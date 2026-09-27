//! Encode-backend benchmark: compares SMG's tokenizer encode path with fastokens.
//!
//! Subcommands (all inputs are local files; nothing is downloaded here):
//!
//! ```text
//! render  --model DIR --conversations FILE --out FILE
//!     Render each conversation with SMG's own chat template; record the
//!     prompt text and its token count under SMG's tokenizer.
//! parity  --model DIR --prompts FILE --stress FILE --conversations FILE --out FILE
//!         [--candidates a,b,...]
//!     Compare every candidate backend's token ids with SMG's, for both
//!     add_special_tokens=false (chat) and true (embeddings), then replay
//!     growing conversations through SMG's L0/L1 caches.
//! time    --model DIR --prompts FILE --variants a,b --rounds N --out FILE.csv
//!         [--warmup FILE] [--build LABEL] [--round-offset K]
//!     Wall and CPU time of every encode, variant order rotated per prompt.
//!     With --warmup the backends first encode that (disjoint) prompt set, so
//!     the timed prompts are new to every cache; without it they warm up on
//!     the timed prompts themselves (exact repeats).
//! turns   --model DIR --conversations FILE --variants a,b --out FILE.csv
//!         [--warmup FILE] [--build LABEL] [--round-offset K]
//!     Multi-turn chats: every user turn re-sends the whole history; turns of
//!     different chats are interleaved; each variant keeps its caches.
//! throughput --model DIR --prompts FILE --variants a,b --threads 1,4 --out FILE.csv
//!         [--warmup FILE] [--build LABEL] [--round-offset K]
//!     One pass over the prompts with N workers pulling from a shared queue
//!     (every prompt encoded exactly once).
//! load    --model DIR --variants a,b [--repeat N] [--out FILE.csv]
//!     Backend construction time (variant `none` loads nothing: RSS baseline).
//! ```
//!
//! Variants: smg (as shipped), hf_encode, hf_encode_fast, fastokens, and for
//! `turns` also smg_cached / fastokens_cached (SMG's L0+L1 caches on top).
//! `FASTOKENS_BPE_THREADS` is read once per process; set it per run.
//!
//! Build it with `--release` so it uses the profile SMG ships.
#![expect(clippy::print_stdout, reason = "benchmark CLI prints its results")]

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fs,
    hint::black_box,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use cpu_time::ProcessTime;
use llm_tokenizer::{
    cache::{CacheConfig, CachedTokenizer},
    chat_template::{
        ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
    },
    create_tokenizer_from_file,
    traits::{
        ChatTemplateOutput, Decoder, Encoder, Encoding, PromptEncoding, RendererCapabilities,
        SpecialTokens, TokenIdType, Tokenizer,
    },
};
use serde_json::{json, Value};

// Same allocator (and the same targets) as the gateway binary.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const USAGE: &str =
    "usage: encode_backends <render|parity|time|turns|throughput|load> --option value ...";
/// Conversations per bucket replayed turn by turn through the caches.
const CACHE_REPLAY_PER_BUCKET: usize = 4;
const MAX_EXAMPLES: usize = 5;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = args.split_first().context(USAGE)?;
    let opts = Opts::parse(rest)?;
    match command.as_str() {
        "render" => render(&opts),
        "parity" => parity(&opts),
        "time" => timing(&opts),
        "turns" => turns(&opts),
        "throughput" => throughput(&opts),
        "load" => load(&opts),
        _ => bail!(USAGE),
    }
}

// ---------------------------------------------------------------------------
// Options and files
// ---------------------------------------------------------------------------

struct Opts(HashMap<String, String>);

impl Opts {
    fn parse(args: &[String]) -> Result<Self> {
        let mut map = HashMap::new();
        let mut it = args.iter();
        while let Some(key) = it.next() {
            let name = key
                .strip_prefix("--")
                .with_context(|| format!("expected --option, got {key}"))?;
            let value = it
                .next()
                .with_context(|| format!("missing value for {key}"))?;
            map.insert(name.to_string(), value.clone());
        }
        Ok(Self(map))
    }

    fn get(&self, name: &str) -> Result<&str> {
        self.0
            .get(name)
            .map(String::as_str)
            .with_context(|| format!("missing --{name}"))
    }

    fn path(&self, name: &str) -> Result<PathBuf> {
        self.get(name).map(PathBuf::from)
    }
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn write_jsonl(path: &Path, rows: &[Value]) -> Result<()> {
    let mut out = BufWriter::new(fs::File::create(path)?);
    for row in rows {
        serde_json::to_writer(&mut out, row)?;
        out.write_all(b"\n")?;
    }
    Ok(out.flush()?)
}

fn load_smg(model: &Path) -> Result<Arc<dyn Tokenizer>> {
    let path = model.to_str().context("model path is not UTF-8")?;
    create_tokenizer_from_file(path)
}

/// The pinned revision recorded by fetch_models.py, if present.
fn model_revision(model: &Path) -> Value {
    fs::read_to_string(model.join("fetch.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .map(|v| json!({"model": v["model"], "revision": v["revision"]}))
        .unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the first `upto` messages of a corpus conversation with SMG's chat
/// template, exactly as the gateway would for a text-only request.
fn render_conversation(
    tok: &dyn Tokenizer,
    conv: &Value,
    upto: Option<usize>,
) -> Result<ChatTemplateOutput> {
    let all = conv["messages"]
        .as_array()
        .context("conversation has no messages")?;
    let openai = tok.chat_template_content_format() == ChatTemplateContentFormat::OpenAI;
    let messages: Vec<Value> = all[..upto.unwrap_or(all.len())]
        .iter()
        .map(|m| {
            if openai {
                json!({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
            } else {
                m.clone()
            }
        })
        .collect();
    let tools = conv["tools"].as_array().filter(|t| !t.is_empty());
    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: tools.map(Vec::as_slice),
        documents: None,
        template_kwargs: None,
        special_tokens: Some(tok.get_special_tokens()),
        thinking: None,
    };
    tok.apply_chat_template_with_encoding(&messages, params, None)
}

fn render(opts: &Opts) -> Result<()> {
    let tok = load_smg(&opts.path("model")?)?;
    let conversations = read_jsonl(&opts.path("conversations")?)?;
    let mut rows = Vec::with_capacity(conversations.len());
    let mut buckets: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for conv in &conversations {
        let rendered = render_conversation(tok.as_ref(), conv, None)
            .with_context(|| format!("render {}", conv["id"]))?;
        let deferred = matches!(rendered.encoding, PromptEncoding::Deferred(_));
        let tokens = tok.encode(&rendered.text, false)?.token_ids().len();
        let bucket = conv["bucket"].as_str().unwrap_or_default().to_string();
        buckets.entry(bucket.clone()).or_default().push(tokens);
        rows.push(json!({
            "id": conv["id"], "bucket": bucket, "kind": conv["kind"],
            "bytes": rendered.text.len(), "tokens": tokens, "deferred": deferred,
            "text": rendered.text,
        }));
    }
    write_jsonl(&opts.path("out")?, &rows)?;
    for (bucket, tokens) in &buckets {
        let (min, max) = (tokens.iter().min(), tokens.iter().max());
        println!(
            "bucket {bucket:>4}: {} prompts, {min:?}..{max:?} tokens",
            tokens.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parity
// ---------------------------------------------------------------------------

/// Index of the first position where two id lists differ; `None` if equal.
fn first_mismatch(expected: &[u32], got: &[u32]) -> Option<usize> {
    match expected.iter().zip(got).position(|(a, b)| a != b) {
        Some(at) => Some(at),
        None if expected.len() != got.len() => Some(expected.len().min(got.len())),
        None => None,
    }
}

/// The ids within `radius` of `at`, clamped to the slice.
fn around(ids: &[u32], at: usize, radius: usize) -> &[u32] {
    let start = at.saturating_sub(radius).min(ids.len());
    let end = at.saturating_add(radius + 1).min(ids.len());
    &ids[start..end]
}

/// Up to `radius` characters of `text` on each side of character `at`.
fn text_around(text: &str, at: usize, radius: usize) -> String {
    let start = at.saturating_sub(radius);
    text.chars().skip(start).take(at + radius - start).collect()
}

fn describe_mismatch(
    smg: &dyn Tokenizer,
    input: &str,
    text: &str,
    expected: &[u32],
    got: &[u32],
    at: usize,
) -> Value {
    let tokens = |ids: &[u32]| -> Vec<Value> {
        ids.iter()
            .map(|&id| json!([id, smg.id_to_token(id)]))
            .collect()
    };
    // Byte-level BPE decodes a prefix of the ids to a prefix of the text, so
    // its length locates the mismatch in the input.
    let at_char = smg
        .decode(&expected[..at], false)
        .map(|s| s.chars().count())
        .unwrap_or(0);
    json!({
        "input": input,
        "index": at,
        "expected_len": expected.len(),
        "got_len": got.len(),
        "expected": tokens(around(expected, at, 5)),
        "got": tokens(around(got, at, 5)),
        "text": format!("{:?}", text_around(text, at_char, 40)),
        "same_decoded_text": smg.decode(expected, false).ok() == smg.decode(got, false).ok(),
    })
}

type EncodeFn = Box<dyn Fn(&str, bool) -> Result<Vec<u32>> + Send + Sync>;

struct Candidate {
    name: &'static str,
    encode: EncodeFn,
}

fn smg_encode(tok: Arc<dyn Tokenizer>) -> EncodeFn {
    Box::new(move |text, special| Ok(tok.encode(text, special)?.token_ids().to_vec()))
}

fn candidates(model: &Path, smg: &Arc<dyn Tokenizer>) -> Result<Vec<Candidate>> {
    // Controls: SMG against itself must never mismatch, and a copy that drops
    // the last id must mismatch on every non-empty input (proves detection).
    let broken = smg.clone();
    let mut out = vec![
        Candidate {
            name: "smg_again",
            encode: smg_encode(smg.clone()),
        },
        Candidate {
            name: "control_drop_last",
            encode: Box::new(move |text, special| {
                let mut ids = broken.encode(text, special)?.token_ids().to_vec();
                ids.pop();
                Ok(ids)
            }),
        },
    ];
    let json_path = model.join("tokenizer.json");
    if !json_path.exists() {
        return Ok(out);
    }
    let fast = Arc::new(load_fastokens(&json_path)?);
    out.push(Candidate {
        name: "fastokens",
        encode: Box::new(move |text, special| Ok(fast.encode_with_special_tokens(text, special)?)),
    });
    let fast_file = fastokens::Tokenizer::from_file(&json_path)?;
    out.push(Candidate {
        name: "fastokens_file",
        encode: Box::new(move |text, special| {
            Ok(fast_file.encode_with_special_tokens(text, special)?)
        }),
    });
    let hf = Arc::new(load_hf(&json_path)?);
    let hf_fast = hf.clone();
    out.push(Candidate {
        name: "hf_raw_encode",
        encode: Box::new(move |text, special| {
            Ok(hf
                .encode(text, special)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec())
        }),
    });
    out.push(Candidate {
        name: "hf_raw_encode_fast",
        encode: Box::new(move |text, special| {
            Ok(hf_fast
                .encode_fast(text, special)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec())
        }),
    });
    Ok(out)
}

/// fastokens from `tokenizer.json` alone. `from_file` parses straight into
/// fastokens' own types (no generic JSON tree), and linking the file into an
/// empty directory keeps it from also merging `tokenizer_config.json` added
/// tokens, which SMG does not do.
fn load_fastokens(path: &Path) -> Result<fastokens::Tokenizer> {
    let dir = tempfile::tempdir()?;
    let isolated = dir.path().join("tokenizer.json");
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        symlink(fs::canonicalize(path)?, &isolated)?;
    }
    #[cfg(not(unix))]
    fs::copy(path, &isolated)?;
    Ok(fastokens::Tokenizer::from_file(&isolated)?)
}

/// SMG's L0 (exact) and L1 (prefix) caches, both on.
fn l0_l1_config() -> CacheConfig {
    CacheConfig {
        enable_l1: true,
        ..CacheConfig::default()
    }
}

#[derive(Default)]
struct Tally {
    compared: usize,
    mismatches: usize,
    errors: usize,
    examples: Vec<Value>,
}

impl Tally {
    fn to_json(&self) -> Value {
        json!({
            "compared": self.compared, "mismatches": self.mismatches,
            "errors": self.errors, "examples": self.examples,
        })
    }
}

fn parity(opts: &Opts) -> Result<()> {
    let model = opts.path("model")?;
    let smg = load_smg(&model)?;
    let mut all = candidates(&model, &smg)?;
    if let Ok(only) = opts.get("candidates") {
        let keep: Vec<&str> = only.split(',').collect();
        all.retain(|c| keep.contains(&c.name));
    }
    let mut inputs: Vec<(String, String)> = Vec::new();
    for key in ["prompts", "stress"] {
        for row in read_jsonl(&opts.path(key)?)? {
            let id = row["id"].as_str().unwrap_or_default().to_string();
            inputs.push((id, row["text"].as_str().unwrap_or_default().to_string()));
        }
    }

    let mut tallies: BTreeMap<(&str, bool), Tally> = BTreeMap::new();
    for (id, text) in &inputs {
        for special in [false, true] {
            let expected = smg.encode(text, special)?.token_ids().to_vec();
            for cand in &all {
                let tally = tallies.entry((cand.name, special)).or_default();
                tally.compared += 1;
                match (cand.encode)(text, special) {
                    Ok(got) => {
                        if let Some(at) = first_mismatch(&expected, &got) {
                            tally.mismatches += 1;
                            if tally.examples.len() < MAX_EXAMPLES {
                                tally.examples.push(describe_mismatch(
                                    smg.as_ref(),
                                    id,
                                    text,
                                    &expected,
                                    &got,
                                    at,
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        tally.errors += 1;
                        if tally.examples.len() < MAX_EXAMPLES {
                            tally
                                .examples
                                .push(json!({"input": id, "error": err.to_string()}));
                        }
                    }
                }
            }
        }
    }

    let conversations = read_jsonl(&opts.path("conversations")?)?;
    let cache = cache_replay(&model, &smg, &conversations)?;
    let vocab = vocab_diff(&model, smg.as_ref())?;

    let mut report = serde_json::Map::new();
    for ((name, special), tally) in &tallies {
        println!(
            "{name:<20} add_special_tokens={special:<5} compared={:>4} mismatches={:>3} errors={}",
            tally.compared, tally.mismatches, tally.errors
        );
        report.entry(name.to_string()).or_insert_with(|| json!({}))[special.to_string()] =
            tally.to_json();
    }
    println!("cache replay: {cache}");
    println!("vocab: {}", vocab["differences"]);
    let summary = json!({
        "model": model_revision(&model),
        "fastokens_bpe_threads": std::env::var("FASTOKENS_BPE_THREADS").unwrap_or_else(|_| "default".into()),
        "inputs": inputs.len(),
        "candidates": report,
        "cache_replay": cache,
        "vocab": vocab,
    });
    fs::write(opts.path("out")?, serde_json::to_string_pretty(&summary)?)?;
    Ok(())
}

/// Replay growing conversations turn by turn (every prefix that ends on a user
/// message) through SMG's L0+L1 caches, over SMG's own tokenizer and over the
/// fastokens-backed wrapper, and compare with an uncached SMG encode.
fn cache_replay(model: &Path, smg: &Arc<dyn Tokenizer>, conversations: &[Value]) -> Result<Value> {
    let config = l0_l1_config();
    let mut cached: Vec<(&str, CachedTokenizer)> = vec![(
        "smg_cached",
        CachedTokenizer::new(smg.clone(), config.clone()),
    )];
    if model.join("tokenizer.json").exists() {
        let wrapper =
            FastEncodeTokenizer::new(load_fastokens(&model.join("tokenizer.json"))?, smg.clone());
        cached.push((
            "fastokens_cached",
            CachedTokenizer::new(Arc::new(wrapper), config),
        ));
    }
    let mut per_bucket: HashMap<&str, usize> = HashMap::new();
    let mut tallies: BTreeMap<&str, Tally> = BTreeMap::new();
    for conv in conversations {
        let bucket = conv["bucket"].as_str().unwrap_or_default();
        if !matches!(bucket, "16k" | "50k") {
            continue;
        }
        let seen = per_bucket.entry(bucket).or_default();
        if *seen >= CACHE_REPLAY_PER_BUCKET {
            continue;
        }
        *seen += 1;
        let messages = conv["messages"].as_array().context("messages")?;
        for upto in (2..=messages.len()).filter(|&k| messages[k - 1]["role"] == "user") {
            let text = render_conversation(smg.as_ref(), conv, Some(upto))?.text;
            let expected = smg.encode(&text, false)?.token_ids().to_vec();
            for (name, tok) in &cached {
                let tally = tallies.entry(name).or_default();
                tally.compared += 1;
                let got = tok.encode(&text, false)?.token_ids().to_vec();
                if let Some(at) = first_mismatch(&expected, &got) {
                    tally.mismatches += 1;
                    if tally.examples.len() < MAX_EXAMPLES {
                        let id = format!("{}@{upto}", conv["id"].as_str().unwrap_or_default());
                        tally.examples.push(describe_mismatch(
                            smg.as_ref(),
                            &id,
                            &text,
                            &expected,
                            &got,
                            at,
                        ));
                    }
                }
            }
        }
    }
    Ok(tallies
        .iter()
        .map(|(name, t)| (name.to_string(), t.to_json()))
        .collect())
}

/// Compare `id_to_token` across the vocabulary (plus a margin for added tokens).
fn vocab_diff(model: &Path, smg: &dyn Tokenizer) -> Result<Value> {
    let json_path = model.join("tokenizer.json");
    if !json_path.exists() {
        return Ok(json!({"differences": "n/a"}));
    }
    let fast = load_fastokens(&json_path)?;
    let limit = smg.vocab_size().max(fast.vocab_size()) as u32 + 2048;
    let mut differences = 0usize;
    let mut examples = Vec::new();
    for id in 0..limit {
        let (a, b) = (
            smg.id_to_token(id),
            fast.id_to_token(id).map(str::to_string),
        );
        if a != b {
            differences += 1;
            if examples.len() < MAX_EXAMPLES {
                examples.push(json!([id, a, b]));
            }
        }
    }
    Ok(json!({"checked": limit, "differences": differences, "examples": examples}))
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Encodes one prompt with a named backend and returns its token count.
type TimedEncode = Box<dyn Fn(&str) -> Result<usize> + Send + Sync>;

/// Variant order for one prompt in one round: rotate so every variant runs
/// first, second, ... equally often (no fixed warm-cache advantage).
fn rotated_order(n: usize, prompt: usize, round: usize) -> Vec<usize> {
    (0..n).map(|i| (i + prompt + round) % n).collect()
}

fn load_hf(path: &Path) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(path).map_err(anyhow::Error::msg)
}

/// The backends a timing run can compare, all with add_special_tokens=false
/// (the chat path):
/// - `smg`: SMG as shipped (`HuggingFaceTokenizer::encode`, or tiktoken-rs)
/// - `hf_encode`: raw HF `encode` (computes offsets SMG never reads)
/// - `hf_encode_fast`: raw HF `encode_fast` (no offsets)
/// - `fastokens`: fastokens built from `tokenizer.json` alone
/// - `smg_cached` / `fastokens_cached`: the same behind SMG's L0+L1 caches
fn variant_encoder(model: &Path, smg: &Arc<dyn Tokenizer>, name: &str) -> Result<TimedEncode> {
    let json_path = model.join("tokenizer.json");
    Ok(match name {
        "smg" => {
            let tok = smg.clone();
            Box::new(move |text| Ok(tok.encode(text, false)?.token_ids().len()))
        }
        "hf_encode" => {
            let hf = load_hf(&json_path)?;
            Box::new(move |text| Ok(hf.encode(text, false).map_err(anyhow::Error::msg)?.len()))
        }
        "hf_encode_fast" => {
            let hf = load_hf(&json_path)?;
            Box::new(move |text| {
                Ok(hf
                    .encode_fast(text, false)
                    .map_err(anyhow::Error::msg)?
                    .len())
            })
        }
        "fastokens" => {
            let fast = load_fastokens(&json_path)?;
            Box::new(move |text| Ok(fast.encode_with_special_tokens(text, false)?.len()))
        }
        "smg_cached" => {
            let cached = CachedTokenizer::new(smg.clone(), l0_l1_config());
            Box::new(move |text| Ok(cached.encode(text, false)?.token_ids().len()))
        }
        "fastokens_cached" => {
            let wrapper = FastEncodeTokenizer::new(load_fastokens(&json_path)?, smg.clone());
            let cached = CachedTokenizer::new(Arc::new(wrapper), l0_l1_config());
            Box::new(move |text| Ok(cached.encode(text, false)?.token_ids().len()))
        }
        other => bail!(
            "unknown variant {other:?} (smg, hf_encode, hf_encode_fast, fastokens, smg_cached, fastokens_cached)"
        ),
    })
}

/// Time to construct one backend from files already in the page cache.
fn load_variant(model: &Path, name: &str) -> Result<Duration> {
    let json_path = model.join("tokenizer.json");
    let start = Instant::now();
    let loaded: Box<dyn Any> = match name {
        "none" => Box::new(()),
        "smg" => Box::new(load_smg(model)?),
        "hf_encode" | "hf_encode_fast" => Box::new(load_hf(&json_path)?),
        "fastokens" => Box::new(load_fastokens(&json_path)?),
        // The proposed backend keeps SMG's tokenizer (decode, templates) too.
        "smg_fastokens" => Box::new((load_smg(model)?, load_fastokens(&json_path)?)),
        other => bail!(
            "unknown variant {other:?} (none, smg, hf_encode, hf_encode_fast, fastokens, smg_fastokens)"
        ),
    };
    let elapsed = start.elapsed();
    drop(loaded);
    Ok(elapsed)
}

fn fastokens_threads_label() -> String {
    std::env::var("FASTOKENS_BPE_THREADS").unwrap_or_else(|_| "default".into())
}

/// One encode per (round, prompt, variant), variants rotated per prompt and
/// round; writes wall and process CPU time (all threads) for every encode.
fn timing(opts: &Opts) -> Result<()> {
    let model = opts.path("model")?;
    let smg = load_smg(&model)?;
    let variants = opts
        .get("variants")?
        .split(',')
        .map(|name| Ok((name, variant_encoder(&model, &smg, name)?)))
        .collect::<Result<Vec<_>>>()?;
    let prompts = read_jsonl(&opts.path("prompts")?)?;
    let rounds: usize = opts.get("rounds")?.parse()?;
    // Separate processes (e.g. alternating builds) pass different offsets so
    // each one rotates the variant order differently.
    let offset: usize = opts.get("round-offset").unwrap_or("0").parse()?;
    let build = opts.get("build").unwrap_or("local");
    let threads = fastokens_threads_label();

    warm_up(opts, &variants, &prompts)?;
    let out_path = opts.path("out")?;
    let mut out = BufWriter::new(fs::File::create(&out_path)?);
    writeln!(
        out,
        "build,fastokens_threads,round,prompt,bucket,kind,bytes,tokens,variant,wall_ns,cpu_ns"
    )?;
    let mut rows = 0usize;
    for round in offset..offset + rounds {
        for (i, prompt) in prompts.iter().enumerate() {
            let text = prompt["text"].as_str().unwrap_or_default();
            for v in rotated_order(variants.len(), i, round) {
                let (name, encode) = &variants[v];
                let cpu_start = ProcessTime::now();
                let wall_start = Instant::now();
                let tokens = black_box(encode(black_box(text))?);
                let wall = wall_start.elapsed();
                let cpu = cpu_start.elapsed();
                writeln!(
                    out,
                    "{build},{threads},{round},{},{},{},{},{tokens},{name},{},{}",
                    prompt["id"].as_str().unwrap_or_default(),
                    prompt["bucket"].as_str().unwrap_or_default(),
                    prompt["kind"].as_str().unwrap_or_default(),
                    text.len(),
                    wall.as_nanos(),
                    cpu.as_nanos(),
                )?;
                rows += 1;
            }
        }
    }
    out.flush()?;
    println!("wrote {rows} timings to {}", out_path.display());
    Ok(())
}

/// Encode `--warmup` (or, without it, the timed prompts) once with every
/// variant before timing.
fn warm_up(opts: &Opts, variants: &[(&str, TimedEncode)], timed: &[Value]) -> Result<()> {
    let warmup = match opts.path("warmup") {
        Ok(path) => read_jsonl(&path)?,
        Err(_) => timed.to_vec(),
    };
    for prompt in &warmup {
        let text = prompt["text"].as_str().unwrap_or_default();
        for (_, encode) in variants {
            black_box(encode(text)?);
        }
    }
    Ok(())
}

/// Saturation run: one pass over the prompts with N workers pulling from a
/// shared queue, like concurrent requests on the gateway's blocking pool.
/// Every prompt is encoded exactly once per (variant, workers) pass.
fn throughput(opts: &Opts) -> Result<()> {
    let model = opts.path("model")?;
    let smg = load_smg(&model)?;
    let texts: Vec<String> = read_jsonl(&opts.path("prompts")?)?
        .iter()
        .map(|p| p["text"].as_str().unwrap_or_default().to_string())
        .collect();
    let threads: Vec<usize> = opts
        .get("threads")?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let build = opts.get("build").unwrap_or("local");
    let offset: usize = opts.get("round-offset").unwrap_or("0").parse()?;
    let fast_threads = fastokens_threads_label();
    let names: Vec<&str> = opts.get("variants")?.split(',').collect();
    let variants = rotated_order(names.len(), 0, offset)
        .into_iter()
        .map(|i| Ok((names[i], variant_encoder(&model, &smg, names[i])?)))
        .collect::<Result<Vec<_>>>()?;
    // Warm up on --warmup only: warming on the timed prompts would make the
    // single pass a pass over exact repeats.
    warm_up(opts, &variants, &[])?;
    let mut out = BufWriter::new(fs::File::create(opts.path("out")?)?);
    writeln!(
        out,
        "build,fastokens_threads,round,variant,threads,seconds,prompts,tokens,cpu_seconds"
    )?;
    for (name, encode) in &variants {
        for &workers in &threads {
            let run = run_throughput(encode, &texts, workers)?;
            let wall = run.wall.as_secs_f64();
            println!(
                "{name:<16} threads={workers:<3} {:>10.0} tokens/s  {:>7.1} prompts/s  cpu/wall={:.2}",
                run.tokens as f64 / wall,
                run.prompts as f64 / wall,
                run.cpu.as_secs_f64() / wall,
            );
            writeln!(
                out,
                "{build},{fast_threads},{offset},{name},{workers},{wall:.3},{},{},{:.3}",
                run.prompts,
                run.tokens,
                run.cpu.as_secs_f64(),
            )?;
        }
    }
    Ok(out.flush()?)
}

struct ThroughputRun {
    prompts: usize,
    tokens: usize,
    wall: Duration,
    cpu: Duration,
}

fn run_throughput(encode: &TimedEncode, texts: &[String], workers: usize) -> Result<ThroughputRun> {
    if texts.is_empty() || workers == 0 {
        bail!("throughput needs at least one prompt and one worker");
    }
    let next = AtomicUsize::new(0);
    let prompts = AtomicUsize::new(0);
    let tokens = AtomicUsize::new(0);
    let cpu_start = ProcessTime::now();
    let start = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(text) = texts.get(i) else {
                            return Ok(());
                        };
                        tokens.fetch_add(encode(text)?, Ordering::Relaxed);
                        prompts.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("throughput worker panicked"))??;
        }
        Ok(())
    })?;
    Ok(ThroughputRun {
        prompts: prompts.into_inner(),
        tokens: tokens.into_inner(),
        wall: start.elapsed(),
        cpu: cpu_start.elapsed(),
    })
}

/// Round-robin order over chats: turn 0 of every chat, then turn 1, ... so
/// the caches see interleaved traffic, as on a busy gateway.
fn turn_schedule(turns: &[usize]) -> Vec<(usize, usize)> {
    let longest = turns.iter().copied().max().unwrap_or(0);
    (0..longest)
        .flat_map(|k| {
            turns
                .iter()
                .enumerate()
                .filter(move |&(_, &n)| k < n)
                .map(move |(chat, _)| (chat, k))
        })
        .collect()
}

/// Multi-turn chats (16k and 50k buckets): each user turn re-sends the whole
/// history, rendered with SMG's chat template. One tokenizer per variant for
/// the whole run, so its caches carry over between turns.
fn turns(opts: &Opts) -> Result<()> {
    let model = opts.path("model")?;
    let smg = load_smg(&model)?;
    let variants = opts
        .get("variants")?
        .split(',')
        .map(|name| Ok((name, variant_encoder(&model, &smg, name)?)))
        .collect::<Result<Vec<_>>>()?;
    let offset: usize = opts.get("round-offset").unwrap_or("0").parse()?;
    let build = opts.get("build").unwrap_or("local");
    let threads = fastokens_threads_label();
    let chats: Vec<Value> = read_jsonl(&opts.path("conversations")?)?
        .into_iter()
        .filter(|c| matches!(c["bucket"].as_str(), Some("16k" | "50k")))
        .collect();
    let mut rendered: Vec<Vec<(usize, String)>> = Vec::with_capacity(chats.len());
    for chat in &chats {
        let messages = chat["messages"].as_array().context("messages")?;
        let mut prefixes = Vec::new();
        for upto in (2..=messages.len()).filter(|&k| messages[k - 1]["role"] == "user") {
            prefixes.push((
                upto,
                render_conversation(smg.as_ref(), chat, Some(upto))?.text,
            ));
        }
        rendered.push(prefixes);
    }
    warm_up(opts, &variants, &[])?;

    let out_path = opts.path("out")?;
    let mut out = BufWriter::new(fs::File::create(&out_path)?);
    writeln!(
        out,
        "build,fastokens_threads,round,prompt,bucket,kind,bytes,tokens,variant,wall_ns,cpu_ns"
    )?;
    let schedule = turn_schedule(&rendered.iter().map(Vec::len).collect::<Vec<_>>());
    for (i, &(chat, k)) in schedule.iter().enumerate() {
        let (upto, text) = &rendered[chat][k];
        for v in rotated_order(variants.len(), i, offset) {
            let (name, encode) = &variants[v];
            let cpu_start = ProcessTime::now();
            let wall_start = Instant::now();
            let tokens = black_box(encode(black_box(text))?);
            let wall = wall_start.elapsed();
            let cpu = cpu_start.elapsed();
            writeln!(
                out,
                "{build},{threads},{offset},{}@{upto},{},{},{},{tokens},{name},{},{}",
                chats[chat]["id"].as_str().unwrap_or_default(),
                chats[chat]["bucket"].as_str().unwrap_or_default(),
                chats[chat]["kind"].as_str().unwrap_or_default(),
                text.len(),
                wall.as_nanos(),
                cpu.as_nanos(),
            )?;
        }
    }
    out.flush()?;
    println!(
        "wrote {} turn timings to {}",
        schedule.len() * variants.len(),
        out_path.display()
    );
    Ok(())
}

/// Median construction time per backend over `repeat` loads.
fn load(opts: &Opts) -> Result<()> {
    let model = opts.path("model")?;
    let repeat: usize = opts.get("repeat").unwrap_or("5").parse()?;
    let mut rows = vec!["variant,repeat,median_ms".to_string()];
    for name in opts.get("variants")?.split(',') {
        let mut times = (0..repeat)
            .map(|_| load_variant(&model, name).map(|d| d.as_secs_f64() * 1e3))
            .collect::<Result<Vec<f64>>>()?;
        times.sort_by(f64::total_cmp);
        let median = times.get(times.len() / 2).copied().unwrap_or_default();
        println!("load {name:<16} median {median:>8.1} ms over {repeat}");
        rows.push(format!("{name},{repeat},{median:.3}"));
    }
    if let Ok(out) = opts.path("out") {
        fs::write(out, rows.join("\n") + "\n")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Proposed backend (prototype)
// ---------------------------------------------------------------------------

/// fastokens encodes; SMG's own tokenizer does everything else (decode,
/// streaming decode, chat templates, vocab, special tokens, EOS ids).
struct FastEncodeTokenizer {
    fast: fastokens::Tokenizer,
    smg: Arc<dyn Tokenizer>,
}

impl FastEncodeTokenizer {
    fn new(fast: fastokens::Tokenizer, smg: Arc<dyn Tokenizer>) -> Self {
        Self { fast, smg }
    }
}

impl Encoder for FastEncodeTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        Ok(Encoding::Plain(
            self.fast
                .encode_with_special_tokens(input, add_special_tokens)?,
        ))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        Ok(self
            .fast
            .encode_batch(inputs, add_special_tokens)?
            .into_iter()
            .map(Encoding::Plain)
            .collect())
    }
}

impl Decoder for FastEncodeTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.smg.decode(token_ids, skip_special_tokens)
    }

    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        self.smg
            .decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }
}

impl Tokenizer for FastEncodeTokenizer {
    fn vocab_size(&self) -> usize {
        self.smg.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.smg.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.smg.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.smg.id_to_token(id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        self.smg.apply_chat_template(messages, params)
    }

    fn apply_chat_template_with_encoding(
        &self,
        messages: &[Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> Result<ChatTemplateOutput> {
        self.smg
            .apply_chat_template_with_encoding(messages, params, assistant_prefix)
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.smg.chat_template_content_format()
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        self.smg.thinking_toggle()
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.smg.thinking_key_name()
    }

    fn native_reasoning_effort_values(&self) -> &'static [&'static str] {
        self.smg.native_reasoning_effort_values()
    }

    fn native_reasoning_effort_off_values(&self) -> &'static [&'static str] {
        self.smg.native_reasoning_effort_off_values()
    }

    fn think_in_prefill(&self) -> bool {
        self.smg.think_in_prefill()
    }

    fn renderer_capabilities(&self) -> RendererCapabilities {
        self.smg.renderer_capabilities()
    }

    fn eos_token_ids(&self) -> &[TokenIdType] {
        self.smg.eos_token_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_mismatch_is_none_for_equal_ids() {
        assert_eq!(first_mismatch(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(first_mismatch(&[], &[]), None);
    }

    #[test]
    fn first_mismatch_finds_the_first_differing_index() {
        assert_eq!(first_mismatch(&[1, 2, 3, 4], &[1, 2, 9, 4]), Some(2));
        assert_eq!(first_mismatch(&[7], &[8]), Some(0));
    }

    #[test]
    fn first_mismatch_reports_the_shorter_length_for_a_prefix() {
        assert_eq!(first_mismatch(&[1, 2, 3], &[1, 2]), Some(2));
        assert_eq!(first_mismatch(&[1, 2], &[1, 2, 3]), Some(2));
    }

    #[test]
    fn around_clamps_to_the_slice() {
        let ids = [10, 11, 12, 13, 14, 15];
        assert_eq!(around(&ids, 3, 1), &[12, 13, 14]);
        assert_eq!(around(&ids, 0, 2), &[10, 11, 12]);
        assert_eq!(around(&ids, 5, 3), &[12, 13, 14, 15]);
        assert_eq!(around(&ids, 9, 1), &[] as &[u32]);
    }

    #[test]
    fn text_around_counts_characters_not_bytes() {
        let text = "花果山上有一塊仙石";
        assert_eq!(text_around(text, 4, 2), "山上有一");
        assert_eq!(text_around(text, 0, 3), "花果山");
        assert_eq!(text_around(text, 20, 3), "");
    }

    #[test]
    fn rotated_order_starts_at_prompt_plus_round() {
        assert_eq!(rotated_order(3, 0, 0), vec![0, 1, 2]);
        assert_eq!(rotated_order(3, 1, 0), vec![1, 2, 0]);
        assert_eq!(rotated_order(3, 0, 1), vec![1, 2, 0]);
        assert_eq!(rotated_order(3, 2, 2), vec![1, 2, 0]);
        assert_eq!(rotated_order(1, 5, 7), vec![0]);
    }

    #[test]
    fn unknown_variant_is_rejected() {
        let smg = create_tokenizer_from_file("mock").unwrap();
        let err = variant_encoder(Path::new("/nonexistent"), &smg, "nope")
            .err()
            .unwrap();
        assert!(err.to_string().contains("unknown variant"), "{err}");
        let err = load_variant(Path::new("/nonexistent"), "nope")
            .err()
            .unwrap();
        assert!(err.to_string().contains("unknown variant"), "{err}");
    }

    #[test]
    fn smg_variant_returns_smg_token_count() {
        let smg = create_tokenizer_from_file("mock").unwrap();
        let encode = variant_encoder(Path::new("/nonexistent"), &smg, "smg").unwrap();
        let expected = smg
            .encode("Hello world test", false)
            .unwrap()
            .token_ids()
            .len();
        assert_eq!(expected, 3);
        assert_eq!(encode("Hello world test").unwrap(), expected);
    }

    fn mock_encode(name: &str) -> TimedEncode {
        let smg = create_tokenizer_from_file("mock").unwrap();
        variant_encoder(Path::new("/nonexistent"), &smg, name).unwrap()
    }

    #[test]
    fn throughput_encodes_every_prompt_exactly_once() {
        let encode = mock_encode("smg");
        let texts: Vec<String> = (0..50)
            .map(|i| {
                if i % 2 == 0 {
                    "Hello world"
                } else {
                    "test token"
                }
                .to_string()
            })
            .collect();
        let run = run_throughput(&encode, &texts, 4).unwrap();
        assert_eq!(run.prompts, 50);
        assert_eq!(run.tokens, 100);
    }

    #[test]
    fn throughput_rejects_an_empty_prompt_list() {
        let encode = mock_encode("smg");
        assert!(run_throughput(&encode, &[], 2).is_err());
    }

    #[test]
    fn turn_schedule_interleaves_conversations_round_robin() {
        assert_eq!(
            turn_schedule(&[2, 3, 1]),
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]
        );
        assert!(turn_schedule(&[]).is_empty());
    }

    #[test]
    fn cached_variant_matches_uncached_token_counts() {
        let plain = mock_encode("smg");
        let cached = mock_encode("smg_cached");
        for text in ["Hello world", "Hello world test", "Hello world"] {
            assert_eq!(cached(text).unwrap(), plain(text).unwrap(), "{text}");
        }
    }

    #[test]
    fn smg_fastokens_is_a_known_load_variant() {
        let err = load_variant(Path::new("/nonexistent"), "smg_fastokens")
            .err()
            .unwrap();
        assert!(!err.to_string().contains("unknown variant"), "{err}");
    }

    /// Needs a downloaded model: ENCODE_BENCH_MODEL_DIR=/path/to/model \
    ///   cargo test -p llm-tokenizer --example encode_backends -- --ignored
    #[test]
    #[ignore = "needs ENCODE_BENCH_MODEL_DIR with a tokenizer.json"]
    fn fast_encode_wrapper_matches_smg_and_delegates_decode() {
        let dir = PathBuf::from(std::env::var("ENCODE_BENCH_MODEL_DIR").unwrap());
        let smg = load_smg(&dir).unwrap();
        let fast = load_fastokens(&dir.join("tokenizer.json")).unwrap();
        let wrapper = FastEncodeTokenizer::new(fast, smg.clone());
        for text in [
            "Hello, world!",
            "fn main() {\n    println!(\"hi\");\n}",
            "花果山上有一塊仙石 🙂",
            "",
        ] {
            let expected = smg.encode(text, false).unwrap().token_ids().to_vec();
            let got = wrapper.encode(text, false).unwrap().token_ids().to_vec();
            assert_eq!(got, expected, "{text:?}");
            assert_eq!(
                wrapper.decode(&got, false).unwrap(),
                smg.decode(&expected, false).unwrap()
            );
        }
        let batch = wrapper.encode_batch(&["a b c", "x"], false).unwrap();
        assert_eq!(
            batch[0].token_ids(),
            smg.encode("a b c", false).unwrap().token_ids()
        );
    }
}
