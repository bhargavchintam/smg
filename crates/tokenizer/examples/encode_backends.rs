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
//! ```
//!
//! Build it with `--release` so it uses the profile SMG ships.
#![expect(clippy::print_stdout, reason = "benchmark CLI prints its results")]

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
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

#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const USAGE: &str = "usage: encode_backends <render|parity> --option value ...";
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
    let fast = Arc::new(fastokens_from_json(&json_path)?);
    out.push(Candidate {
        name: "fastokens_json",
        encode: Box::new(move |text, special| Ok(fast.encode_with_special_tokens(text, special)?)),
    });
    let fast_file = fastokens::Tokenizer::from_file(&json_path)?;
    out.push(Candidate {
        name: "fastokens_file",
        encode: Box::new(move |text, special| {
            Ok(fast_file.encode_with_special_tokens(text, special)?)
        }),
    });
    let hf = Arc::new(tokenizers::Tokenizer::from_file(&json_path).map_err(anyhow::Error::msg)?);
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

/// fastokens built from the raw `tokenizer.json` only. `from_file` also merges
/// `tokenizer_config.json` added tokens, which SMG does not do.
fn fastokens_from_json(path: &Path) -> Result<fastokens::Tokenizer> {
    let raw: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(fastokens::Tokenizer::from_json(raw)?)
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
    let config = CacheConfig {
        enable_l1: true,
        ..CacheConfig::default()
    };
    let mut cached: Vec<(&str, CachedTokenizer)> = vec![(
        "smg_cached",
        CachedTokenizer::new(smg.clone(), config.clone()),
    )];
    if model.join("tokenizer.json").exists() {
        let wrapper = FastEncodeTokenizer::new(
            fastokens_from_json(&model.join("tokenizer.json"))?,
            smg.clone(),
        );
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
    let fast = fastokens_from_json(&json_path)?;
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

    /// Needs a downloaded model: ENCODE_BENCH_MODEL_DIR=/path/to/model \
    ///   cargo test -p llm-tokenizer --example encode_backends -- --ignored
    #[test]
    #[ignore = "needs ENCODE_BENCH_MODEL_DIR with a tokenizer.json"]
    fn fast_encode_wrapper_matches_smg_and_delegates_decode() {
        let dir = PathBuf::from(std::env::var("ENCODE_BENCH_MODEL_DIR").unwrap());
        let smg = load_smg(&dir).unwrap();
        let fast = fastokens_from_json(&dir.join("tokenizer.json")).unwrap();
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
