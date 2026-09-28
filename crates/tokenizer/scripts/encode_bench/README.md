# Encode-backend benchmark

Measures how fast SMG turns a rendered prompt into token ids, as shipped and
with faster alternatives, and checks that every alternative returns exactly
the same ids as SMG.

- **Harness:** `crates/tokenizer/examples/encode_backends.rs`, a Rust example with the subcommands `render`, `parity`, `time`, `turns`, `throughput` and `load`.
- **Scripts:** Python 3.8+, standard library only.

## Backends compared

| variant | what it is |
|---|---|
| `smg` | `HuggingFaceTokenizer::encode` as shipped (full HF `encode`, offsets included) |
| `hf_encode` / `hf_encode_fast` | raw HF `tokenizers` `encode` / `encode_fast` (SMG only reads the ids) |
| `fastokens` | [fastokens](https://crates.io/crates/fastokens) 0.3.2 built from `tokenizer.json` alone |
| `smg_cached` / `fastokens_cached` | the same behind SMG's L0+L1 caches (multi-turn runs) |

Builds of the same code:

| build | what it is |
|---|---|
| `z` | SMG's release profile as shipped (`opt-level = "z"`) |
| `tok` | `z`, plus opt-level 2 for `tokenizers` |
| `toksys` | `tok`, plus opt-level 2 for `onig` and `onig_sys` |
| `hotdeps` | `z`, plus opt-level 2 for every crate on HF's encode path |
| `o2` | everything at opt-level 2 (upper bound) |

## Run it

Set `DATA` to any directory outside the repository.

```bash
DATA=/path/to/encode-bench-data
cd crates/tokenizer/scripts/encode_bench

# 1. Tokenizer files at the revisions in pins.json (downloads are sha256-checked)
python3 fetch_models.py --out "$DATA/models"

# 2. Public-domain Chinese text (Project Gutenberg #23962)
mkdir -p "$DATA/sources"
curl -sSfL -o "$DATA/sources/gutenberg-23962-xiyouji.txt" https://www.gutenberg.org/ebooks/23962.txt.utf-8
echo "af3c9e408c0c58595b666ed9981b6fa1e9343f4bbc78309b1cb0818c32fc1f58  $DATA/sources/gutenberg-23962-xiyouji.txt" | shasum -a 256 -c -

# 3. Three corpora from git-tracked SMG sources: timed, warm-up, throughput.
#    CI reads those sources at upstream commit 33dd6c4e, so a branch that edits
#    them (a fix under test) keeps the same prompts. To do the same here, pass a
#    checkout of that commit as --repo.
CJK="$DATA/sources/gutenberg-23962-xiyouji.txt"
python3 build_corpus.py --repo ../../../.. --cjk "$CJK" --out "$DATA/corpus"
python3 build_corpus.py --repo ../../../.. --cjk "$CJK" --out "$DATA/corpus-warmup" --seed 99 --per-bucket 20 --stress-slices 3
python3 build_corpus.py --repo ../../../.. --cjk "$CJK" --out "$DATA/corpus-throughput" --seed 7 --per-bucket 100 --stress-slices 3

# 4. Everything else: builds, render, parity, timing, multi-turn, throughput,
#    memory, results.md. On a Mac, plug in (or pass --allow-battery).
python3 run_bench.py --data "$DATA" --legs 10
```

The results land in `$DATA/results/<run-id>/results.md`, next to the raw CSV and JSON files.

- **Script tests:** `python3 -m unittest`
- **Harness tests:** `cargo test -p llm-tokenizer --example encode_backends`. To include the test that needs a downloaded model, set `ENCODE_BENCH_MODEL_DIR` and add `-- --include-ignored`.

## How the numbers are made

- **Parity is a gate.** Every candidate is compared with SMG's ids for `add_special_tokens` false (chat) and true (embeddings) on:
  - all rendered prompts
  - a stress set: base64, minified JSON, emoji ZWJ, combining marks, literal special tokens, and 60–140 KB slices that cross fastokens' 64 KiB parallel threshold
  - growing chats replayed through SMG's caches
  - fastokens at several BPE thread counts

  Two controls check the harness. SMG against itself must show 0 mismatches. A copy that drops the last id must mismatch on every non-empty input.
- **Fresh prompts are the headline.** Each backend first encodes the warm-up corpus: the same sources as the timed corpus but different random draws. Its caches then hold common word pieces, as on a gateway that has served similar traffic, but the timed prompts themselves are new to the process.
  - fastokens keeps up to about 2M cached pretokens per thread; HF `tokenizers` stops adding to its cache after 10k. So exact repeats are timed separately and labelled.
- **Multi-turn chats:** every user turn re-sends the whole history, with chats interleaved round-robin. Each variant keeps one tokenizer and its caches for the whole run. This is where SMG's L1 prefix cache competes.
- **Timing method:**
  - Caches are off unless labelled, as in the gateway's default.
  - Each leg is one process: one encode per prompt per variant, with the variant order rotated per prompt.
  - Builds alternate in palindromic order (`z tok toksys hotdeps o2 o2 hotdeps ...`).
  - Per prompt, the median over legs; per bucket, the median and p90 across prompts. The p90 is spread over prompt sizes, not tail latency.
  - CPU time counts all threads, because fastokens can split one large encode across cores.
- **Throughput:** one pass over the throughput corpus, with N workers on one shared queue, so every prompt is encoded once. Legs alternate builds and variant order.
- **Memory:** peak RSS after loading each backend, and after a full-concurrency throughput pass. The second shows runtime cache growth. `smg_fastokens` is the proposed wrapper, which holds both tokenizers.
- **Environment:**
  - `FASTOKENS_BPE_THREADS` and `FASTOKENS_INPUT_CACHE` are read once per process. Default runs unset both.
  - `PCRE2_SYS_STATIC=1` (in `.cargo/config.toml`) builds the bundled PCRE2 into the binary, so every machine uses the same PCRE2 code.
  - The allocator is jemalloc, as in the gateway.
