# Encode-backend benchmark

Measures how fast SMG turns a rendered prompt into token ids, as shipped and
with faster alternatives, and checks that every alternative returns exactly
the same ids as SMG.

- **Harness:** `crates/tokenizer/examples/encode_backends.rs`, a Rust example with the subcommands `render`, `parity`, `time`, `throughput` and `load`.
- **Scripts:** Python 3.8+, standard library only.

## Backends compared

| variant | what it is |
|---|---|
| `smg` | `HuggingFaceTokenizer::encode` as shipped (full HF `encode`, offsets included) |
| `hf_encode` / `hf_encode_fast` | raw HF `tokenizers` `encode` / `encode_fast` (SMG only reads the ids) |
| `fastokens` | [fastokens](https://crates.io/crates/fastokens) 0.3.2 built from the raw `tokenizer.json` |

The harness builds three binaries:

- **`z`:** the release profile SMG ships (`opt-level = "z"`).
- **`hotdeps`:** the same, plus opt-level 2 for the crates on HF's encode path.
- **`o2`:** everything at opt-level 2, as an upper bound.

## Run it

```bash
DATA=~/Downloads/oss/encode-bench-data
cd crates/tokenizer/scripts/encode_bench

# 1. tokenizer files at pinned revisions (Qwen3, DeepSeek-V3.2, Kimi-K2)
python3 fetch_models.py --out $DATA/models

# 2. corpus: SMG's own sources and docs, plus a public-domain Chinese text
#    (Project Gutenberg #23962); keep it outside the repo
python3 build_corpus.py --repo ../../../.. \
    --cjk $DATA/sources/gutenberg-23962-xiyouji.txt --out $DATA/corpus

# 3. everything else: builds, render, parity, timing, throughput, load, results.md
python3 run_bench.py --data $DATA --legs 10
```

The results land in `$DATA/results/<run-id>/results.md`. To check the scripts' own tests, run `python3 -m unittest`, and for the harness, `cargo test -p llm-tokenizer --example encode_backends`. The harness has one test that needs a downloaded model; set `ENCODE_BENCH_MODEL_DIR` and add `-- --include-ignored` to run it.

## How the numbers are made

- **Parity is a gate.** Every candidate is compared with SMG's ids for `add_special_tokens` false (chat) and true (embeddings) on:
  - all rendered prompts
  - a stress set: base64, minified JSON, emoji ZWJ, combining marks, literal special tokens, and 60–140 KB slices that cross fastokens' 64 KiB parallel threshold
  - growing chats replayed through SMG's L0/L1 caches
- **Two controls check the harness.** SMG against itself must show 0 mismatches. A copy that drops the last id must mismatch on every non-empty input.
- **Timing:**
  - Caches are off, as in the gateway's default.
  - Each leg is one process: one encode per prompt per variant, with the variant order rotated per prompt.
  - Builds alternate in palindromic order (`z hotdeps o2 o2 hotdeps z ...`).
  - Per prompt, the median over legs; per bucket, the median and p90 over prompts.
  - CPU time counts all threads. fastokens can split one large encode across cores, so compare CPU ms as well as wall ms.
- **Environment variables:** `FASTOKENS_BPE_THREADS` and `FASTOKENS_INPUT_CACHE` are read once per process. Default runs unset both; one extra run pins fastokens to 1 thread.
- **PCRE2:** `PCRE2_SYS_STATIC=1` (in `.cargo/config.toml`) builds the bundled PCRE2 into the binary, so every machine uses the same PCRE2 code.
