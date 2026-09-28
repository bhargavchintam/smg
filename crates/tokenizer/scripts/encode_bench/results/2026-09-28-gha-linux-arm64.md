# Encode benchmark results (ubuntu-24.04-arm-36398002868)

## Setup

- **machine**: Neoverse-N2 (4 logical, aarch64)
- **os**: Linux 6.17.0-1022-azure
- **power**: n/a
- **rustc**: rustc 1.98.0 (88d9e12ae 2026-08-18)
- **smg_commit**: 896071322ab95c72863138b1d6764f88bc0cfd2f
- **fastokens**: 0.3.2
- **tokenizers**: 0.23.1
- **corpus_sha256**: {"corpus": ["b5050a08bf6775c3c2666658ac0ee7dad765d9f1a4c1088325042f0054e5b887  conversations.jsonl", "e98a7fe74c51af61e69f4da2b6cf4726c89c679836d09579cdcead026f10c25b  stress.jsonl"], "corpus-warmup": ["d53964f8a38c1a7beebcffb7fad973f84b2c51cb2100ffcd903ba577096d4797  conversations.jsonl", "2b2cc0d9799f37624f7c068472a72d04f8e654b1b434402e1f3db3bf7734dfdc  stress.jsonl"], "corpus-throughput": ["3de5bd7ff4378daabe38d5af4fb9399b6d598fe4e45070eaa317688672c335f3  conversations.jsonl", "66e2ffb8b6911a16a734e6ffd6d7aa35547f14818d802e518227397c730bd68d  stress.jsonl"]}
- **rounds**: 6 legs per build, 1 round per leg, palindromic build order
- **builds**: {"z": "(release profile as shipped)", "tok": "--config profile.release.package.tokenizers.opt-level=2", "toksys": "--config profile.release.package.tokenizers.opt-level=2 --config profile.release.package.onig.opt-level=2 --config profile.release.package.onig_sys.opt-level=2", "hotdeps": "--config profile.release.package.tokenizers.opt-level=2 --config profile.release.package.onig.opt-level=2 --config profile.release.package.onig_sys.opt-level=2 --config profile.release.package.daachorse.opt-level=2 --config profile.release.package.dary_heap.opt-level=2 --config profile.release.package.ahash.opt-level=2 --config profile.release.package.aho-corasick.opt-level=2 --config profile.release.package.compact_str.opt-level=2 --config profile.release.package.regex.opt-level=2 --config profile.release.package.unicode-normalization-alignments.opt-level=2 --config profile.release.package.unicode-segmentation.opt-level=2 --config profile.release.package.unicode_categories.opt-level=2", "o2": "--config profile.release.opt-level=2"}
- **model deepseek-v3.2**: a7e62ac04ecb2c0a54d736dc46601c5606cf10a6
- **model qwen3**: cdbee75f17c01a7cc42f958dc650907174af0554

## deepseek-v3.2

### Token-id parity against SMG (mismatches / inputs)

| build | model | fastokens threads | candidate | chat (add_special_tokens=false) | embeddings (true) |
|---|---|---|---|---|---|
| hotdeps | deepseek-ai/DeepSeek-V3.2 | default | fastokens | 0/119 | 0/119 |
| hotdeps | deepseek-ai/DeepSeek-V3.2 | default | fastokens_file | 0/119 | 0/119 |
| hotdeps | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode | 0/119 | 0/119 |
| hotdeps | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| hotdeps | deepseek-ai/DeepSeek-V3.2 | default | fastokens_cached (cache replay) | 0/72 | — |
| hotdeps | deepseek-ai/DeepSeek-V3.2 | default | smg_cached (cache replay) | 0/72 | — |
| o2 | deepseek-ai/DeepSeek-V3.2 | default | fastokens | 0/119 | 0/119 |
| o2 | deepseek-ai/DeepSeek-V3.2 | default | fastokens_file | 0/119 | 0/119 |
| o2 | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode | 0/119 | 0/119 |
| o2 | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| o2 | deepseek-ai/DeepSeek-V3.2 | default | fastokens_cached (cache replay) | 0/72 | — |
| o2 | deepseek-ai/DeepSeek-V3.2 | default | smg_cached (cache replay) | 0/72 | — |
| tok | deepseek-ai/DeepSeek-V3.2 | default | fastokens | 0/119 | 0/119 |
| tok | deepseek-ai/DeepSeek-V3.2 | default | fastokens_file | 0/119 | 0/119 |
| tok | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode | 0/119 | 0/119 |
| tok | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| tok | deepseek-ai/DeepSeek-V3.2 | default | fastokens_cached (cache replay) | 0/72 | — |
| tok | deepseek-ai/DeepSeek-V3.2 | default | smg_cached (cache replay) | 0/72 | — |
| toksys | deepseek-ai/DeepSeek-V3.2 | default | fastokens | 0/119 | 0/119 |
| toksys | deepseek-ai/DeepSeek-V3.2 | default | fastokens_file | 0/119 | 0/119 |
| toksys | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode | 0/119 | 0/119 |
| toksys | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| toksys | deepseek-ai/DeepSeek-V3.2 | default | fastokens_cached (cache replay) | 0/72 | — |
| toksys | deepseek-ai/DeepSeek-V3.2 | default | smg_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 1 | fastokens | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 1 | fastokens_file | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 1 | fastokens_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 1 | smg_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 2 | fastokens | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 2 | fastokens_file | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 2 | fastokens_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 2 | smg_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 3 | fastokens | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 3 | fastokens_file | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 3 | fastokens_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 3 | smg_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 4 | fastokens | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 4 | fastokens_file | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | 4 | fastokens_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | 4 | smg_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | default | fastokens | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | default | fastokens_file | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| z | deepseek-ai/DeepSeek-V3.2 | default | fastokens_cached (cache replay) | 0/72 | — |
| z | deepseek-ai/DeepSeek-V3.2 | default | smg_cached (cache replay) | 0/72 | — |

`id_to_token` differences over the whole vocabulary (SMG vs fastokens): 0

### Encode latency on fresh prompts, median ms per prompt (speed-up), caches off

Every backend first encodes a separate warm-up prompt set, so its caches hold common word pieces but never the timed prompt.

| bucket | tokens (median) | SMG as shipped | SMG, tokenizers at O2 | SMG, tokenizers+onig at O2 | SMG, HF hot deps at O2 | SMG, all O2 | HF encode_fast | fastokens | fastokens, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 109 | 0.25 | 0.18 (1.4×) | 0.17 (1.4×) | 0.17 (1.5×) | 0.16 (1.6×) | 0.25 (1.0×) | 0.06 (4.0×) | 0.04 (5.4×) |
| 2k | 2,930 | 5.28 | 3.71 (1.4×) | 3.58 (1.5×) | 3.60 (1.5×) | 3.33 (1.6×) | 5.15 (1.0×) | 0.92 (5.7×) | 0.80 (6.3×) |
| 16k | 18,922 | 31.57 | 22.36 (1.4×) | 21.56 (1.5×) | 21.82 (1.4×) | 20.40 (1.5×) | 30.75 (1.0×) | 3.81 (8.3×) | 4.20 (7.2×) |
| 50k | 58,296 | 95.36 | 66.22 (1.4×) | 65.01 (1.5×) | 65.21 (1.5×) | 60.85 (1.6×) | 92.10 (1.0×) | 10.08 (9.5×) | 12.07 (7.6×) |

### p90 across the bucket's prompts (ms; spread over prompt sizes, not tail latency)

| bucket | tokens (median) | SMG as shipped | SMG, tokenizers at O2 | SMG, tokenizers+onig at O2 | SMG, HF hot deps at O2 | SMG, all O2 | HF encode_fast | fastokens | fastokens, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 109 | 0.28 | 0.19 (1.4×) | 0.20 (1.4×) | 0.20 (1.4×) | 0.18 (1.5×) | 0.27 (1.0×) | 0.12 (2.3×) | 0.08 (3.3×) |
| 2k | 2,930 | 5.42 | 3.86 (1.4×) | 3.70 (1.5×) | 3.67 (1.5×) | 3.48 (1.6×) | 5.28 (1.0×) | 1.46 (3.7×) | 1.38 (3.7×) |
| 16k | 18,922 | 33.72 | 23.69 (1.4×) | 22.72 (1.5×) | 22.81 (1.5×) | 21.18 (1.6×) | 33.48 (1.0×) | 4.78 (7.1×) | 7.87 (4.2×) |
| 50k | 58,296 | 106.07 | 74.86 (1.4×) | 71.57 (1.5×) | 72.21 (1.5×) | 67.18 (1.6×) | 105.66 (1.0×) | 11.20 (9.5×) | 21.24 (4.9×) |

### CPU ms per prompt (all threads; fastokens can split one large encode across cores)

| bucket | tokens (median) | SMG as shipped | SMG, tokenizers at O2 | SMG, tokenizers+onig at O2 | SMG, HF hot deps at O2 | SMG, all O2 | HF encode_fast | fastokens | fastokens, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 109 | 0.25 | 0.18 (1.4×) | 0.17 (1.4×) | 0.17 (1.5×) | 0.16 (1.6×) | 0.25 (1.0×) | 0.06 (4.0×) | 0.04 (5.4×) |
| 2k | 2,930 | 5.28 | 3.71 (1.4×) | 3.58 (1.5×) | 3.61 (1.5×) | 3.33 (1.6×) | 5.15 (1.0×) | 0.92 (5.7×) | 0.80 (6.3×) |
| 16k | 18,922 | 31.59 | 22.36 (1.4×) | 21.57 (1.5×) | 21.83 (1.4×) | 20.43 (1.5×) | 30.75 (1.0×) | 4.79 (6.6×) | 4.20 (7.2×) |
| 50k | 58,296 | 95.36 | 66.23 (1.4×) | 65.00 (1.5×) | 65.22 (1.5×) | 60.86 (1.6×) | 92.10 (1.0×) | 12.56 (7.6×) | 12.07 (7.6×) |

### Stability: spread of per-leg medians

| column | largest spread (any bucket) | median spread |
|---|---|---|
| SMG as shipped | 9.5% | 5.8% |
| SMG, tokenizers at O2 | 11.5% | 6.7% |
| SMG, tokenizers+onig at O2 | 8.5% | 6.8% |
| SMG, HF hot deps at O2 | 9.4% | 4.1% |
| SMG, all O2 | 7.3% | 5.1% |
| HF encode_fast | 6.5% | 5.0% |
| fastokens | 19.2% | 6.1% |
| fastokens, 1 thread | 14.0% | 9.5% |

### Warm caches: exact repeats (warm-up on the timed prompts themselves), median ms

| bucket | tokens (median) | SMG as shipped | HF encode_fast | fastokens |
|---|---|---|---|---|
| 100 | 109 | 0.17 | 0.16 (1.0×) | 0.04 (4.2×) |
| 2k | 2,930 | 4.46 | 4.39 (1.0×) | 0.75 (5.9×) |
| 16k | 18,922 | 30.63 | 29.87 (1.0×) | 3.38 (9.1×) |
| 50k | 58,296 | 91.89 | 90.26 (1.0×) | 8.31 (11.1×) |

### Growing chats (multi-turn): median ms per turn, every turn re-sends the whole history

| bucket | tokens (median) | SMG as shipped | SMG + L0/L1 caches | fastokens | fastokens + L0/L1 caches |
|---|---|---|---|---|---|
| 16k | 12,158 | 19.89 | 27.12 (0.7×) | 3.10 (6.4×) | 10.14 (2.0×) |
| 50k | 30,766 | 52.08 | 71.03 (0.7×) | 5.09 (10.2×) | 22.47 (2.3×) |

### Throughput: one pass over fresh prompts, N workers on one shared queue (median over legs)

| build | variant | fastokens threads | workers | M tokens/s | CPU/wall | legs | prompts |
|---|---|---|---|---|---|---|---|
| hotdeps | fastokens | default | 1 | 5.10 | 1.39 | 2 | 400 |
| hotdeps | fastokens | default | 4 | 14.14 | 3.43 | 2 | 400 |
| hotdeps | smg | default | 1 | 0.89 | 1.00 | 2 | 400 |
| hotdeps | smg | default | 4 | 3.48 | 3.93 | 2 | 400 |
| z | fastokens | default | 1 | 4.96 | 1.40 | 2 | 400 |
| z | fastokens | default | 4 | 13.53 | 3.44 | 2 | 400 |
| z | smg | default | 1 | 0.60 | 1.00 | 2 | 400 |
| z | smg | default | 4 | 2.33 | 3.94 | 2 | 400 |

### Load time and memory (MB above an empty process)

| backend | load ms (median) | after load | after a full-concurrency pass |
|---|---|---|---|
| none | 0 | 0 | — |
| smg | 273 | 67 | 307 |
| hf_encode | 205 | 67 | — |
| fastokens | 794 | 102 | 256 |
| smg_fastokens | 1083 | 151 | — |
| smg_cached | — | — | 368 |
| fastokens_cached | — | — | 317 |

## qwen3

### Token-id parity against SMG (mismatches / inputs)

| build | model | fastokens threads | candidate | chat (add_special_tokens=false) | embeddings (true) |
|---|---|---|---|---|---|
| hotdeps | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens | 0/119 | 0/119 |
| hotdeps | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_file | 0/119 | 0/119 |
| hotdeps | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode | 0/119 | 0/119 |
| hotdeps | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| hotdeps | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_cached (cache replay) | 0/72 | — |
| hotdeps | Qwen/Qwen3-4B-Instruct-2507 | default | smg_cached (cache replay) | 0/72 | — |
| o2 | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens | 0/119 | 0/119 |
| o2 | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_file | 0/119 | 0/119 |
| o2 | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode | 0/119 | 0/119 |
| o2 | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| o2 | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_cached (cache replay) | 0/72 | — |
| o2 | Qwen/Qwen3-4B-Instruct-2507 | default | smg_cached (cache replay) | 0/72 | — |
| tok | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens | 0/119 | 0/119 |
| tok | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_file | 0/119 | 0/119 |
| tok | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode | 0/119 | 0/119 |
| tok | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| tok | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_cached (cache replay) | 0/72 | — |
| tok | Qwen/Qwen3-4B-Instruct-2507 | default | smg_cached (cache replay) | 0/72 | — |
| toksys | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens | 0/119 | 0/119 |
| toksys | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_file | 0/119 | 0/119 |
| toksys | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode | 0/119 | 0/119 |
| toksys | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| toksys | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_cached (cache replay) | 0/72 | — |
| toksys | Qwen/Qwen3-4B-Instruct-2507 | default | smg_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 1 | fastokens | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 1 | fastokens_file | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 1 | fastokens_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 1 | smg_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 2 | fastokens | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 2 | fastokens_file | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 2 | fastokens_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 2 | smg_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 3 | fastokens | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 3 | fastokens_file | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 3 | fastokens_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 3 | smg_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 4 | fastokens | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 4 | fastokens_file | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | 4 | fastokens_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | 4 | smg_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_file | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | default | hf_raw_encode_fast | 0/119 | 0/119 |
| z | Qwen/Qwen3-4B-Instruct-2507 | default | fastokens_cached (cache replay) | 0/72 | — |
| z | Qwen/Qwen3-4B-Instruct-2507 | default | smg_cached (cache replay) | 0/72 | — |

`id_to_token` differences over the whole vocabulary (SMG vs fastokens): 0

### Encode latency on fresh prompts, median ms per prompt (speed-up), caches off

Every backend first encodes a separate warm-up prompt set, so its caches hold common word pieces but never the timed prompt.

| bucket | tokens (median) | SMG as shipped | SMG, tokenizers at O2 | SMG, tokenizers+onig at O2 | SMG, HF hot deps at O2 | SMG, all O2 | HF encode_fast | fastokens | fastokens, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 128 | 0.25 | 0.17 (1.4×) | 0.17 (1.4×) | 0.16 (1.5×) | 0.16 (1.6×) | 0.24 (1.0×) | 0.04 (5.6×) | 0.03 (8.8×) |
| 2k | 3,043 | 4.87 | 3.30 (1.5×) | 3.18 (1.5×) | 3.14 (1.5×) | 2.95 (1.7×) | 4.80 (1.0×) | 0.60 (8.1×) | 0.45 (10.7×) |
| 16k | 20,374 | 31.93 | 22.21 (1.4×) | 21.37 (1.5×) | 21.23 (1.5×) | 19.99 (1.6×) | 31.79 (1.0×) | 2.60 (12.3×) | 2.68 (11.6×) |
| 50k | 63,446 | 94.02 | 65.54 (1.4×) | 63.18 (1.5×) | 62.99 (1.5×) | 59.59 (1.6×) | 92.75 (1.0×) | 5.49 (17.1×) | 8.11 (11.6×) |

### p90 across the bucket's prompts (ms; spread over prompt sizes, not tail latency)

| bucket | tokens (median) | SMG as shipped | SMG, tokenizers at O2 | SMG, tokenizers+onig at O2 | SMG, HF hot deps at O2 | SMG, all O2 | HF encode_fast | fastokens | fastokens, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 128 | 0.29 | 0.20 (1.4×) | 0.21 (1.4×) | 0.20 (1.4×) | 0.19 (1.5×) | 0.29 (1.0×) | 0.07 (4.3×) | 0.04 (6.7×) |
| 2k | 3,043 | 5.08 | 3.54 (1.4×) | 3.46 (1.5×) | 3.38 (1.5×) | 3.22 (1.6×) | 4.97 (1.0×) | 0.73 (6.9×) | 0.65 (7.7×) |
| 16k | 20,374 | 33.30 | 22.60 (1.5×) | 22.14 (1.5×) | 21.73 (1.5×) | 20.65 (1.6×) | 33.53 (1.0×) | 2.74 (12.1×) | 3.35 (9.8×) |
| 50k | 63,446 | 106.64 | 73.52 (1.5×) | 70.57 (1.5×) | 70.59 (1.5×) | 66.31 (1.6×) | 106.06 (1.0×) | 7.16 (14.9×) | 9.41 (11.3×) |

### CPU ms per prompt (all threads; fastokens can split one large encode across cores)

| bucket | tokens (median) | SMG as shipped | SMG, tokenizers at O2 | SMG, tokenizers+onig at O2 | SMG, HF hot deps at O2 | SMG, all O2 | HF encode_fast | fastokens | fastokens, 1 thread |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 128 | 0.25 | 0.18 (1.4×) | 0.18 (1.4×) | 0.16 (1.5×) | 0.16 (1.6×) | 0.25 (1.0×) | 0.04 (5.6×) | 0.03 (8.6×) |
| 2k | 3,043 | 4.86 | 3.30 (1.5×) | 3.18 (1.5×) | 3.15 (1.5×) | 2.95 (1.6×) | 4.81 (1.0×) | 0.60 (8.1×) | 0.45 (10.7×) |
| 16k | 20,374 | 31.97 | 22.21 (1.4×) | 21.39 (1.5×) | 21.24 (1.5×) | 19.94 (1.6×) | 31.78 (1.0×) | 3.25 (9.8×) | 2.68 (11.6×) |
| 50k | 63,446 | 93.95 | 65.55 (1.4×) | 63.18 (1.5×) | 63.01 (1.5×) | 59.60 (1.6×) | 92.75 (1.0×) | 8.76 (10.7×) | 8.11 (11.6×) |

### Stability: spread of per-leg medians

| column | largest spread (any bucket) | median spread |
|---|---|---|
| SMG as shipped | 6.2% | 4.0% |
| SMG, tokenizers at O2 | 15.8% | 6.6% |
| SMG, tokenizers+onig at O2 | 15.1% | 6.5% |
| SMG, HF hot deps at O2 | 12.9% | 6.7% |
| SMG, all O2 | 12.5% | 6.8% |
| HF encode_fast | 7.6% | 3.8% |
| fastokens | 9.2% | 6.3% |
| fastokens, 1 thread | 11.1% | 6.2% |

### Warm caches: exact repeats (warm-up on the timed prompts themselves), median ms

| bucket | tokens (median) | SMG as shipped | HF encode_fast | fastokens |
|---|---|---|---|---|
| 100 | 128 | 0.18 | 0.18 (1.0×) | 0.03 (7.3×) |
| 2k | 3,043 | 4.21 | 4.13 (1.0×) | 0.42 (10.0×) |
| 16k | 20,374 | 31.12 | 30.84 (1.0×) | 2.46 (12.6×) |
| 50k | 63,446 | 93.05 | 92.11 (1.0×) | 5.45 (17.1×) |

### Growing chats (multi-turn): median ms per turn, every turn re-sends the whole history

| bucket | tokens (median) | SMG as shipped | SMG + L0/L1 caches | fastokens | fastokens + L0/L1 caches |
|---|---|---|---|---|---|
| 16k | 12,877 | 19.81 | 20.10 (1.0×) | 2.40 (8.2×) | 2.72 (7.3×) |
| 50k | 32,136 | 49.02 | 49.90 (1.0×) | 3.70 (13.3×) | 4.42 (11.1×) |

### Throughput: one pass over fresh prompts, N workers on one shared queue (median over legs)

| build | variant | fastokens threads | workers | M tokens/s | CPU/wall | legs | prompts |
|---|---|---|---|---|---|---|---|
| hotdeps | fastokens | default | 1 | 7.63 | 1.38 | 2 | 400 |
| hotdeps | fastokens | default | 4 | 19.84 | 3.43 | 2 | 400 |
| hotdeps | smg | default | 1 | 0.97 | 1.00 | 2 | 400 |
| hotdeps | smg | default | 4 | 3.82 | 3.95 | 2 | 400 |
| z | fastokens | default | 1 | 7.60 | 1.37 | 2 | 400 |
| z | fastokens | default | 4 | 20.01 | 3.46 | 2 | 400 |
| z | smg | default | 1 | 0.64 | 1.00 | 2 | 400 |
| z | smg | default | 4 | 2.48 | 3.93 | 2 | 400 |

### Load time and memory (MB above an empty process)

| backend | load ms (median) | after load | after a full-concurrency pass |
|---|---|---|---|
| none | 0 | 0 | — |
| smg | 427 | 150 | 342 |
| hf_encode | 318 | 149 | — |
| fastokens | 1015 | 190 | 262 |
| smg_fastokens | 1444 | 235 | — |
| smg_cached | — | — | 411 |
| fastokens_cached | — | — | 326 |
