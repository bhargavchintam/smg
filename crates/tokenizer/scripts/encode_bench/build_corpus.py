"""Build the deterministic prompt corpus for the encode-backend benchmark.

Writes four files into --out:
  conversations.jsonl  one chat per line: id, bucket, target_bytes, kind, messages, tools
  stress.jsonl         edge-case strings: id, category, text
  manifest.json        seed, buckets, counts, and the path + sha256 of every source file
  SHA256SUMS           sha256 of the three files above (`shasum -a 256` format)

Keep the corpus outside the repository and share SHA256SUMS, so anyone can
check they rebuilt exactly the same prompts.

Example:
  python3 build_corpus.py --repo ~/Downloads/oss/smg \
      --cjk ~/Downloads/oss/encode-bench-data/sources/gutenberg-23962-xiyouji.txt \
      --out ~/Downloads/oss/encode-bench-data/corpus
"""

import argparse
import base64
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

KINDS = ["code", "prose", "cjk", "json", "mixed"]
STRESS_CATEGORIES = [
    "tiny",
    "base64",
    "minified_json",
    "emoji_zwj",
    "combining_marks",
    "special_token_literals",
    "whitespace_runs",
    "long_word",
    "mixed_scripts",
    "crlf_tabs",
    "slice",
]
# (bucket name, content bytes). About 4 bytes per token for English and code.
DEFAULT_BUCKETS = [("100", 400), ("2k", 8_000), ("16k", 64_000), ("50k", 200_000)]
TOOLS_FROM_BYTES = 8_000
MAX_TURNS = 24
SEP = "\n\n"
CLOSING = "Please continue with the next step."

SHORT_SYSTEM = "You are a helpful assistant."
AGENT_SYSTEM = (
    "You are a senior engineer working inside a coding agent. Read every file, log and tool "
    "result the user shares, reason step by step, and call the provided tools when you need "
    "more context. Keep answers precise, cite the lines you rely on, and say clearly when "
    "something is uncertain. Reply in the language the user writes in."
)
USER_LEADS = {
    "code": ["Review this code and point out bugs:", "Explain what this function does:", "Refactor this for readability:"],
    "prose": ["Summarize the following notes:", "Rewrite this section more clearly:", "What are the key points here?"],
    "cjk": ["请总结下面这段文字：", "把下面的段落翻译成英文：", "这段话讲了什么？"],
    "json": ["Here is the tool output:", "The API returned this result:", "Parse this response and list the failures:"],
    "mixed": ["Here is more context for the task:", "Use this material to answer:", "Continue the analysis with this:"],
}
ASSISTANT_LEADS = ["Sure. Here is what I found:", "Understood. The relevant part:", "好的，相关内容如下：", "Here is the next piece:"]
WORDS = [
    "alpha", "router", "worker", "cache", "token", "prefill", "decode", "latency", "shard", "replica",
    "queue", "stream", "policy", "health", "metric", "gateway", "tenant", "budget", "retry", "timeout",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository and return its text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the repository root."},
                    "start_line": {"type": "integer", "description": "First line to return (1-based)."},
                    "end_line": {"type": "integer", "description": "Last line to return (inclusive)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the sandbox and return stdout, stderr and the exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."},
                    "timeout_s": {"type": "integer", "description": "Kill the command after this many seconds."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository with a regular expression and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "glob": {"type": "string", "description": "Only search files matching this glob."},
                    "max_results": {"type": "integer", "description": "Stop after this many matches."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Send an HTTP request to an internal service and return the response body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "url": {"type": "string", "description": "Absolute URL of the endpoint."},
                    "body": {"type": "object", "description": "JSON body for POST and PUT."},
                },
                "required": ["method", "url"],
            },
        },
    },
]
SPECIAL_TOKEN_LITERALS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|begin_of_text|>", "<|eot_id|>",
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>",
    "<think>", "</think>", "<tool_call>", "</tool_call>", "[INST]", "[/INST]", "<s>", "</s>",
]
MIXED_SCRIPT_PHRASES = [
    "مرحبا بالعالم، كيف حالك اليوم؟",
    "שלום עולם, מה שלומך היום?",
    "नमस्ते दुनिया, आज आप कैसे हैं?",
    "สวัสดีชาวโลกวันนี้คุณเป็นอย่างไรบ้าง",
    "Привет, мир! Как дела сегодня?",
    "Γειά σου κόσμε, τι κάνεις σήμερα;",
    "안녕하세요 세계, 오늘 어떠세요?",
    "こんにちは世界、今日はお元気ですか？",
]


def _nbytes(text):
    return len(text.encode("utf-8"))


def _truncate(text, max_bytes):
    """Longest prefix of text that fits in max_bytes (never splits a character)."""
    return text.encode("utf-8")[:max(0, max_bytes)].decode("utf-8", "ignore")


def _fit(text, size):
    """Cut or pad (with spaces) text to exactly `size` UTF-8 bytes."""
    text = _truncate(text, size)
    return text + " " * (size - _nbytes(text))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strip_gutenberg(text):
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    begin = start.end() if start else 0
    finish = end.start() if end and end.start() > begin else len(text)
    return text[begin:finish]


def _blocks(files, min_bytes, split):
    """Split every file into non-trivial blocks, keeping file order stable."""
    blocks = []
    for path in sorted(str(p) for p in files):
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for block in split(text):
            block = block.strip()
            if _nbytes(block) >= min_bytes:
                blocks.append(block)
    if not blocks:
        raise ValueError(f"no usable text in {list(files)}")
    return blocks


def _load_pools(code_files, prose_files, cjk_files):
    paragraphs = lambda text: re.split(r"\n\s*\n", text)  # noqa: E731
    return {
        "code": _blocks(code_files, 40, paragraphs),
        "prose": _blocks(prose_files, 40, paragraphs),
        "cjk": _blocks(cjk_files, 20, lambda text: _strip_gutenberg(text).splitlines()),
    }


def _consecutive(pool, rng, low, high, joiner):
    start = rng.randrange(len(pool))
    return joiner.join(pool[start : start + rng.randint(low, high)])


def _json_chunk(rng):
    items = [
        {
            "id": rng.randrange(10**6),
            "name": f"{rng.choice(WORDS)}_{rng.randrange(1000)}",
            "status": rng.choice(["ok", "failed", "pending"]),
            "latency_ms": round(rng.uniform(0.1, 900.0), 3),
            "tags": rng.sample(WORDS, 3),
        }
        for _ in range(rng.randint(3, 12))
    ]
    obj = {
        "request_id": f"req-{rng.getrandbits(64):016x}",
        "items": items,
        "next_cursor": base64.b64encode(rng.getrandbits(96).to_bytes(12, "big")).decode(),
    }
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"


def _drawers(pools, rng):
    draw = {
        "code": lambda: _consecutive(pools["code"], rng, 1, 3, SEP),
        "prose": lambda: _consecutive(pools["prose"], rng, 1, 3, SEP),
        "cjk": lambda: _consecutive(pools["cjk"], rng, 3, 8, "\n"),
        "json": lambda: _json_chunk(rng),
    }
    draw["mixed"] = lambda: draw[rng.choice(["code", "prose", "cjk", "json"])]()
    return draw


def _fill(size, lead, draw):
    """Text of exactly `size` bytes: the lead line followed by drawn chunks."""
    text = lead
    while _nbytes(text) < size:
        text += SEP + draw()
    return _fit(text, size)


def _conversation(conv_id, bucket, target, kind, pools, rng):
    system = AGENT_SYSTEM if target >= TOOLS_FROM_BYTES else SHORT_SYSTEM
    n_turns = max(1, min(MAX_TURNS, target // TOOLS_FROM_BYTES))
    # An odd number of turns ends on a user turn, so the closing line joins it;
    # otherwise the closing line is a user message of its own.
    closing_cost = _nbytes(SEP + CLOSING) if n_turns % 2 else _nbytes(CLOSING)
    body = target - _nbytes(system) - closing_cost
    if body < n_turns:
        raise ValueError(f"bucket {bucket} ({target} bytes) is too small for a conversation")
    base = body // n_turns
    sizes = [base] * (n_turns - 1) + [body - base * (n_turns - 1)]

    draw = _drawers(pools, rng)[kind]
    messages = [{"role": "system", "content": system}]
    for turn, size in enumerate(sizes):
        if turn % 2 == 0:
            messages.append({"role": "user", "content": _fill(size, rng.choice(USER_LEADS[kind]), draw)})
        else:
            messages.append({"role": "assistant", "content": _fill(size, rng.choice(ASSISTANT_LEADS), draw)})
    if n_turns % 2:
        messages[-1]["content"] += SEP + CLOSING
    else:
        messages.append({"role": "user", "content": CLOSING})
    return {
        "id": conv_id,
        "bucket": bucket,
        "target_bytes": target,
        "kind": kind,
        "messages": messages,
        "tools": TOOLS if target >= TOOLS_FROM_BYTES else [],
    }


def _stress(pools, rng, slices):
    def items(category, texts):
        return [{"id": f"stress-{category}-{i}", "category": category, "text": t} for i, t in enumerate(texts)]

    big = {
        "rows": [
            {"k": f"{rng.choice(WORDS)}{i}", "v": rng.random(), "n": rng.randrange(10**9), "s": rng.sample(WORDS, 4)}
            for i in range(700)
        ],
        "note": "中文说明：这是压缩后的 JSON。 emoji 🙂 and quotes \" \\ /",
    }
    b64 = [base64.b64encode(rng.getrandbits(8 * n).to_bytes(n, "big")).decode() for n in (12_000, 72_000)]
    code = pools["code"][rng.randrange(len(pools["code"]))]
    long_doc = []
    while sum(_nbytes(part) for part in long_doc) < 300_000:
        long_doc.append(_drawers(pools, rng)[rng.choice(["code", "prose", "cjk"])]())
    doc = SEP.join(long_doc)
    step = 80_000 // max(1, slices - 1)
    slice_texts = []
    for i in range(slices):
        start = rng.randrange(len(doc) // 2)
        slice_texts.append(_fit(doc[start:], 60_000 + i * step))

    out = []
    out += items("tiny", ["", " ", "a", "\n", "中", "🙂"])
    out += items("base64", b64 + ["\n".join(b64[0][i : i + 76] for i in range(0, len(b64[0]), 76))])
    out += items(
        "minified_json",
        [json.dumps(big, separators=(",", ":"), ensure_ascii=False), json.dumps(big, separators=(",", ":"))],
    )
    out += items(
        "emoji_zwj",
        [
            " ".join(["👨‍👩‍👧‍👦", "🏳️‍🌈", "👩🏽‍💻", "🇮🇳🇺🇸🇯🇵", "1️⃣2️⃣3️⃣", "🧑🏿‍🚀"] * 200),
            "".join(["👨‍👩‍👧‍👦🏳️‍🌈👩🏽‍💻"] * 300),
        ],
    )
    out += items(
        "combining_marks",
        [
            "é" * 2000 + " " + "äöü" * 500,
            "Z͓͑͒a͔͕l͖͗g͙͘o͚ " * 400,
            "Tiếng Việt có nhiều dấu thanh. " * 200 + "हिन्दी भाषा " * 200 + "각" * 300,
        ],
    )
    out += items(
        "special_token_literals",
        [
            " ".join(SPECIAL_TOKEN_LITERALS * 20),
            "".join(SPECIAL_TOKEN_LITERALS * 20),
            " x ".join(f"word{tok}word" for tok in SPECIAL_TOKEN_LITERALS * 10),
        ],
    )
    out += items(
        "whitespace_runs",
        [
            "".join(" " * n + "x" for n in range(1, 65)) + "\t\t\tx\n\n\n\nx\r\n\r\nx   ",
            " x　y​z w v " * 300,
        ],
    )
    out += items("long_word", ["a" * 20_000, "ab" * 10_000, "1234567890" * 2_000])
    out += items("mixed_scripts", [" ".join(MIXED_SCRIPT_PHRASES * 60), "".join(MIXED_SCRIPT_PHRASES * 60)])
    out += items("crlf_tabs", [code.replace("    ", "\t").replace("\n", "\r\n")])
    out += items("slice", slice_texts)
    return out


def _write_jsonl(path, rows):
    # ASCII-only JSON: raw U+2028/U+2029 (and similar) would break line-based readers.
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_corpus(out_dir, code_files, prose_files, cjk_files, seed=1234, per_bucket=20, buckets=None, stress_slices=12):
    buckets = list(buckets or DEFAULT_BUCKETS)
    pools = _load_pools(code_files, prose_files, cjk_files)
    conversations = []
    for bucket, target in buckets:
        for i in range(per_bucket):
            rng = random.Random(f"{seed}-{bucket}-{i}")
            kind = KINDS[i % len(KINDS)]
            conversations.append(_conversation(f"{bucket}-{i:03d}-{kind}", bucket, target, kind, pools, rng))
    stress = _stress(pools, random.Random(f"{seed}-stress"), stress_slices)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "conversations.jsonl", conversations)
    _write_jsonl(out_dir / "stress.jsonl", stress)
    manifest = {
        "seed": seed,
        "buckets": buckets,
        "per_bucket": per_bucket,
        "stress_slices": stress_slices,
        "counts": {"conversations": len(conversations), "stress": len(stress)},
        "sources": [
            {"path": str(p), "sha256": _sha256(p)} for p in sorted(str(p) for p in [*code_files, *prose_files, *cjk_files])
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sums = [f"{_sha256(out_dir / name)}  {name}" for name in ("conversations.jsonl", "stress.jsonl", "manifest.json")]
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return manifest


def _repo_sources(repo):
    """Git-tracked Rust sources and Markdown docs only, so any clean checkout of
    the same commit rebuilds exactly the same corpus (local notes never leak in)."""
    repo = Path(repo)
    listing = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], check=True, capture_output=True).stdout
    files = [f for f in listing.decode("utf-8").split("\0") if f and "scripts/encode_bench/" not in f]
    code = [f for f in files if f.endswith(".rs") and re.match(r"(crates/[^/]+/src|model_gateway/src)/", f)]
    prose = [f for f in files if f.endswith(".md")]
    return sorted(repo / f for f in code), sorted(repo / f for f in prose)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="SMG checkout: its Rust sources and Markdown docs feed the corpus")
    parser.add_argument("--cjk", required=True, nargs="+", help="Chinese/Japanese plain-text files (public domain)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--per-bucket", type=int, default=20)
    parser.add_argument("--stress-slices", type=int, default=12)
    args = parser.parse_args()
    code, prose = _repo_sources(args.repo)
    manifest = build_corpus(
        out_dir=args.out,
        code_files=code,
        prose_files=prose,
        cjk_files=[Path(p) for p in args.cjk],
        seed=args.seed,
        per_bucket=args.per_bucket,
        stress_slices=args.stress_slices,
    )
    print(json.dumps(manifest["counts"]), f"-> {args.out}")


if __name__ == "__main__":
    main()
