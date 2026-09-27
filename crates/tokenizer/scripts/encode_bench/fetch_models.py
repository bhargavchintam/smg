"""Download tokenizer files for the benchmark models at pinned revisions.

Each model lands in <out>/<org>__<name>@<commit12>/ with a fetch.json that
records the full commit and the sha256 of every file, so a run can always be
traced back to the exact tokenizer it used.

Example:
  python3 fetch_models.py --out ~/Downloads/oss/encode-bench-data/models
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "deepseek-ai/DeepSeek-V3.2", "moonshotai/Kimi-K2-Instruct"]
WANTED = {
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tiktoken.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def wanted_files(siblings):
    """Tokenizer, template and config files; never weights or custom Python code."""
    keep = [
        f
        for f in siblings
        if f in WANTED or f.endswith(".tiktoken") or (f.startswith("encoding/") and f.endswith(".py"))
    ]
    return sorted(keep)


def model_dir_name(model_id, sha):
    return f"{model_id.replace('/', '__')}@{sha[:12]}"


def file_url(model_id, sha, filename):
    return f"https://huggingface.co/{model_id}/resolve/{sha}/{filename}"


def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https host)
        return json.load(resp)


def fetch(model_id, out_root, revision="main"):
    info = _get_json(f"https://huggingface.co/api/models/{model_id}/revision/{revision}")
    sha = info["sha"]
    dest = Path(out_root) / model_dir_name(model_id, sha)
    files = wanted_files([s["rfilename"] for s in info.get("siblings", [])])
    record = {"model": model_id, "revision": sha, "files": {}}
    for name in files:
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(file_url(model_id, sha, name), target)  # noqa: S310
        record["files"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest / "fetch.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", action="append", help="Hugging Face model id (repeatable); default: the benchmark set")
    args = parser.parse_args()
    for model_id in args.model or MODELS:
        print(fetch(model_id, args.out))


if __name__ == "__main__":
    main()
