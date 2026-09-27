"""Download tokenizer files for the benchmark models at pinned revisions.

pins.json (next to this script) records the exact commit of every model and
the sha256 of every file the benchmark reads. Downloads are checked against
it, so every machine benchmarks byte-identical tokenizers. Each model lands in
<out>/<org>__<name>@<commit12>/ with a fetch.json copy of its pin.

Example:
  python3 fetch_models.py --out "$DATA/models"
  python3 fetch_models.py --repin --model Qwen/Qwen3-4B-Instruct-2507  # print a new pin
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

PINS = json.loads(Path(__file__).with_name("pins.json").read_text(encoding="utf-8"))
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


def pinned_dir(root, model_id):
    return Path(root) / model_dir_name(model_id, PINS[model_id]["revision"])


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(dest, expected):
    """Raise if any expected file is missing or differs from its pinned sha256."""
    problems = []
    for name, digest in sorted(expected.items()):
        path = Path(dest) / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif _sha256(path) != digest:
            problems.append(f"{name}: sha256 differs from pins.json")
    if problems:
        raise ValueError(f"{dest}: " + "; ".join(problems))


def fetch(model_id, out_root):
    pin = PINS[model_id]
    dest = pinned_dir(out_root, model_id)
    for name in sorted(pin["files"]):
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(file_url(model_id, pin["revision"], name), target)  # noqa: S310
    verify_files(dest, pin["files"])
    record = {"model": model_id, "revision": pin["revision"], "files": pin["files"]}
    (dest / "fetch.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return dest


def repin(model_id, out_root, revision="main"):
    """Resolve `revision` on the Hub, download its files and return a new pin."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_id}/revision/{revision}", timeout=60) as resp:  # noqa: S310
        info = json.load(resp)
    sha = info["sha"]
    dest = Path(out_root) / model_dir_name(model_id, sha)
    files = {}
    for name in wanted_files([s["rfilename"] for s in info.get("siblings", [])]):
        target = dest / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(file_url(model_id, sha, name), target)  # noqa: S310
        files[name] = _sha256(target)
    return {"revision": sha, "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", action="append", help="Hugging Face model id (repeatable); default: every pinned model")
    parser.add_argument("--repin", action="store_true", help="resolve main on the Hub and print new pins instead")
    args = parser.parse_args()
    models = args.model or sorted(PINS)
    if args.repin:
        print(json.dumps({m: repin(m, args.out) for m in models}, indent=2))
        return
    for model_id in models:
        print(fetch(model_id, args.out))


if __name__ == "__main__":
    main()
