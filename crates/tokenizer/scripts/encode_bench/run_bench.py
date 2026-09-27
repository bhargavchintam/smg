"""Drive the encode benchmark end to end and write results.md.

Steps, for each model (tokenizer files pinned in pins.json):
  1. build the example once per build profile (z = SMG's release profile as
     shipped; tok/toksys/hotdeps = some of HF's encode crates at opt-level 2;
     o2 = everything at opt-level 2)
  2. render the corpus, a disjoint warm-up corpus and a larger throughput
     corpus with SMG's chat templates (fresh for every run)
  3. parity with every build, and fastokens at several BPE thread counts
  4. timing legs in palindromic build order (z tok ... tok z ...):
     fresh prompts after the warm-up corpus (headline), plus a 1-thread
     fastokens process that also times smg as its in-process baseline
  5. exact repeats and multi-turn chats (SMG caches on/off) on the z build
  6. throughput legs (one pass over fresh prompts), load time, peak RSS
  7. make_tables.py -> <run>/results.md

On macOS the run refuses to start on battery unless --allow-battery is given,
and keeps the machine awake while it runs.

Example:
  python3 crates/tokenizer/scripts/encode_bench/run_bench.py --data "$DATA" --legs 10
"""

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

import fetch_models
import make_tables

EXAMPLE = "encode_backends"
REPO = Path(__file__).resolve().parents[4]
# Crates on HF tokenizers' encode path that SMG's release profile builds at "z".
HF_HOT_DEPS = [
    "tokenizers",
    "onig",
    "onig_sys",
    "daachorse",
    "dary_heap",
    "ahash",
    "aho-corasick",
    "compact_str",
    "regex",
    "unicode-normalization-alignments",
    "unicode-segmentation",
    "unicode_categories",
]


def _opt2(crates):
    return [arg for crate in crates for arg in ("--config", f"profile.release.package.{crate}.opt-level=2")]


BUILDS = {
    "z": [],
    # Ablation: which of HF's encode-path crates carry the opt-level win.
    "tok": _opt2(["tokenizers"]),
    "toksys": _opt2(["tokenizers", "onig", "onig_sys"]),
    "hotdeps": _opt2(HF_HOT_DEPS),
    "o2": ["--config", "profile.release.opt-level=2"],
}
MODEL_IDS = {"qwen3": "Qwen/Qwen3-4B-Instruct-2507", "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2"}
TIME_VARIANTS = "smg,hf_encode,hf_encode_fast,fastokens"
REPEAT_VARIANTS = "smg,hf_encode_fast,fastokens"
TURN_VARIANTS = "smg,smg_cached,fastokens,fastokens_cached"
LOAD_VARIANTS = ["none", "smg", "hf_encode", "fastokens", "smg_fastokens"]
RUNTIME_RSS_VARIANTS = ["smg", "smg_cached", "fastokens", "fastokens_cached"]
# fastokens reads these once per process; "default" runs must not inherit them.
FASTOKENS_ENV = ("FASTOKENS_BPE_THREADS", "FASTOKENS_INPUT_CACHE")


def palindrome_schedule(builds, legs_per_build):
    """Build order for timing legs: A B C C B A, repeated, so drift over the
    run (thermals, background load) hits every build equally."""
    if legs_per_build < 2 or legs_per_build % 2:
        raise ValueError("legs_per_build must be an even number of at least 2")
    forward = list(builds)
    return (forward + forward[::-1]) * (legs_per_build // 2)


def build_args(name):
    if name not in BUILDS:
        raise ValueError(f"unknown build {name!r}; known: {sorted(BUILDS)}")
    return [
        "cargo", "build", "--release", "-p", "llm-tokenizer", "--example", EXAMPLE,
        "--target-dir", f"target/tb-{name}", *BUILDS[name],
    ]  # fmt: skip


def parse_max_rss_mb(text):
    """Peak RSS in MiB from `/usr/bin/time -l` (macOS, bytes) or `-v` (GNU, KiB)."""
    match = re.search(r"^\s*(\d+)\s+maximum resident set size", text, re.M)
    if match:
        return int(match.group(1)) / (1024 * 1024)
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match:
        return int(match.group(1)) / 1024
    return None


def power_source(text):
    """'AC' or 'battery' from `pmset -g batt`; None when it cannot tell."""
    match = re.search(r"drawing from '([^']+)'", text)
    if not match:
        return None
    return "AC" if match.group(1).startswith("AC") else "battery"


def resolve_models(data, names):
    """Model directories at their pinned revisions (fetch_models.py fetches them)."""
    root = Path(data) / "models"
    models = {}
    for name in names:
        if name not in MODEL_IDS:
            raise SystemExit(f"unknown model {name!r}; known: {sorted(MODEL_IDS)}")
        path = fetch_models.pinned_dir(root, MODEL_IDS[name])
        if not path.is_dir():
            raise SystemExit(f"{path} is missing; run fetch_models.py --out {root}")
        models[name] = path
    return models


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in FASTOKENS_ENV}
    env.update(extra)
    return env


def _run(cmd, env=None, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, env=env or _clean_env(), text=True, **kwargs)


def _output(cmd):
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _binary(build):
    return REPO / "target" / f"tb-{build}" / "release" / "examples" / EXAMPLE


def _lock_version(crate):
    match = re.search(rf'name = "{re.escape(crate)}"\nversion = "([^"]+)"', (REPO / "Cargo.lock").read_text())
    return match.group(1) if match else "?"


def _peak_rss(cmd):
    flag = "-l" if platform.system() == "Darwin" else "-v"
    proc = subprocess.run(["/usr/bin/time", flag, *[str(c) for c in cmd]], check=True, capture_output=True, text=True, env=_clean_env())
    return parse_max_rss_mb(proc.stderr)


def _linux_cpu_name():
    info = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    match = re.search(r"model name\s*:\s*(.+)", info) or re.search(r"Model name:\s*(.+)", _output(["lscpu"]))
    return match.group(1).strip() if match else platform.machine()


def environment(data, models, builds, legs):
    system = platform.system()
    if system == "Darwin":
        machine = "{} ({} P-cores, {} logical, {:.0f} GiB)".format(
            _output(["sysctl", "-n", "machdep.cpu.brand_string"]),
            _output(["sysctl", "-n", "hw.perflevel0.logicalcpu"]),
            _output(["sysctl", "-n", "hw.logicalcpu"]),
            int(_output(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30,
        )
        os_name = f"macOS {_output(['sw_vers', '-productVersion'])}"
        low_power = re.search(r"lowpowermode\s+(\d)", _output(["pmset", "-g"]))
        power = f"{power_source(_output(['pmset', '-g', 'batt']))}, Low Power Mode {'on' if low_power and low_power.group(1) == '1' else 'off'}"
    else:
        machine = f"{_linux_cpu_name()} ({os.cpu_count()} logical, {platform.machine()})"
        os_name = f"{system} {platform.release()}"
        power = "n/a"
    dirty = bool(_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]))
    sums = {
        corpus: (Path(data) / corpus / "SHA256SUMS").read_text().splitlines()[:2]
        for corpus in ("corpus", "corpus-warmup", "corpus-throughput")
        if (Path(data) / corpus / "SHA256SUMS").exists()
    }
    return {
        "machine": machine,
        "os": os_name,
        "power": power,
        "rustc": _output(["rustc", "-V"]),
        "smg_commit": _output(["git", "-C", str(REPO), "rev-parse", "HEAD"]) + (" (dirty)" if dirty else ""),
        "fastokens": _lock_version("fastokens"),
        "tokenizers": _lock_version("tokenizers"),
        "corpus_sha256": sums,
        "rounds": f"{legs} legs per build, 1 round per leg, palindromic build order",
        "builds": {name: " ".join(BUILDS[name]) or "(release profile as shipped)" for name in builds},
        "models": {name: fetch_models.PINS[MODEL_IDS[name]]["revision"] for name in models},
    }


def bench_model(model, data, out, args):
    builds = args.builds.split(",")
    z = _binary("z")
    out.mkdir(parents=True, exist_ok=True)
    rendered = {}
    for label, corpus in (("prompts", args.corpus), ("warmup", args.warmup_corpus), ("throughput", args.throughput_corpus)):
        rendered[label] = out / f"{label}.jsonl"
        _run([z, "render", "--model", model, "--conversations", Path(corpus) / "conversations.jsonl", "--out", rendered[label]])
    prompts, warmup = rendered["prompts"], rendered["warmup"]
    conversations = Path(args.corpus) / "conversations.jsonl"

    parity = out / "parity"
    parity.mkdir(exist_ok=True)
    common = ["--model", model, "--prompts", prompts, "--stress", Path(args.corpus) / "stress.jsonl", "--conversations", conversations]
    for build in builds:
        _run([_binary(build), "parity", *common, "--out", parity / f"{build}-tdefault.json"])
    for threads in args.parity_threads.split(","):
        _run(
            [z, "parity", *common, "--candidates", "fastokens,fastokens_file", "--out", parity / f"z-t{threads}.json"],
            env=_clean_env(FASTOKENS_BPE_THREADS=threads),
        )

    for leg, build in enumerate(palindrome_schedule(builds, args.legs)):
        timing = ["time", "--model", model, "--prompts", prompts, "--warmup", warmup, "--rounds", "1", "--round-offset", str(leg), "--build", build]
        _run([_binary(build), *timing, "--variants", TIME_VARIANTS, "--out", out / f"timing-{build}-leg{leg:02d}-tdefault.csv"])
        _run(
            [_binary(build), *timing, "--variants", "smg,fastokens", "--out", out / f"timing-{build}-leg{leg:02d}-t1.csv"],
            env=_clean_env(FASTOKENS_BPE_THREADS="1"),
        )
    for leg in range(args.side_legs):
        side = ["--model", model, "--round-offset", str(leg), "--build", "z"]
        _run([z, "time", *side, "--prompts", prompts, "--rounds", "1", "--variants", REPEAT_VARIANTS, "--out", out / f"repeat-z-leg{leg:02d}.csv"])
        _run([z, "turns", *side, "--conversations", conversations, "--warmup", warmup, "--variants", TURN_VARIANTS, "--out", out / f"turns-z-leg{leg:02d}.csv"])

    tp_builds = args.throughput_builds.split(",")
    for leg, build in enumerate(palindrome_schedule(tp_builds, 2)):
        _run(
            [_binary(build), "throughput", "--model", model, "--prompts", rendered["throughput"], "--warmup", warmup,
             "--variants", "smg,fastokens", "--threads", args.throughput_threads, "--round-offset", str(leg),
             "--build", build, "--out", out / f"throughput-{build}-leg{leg:02d}.csv"]
        )  # fmt: skip

    _run([z, "load", "--model", model, "--variants", ",".join(LOAD_VARIANTS), "--repeat", "5", "--out", out / "load-z.csv"])
    rows = ["variant,max_rss_mb"]
    for variant in LOAD_VARIANTS:
        rss = _peak_rss([z, "load", "--model", model, "--variants", variant, "--repeat", "1"])
        rows.append(f"{variant},{rss:.1f}" if rss is not None else f"{variant},")
    (out / "rss.csv").write_text("\n".join(rows) + "\n")
    rows = ["variant,max_rss_mb"]
    max_threads = str(os.cpu_count())
    for variant in RUNTIME_RSS_VARIANTS:
        rss = _peak_rss(
            [z, "throughput", "--model", model, "--prompts", rendered["throughput"], "--warmup", warmup,
             "--variants", variant, "--threads", max_threads, "--out", out / f"rss-pass-{variant}.csv.tmp"]
        )  # fmt: skip
        rows.append(f"{variant},{rss:.1f}" if rss is not None else f"{variant},")
    (out / "rss_runtime.csv").write_text("\n".join(rows) + "\n")


def _check_power(allow_battery):
    if platform.system() != "Darwin":
        return
    source = power_source(_output(["pmset", "-g", "batt"]))
    if source == "battery" and not allow_battery:
        raise SystemExit("On battery: plug in for comparable numbers, or pass --allow-battery (recorded in env.json).")
    # Keep the machine awake (no idle sleep) until this process exits.
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # noqa: S603,S607


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="directory holding models/ and the three corpora")
    parser.add_argument("--models", default="qwen3,deepseek-v3.2", help=f"comma-separated, from {sorted(MODEL_IDS)}")
    parser.add_argument("--builds", default="z,tok,toksys,hotdeps,o2")
    parser.add_argument("--legs", type=int, default=10, help="timing legs per build (even)")
    parser.add_argument("--side-legs", type=int, default=4, help="legs for exact-repeat and multi-turn runs (z build)")
    parser.add_argument("--parity-threads", default=f"1,2,3,{os.cpu_count()}", help="FASTOKENS_BPE_THREADS values for parity")
    parser.add_argument("--throughput-builds", default="z,hotdeps")
    parser.add_argument("--throughput-threads", default=f"1,4,{os.cpu_count()}")
    parser.add_argument("--corpus", help="default: <data>/corpus")
    parser.add_argument("--warmup-corpus", help="default: <data>/corpus-warmup")
    parser.add_argument("--throughput-corpus", help="default: <data>/corpus-throughput")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + platform.node().split(".")[0])
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    data = Path(args.data)
    args.corpus = args.corpus or data / "corpus"
    args.warmup_corpus = args.warmup_corpus or data / "corpus-warmup"
    args.throughput_corpus = args.throughput_corpus or data / "corpus-throughput"

    builds = args.builds.split(",")
    palindrome_schedule(builds, args.legs)  # validate before any work
    models = resolve_models(data, args.models.split(","))
    for name, path in models.items():
        fetch_models.verify_files(path, fetch_models.PINS[MODEL_IDS[name]]["files"])
    _check_power(args.allow_battery)
    if not args.skip_build:
        for build in sorted(set(builds) | set(args.throughput_builds.split(","))):
            _run(build_args(build), cwd=REPO)
    run_dir = data / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = environment(data, models, builds, args.legs)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    for name, model in models.items():
        bench_model(model, data, run_dir / name, args)
    report = run_dir / "results.md"
    report.write_text(make_tables.render_report(run_dir), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
