"""Drive the encode benchmark end to end and write results.md.

Steps:
  1. build the example once per build profile (z = SMG's shipped release
     profile, hotdeps = HF encode crates at opt-level 2, o2 = everything at 2)
  2. render the corpus with SMG's chat templates (fresh for every run)
  3. parity with every build, plus fastokens pinned to 1 BPE thread
  4. time every encode in palindromic build order (z hotdeps o2 o2 hotdeps z ...)
  5. throughput, load time and peak RSS
  6. make_tables.py -> <run>/results.md

Example (any working directory):
  python3 crates/tokenizer/scripts/encode_bench/run_bench.py \
      --data ~/Downloads/oss/encode-bench-data --legs 10
"""

import argparse
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

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
BUILDS = {
    "z": [],
    "hotdeps": [arg for crate in HF_HOT_DEPS for arg in ("--config", f"profile.release.package.{crate}.opt-level=2")],
    "o2": ["--config", "profile.release.opt-level=2"],
}
DEFAULT_MODELS = {"qwen3": "Qwen__Qwen3-4B-Instruct-2507@", "deepseek-v3.2": "deepseek-ai__DeepSeek-V3.2@"}
TIME_VARIANTS = "smg,hf_encode,hf_encode_fast,fastokens"
LOAD_VARIANTS = ["none", "smg", "hf_encode", "fastokens"]
# fastokens reads these once per process; "default" runs must not inherit them.
FASTOKENS_ENV = ("FASTOKENS_BPE_THREADS", "FASTOKENS_INPUT_CACHE")


def palindrome_schedule(builds, legs_per_build):
    """Build order for the timing legs: A B C C B A, repeated, so drift over
    the run (thermals, background load) hits every build equally."""
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
        power = (_output(["pmset", "-g", "batt"]).splitlines() or ["?"])[0]
    else:
        cpu = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text()) if Path("/proc/cpuinfo").exists() else None
        machine = f"{cpu.group(1) if cpu else platform.processor()} ({os.cpu_count()} logical)"
        os_name = f"{system} {platform.release()}"
        power = "n/a"
    dirty = bool(_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]))
    sums = Path(data) / "corpus" / "SHA256SUMS"
    return {
        "machine": machine,
        "os": os_name,
        "power": power,
        "rustc": _output(["rustc", "-V"]),
        "smg_commit": _output(["git", "-C", str(REPO), "rev-parse", "HEAD"]) + (" (dirty)" if dirty else ""),
        "fastokens": _lock_version("fastokens"),
        "tokenizers": _lock_version("tokenizers"),
        "corpus_sha256": sums.read_text().splitlines() if sums.exists() else [],
        "rounds": f"{legs} legs per build, 1 round per leg, palindromic build order",
        "builds": {name: " ".join(BUILDS[name]) or "(release profile as shipped)" for name in builds},
        "models": {
            name: json.loads((path / "fetch.json").read_text()).get("revision", "?") if (path / "fetch.json").exists() else "?"
            for name, path in models.items()
        },
    }


def _resolve_models(data, specs):
    root = Path(data) / "models"
    wanted = dict(spec.split("=", 1) for spec in specs) if specs else DEFAULT_MODELS
    models = {}
    for name, prefix in wanted.items():
        matches = sorted(p for p in root.iterdir() if p.name.startswith(prefix))
        if not matches:
            raise SystemExit(f"no model directory starting with {prefix!r} in {root}; run fetch_models.py first")
        models[name] = matches[-1]
    return models


def bench_model(name, model, data, out, builds, legs, tp_threads, tp_seconds):
    corpus = Path(data) / "corpus"
    z = _binary("z")
    # Render fresh for every run: prompts always match this corpus and this commit.
    out.mkdir(parents=True, exist_ok=True)
    prompts = out / "prompts.jsonl"
    _run([z, "render", "--model", model, "--conversations", corpus / "conversations.jsonl", "--out", prompts])
    parity = out / "parity"
    parity.mkdir(parents=True, exist_ok=True)
    common = ["--model", model, "--prompts", prompts, "--stress", corpus / "stress.jsonl", "--conversations", corpus / "conversations.jsonl"]
    for build in builds:
        _run([_binary(build), "parity", *common, "--out", parity / f"{build}-tdefault.json"])
    _run(
        [z, "parity", *common, "--candidates", "fastokens_json", "--out", parity / "z-t1.json"],
        env=_clean_env(FASTOKENS_BPE_THREADS="1"),
    )

    for leg, build in enumerate(palindrome_schedule(builds, legs)):
        timing = ["time", "--model", model, "--prompts", prompts, "--rounds", "1", "--round-offset", str(leg), "--build", build]
        _run([_binary(build), *timing, "--variants", TIME_VARIANTS, "--out", out / f"timing-{build}-leg{leg:02d}-tdefault.csv"])
        _run(
            [_binary(build), *timing, "--variants", "fastokens", "--out", out / f"timing-{build}-leg{leg:02d}-t1.csv"],
            env=_clean_env(FASTOKENS_BPE_THREADS="1"),
        )

    for build in builds:
        _run(
            [_binary(build), "throughput", "--model", model, "--prompts", prompts, "--variants", "smg,fastokens",
             "--threads", tp_threads, "--seconds", str(tp_seconds), "--build", build, "--out", out / f"throughput-{build}.csv"]
        )  # fmt: skip

    _run([z, "load", "--model", model, "--variants", ",".join(LOAD_VARIANTS), "--repeat", "5", "--out", out / "load-z.csv"])
    flag = "-l" if platform.system() == "Darwin" else "-v"
    rows = ["variant,max_rss_mb"]
    for variant in LOAD_VARIANTS:
        proc = subprocess.run(
            ["/usr/bin/time", flag, str(z), "load", "--model", str(model), "--variants", variant, "--repeat", "1"],
            check=True, capture_output=True, text=True, env=_clean_env(),
        )  # fmt: skip
        rss = parse_max_rss_mb(proc.stderr)
        rows.append(f"{variant},{rss:.1f}" if rss is not None else f"{variant},")
    (out / "rss.csv").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="directory holding corpus/, models/ (and prompts/, results/)")
    parser.add_argument("--models", nargs="*", help="name=model-dir-prefix pairs (default: qwen3 and deepseek-v3.2)")
    parser.add_argument("--builds", default="z,hotdeps,o2")
    parser.add_argument("--legs", type=int, default=10, help="timing legs per build (even)")
    parser.add_argument("--throughput-threads", default=f"1,4,{os.cpu_count()}")
    parser.add_argument("--throughput-seconds", type=float, default=5.0)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + platform.node().split(".")[0])
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    builds = args.builds.split(",")
    palindrome_schedule(builds, args.legs)  # validate before any work
    if not args.skip_build:
        for build in builds:
            _run(build_args(build), cwd=REPO)
    models = _resolve_models(args.data, args.models)
    run_dir = Path(args.data) / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = environment(args.data, models, builds, args.legs)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    for name, model in models.items():
        bench_model(name, model, args.data, run_dir / name, builds, args.legs, args.throughput_threads, args.throughput_seconds)
    report = run_dir / "results.md"
    report.write_text(make_tables.render_report(run_dir), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
