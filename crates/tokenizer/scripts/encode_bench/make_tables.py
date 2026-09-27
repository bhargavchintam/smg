"""Turn encode-benchmark outputs into results.md.

Reads one run directory written by run_bench.py:
  env.json                      machine, toolchain, commits, corpus and model revisions
  <model>/timing-*.csv          per-encode wall/CPU time (encode_backends time)
  <model>/throughput-*.csv      tokens/s with N workers (encode_backends throughput)
  <model>/load-*.csv            backend construction time (encode_backends load)
  <model>/rss.csv               peak RSS per backend (/usr/bin/time)
  <model>/parity/*.json         parity reports (encode_backends parity)

Example:
  python3 make_tables.py ~/Downloads/oss/encode-bench-data/results/<run-id>
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

DASH = "—"
BUCKETS = ["100", "2k", "16k", "50k"]
CONTROLS = ("smg_again", "control_drop_last")
# (label, build, fastokens threads, variant); the first column is the baseline.
LATENCY_COLUMNS = [
    ("SMG as shipped", "z", "default", "smg"),
    ("SMG, tokenizers at O2", "tok", "default", "smg"),
    ("SMG, tokenizers+onig at O2", "toksys", "default", "smg"),
    ("SMG, HF hot deps at O2", "hotdeps", "default", "smg"),
    ("SMG, all O2", "o2", "default", "smg"),
    ("HF encode_fast", "z", "default", "hf_encode_fast"),
    ("fastokens", "z", "default", "fastokens"),
    ("fastokens, 1 thread", "z", "1", "fastokens"),
]


def median(xs):
    xs = sorted(xs)
    if not xs:
        raise ValueError("median of an empty list")
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def p90(xs):
    """Nearest-rank 90th percentile."""
    xs = sorted(xs)
    if not xs:
        raise ValueError("p90 of an empty list")
    return xs[max(0, math.ceil(0.9 * len(xs)) - 1)]


def summarize_timing(rows):
    """Per (build, threads, variant, bucket): statistics over per-input medians.

    Each prompt is timed once per round; its median over rounds is its time.
    The bucket's median/p90 are taken over those per-prompt medians.
    """
    per_input = defaultdict(lambda: {"wall": [], "cpu": [], "tokens": 0})
    for r in rows:
        key = (r["build"], r["fastokens_threads"], r["variant"], r["bucket"], r["prompt"])
        cell = per_input[key]
        cell["wall"].append(int(r["wall_ns"]) / 1e6)
        cell["cpu"].append(int(r["cpu_ns"]) / 1e6)
        cell["tokens"] = int(r["tokens"])
    groups = defaultdict(list)
    for (build, threads, variant, bucket, _prompt), cell in per_input.items():
        groups[(build, threads, variant, bucket)].append(cell)
    summary = {}
    for key, cells in groups.items():
        medians = [median(c["wall"]) for c in cells]
        total_wall = sum(sum(c["wall"]) for c in cells)
        total_cpu = sum(sum(c["cpu"]) for c in cells)
        summary[key] = {
            "inputs": len(cells),
            "median_ms": median(medians),
            "p90_ms": p90(medians),
            "median_tokens": median([c["tokens"] for c in cells]),
            "tokens_per_s": sum(c["tokens"] for c in cells) / (sum(medians) / 1000),
            "cpu_ratio": total_cpu / total_wall if total_wall else float("nan"),
            "cpu_median_ms": median([median(c["cpu"]) for c in cells]),
        }
    return summary


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def latency_table(summary, columns, buckets, metric="median_ms"):
    """Median ms per prompt; every column after the first also shows its speed-up over the first."""
    headers = ["bucket", "tokens (median)"] + [label for label, *_ in columns]
    rows = []
    for bucket in buckets:
        cells = [summary.get((build, threads, variant, bucket)) for _, build, threads, variant in columns]
        base = cells[0]
        row = [bucket, f"{base['median_tokens']:,.0f}" if base else DASH]
        for i, cell in enumerate(cells):
            if cell is None:
                row.append(DASH)
            elif i == 0 or base is None:
                row.append(f"{cell[metric]:.2f}")
            else:
                row.append(f"{cell[metric]:.2f} ({base[metric] / cell[metric]:.1f}×)")
        rows.append(row)
    return markdown_table(headers, rows)


def _tally_cell(tally):
    if not tally:
        return DASH
    text = f"{tally['mismatches']}/{tally['compared']}"
    if tally.get("errors"):
        text += f" ({tally['errors']} errors)"
    return text


def parity_rows(paths):
    """One row per candidate per report; controls are checked, not listed.

    Reports are named <build>-t<threads>.json by run_bench.py.
    """
    rows = []
    for path in sorted(Path(p) for p in paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        build = path.stem.split("-t")[0]
        model_info = report.get("model")
        model = model_info.get("model") if isinstance(model_info, dict) else str(model_info)
        threads = report.get("fastokens_bpe_threads", "default")
        candidates = report.get("candidates", {})
        drop_last = candidates.get("control_drop_last")
        if drop_last is not None and not any(t["mismatches"] for t in drop_last.values()):
            raise ValueError(f"{path}: control_drop_last found no mismatches, so the harness is not detecting differences")
        same = candidates.get("smg_again")
        if same is not None and any(t["mismatches"] or t["errors"] for t in same.values()):
            raise ValueError(f"{path}: smg_again mismatched SMG itself")
        for name, flags in candidates.items():
            if name in CONTROLS:
                continue
            rows.append([build, model, threads, name, _tally_cell(flags.get("false")), _tally_cell(flags.get("true"))])
        for name, tally in report.get("cache_replay", {}).items():
            rows.append([build, model, threads, f"{name} (cache replay)", _tally_cell(tally), DASH])
    return rows


def _read_csv(paths):
    rows = []
    for path in sorted(paths):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _env_section(env):
    lines = ["## Setup", ""]
    for key in ("machine", "os", "power", "rustc", "smg_commit", "fastokens", "tokenizers", "corpus_sha256", "rounds", "builds"):
        if key in env:
            value = env[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            lines.append(f"- **{key}**: {value}")
    for model, info in sorted(env.get("models", {}).items()):
        lines.append(f"- **model {model}**: {info}")
    return "\n".join(lines)


def render_report(run_dir):
    run_dir = Path(run_dir)
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8")) if (run_dir / "env.json").exists() else {}
    out = [f"# Encode benchmark results ({run_dir.name})", "", _env_section(env), ""]
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        out += [f"## {model}", ""]
        parity = sorted((model_dir / "parity").glob("*.json"))
        if parity:
            out += [
                "### Token-id parity against SMG (mismatches / inputs)",
                "",
                markdown_table(
                    ["build", "model", "fastokens threads", "candidate", "chat (add_special_tokens=false)", "embeddings (true)"],
                    parity_rows(parity),
                ),
                "",
            ]
        timing = _read_csv(model_dir.glob("timing-*.csv"))
        if timing:
            summary = summarize_timing(timing)
            present = [c for c in LATENCY_COLUMNS if any(k[:3] == c[1:] for k in summary)]
            out += [
                "### Encode latency, median ms per prompt (speed-up vs SMG as shipped), caches off",
                "",
                latency_table(summary, present, BUCKETS),
                "",
                "### p90 ms per prompt",
                "",
                latency_table(summary, present, BUCKETS, metric="p90_ms"),
                "",
                "### CPU ms per prompt (all threads; fastokens can split one large encode across cores)",
                "",
                latency_table(summary, present, BUCKETS, metric="cpu_median_ms"),
                "",
            ]
        throughput = _read_csv(model_dir.glob("throughput-*.csv"))
        if throughput:
            rows = [
                [
                    r["build"],
                    r["variant"],
                    r["fastokens_threads"],
                    r["threads"],
                    f"{int(r['tokens']) / float(r['seconds']) / 1e6:.2f}",
                    f"{int(r['prompts']) / float(r['seconds']):.1f}",
                    f"{float(r['cpu_seconds']) / float(r['seconds']):.2f}",
                ]
                for r in throughput
            ]
            out += [
                "### Throughput with N workers on one shared prompt queue",
                "",
                markdown_table(["build", "variant", "fastokens threads", "workers", "M tokens/s", "prompts/s", "CPU/wall"], rows),
                "",
            ]
        loads = _read_csv(model_dir.glob("load-*.csv"))
        rss = {r["variant"]: r for r in _read_csv(model_dir.glob("rss.csv"))}
        if loads:
            base_rss = float(rss["none"]["max_rss_mb"]) if "none" in rss else None
            rows = []
            for r in loads:
                mem = rss.get(r["variant"])
                extra = f"{float(mem['max_rss_mb']) - base_rss:.0f}" if mem and base_rss is not None else DASH
                rows.append([r["variant"], f"{float(r['median_ms']):.0f}", extra])
            out += ["### Load time and memory", "", markdown_table(["backend", "load ms (median)", "extra peak RSS MB"], rows), ""]
    return "\n".join(out).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", help="default: <run_dir>/results.md")
    args = parser.parse_args()
    report = render_report(args.run_dir)
    out = Path(args.out) if args.out else Path(args.run_dir) / "results.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
