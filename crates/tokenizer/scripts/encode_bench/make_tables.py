"""Turn encode-benchmark outputs into results.md.

Reads one run directory written by run_bench.py:
  env.json                         machine, toolchain, commits, corpus and model pins
  <model>/timing-*.csv             fresh prompts (after a disjoint warm-up): the headline
  <model>/repeat-*.csv             exact repeats (warm-up on the timed prompts)
  <model>/turns-*.csv              multi-turn chats, SMG caches on and off
  <model>/throughput-*.csv         one pass, N workers, shared queue
  <model>/load-*.csv, rss.csv, rss_runtime.csv   construction time and peak memory
  <model>/parity/*.json            token-id parity reports

Example:
  python3 make_tables.py "$DATA/results/<run-id>"
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
# (label, build, fastokens threads, variant[, baseline (build, threads, variant)]).
# Without a baseline, a column is compared with the first column.
LATENCY_COLUMNS = [
    ("SMG as shipped", "z", "default", "smg"),
    ("SMG, tokenizers at O2", "tok", "default", "smg"),
    ("SMG, tokenizers+onig at O2", "toksys", "default", "smg"),
    ("SMG, HF hot deps at O2", "hotdeps", "default", "smg"),
    ("SMG, all O2", "o2", "default", "smg"),
    ("HF encode_fast", "z", "default", "hf_encode_fast"),
    ("fastokens", "z", "default", "fastokens"),
    ("fastokens, 1 thread", "z", "1", "fastokens", ("z", "1", "smg")),
]
REPEAT_COLUMNS = [
    ("SMG as shipped", "z", "default", "smg"),
    ("HF encode_fast", "z", "default", "hf_encode_fast"),
    ("fastokens", "z", "default", "fastokens"),
]
TURN_COLUMNS = [
    ("SMG as shipped", "z", "default", "smg"),
    ("SMG + L0/L1 caches", "z", "default", "smg_cached"),
    ("fastokens", "z", "default", "fastokens"),
    ("fastokens + L0/L1 caches", "z", "default", "fastokens_cached"),
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

    Each prompt is timed once per leg; its median over legs is its time.
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


def leg_spread(rows):
    """(max - min) / median of the per-leg bucket medians: how much legs disagree."""
    per_leg = defaultdict(list)
    for r in rows:
        per_leg[(r["build"], r["fastokens_threads"], r["variant"], r["bucket"], r["round"])].append(int(r["wall_ns"]) / 1e6)
    legs = defaultdict(list)
    for (build, threads, variant, bucket, _leg), walls in per_leg.items():
        legs[(build, threads, variant, bucket)].append(median(walls))
    return {key: (max(ms) - min(ms)) / median(ms) for key, ms in legs.items()}


def summarize_throughput(rows):
    """Per (build, threads, variant, workers): medians over legs."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["build"], r["fastokens_threads"], r["variant"], r["threads"])].append(r)
    out = {}
    for key, group in groups.items():
        tok_s = [int(r["tokens"]) / float(r["seconds"]) for r in group]
        cpu = [float(r["cpu_seconds"]) / float(r["seconds"]) for r in group]
        out[key] = {"mtok_s": median(tok_s) / 1e6, "cpu_ratio": median(cpu), "legs": len(group), "prompts": int(group[0]["prompts"])}
    return out


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def latency_table(summary, columns, buckets, metric="median_ms"):
    """One row per bucket; every column after the first shows its speed-up over
    its baseline (its own, if given, else the first column)."""
    headers = ["bucket", "tokens (median)"] + [col[0] for col in columns]
    first = tuple(columns[0][1:4])
    rows = []
    for bucket in buckets:
        base_cell = summary.get(first + (bucket,))
        row = [bucket, f"{base_cell['median_tokens']:,.0f}" if base_cell else DASH]
        for i, col in enumerate(columns):
            cell = summary.get(tuple(col[1:4]) + (bucket,))
            baseline = summary.get(tuple(col[4] if len(col) > 4 else first) + (bucket,))
            if cell is None:
                row.append(DASH)
            elif i == 0 or baseline is None:
                row.append(f"{cell[metric]:.2f}")
            else:
                row.append(f"{cell[metric]:.2f} ({baseline[metric] / cell[metric]:.1f}×)")
        rows.append(row)
    return markdown_table(headers, rows)


def spread_table(rows, columns):
    spread = leg_spread(rows)
    body = []
    for col in columns:
        values = [v for k, v in spread.items() if k[:3] == tuple(col[1:4])]
        if values:
            body.append([col[0], f"{100 * max(values):.1f}%", f"{100 * median(values):.1f}%"])
    return markdown_table(["column", "largest spread (any bucket)", "median spread"], body)


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


def _mb(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _present(columns, summary):
    return [c for c in columns if any(k[:3] == tuple(c[1:4]) for k in summary)]


def _model_section(model_dir):
    out = [f"## {model_dir.name}", ""]
    parity = sorted((model_dir / "parity").glob("*.json"))
    if parity:
        vocab = sorted({json.loads(p.read_text()).get("vocab", {}).get("differences", DASH) for p in parity}, key=str)
        out += [
            "### Token-id parity against SMG (mismatches / inputs)",
            "",
            markdown_table(
                ["build", "model", "fastokens threads", "candidate", "chat (add_special_tokens=false)", "embeddings (true)"],
                parity_rows(parity),
            ),
            "",
            f"`id_to_token` differences over the whole vocabulary (SMG vs fastokens): {', '.join(str(v) for v in vocab)}",
            "",
        ]
    timing = _read_csv(model_dir.glob("timing-*.csv"))
    if timing:
        summary = summarize_timing(timing)
        cols = _present(LATENCY_COLUMNS, summary)
        out += [
            "### Encode latency on fresh prompts, median ms per prompt (speed-up), caches off",
            "",
            "Every backend first encodes a separate warm-up prompt set, so its caches hold common word pieces but never the timed prompt.",
            "",
            latency_table(summary, cols, BUCKETS),
            "",
            "### p90 across the bucket's prompts (ms; spread over prompt sizes, not tail latency)",
            "",
            latency_table(summary, cols, BUCKETS, metric="p90_ms"),
            "",
            "### CPU ms per prompt (all threads; fastokens can split one large encode across cores)",
            "",
            latency_table(summary, cols, BUCKETS, metric="cpu_median_ms"),
            "",
            "### Stability: spread of per-leg medians",
            "",
            spread_table(timing, cols),
            "",
        ]
    repeat = _read_csv(model_dir.glob("repeat-*.csv"))
    if repeat:
        summary = summarize_timing(repeat)
        out += [
            "### Warm caches: exact repeats (warm-up on the timed prompts themselves), median ms",
            "",
            latency_table(summary, _present(REPEAT_COLUMNS, summary), BUCKETS),
            "",
        ]
    turns = _read_csv(model_dir.glob("turns-*.csv"))
    if turns:
        summary = summarize_timing(turns)
        out += [
            "### Growing chats (multi-turn): median ms per turn, every turn re-sends the whole history",
            "",
            latency_table(summary, _present(TURN_COLUMNS, summary), ["16k", "50k"]),
            "",
        ]
    throughput = _read_csv(model_dir.glob("throughput-*.csv"))
    if throughput:
        cells = summarize_throughput(throughput)
        rows = [
            [build, variant, threads, workers, f"{c['mtok_s']:.2f}", f"{c['cpu_ratio']:.2f}", c["legs"], c["prompts"]]
            for (build, threads, variant, workers), c in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][2], int(kv[0][3])))
        ]
        out += [
            "### Throughput: one pass over fresh prompts, N workers on one shared queue (median over legs)",
            "",
            markdown_table(["build", "variant", "fastokens threads", "workers", "M tokens/s", "CPU/wall", "legs", "prompts"], rows),
            "",
        ]
    loads = _read_csv(model_dir.glob("load-*.csv"))
    if loads:
        rss = {r["variant"]: _mb(r.get("max_rss_mb")) for r in _read_csv(model_dir.glob("rss.csv"))}
        runtime = {r["variant"]: _mb(r.get("max_rss_mb")) for r in _read_csv(model_dir.glob("rss_runtime.csv"))}
        base = rss.get("none")
        loaded = {r["variant"] for r in loads}
        rows = []
        for r in loads:
            load_mb, run_mb = rss.get(r["variant"]), runtime.get(r["variant"])
            rows.append(
                [
                    r["variant"],
                    f"{float(r['median_ms']):.0f}",
                    f"{load_mb - base:.0f}" if load_mb is not None and base is not None else DASH,
                    f"{run_mb - base:.0f}" if run_mb is not None and base is not None else DASH,
                ]
            )
        rows += [
            [variant, DASH, DASH, f"{mb - base:.0f}"]
            for variant, mb in runtime.items()
            if variant not in loaded and mb is not None and base is not None
        ]
        out += [
            "### Load time and memory (MB above an empty process)",
            "",
            markdown_table(["backend", "load ms (median)", "after load", "after a full-concurrency pass"], rows),
            "",
        ]
    return out


def render_report(run_dir):
    run_dir = Path(run_dir)
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8")) if (run_dir / "env.json").exists() else {}
    out = [f"# Encode benchmark results ({run_dir.name})", "", _env_section(env), ""]
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        out += _model_section(model_dir)
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
