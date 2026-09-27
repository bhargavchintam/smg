"""Tests for make_tables.py. Run: python3 -m unittest -v test_make_tables"""

import json
import tempfile
import unittest
from pathlib import Path

import make_tables as mt


def row(prompt, bucket, variant, wall_ms, tokens=1000, cpu_ms=None, build="z", threads="default", rnd=0):
    return {
        "build": build,
        "fastokens_threads": threads,
        "round": str(rnd),
        "prompt": prompt,
        "bucket": bucket,
        "kind": "code",
        "bytes": "4000",
        "tokens": str(tokens),
        "variant": variant,
        "wall_ns": str(int(wall_ms * 1e6)),
        "cpu_ns": str(int((cpu_ms if cpu_ms is not None else wall_ms) * 1e6)),
    }


class StatsTest(unittest.TestCase):
    def test_median_handles_even_and_odd_lengths(self):
        self.assertEqual(mt.median([3, 1, 2]), 2)
        self.assertEqual(mt.median([4, 1, 3, 2]), 2.5)

    def test_p90_uses_nearest_rank(self):
        self.assertEqual(mt.p90(list(range(1, 11))), 9)
        self.assertEqual(mt.p90([5]), 5)


class SummarizeTimingTest(unittest.TestCase):
    def setUp(self):
        rows = []
        # prompt a: rounds 10, 12, 50 ms -> per-input median 12
        # prompt b: rounds 20, 22, 21 ms -> per-input median 21
        for rnd, (a, b) in enumerate([(10, 20), (12, 22), (50, 21)]):
            rows.append(row("a", "2k", "smg", a, tokens=1000, cpu_ms=a, rnd=rnd))
            rows.append(row("b", "2k", "smg", b, tokens=3000, cpu_ms=2 * b, rnd=rnd))
        self.summary = mt.summarize_timing(rows)

    @property
    def cell(self):
        key = ("z", "default", "smg", "2k")
        self.assertIn(key, self.summary)
        return self.summary[key]

    def test_bucket_median_is_median_of_per_input_medians(self):
        self.assertAlmostEqual(self.cell["median_ms"], (12 + 21) / 2)

    def test_p90_is_over_per_input_medians(self):
        self.assertAlmostEqual(self.cell["p90_ms"], 21)

    def test_tokens_per_second_uses_per_input_medians(self):
        self.assertAlmostEqual(self.cell["tokens_per_s"], (1000 + 3000) / ((12 + 21) / 1000), places=3)

    def test_cpu_ratio_is_total_cpu_over_total_wall(self):
        wall = 10 + 12 + 50 + 20 + 22 + 21
        cpu = 10 + 12 + 50 + 2 * (20 + 22 + 21)
        self.assertAlmostEqual(self.cell["cpu_ratio"], cpu / wall)

    def test_counts_inputs_and_median_tokens(self):
        self.assertEqual(self.cell["inputs"], 2)
        self.assertEqual(self.cell["median_tokens"], 2000)


class LatencyTableTest(unittest.TestCase):
    def test_speedup_is_against_shipped_smg(self):
        rows = [row("a", "2k", "smg", 10), row("a", "2k", "fastokens", 1), row("a", "2k", "smg", 5, build="o2")]
        summary = mt.summarize_timing(rows)
        columns = [("SMG shipped", "z", "default", "smg"), ("SMG O2", "o2", "default", "smg"), ("fastokens", "z", "default", "fastokens")]
        table = mt.latency_table(summary, columns, buckets=["2k"])
        self.assertIn("| 2k |", table)
        self.assertIn("10.00", table)
        self.assertIn("5.00 (2.0×)", table)
        self.assertIn("1.00 (10.0×)", table)

    def test_missing_cell_renders_as_dash(self):
        summary = mt.summarize_timing([row("a", "2k", "smg", 10)])
        table = mt.latency_table(summary, [("SMG", "z", "default", "smg"), ("x", "z", "default", "nope")], buckets=["2k"])
        self.assertIn("| — |", table)


class MarkdownTest(unittest.TestCase):
    def test_markdown_table_has_header_separator_and_rows(self):
        text = mt.markdown_table(["a", "b"], [[1, "x"], [2, "y"]])
        self.assertEqual(text.splitlines(), ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"])


class ParitySummaryTest(unittest.TestCase):
    def test_counts_mismatches_and_skips_controls(self):
        report = {
            "model": {"model": "Qwen/Qwen3", "revision": "abc"},
            "fastokens_bpe_threads": "default",
            "inputs": 119,
            "candidates": {
                "control_drop_last": {"false": {"compared": 119, "mismatches": 118, "errors": 0}},
                "fastokens_json": {
                    "false": {"compared": 119, "mismatches": 0, "errors": 0},
                    "true": {"compared": 119, "mismatches": 2, "errors": 1},
                },
            },
            "cache_replay": {"fastokens_cached": {"compared": 72, "mismatches": 0, "errors": 0}},
            "vocab": {"differences": 0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hotdeps-tdefault.json"
            path.write_text(json.dumps(report))
            rows = mt.parity_rows([path])
        self.assertEqual(
            rows,
            [
                ["hotdeps", "Qwen/Qwen3", "default", "fastokens_json", "0/119", "2/119 (1 errors)"],
                ["hotdeps", "Qwen/Qwen3", "default", "fastokens_cached (cache replay)", "0/72", "—"],
            ],
        )

    def test_control_must_detect_mismatches(self):
        report = {
            "model": {"model": "m"},
            "fastokens_bpe_threads": "default",
            "candidates": {"control_drop_last": {"false": {"compared": 10, "mismatches": 0, "errors": 0}}},
            "cache_replay": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                mt.parity_rows([path])


if __name__ == "__main__":
    unittest.main()
