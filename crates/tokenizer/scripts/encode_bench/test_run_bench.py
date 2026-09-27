"""Tests for run_bench.py. Run: python3 -m unittest -v test_run_bench"""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import run_bench as rb


class ScheduleTest(unittest.TestCase):
    def test_palindromes_balance_order_effects(self):
        order = rb.palindrome_schedule(["z", "hotdeps", "o2"], 4)
        self.assertEqual(order, ["z", "hotdeps", "o2", "o2", "hotdeps", "z"] * 2)

    def test_every_build_gets_the_same_number_of_legs(self):
        order = rb.palindrome_schedule(["a", "b"], 10)
        self.assertEqual(Counter(order), {"a": 10, "b": 10})

    def test_odd_or_too_few_legs_are_rejected(self):
        for legs in (0, 1, 3):
            with self.assertRaises(ValueError):
                rb.palindrome_schedule(["a", "b"], legs)


class BuildArgsTest(unittest.TestCase):
    def test_shipped_build_uses_the_release_profile_unchanged(self):
        args = rb.build_args("z")
        self.assertEqual(args[:3], ["cargo", "build", "--release"])
        self.assertNotIn("--config", args)
        self.assertEqual(args[args.index("--target-dir") + 1], "target/tb-z")

    def test_o2_build_overrides_the_whole_profile(self):
        self.assertIn("profile.release.opt-level=2", rb.build_args("o2"))

    def test_hotdeps_build_raises_only_hf_encode_crates(self):
        args = rb.build_args("hotdeps")
        self.assertIn("profile.release.package.tokenizers.opt-level=2", args)
        self.assertIn("profile.release.package.onig_sys.opt-level=2", args)
        self.assertNotIn("profile.release.opt-level=2", args)

    def test_ablation_builds_raise_one_crate_set_each(self):
        self.assertIn("tok", rb.BUILDS)
        self.assertIn("toksys", rb.BUILDS)
        tok = rb.build_args("tok")
        self.assertIn("profile.release.package.tokenizers.opt-level=2", tok)
        self.assertNotIn("profile.release.package.onig_sys.opt-level=2", tok)
        toksys = rb.build_args("toksys")
        for crate in ("tokenizers", "onig", "onig_sys"):
            self.assertIn(f"profile.release.package.{crate}.opt-level=2", toksys)
        self.assertNotIn("profile.release.package.daachorse.opt-level=2", toksys)

    def test_unknown_build_is_rejected(self):
        with self.assertRaises(ValueError):
            rb.build_args("nope")


class MaxRssTest(unittest.TestCase):
    def test_parses_macos_time_l_bytes(self):
        text = "        0.21 real         0.15 user\n  52428800  maximum resident set size\n  0  page reclaims\n"
        self.assertAlmostEqual(rb.parse_max_rss_mb(text), 50.0)

    def test_parses_gnu_time_v_kbytes(self):
        text = "\tMaximum resident set size (kbytes): 51200\n\tExit status: 0\n"
        self.assertAlmostEqual(rb.parse_max_rss_mb(text), 50.0)

    def test_returns_none_when_absent(self):
        self.assertIsNone(rb.parse_max_rss_mb("no memory line here"))


class PowerTest(unittest.TestCase):
    def test_reads_ac_and_battery_from_pmset(self):
        self.assertEqual(rb.power_source("Now drawing from 'AC Power'\n -InternalBattery-0 99%"), "AC")
        self.assertEqual(rb.power_source("Now drawing from 'Battery Power'\n"), "battery")
        self.assertIsNone(rb.power_source(""))


class ResolveModelsTest(unittest.TestCase):
    def test_picks_the_pinned_revision_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            (root / "Qwen__Qwen3-4B-Instruct-2507@000000000000").mkdir(parents=True)
            pinned = root / "Qwen__Qwen3-4B-Instruct-2507@cdbee75f17c0"
            pinned.mkdir()
            self.assertEqual(rb.resolve_models(tmp, ["qwen3"]), {"qwen3": pinned})

    def test_missing_pinned_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models").mkdir()
            with self.assertRaises(SystemExit):
                rb.resolve_models(tmp, ["qwen3"])


if __name__ == "__main__":
    unittest.main()
