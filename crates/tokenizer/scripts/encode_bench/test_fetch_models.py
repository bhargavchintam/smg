"""Tests for fetch_models.py. Run: python3 -m unittest -v test_fetch_models"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import fetch_models as fm


class FetchModelsTest(unittest.TestCase):
    def test_wanted_files_keep_tokenizer_files_and_skip_weights(self):
        siblings = [
            "config.json",
            "generation_config.json",
            "model-00001-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tiktoken.model",
            "o200k_base.tiktoken",
            "encoding/encoding_dsv32.py",
            "chat_template.jinja",
            "README.md",
            "tokenization_kimi.py",
        ]
        self.assertEqual(
            fm.wanted_files(siblings),
            [
                "chat_template.jinja",
                "config.json",
                "encoding/encoding_dsv32.py",
                "generation_config.json",
                "o200k_base.tiktoken",
                "tiktoken.model",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )

    def test_model_dir_name_pins_revision_and_is_path_safe(self):
        name = fm.model_dir_name("deepseek-ai/DeepSeek-V3.2", "a" * 40)
        self.assertEqual(name, "deepseek-ai__DeepSeek-V3.2@aaaaaaaaaaaa")
        self.assertNotIn("/", name)

    def test_file_url_uses_the_pinned_commit(self):
        self.assertEqual(
            fm.file_url("Qwen/Qwen3-4B-Instruct-2507", "abc123", "tokenizer.json"),
            "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/abc123/tokenizer.json",
        )


class PinsTest(unittest.TestCase):
    def test_pins_cover_the_benchmark_models_with_full_revisions(self):
        for model in ("Qwen/Qwen3-4B-Instruct-2507", "deepseek-ai/DeepSeek-V3.2"):
            pin = fm.PINS[model]
            self.assertEqual(len(pin["revision"]), 40)
            self.assertIn("tokenizer.json", pin["files"])

    def test_verify_files_accepts_matching_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.json").write_text("x")
            fm.verify_files(Path(tmp), {"a.json": hashlib.sha256(b"x").hexdigest()})

    def test_verify_files_names_changed_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.json").write_text("changed")
            with self.assertRaises(ValueError) as ctx:
                fm.verify_files(Path(tmp), {"a.json": hashlib.sha256(b"x").hexdigest(), "b.json": "0" * 64})
        self.assertIn("a.json", str(ctx.exception))
        self.assertIn("b.json", str(ctx.exception))

    def test_pinned_dir_uses_the_pinned_revision(self):
        self.assertEqual(
            fm.pinned_dir(Path("/m"), "Qwen/Qwen3-4B-Instruct-2507"),
            Path("/m") / "Qwen__Qwen3-4B-Instruct-2507@cdbee75f17c0",
        )


if __name__ == "__main__":
    unittest.main()
