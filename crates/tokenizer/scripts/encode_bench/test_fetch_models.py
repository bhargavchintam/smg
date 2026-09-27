"""Tests for fetch_models.py. Run: python3 -m unittest -v test_fetch_models"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
