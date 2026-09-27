"""Tests for build_corpus.py. Run: python3 -m unittest -v test_build_corpus"""

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import build_corpus as bc

CJK_RE = re.compile(r"[一-鿿]")
SMALL_BUCKETS = [("s", 400), ("m", 8_000), ("l", 70_000)]


def make_sources(root: Path) -> dict:
    """Small but realistic source files: Rust code, markdown prose, a Chinese text."""
    code = root / "src.rs"
    code.write_text(
        "\n\n".join(
            f"/// Handles request number {i}.\n"
            f"pub fn handle_{i}(req: &Request) -> Result<Response, Error> {{\n"
            f"    let body = req.body().map_err(|e| Error::new(e, {i}))?;\n"
            f"    Ok(Response::new(body.len() * {i}))\n}}"
            for i in range(200)
        ),
        encoding="utf-8",
    )
    prose = root / "README.md"
    prose.write_text(
        "\n\n".join(
            f"Paragraph {i}: the gateway routes each request to a worker, keeps "
            f"the tokenizer warm, and reports latency for every stage it runs."
            for i in range(200)
        ),
        encoding="utf-8",
    )
    cjk = root / "gutenberg-cjk.txt"
    cjk.write_text(
        "Project Gutenberg header that must be dropped\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + "\n".join(f"第{i}回　花果山上有一塊仙石，天產石猴，每日在山中行走跳躍。" for i in range(400))
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense footer\n",
        encoding="utf-8",
    )
    return {"code_files": [code], "prose_files": [prose], "cjk_files": [cjk]}


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def content_bytes(conv: dict) -> int:
    return sum(len(m["content"].encode("utf-8")) for m in conv["messages"])


class BuildCorpusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sources = make_sources(self.root)
        self.out = self.root / "out"
        self.manifest = bc.build_corpus(
            out_dir=self.out, seed=7, per_bucket=5, buckets=SMALL_BUCKETS, stress_slices=3, **self.sources
        )
        self.convs = read_jsonl(self.out / "conversations.jsonl")
        self.stress = read_jsonl(self.out / "stress.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_seed_gives_identical_files(self):
        self.assertTrue(self.convs and self.stress)
        other = self.root / "again"
        bc.build_corpus(out_dir=other, seed=7, per_bucket=5, buckets=SMALL_BUCKETS, stress_slices=3, **self.sources)
        for name in ("conversations.jsonl", "stress.jsonl"):
            self.assertEqual((self.out / name).read_bytes(), (other / name).read_bytes(), name)

    def test_different_seed_changes_conversations(self):
        other = self.root / "seed8"
        bc.build_corpus(out_dir=other, seed=8, per_bucket=5, buckets=SMALL_BUCKETS, stress_slices=3, **self.sources)
        self.assertNotEqual((self.out / "conversations.jsonl").read_bytes(), (other / "conversations.jsonl").read_bytes())

    def test_each_bucket_has_requested_count(self):
        for name, _ in SMALL_BUCKETS:
            self.assertEqual(sum(1 for c in self.convs if c["bucket"] == name), 5, name)

    def test_content_bytes_land_on_target(self):
        self.assertEqual(len(self.convs), 15)
        for conv in self.convs:
            size, target = content_bytes(conv), conv["target_bytes"]
            self.assertGreaterEqual(size, target, conv["id"])
            self.assertLessEqual(size, target + 8, conv["id"])

    def test_roles_start_with_system_alternate_and_end_with_user(self):
        self.assertEqual(len(self.convs), 15)
        for conv in self.convs:
            roles = [m["role"] for m in conv["messages"]]
            self.assertEqual(roles[0], "system", conv["id"])
            for i, role in enumerate(roles[1:]):
                self.assertEqual(role, "user" if i % 2 == 0 else "assistant", conv["id"])
            self.assertEqual(roles[-1], "user", conv["id"])

    def test_every_kind_appears_in_each_bucket(self):
        self.assertEqual(set(bc.KINDS), {"code", "prose", "cjk", "json", "mixed"})
        for name, _ in SMALL_BUCKETS:
            kinds = {c["kind"] for c in self.convs if c["bucket"] == name}
            self.assertEqual(kinds, set(bc.KINDS), name)

    def test_tools_only_for_buckets_from_8000_bytes(self):
        self.assertEqual(len(self.convs), 15)
        for conv in self.convs:
            if conv["target_bytes"] >= 8_000:
                self.assertTrue(conv["tools"], conv["id"])
            else:
                self.assertEqual(conv["tools"], [], conv["id"])

    def test_tools_are_function_schemas(self):
        tool_lists = [c["tools"] for c in self.convs if c["tools"]]
        self.assertTrue(tool_lists)
        tools = tool_lists[0]
        for tool in tools:
            self.assertEqual(tool["type"], "function")
            fn = tool["function"]
            self.assertTrue(fn["name"] and fn["description"])
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertTrue(set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"]))

    def test_mixed_kind_has_code_prose_and_cjk(self):
        text = "".join(m["content"] for c in self.convs if c["kind"] == "mixed" and c["bucket"] == "l" for m in c["messages"])
        self.assertTrue(CJK_RE.search(text))
        self.assertIn("pub fn handle_", text)
        self.assertIn("the gateway routes", text)

    def test_json_kind_embeds_parseable_json_blocks(self):
        candidates = [c for c in self.convs if c["kind"] == "json" and c["bucket"] == "m"]
        self.assertTrue(candidates)
        conv = candidates[0]
        blocks = re.findall(r"```json\n(.*?)\n```", "".join(m["content"] for m in conv["messages"]), re.S)
        self.assertTrue(blocks)
        json.loads(blocks[0])

    def test_gutenberg_header_and_footer_are_dropped(self):
        text = "".join(m["content"] for c in self.convs for m in c["messages"])
        self.assertIn("花果山", text)
        self.assertNotIn("Project Gutenberg header", text)
        self.assertNotIn("license footer", text)

    def test_stress_has_every_category(self):
        expected = {"tiny", "base64", "minified_json", "emoji_zwj", "combining_marks", "special_token_literals",
                    "whitespace_runs", "long_word", "mixed_scripts", "crlf_tabs", "slice"}
        self.assertEqual(set(bc.STRESS_CATEGORIES), expected)
        self.assertEqual({s["category"] for s in self.stress}, expected)

    def test_stress_includes_empty_string(self):
        self.assertIn("", [s["text"] for s in self.stress if s["category"] == "tiny"])

    def test_stress_slices_span_the_64k_parallel_threshold(self):
        sizes = [len(s["text"].encode("utf-8")) for s in self.stress if s["category"] == "slice"]
        self.assertEqual(len(sizes), 3)
        for size in sizes:
            self.assertTrue(60_000 <= size <= 140_000, size)
        self.assertTrue(any(size > 65_536 for size in sizes))

    def test_sha256sums_cover_outputs(self):
        lines = (self.out / "SHA256SUMS").read_text().splitlines()
        listed = dict(reversed(line.split("  ", 1)) for line in lines)
        self.assertEqual(set(listed), {"conversations.jsonl", "stress.jsonl", "manifest.json"})
        for name, digest in listed.items():
            self.assertEqual(hashlib.sha256((self.out / name).read_bytes()).hexdigest(), digest, name)

    def test_manifest_records_seed_and_source_hashes(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest.get("seed"), 7)
        cjk = self.sources["cjk_files"][0]
        self.assertIn(
            {"path": str(cjk), "sha256": hashlib.sha256(cjk.read_bytes()).hexdigest()}, manifest.get("sources", [])
        )


class RepoSourcesTest(unittest.TestCase):
    def test_only_git_tracked_files_are_used(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "crates" / "a" / "src").mkdir(parents=True)
            (repo / "model_gateway" / "src").mkdir(parents=True)
            (repo / "crates" / "a" / "src" / "lib.rs").write_text("pub fn a() {}\n")
            (repo / "model_gateway" / "src" / "main.rs").write_text("fn main() {}\n")
            (repo / "README.md").write_text("# readme\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            (repo / "notes.md").write_text("untracked local notes\n")
            (repo / "crates" / "a" / "src" / "scratch.rs").write_text("// untracked\n")
            code, prose = bc._repo_sources(repo)
        self.assertEqual([p.relative_to(repo).as_posix() for p in code], ["crates/a/src/lib.rs", "model_gateway/src/main.rs"])
        self.assertEqual([p.relative_to(repo).as_posix() for p in prose], ["README.md"])


if __name__ == "__main__":
    unittest.main()
