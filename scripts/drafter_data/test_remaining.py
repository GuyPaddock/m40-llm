"""Remaining CPU audits must preserve semantics and inspect archives inertly."""

import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from exclusion_inventory import parse_text, read_texts
from mbpp_remaining import scan_exclusions
from template_audit import normalized_solution, shared_span_edges


class RemainingAuditTests(unittest.TestCase):
    """Compare observable template identities, span edges and extracted texts."""

    def test_alpha_renaming_groups_equivalent_simple_templates(self) -> None:
        """Renamed function arguments and locals should share a template ID."""
        left = "def first(x):\n total=x+1\n return total"
        right = "def second(data):\n answer=data+1\n return answer"
        self.assertIsNotNone(normalized_solution(left))
        self.assertEqual(normalized_solution(left), normalized_solution(right))

    def test_preserves_constants_and_external_calls(self) -> None:
        """Different operations must not merge just because locals were renamed."""
        baseline = normalized_solution("def f(x):\n return sum(x)+1")
        self.assertNotEqual(baseline, normalized_solution("def g(y):\n return max(y)+1"))
        self.assertNotEqual(baseline, normalized_solution("def g(y):\n return sum(y)+2"))

    def test_nested_scope_is_unresolved(self) -> None:
        """Unsupported binding scopes need review rather than false equivalence."""
        self.assertIsNone(normalized_solution("def f(x):\n return [y for y in x]"))

    def test_span_threshold_and_offset(self) -> None:
        """A 32-token span matches at any offset; 31 tokens do not satisfy it."""
        ids = list(range(32))
        self.assertEqual(shared_span_edges({1: ids, 2: [99] + ids, 3: ids[:31]}), [[1, 2]])

    def test_extracts_json_answers_and_raw_prompt_literals(self) -> None:
        """Generated answers and multiline raw prompts belong to exclusions."""
        values, errors = parse_text('{"prompt":"input", "output":"answer", "ids":[1,2]}', ".json")
        self.assertEqual(values, ["input", "answer"])
        self.assertEqual(errors, [])
        values, errors = parse_text('let x = r#"first\nsecond"#;', ".rs")
        self.assertIn("first\nsecond", values)

    def test_archive_members_are_read_without_path_extraction(self) -> None:
        """Even a traversal member name cannot write outside the archive."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tar.gz"
            payload = gzip.compress(json.dumps({"output": "retained generated answer"}).encode())
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo("../../must-not-extract.json.gz")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            texts, errors = read_texts(path)
            self.assertIn("retained generated answer", texts)
            self.assertEqual(errors, [])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_benchmark_witnesses_cover_duplicate_sources_and_spans(self) -> None:
        """All duplicate tasks inherit exclusion, including a token-only hit."""
        with tempfile.TemporaryDirectory() as directory:
            texts = Path(directory) / "texts.jsonl"
            tokens = Path(directory) / "tokens.jsonl"
            texts.write_text(json.dumps({"id": "witness", "comparison_text": "same task"}) + "\n")
            tokens.write_text(json.dumps({"id": "witness", "comparison_ids": [999] + list(range(32))}) + "\n")
            hits = scan_exclusions(texts, tokens, {1: "same task", 2: "same task", 3: "different"},
                                   {1: [], 2: [], 3: list(range(32))})
            self.assertEqual(set(hits), {1, 2, 3})
            self.assertIn("shared_32_runtime_tokens", [hit["reason"] for hit in hits[3]])

    def test_missing_benchmark_tokens_cannot_clear_exclusions(self) -> None:
        """Incomplete tokenizer coverage must fail rather than report clean."""
        with tempfile.TemporaryDirectory() as directory:
            texts = Path(directory) / "texts.jsonl"
            tokens = Path(directory) / "tokens.jsonl"
            texts.write_text(json.dumps({"id": "missing", "comparison_text": "task"}) + "\n")
            tokens.write_text("")
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                scan_exclusions(texts, tokens, {1: "unrelated"}, {1: []})


if __name__ == "__main__":
    unittest.main()
