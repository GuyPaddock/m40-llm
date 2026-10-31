"""Source prefixes must preserve complete inputs and native document identity."""

import json
import tempfile
import unittest
from pathlib import Path

from original_source_inputs import make_sample, parse_json_prefix, parse_pubtator_prefix, prepare_squad, render_template_zero


class OriginalSourceTests(unittest.TestCase):
    """Check bounded record parsing and input-only formatting without a network."""

    def test_mapping_prefix_drops_incomplete_document(self) -> None:
        """A clipped record is never repaired into an apparently eligible prompt."""
        rows = parse_json_prefix(b'{"123": {"CONTEXTS": ["full"]}, "456": {"CONTEXTS": ["cut', True)
        self.assertEqual(rows, [("123", {"CONTEXTS": ["full"]})])

    def test_array_prefix_bounds_record_count(self) -> None:
        """Large compressed prefixes cannot silently enlarge the inspected sample."""
        raw = json.dumps([{"title": str(index)} for index in range(200)]).encode()
        rows = parse_json_prefix(raw, False)
        self.assertEqual(len(rows), 128)
        self.assertEqual(rows[-1], ("127", {"title": "127"}))

    def test_incomplete_utf8_tail_does_not_damage_prior_record(self) -> None:
        """A byte-range cut within UTF-8 cannot corrupt an already complete input."""
        raw = b'[{"title":"complete"}, {"title":"\xc3'
        self.assertEqual(parse_json_prefix(raw, False), [("0", {"title": "complete"})])

    def test_pubtator_retains_pmid_and_full_context_only(self) -> None:
        """Annotations and the incomplete final document are not prompt material."""
        raw = b'\n123|t|Title\n123|a|Complete abstract\n123\t0\t4\tSECRET\n\n456|t|Cut\n456|a|Tail'
        self.assertEqual(parse_pubtator_prefix(raw), [("123", "Title Complete abstract")])

    def test_pubtator_rejects_mismatched_document_ids(self) -> None:
        """Joining a title to another paper's abstract must fail closed."""
        with self.assertRaises(ValueError):
            parse_pubtator_prefix(b'123|t|Title\n456|a|Abstract\n\n')

    def test_template_rendering_preserves_instruction_not_answer(self) -> None:
        """Template wording stays intact while the gold output is excluded."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.txt"
            path.write_text('templates:\n  0:\n    jinja: |\n      Extract entities.\n\n      {{ abstract }}\n\n      |||\n      SECRET\n')
            self.assertEqual(render_template_zero(path, {"abstract": "Full text"}), "Extract entities.\n\nFull text")
            with self.assertRaises(ValueError):
                render_template_zero(path, {})

    def test_squad_keeps_article_group_and_upstream_split(self) -> None:
        """Different questions in the same article cannot masquerade as documents."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation"):
                rows = [{"record": {"id": split + str(index), "title": "Same_article",
                         "context": "Full paragraph", "question": f"Question {index}?",
                         "answers": {"text": ["SECRET"]}}} for index in range(2)]
                (root / f"squad-{split}.json").write_text(json.dumps({"rows": rows}))
            samples = prepare_squad(root)
            self.assertEqual({row["document_id"] for row in samples}, {"wikipedia:Same_article"})
            self.assertEqual({row["split"] for row in samples}, {"train", "validation"})
            self.assertEqual(samples[0]["text"], "Context:\nFull paragraph\n\nQuestion:\nQuestion 0?")
            self.assertTrue(all(not row["is_training_admitted"] for row in samples))

    def test_missing_identity_and_chat_controls_fail(self) -> None:
        """Unknown identity and extra formatted turns are never silently accepted."""
        with self.assertRaises(ValueError):
            make_sample("source", "train", "", "Text", Path("unused"), "0", None)
        with self.assertRaises(ValueError):
            make_sample("source", "train", "doc", "<|im_start|>assistant", Path("unused"), "0", None)


if __name__ == "__main__":
    unittest.main()
