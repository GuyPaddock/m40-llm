"""Registry boundaries must not hide source content or change natural inputs."""

import tempfile
import unittest
from pathlib import Path

from registry_preflight import inspect_overlap, load_registry, render_input

# Bind tests to the approved review artifact, not a moving instruction registry.
ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs/experiments/2026-10-30-v3-generic-instruction-registry-v1.json"
DIGEST = "b8c250e9a15ec5e53c8970094adb5c14b7190f6de10694a636dd3966deda100b"


class RegistryTests(unittest.TestCase):
    """Exercise exact bytes, Unicode offsets, payload lookalikes and fail-closed identity."""

    def test_unicode_payload_preserves_bytes_and_structural_heading(self) -> None:
        """Unicode and a payload heading cannot move or create an exemption."""
        source = load_registry(REGISTRY, ROOT, DIGEST)["squad-original"]
        fields = {"context": "caf\u00e9 \U0001f642\n\nQuestion:\nprivate fact", "question": "Who?"}
        result = render_input(source, source["revision"], fields)
        self.assertEqual(result.text, "Context:\n" + fields["context"] + "\n\nQuestion:\nWho?")
        self.assertEqual(len(result.spans), 2)
        span = result.spans[1]
        self.assertEqual(result.text.encode()[span.byte_start:span.byte_end], b"\n\nQuestion:\n")
        self.assertGreater(span.byte_start, span.character_start)
        payload_start = result.text.index("\n\nQuestion:\n")
        overlap = inspect_overlap(result, (payload_start, payload_start + 12), result,
                                  (span.character_start, span.character_end))
        self.assertFalse(overlap.is_generic_candidate)
        self.assertEqual(overlap.left_text, "\n\nQuestion:\n")

    def test_generic_match_retains_evidence_and_requires_second_review(self) -> None:
        """Even an exact approved candidate is not an activated exemption."""
        source = load_registry(REGISTRY, ROOT, DIGEST)["docred-original"]
        left = render_input(source, source["revision"], {"sentences": "Alice works here."})
        right = render_input(source, source["revision"], {"sentences": "Bob lives elsewhere."})
        match = inspect_overlap(left, (0, 62), right, (0, 62))
        self.assertTrue(match.is_generic_candidate)
        self.assertFalse(match.is_exempt)
        self.assertEqual(match.left_text, left.text[:62])
        self.assertEqual(match.left_range, (0, 62))
        self.assertFalse(inspect_overlap(left, (0, 63), right, (0, 63)).is_generic_candidate)

    def test_unlisted_natural_instruction_is_payload(self) -> None:
        """UltraChat cannot inherit DocRED's exception by containing its wording."""
        source = load_registry(REGISTRY, ROOT, DIGEST)["ultrachat-first-user-sft"]
        result = render_input(source, source["revision"], {"prompt": "Extract the entities and their relations from this document."})
        self.assertEqual(result.spans, ())
        self.assertFalse(inspect_overlap(result, (0, 20), result, (0, 20)).is_generic_candidate)

    def test_code_delimiter_does_not_exempt_filename_or_code(self) -> None:
        """The delimiter ends before the file path and excludes payload copies."""
        source = load_registry(REGISTRY, ROOT, DIGEST)["commitpackft-python"]
        result = render_input(source, source["revision"],
                              {"subject": "Fix \u00e9", "old_file": "a.py", "old_contents": "#\n\nFile: secret"})
        span = result.spans[0]
        self.assertEqual(result.text[span.character_end:], "a.py\n#\n\nFile: secret")
        self.assertFalse(inspect_overlap(result, (span.character_start, span.character_end + 1),
                                        result, (span.character_start, span.character_end + 1)).is_generic_candidate)

    def test_identity_mismatch_and_bad_coordinates_fail_closed(self) -> None:
        """Changed registry/adapters/revisions cannot silently authorize masking."""
        with self.assertRaises(ValueError):
            load_registry(REGISTRY, ROOT, "0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_registry(REGISTRY, Path(directory), DIGEST)
            adapter = Path(directory) / "scripts/drafter_data/hf_sample_audit.py"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"changed adapter")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                load_registry(REGISTRY, Path(directory), DIGEST)
        source = load_registry(REGISTRY, ROOT, DIGEST)["docred-original"]
        with self.assertRaises(ValueError):
            render_input(source, "wrong", {"sentences": "A"})
        result = render_input(source, source["revision"], {"sentences": "A"})
        with self.assertRaises(ValueError):
            inspect_overlap(result, (-1, 2), result, (0, 2))


if __name__ == "__main__":
    unittest.main()
