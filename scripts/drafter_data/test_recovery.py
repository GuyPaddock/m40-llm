"""Provenance recovery must distinguish passage evidence from search hints."""

import tempfile
import unittest
from pathlib import Path

from recover_dolly import WikiProbe, compare_passage


class SavedWikiProbe(WikiProbe):
    """Serve inert saved API-shaped records without any network calls.

    Attributes:
        responses: Remaining fixture responses in request order.
    """

    def __init__(self, directory: Path, responses: list[dict[str, object]]) -> None:
        """Prepare a local evidence owner and a deterministic response sequence.

        Args:
            directory: Fresh output directory.
            responses: API fixture records in expected consumption order.
        """
        super().__init__(directory)
        self.responses = responses

    def fetch(self, parameters: dict[str, str]) -> dict[str, object]:
        """Return saved evidence while preserving request-count behavior.

        Args:
            parameters: Query to record, never sent over the network.

        Returns:
            Next fixture response.
        """
        self.requests.append({"parameters": parameters})
        return self.responses.pop(0)


class RecoveryTests(unittest.TestCase):
    """Exercise provenance classification without a model, network or GPU."""

    def test_full_passage_ignores_numeric_citations(self) -> None:
        """Rendering and numeric references may differ while every word matches."""
        context = " ".join(f"word{index}" for index in range(40))
        result = compare_passage(context + "[17]", f"<p>{context}</p><sup>[17]</sup>")
        self.assertTrue(result["is_full_passage_match"])
        self.assertEqual(result["coverage"], 1)

    def test_title_and_partial_matches_cannot_establish_identity(self) -> None:
        """A title or shared introduction is not the entire source passage."""
        context = " ".join(f"word{index}" for index in range(40))
        result = compare_passage(context, "<h1>word0</h1><p>word1 word2</p>")
        self.assertFalse(result["is_full_passage_match"])

    def test_embedded_scripts_do_not_supply_passage_evidence(self) -> None:
        """Unrendered executable text is neither executed nor matched as prose."""
        context = " ".join(f"word{index}" for index in range(40))
        self.assertFalse(compare_passage(context, f"<script>{context}</script>")["is_full_passage_match"])

    def test_short_common_phrase_remains_unresolved(self) -> None:
        """A ubiquitous short phrase cannot identify an upstream document."""
        self.assertFalse(compare_passage("short text", "<p>short text</p>")["is_full_passage_match"])

    def test_two_matching_documents_are_ambiguous(self) -> None:
        """Duplicate passage evidence cannot arbitrarily choose one document."""
        context = " ".join(f"word{index}" for index in range(40))
        responses: list[dict[str, object]] = [{"query": {"search": [
            {"pageid": 1, "title": "First"}, {"pageid": 2, "title": "Second"}]}}]
        for page_id in (1, 2):
            responses.extend([
                {"query": {"pages": [{"title": str(page_id), "revisions": [{"revid": page_id}]}]}},
                {"parse": {"text": f"<p>{context}</p>", "revid": page_id}},
            ])
        with tempfile.TemporaryDirectory() as directory:
            probe = SavedWikiProbe(Path(directory) / "probe", responses)
            result = probe.recover(1, context)
            self.assertIsNone(result["canonical_document"])
            self.assertEqual(result["confidence"], "unresolved")
            self.assertFalse(result["is_training_admitted"])

    def test_single_verified_revision_is_not_editor_attestation(self) -> None:
        """Reconstruction identifies an article without inventing editorial provenance."""
        context = " ".join(f"word{index}" for index in range(40))
        responses = [
            {"query": {"search": [{"pageid": 1, "title": "First"}]}},
            {"query": {"pages": [{"title": "First", "revisions": [{"revid": 20}]}]}},
            {"parse": {"text": f"<p>{context}</p>", "revid": 20}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = SavedWikiProbe(Path(directory) / "probe", responses).recover(1, context)
            self.assertEqual(result["canonical_document"], "enwiki:1")
            self.assertFalse(result["is_editor_source_attested"])
            self.assertFalse(result["is_training_admitted"])


if __name__ == "__main__":
    unittest.main()
