"""Fixed review selection and dependency closure cannot masquerade as admission."""

import unittest
from dataclasses import replace

from review_dependencies import build_dependencies
from review_records import ReviewRecord, select_review_records
from run_review_cost import create_packets


def make_record(identity: str, source: str, split: str, **metadata: object) -> ReviewRecord:
    """Construct a small inert review fixture with explicit source metadata.

    Args:
        identity: Unique locator.
        source: Frozen source family.
        split: Native split alias.
        metadata: Provisional source or license fields, not clearance.

    Returns:
        A natural-length synthetic record.
    """
    return ReviewRecord(identity, source, split, "fixture", (), 100, metadata)


def sample_fixture() -> list[ReviewRecord]:
    """Make all sixteen preregistered strata without content-based rescue."""
    rows = []
    for split in ("train", "test"):
        rows.extend(make_record(f"u-{split}-{i}", "ultrachat", split) for i in range(2))
    rows.extend(make_record(f"c-{i}", "commitpack", "python", license_label=label)
                for i, label in enumerate(("mit", "bsd-3-clause", "agpl-3.0", "epl-1.0")))
    for split in ("train", "validation"):
        rows.extend(make_record(f"s-{split}-{i}", "squad-original", split, provisional_group=split)
                    for i in range(2))
    rows.extend(make_record(f"d-{i}", "docred-original", split)
                for i, split in enumerate(("train_annotated", "train_annotated", "dev", "test")))
    return rows


class ReviewTests(unittest.TestCase):
    """Assert fixed locators and conservative local dependencies, not source text."""

    def test_fixed_sample_and_no_missing_stratum_rescue(self) -> None:
        """A missing license/native stratum stops rather than substituting easier rows."""
        rows = sample_fixture()
        self.assertEqual(select_review_records(rows), rows)
        with self.assertRaises(ValueError):
            select_review_records(rows[:-1])
        with self.assertRaises(ValueError):
            select_review_records([replace(r, prompt_tokens=64) if r.identity == "c-3" else r for r in rows])

    def test_unselected_ineligible_bridge_remains_in_closure(self) -> None:
        """Same-document and raw-match edges transit through unused out-of-range rows."""
        rows = [make_record("a", "squad-original", "train"),
                replace(make_record("bridge", "squad-original", "train"), prompt_tokens=1000),
                make_record("b", "docred-original", "test")]
        raw = {"a": {"title": "A"}, "bridge": {"title": "A"}, "b": {"title": "B"}}
        graph = build_dependencies(rows, raw, [(1, 2, "shared_32_token_span")])
        self.assertEqual(graph["membership"]["a"], [0, 1, 2])
        self.assertFalse(graph["has_complete_semantic_closure"])

    def test_code_aliases_and_unresolved_ast_are_retained(self) -> None:
        """Repository aliases connect code even when unsupported syntax prevents AST clearance."""
        rows = [make_record("a", "commitpack", "python"), make_record("b", "commitpack", "python")]
        raw = {"a": {"repos": "One/Repo, fork/repo", "commit": "1", "old_contents": "class A: pass"},
               "b": {"repos": "one/repo", "commit": "2", "old_contents": "class B: pass"}}
        graph = build_dependencies(rows, raw, [])
        self.assertEqual(graph["membership"]["b"], [0, 1])
        self.assertEqual(len(graph["unresolved_code_fields"]), 4)

    def test_literal_negative_packet_is_not_human_clearance(self) -> None:
        """No literal hits cannot clear unavailable messages, rights or human review."""
        row = make_record("a", "ultrachat", "train", sample_sha256="fixture")
        graph = {"membership": {"a": [0]}, "unresolved_code_fields": []}
        packet = create_packets([row], [row], {"a": {"prompt": "text", "prompt_id": "a"}},
                                graph, {"matches": {}})[0]
        self.assertEqual(packet["status"], "blocked")
        self.assertIn("independent_human_review_unavailable", packet["blockers"])
        self.assertIn("first_user_message_consistency_unverified", packet["blockers"])
        self.assertFalse(packet["has_complete_dependency_closure"])


if __name__ == "__main__":
    unittest.main()
