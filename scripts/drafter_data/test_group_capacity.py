"""CPU admission must quarantine duplicates without moving official partitions."""

import json
import tempfile
import unittest
from pathlib import Path

from group_capacity import Task, audit_groups
from prepare_v2 import audit_dolly, require_hash, write_new


class GroupCapacityTests(unittest.TestCase):
    """Check observable candidate dispositions and retained evidence ownership."""

    def test_cross_partition_component_is_transitive(self) -> None:
        """Text and solution edges together quarantine the entire component."""
        tasks = [
            Task(601, "train", "Count apples", "answer = 1", 16),
            Task(602, "train", "  COUNT apples  ", "answer = 2", 16),
            Task(511, "calibration", "Count pears", "answer=2\n", 17),
            Task(11, "held_out", "Find the maximum", "answer = 3", 20),
        ]
        result = audit_groups(tasks, [[1, 64]])
        self.assertEqual(result["residual_row_upper_bounds"], {"train": 0, "calibration": 0, "held_out": 1})
        rows = {row["task_id"]: row for row in result["rows"]}
        for task_id in (601, 602, 511):
            self.assertIn("cross_partition_component", rows[task_id]["quarantine_reasons"])
        self.assertEqual(rows[601]["source_split"], "train")
        self.assertEqual(rows[511]["source_split"], "calibration")
        self.assertFalse(result["is_r1_admitted"])

    def test_five_word_shingles_quarantine_near_duplicates(self) -> None:
        """The frozen Jaccard rule joins similar statements across partitions."""
        tasks = [Task(601, "train", "one two three four five six", "a = 1", 20),
                 Task(11, "held_out", "one two three four five six seven", "a = 2", 21)]
        result = audit_groups(tasks, [[1, 64]])
        self.assertEqual(result["residual_row_upper_bounds"]["train"], 0)
        self.assertEqual(result["residual_row_upper_bounds"]["held_out"], 0)

    def test_short_independent_statements_do_not_share_empty_shingles(self) -> None:
        """Empty shingle sets are not evidence of identical task templates."""
        result = audit_groups([Task(601, "train", "Sum values", "a=1", 16),
                               Task(11, "held_out", "Sort names", "a=2", 17)], [[1, 64]])
        self.assertEqual(result["residual_row_upper_bounds"]["train"], 1)
        self.assertEqual(result["residual_row_upper_bounds"]["held_out"], 1)
        self.assertEqual(result["edges"], [])

    def test_length_ineligible_row_still_connects_cross_partition_group(self) -> None:
        """Filtering before grouping must not conceal a partition conflict."""
        result = audit_groups([Task(601, "train", "same text", "a=1", 16),
                               Task(11, "held_out", "same text", "a=2", 500)], [[1, 64]])
        self.assertEqual(result["residual_row_upper_bounds"]["train"], 0)

    def test_solution_source_is_never_executed(self) -> None:
        """Parsing reference code cannot execute its file-writing side effects."""
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "must-not-exist"
            program = f"open({str(forbidden)!r}, 'w').write('bad')"
            result = audit_groups([Task(601, "train", "Read text", program, 16)], [[1, 64]])
            self.assertEqual(result["residual_row_upper_bounds"]["train"], 1)
            self.assertFalse(forbidden.exists())

    def test_unparseable_solution_is_unresolved_not_admitted(self) -> None:
        """A missing solution audit cannot silently increase eligible capacity."""
        result = audit_groups([Task(601, "train", "Read text", "not python !", 16)], [[1, 64]])
        self.assertEqual(result["residual_row_upper_bounds"]["train"], 0)
        self.assertIn("unresolved_solution_ast", result["rows"][0]["quarantine_reasons"])

    def test_dolly_context_hash_is_not_document_identity(self) -> None:
        """A context paragraph remains quarantined without reviewed provenance."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dolly.jsonl"
            path.write_text(json.dumps({"instruction": "Who?", "context": "A passage",
                                        "response": "Someone", "category": "closed_qa"}) + "\n",
                            encoding="utf-8")
            result = audit_dolly(path)
            self.assertEqual(result["strata"]["dolly_grounded_qa"]["unresolved_context_rows"], 1)
            self.assertIsNone(result["rows"][0]["canonical_upstream_document"])
            self.assertFalse(result["rows"][0]["is_selected"])

    def test_rejects_evidence_replacement_and_hash_mismatch(self) -> None:
        """Previously recorded artifacts cannot be overwritten or substituted."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            write_new(path, {"value": 1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, {"value": 2})
            with self.assertRaisesRegex(ValueError, "hash differs"):
                require_hash(path, "0" * 64)
            self.assertEqual(path.read_bytes(), original)

    def test_duplicate_ids_cannot_inflate_capacity(self) -> None:
        """Repeated source IDs fail the audit rather than becoming extra rows."""
        task = Task(601, "train", "One task", "a=1", 16)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_groups([task, task], [[1, 64]])


if __name__ == "__main__":
    unittest.main()
