"""Lossless candidate pruning must reproduce the exhaustive literal oracle."""

import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from census_matching import CandidateIndex, GroupRecord, IncompleteAudit, group_records, match_reasons


class MatchingTests(unittest.TestCase):
    """Check candidate completeness, retained raw pairs and conservative stops."""

    def test_prefix_candidates_match_exhaustive_oracle(self) -> None:
        """Indexed candidates retain all three rules on deterministic mixed content."""
        rng = random.Random(20261029)
        rows = [(" ".join(rng.choices([f"word{i}" for i in range(8)], k=20)),
                 tuple(rng.randrange(5) for _ in range(40))) for _ in range(60)]
        rows += [("one two three four five six", tuple(range(40))),
                 ("one two three four five six seven", tuple(range(32))),
                 (" ONE  TWO three four five six ", ()), ("", ()), ("", ())]
        expected = {(left, right): match_reasons(*rows[left], *rows[right])
                    for left, right in itertools.combinations(range(len(rows)), 2)
                    if match_reasons(*rows[left], *rows[right])}
        with tempfile.TemporaryDirectory() as directory:
            index = CandidateIndex(Path(directory) / "index.sqlite")
            try:
                for number, row in enumerate(rows):
                    index.add(number, *row)
                actual = {(a, b): tuple(json.loads(reasons)) for a, b, reasons
                          in index.connection.execute("SELECT * FROM matches")}
                self.assertEqual(actual, expected)
                self.assertEqual(index.connection.execute("SELECT raw FROM records WHERE id=62").fetchone()[0], rows[62][0])
            finally:
                index.close()

    def test_frequent_key_limit_is_incomplete_not_suppression(self) -> None:
        """A repeated template cannot be dropped to make the audit finish."""
        with tempfile.TemporaryDirectory() as directory:
            index = CandidateIndex(Path(directory) / "index.sqlite", max_candidates=1)
            try:
                index.add(1, "same raw text", ())
                index.add(2, "same raw text", ())
                with self.assertRaises(IncompleteAudit):
                    index.add(3, "same raw text", ())
                self.assertEqual(index.connection.execute("SELECT count(*) FROM matches").fetchone()[0], 1)
                with self.assertRaises(ValueError):
                    index.add(1, "replacement", ())
            finally:
                index.close()

    def test_transitive_native_and_family_quarantine(self) -> None:
        """Connecting through an ineligible vertex cannot split a leaked component."""
        rows = [GroupRecord(0, "squad", "train", "qa"),
                GroupRecord(1, "squad", "train", "qa"),
                GroupRecord(2, "squad", "validation", "qa"),
                GroupRecord(3, "docred", "test", "extraction"),
                GroupRecord(4, "ultra", "test_sft", "prose")]
        result = group_records(rows, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(result[0], ([0, 1, 2, 3], True))
        self.assertEqual(result[4], ([4], False))
        with self.assertRaises(ValueError):
            group_records(rows, [(0, 99)])


if __name__ == "__main__":
    unittest.main()
