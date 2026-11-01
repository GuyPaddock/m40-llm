"""Frequency ordering must preserve literal matches, including common material."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from census_matching import IncompleteAudit, match_reasons
from frequency_matching import FrequencyIndex


class FrequencyTests(unittest.TestCase):
    """Compare the fixed index with the unchanged exhaustive full-content oracle."""

    def test_adversarial_matches_equal_exhaustive_oracle(self) -> None:
        """Headers, copied passages, exact boundaries and Jaccard=0.5 cannot disappear."""
        header = "Extract the entities and their relations from this document. "
        rows = [(header + " ".join(f"r{i}w{j}" for j in range(40)), tuple(range(i * 80, i * 80 + 60)))
                for i in range(20)]
        rows += [("one two three four five", ()), ("one two three four five six", ()),
                 ("one two three four five six seven", ()),
                 (" copied passage ", tuple(range(30)) + (999, 998, 997)),
                 ("COPIED   passage", tuple(range(30)) + (999, 998, 996)),
                 ("different raw content", tuple(range(31))), ("", ())]
        expected = {(a, b): match_reasons(*rows[a], *rows[b])
                    for a, b in itertools.combinations(range(len(rows)), 2)
                    if match_reasons(*rows[a], *rows[b])}
        for order in (list(range(len(rows))), list(reversed(range(len(rows))))):
            with tempfile.TemporaryDirectory() as directory:
                index = FrequencyIndex(Path(directory) / "index.sqlite")
                try:
                    for number in order:
                        index.add(number, *rows[number])
                    index.finish()
                    actual = {(a, b): tuple(json.loads(reasons)) for a, b, reasons
                              in index.connection.execute("SELECT * FROM matches")}
                    self.assertEqual(actual, expected)
                    self.assertIn("shared_32_token_span", actual[23, 24])
                    self.assertIn("five_word_jaccard_at_least_half", actual[20, 21])
                    self.assertEqual(index.connection.execute("SELECT raw FROM records WHERE id=23").fetchone()[0], rows[23][0])
                finally:
                    index.close()

    def test_repeated_header_no_longer_generates_all_pairs(self) -> None:
        """Rare-first order avoids false pairs without dropping the shared header."""
        header = " ".join(f"common{i}" for i in range(20)) + " "
        with tempfile.TemporaryDirectory() as directory:
            index = FrequencyIndex(Path(directory) / "index.sqlite")
            try:
                for i in range(200):
                    index.add(i, header + " ".join(f"r{i}w{j}" for j in range(60)), ())
                index.finish()
                self.assertEqual(index.candidate_checks, 0)
                self.assertEqual(index.connection.execute("SELECT count(*) FROM records").fetchone()[0], 200)
            finally:
                index.close()

    def test_true_common_matches_are_all_retained(self) -> None:
        """A frequent literal passage still creates every required pair."""
        with tempfile.TemporaryDirectory() as directory:
            index = FrequencyIndex(Path(directory) / "index.sqlite")
            try:
                for i in range(30):
                    index.add(i, "same boilerplate " + str(i), tuple(range(32)))
                index.finish()
                self.assertEqual(index.connection.execute("SELECT count(*) FROM matches").fetchone()[0], 435)
                self.assertEqual(index.candidate_checks, 435)
                with self.assertRaises(ValueError):
                    index.add(99, "late row", ())
            finally:
                index.close()

    def test_incomplete_index_never_reports_complete(self) -> None:
        """A forced deadline leaves explicit unknown evidence rather than empty clearance."""
        with tempfile.TemporaryDirectory() as directory:
            index = FrequencyIndex(Path(directory) / "index.sqlite")
            try:
                index.add(0, "text", ())
                index.cpu_deadline = 0
                with self.assertRaises(IncompleteAudit):
                    index.finish()
                self.assertEqual(index.connection.execute("SELECT status FROM state").fetchone()[0], "unknown")
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
