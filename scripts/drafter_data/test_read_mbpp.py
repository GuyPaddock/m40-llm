"""Test that the MBPP conversion preserves inert data and evidence ownership."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow
import pyarrow.parquet

from read_mbpp import convert_partition, read_prompt_rows


class MbppConversionTests(unittest.TestCase):
    """Exercise real Parquet input without network, models or a GPU."""

    def test_preserves_prompt_and_source_identity(self) -> None:
        """Keep Unicode and whitespace, excluding code and tests from input."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.parquet"
            destination = Path(temporary) / "converted.json"
            text = "  Count caf\u00e9 records.\n"
            table = pyarrow.table({
                "task_id": [601], "text": [text],
                "code": ["raise RuntimeError('must not execute')"],
                "test_list": [["assert False"]],
            })
            pyarrow.parquet.write_table(table, source)
            convert_partition(source, destination)
            result = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(result["rows"], [{"task_id": 601, "text": text}])
            self.assertEqual(result["source_rows"], 1)
            self.assertEqual(result["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                convert_partition(source, destination)

    def test_rejects_duplicate_source_ids(self) -> None:
        """Duplicates cannot inflate a partition's upper-bound capacity."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.parquet"
            pyarrow.parquet.write_table(pyarrow.table({
                "task_id": [601, 601], "text": ["first", "second"],
            }), source)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_prompt_rows(source)

    def test_rejects_null_prompt(self) -> None:
        """Missing prompts fail conversion instead of becoming empty examples."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.parquet"
            pyarrow.parquet.write_table(pyarrow.table({
                "task_id": [601], "text": [None],
            }), source)
            with self.assertRaisesRegex(ValueError, "string"):
                read_prompt_rows(source)


if __name__ == "__main__":
    unittest.main()
