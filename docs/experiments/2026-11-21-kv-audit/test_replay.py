"""Independent controls for the retained-logit audit."""

import array
import math
import struct
import tempfile
import unittest
from pathlib import Path

from replay import check_value, compare, read_logits


class LogitReplayTests(unittest.TestCase):
    """Exercise probability, tie-breaking and evidence corruption contracts."""

    def test_uniform_reference_and_changed_candidate(self) -> None:
        """A uniform dense vector gives a hand-computable two-token KL."""
        dense = array.array("f", [0., 0.])
        other = array.array("f", [0., math.log(3)])
        result = compare(dense, other, 0)
        self.assertAlmostEqual(result["kl_reference_candidate"], .5 * math.log(4/3))
        self.assertEqual(result["reference"]["top_indices"], [0, 1])
        self.assertEqual(result["candidate"]["target_rank"], 2)
        self.assertEqual(result["top10_overlap"], 2)
        with self.assertRaisesRegex(ValueError, "target_rank"):
            check_value(result["candidate"]["target_rank"], 1, "target_rank")

    def test_truncated_and_nonfinite_vectors(self) -> None:
        """A missing vocabulary element or a NaN invalidates the readback."""
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "logits"
            file.write_bytes(struct.pack("<f", 0.))
            with self.assertRaisesRegex(ValueError, "size"):
                read_logits(file, 2)
            file.write_bytes(struct.pack("<ff", 0., math.nan))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                read_logits(file, 2)

    def test_target_preserves_binary32_round_trip(self) -> None:
        """JSON decimal formatting can differ while the target's FP32 bits agree."""
        check_value(1.0000001192092896, 1.0000001, "candidate.target_logit")
        with self.assertRaisesRegex(ValueError, "bits differ"):
            check_value(1.0000001192092896, 1.0000002, "candidate.target_logit")
