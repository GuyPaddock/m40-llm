"""Exercise boundary-specific cache geometry and answer scoring."""

import unittest

from history import Geometry, check_cache, passed


class BoundaryAuditTests(unittest.TestCase):
    """Reject wrong selected coverage and values of the wrong JSON type."""

    def test_clipping_begins_after_sixty_four_full_blocks(self) -> None:
        """At position 3,073 the decoder holds 65 blocks and selects at most 64."""
        state = Geometry(block_size=32, configured_old_block_budget=64, old_blocks=65,
                         consumed_tokens=3073, capacity=4096, allocated_bytes=143931024,
                         old_tokens=2049, physical_lengths=[3073]*36, is_compressed=True,
                         all_old_blocks_fit_top64=False)
        check_cache(state, "top64", 3073, False)
        state["configured_old_block_budget"] = 65
        with self.assertRaisesRegex(ValueError, "configured_old_block_budget"):
            check_cache(state, "top64", 3073, False)
        state["configured_old_block_budget"] = 64
        state["physical_lengths"][35] = 3072
        with self.assertRaisesRegex(ValueError, "physical_lengths"):
            check_cache(state, "top64", 3073, False)

    def test_fenced_json_preserves_boolean_and_number_types(self) -> None:
        """The frozen rubric rejects a numeric one in place of a JSON true."""
        rubric = {"kind":"json_object","expected":{"passed":True,"port":23}}
        self.assertTrue(passed('```json\n{"passed":true,"port":23}\n```', rubric))
        self.assertFalse(passed('{"passed":1,"port":23}', rubric))
        self.assertFalse(passed('{"passed":true,"port":23,"extra":1}', rubric))
        self.assertFalse(passed('invalid', rubric))
        self.assertTrue(passed(" KeSTrel\n", {"kind":"exact","answers":["Kestrel"]}))
