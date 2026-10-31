"""The established optimizer must reproduce the independent integer reference."""

import unittest
from dataclasses import replace

from cp_sat_allocation import allocate
from quota_preflight import Component, Requirements, check_witness, solve


class OptimizerTests(unittest.TestCase):
    """Compare exact outcomes and canonical witnesses, not solver implementation text."""

    def test_small_canonical_results_match_exhaustive_reference(self) -> None:
        """Flexible roles, bucket choices and hash departures retain the exact objective."""
        problems = [
            ([Component("g0", 0, (0, 1), (2, 0, 0)), Component("g1", 0, (0, 1), (1, 0, 0))],
             Requirements(((2, 1, 0),), (0,), (0, 0, 0))),
            ([Component(str(i), 0, (0, 1, 2), (1, 1, 0)) for i in range(3)],
             Requirements(((1, 1, 1),), (1,), (1, 0, 0))),
            ([Component("a", 0, (2,), (2, 1, 0)), Component("b", 1, (2,), (1, 2, 0))],
             Requirements(((0, 0, 1), (0, 0, 1)), (1, 1), (1, 1, 0))),
        ]
        for groups, required in problems:
            expected = solve(groups, required)
            self.assertEqual(expected.status, "feasible")
            for order in (groups, list(reversed(groups))):
                result = allocate(order, required)
                self.assertTrue(result.is_canonical)
                self.assertTrue(result.is_minimum_departure)
                self.assertEqual(result.witness, expected.witness)

    def test_infeasible_and_unknown_remain_distinct(self) -> None:
        """Native shortages and indivisibility have proofs; no-budget runs do not."""
        groups = [Component("one", 0, (0, 1), (2, 0, 0))]
        required = Requirements(((1, 1, 0),), (0,), (0, 0, 0))
        result = allocate(groups, required)
        self.assertEqual(result.feasibility, "infeasible")
        self.assertIn("independent exhaustive", result.reason)
        self.assertEqual(allocate(groups, required, max_calls=0).feasibility, "unknown")
        self.assertEqual(allocate(groups, required, cpu_seconds=0).feasibility, "unknown")
        self.assertEqual(allocate([replace(groups[0], roles=(2,))], required).feasibility, "infeasible")

    def test_interrupted_canonicalization_keeps_checked_feasibility(self) -> None:
        """A minimum-departure witness is not falsely advertised as canonical."""
        groups = [Component(str(i), 0, (0, 1, 2), (1, 1, 0)) for i in range(3)]
        required = Requirements(((1, 1, 1),), (1,), (1, 0, 0))
        result = allocate(groups, required, max_calls=1)
        self.assertEqual(result.feasibility, "feasible")
        self.assertFalse(result.is_canonical)
        self.assertTrue(check_witness(groups, required, result.witness))
        self.assertFalse(check_witness(groups, required, replace(result.witness, counts=((99, 0, 0),) * 3)))

    def test_contributing_group_floor_and_global_buckets(self) -> None:
        """Assigned empty groups cannot count; bucket requirements are global."""
        groups = [Component("a", 0, (2,), (3, 0, 0)), Component("b", 0, (2,), (0, 1, 0))]
        required = Requirements(((0, 0, 2),), (2,), (1, 1, 0))
        result = allocate(groups, required)
        self.assertTrue(result.is_canonical)
        self.assertEqual(result.witness, solve(groups, required).witness)
        self.assertEqual(allocate(groups, replace(required, heldout_groups=(3,))).feasibility, "infeasible")


if __name__ == "__main__":
    unittest.main()
