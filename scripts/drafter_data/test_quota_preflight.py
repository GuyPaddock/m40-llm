"""Allocation evidence must separate hash accidents, true constraints and unknown."""

import itertools
import unittest
from dataclasses import replace

from quota_preflight import Component, Requirements, check_witness, hash_role, native_roles, solve


class QuotaTests(unittest.TestCase):
    """Exercise count-only synthetic problems, never actual source selection."""

    def test_hash_shortfall_has_legal_quota_aware_witness(self) -> None:
        """Two synthetic groups hashing to train still permit calibration."""
        groups = [Component("g0", 0, (0, 1), (2, 0, 0)), Component("g1", 0, (0, 1), (1, 0, 0))]
        required = Requirements(((2, 1, 0),), (0,), (0, 0, 0))
        fixed = [replace(g, roles=(hash_role(g),)) for g in groups]
        result = solve(groups, required)
        self.assertEqual(result.status, "feasible")
        self.assertTrue(check_witness(groups, required, result.witness))
        self.assertEqual(solve(fixed, required).status, "infeasible")

    def test_native_role_and_bucket_shortfalls_are_real(self) -> None:
        """Native heldout data cannot repair train, nor long rows short buckets."""
        group = Component("a", 0, native_roles("docred", "test"), (0, 0, 10))
        train = Requirements(((1, 0, 0),), (0,), (0, 0, 0))
        bucket = Requirements(((0, 0, 1),), (1,), (1, 0, 0))
        self.assertEqual(solve([group], train).status, "infeasible")
        self.assertEqual(solve([group], bucket).status, "infeasible")
        with self.assertRaises(ValueError):
            native_roles("docred", "train_distant")

    def test_only_contributing_groups_count_toward_minimum(self) -> None:
        """Assigned but unused components cannot satisfy heldout independence."""
        groups = [Component("a", 0, (2,), (3, 0, 0)), Component("b", 0, (2,), (0, 1, 0))]
        required = Requirements(((0, 0, 2),), (2,), (1, 1, 0))
        result = solve(groups, required)
        self.assertEqual(result.status, "feasible")
        self.assertTrue(all(sum(row) > 0 for row in result.witness.counts))
        self.assertEqual(solve(groups, replace(required, heldout_groups=(3,))).status, "infeasible")

    def test_indivisible_group_competition_requires_joint_proof(self) -> None:
        """Per-role capacity can pass while one component cannot fill both roles."""
        groups = [Component("a", 0, (0, 1), (2, 0, 0))]
        result = solve(groups, Requirements(((1, 1, 0),), (0,), (0, 0, 0)))
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.reason, "complete integer enumeration")

    def test_global_buckets_can_be_supplied_by_different_families(self) -> None:
        """No extra per-family bucket rule is silently added."""
        groups = [Component("a", 0, (2,), (1, 0, 0)), Component("b", 1, (2,), (0, 1, 0))]
        required = Requirements(((0, 0, 1), (0, 0, 1)), (1, 1), (1, 1, 0))
        self.assertEqual(solve(groups, required).status, "feasible")

    def test_canonical_assignment_is_permutation_invariant(self) -> None:
        """Input ordering cannot change destinations or selected bucket counts."""
        groups = [Component(str(i), 0, (0, 1, 2), (1, 1, 0)) for i in range(3)]
        required = Requirements(((1, 1, 1),), (1,), (1, 0, 0))
        expected = solve(groups, required)
        self.assertEqual(expected.status, "feasible")
        for permutation in itertools.permutations(groups):
            self.assertEqual(solve(list(permutation), required).witness, expected.witness)
        self.assertFalse(check_witness(groups, required, replace(expected.witness, departures=999)))

    def test_exhausted_or_oversized_solve_is_unknown(self) -> None:
        """No proof can be inferred from a timeout or size cap."""
        groups = [Component("a", 0, (0, 1), (2, 0, 0))]
        required = Requirements(((1, 0, 0),), (0,), (0, 0, 0))
        self.assertEqual(solve(groups, required, max_states=0).status, "unknown")
        self.assertEqual(solve(groups, required, cpu_seconds=0).status, "unknown")
        self.assertEqual(solve(groups, required, max_groups=0).status, "unknown")

    def test_duplicate_and_negative_capacity_cannot_inflate_counts(self) -> None:
        """Malformed source counts fail validation instead of proving feasibility."""
        group = Component("a", 0, (0,), (1, 0, 0))
        required = Requirements(((1, 0, 0),), (0,), (0, 0, 0))
        with self.assertRaises(ValueError):
            solve([group, group], required)
        with self.assertRaises(ValueError):
            solve([replace(group, buckets=(-1, 0, 0))], required)


if __name__ == "__main__":
    unittest.main()
