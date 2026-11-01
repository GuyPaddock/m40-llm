"""The amended last objective must not weaken feasibility or destination priorities."""

import itertools
import unittest
from dataclasses import replace

from quota_preflight import Component, Requirements, Witness, check_witness, group_hash, hash_role, solve
from rank_allocation import allocate_rank
from rank_audit import audit_witness, rank_cost


def enumerate_optimum(groups: list[Component], required: Requirements) -> tuple[int, tuple[int, ...], int] | None:
    """Independently enumerate tiny integer fixtures under V2's objective order.

    Args:
        groups: At most four tiny components.
        required: Explicit tiny constraints.

    Returns:
        Optimal departure/destination/rank tuple, or None if proven infeasible.
    """
    if len(groups) > 4 or sum(sum(g.buckets) for g in groups) > 12:
        raise ValueError("exhaustive fixture too large")
    ordered = sorted(groups, key=lambda g: (group_hash(g.component_id), g.component_id))
    best = None
    options = [list(itertools.product(*(range(n + 1) for n in g.buckets))) for g in ordered]
    for roles in itertools.product(*(g.roles for g in ordered)):
        for counts in itertools.product(*options):
            candidate = Witness(tuple(g.component_id for g in ordered), roles, counts,
                                sum(role != hash_role(g) for g, role in zip(ordered, roles)))
            if check_witness(groups, required, candidate):
                # Deliberately do not call the production objective calculator.
                cost = sum((3 * i + b + 1) * n for i, row in enumerate(counts) for b, n in enumerate(row))
                key = candidate.departures, candidate.roles, cost
                if best is None or key < best:
                    best = key
    return best


class RankTests(unittest.TestCase):
    """Compare observable witnesses, proof flags, interruption and concentration."""

    def test_exact_objective_and_repeatability(self) -> None:
        """Global buckets, native roles and nonzero hash repairs retain exact priorities."""
        cases = [
            ([Component("g0", 0, (0, 1), (2, 0, 0)), Component("g1", 0, (0, 1), (1, 0, 0))],
             Requirements(((2, 1, 0),), (0,), (0, 0, 0))),
            ([Component(str(i), 0, (0, 1, 2), (1, 1, 0)) for i in range(3)],
             Requirements(((1, 1, 1),), (1,), (1, 0, 0))),
            ([Component("a", 0, (2,), (2, 1, 0)), Component("b", 1, (2,), (1, 2, 0))],
             Requirements(((0, 0, 1), (0, 0, 1)), (1, 1), (1, 1, 0))),
        ]
        for groups, required in cases:
            expected = enumerate_optimum(groups, required)
            original = solve(groups, required)
            witnesses = []
            for order in (groups, list(reversed(groups)), groups):
                result = allocate_rank(order, required)
                self.assertTrue(result.is_rank_optimal)
                self.assertTrue(result.is_minimum_departure)
                self.assertTrue(result.has_canonical_destinations)
                self.assertEqual((result.witness.departures, result.witness.roles, result.rank_cost), expected)
                self.assertEqual(result.witness.roles, original.witness.roles)
                audit_witness(groups, required, result.witness)
                witnesses.append(result.witness)
            self.assertEqual(witnesses[0], witnesses[1])
            self.assertEqual(witnesses[1], witnesses[2])

    def test_only_count_objective_changes(self) -> None:
        """V1 chooses later counts; V2's lower rank cost may choose earlier groups."""
        groups = [Component(str(i), 0, (2,), (1, 0, 0)) for i in range(4)]
        required = Requirements(((0, 0, 2),), (2,), (2, 0, 0))
        original = solve(groups, required).witness
        result = allocate_rank(groups, required)
        self.assertTrue(result.is_rank_optimal)
        self.assertLess(result.rank_cost, rank_cost(original))
        self.assertEqual(result.witness.roles, original.roles)
        self.assertEqual(result.witness.departures, original.departures)
        audit = audit_witness(groups, required, result.witness)
        self.assertEqual(audit["concentration"][2]["contributing_groups"], 2)
        self.assertEqual(audit["concentration"][2]["herfindahl"], 0.5)

    def test_incomplete_and_infeasible_are_distinct(self) -> None:
        """Resource stops do not certify optimality or fabricate quota failures."""
        groups = [Component(str(i), 0, (0, 1, 2), (1, 1, 0)) for i in range(3)]
        required = Requirements(((1, 1, 1),), (1,), (1, 0, 0))
        partial = allocate_rank(groups, required, max_calls=1)
        self.assertEqual(partial.feasibility, "feasible")
        self.assertFalse(partial.is_rank_optimal)
        self.assertEqual(allocate_rank(groups, required, max_calls=0).feasibility, "unknown")
        impossible = [Component("one", 0, (0, 1), (2, 0, 0))]
        quota = Requirements(((1, 1, 0),), (0,), (0, 0, 0))
        self.assertIsNone(enumerate_optimum(impossible, quota))
        self.assertEqual(allocate_rank(impossible, quota).feasibility, "infeasible")

    def test_witness_corruption_and_independence_floor(self) -> None:
        """Empty groups cannot satisfy independence; malformed serialized counts fail."""
        groups = [Component("a", 0, (2,), (2, 0, 0)), Component("b", 0, (2,), (0, 2, 0))]
        required = Requirements(((0, 0, 2),), (2,), (1, 1, 0))
        witness = allocate_rank(groups, required).witness
        with self.assertRaises(ValueError):
            audit_witness(groups, required, replace(witness, counts=((2, 0, 0), (0, 0, 0))))
        with self.assertRaises(ValueError):
            audit_witness(groups, required, replace(witness, departures=False))
        self.assertEqual(allocate_rank(groups, replace(required, heldout_groups=(3,))).feasibility, "infeasible")


if __name__ == "__main__":
    unittest.main()
