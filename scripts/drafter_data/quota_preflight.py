"""Exact synthetic allocation separates capacity from a particular hash split.

This bounded reference enumerates the frozen integer formulation and validates
witnesses independently. It is not a production census solver: uncompleted
search or oversized problems return unknown, never a quota failure.
"""

import hashlib
import itertools
import time
from dataclasses import dataclass

# Destination order and seed are part of the approved deterministic contract.
ROLES = (0, 1, 2)
SEED = "20261029"


@dataclass(frozen=True)
class Component:
    """An audited group has one permitted destination, even if some rows are unused.

    Attributes:
        component_id: Stable group identity, synthetic in this preflight.
        family: Zero-based frozen family index.
        roles: Permitted native destinations: train=0, calibration=1, heldout=2.
        buckets: Unique eligible row capacities in the three natural-length buckets.
    """

    component_id: str
    family: int
    roles: tuple[int, ...]
    buckets: tuple[int, int, int]


@dataclass(frozen=True)
class Requirements:
    """All quotas and independence floors must be satisfied jointly.

    Attributes:
        quotas: Per-family train/calibration/heldout row counts.
        heldout_groups: Per-family minimum contributing heldout components.
        heldout_buckets: Global heldout row minima per length bucket.
    """

    quotas: tuple[tuple[int, int, int], ...]
    heldout_groups: tuple[int, ...]
    heldout_buckets: tuple[int, int, int]


@dataclass(frozen=True)
class Witness:
    """Count-only witness in seeded component order, never a source-row manifest.

    Attributes:
        component_ids: Explicit ordered group IDs binding the vector.
        roles: One role per component.
        counts: Selected row counts by bucket for that component's role.
        departures: Number of assignments differing from the historical hash.
    """

    component_ids: tuple[str, ...]
    roles: tuple[int, ...]
    counts: tuple[tuple[int, int, int], ...]
    departures: int


@dataclass(frozen=True)
class SolveResult:
    """Only completed proof/search certifies the canonical allocation result.

    Attributes:
        status: feasible, infeasible, or unknown.
        reason: Proof kind or incomplete-work reason.
        states: Number of exhaustive search states visited.
        witness: Checked feasible counts, possibly noncanonical on timeout.
    """

    status: str
    reason: str
    states: int
    witness: Witness | None


def native_roles(source: str, split: str) -> tuple[int, ...]:
    """Keep the approved native split restrictions explicit.

    Args:
        source: One of the four provisional source names.
        split: Native split, never a reassigned research role.

    Returns:
        Allowed research destinations.

    Raises:
        ValueError: A source or split is outside the approved scope.
    """
    mapping = {("ultrachat", "train_sft"): (0, 1), ("ultrachat", "test_sft"): (2,),
               ("commitpack", "python"): ROLES, ("squad", "train"): (0, 1),
               ("squad", "validation"): (2,), ("docred", "train_annotated"): (0,),
               ("docred", "dev"): (1,), ("docred", "test"): (2,)}
    if (source, split) not in mapping:
        raise ValueError("unapproved native source/split")
    return mapping[source, split]


def group_hash(component_id: str) -> int:
    """Compute the immutable seed's big-endian allocation hash.

    Args:
        component_id: Canonical group identity.

    Returns:
        SHA256 interpreted as an unsigned integer.
    """
    return int.from_bytes(hashlib.sha256(f"{SEED}:{component_id}".encode()).digest(), "big")


def hash_role(group: Component) -> int:
    """Apply the original allocator once, without trying alternative seeds.

    Args:
        group: Component with frozen allowed roles.

    Returns:
        Original permitted destination.
    """
    if len(group.roles) == 1:
        return group.roles[0]
    value = group_hash(group.component_id)
    if group.roles == (0, 1):
        return 1 if value % 304 < 48 else 0
    if group.roles != ROLES:
        raise ValueError("unsupported permitted destination set")
    remainder = value % 400
    return 0 if remainder < 256 else 1 if remainder < 304 else 2


def validate_problem(groups: list[Component], required: Requirements) -> None:
    """Reject malformed capacities instead of letting them certify a quota.

    Args:
        groups: All usable synthetic/audited components.
        required: Explicit quotas and minima.
    """
    if (len({g.component_id for g in groups}) != len(groups)
            or len(required.quotas) != len(required.heldout_groups)
            or any(len(row) != 3 for row in required.quotas)
            or len(required.heldout_buckets) != 3):
        raise ValueError("invalid identities or quota dimensions")
    numbers = [n for row in required.quotas for n in row] + list(required.heldout_groups) + list(required.heldout_buckets)
    for group in groups:
        if (not 0 <= group.family < len(required.quotas) or len(group.buckets) != 3
                or group.roles not in ((0,), (1,), (2,), (0, 1), ROLES)):
            raise ValueError("invalid component dimensions or native roles")
        numbers.extend(group.buckets)
    if any(type(n) is not int or n < 0 for n in numbers):
        raise ValueError("counts must be nonnegative integers")


def capacity_bound(groups: list[Component], required: Requirements) -> str | None:
    """Return a sufficient, independently recomputable impossibility bound.

    Args:
        groups: Complete scope for this synthetic capacity problem.
        required: Unchanged row and group requirements.

    Returns:
        A violated upper bound, or None; None is not proof of feasibility.
    """
    for family, quotas in enumerate(required.quotas):
        subset = [g for g in groups if g.family == family]
        for width in (1, 2, 3):
            for roles in itertools.combinations(ROLES, width):
                available = sum(sum(g.buckets) for g in subset if set(g.roles) & set(roles))
                needed = sum(quotas[role] for role in roles)
                if available < needed:
                    return f"family={family} roles={roles} capacity={available} required={needed}"
        count = sum(2 in g.roles and sum(g.buckets) > 0 for g in subset)
        if min(count, quotas[2]) < required.heldout_groups[family]:
            return f"family={family} heldout_components={count} row_quota={quotas[2]} minimum={required.heldout_groups[family]}"
    for bucket, minimum in enumerate(required.heldout_buckets):
        available = sum(g.buckets[bucket] for g in groups if 2 in g.roles)
        if available < minimum:
            return f"bucket={bucket} capacity={available} minimum={minimum}"
    if sum(row[2] for row in required.quotas) < sum(required.heldout_buckets):
        return "heldout row quota below sum of disjoint bucket minima"
    return None


def check_witness(groups: list[Component], required: Requirements, witness: Witness) -> bool:
    """Check integer constraints independently of search traversal and objective.

    Args:
        groups: Input capacities with native role restrictions.
        required: All quotas and independence floors.
        witness: Count-only assignment to verify.

    Returns:
        Whether the assignment satisfies every frozen feasibility constraint.
    """
    ordered = sorted(groups, key=lambda g: (group_hash(g.component_id), g.component_id))
    if (witness.component_ids != tuple(g.component_id for g in ordered)
            or len(witness.roles) != len(ordered) or len(witness.counts) != len(ordered)):
        return False
    counts = [[0, 0, 0] for _ in required.quotas]
    heldout = [0] * len(counts)
    buckets = [0, 0, 0]
    departures = 0
    for group, role, amounts in zip(ordered, witness.roles, witness.counts, strict=True):
        if (role not in group.roles or len(amounts) != 3
                or any(type(n) is not int or not 0 <= n <= cap for n, cap in zip(amounts, group.buckets))):
            return False
        counts[group.family][role] += sum(amounts)
        departures += role != hash_role(group)
        if role == 2:
            heldout[group.family] += sum(amounts) > 0
            buckets = [a + b for a, b in zip(buckets, amounts)]
    return (tuple(tuple(row) for row in counts) == required.quotas
            and all(a >= b for a, b in zip(heldout, required.heldout_groups))
            and all(a >= b for a, b in zip(buckets, required.heldout_buckets))
            and departures == witness.departures)


class ExhaustiveAllocator:
    """Finite integer reference with a deterministic lexicographic optimum.

    Attributes:
        groups: Seed-ordered components.
        required: Frozen count constraints.
        max_states: Search budget, not a data eligibility criterion.
        deadline: Process CPU deadline.
        states: States visited, including unfinished searches.
        best: Best checked witness found so far.
        best_key: Departures, destination vector, then flattened bucket counts.
    """

    def __init__(self, groups: list[Component], required: Requirements,
                 max_states: int, cpu_seconds: float) -> None:
        """Bind one bounded search without source rows or target-model outputs.

        Args:
            groups: Synthetic or audited count-only components.
            required: Joint constraints.
            max_states: Hard explored-state cap.
            cpu_seconds: Hard process CPU budget in seconds.
        """
        self.groups = sorted(groups, key=lambda g: (group_hash(g.component_id), g.component_id))
        self.required = required
        self.max_states = max_states
        self.deadline = time.process_time() + cpu_seconds
        self.states = 0
        self.best: Witness | None = None
        self.best_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None

    def visit(self, roles: tuple[int, ...], amounts: tuple[tuple[int, int, int], ...],
              remaining: tuple[tuple[int, int, int], ...], departures: int) -> None:
        """Enumerate integer y/x assignments; reject incomplete optimum claims.

        Args:
            roles: Assigned destination prefix.
            amounts: Assigned bucket-count prefix.
            remaining: Per-family row quotas remaining by destination.
            departures: Hash departures in the assigned prefix.

        Raises:
            TimeoutError: State or CPU cap prevents complete canonical search.
        """
        if self.states >= self.max_states or time.process_time() >= self.deadline:
            raise TimeoutError("state or CPU limit")
        self.states += 1
        if self.best_key is not None and departures > self.best_key[0]:
            return
        if len(roles) == len(self.groups):
            candidate = Witness(tuple(g.component_id for g in self.groups), roles, amounts, departures)
            if not any(sum(row) for row in remaining) and check_witness(self.groups, self.required, candidate):
                key = (departures, roles, tuple(n for row in amounts for n in row))
                if self.best_key is None or key < self.best_key:
                    self.best, self.best_key = candidate, key
            return
        group = self.groups[len(roles)]
        for role in group.roles:
            quota = remaining[group.family][role]
            choices = [range(min(capacity, quota) + 1) for capacity in group.buckets]
            for counts in itertools.product(*choices):
                # Count every attempted combination, even if its sum exceeds quota.
                if self.states >= self.max_states or time.process_time() >= self.deadline:
                    raise TimeoutError("state or CPU limit")
                self.states += 1
                if sum(counts) <= quota:
                    updated = [list(row) for row in remaining]
                    updated[group.family][role] -= sum(counts)
                    self.visit(roles + (role,), amounts + (counts,), tuple(tuple(row) for row in updated),
                               departures + (role != hash_role(group)))


def solve(groups: list[Component], required: Requirements, max_states: int = 200000,
          cpu_seconds: float = 10.0, max_groups: int = 12) -> SolveResult:
    """Prove small fixtures exactly; report larger or timed-out instances unknown.

    Args:
        groups: Count-only components; no selected source manifest is produced.
        required: Explicit quotas, fixed to the protocol for population estimates.
        max_states: Bounded enumeration work.
        cpu_seconds: CPU search allowance.
        max_groups: Reference-solver size guard, not a census grouping rule.

    Returns:
        Canonical feasible witness, certified infeasibility, or unknown.
    """
    validate_problem(groups, required)
    if max_states < 0 or cpu_seconds < 0 or not 0 <= max_groups <= 100:
        raise ValueError("invalid reference solver limit")
    bound = capacity_bound(groups, required)
    if bound is not None:
        return SolveResult("infeasible", "upper-bound: " + bound, 0, None)
    if len(groups) > max_groups:
        return SolveResult("unknown", "reference solver component limit; scalable solver required", 0, None)
    allocator = ExhaustiveAllocator(groups, required, max_states, cpu_seconds)
    try:
        allocator.visit((), (), required.quotas, 0)
    except TimeoutError:
        return SolveResult("unknown", "incomplete canonical solve", allocator.states, allocator.best)
    if allocator.best is None:
        return SolveResult("infeasible", "complete integer enumeration", allocator.states, None)
    return SolveResult("feasible", "complete lexicographic optimum", allocator.states, allocator.best)
