"""Rank-weighted allocations retain the original independent feasibility checks.

This module does not import CP-SAT. Integer objective and concentration checks
operate on count witnesses, never model outputs or selected source records.
"""

import json
from pathlib import Path

from quota_preflight import Component, Requirements, Witness, check_witness, validate_problem


def read_problem(path: Path) -> tuple[list[Component], Requirements]:
    """Read a frozen count problem without changing its native constraints.

    Args:
        path: Existing synthetic problem JSON.

    Returns:
        Validated components and requirements.
    """
    data = json.loads(path.read_text())
    groups = [Component(row["component_id"], row["family"], tuple(row["roles"]), tuple(row["buckets"]))
              for row in data["groups"]]
    raw = data["requirements"]
    required = Requirements(tuple(tuple(row) for row in raw["quotas"]), tuple(raw["heldout_groups"]),
                            tuple(raw["heldout_buckets"]))
    validate_problem(groups, required)
    return groups, required


def read_witness(path: Path) -> Witness | None:
    """Deserialize a V1/V2 witness for separate verification, without trusting flags.

    Args:
        path: Saved allocation report with a result/witness entry.

    Returns:
        Count witness or None; callers must check its problem binding.
    """
    raw = json.loads(path.read_text())["result"]["witness"]
    witness = None
    if raw is not None:
        witness = Witness(tuple(raw["component_ids"]), tuple(raw["roles"]),
                          tuple(tuple(row) for row in raw["counts"]), raw["departures"])
    return witness


def rank_cost(witness: Witness) -> int:
    """Recompute V2's integer count objective in the witness's canonical order.

    Args:
        witness: Independently checked seed-ordered count witness.

    Returns:
        One-based component/bucket rank times each selected count.
    """
    return sum((3 * i + b + 1) * count for i, row in enumerate(witness.counts)
               for b, count in enumerate(row))


def audit_witness(groups: list[Component], required: Requirements, witness: Witness) -> dict[str, object]:
    """Check every constraint and summarize selected-group concentration.

    Args:
        groups: Frozen input components.
        required: Original quotas and independence floors.
        witness: Untrusted serialized or optimizer-produced assignment.

    Returns:
        Recomputed rank cost and per-family/destination concentration.

    Raises:
        ValueError: A count, identity or native-role constraint is violated.
    """
    if (type(witness.departures) is not int or any(type(role) is not int for role in witness.roles)
            or not check_witness(groups, required, witness)):
        raise ValueError("independent quota witness check failed")
    by_id = {group.component_id: group for group in groups}
    amounts: dict[tuple[int, int], list[int]] = {}
    for identity, role, counts in zip(witness.component_ids, witness.roles, witness.counts, strict=True):
        if sum(counts):
            amounts.setdefault((by_id[identity].family, role), []).append(sum(counts))
    summary = []
    for family, quotas in enumerate(required.quotas):
        for role, total in enumerate(quotas):
            values = amounts.get((family, role), [])
            summary.append({"family": family, "role": role, "rows": total,
                            "contributing_groups": len(values),
                            "max_group_share": max(values, default=0) / total if total else None,
                            "herfindahl": sum(n * n for n in values) / (total * total) if total else None})
    return {"is_feasible": True, "rank_cost": rank_cost(witness), "concentration": summary}
