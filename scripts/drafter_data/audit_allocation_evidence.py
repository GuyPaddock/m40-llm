"""Audit serialized count witnesses without importing or trusting CP-SAT.

Feasibility uses the preserved integer checker. Independent canonical checks
use exhaustive small problems or an analytically forced fixed-role instance;
larger solver proofs are reported as not independently replayed.
"""

import json
from pathlib import Path

from prepare_v2 import file_hash
from quota_preflight import Component, Requirements, Witness, check_witness, solve, validate_problem


def audit_evidence(problem_path: Path, result_path: Path) -> dict[str, object]:
    """Verify the problem binding and every serialized witness count independently.

    Args:
        problem_path: Frozen synthetic count problem, not source-row selection.
        result_path: Optimizer evidence bound to that problem's SHA256.

    Returns:
        Feasibility check and scope of independently reproduced canonical proof.

    Raises:
        ValueError: Identity, integer constraints or proof flags are inconsistent.
    """
    problem = json.loads(problem_path.read_text())
    report = json.loads(result_path.read_text())
    if report["problem_sha256"] != file_hash(problem_path):
        raise ValueError("problem identity differs")
    groups = [Component(row["component_id"], row["family"], tuple(row["roles"]), tuple(row["buckets"]))
              for row in problem["groups"]]
    raw = problem["requirements"]
    required = Requirements(tuple(tuple(row) for row in raw["quotas"]), tuple(raw["heldout_groups"]),
                            tuple(raw["heldout_buckets"]))
    validate_problem(groups, required)
    result = report["result"]
    canonical_check = "not independently replayed"
    is_feasible = False
    if result["witness"] is not None:
        row = result["witness"]
        witness = Witness(tuple(row["component_ids"]), tuple(row["roles"]),
                          tuple(tuple(counts) for counts in row["counts"]), row["departures"])
        if (any(type(role) is not int for role in witness.roles)
                or not check_witness(groups, required, witness)):
            raise ValueError("serialized witness violates integer constraints")
        if result["feasibility"] != "feasible":
            raise ValueError("witness and feasibility status disagree")
        is_feasible = True
        if result["is_canonical"] and not result["is_minimum_departure"]:
            raise ValueError("canonical claim lacks minimum-departure proof")
        if len(groups) <= 12:
            reference = solve(groups, required)
            if reference.status == "feasible":
                if result["is_canonical"] and witness != reference.witness:
                    raise ValueError("canonical result differs from exhaustive oracle")
                canonical_check = "exhaustive reference" if result["is_canonical"] else "claim absent"
        elif all(len(g.roles) == 1 for g in groups):
            totals = [[0, 0, 0] for _ in required.quotas]
            for group in groups:
                totals[group.family][group.roles[0]] += sum(group.buckets)
            if tuple(tuple(row) for row in totals) == required.quotas:
                canonical_check = "unique fixed-role all-capacity solution"
    elif result["feasibility"] == "feasible" or result["is_canonical"]:
        raise ValueError("claimed feasibility without witness")
    return {"problem_sha256": file_hash(problem_path), "result_sha256": file_hash(result_path),
            "is_witness_feasible": is_feasible, "canonical_check": canonical_check,
            "is_independent_human_review": False}
