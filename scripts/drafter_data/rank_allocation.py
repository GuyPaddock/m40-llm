"""V2 replaces only the final count-vector objective with a fixed rank sum.

All constraints and exact destination priorities come from the preserved V1
model. Optimality and reproducibility are separate: equal rank optima do not
have V1's solver-independent unique count vector.
"""

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from cp_sat_allocation import QuotaModel
from quota_preflight import Component, Requirements, Witness, capacity_bound, hash_role, solve, validate_problem
from rank_audit import rank_cost


@dataclass(frozen=True)
class RankResult:
    """Proof flags describe each objective, never an inferred dataset admission.

    Attributes:
        feasibility: feasible, infeasible or unknown.
        is_minimum_departure: Whether the original first objective was proven.
        has_canonical_destinations: Whether the unchanged destination tie-break completed.
        is_rank_optimal: Whether the amended final objective completed optimally.
        rank_cost: Independently recomputed value of the available witness.
        reason: Proof or resource-stop explanation.
        solver_status: Last native solver status.
        solve_calls: Attempted solver invocations.
        witness: Independently checked count-only result, if available.
        cpu_seconds: Whole allocator CPU time, including construction.
        wall_seconds: Whole allocator wall time.
    """

    feasibility: str
    is_minimum_departure: bool
    has_canonical_destinations: bool
    is_rank_optimal: bool
    rank_cost: int | None
    reason: str
    solver_status: str
    solve_calls: int
    witness: Witness | None
    cpu_seconds: float
    wall_seconds: float


def fix_destinations(model: QuotaModel, optimum: int) -> bool:
    """Preserve V1's departure optimum and exact destination-vector tie-break.

    Args:
        model: Feasible model with proven minimum departures.
        optimum: Exact minimum-departure value.

    Returns:
        True only when all destinations are fixed canonically.
    """
    model.model.add(model.departures == optimum)
    if optimum == 0:
        for group, variables in zip(model.groups, model.variables):
            model.model.add(variables.role == hash_role(group))
        return True
    return model.fix_lexicographic([(v.role, 2) for g, v in zip(model.groups, model.variables)
                                   if len(g.roles) > 1])


def minimize_counts(model: QuotaModel) -> bool:
    """Optimize one bounded integer rank sum after destinations are fixed.

    Args:
        model: Model whose checked witness uses canonical fixed destinations.

    Returns:
        True only for proven rank optimality with a matching independent value.
    """
    if model.witness is None:
        raise ValueError("rank objective requires a checked witness")
    if 3 * len(model.groups) * sum(sum(q) for q in model.required.quotas) >= 2**60:
        raise ValueError("rank objective exceeds declared integer range")
    objective = sum((3 * i + b + 1) * count
                    for i, (variables, role) in enumerate(zip(model.variables, model.witness.roles))
                    for b, count in enumerate(variables.rows[role]))
    status, value = model.optimize(objective)
    if value is not None and (model.witness is None or value != rank_cost(model.witness)):
        raise ValueError("independent rank objective mismatch")
    return status == cp_model.OPTIMAL


def allocate_rank(groups: list[Component], required: Requirements, cpu_seconds: float = 90,
                  wall_seconds: float = 120, max_calls: int = 128) -> RankResult:
    """Evaluate the versioned count rule without changing feasibility or earlier objectives.

    Args:
        groups: Synthetic or audited count components, not selected source rows.
        required: Frozen substantive constraints.
        cpu_seconds: Whole allocator CPU allowance.
        wall_seconds: Whole allocator wall allowance.
        max_calls: Resource guard; unfinished objectives remain incomplete.

    Returns:
        Separate feasibility/optimality flags and a checked witness if available.
    """
    if cpu_seconds < 0 or wall_seconds < 0 or max_calls < 0:
        raise ValueError("invalid solver limits")
    model = QuotaModel(groups, required, cpu_seconds, wall_seconds, max_calls)
    validate_problem(groups, required)
    feasibility, minimum, destinations, optimal = "unknown", False, False, False
    reason = "no checked witness before solver limit"
    try:
        bound = capacity_bound(groups, required)
        if bound is not None:
            feasibility, reason = "infeasible", "independent upper bound: " + bound
        else:
            model.build()
            status, value = model.optimize(model.departures)
            if status == cp_model.INFEASIBLE:
                reference = solve(groups, required, cpu_seconds=max(0, model.remaining()))
                if reference.status == "infeasible":
                    feasibility, reason = "infeasible", "CP-SAT and independent exhaustive proof"
                else:
                    reason = "CP-SAT infeasible; independent proof unavailable"
            elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                minimum = status == cp_model.OPTIMAL
                destinations = minimum and value is not None and fix_destinations(model, value)
                optimal = destinations and minimize_counts(model)
                reason = "all V2 objectives proven" if optimal else "checked feasible; objectives incomplete"
    except TimeoutError as error:
        reason = str(error)
    except MemoryError:
        reason = "memory limit; objectives incomplete"
    if model.witness is not None:
        feasibility = "feasible"
    return RankResult(feasibility, minimum, destinations, optimal,
                      rank_cost(model.witness) if model.witness is not None else None,
                      reason, model.last_status, model.calls, model.witness,
                      time.process_time() - model.cpu_start, time.monotonic() - model.wall_start)
