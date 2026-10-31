"""CP-SAT evaluates the frozen whole-component objective without greedy repair.

Feasibility, minimum departures and canonical tie-breaking have separate
statuses. Count witnesses are checked by the independent reference checker;
no source-row manifest is produced. Exact lexicographic blocks use integer
mixed-radix weights, not epsilon penalties or giant floating-point objectives.
"""

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from quota_preflight import (Component, Requirements, Witness, capacity_bound,
                            check_witness, group_hash, hash_role, solve, validate_problem)


@dataclass
class Variables:
    """One component's exact assignment and row counts.

    Attributes:
        role: Integer destination expression.
        rows: Role-keyed bucket variables; impossible/zero-capacity slots are zero.
    """

    role: cp_model.LinearExpr | int
    rows: dict[int, tuple[cp_model.IntVar | int, ...]]


@dataclass(frozen=True)
class AllocationResult:
    """Checked feasibility is distinct from completion of the canonical objective.

    Attributes:
        feasibility: feasible, infeasible or unknown, with proof limitations below.
        is_minimum_departure: Whether the first objective was proven optimal.
        is_canonical: Whether every frozen lexicographic objective was completed.
        reason: Stop/proof reason.
        solver_status: Last native CP-SAT status, never silently promoted to proof.
        solve_calls: Number of optimizer calls, including interrupted calls.
        witness: Independently checked count-only assignment, if available.
        cpu_seconds: Whole invocation CPU time, including construction/checks.
        wall_seconds: Whole invocation wall time.
    """

    feasibility: str
    is_minimum_departure: bool
    is_canonical: bool
    reason: str
    solver_status: str
    solve_calls: int
    witness: Witness | None
    cpu_seconds: float
    wall_seconds: float


class QuotaModel:
    """One fixed integer formulation with deadline-aware lexicographic evaluation.

    Attributes:
        groups: Canonical seed-ordered count-only components.
        required: Frozen family quotas and global coverage floors.
        model: Owned CP-SAT model.
        variables: Component variables in matching order.
        departures: Exact number of non-hash assignments.
        cpu_start: CPU clock before model construction.
        wall_start: Wall clock before model construction.
        cpu_limit: Entire invocation CPU seconds.
        wall_limit: Entire invocation wall seconds.
        call_limit: Safety cap; unfinished canonical work remains unknown.
        calls: Actual optimizer invocations.
        last_status: Native status from the last invocation.
        witness: Last independently verified feasible assignment.
    """

    def __init__(self, groups: list[Component], required: Requirements,
                 cpu_seconds: float, wall_seconds: float, max_calls: int) -> None:
        """Construct the same y/x/z constraints as the exhaustive reference.

        Args:
            groups: Count-only synthetic or audited components.
            required: Explicit row, contributing-group and bucket constraints.
            cpu_seconds: Aggregate CPU deadline including model construction.
            wall_seconds: Wall deadline including model construction.
            max_calls: Bounded evaluation count, not a changed objective.
        """
        self.cpu_start, self.wall_start = time.process_time(), time.monotonic()
        self.cpu_limit, self.wall_limit, self.call_limit = cpu_seconds, wall_seconds, max_calls
        self.calls = 0
        self.last_status = "NOT_RUN"
        self.witness: Witness | None = None
        self.groups = sorted(groups, key=lambda g: (group_hash(g.component_id), g.component_id))
        self.required = required
        self.model = cp_model.CpModel()
        self.variables: list[Variables] = []
        self.departures: cp_model.LinearExpr | int = 0

    def remaining(self) -> float:
        """Return the smaller remaining whole-invocation CPU/wall allowance."""
        return min(self.cpu_limit - (time.process_time() - self.cpu_start),
                   self.wall_limit - (time.monotonic() - self.wall_start))

    def build(self) -> None:
        """Populate role, exact-quota, positive-contribution and bucket constraints."""
        family_rows: dict[tuple[int, int], list[cp_model.LinearExpr | int]] = {}
        held_groups: dict[int, list[cp_model.IntVar]] = {}
        buckets: list[list[cp_model.IntVar | int]] = [[], [], []]
        for number, group in enumerate(self.groups):
            if self.remaining() <= 0:
                raise TimeoutError("model construction deadline")
            assigned = {role: self.model.new_bool_var(f"y{number}_{role}") for role in group.roles}
            self.model.add_exactly_one(assigned.values())
            rows: dict[int, tuple[cp_model.IntVar | int, ...]] = {}
            for role in group.roles:
                limits = [min(n, self.required.quotas[group.family][role]) for n in group.buckets]
                values = tuple(self.model.new_int_var(0, limit, f"x{number}_{role}_{b}") if limit else 0
                               for b, limit in enumerate(limits))
                for variable, limit in zip(values, limits):
                    if limit:
                        self.model.add(variable <= limit * assigned[role])
                rows[role] = values
                family_rows.setdefault((group.family, role), []).append(sum(values))
                if role == 2:
                    contributes = self.model.new_bool_var(f"z{number}")
                    self.model.add(sum(values) >= contributes)
                    self.model.add(sum(values) <= self.required.quotas[group.family][2] * contributes)
                    held_groups.setdefault(group.family, []).append(contributes)
                    for b, variable in enumerate(values):
                        buckets[b].append(variable)
            self.variables.append(Variables(sum(role * assigned[role] for role in group.roles), rows))
            self.departures += sum(assigned[role] for role in group.roles if role != hash_role(group))
        for family, quotas in enumerate(self.required.quotas):
            for role, count in enumerate(quotas):
                self.model.add(sum(family_rows.get((family, role), [])) == count)
            self.model.add(sum(held_groups.get(family, [])) >= self.required.heldout_groups[family])
        for b, minimum in enumerate(self.required.heldout_buckets):
            self.model.add(sum(buckets[b]) >= minimum)

    def optimize(self, objective: cp_model.LinearExpr | int) -> tuple[int, int | None]:
        """Optimize one exact objective and independently check every feasible result.

        Args:
            objective: Integer expression with bounded coefficients.

        Returns:
            Native status and exact integer optimum/value if a witness exists.

        Raises:
            TimeoutError: Whole-invocation budget or call cap is exhausted.
            ValueError: Solver model or returned witness violates the contract.
        """
        if self.remaining() <= 0 or self.calls >= self.call_limit:
            raise TimeoutError("canonical solve deadline/call cap")
        self.model.minimize(objective)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 20261029
        solver.parameters.max_time_in_seconds = max(0.001, self.remaining())
        solver.parameters.max_memory_in_mb = 3072
        self.calls += 1
        status = solver.solve(self.model)
        self.last_status = solver.status_name(status)
        if status == cp_model.MODEL_INVALID:
            raise ValueError("CP-SAT model invalid: " + solver.solution_info())
        value = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            roles = tuple(int(solver.value(variables.role)) for variables in self.variables)
            counts = tuple(tuple(int(solver.value(x)) for x in variables.rows[role])
                           for variables, role in zip(self.variables, roles))
            witness = Witness(tuple(g.component_id for g in self.groups), roles, counts,
                              sum(role != hash_role(g) for role, g in zip(roles, self.groups)))
            if not check_witness(self.groups, self.required, witness):
                raise ValueError("independent witness verification failed")
            self.witness = witness
            value = int(solver.value(objective))
        elif status == cp_model.INFEASIBLE and self.witness is not None:
            raise ValueError("canonical restriction invalidated an existing witness")
        return status, value

    def fix_lexicographic(self, terms: list[tuple[cp_model.LinearExpr | int, int]]) -> bool:
        """Optimize bounded mixed-radix chunks in exact lexicographic order.

        Args:
            terms: Ordered nonnegative expressions and valid inclusive upper bounds.

        Returns:
            True only if every chunk is proven optimal and fixed.
        """
        offset = 0
        while offset < len(terms):
            if self.remaining() <= 0:
                raise TimeoutError("canonical objective construction deadline")
            end, radix = offset, 1
            while end < len(terms) and radix * (terms[end][1] + 1) <= 2**50:
                radix *= terms[end][1] + 1
                end += 1
            if end == offset:
                raise ValueError("lexicographic digit exceeds fixed integer limit")
            expression: cp_model.LinearExpr | int = 0
            for variable, maximum in terms[offset:end]:
                expression = expression * (maximum + 1) + variable
            status, optimum = self.optimize(expression)
            if status != cp_model.OPTIMAL:
                return False
            self.model.add(expression == optimum)
            offset = end
        return True

    def canonicalize(self, optimum: int) -> bool:
        """Fix the optimal departure count, destinations, then flattened bucket counts.

        Args:
            optimum: Proven minimum departure objective.

        Returns:
            Whether the full frozen canonical objective completed.
        """
        self.model.add(self.departures == optimum)
        if optimum == 0:
            # A zero sum of departure indicators forces every original hash role.
            for group, variables in zip(self.groups, self.variables):
                self.model.add(variables.role == hash_role(group))
        elif not self.fix_lexicographic([(v.role, 2) for g, v in zip(self.groups, self.variables) if len(g.roles) > 1]):
            return False
        if self.witness is None:
            raise ValueError("canonicalization requires a checked witness")
        terms = []
        for group, variables, role in zip(self.groups, self.variables, self.witness.roles):
            for capacity, variable in zip(group.buckets, variables.rows[role]):
                maximum = min(capacity, self.required.quotas[group.family][role])
                if maximum:
                    terms.append((variable, maximum))
        return self.fix_lexicographic(terms)


def allocate(groups: list[Component], required: Requirements, cpu_seconds: float = 60,
             wall_seconds: float = 90, max_calls: int = 128) -> AllocationResult:
    """Evaluate one fixed solver while separating checked feasibility and canonicality.

    Args:
        groups: Synthetic/audited components, never raw source records.
        required: Frozen count constraints.
        cpu_seconds: Total CPU limit, not a per-lexicographic-call allowance.
        wall_seconds: Total wall limit.
        max_calls: Bounded optimizer invocations; no incomplete canonical pass.

    Returns:
        Checked witness and explicit objective/proof completion status.
    """
    if cpu_seconds < 0 or wall_seconds < 0 or max_calls < 0:
        raise ValueError("invalid solver limits")
    model = QuotaModel(groups, required, cpu_seconds, wall_seconds, max_calls)
    validate_problem(groups, required)
    feasibility, minimum, canonical = "unknown", False, False
    reason = "not run"
    bound = capacity_bound(groups, required)
    try:
        if bound is not None:
            feasibility, reason = "infeasible", "independent upper bound: " + bound
        else:
            model.build()
            status, optimum = model.optimize(model.departures)
            if status == cp_model.INFEASIBLE:
                reference = solve(groups, required, cpu_seconds=max(0, model.remaining()), max_states=200000)
                if reference.status == "infeasible":
                    feasibility, reason = "infeasible", "CP-SAT and independent exhaustive proof"
                else:
                    reason = "CP-SAT infeasible; independent proof unavailable"
            elif status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                feasibility = "feasible"
                minimum = status == cp_model.OPTIMAL
                canonical = minimum and optimum is not None and model.canonicalize(optimum)
                reason = "canonical optimum verified" if canonical else "checked feasible; canonical objective incomplete"
            else:
                reason = "no checked witness before solver limit"
    except TimeoutError as error:
        reason = str(error)
    except MemoryError:
        reason = "memory limit; canonical objective incomplete"
    if model.witness is not None:
        feasibility = "feasible"
    return AllocationResult(feasibility, minimum, canonical, reason, model.last_status,
                            model.calls, model.witness, time.process_time() - model.cpu_start,
                            time.monotonic() - model.wall_start)
