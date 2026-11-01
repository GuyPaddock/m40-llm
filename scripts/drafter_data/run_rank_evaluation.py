"""Measure the amended count objective on hash-bound V1 synthetic fixtures.

Each invocation is isolated and bounded. Historical comparisons identify
incomplete controls; repeatability requires two fully optimal fresh witnesses.
"""

import argparse
import hashlib
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

from cp_sat_allocation import allocate
from prepare_v2 import file_hash
from rank_allocation import allocate_rank
from rank_audit import audit_witness, read_problem, read_witness


def evaluate(case: str, output: Path, is_original: bool, first: Path | None) -> dict[str, object]:
    """Measure one frozen case and check persisted witnesses independently.

    Args:
        case: Original synthetic case name.
        output: Fresh artifact directory.
        is_original: Whether to rerun the sole contemporaneous V1 small control.
        first: Completed V2 result directory authorizing one reversed-input repeat.

    Returns:
        Resource, objective, concentration and reproducibility evidence.
    """
    baseline = Path("local-artifacts/2026-10-31-v3-engineering") / case
    historical = next(row for row in json.loads(Path(
        "docs/experiments/2026-10-31-v3-engineering-results.json").read_text())["probes"]
                      if row["kind"] == "solver" and row["case"] == case)
    identity = historical["measurement"]["independent_audit"]
    problem = baseline / "problem.json"
    if (file_hash(problem) != identity["problem_sha256"]
            or file_hash(baseline / "allocation.json") != identity["result_sha256"]):
        raise ValueError("historical synthetic identity mismatch")
    groups, required = read_problem(problem)
    first_report = None
    if first is not None:
        first_report = json.loads((first / "allocation.json").read_text())
        if first_report["problem_sha256"] != file_hash(problem) or not first_report["result"]["is_rank_optimal"]:
            raise ValueError("repeat requires a fully optimal V2 result for this problem")
        groups.reverse()
    result = (allocate(groups, required, cpu_seconds=90, wall_seconds=120, max_calls=128) if is_original
              else allocate_rank(groups, required, cpu_seconds=90, wall_seconds=120, max_calls=128))
    result_path = output / "allocation.json"
    with result_path.open("x") as stream:
        json.dump({"version": "v1" if is_original else "quota-aware-rank-v2",
                   "problem_sha256": file_hash(problem), "result": asdict(result)}, stream, separators=(",", ":"))
    witness = read_witness(result_path)
    audit = audit_witness(groups, required, witness) if witness is not None else None
    original = read_witness(baseline / "allocation.json")
    original_audit = audit_witness(groups, required, original) if original is not None else None
    overlap = None
    if witness is not None and original is not None:
        current_ids = {name for name, counts in zip(witness.component_ids, witness.counts) if sum(counts)}
        prior_ids = {name for name, counts in zip(original.component_ids, original.counts) if sum(counts)}
        overlap = {"current": len(current_ids), "prior": len(prior_ids),
                   "intersection": len(current_ids & prior_ids), "union": len(current_ids | prior_ids)}
    reproducible = None
    if first_report is not None:
        reproducible = result.is_rank_optimal and witness == read_witness(first / "allocation.json")
    digest = (hashlib.sha256(json.dumps(asdict(witness), sort_keys=True).encode()).hexdigest()
              if witness is not None else None)
    return {"case": case, "version": "v1" if is_original else "quota-aware-rank-v2",
            "input_order": "reverse" if first is not None else "forward", "problem_sha256": file_hash(problem),
            "result_sha256": file_hash(result_path), "witness_sha256": digest,
            "result": {key: value for key, value in asdict(result).items() if key != "witness"},
            "independent_audit": audit, "historical_v1_audit": original_audit,
            "historical_v1_result": historical["measurement"]["result"],
            "selected_group_overlap_with_v1": overlap, "is_reproducible": reproducible,
            "is_source_selection": False, "has_human_review": False}


def main() -> None:
    """Run a single CPU probe without changing caps after observing outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("fixed72", "small", "medium", "large", "upper"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-control", action="store_true")
    parser.add_argument("--repeat-of", type=Path)
    args = parser.parse_args()
    if args.original_control and (args.case != "small" or args.repeat_of is not None):
        raise ValueError("only the frozen forward V1 small control is authorized")
    args.output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpu, wall = time.process_time(), time.monotonic()
    report = evaluate(args.case, args.output, args.original_control, args.repeat_of)
    report.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("case", "version", "input_order", "result",
                                                   "cpu_seconds", "wall_seconds", "peak_rss_kib", "is_reproducible")}))


if __name__ == "__main__":
    main()
