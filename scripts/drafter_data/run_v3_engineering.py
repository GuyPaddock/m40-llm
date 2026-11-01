"""Run one frozen CPU engineering probe per process for independent resource peaks.

All populations are retained samples or deterministic synthetic stress inputs.
No network, source selection, GPU initialization or model inference is used.
"""

import argparse
import json
import platform
import resource
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import ortools

from audit_allocation_evidence import audit_evidence
from census_matching import IncompleteAudit
from cp_sat_allocation import allocate
from frequency_matching import FrequencyIndex
from prepare_v2 import file_hash
from quota_preflight import Component, Requirements
from run_v3_preflight import retained_inputs, synthetic_rows, verify_retained_matches


def synthetic_components(case: str) -> list[Component]:
    """Construct fixed count/bucket envelopes without inferring corpus eligibility.

    Args:
        case: fixed72, small, medium, large or upper.

    Returns:
        Count-only components; upper uses one row per native-record upper bound.
    """
    groups = []
    if case == "fixed72":
        for family in range(4):
            groups.extend([Component(f"f{family}-train", family, (0,), (256, 0, 0)),
                           Component(f"f{family}-cal", family, (1,), (0, 48, 0))])
            groups.extend(Component(f"f{family}-held-{i}", family, (2,), (2, 2, 2)) for i in range(16))
        return groups
    populations = {"small": (2048, 512, 128, 512), "medium": (8192, 2048, 512, 1024),
                   "large": (65536, 8192, 1024, 5051), "upper": (230975, 56025, 98169, 5051)}
    limits = populations[case]
    boundaries = ((65, 84), (9, 53), (7, 83), (0, 58))
    for family, count in enumerate(limits):
        for i in range(count):
            marker = i % 100
            low, middle = boundaries[family]
            bucket = 0 if marker < low else 1 if marker < middle else 2
            capacity = 1 if case == "upper" or family in (0, 3) else 1 + i % 3 if family == 1 else 16
            buckets = tuple(capacity if b == bucket else 0 for b in range(3))
            if family in (0, 2):
                # Cycle independently of bucket marker to avoid artificial correlation.
                roles = (2,) if (i // 100) % 10 == 0 else (0, 1)
            elif family == 1:
                roles = (0, 1, 2)
            else:
                roles = (0,) if i < count * 3053 // 5051 else (1,) if i < count * 4051 // 5051 else (2,)
            groups.append(Component(f"synthetic-f{family}-{i:06d}", family, roles, buckets))
    return groups


def measure_matching(case: str, output: Path, root: Path) -> dict[str, object]:
    """Measure complete frequency preparation and matching, preserving unknown stops.

    Args:
        case: Frozen retained/header/copied/population case.
        output: Fresh artifact directory.
        root: Repository containing retained raw evidence.

    Returns:
        Actual costs and independent retained oracle comparison when applicable.
    """
    retained = retained_inputs(root) if case == "retained" else []
    shape = {"header2k": (2000, 256), "header10k": (10000, 256),
             "copied200": (200, 64), "population": (390220, 64)}
    if case == "retained":
        count, width = len(retained), None
        rows = ((text, tokens) for _, text, tokens in retained)
    elif case == "copied200":
        count, width = shape[case]
        rows = ((f"row {i} " + " ".join(f"copied{j}" for j in range(64)), tuple(range(64))) for i in range(count))
    else:
        count, width = shape[case]
        rows = synthetic_rows(count, width, has_common_header=True)
    index = FrequencyIndex(output / "matches.sqlite", cpu_seconds=600, wall_seconds=900)
    start = time.process_time()
    staged = 0
    status, reason = "complete", None
    try:
        for i, (text, tokens) in enumerate(rows):
            index.add(i, text, tokens)
            staged += 1
        staging_seconds = time.process_time() - start
        index.finish()
    except (IncompleteAudit, sqlite3.OperationalError, MemoryError) as error:
        status, reason = "unknown", str(error)
        staging_seconds = None
    finally:
        index.close()
    connection = sqlite3.connect(output / "matches.sqlite")
    try:
        record_count = connection.execute("SELECT count(*) FROM records").fetchone()[0]
        matches = connection.execute("SELECT count(*) FROM matches").fetchone()[0]
        frequency_keys = connection.execute("SELECT count(*) FROM frequencies").fetchone()[0]
    finally:
        connection.close()
    result: dict[str, object] = {"status": status, "reason": reason, "intended_records": count,
        "staged_records": staged, "persisted_records": record_count, "synthetic_width": width,
        "staging_cpu_seconds": staging_seconds, "index_cpu_seconds": time.process_time() - start,
        "candidate_checks": index.candidate_checks, "raw_matching_pairs": matches,
        "postings": index.posting_count, "frequency_keys": frequency_keys,
        "database_bytes": (output / "matches.sqlite").stat().st_size,
        "database_sha256": file_hash(output / "matches.sqlite")}
    if status == "complete" and case == "retained":
        result["independent_oracle"] = verify_retained_matches(output / "matches.sqlite", retained)
    if status == "complete" and case == "copied200" and matches != count * (count - 1) // 2:
        raise ValueError("common-passage matches lost")
    return result


def measure_solver(case: str, output: Path) -> dict[str, object]:
    """Save and independently audit one frozen synthetic allocation problem.

    Args:
        case: Synthetic component envelope, never a real data selection.
        output: Fresh artifact directory.

    Returns:
        Compact status summary; full synthetic witness remains in local evidence.
    """
    groups = synthetic_components(case)
    required = Requirements(((256, 48, 96),) * 4, (16,) * 4, (32, 32, 32))
    problem = {"kind": "synthetic_count_problem", "case": case,
               "groups": [asdict(g) for g in groups], "requirements": asdict(required)}
    path = output / "problem.json"
    with path.open("x") as stream:
        json.dump(problem, stream, separators=(",", ":"))
    del problem
    result = allocate(groups, required, cpu_seconds=90, wall_seconds=120, max_calls=128)
    with (output / "allocation.json").open("x") as stream:
        json.dump({"problem_sha256": file_hash(path), "result": asdict(result)}, stream, separators=(",", ":"))
    audit = audit_evidence(path, output / "allocation.json")
    return {"components": len(groups), "result": {key: value for key, value in asdict(result).items() if key != "witness"},
            "has_checked_witness": result.witness is not None,
            "departures": result.witness.departures if result.witness is not None else None,
            "independent_audit": audit}


def main() -> None:
    """Execute one bounded probe and emit a new, explicitly non-admitting report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("matching", "solver"))
    parser.add_argument("case")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case not in ({"retained", "header2k", "header10k", "copied200", "population"}
                         if args.kind == "matching" else {"fixed72", "small", "medium", "large", "upper"}):
        raise ValueError("unfrozen probe case")
    args.output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (720, 720))
    cpu, wall = time.process_time(), time.monotonic()
    result = (measure_matching(args.case, args.output, Path.cwd()) if args.kind == "matching"
              else measure_solver(args.case, args.output))
    report = {"kind": args.kind, "case": args.case, "python": platform.python_version(),
              "ortools": ortools.__version__, "sqlite": sqlite3.sqlite_version,
              "is_census_admitted": False, "gpu_hours": 0, "has_acquisition": False,
              "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - wall,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "measurement": result}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
