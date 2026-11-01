"""Measure automated local review preparation, never simulated human review time.

Every exported packet remains blocked. Existing local evidence can support
literal checks and dependency inventories, not independent human adjudication.
"""

import argparse
import json
import resource
import signal
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import TypeVar

from mbpp_remaining import scan_exclusions
from prepare_v2 import file_hash
from review_dependencies import build_dependencies, read_raw_pairs
from review_records import ReviewRecord, freeze_manifest, read_raw_records, read_records, select_review_records

# Preserve each timed operation's concrete return type.
T = TypeVar("T")


def measure_stage(name: str, operation: Callable[[], T], timings: list[dict[str, object]]) -> T:
    """Record automated effort even when a stage stops without completion.

    Args:
        name: Explicit machine-work stage name, not a human-review label.
        operation: Bounded local operation.
        timings: Output ledger to append one completed/unknown record.

    Returns:
        The operation's result; exceptions propagate after timing is retained.
    """
    cpu, wall, complete = time.process_time(), time.monotonic(), False
    try:
        result = operation()
        complete = True
        return result
    finally:
        timings.append({"stage": name, "cpu_seconds": time.process_time() - cpu,
                        "wall_seconds": time.monotonic() - wall,
                        "status": "complete" if complete else "unknown", "actor": "automation"})


def inspect_sources(manifest: dict[str, object], root: Path) -> tuple[list[ReviewRecord], dict[str, dict[str, object]]]:
    """Validate frozen sample identities and inspect retained raw field availability.

    Args:
        manifest: Preregistered local review locators and ledger hashes.
        root: Repository root.

    Returns:
        All retained records and their available raw audit fields.
    """
    for filename, digest in manifest["inputs"].items():
        if file_hash(root / filename) != digest:
            raise ValueError("review input ledger changed")
    records = read_records(root)
    if [r.identity for r in select_review_records(records)] != [r["id"] for r in manifest["rows"]]:
        raise ValueError("fixed review sample differs")
    return records, read_raw_records(records, root)


def inventory_rights(root: Path) -> dict[str, str]:
    """Inventory local rights/provenance evidence without issuing legal clearance.

    Args:
        root: Repository holding the already retained source metadata.

    Returns:
        Relative evidence paths bound to file hashes.
    """
    paths = []
    for directory, sources in (("2026-10-30-hf-feasibility", ("ultrachat", "commitpack")),
                               ("2026-10-30-source-selection", ("squad", "docred"))):
        for source in sources:
            paths.extend(Path("local-artifacts") / directory / f"{source}-{suffix}"
                         for suffix in ("card.md", "api.json"))
    paths.append(Path("local-artifacts/2026-10-30-source-selection/docred-upstream-LICENSE.txt"))
    return {str(path): file_hash(root / path) for path in paths}


def inspect_benchmarks(root: Path, selected: list[ReviewRecord]) -> dict[str, object]:
    """Check packet inputs against the frozen literal inventory, retaining its gaps.

    Args:
        root: Repository with existing benchmark input/token evidence.
        selected: Frozen review sample, not training selection.

    Returns:
        Raw matches and explicit coverage limits, without semantic clearance.
    """
    directory = root / "local-artifacts/2026-10-30-drafter-recovery/exclusions-final"
    inventory = json.loads((directory / "inventory.json").read_text())
    matches = scan_exclusions(directory / "inputs.jsonl", directory / "tokens.jsonl",
                              {i: row.text for i, row in enumerate(selected)},
                              {i: list(row.tokens) for i, row in enumerate(selected)})
    return {"matches": {selected[i].identity: values for i, values in matches.items()},
            "inventory_sha256": file_hash(directory / "inventory.json"),
            "inputs_sha256": file_hash(directory / "inputs.jsonl"),
            "tokens_sha256": file_hash(directory / "tokens.jsonl"),
            "unique_texts": inventory["unique_texts"],
            "missing_local_references": inventory["missing_local_references"],
            "has_semantic_clearance": False, "has_complete_audit_field_coverage": False}


def create_packets(records: list[ReviewRecord], selected: list[ReviewRecord],
                   raw: dict[str, dict[str, object]], graph: dict[str, object],
                   benchmark: dict[str, object]) -> list[dict[str, object]]:
    """Export blocked review packets with all known unselected local dependencies.

    Args:
        records: Full retained sample scope.
        selected: Frozen sixteen review locators.
        raw: Hash-bound available source fields.
        graph: Conservative local transitive dependency evidence.
        benchmark: Literal input screening evidence with coverage limits.

    Returns:
        Metadata-only packets; no source text or dataset answer becomes a label.
    """
    selected_ids = {r.identity for r in selected}
    packets = []
    for row in selected:
        fields = raw[row.identity]
        members = [records[i] for i in graph["membership"][row.identity]]
        blockers = ["independent_human_review_unavailable", "external_dependency_closure_unknown",
                    "semantic_benchmark_review_pending", "rights_chain_unresolved",
                    "audit_only_fields_not_fully_screened", "registry_second_review_pending"]
        if row.source == "ultrachat" and "messages" not in fields:
            blockers.append("first_user_message_consistency_unverified")
        if row.source in ("squad-original", "docred-original"):
            blockers.append("historical_article_identity_unresolved")
        if row.source == "squad-original" and "answers" not in fields:
            blockers.append("source_answers_unavailable")
        if row.source == "docred-original" and "labels" not in fields:
            blockers.append("source_relation_labels_unavailable")
        unresolved_ast = [entry for entry in graph["unresolved_code_fields"] if entry["id"] in {r.identity for r in members}]
        if unresolved_ast:
            blockers.append("dependent_code_template_review_unresolved")
        if row.identity in benchmark["matches"]:
            blockers.append("literal_benchmark_overlap")
        packets.append({"id": row.identity, "source": row.source, "split": row.split,
                        "prompt_tokens": row.prompt_tokens, "available_fields": sorted(fields),
                        "source_file_sha256": row.metadata["sample_sha256"],
                        "license_label": row.metadata.get("license_label"), "status": "blocked",
                        "blockers": blockers, "local_dependency_ids": [r.identity for r in members],
                        "local_dependency_count": len(members),
                        "unselected_dependency_count": sum(r.identity not in selected_ids for r in members),
                        "has_cross_family_dependency": len({r.source for r in members}) > 1,
                        "has_cross_native_split_dependency": len({(r.source, r.split) for r in members}) > 1,
                        "unresolved_code_fields": unresolved_ast,
                        "benchmark_matches": benchmark["matches"].get(row.identity, []),
                        "has_complete_dependency_closure": False})
    return packets


def stop_probe(signum: int, frame: FrameType | None) -> None:
    """Interrupt an exhausted CPU/wall probe without converting it into clearance.

    Args:
        signum: Expired timer signal.
        frame: Interrupted Python frame, unused.
    """
    raise TimeoutError(f"review automation deadline reached: signal {signum}")


def run_probe(root: Path, manifest_path: Path, output: Path) -> None:
    """Run the one authorized local automation measurement, leaving human stages null.

    Args:
        root: Repository root.
        manifest_path: Committed preregistered review-only locators.
        output: Fresh evidence directory.
    """
    output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (660, 660))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGPROF, stop_probe)
    signal.signal(signal.SIGALRM, stop_probe)
    signal.setitimer(signal.ITIMER_PROF, 600)
    signal.setitimer(signal.ITIMER_REAL, 900)
    cpu, wall = time.process_time(), time.monotonic()
    timings: list[dict[str, object]] = []
    report: dict[str, object] = {"status": "unknown", "manifest_sha256": file_hash(manifest_path),
        "human_reviewer_count": 0, "human_review_seconds": None, "reviewer_agreement": None,
        "human_adjudication_seconds": None, "independent_second_review_seconds": None,
        "human_stages_status": "not performed; reviewers unavailable",
        "is_census_admitted": False, "additional_gpu_hours": 0, "timings": timings}
    try:
        manifest = json.loads(manifest_path.read_text())
        records, raw = measure_stage("source_identity_and_field_inventory", lambda: inspect_sources(manifest, root), timings)
        selected = select_review_records(records)
        pairs = measure_stage("raw_overlap_identity_replay", lambda: read_raw_pairs(root, records), timings)
        graph = measure_stage("local_dependency_graph", lambda: build_dependencies(records, raw, pairs), timings)
        rights = measure_stage("rights_evidence_inventory", lambda: inventory_rights(root), timings)
        benchmark = measure_stage("literal_benchmark_input_screen", lambda: inspect_benchmarks(root, selected), timings)
        packets = measure_stage("dependency_closure_and_blocked_packets",
                                lambda: create_packets(records, selected, raw, graph, benchmark), timings)
        report.update(status="automation_complete_human_review_unavailable", retained_records=len(records),
                      review_records=len(selected), local_components=graph["component_count"],
                      dependency_edges=graph["edges"], rights_evidence=rights, benchmark=benchmark, packets=packets,
                      unresolved_code_fields=graph["unresolved_code_fields"],
                      blocked_by_reason=dict(Counter(reason for packet in packets for reason in packet["blockers"])))
    except (TimeoutError, MemoryError) as error:
        report["stop_reason"] = str(error)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.setitimer(signal.ITIMER_REAL, 0)
    report.update(cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "cpu_seconds", "wall_seconds", "peak_rss_kib", "timings")}))


def main() -> None:
    """Freeze locators or run the separately committed bounded measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.output is not None:
            raise ValueError("prepare only writes the new review manifest")
        freeze_manifest(Path.cwd(), args.manifest)
    else:
        if args.output is None:
            raise ValueError("measurement needs a fresh output directory")
        run_probe(Path.cwd(), args.manifest, args.output)


if __name__ == "__main__":
    main()
