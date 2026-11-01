"""A versioned data specification does not establish dataset isolation.

This CPU preparation binds the new eligibility rule to retained source files
and runtime token IDs, then records conservative quarantine decisions. It
cannot authorize GPU work or label an unresolved corpus a training manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet

from group_capacity import Task, audit_groups


# Frozen specifications and previously validated runtime-tokenizer ledger.
PARENT_SHA256 = "6ba7fc6606525bb142de110c7a7cd78cbe26d01a6f1edb8f03646871095d4bcd"
SPEC_SHA256 = "f0a9db0ecb1bc292001c80dbe6cb75d0b7a17174533bfa00d05279e92609ec91"
LEDGER_SHA256 = "c6c16a3e2ed778bb2564959ad4797adb69097d3ec37fa3a34fc2c4e1bab5d721"
DECISION_SHA256 = "a93b19567d1adc234f86fcfa259ba9997341c420f41a6cfc868a583fddb712b0"


def file_hash(path: Path) -> str:
    """Hash an existing evidence file without modifying it.

    Args:
        path: Artifact to identify.

    Returns:
        SHA256 hex digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(path: Path, expected: str) -> None:
    """Reject changed evidence before interpreting its content.

    Args:
        path: Retained artifact.
        expected: Frozen digest.

    Raises:
        ValueError: Artifact contents differ from their frozen identity.
    """
    if file_hash(path) != expected:
        raise ValueError(f"artifact hash differs: {path}")


def read_tasks(sources: Path, ledger_path: Path, decision_path: Path) -> list[Task]:
    """Bind inert reference programs to the original validated prompt census.

    Args:
        sources: Retained source directory containing MBPP and Dolly payloads.
        ledger_path: Immutable runtime prompt/token ledger.
        decision_path: Original admission decision with source payload hashes.

    Returns:
        All official MBPP rows with their unchanged split and prompt length.

    Raises:
        ValueError: Source, partition or tokenizer-ledger identity differs.
    """
    require_hash(ledger_path, LEDGER_SHA256)
    require_hash(decision_path, DECISION_SHA256)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    by_id = {row["task_id"]: row for row in ledger}
    if len(by_id) != 964 or len(ledger) != 964:
        raise ValueError("runtime ledger must contain exactly 964 unique tasks")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    payloads = {row["filename"]: row["sha256"] for row in decision["source_payloads"]}
    tasks: list[Task] = []
    for filename, split, first, last in [
        ("train", "train", 601, 974), ("validation", "calibration", 511, 600),
        ("test", "held_out", 11, 510),
    ]:
        source = sources / "mbpp" / f"{filename}.parquet"
        require_hash(source, payloads[f"full/{filename}-00000-of-00001.parquet"])
        table = pyarrow.parquet.read_table(source, columns=["task_id", "text", "code"])
        rows = table.to_pylist()
        if len(rows) != last - first + 1 or {row["task_id"] for row in rows} != set(range(first, last + 1)):
            raise ValueError("official MBPP partition differs")
        for row in rows:
            previous = by_id[row["task_id"]]
            if previous["split"] != split or previous["source_text"] != row["text"]:
                raise ValueError("source text/split differs from runtime census")
            if not isinstance(row["code"], str):
                raise ValueError("reference program must be inert source text")
            tasks.append(Task(row["task_id"], split, row["text"], row["code"],
                              len(previous["prompt_token_ids"])))
    return tasks


def audit_dolly(source: Path) -> dict[str, object]:
    """Record unresolved context provenance without inventing document IDs.

    No source document map has been reviewed for this payload. Context hashes
    can detect identical passages, but cannot establish document isolation.

    Args:
        source: Original Dolly JSONL, already checked against its pinned hash.

    Returns:
        Content-free per-row disposition and stratum provenance counts.

    Raises:
        ValueError: Source fields have unexpected types.
    """
    strata = {
        "creative_writing": "dolly_prose_open_qa", "open_qa": "dolly_prose_open_qa",
        "closed_qa": "dolly_grounded_qa",
        "information_extraction": "dolly_extraction_classification",
        "classification": "dolly_extraction_classification",
    }
    rows: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    with source.open(encoding="utf-8") as stream:
        for line, content in enumerate(stream, start=1):
            row = json.loads(content)
            if not all(isinstance(row.get(key), str) for key in ("instruction", "context", "category", "response")):
                raise ValueError(f"invalid Dolly fields at line {line}")
            stratum = strata.get(row["category"])
            reasons: list[str] = []
            if stratum is None:
                reasons.append("category_outside_specification")
            elif row["context"]:
                reasons.append("unresolved_upstream_document")
            if stratum is not None:
                count = counts.setdefault(stratum, {"source_rows": 0, "unresolved_context_rows": 0})
                count["source_rows"] += 1
                count["unresolved_context_rows"] += bool(row["context"])
            rows.append({"source_line": line, "source_record_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                         "category": row["category"], "stratum": stratum,
                         "context_sha256": hashlib.sha256(row["context"].encode("utf-8")).hexdigest(),
                         "canonical_upstream_document": None, "manual_reviewer": None,
                         "quarantine_reasons": reasons, "is_selected": False})
    return {"rows": rows, "strata": counts,
            "has_reviewed_document_map": False, "is_document_isolation_admitted": False}


def write_new(path: Path, record: dict[str, object]) -> None:
    """Preserve earlier evidence by refusing to replace an output artifact.

    Args:
        path: New artifact destination.
        record: JSON-serializable evidence.

    Raises:
        FileExistsError: The destination already exists.
    """
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")


def prepare(spec_path: Path, sources: Path, ledger: Path, decision: Path,
            output: Path, candidates: Path) -> None:
    """Write necessary-gate evidence, without selecting unreviewed training rows.

    Args:
        spec_path: Frozen v2 specification.
        sources: Original retained source directory.
        ledger: Original runtime-tokenized MBPP ledger.
        decision: Original source acquisition and failed admission record.
        output: New content-free summary report.
        candidates: New local per-row quarantine manifest, not a training split.

    Raises:
        ValueError: Evidence is changed or output paths are not distinct/new.
    """
    if output.resolve() == candidates.resolve() or output.exists() or candidates.exists():
        raise ValueError("output paths must be distinct and new")
    require_hash(spec_path, SPEC_SHA256)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    require_hash(Path(spec["parent_specification"]["path"]), PARENT_SHA256)
    require_hash(decision, DECISION_SHA256)
    original = json.loads(decision.read_text(encoding="utf-8"))
    for payload in original["source_payloads"]:
        if payload["repository"] == "databricks/databricks-dolly-15k":
            require_hash(sources / "dolly" / payload["filename"], payload["sha256"])
    tasks = read_tasks(sources, ledger, decision)
    groups = audit_groups(tasks, spec["selection"]["prompt_token_buckets_by_stratum"]["mbpp_code"])
    dolly = audit_dolly(sources / "dolly" / "databricks-dolly-15k.jsonl")
    candidate_manifest = {"kind": "unselected_cpu_candidate_quarantine_manifest", "schema_version": 2,
                          "spec_sha256": SPEC_SHA256, "is_realized_training_manifest": False,
                          "is_r1_admitted": False, "mbpp": groups, "dolly": dolly}
    write_new(candidates, candidate_manifest)
    quotas = next(row for row in spec["selection"]["strata"] if row["name"] == "mbpp_code")
    report = {
        "kind": "v2_cpu_manifest_preparation_checkpoint", "schema_version": 2,
        "spec_sha256": SPEC_SHA256, "parent_spec_sha256": PARENT_SHA256,
        "original_token_ledger_sha256": LEDGER_SHA256,
        "source_identity_record_sha256": file_hash(decision),
        "source_payloads": original["source_payloads"],
        "candidate_manifest_path": str(candidates), "candidate_manifest_sha256": file_hash(candidates),
        "status": "blocked_before_realized_training_manifest",
        "is_r1_admitted": False, "has_realized_training_manifest": False,
        "additional_gpu_hours": 0, "selected_training_rows": 0,
        "mbpp_length_eligible_upper_bounds": {
            split: sum(task.split == split and 1 <= task.prompt_tokens <= 447 and bool(task.text.strip()) for task in tasks)
            for split in ("train", "calibration", "held_out")},
        "mbpp_residual_row_upper_bounds": groups["residual_row_upper_bounds"],
        "mbpp_required_quotas": {split: quotas[split] for split in ("train", "calibration", "held_out")},
        "dolly_provenance": dolly["strata"],
        "mandatory_unfinished_checks": ["Dolly upstream document resolution", "normalized shared32-token audit",
                                       "manual paraphrase/template and normalized-AST review", "expanded benchmark exclusions",
                                       "Dolly runtime tokenization and group allocation", "realized manifest isolation",
                                       "combined source/license review"],
        "next_action": "Review failures before further admission; never substitute sources or relax isolation.",
    }
    write_new(output, report)


def main() -> None:
    """Run the explicitly supplied local CPU-only preparation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "sources", "ledger", "decision", "output", "candidates"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.spec, args.sources, args.ledger, args.decision, args.output, args.candidates)
    raise SystemExit("Preparation recorded; no realized training manifest is admitted")


if __name__ == "__main__":
    main()
