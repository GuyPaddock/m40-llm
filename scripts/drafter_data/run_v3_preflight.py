"""Measure CPU preflight on retained samples and explicitly synthetic populations.

No network client, source acquisition, source-row selection or model inference
is performed. Reports distinguish complete checks, unknown work and projected
costs; the original registry and all prior evidence stay untouched.
"""

import argparse
import itertools
import json
import platform
import resource
import sqlite3
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from census_matching import CandidateIndex, GroupRecord, IncompleteAudit, group_records, token_spans
from group_capacity import normalize_text, word_shingles
from original_source_inputs import parse_json_prefix
from prepare_v2 import file_hash
from quota_preflight import Component, Requirements, hash_role, solve
from registry_preflight import load_registry, render_input

# Approved immutable registry and source sample roots, never download locations.
REGISTRY_DIGEST = "b8c250e9a15ec5e53c8970094adb5c14b7190f6de10694a636dd3966deda100b"
HF_ROOT = Path("local-artifacts/2026-10-30-hf-feasibility")
ORIGINAL_ROOT = Path("local-artifacts/2026-10-30-source-selection")


def local_registry_rows(root: Path) -> Iterator[tuple[str, dict[str, str]]]:
    """Reconstruct only previously retained complete records from source prefixes.

    Args:
        root: Repository root holding immutable local evidence.

    Yields:
        Registry source name and natural fields without source answer labels.
    """
    for split in ("train", "test"):
        path = root / HF_ROOT / f"ultrachat-{split}.json"
        for row in json.loads(path.read_text())["rows"]:
            record = row["record"]
            if "messages" in record:
                users = [message["content"] for message in record["messages"] if message["role"] == "user"]
                if not users or users[0] != record["prompt"]:
                    raise ValueError("first-user prompt inconsistency")
            yield "ultrachat-first-user-sft", {"prompt": record["prompt"]}
    for line in (root / HF_ROOT / "commitpack-python-prefix.bin").read_bytes().split(b"\n")[:-1][:128]:
        record = json.loads(line)
        yield "commitpackft-python", {key: record[key] for key in ("subject", "old_file", "old_contents")}
    for split in ("train", "validation"):
        for row in json.loads((root / ORIGINAL_ROOT / f"squad-{split}.json").read_text())["rows"]:
            yield "squad-original", {key: row["record"][key] for key in ("context", "question")}
    for split in ("train_annotated", "dev", "test"):
        raw = zlib.decompressobj(31).decompress((root / ORIGINAL_ROOT / f"docred-{split}.gzpart").read_bytes(), 4 * 1024 * 1024)
        for _, record in parse_json_prefix(raw, False):
            sentences = record["sents"]
            if not isinstance(sentences, list):
                raise ValueError("malformed retained document")
            yield "docred-original", {"sentences": "\n".join(" ".join(sentence) for sentence in sentences)}


def retained_inputs(root: Path) -> list[tuple[str, str, tuple[int, ...]]]:
    """Read prior prompts and original-Qwen comparison IDs without retokenizing.

    Args:
        root: Repository with checksum-verified historical evidence.

    Returns:
        Source names, complete raw texts and retained comparison token IDs.
    """
    aliases = {"ultrachat": "ultrachat-first-user-sft", "commitpack": "commitpackft-python",
               "squad-original": "squad-original", "docred-original": "docred-original"}
    rows = []
    for directory in (HF_ROOT, ORIGINAL_ROOT):
        with (root / directory / "sample-tokens.jsonl").open() as stream:
            tokens = {row["id"]: row for row in map(json.loads, stream)}
        with (root / directory / "sample-inputs.jsonl").open() as stream:
            for row in map(json.loads, stream):
                if row["source"] in aliases:
                    rows.append((aliases[row["source"]], row["text"], tuple(tokens[row["id"]]["comparison_ids"])))
    return rows


def registry_report(root: Path, rows: list[tuple[str, str, tuple[int, ...]]]) -> dict[str, object]:
    """Replay raw adapters and prove byte preservation against historical inputs.

    Args:
        root: Repository root.
        rows: Original retained input/token ledger, not an admitted manifest.

    Returns:
        Source counts, verified span counts and explicit second-review blocker.
    """
    path = root / "docs/experiments/2026-10-30-v3-generic-instruction-registry-v1.json"
    registry = load_registry(path, root, REGISTRY_DIGEST)
    counts: Counter[str] = Counter()
    rendered: Counter[tuple[str, str]] = Counter()
    spans = 0
    for source_id, fields in local_registry_rows(root):
        source = registry[source_id]
        result = render_input(source, source["revision"], fields)
        counts[source_id] += 1
        rendered[source_id, result.text] += 1
        spans += len(result.spans)
    expected = Counter((source, text) for source, text, _ in rows)
    if rendered != expected:
        raise ValueError("rendered input differs from frozen original evidence")
    return {"source_counts": dict(counts), "verified_spans": spans,
            "has_identical_prompt_bytes": True, "tokenizer_rerun": False,
            "token_lengths": "unchanged prior evidence; same prompt bytes and tokenizer",
            "registry_sha256": file_hash(path), "user_candidate_approval": True,
            "ultrachat_first_user_consistency": "unverified: retained projection lacks messages",
            "independent_second_reviewer": None, "is_exemption_active": False}


def measure_index(path: Path, rows: Iterable[tuple[str, tuple[int, ...]]],
                  expected_rows: int, max_candidates: int = 1000000) -> dict[str, object]:
    """Measure a fresh bounded raw-matching database without suppressing any key.

    Args:
        path: New local SQLite artifact.
        rows: Retained or explicitly synthetic inputs.
        expected_rows: Full population for this probe only.
        max_candidates: Pair verification ceiling, not a changed matching rule.

    Returns:
        Actual completed work, CPU/wall/RSS/storage and incomplete status.
    """
    cpu, wall = time.process_time(), time.monotonic()
    index = CandidateIndex(path, max_candidates=max_candidates, cpu_seconds=180)
    status, reason, completed = "complete", None, 0
    try:
        for number, (text, tokens) in enumerate(rows):
            index.add(number, text, tokens)
            completed += 1
    except IncompleteAudit as error:
        status, reason = "unknown", str(error)
    finally:
        matches = index.connection.execute("SELECT count(*) FROM matches").fetchone()[0]
        index.close()
    return {"status": status, "reason": reason, "expected_rows": expected_rows,
            "completed_rows": completed, "candidate_checks": index.candidate_checks,
            "postings": index.posting_count, "matches": matches, "database_bytes": path.stat().st_size,
            "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - wall,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def verify_retained_matches(path: Path, rows: list[tuple[str, str, tuple[int, ...]]]) -> dict[str, int]:
    """Independently compare every retained pair against cached exact rule sets.

    Args:
        path: Completed local matching database.
        rows: Retained raw inputs and original token IDs.

    Returns:
        Exhaustive comparison count and verified matching pairs.
    """
    texts = [normalize_text(text) for _, text, _ in rows]
    words = [word_shingles(text) for text in texts]
    spans = [token_spans(tokens) for _, _, tokens in rows]
    expected = {}
    for left, right in itertools.combinations(range(len(rows)), 2):
        reasons = []
        if texts[left] and texts[left] == texts[right]:
            reasons.append("exact_normalized_text")
        if spans[left] & spans[right]:
            reasons.append("shared_32_token_span")
        union = words[left] | words[right]
        if union and 2 * len(words[left] & words[right]) >= len(union):
            reasons.append("five_word_jaccard_at_least_half")
        if reasons:
            expected[left, right] = tuple(reasons)
    connection = sqlite3.connect(path)
    try:
        actual = {(a, b): tuple(json.loads(reasons)) for a, b, reasons in connection.execute("SELECT * FROM matches")}
    finally:
        connection.close()
    if expected != actual:
        raise ValueError("indexed matching differs from retained exhaustive oracle")
    return {"compared_pairs": len(rows) * (len(rows) - 1) // 2, "matching_pairs": len(actual)}


def synthetic_rows(count: int, width: int, has_common_header: bool = False) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Make resource fixtures, not artificial corpus evidence or tokenizer timings.

    Args:
        count: Number of deterministic synthetic records.
        width: Words and token IDs per record, unrelated to Qwen eligibility.
        has_common_header: Whether twenty generic words cause posting contention.

    Yields:
        Text and disjoint synthetic token IDs; no actual model data.
    """
    for record in range(count):
        header = [f"common{i}" for i in range(20)] if has_common_header else []
        words = header + [f"r{record}w{i}" for i in range(width - len(header))]
        yield " ".join(words), tuple(range(record * width, (record + 1) * width))


def scale_grouping(count: int) -> dict[str, object]:
    """Measure graph union at expected record count, without claiming semantic edges.

    Args:
        count: Synthetic vertex count.

    Returns:
        Actual graph time, memory and quarantine count.
    """
    cpu = time.process_time()
    rows = [GroupRecord(i, "synthetic", "test" if i % 1024 == 0 else "train", "prose") for i in range(count)]
    edges = [(i, i + 1) for i in range(count - 1) if i % 128 != 127]
    groups = group_records(rows, edges)
    return {"vertices": count, "edges": len(edges), "components": len(groups),
            "quarantined_components": sum(flag for _, flag in groups.values()),
            "cpu_seconds": time.process_time() - cpu,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope": "synthetic union only; not identity discovery or semantic review"}


def allocation_probes() -> list[dict[str, object]]:
    """Measure canonical search limits separately from population feasibility.

    Returns:
        Frozen synthetic cases and statuses, never real source assignments.
    """
    results = []
    for count in (4, 8, 12, 1000, 390220):
        groups = [Component(f"synthetic-{i}", i % 4, (0, 1, 2), (1, 1, 1)) for i in range(count)]
        required = (Requirements(tuple((256, 48, 96) for _ in range(4)), (16,) * 4, (32, 32, 32))
                    if count >= 1000 else Requirements(tuple((0, 0, 1) for _ in range(4)), (1,) * 4, (1, 1, 1)))
        cpu = time.process_time()
        result = solve(groups, required, max_states=50000, cpu_seconds=5)
        results.append({"components": count, "requirements": asdict(required),
                        "status": result.status, "reason": result.reason, "states": result.states,
                        "has_feasible_witness": result.witness is not None,
                        "cpu_seconds": time.process_time() - cpu,
                        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    groups = [Component("g0", 0, (0, 1), (2, 0, 0)), Component("g1", 0, (0, 1), (1, 0, 0))]
    result = solve(groups, Requirements(((2, 1, 0),), (0,), (0, 0, 0)))
    results.append({"case": "hash_shortfall", "hash_roles": [hash_role(g) for g in groups],
                    "result": asdict(result)})
    return results


def main() -> None:
    """Run only local/synthetic CPU probes and refuse to overwrite any evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))
    cpu = time.process_time()
    rows = retained_inputs(args.root)
    report: dict[str, object] = {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                               "is_full_census": False, "is_training_manifest": False,
                               "gpu_hours": 0, "has_network_acquisition": False}
    report["registry"] = registry_report(args.root, rows)
    path = args.output / "retained.sqlite"
    report["retained_matching"] = measure_index(path, ((text, ids) for _, text, ids in rows), len(rows))
    report["retained_exhaustive_oracle"] = verify_retained_matches(path, rows)
    measurements = []
    for count, width, common in ((1000, 256, False), (10000, 256, False), (2000, 256, True), (390220, 1, False)):
        label = f"synthetic-{count}-{width}-common{int(common)}"
        result = measure_index(args.output / f"{label}.sqlite", synthetic_rows(count, width, common),
                               count, max_candidates=20000 if common else 1000000)
        measurements.append({"label": label, "synthetic_width": width, **result})
        print(json.dumps(measurements[-1]), flush=True)
    report["matching_scale"] = measurements
    report["grouping_scale"] = scale_grouping(390220)
    report["allocation"] = allocation_probes()
    report["total_cpu_seconds"] = time.process_time() - cpu
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["full_census_recommendation"] = "no-go pending scalable canonical solver, matching stress coverage and independent review"
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
