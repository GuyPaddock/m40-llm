"""Local review packets bind retained prompts to source fields without admission.

The fixed sample is purposive and does not select training examples. Original
Qwen lengths and all generic instruction bytes remain unchanged.
"""

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

from hf_sample_audit import prompt_text
from original_source_inputs import parse_json_prefix
from prepare_v2 import file_hash
from registry_preflight import load_registry, render_input
from run_v3_preflight import HF_ROOT, ORIGINAL_ROOT, REGISTRY_DIGEST


@dataclass(frozen=True)
class ReviewRecord:
    """One retained input plus audit locators, not a verified upstream identity.

    Attributes:
        identity: Historical input-ledger ID.
        source: Frozen source adapter name.
        split: Historical native-split alias, never a reassignment.
        text: Untouched input text.
        tokens: Original normalized comparison IDs.
        prompt_tokens: Original natural QwenChat length.
        metadata: Retained source locators, provisional groups and evidence hashes.
    """

    identity: str
    source: str
    split: str
    text: str
    tokens: tuple[int, ...]
    prompt_tokens: int
    metadata: dict[str, object]


def read_records(root: Path) -> list[ReviewRecord]:
    """Load the original four-source input/token ledgers, including ineligible rows.

    Args:
        root: Repository containing retained evidence.

    Returns:
        Records in the same order as the preserved raw-match index.
    """
    records = []
    for directory in (HF_ROOT, ORIGINAL_ROOT):
        with (root / directory / "sample-tokens.jsonl").open() as stream:
            tokens = {row["id"]: row for row in map(json.loads, stream)}
        with (root / directory / "sample-inputs.jsonl").open() as stream:
            for row in map(json.loads, stream):
                if row["source"] in ("ultrachat", "commitpack", "squad-original", "docred-original"):
                    token = tokens[row["id"]]
                    records.append(ReviewRecord(row["id"], row["source"], row["split"], row["text"],
                                                tuple(token["comparison_ids"]), token["prompt_tokens"], row))
    if len({record.identity for record in records}) != len(records):
        raise ValueError("duplicate retained input identity")
    return records


def select_review_records(records: list[ReviewRecord]) -> list[ReviewRecord]:
    """Apply the preregistered sixteen-record rule without outcomes or replacements.

    Args:
        records: Retained ledger order, including natural lengths.

    Returns:
        Exactly four review records per family.

    Raises:
        ValueError: A frozen stratum is unavailable or produces duplicate locators.
    """
    eligible = [row for row in records if 65 <= row.prompt_tokens <= 447]
    strata = []
    for split in ("train", "test"):
        strata.append(([r for r in eligible if r.source == "ultrachat" and r.split == split], 2))
    code = [r for r in eligible if r.source == "commitpack"]
    permissive = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"}
    strata.append(([r for r in code if str(r.metadata.get("license_label", "")).lower() in permissive], 2))
    strata.append(([r for r in code if str(r.metadata.get("license_label", "")).lower().startswith(("agpl", "epl"))], 2))
    for split in ("train", "validation"):
        rows = [r for r in eligible if r.source == "squad-original" and r.split == split]
        if not rows:
            raise ValueError("missing SQuAD review stratum")
        strata.append(([r for r in rows if r.metadata["provisional_group"] == rows[0].metadata["provisional_group"]], 2))
    for split, count in (("train_annotated", 2), ("dev", 1), ("test", 1)):
        strata.append(([r for r in eligible if r.source == "docred-original" and r.split == split], count))
    selected = []
    for rows, count in strata:
        if len(rows) < count:
            raise ValueError("fixed review stratum lacks records; no replacement allowed")
        selected.extend(rows[:count])
    if len({r.identity for r in selected}) != 16:
        raise ValueError("review sample must have sixteen distinct locators")
    return selected


def read_raw_records(records: list[ReviewRecord], root: Path) -> dict[str, dict[str, object]]:
    """Bind every retained prompt to available raw fields and unchanged rendering.

    Args:
        records: Complete retained sample, including unselected dependencies.
        root: Repository containing frozen source files and registry.

    Returns:
        Raw audit records keyed by historical input IDs; missing fields stay absent.
    """
    registry = load_registry(root / "docs/experiments/2026-10-30-v3-generic-instruction-registry-v1.json",
                             root, REGISTRY_DIGEST)
    tables: dict[str, dict[str, dict[str, object]]] = {}
    result = {}
    for row in records:
        filename = str(row.metadata["sample_file"])
        path = root / filename
        if filename not in tables:
            if file_hash(path) != row.metadata["sample_sha256"]:
                raise ValueError("retained source hash differs")
            if row.source == "commitpack":
                raw = [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1][:128]]
                tables[filename] = {str(i + 1): value for i, value in enumerate(raw)}
            elif row.source == "docred-original":
                payload = zlib.decompressobj(31).decompress(path.read_bytes(), 4 * 1024 * 1024)
                tables[filename] = dict(parse_json_prefix(payload, False))
            else:
                key = "prompt_id" if row.source == "ultrachat" else "id"
                tables[filename] = {r["record"][key]: r["record"] for r in json.loads(path.read_text())["rows"]}
        locator = str(row.metadata["source_id"])
        if row.source == "commitpack":
            locator = locator.rsplit(":", 1)[1]
        raw = tables[filename][locator]
        if row.source in ("ultrachat", "commitpack"):
            text = prompt_text(row.source, raw)
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            if digest != row.metadata["record_sha256"]:
                raise ValueError("retained raw record identity differs")
        else:
            fields = ({"context": raw["context"], "question": raw["question"]} if row.source == "squad-original"
                      else {"sentences": "\n".join(" ".join(sentence) for sentence in raw["sents"])})
            text = render_input(registry[row.source], registry[row.source]["revision"], fields).text
        if text != row.text:
            raise ValueError("raw source no longer reproduces retained input")
        result[row.identity] = raw
    return result


def freeze_manifest(root: Path, output: Path) -> None:
    """Freeze the sixteen review locators and input identities before timed work.

    Args:
        root: Repository containing immutable ledgers.
        output: New public metadata-only manifest path.
    """
    rows = select_review_records(read_records(root))
    registry = load_registry(root / "docs/experiments/2026-10-30-v3-generic-instruction-registry-v1.json",
                             root, REGISTRY_DIGEST)
    aliases = {"ultrachat": "ultrachat-first-user-sft", "commitpack": "commitpackft-python"}
    manifest = {"version": "v3-review-cost-1", "is_training_manifest": False,
                "human_reviewers_available": 0, "rows": [], "inputs": {}}
    for directory in (HF_ROOT, ORIGINAL_ROOT):
        for name in ("sample-inputs.jsonl", "sample-tokens.jsonl"):
            path = directory / name
            manifest["inputs"][str(path)] = file_hash(root / path)
    for row in rows:
        manifest["rows"].append({"id": row.identity, "source": row.source, "split": row.split,
                                 "revision": registry[aliases.get(row.source, row.source)]["revision"],
                                 "prompt_tokens": row.prompt_tokens,
                                 "prompt_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                                 "sample_file": row.metadata["sample_file"],
                                 "sample_sha256": row.metadata["sample_sha256"],
                                 "license_label": row.metadata.get("license_label")})
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
