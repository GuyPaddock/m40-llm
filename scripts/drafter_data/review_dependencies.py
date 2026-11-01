"""Review queues include unselected local dependencies and unresolved outside links.

This graph conservatively joins existing raw matches and source identity/code
candidates. It is not a semantic clearance engine or a new corpus index.
"""

import json
import sqlite3
from pathlib import Path

from group_capacity import find_root
from prepare_v2 import file_hash
from review_records import ReviewRecord
from template_audit import normalized_solution


def read_raw_pairs(root: Path, records: list[ReviewRecord]) -> list[tuple[int, int, str]]:
    """Bind reusable raw overlap evidence to the exact retained ledger order.

    Args:
        root: Repository with the immutable October 31 retained index.
        records: Complete four-source retained inputs in original order.

    Returns:
        All stored raw pair/reason tuples, with no registry masking.
    """
    path = root / "local-artifacts/2026-10-31-v3-engineering/retained/matches.sqlite"
    reports = json.loads((root / "docs/experiments/2026-10-31-v3-engineering-results.json").read_text())
    expected = next(row["measurement"]["database_sha256"] for row in reports["probes"] if row["case"] == "retained")
    if file_hash(path) != expected:
        raise ValueError("raw matching evidence identity differs")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        stored = list(connection.execute("SELECT id,raw,tokens FROM records ORDER BY id"))
        if len(stored) != len(records):
            raise ValueError("raw matching scope differs")
        for (number, text, ids), row in zip(stored, records, strict=True):
            if records[number] != row or text != row.text or tuple(json.loads(ids)) != row.tokens:
                raise ValueError("raw matching ledger order/content differs")
        pairs = [(left, right, reason) for left, right, reasons in connection.execute("SELECT * FROM matches")
                 for reason in json.loads(reasons)]
    finally:
        connection.close()
    return pairs


def build_dependencies(records: list[ReviewRecord], raw: dict[str, dict[str, object]],
                       pairs: list[tuple[int, int, str]]) -> dict[str, object]:
    """Join literal, provisional identity and supported code-template candidates.

    Args:
        records: All retained records, including length-ineligible/unselected rows.
        raw: Hash-bound raw source fields.
        pairs: Original raw overlaps, never reclassified or dropped.

    Returns:
        Transitive component membership, typed edges and unresolved code locators.
    """
    parents = {i: i for i in range(len(records))}
    edges = list(pairs)
    owners: dict[str, int] = {}
    unresolved = []
    for i, row in enumerate(records):
        fields = raw[row.identity]
        keys = []
        if row.source in ("squad-original", "docred-original"):
            keys.append("provisional_article:" + str(fields["title"]))
        elif row.source == "ultrachat":
            keys.append("conversation:" + str(fields["prompt_id"]))
        elif row.source == "commitpack":
            keys.extend("repository_alias:" + name.strip().lower() for name in str(fields["repos"]).split(","))
            keys.append("commit:" + str(fields["commit"]))
            for field in ("old_contents", "new_contents"):
                code = fields.get(field)
                fingerprint = normalized_solution(code) if isinstance(code, str) else None
                if fingerprint is None:
                    unresolved.append({"id": row.identity, "field": field})
                else:
                    keys.append("code_template:" + fingerprint)
        for key in sorted(set(keys)):
            if key in owners and owners[key] != i:
                edges.append((owners[key], i, key.split(":", 1)[0]))
            else:
                owners[key] = i
    for left, right, _ in edges:
        parents[find_root(parents, left)] = find_root(parents, right)
    components: dict[int, list[int]] = {}
    for i in range(len(records)):
        components.setdefault(find_root(parents, i), []).append(i)
    membership = {records[i].identity: members for members in components.values() for i in members}
    return {"membership": membership, "edges": edges, "unresolved_code_fields": unresolved,
            "component_count": len(components), "has_complete_semantic_closure": False}
