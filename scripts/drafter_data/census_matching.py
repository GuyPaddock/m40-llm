"""Lossless literal candidates reduce pair work without granting isolation.

SQLite postings retain raw records and every verified matching pair. A common
posting may still be quadratic; resource exhaustion stops as unknown, never
by dropping frequent shingles. Semantic and provenance edges remain external.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from group_capacity import find_root, normalize_text, word_shingles


class IncompleteAudit(RuntimeError):
    """A bounded audit has unfinished work and cannot clear the population."""


def token_spans(tokens: tuple[int, ...]) -> set[tuple[int, ...]]:
    """Return exact contiguous 32-token windows, without changing tokenization.

    Args:
        tokens: Original-runtime comparison IDs, or explicitly synthetic IDs.

    Returns:
        Unique windows for literal verification.
    """
    return {tokens[start:start + 32] for start in range(len(tokens) - 31)}


def digest_key(value: str | tuple[str, ...] | tuple[int, ...]) -> bytes:
    """Build a posting key; collisions must still pass full-content verification.

    Args:
        value: Normalized text, word shingle or token window.

    Returns:
        SHA256 posting key, not an identity or match certificate.
    """
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).digest()


def match_reasons(left: str, left_tokens: tuple[int, ...], right: str,
                  right_tokens: tuple[int, ...]) -> tuple[str, ...]:
    """Verify the unchanged literal thresholds against full comparison content.

    Args:
        left: Original or normalized text.
        left_tokens: Original-runtime comparison IDs.
        right: Original or normalized text.
        right_tokens: Original-runtime comparison IDs.

    Returns:
        All satisfied literal rules; no generic exemption is applied.
    """
    a, b = normalize_text(left), normalize_text(right)
    reasons = []
    if a and a == b:
        reasons.append("exact_normalized_text")
    if token_spans(left_tokens) & token_spans(right_tokens):
        reasons.append("shared_32_token_span")
    sa, sb = word_shingles(a), word_shingles(b)
    if sa | sb and 2 * len(sa & sb) >= len(sa | sb):
        reasons.append("five_word_jaccard_at_least_half")
    return tuple(reasons)


class CandidateIndex:
    """Disk-backed raw matching with complete prefix candidates for Jaccard >=0.5.

    Attributes:
        connection: Owned SQLite database, closed explicitly by the caller.
        candidate_checks: Number of verified candidate pairs.
        posting_count: Retained posting count for resource estimates.
        max_candidates: Hard verification ceiling; exceeding it is unknown.
        deadline: Process CPU-time deadline in seconds.
    """

    def __init__(self, path: Path, max_candidates: int = 1000000,
                 cpu_seconds: float = 300.0) -> None:
        """Create a fresh bounded index; never replace an evidence database.

        Args:
            path: New output database path.
            max_candidates: Maximum verified pairs for this attempt.
            cpu_seconds: Total process CPU budget from construction.
        """
        if path.exists():
            raise FileExistsError(path)
        if max_candidates < 0 or cpu_seconds <= 0:
            raise ValueError("invalid matching limits")
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            PRAGMA cache_size=-16384;
            PRAGMA temp_store=FILE;
            PRAGMA max_page_count=3145728;
            CREATE TABLE records(id INTEGER PRIMARY KEY, raw TEXT, tokens TEXT);
            CREATE TABLE postings(kind TEXT, key BLOB, id INTEGER,
                                  PRIMARY KEY(kind,key,id)) WITHOUT ROWID;
            CREATE TABLE matches(left_id INTEGER, right_id INTEGER, reasons TEXT,
                                 PRIMARY KEY(left_id,right_id)) WITHOUT ROWID;
        """)
        self.candidate_checks = 0
        self.posting_count = 0
        self.max_candidates = max_candidates
        self.deadline = time.process_time() + cpu_seconds

    def check_budget(self) -> None:
        """Stop on unfinished CPU work without treating it as quota failure."""
        if time.process_time() >= self.deadline:
            raise IncompleteAudit("matching CPU budget exhausted")

    def add(self, record_id: int, text: str, tokens: tuple[int, ...]) -> None:
        """Index a complete raw record and retain all verified earlier pairs.

        Args:
            record_id: Unique numeric audit locator, not a source document ID.
            text: Complete field/view with no instruction removal.
            tokens: IDs from the frozen tokenizer for this exact comparison view.

        Raises:
            IncompleteAudit: Candidate or CPU limit prevents a complete result.
        """
        self.check_budget()
        if self.connection.execute("SELECT 1 FROM records WHERE id=?", (record_id,)).fetchone():
            raise ValueError("duplicate record ID")
        normalized = normalize_text(text)
        shingles = sorted(word_shingles(normalized), key=lambda value: (digest_key(value), value))
        # Any sets with Jaccard >= 1/2 share a globally ordered prefix key.
        prefix_length = len(shingles) - (len(shingles) + 1) // 2 + 1
        keys = [("word", digest_key(value)) for value in shingles[:prefix_length]]
        keys.extend(("span", digest_key(value)) for value in token_spans(tokens))
        if normalized:
            keys.append(("exact", digest_key(normalized)))
        candidates: set[int] = set()
        for kind, key in keys:
            self.check_budget()
            for row in self.connection.execute("SELECT id FROM postings WHERE kind=? AND key=?", (kind, key)):
                candidates.add(row[0])
                if len(candidates) + self.candidate_checks > self.max_candidates:
                    raise IncompleteAudit("candidate budget exhausted; no frequent keys dropped")
        for other in sorted(candidates):
            self.check_budget()
            raw, ids = self.connection.execute("SELECT raw,tokens FROM records WHERE id=?", (other,)).fetchone()
            self.candidate_checks += 1
            reasons = match_reasons(raw, tuple(json.loads(ids)), text, tokens)
            if reasons:
                left, right = sorted((other, record_id))
                self.connection.execute("INSERT INTO matches VALUES(?,?,?)", (left, right, json.dumps(reasons)))
        self.connection.execute("INSERT INTO records VALUES(?,?,?)", (record_id, text, json.dumps(tokens)))
        self.connection.executemany("INSERT INTO postings VALUES(?,?,?)",
                                    ((kind, key, record_id) for kind, key in keys))
        self.posting_count += len(keys)

    def close(self) -> None:
        """Flush partial evidence even on an incomplete attempt and release ownership."""
        self.connection.commit()
        self.connection.close()


@dataclass(frozen=True)
class GroupRecord:
    """Native identity constraints remain attached to every graph vertex.

    Attributes:
        record_id: Unique vertex, including length-ineligible records.
        source: Native dataset identity.
        split: Unmodified native split label.
        family: Frozen task family.
    """

    record_id: int
    source: str
    split: str
    family: str


def group_records(records: list[GroupRecord], edges: list[tuple[int, int]]) -> dict[int, tuple[list[int], bool]]:
    """Union verified edges and quarantine cross-native-split/family components.

    Args:
        records: All vertices before filtering; caller supplies reviewed IDs.
        edges: Literal, document, code, fact or semantic connections.

    Returns:
        Minimum-ID roots with member IDs and mandatory quarantine flags.

    Raises:
        ValueError: Duplicate vertices or an edge endpoint is missing.
    """
    parents = {record.record_id: record.record_id for record in records}
    if len(parents) != len(records):
        raise ValueError("duplicate graph vertex")
    for left, right in edges:
        if left not in parents or right not in parents:
            raise ValueError("unknown graph endpoint")
        a, b = sorted((find_root(parents, left), find_root(parents, right)))
        parents[b] = a
    components: dict[int, list[GroupRecord]] = {}
    for record in records:
        components.setdefault(find_root(parents, record.record_id), []).append(record)
    result = {}
    for root, members in components.items():
        splits: dict[str, set[str]] = {}
        for member in members:
            splits.setdefault(member.source, set()).add(member.split)
        is_quarantined = (len({member.family for member in members}) > 1
                          or any(len(labels) > 1 for labels in splits.values()))
        result[root] = (sorted(member.record_id for member in members), is_quarantined)
    return result
