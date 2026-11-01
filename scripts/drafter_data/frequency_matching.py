"""Two-pass global frequency ordering avoids common-header prefix candidates.

Every raw record and required match remains auditable. Rare-first ordering
changes candidate generation only; complete-content checks and all true
32-token/exact matches remain unchanged. No registry exemptions are applied.
"""

import json
import sqlite3
import time
from pathlib import Path

from census_matching import IncompleteAudit, digest_key, match_reasons, token_spans
from group_capacity import normalize_text, word_shingles


class FrequencyIndex:
    """Own a disk-backed, two-phase lossless literal index.

    Attributes:
        connection: SQLite evidence database, explicitly closed by its owner.
        candidate_checks: Number of full-content pair verifications.
        posting_count: Retained exact/span/prefix postings.
        cpu_deadline: Process CPU budget, including staging and frequency counting.
        wall_deadline: Elapsed-time budget, including storage waits.
        is_frozen: True after frequency counting; additions are then forbidden.
    """

    def __init__(self, path: Path, cpu_seconds: float = 300.0,
                 wall_seconds: float = 600.0) -> None:
        """Create fresh evidence with bounded SQLite memory and explicit deadlines.

        Args:
            path: New database path; never overwrite an earlier experiment.
            cpu_seconds: Entire index's aggregate process CPU allowance.
            wall_seconds: Entire index's wall-time allowance.
        """
        if path.exists():
            raise FileExistsError(path)
        if cpu_seconds <= 0 or wall_seconds <= 0:
            raise ValueError("positive deadlines required")
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            PRAGMA cache_size=-16384;
            PRAGMA temp_store=FILE;
            PRAGMA max_page_count=2621440;
            CREATE TABLE records(id INTEGER PRIMARY KEY, raw TEXT, tokens TEXT, shingles INTEGER);
            CREATE TABLE frequencies(key BLOB PRIMARY KEY, frequency INTEGER) WITHOUT ROWID;
            CREATE TABLE postings(kind TEXT, key BLOB, id INTEGER,
                                  PRIMARY KEY(kind,key,id)) WITHOUT ROWID;
            CREATE TABLE candidates(id INTEGER PRIMARY KEY);
            CREATE TABLE matches(left_id INTEGER, right_id INTEGER, reasons TEXT,
                                 PRIMARY KEY(left_id,right_id)) WITHOUT ROWID;
            CREATE TABLE state(status TEXT);
            INSERT INTO state VALUES('staging');
        """)
        self.candidate_checks = 0
        self.posting_count = 0
        self.cpu_deadline = time.process_time() + cpu_seconds
        self.wall_deadline = time.monotonic() + wall_seconds
        self.is_frozen = False
        self.connection.set_progress_handler(self.should_interrupt, 10000)

    def should_interrupt(self) -> int:
        """Interrupt long SQLite operations when either whole-index budget expires."""
        return int(time.process_time() >= self.cpu_deadline or time.monotonic() >= self.wall_deadline)

    def check_budget(self) -> None:
        """Incomplete work cannot be called a complete match audit."""
        if self.should_interrupt():
            raise IncompleteAudit("frequency index CPU/wall budget exhausted")

    def add(self, record_id: int, text: str, tokens: tuple[int, ...]) -> None:
        """Stage full content and count global shingle frequencies once per record.

        Args:
            record_id: Unique audit locator; not an inferred document identity.
            text: Original unmasked input or audit field.
            tokens: Frozen-runtime normalized comparison IDs, or synthetic IDs.
        """
        self.check_budget()
        if self.is_frozen:
            raise ValueError("cannot add after global frequency freeze")
        shingles = word_shingles(normalize_text(text))
        self.connection.execute("INSERT INTO records VALUES(?,?,?,?)",
                                (record_id, text, json.dumps(tokens), len(shingles)))
        self.connection.executemany(
            "INSERT INTO frequencies VALUES(?,1) ON CONFLICT(key) DO UPDATE SET frequency=frequency+1",
            ((digest_key(value),) for value in shingles))
        if record_id % 1000 == 0:
            self.connection.commit()

    def ordered_shingles(self, text: str) -> list[tuple[str, ...]]:
        """Resolve one immutable global ordering, including deterministic hash ties.

        Args:
            text: Full original record whose keys were counted in pass one.

        Returns:
            Shingles in increasing global frequency/key/content order.
        """
        shingles = word_shingles(normalize_text(text))
        hashes = {value: digest_key(value) for value in shingles}
        keys = list(set(hashes.values()))
        frequencies = {}
        for offset in range(0, len(keys), 500):
            self.check_budget()
            chunk = keys[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            frequencies.update(self.connection.execute(
                f"SELECT key,frequency FROM frequencies WHERE key IN ({placeholders})", chunk))
        return sorted(shingles, key=lambda value: (frequencies[hashes[value]], hashes[value], value))

    def verify_record(self, record_id: int, text: str, tokens: tuple[int, ...]) -> None:
        """Verify every complete prefix/exact/span candidate and retain all matches.

        Args:
            record_id: Frozen record locator.
            text: Original unmasked content.
            tokens: Frozen comparison IDs.
        """
        shingles = self.ordered_shingles(text)
        prefix = len(shingles) - (len(shingles) + 1) // 2 + 1
        keys = [("word", digest_key(value)) for value in shingles[:prefix]]
        keys.extend(("span", digest_key(value)) for value in token_spans(tokens))
        normalized = normalize_text(text)
        if normalized:
            keys.append(("exact", digest_key(normalized)))
        self.connection.execute("DELETE FROM candidates")
        for kind, key in keys:
            self.check_budget()
            if kind == "word":
                self.connection.execute("""INSERT OR IGNORE INTO candidates
                    SELECT p.id FROM postings p JOIN records r ON r.id=p.id
                    WHERE p.kind=? AND p.key=? AND 2*r.shingles>=? AND r.shingles<=2*?""",
                                        (kind, key, len(shingles), len(shingles)))
            else:
                self.connection.execute("INSERT OR IGNORE INTO candidates SELECT id FROM postings WHERE kind=? AND key=?",
                                        (kind, key))
        for other, raw, ids in self.connection.execute(
                "SELECT r.id,r.raw,r.tokens FROM candidates c JOIN records r ON r.id=c.id ORDER BY r.id"):
            self.check_budget()
            self.candidate_checks += 1
            reasons = match_reasons(raw, tuple(json.loads(ids)), text, tokens)
            if reasons:
                left, right = sorted((other, record_id))
                self.connection.execute("INSERT INTO matches VALUES(?,?,?)", (left, right, json.dumps(reasons)))
        before = self.connection.total_changes
        self.connection.executemany("INSERT OR IGNORE INTO postings VALUES(?,?,?)",
                                    ((kind, key, record_id) for kind, key in keys))
        self.posting_count += self.connection.total_changes - before

    def finish(self) -> None:
        """Freeze frequencies and match every staged record, or leave unknown evidence.

        Raises:
            IncompleteAudit: Time or SQLite resource failure prevents completion.
        """
        if self.is_frozen:
            raise ValueError("index already frozen")
        self.is_frozen = True
        self.connection.execute("UPDATE state SET status='matching'")
        self.connection.commit()
        try:
            for number, text, ids in self.connection.execute("SELECT id,raw,tokens FROM records ORDER BY id"):
                self.check_budget()
                self.verify_record(number, text, tuple(json.loads(ids)))
            self.check_budget()
            self.connection.execute("UPDATE state SET status='complete'")
            self.connection.commit()
        except (IncompleteAudit, sqlite3.OperationalError) as error:
            self.connection.set_progress_handler(None, 0)
            self.connection.execute("UPDATE state SET status='unknown'")
            self.connection.commit()
            raise IncompleteAudit(str(error)) from error

    def close(self) -> None:
        """Preserve partial evidence and close, never upgrading an unfinished status."""
        self.connection.set_progress_handler(None, 0)
        self.connection.execute("UPDATE state SET status='unknown' WHERE status!='complete'")
        self.connection.commit()
        self.connection.close()
