"""Verify integration sources and October inventories against immutable Git objects."""

import csv
import hashlib
import json
import posixpath
import re
import subprocess
from functools import cache
from pathlib import Path

MAIN = "e1403e0dbe2037cd8fc2b0af32addf97a4df3488"
PHASE4 = "48c63332462e009964e891be1cabaf139e3b50f8"
EXPERIMENTS = Path("docs/experiments")


def read_git(*args: str) -> bytes:
    """Return original object or command output, failing on missing evidence."""
    return subprocess.run(["git", *args], check=True, stdout=subprocess.PIPE).stdout


def read_table(path: Path) -> list[dict[str, str]]:
    """Read named TSV columns and reject incomplete rows."""
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if any(None in row or not all(row.values()) for row in rows):
        raise ValueError(f"Missing or extra fields in {path}")
    return rows


@cache
def blobs(revision: str) -> dict[str, str]:
    """Index tracked file blobs at one exact source revision."""
    files = {}
    for line in read_git("ls-tree", "-rz", revision).split(b"\0"):
        if line:
            metadata, path = line.split(b"\t", 1)
            _, kind, oid = metadata.split()
            if kind == b"blob":
                files[path.decode()] = oid.decode()
    return files


def check_inventory() -> dict[str, int]:
    """Reconstruct all per-topic diffs, implementation commits and report hashes."""
    folder = EXPERIMENTS / "2026-11-21-inventory"
    topics = read_table(folder / "branches.tsv")
    recorded_paths = read_table(folder / "paths.tsv")
    recorded_commits = read_table(folder / "implementation-commits.tsv")
    paths = []
    commits = []
    for topic in topics:
        branch, tip = topic["branch"], topic["tip"]
        read_git("merge-base", "--is-ancestor", tip, branch)
        base = read_git("merge-base", PHASE4, tip).decode().strip()
        report = read_git("show", f"{tip}:{topic['report_path']}")
        if base != topic["merge_base_with_phase4"] or hashlib.sha256(report).hexdigest() != topic["report_sha256"]:
            raise ValueError(f"{branch} report or merge base differs")
        for line in read_git("diff", "--name-status", base, tip).decode().splitlines():
            change, path = line.split("\t")
            paths.append([branch, change, path, *(blobs(rev).get(path, "absent") for rev in (tip, PHASE4, MAIN))])
        history = read_git("log", "--reverse", "--format=%H%x09%s", f"{base}..{tip}", "--", ".", ":!docs")
        commits.extend([branch, *line.split("\t", 1)] for line in history.decode().splitlines())
    if sorted(paths) != sorted(list(row.values()) for row in recorded_paths):
        raise ValueError("Modified paths or blob identities differ")
    if commits != [list(row.values()) for row in recorded_commits]:
        raise ValueError("Implementation commit sequence differs")
    return {"topics": len(topics), "path_rows": len(paths), "implementation_commits": len(commits)}


def check_decisions() -> dict[str, int]:
    """Resolve each recommendation and every relative Markdown link in its source."""
    decisions = read_table(EXPERIMENTS / "2026-11-21-integration/candidates.tsv")
    reports = set()
    linked_paths = 0
    for row in decisions:
        sha = row["source_tip"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"Incomplete source SHA: {sha}")
        read_git("merge-base", "--is-ancestor", sha, row["source_branch"])
        prefix = f"https://github.com/GuyPaddock/m40-llm/blob/{sha}/"
        url = row["source_report"]
        if not url.startswith(prefix):
            raise ValueError(f"Source URL does not match commit: {url}")
        path = url[len(prefix):]
        if (sha, path) in reports:
            continue
        reports.add((sha, path))
        data = read_git("show", f"{sha}:{path}").decode()
        if not path.endswith(".md"):
            continue
        for link in re.findall(r"\]\(([^ )]+)\)", data):
            if link.startswith(("#", "http://", "https://")):
                continue
            filename = posixpath.normpath(posixpath.join(posixpath.dirname(path), link.split("#")[0]))
            read_git("cat-file", "-e", f"{sha}:{filename}")
            linked_paths += 1
    return {"decisions": len(decisions), "source_reports": len(reports), "relative_links": linked_paths}


if __name__ == "__main__":
    print(json.dumps({"inventory": check_inventory(), "integration": check_decisions()}, indent=2))
