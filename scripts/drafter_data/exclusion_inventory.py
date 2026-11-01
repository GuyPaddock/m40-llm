"""Benchmark exclusions include saved answers as well as prompt fixtures.

This inventory hashes tracked fixtures and retained textual experiment files,
deduplicates comparison text, and records missing local artifact references.
It does not claim semantic/template clearance from literal scans alone.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

from group_capacity import normalize_text
from prepare_v2 import file_hash, write_new


def text_values(value: object) -> list[str]:
    """Collect JSON string values without treating numbers as generated prose.

    Args:
        value: Decoded JSON record, array or scalar.

    Returns:
        String leaves, including prompts, outputs and referenced paths.
    """
    result: list[str] = []
    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            result.extend(text_values(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(text_values(child))
    return result


def parse_text(text: str, suffix: str) -> tuple[list[str], list[str]]:
    """Extract literals from already decoded evidence without evaluating code.

    Args:
        text: UTF-8 evidence content.
        suffix: Original textual format suffix.

    Returns:
        Comparison texts and unresolved parse/size errors.
    """
    result: list[str] = []
    errors: list[str] = []
    if suffix == ".json":
        try:
            result = text_values(json.loads(text))
        except ValueError:
            errors.append("invalid_json")
    elif suffix == ".jsonl":
        for line, record in enumerate(text.splitlines(), 1):
            if record.strip():
                try:
                    result.extend(text_values(json.loads(record)))
                except ValueError:
                    errors.append(f"invalid_jsonl_line:{line}")
    else:
        result = text.splitlines()
        # Extract ordinary and raw literals too; unexpanded dynamic templates
        # remain subject to the manual semantic audit, never an automatic pass.
        for literal in re.findall(r'"(?:\\.|[^"\\])*"', text):
            try:
                result.append(json.loads(literal))
            except ValueError:
                pass
        result.extend(re.findall(r'r#"(.*?)"#', text, flags=re.DOTALL))
    return result, errors


def read_texts(path: Path) -> tuple[list[str], list[str]]:
    """Read text and archive members in memory, never extracting archive paths.

    Args:
        path: Tracked fixture or retained textual experiment artifact.

    Returns:
        Literal texts and explicit parse/size gaps.
    """
    if path.name.endswith(".tar.gz"):
        texts: list[str] = []
        errors: list[str] = []
        total = 0
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not is_text_artifact(Path(member.name)):
                    continue
                total += member.size
                if total > 128 * 1024 * 1024:
                    errors.append("archive_text_size_limit")
                    break
                stream = archive.extractfile(member)
                if stream is None:
                    errors.append(f"unreadable_member:{member.name}")
                    continue
                try:
                    with stream:
                        raw = stream.read()
                    member_path = Path(member.name)
                    if member.name.endswith(".gz"):
                        # Source snapshots repeat measured artifacts already inventoried.
                        if member_path.is_file() and hashlib.sha256(raw).hexdigest() == file_hash(member_path):
                            continue
                        if member.name.endswith(".tar.gz"):
                            errors.append(f"unresolved_nested_archive:{member.name}")
                            continue
                        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                            raw = compressed.read(128 * 1024 * 1024 - total + 1)
                        total += len(raw)
                        if total > 128 * 1024 * 1024:
                            errors.append("archive_text_size_limit")
                            break
                        member_path = member_path.with_suffix("")
                    content = raw.decode("utf-8")
                    values, failures = parse_text(content, member_path.suffix)
                    texts.extend(values)
                    errors.extend(f"{member.name}:{failure}" for failure in failures)
                except UnicodeDecodeError:
                    errors.append(f"invalid_utf8:{member.name}")
        return texts, errors
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024:
        return [], ["decompressed_size_limit"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [], ["invalid_utf8"]
    suffix = path.with_suffix("").suffix if path.suffix == ".gz" else path.suffix
    return parse_text(text, suffix)


def is_text_artifact(path: Path) -> bool:
    """Recognize bounded textual evidence, not models or executable binaries.

    Args:
        path: Candidate artifact path.

    Returns:
        Whether the extension is part of the evidence inventory.
    """
    return path.suffix in {".rs", ".py", ".md", ".json", ".jsonl", ".txt", ".log", ".gz", ".sha256"}


def build_inventory(directory: Path) -> None:
    """Freeze an expanded literal corpus and all unresolved artifact references.

    Args:
        directory: New local directory for input text and hash inventory.
    """
    directory.mkdir()
    paths = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    selected = {Path(path) for path in paths if is_text_artifact(Path(path)) and
                path.startswith(("data/", "src/", "tests/", "examples/", "benches/", "docs/experiments/", "docs/perf_baselines")) and
                not path.startswith("docs/experiments/2026-10-29-drafter-")}
    for path in Path("local-artifacts").rglob("*"):
        if path.is_file() and is_text_artifact(path) and "drafter-data" not in str(path) and "drafter-recovery" not in str(path):
            selected.add(path)
    inventory: list[dict[str, object]] = []
    texts: dict[str, str] = {}
    occurrences: dict[str, list[str]] = {}
    missing: set[str] = set()
    for path in sorted(selected):
        values, errors = read_texts(path)
        inventory.append({"path": str(path), "sha256": file_hash(path), "errors": errors})
        for value in values:
            for reference in re.findall(r'local-artifacts/[^\s"\x27<>`]+', value):
                reference = reference.rstrip(".,);:")
                candidate = Path(reference)
                if "drafter-data" not in reference and "drafter-recovery" not in reference and is_text_artifact(candidate) and not candidate.exists():
                    missing.add(reference)
            normalized = normalize_text(value)
            if not normalized:
                continue
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            texts[digest] = normalized
            occurrences.setdefault(digest, [])
            if str(path) not in occurrences[digest]:
                occurrences[digest].append(str(path))
    with (directory / "inputs.jsonl").open("x", encoding="utf-8") as stream:
        for digest, text in sorted(texts.items()):
            stream.write(json.dumps({"id": f"exclude:{digest}", "text": text, "comparison_text": text}, ensure_ascii=False) + "\n")
    write_new(directory / "inventory.json", {"files": inventory, "text_occurrences": occurrences,
              "missing_local_references": sorted(missing), "unique_texts": len(texts),
              "is_manual_template_audit_complete": False, "is_r1_admitted": False})


def main() -> None:
    """Create an explicit local benchmark corpus without network or GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_inventory(args.output)


if __name__ == "__main__":
    main()
