"""Pinned dataset samples provide feasibility evidence, not corpus admission.

HTTP reads are cached with hashes and strict byte budgets. Parquet ranges are
validated before decoding; no remote dataset loader or reference code runs.
"""

import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet

from prepare_v2 import write_new


class BoundedSource:
    """Retain at most 32 MiB of public dataset evidence per source directory.

    Attributes:
        directory: Receipt and raw response directory, reused across reads.
    """

    def __init__(self, directory: Path) -> None:
        """Create a local cache without contacting the source.

        Args:
            directory: Source-specific evidence location.
        """
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, start: int | None = None, size: int = 2 * 1024 * 1024) -> tuple[bytes, dict[str, object]]:
        """Read a bounded resource or exact byte range, preserving its receipt.

        Args:
            url: Public HTTPS URL, pinned for all dataset payload reads.
            start: Byte offset for a range request, or None for metadata.
            size: Maximum bytes for metadata, exact byte count for a range.

        Returns:
            Raw bytes and their URL/hash/range receipt.

        Raises:
            ValueError: Budget exhausted or server ignored the required range.
        """
        key = hashlib.sha256(f"{url}:{start}:{size}".encode()).hexdigest()
        payload = self.directory / f"{key}.bin"
        receipt_path = self.directory / f"{key}.json"
        if payload.exists() and receipt_path.exists():
            raw = payload.read_bytes()
            receipt = json.loads(receipt_path.read_text())
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("cached evidence hash differs")
            return raw, receipt
        used = sum(path.stat().st_size for path in self.directory.glob("*.bin"))
        if size <= 0 or used + size > 32 * 1024 * 1024:
            raise ValueError("32 MiB per-source evidence budget exhausted")
        if len(list(self.directory.glob("*.json"))) >= 100:
            raise ValueError("100-request per-source limit reached")
        headers = {"User-Agent": "m40-llm-cpu-dataset-feasibility/1.0", "Accept-Encoding": "identity"}
        if start is not None:
            headers["Range"] = f"bytes={start}-{start + size - 1}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=40) as response:
            content_range = response.headers.get("Content-Range")
            if start is not None:
                validate_range(response.status, content_range, start, size)
            raw = response.read(size + 1)
            if len(raw) > size or (start is not None and len(raw) != size):
                raise ValueError("response size differs from bounded request")
            receipt = {"url": url, "status": response.status, "content_range": content_range,
                       "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                       "retrieved_unix": time.time(), "etag": response.headers.get("ETag")}
        payload.write_bytes(raw)
        write_new(receipt_path, receipt)
        return raw, receipt


def validate_range(status: int, content_range: str | None, start: int, size: int) -> int:
    """Reject ignored, truncated or mismatched ranges before reading a body.

    Args:
        status: HTTP response status.
        content_range: Server range declaration.
        start: Requested first byte.
        size: Requested byte count.

    Returns:
        Declared total resource length.
    """
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
    if status != 206 or match is None:
        raise ValueError("server did not honor byte range")
    first, last, total = map(int, match.groups())
    if first != start or last != start + size - 1 or total <= last:
        raise ValueError("server returned a different byte range")
    return total


class PinnedRangeReader(io.RawIOBase):
    """Expose a pinned Parquet object through bounded, recorded HTTP ranges.

    Attributes:
        source: Shared per-source budget and response cache.
        url: Immutable dataset file URL.
        position: Current byte offset for the file interface.
        length: Server-declared total length, not downloaded in full.
    """

    def __init__(self, source: BoundedSource, url: str) -> None:
        """Inspect four bytes to establish the remote file size.

        Args:
            source: Budget shared by all sampled splits for this source.
            url: Revision-pinned dataset payload URL.
        """
        super().__init__()
        self.source = source
        self.url = url
        self.position = 0
        raw, receipt = source.fetch(url + "?probe=0-4", 0, 4)
        if raw != b"PAR1":
            raise ValueError("not a Parquet payload")
        self.length = validate_range(206, str(receipt["content_range"]), 0, 4)

    def readable(self) -> bool:
        """Report support for PyArrow's read-only file interface."""
        return True

    def seekable(self) -> bool:
        """Allow footer and selected column reads without downloading the file."""
        return True

    def tell(self) -> int:
        """Return the current byte offset."""
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        """Position the reader without an HTTP request.

        Args:
            offset: Relative or absolute byte displacement.
            whence: Standard beginning, current or end origin.

        Returns:
            New absolute byte offset.
        """
        origin = {0: 0, 1: self.position, 2: self.length}[whence]
        position = origin + offset
        if not 0 <= position <= self.length:
            raise ValueError("seek outside remote object")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        """Fetch only the requested bytes under the shared source budget.

        Args:
            size: Requested byte count; -1 requests the bounded remainder.

        Returns:
            Requested bytes, or empty bytes at EOF.
        """
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size == 0:
            return b""
        raw, _ = self.source.fetch(self.url + f"?probe={self.position}-{size}", self.position, size)
        self.position += len(raw)
        return raw


def sample_parquet(source: BoundedSource, url: str, output: Path, group_count: int = 3,
                   columns: list[str] | None = None) -> None:
    """Inspect at most 128 rows spread over evenly spaced row groups.

    Args:
        source: Budget and receipt storage.
        url: Pinned Parquet shard URL.
        output: New local evidence file, never a training manifest.
        group_count: Fixed number of row groups, bounded to eight.
        columns: Optional column subset to avoid acquiring unused answers.
    """
    with PinnedRangeReader(source, url) as reader:
        table = pyarrow.parquet.ParquetFile(reader)
        if not 2 <= group_count <= 8:
            raise ValueError("sample group count must be between two and eight")
        groups = sorted({index * (table.num_row_groups - 1) // (group_count - 1) for index in range(group_count)})
        rows: list[dict[str, object]] = []
        for group in groups:
            batch = table.read_row_group(group, columns=columns)
            count = min(128 // len(groups), batch.num_rows)
            indices = [index * batch.num_rows // count for index in range(count)]
            for index, record in zip(indices, batch.take(indices).to_pylist()):
                rows.append({"row_group": group, "row_in_group": index, "record": record})
        write_new(output, {"url": url, "total_shard_rows": table.metadata.num_rows,
                           "row_groups": table.num_row_groups, "sampled_groups": groups,
                           "schema": str(table.schema_arrow), "rows": rows,
                           "is_representative_random_sample": False, "is_training_admitted": False})


def main() -> None:
    """Fetch explicitly named metadata or bounded Parquet samples only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument("--prefix-bytes", type=int)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--columns", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace evidence")
    source = BoundedSource(args.cache)
    if args.parquet:
        sample_parquet(source, args.url, args.output, args.groups, args.columns)
    else:
        raw, receipt = source.fetch(args.url, 0 if args.prefix_bytes else None,
                                    args.prefix_bytes or 2 * 1024 * 1024)
        args.output.write_bytes(raw)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
