"""Dolly passages lack document identifiers, so recovery needs source evidence.

This bounded probe searches Wikipedia and compares complete passages against
historical revisions. Search hits and partial matches remain quarantined.
Original dataset text and previously frozen specifications are never changed.
"""

import argparse
import difflib
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from group_capacity import normalize_text
from prepare_v2 import file_hash, require_hash, write_new


class ArticleText(HTMLParser):
    """Rendered article text excludes scripts, styles and reference markers.

    Attributes:
        parts: Visible text fragments in document order.
        suppressed: Nesting depth of ignored script/style/sup elements.
    """

    def __init__(self) -> None:
        """Initialize an inert text extractor; no HTML resources are loaded."""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Suppress non-content elements while ignoring all HTML attributes.

        Args:
            tag: Parsed element name.
            attrs: Inert attributes, never interpreted as resource requests.
        """
        if tag in {"script", "style", "sup"}:
            self.suppressed += 1

    def handle_endtag(self, tag: str) -> None:
        """Restore text collection after an ignored element.

        Args:
            tag: Closed element name.
        """
        if tag in {"script", "style", "sup"}:
            self.suppressed = max(0, self.suppressed - 1)

    def handle_data(self, data: str) -> None:
        """Collect visible text without executing embedded code.

        Args:
            data: Literal decoded HTML text.
        """
        if not self.suppressed:
            self.parts.append(data)


def passage_words(text: str) -> list[str]:
    """Normalize comparison words without changing retained source bytes.

    Args:
        text: Dataset passage or rendered article text.

    Returns:
        NFKC/lowercase Unicode words excluding numeric citation markers.
    """
    return re.findall(r"\w+", normalize_text(re.sub(r"\[\d+\]", " ", text)))


def compare_passage(context: str, html: str) -> dict[str, object]:
    """Require a full contiguous passage, not a title or partial search hit.

    Args:
        context: Original Dolly context.
        html: Saved revision HTML from Wikipedia's parse endpoint.

    Returns:
        Full-match flag, matched word count and partial-match diagnostic.
    """
    parser = ArticleText()
    parser.feed(html)
    needle = passage_words(context)
    article = passage_words(" ".join(parser.parts))
    match = difflib.SequenceMatcher(None, needle, article, autojunk=False).find_longest_match()
    return {"context_words": len(needle), "longest_contiguous_words": match.size,
            "coverage": match.size / len(needle) if needle else 0,
            "is_full_passage_match": len(needle) >= 32 and match.size == len(needle)}


class WikiProbe:
    """A fixed network budget bounds evidence retrieval, not admission criteria.

    Attributes:
        directory: Fresh directory retaining every API response and ledger.
        requests: Recorded request identities and outcomes.
        received_bytes: Total downloaded bytes, including rejected responses.
    """

    def __init__(self, directory: Path) -> None:
        """Reserve a fresh evidence directory; never overwrite an earlier run.

        Args:
            directory: Nonexistent output directory.
        """
        directory.mkdir()
        self.directory = directory
        self.requests: list[dict[str, object]] = []
        self.received_bytes = 0

    def fetch(self, parameters: dict[str, str]) -> dict[str, object]:
        """Fetch one bounded Wikipedia API response and retain its raw hash.

        Args:
            parameters: Public article query parameters, never private data.

        Returns:
            Decoded JSON or an explicit error record on a failed request.
        """
        if len(self.requests) >= 96 or self.received_bytes >= 64 * 1024 * 1024:
            return {"error": "fixed_network_budget_exhausted"}
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"format": "json", "formatversion": "2", **parameters})
        record: dict[str, object] = {"url": url, "retrieved_unix": time.time()}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "M40LLM-ProvenanceAudit/1.0 (bounded research)"})
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            self.received_bytes += len(raw)
            path = self.directory / f"response-{len(self.requests):03}.json"
            with path.open("xb") as stream:
                stream.write(raw)
            record.update({"path": str(path), "sha256": file_hash(path), "bytes": len(raw)})
            if len(raw) > 4 * 1024 * 1024:
                result = {"error": "response_size_limit"}
            else:
                result = json.loads(raw)
                if not isinstance(result, dict):
                    result = {"error": "unexpected_json_type"}
        except (OSError, ValueError) as error:
            result = {"error": str(error)}
        record["error"] = result.get("error")
        self.requests.append(record)
        time.sleep(0.2)
        return result

    def recover(self, line: int, context: str) -> dict[str, object]:
        """Compare at most two search candidates against a pre-release revision.

        Args:
            line: One-based Dolly source line.
            context: Original context, not its answer or model output.

        Returns:
            Evidence-linked confidence classification, not training admission.
        """
        start = len(self.requests)
        phrase = " ".join(passage_words(context)[:10])
        search = self.fetch({"action": "query", "list": "search", "srsearch": f'"{phrase}"', "srlimit": "2"})
        candidates: list[dict[str, object]] = []
        for hit in search.get("query", {}).get("search", []):
            info = self.fetch({"action": "query", "prop": "revisions", "pageids": str(hit["pageid"]),
                               "rvstart": "2023-04-12T23:59:59Z", "rvdir": "older", "rvlimit": "1", "rvprop": "ids|timestamp"})
            pages = info.get("query", {}).get("pages", [])
            if not pages or not pages[0].get("revisions"):
                candidates.append({"pageid": hit["pageid"], "title": hit["title"], "status": "revision_unavailable"})
                continue
            revision = pages[0]["revisions"][0]
            parsed = self.fetch({"action": "parse", "oldid": str(revision["revid"]), "prop": "text|revid"})
            html = parsed.get("parse", {}).get("text")
            candidate = {"pageid": hit["pageid"], "title": pages[0]["title"],
                         "revision": revision, "url": f'https://en.wikipedia.org/w/index.php?oldid={revision["revid"]}'}
            if isinstance(html, str) and parsed["parse"].get("revid") == revision["revid"]:
                candidate.update(compare_passage(context, html))
            else:
                candidate["status"] = "revision_text_unavailable"
            candidates.append(candidate)
        matches = [item for item in candidates if item.get("is_full_passage_match")]
        return {"source_line": line, "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "request_indices": list(range(start, len(self.requests))), "candidates": candidates,
                "confidence": "full_passage_in_one_historical_candidate" if len(matches) == 1 else "unresolved",
                "canonical_document": f'enwiki:{matches[0]["pageid"]}' if len(matches) == 1 else None,
                "is_editor_source_attested": False, "is_training_admitted": False}


def main() -> None:
    """Probe exactly sixteen deterministically ordered eligible source passages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path("local-artifacts/2026-10-29-drafter-data/sources/dolly/databricks-dolly-15k.jsonl")
    require_hash(source, "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec")
    with args.tokens.open() as stream:
        lengths = {row["id"]: row["prompt_tokens"] for row in map(json.loads, stream)}
    with source.open() as stream:
        rows = [(line, row) for line, row in enumerate(map(json.loads, stream), 1)
                if row["category"] == "closed_qa" and 65 <= lengths[f"dolly:{line}"] <= 447]
    selected = sorted(rows, key=lambda item: hashlib.sha256(f"20261029:dolly:{item[0]}".encode()).hexdigest())[:16]
    probe = WikiProbe(args.output)
    write_new(args.output / "selection.json", {"source_sha256": file_hash(source), "tokens_sha256": file_hash(args.tokens),
              "eligible_source_count": len(rows), "selected_source_lines": [line for line, _ in selected],
              "maximum_requests": 96, "maximum_rows": 16, "maximum_response_bytes": 4 * 1024 * 1024})
    results = []
    for line, row in selected:
        result = probe.recover(line, row["context"])
        write_new(args.output / f"row-{line}.json", result)
        results.append(result)
    write_new(args.output / "report.json", {"rows": results, "requests": probe.requests,
              "received_bytes": probe.received_bytes, "additional_gpu_hours": 0,
              "is_r1_admitted": False, "is_exhaustive_provenance_search": False})


if __name__ == "__main__":
    main()
