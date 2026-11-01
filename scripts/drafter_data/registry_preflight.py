"""Generic instruction exceptions belong to adapter spans, not matching strings.

This preflight verifies the frozen registry and records candidate dispositions.
It cannot activate exceptions or certify the independent second review.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from hf_sample_audit import prompt_text
from prepare_v2 import require_hash


class RegistryEntry(TypedDict):
    """Frozen text and lengths used to check one structural span."""

    entry_id: str
    text: str
    utf8_bytes: int
    unicode_scalars: int
    sha256: str


class RegistrySource(TypedDict):
    """Source and adapter identities bound to the reviewed instruction text."""

    source_id: str
    revision: str
    adapter_path: str
    adapter_sha256: str
    entries: list[RegistryEntry]


@dataclass(frozen=True)
class Span:
    """One half-open registry span in original user content.

    Attributes:
        entry_id: Frozen registry identifier.
        character_start: Unicode scalar offset before ChatML wrapping.
        character_end: Exclusive scalar offset.
        byte_start: UTF-8 offset without normalization.
        byte_end: Exclusive UTF-8 offset.
    """

    entry_id: str
    character_start: int
    character_end: int
    byte_start: int
    byte_end: int


@dataclass(frozen=True)
class RenderedInput:
    """Unchanged prompt bytes and proposed generic spans, never a masked input.

    Attributes:
        text: Complete natural user content.
        spans: Verified structural spans for diagnostic classification only.
    """

    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Overlap:
    """Raw overlap evidence survives a proposed generic-only disposition.

    Attributes:
        left_range: Original scalar coordinates.
        right_range: Original scalar coordinates.
        left_text: Original matched text, including payload if present.
        right_text: Original matched text.
        registry_entries: Both enclosing entries, or empty for payload matches.
        is_generic_candidate: True only when both complete matches are enclosed.
        is_exempt: Always false until an independent review is supplied later.
    """

    left_range: tuple[int, int]
    right_range: tuple[int, int]
    left_text: str
    right_text: str
    registry_entries: tuple[str, ...]
    is_generic_candidate: bool
    is_exempt: bool = False


def load_registry(path: Path, root: Path, expected_sha256: str) -> dict[str, RegistrySource]:
    """Verify the immutable file, adapters and all text digests locally.

    Args:
        path: Frozen review-only registry file.
        root: Repository containing the pinned adapters.
        expected_sha256: Digest from the approved checkpoint.

    Returns:
        Source-bound entries; no permission to exempt matches.

    Raises:
        ValueError: File, adapter or text identity differs.
    """
    require_hash(path, expected_sha256)
    document = json.loads(path.read_text(encoding="utf-8"))
    sources = cast(list[RegistrySource], document["sources"])
    result = {}
    for source in sources:
        require_hash(root / source["adapter_path"], source["adapter_sha256"])
        for entry in source["entries"]:
            raw = entry["text"].encode("utf-8")
            if (hashlib.sha256(raw).hexdigest() != entry["sha256"]
                    or len(raw) != entry["utf8_bytes"]
                    or len(entry["text"]) != entry["unicode_scalars"]):
                raise ValueError("registry text identity differs")
        result[source["source_id"]] = source
    return result


def render_input(source: RegistrySource, revision: str, fields: dict[str, str]) -> RenderedInput:
    """Resolve exact spans from source fields, never by searching payload text.

    Args:
        source: Hash-verified registry source.
        revision: Actual source revision being inspected.
        fields: Natural input fields; DocRED supplies already joined sentences.

    Returns:
        Complete user content and byte/scalar boundaries.

    Raises:
        ValueError: Revision, input controls or supported adapter identity differs.
    """
    if revision != source["revision"]:
        raise ValueError("source revision differs")
    source_id = source["source_id"]
    if source_id == "squad-original":
        text = "Context:\n" + fields["context"] + "\n\nQuestion:\n" + fields["question"]
        starts = [0, 9 + len(fields["context"])]
    elif source_id == "docred-original":
        text = "Extract the entities and their relations from this document.\n\n" + fields["sentences"]
        starts = [0]
    elif source_id == "commitpackft-python":
        text = prompt_text("commitpack", dict(fields))
        starts = [len(fields["subject"])]
    elif source_id == "ultrachat-first-user-sft":
        text = prompt_text("ultrachat", dict(fields))
        starts = []
    else:
        raise ValueError("unregistered adapter")
    if "<|im_start|>" in text or "<|im_end|>" in text:
        raise ValueError("embedded ChatML controls")
    spans = []
    for entry, start in zip(source["entries"], starts, strict=True):
        end = start + entry["unicode_scalars"]
        if text[start:end] != entry["text"]:
            raise ValueError("structural span differs")
        spans.append(Span(entry["entry_id"], start, end,
                          len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))))
    return RenderedInput(text, tuple(spans))


def inspect_overlap(left: RenderedInput, left_range: tuple[int, int],
                    right: RenderedInput, right_range: tuple[int, int]) -> Overlap:
    """Retain raw evidence and classify only wholly enclosed generic spans.

    Args:
        left: Original left input with verified spans.
        left_range: Half-open scalar bounds supplied by a matching audit.
        right: Original right input with verified spans.
        right_range: Half-open scalar bounds supplied by a matching audit.

    Returns:
        Candidate-only disposition; payload and cross-boundary matches stay raw.

    Raises:
        ValueError: A match is empty, reversed or outside its original input.
    """
    entries = []
    for record, bounds in ((left, left_range), (right, right_range)):
        start, end = bounds
        if not 0 <= start < end <= len(record.text):
            raise ValueError("invalid raw match coordinates")
        entries.extend(span.entry_id for span in record.spans
                       if span.character_start <= start and end <= span.character_end)
    return Overlap(left_range, right_range, left.text[slice(*left_range)],
                   right.text[slice(*right_range)], tuple(entries) if len(entries) == 2 else (),
                   len(entries) == 2)
