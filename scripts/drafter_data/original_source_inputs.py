"""Original document samples preserve source boundaries before admission.

These adapters inspect complete records inside bounded prefixes, never infer
missing context, and export inputs rather than source answers. The resulting
ledger is evidence for source selection, not an admitted training manifest.
"""

import argparse
import codecs
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

from group_capacity import normalize_text
from prepare_v2 import file_hash


def parse_json_prefix(raw: bytes, is_mapping: bool) -> list[tuple[str, dict[str, object]]]:
    """Decode complete records without repairing an incomplete last record.

    Args:
        raw: UTF-8 prefix of a top-level JSON array or mapping.
        is_mapping: Whether native document IDs are mapping keys.

    Returns:
        Up to 128 complete ID/record pairs in original order. Array indexes
        are locators only, never claimed as upstream document IDs.
    """
    text = codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
    decoder = json.JSONDecoder()
    expected = "{" if is_mapping else "["
    if not text.lstrip().startswith(expected):
        raise ValueError("unexpected JSON prefix container")
    position = text.index(expected) + 1
    rows = []
    while len(rows) < 128:
        while position < len(text) and text[position] in " \n\t\r,":
            position += 1
        if position >= len(text) or text[position] in "}]":
            break
        try:
            key = str(len(rows))
            if is_mapping:
                key, position = decoder.raw_decode(text, position)
                while position < len(text) and text[position].isspace():
                    position += 1
                if position >= len(text) or text[position] != ":":
                    break
                position += 1
                while position < len(text) and text[position].isspace():
                    position += 1
            record, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if not isinstance(key, str) or not isinstance(record, dict):
            raise ValueError("source record must be a keyed object")
        rows.append((key, record))
    return rows


def parse_pubtator_prefix(raw: bytes) -> list[tuple[str, str]]:
    """Retain complete PMID-delimited documents and omit annotation labels.

    Args:
        raw: PubTator prefix with title, abstract and annotation lines.

    Returns:
        PMID and complete title/abstract joined as in the BigBio formatter.
    """
    text = codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
    documents = []
    for block in text.replace("\r\n", "\n").split("\n\n")[:-1]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        title = lines[0].split("|t|", 1)
        abstract = lines[1].split("|a|", 1) if len(lines) > 1 else []
        if len(title) != 2 or len(abstract) != 2 or title[0] != abstract[0]:
            raise ValueError("mismatched or missing PMID/title/abstract")
        documents.append((title[0], title[1].strip() + " " + abstract[1].strip()))
        if len(documents) == 128:
            break
    return documents


def render_template_zero(path: Path, fields: dict[str, str]) -> str:
    """Render only the input side of the retained SciRIFF template zero.

    Args:
        path: Pinned inert YAML template, not executable remote code.
        fields: Exact literal placeholder substitutions, without gold answers.

    Returns:
        Input text retaining the substantive source instruction.
    """
    text = path.read_text().split("    jinja: |\n", 1)[1].split("      |||", 1)[0]
    text = textwrap.dedent(text).strip()
    for name, value in fields.items():
        text = text.replace("{{ " + name + " }}", value)
    if "{{" in text or "{%" in text:
        raise ValueError("unresolved source template expression")
    return text


def make_sample(source: str, split: str, document_id: str, text: str,
                path: Path, locator: str, template_group: str | None) -> dict[str, object]:
    """Attach source identity and unresolved isolation to a natural input.

    Args:
        source: Distinguishes original and SciRIFF-rendered diagnostic inputs.
        split: Upstream partition, or explicitly unresolved CV membership.
        document_id: Native document identifier, not a content hash substitute.
        text: Complete input without teacher labels or length adjustment.
        path: Retained source evidence whose checksum binds the input.
        locator: Record ID or offset within the pinned evidence.
        template_group: Known substantive task dependency, None if unaudited.

    Returns:
        Tokenizer-compatible sample with no training admission.
    """
    if not document_id or any(token in text for token in ("<|im_start|>", "<|im_end|>")):
        raise ValueError("missing document identity or embedded ChatML controls")
    return {"id": f"{source}:{split}:{locator}", "source": source, "split": split,
            "source_id": locator, "document_id": document_id,
            "provisional_group": document_id, "template_group": template_group,
            "text": text, "comparison_text": normalize_text(text),
            "sample_file": str(path), "sample_sha256": file_hash(path),
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "has_semantic_clearance": False, "is_training_admitted": False}


def prepare_pubmed(root: Path, previous: Path) -> list[dict[str, object]]:
    """Compare natural PubMed questions with unchanged SciRIFF instructions.

    Args:
        root: Current bounded sources and original test-ID metadata.
        previous: Immutable prior evidence containing the PubMedQA template.

    Returns:
        Two input variants per complete record, never two independent papers.
    """
    path = root / "pubmedqa-prefix.jsonpart"
    test_ids = json.loads((root / "pubmedqa-test_ground_truth.json.txt").read_text())
    samples = []
    for pmid, record in parse_json_prefix(path.read_bytes(), True):
        context = "\n".join(f"({label}) {paragraph}" for label, paragraph
                            in zip(record["LABELS"], record["CONTEXTS"], strict=True))
        question = record["QUESTION"]
        split = "test" if pmid in test_ids else "cv_unresolved"
        natural = f"Context:\n{context}\n\nQuestion:\n{question}"
        formatted = render_template_zero(previous / "sciriff-pubmedqa-template.txt",
                                          {"context": context, "question": question})
        for source, text, group in [("pubmedqa-original", natural, None),
                                     ("pubmedqa-sciriff", formatted, "pubmedqa-yes-no-maybe")]:
            samples.append(make_sample(source, split, "pmid:" + pmid, text, path, pmid, group))
    return samples


def prepare_documents(root: Path) -> list[dict[str, object]]:
    """Compare fixed extraction tasks without inventing independent templates.

    Args:
        root: Retained NCBI and DocRED source prefixes and SciRIFF template.

    Returns:
        Complete document inputs with explicit shared task-template groups.
    """
    samples = []
    for split in ("train", "devel", "test"):
        path = root / f"ncbi-{split}.txtpart"
        for pmid, abstract in parse_pubtator_prefix(path.read_bytes()):
            text = render_template_zero(root / "sciriff-ncbi_ner.yaml.txt",
                                         {"abstract": abstract, "ner_type_list | tojson": '["Disease"]'})
            samples.append(make_sample("ncbi-sciriff", split, "pmid:" + pmid,
                                        text, path, pmid, "ncbi-disease-extraction"))
    for split in ("train_annotated", "dev", "test"):
        path = root / f"docred-{split}.gzpart"
        # A bounded partial gzip stream is intentional; its incomplete tail is discarded.
        raw = zlib.decompressobj(31).decompress(path.read_bytes(), 4 * 1024 * 1024)
        for locator, record in parse_json_prefix(raw, False):
            sentences = "\n".join(" ".join(sentence) for sentence in record["sents"])
            text = "Extract the entities and their relations from this document.\n\n" + sentences
            samples.append(make_sample("docred-original", split, "wikipedia:" + record["title"],
                                        text, path, locator, "docred-entity-relation-extraction"))
    return samples


def prepare_squad(root: Path) -> list[dict[str, object]]:
    """Retain original paragraphs and authored questions with article grouping.

    Args:
        root: Bounded pinned SQuAD Parquet sample files.

    Returns:
        Natural QA samples; titles are article keys, not page revision IDs.
    """
    samples = []
    for split in ("train", "validation"):
        path = root / f"squad-{split}.json"
        for row in json.loads(path.read_text())["rows"]:
            record = row["record"]
            text = f'Context:\n{record["context"]}\n\nQuestion:\n{record["question"]}'
            samples.append(make_sample("squad-original", split, "wikipedia:" + record["title"],
                                        text, path, record["id"], None))
    return samples


def main() -> None:
    """Write a new local tokenizer ledger; never overwrite prior evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    samples = prepare_pubmed(args.root, args.previous) + prepare_documents(args.root) + prepare_squad(args.root)
    with (args.root / "sample-inputs.jsonl").open("x") as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
