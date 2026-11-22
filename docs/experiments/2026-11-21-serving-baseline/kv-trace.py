"""Validate TinyLlama historical KV dispatch on the full profiled request."""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median

ATTENTION_ROW = re.compile(
    r"^\[profile\] forward\.layer\.(\d+)\.seq_len\.(\d+)\.attention: "
    r"op=attention_last_token_f32_gqa_block_select_exact_(\S+) "
    r"launches=(\d+) .* elapsed_us=([0-9.]+)$"
)
BACKENDS = ("staged_fp16_k_q4_v_old", "fp16_k_q4_v_old_direct")
LAYERS = 22
PROMPT_TOKENS = 1390
OUTPUT_CAP = 32
RECENT = 1024


def dispatch_rows(path: Path, expected_backend: str, final_length: int) -> dict[tuple[int, int], float]:
    """Collect and validate one wrapper elapsed time per layer and input position."""
    rows: dict[tuple[int, int], float] = {}
    with path.open() as log:
        for line in log:
            match = ATTENTION_ROW.fullmatch(line.rstrip("\n"))
            if match is None:
                continue
            layer, length, backend, launches, elapsed = match.groups()
            key = int(layer), int(length)
            elapsed_us = float(elapsed)
            if backend != expected_backend or int(launches) != 1:
                raise ValueError(f"Wrong consumer or launch count at layer/length {key}")
            if key in rows or not math.isfinite(elapsed_us) or elapsed_us < 0:
                raise ValueError(f"Duplicate or invalid attention row at layer/length {key}")
            rows[key] = elapsed_us
    expected = {
        (layer, length)
        for layer in range(LAYERS)
        for length in range(1, final_length + 1)
    }
    if rows.keys() != expected:
        missing = sorted(expected - rows.keys())
        extra = sorted(rows.keys() - expected)
        raise ValueError(f"Incomplete dispatch grid: missing {missing[:5]}, extra {extra[:5]}")
    return rows


def summarize(directory: Path) -> dict[str, object]:
    """Replay the dispatch checks independently of the server's exit status."""
    arms: list[dict[str, object]] = []
    terminal_ids: list[list[int]] = []
    for arm, backend in enumerate(BACKENDS):
        response = json.loads((directory / f"calibrate-{arm}.json").read_text())
        if not response["success"] or len(response["records"]) != 1:
            raise ValueError(f"Invalid diagnostic response in arm {arm}")
        metrics = response["records"][0]["response"]["metrics"]
        ids = json.loads((directory / f"ids-{arm}.json").read_text())
        outputs = metrics["generated_tokens"]
        if (
            metrics["prompt_tokens"] != PROMPT_TOKENS
            or not 1 <= outputs <= OUTPUT_CAP
            or len(ids) != outputs
        ):
            raise ValueError(f"Unexpected diagnostic work in arm {arm}")
        terminal_ids.append(ids)
        final_length = PROMPT_TOKENS + outputs - 1
        rows = dispatch_rows(directory / f"trace-{arm}.log", backend, final_length)
        scopes = (
            ("recent_only_prefill", 1, RECENT),
            ("historical_prefill", RECENT + 1, PROMPT_TOKENS),
            ("decode_after_first_sample", PROMPT_TOKENS + 1, final_length),
        )
        bands: list[dict[str, object]] = []
        for band, first, last in scopes:
            if first > last:
                continue
            spans = [rows[layer, length] for layer in range(LAYERS) for length in range(first, last + 1)]
            bands.append({
                "band": band,
                "input_length_bounds": [first, last],
                "rows": len(spans),
                "median_wrapper_elapsed_us": median(spans),
            })
        arms.append({
            "consumer": backend,
            "attention_rows": len(rows),
            "generated_tokens": outputs,
            "stop_reason": metrics["stop_reason"],
            "bands": bands,
        })
    return {"arms": arms, "same_greedy_ids": terminal_ids[0] == terminal_ids[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
