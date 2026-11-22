"""Score complete charged Q4 operators with task-clustered uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict

ARMS = ("native", "packed", "zigzag", "threshold_native", "threshold_packed", "threshold_zigzag")
METRICS = ("event_us", "wall_us")


class Trial(TypedDict):
    arm: str
    event_us: float
    wall_us: float
    submit_us: float
    repeats: int
    order: int
    layer: int
    row: int
    level: int
    round: int
    kind: str


class CaptureRow(TypedDict):
    task: str
    position: int
    token: int


class Layer(TypedDict):
    layer: int
    cutoffs: list[float]
    rows: list[CaptureRow]


class Sample(TypedDict):
    task: str
    gain: float
    noise: float


class Decision(TypedDict):
    gain: float
    lower: float
    upper: float
    twice_noise: float
    tasks: int
    decision: Literal["PASS", "FAIL", "INCONCLUSIVE"]


class Contrast(TypedDict):
    controls: dict[str, Decision]
    passed: bool
    failed: bool
    median_density: float
    median_retained_mass: float
    max_native_relative_rms: float
    min_native_cosine: float
    max_layout_relative_rms: float
    min_layout_cosine: float


class LayerResult(TypedDict):
    layer: int
    rows: int
    checks: int
    intervals: int
    comparisons: dict[str, Contrast]


def decide(samples: list[Sample]) -> Decision:
    """Give each task equal weight and resample whole tasks, retaining paired row gains."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        by_task[sample["task"]].append(sample["gain"])
    tasks = [median(values) for values in by_task.values()]
    if not tasks:
        raise ValueError("no held-out tasks")
    gain = median(tasks)
    noise = 2 * median(s["noise"] for s in samples)
    rng = random.Random(1121)
    resampled = sorted(median(rng.choices(tasks, k=len(tasks))) for _ in range(4096))
    lower, upper = resampled[102], resampled[-103]
    status: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "INCONCLUSIVE"
    if len(tasks) >= 3:
        if gain >= 0.10 and gain > noise and lower > 0:
            status = "PASS"
        elif upper < 0.10:
            status = "FAIL"
    return Decision(gain=gain, lower=lower, upper=upper, twice_noise=noise,
                    tasks=len(tasks), decision=status)


def paired(a: list[float], b: list[float], task: str) -> Sample:
    """Calculate paired fractional latency reduction and within-row repeat noise."""
    def spread(v: list[float]) -> float:
        center = median(v)
        return median(abs(t-center) for t in v) / center

    return Sample(task=task, gain=median((x-y)/x for x,y in zip(a,b,strict=True)),
                  noise=max(spread(a),spread(b)))


def layer_result(layer: Layer, path: Path) -> LayerResult:
    """Check every row, arm and interval in one complete device run."""
    records = [json.loads(s) for s in path.read_text().splitlines()]
    rows = layer["rows"]
    ident = layer["layer"]
    last = records[-1]
    if (last["kind"], last["layer"], last["rows"]) != ("complete",ident,len(rows)):
        raise ValueError("missing layer completion")
    checks = {}
    trials: dict[tuple[int,int,int,str], Trial] = {}
    for item in records[:-1]:
        row, level, arm = item["row"], item["level"], item["arm"]
        if item["layer"] != ident or row not in range(-1,len(rows)) or level not in range(3) or arm not in ARMS:
            raise ValueError("unknown check or timing identity")
        if item["kind"] == "check":
            key = (row,level,arm)
            if key in checks or not math.isclose(item["cutoff"], layer["cutoffs"][level], rel_tol=1e-7):
                raise ValueError("duplicate check or changed cutoff")
            if not 0 <= item["count"] <= 11008 or not 0 <= item["mass_fraction"] <= 1.000000000001:
                raise ValueError("invalid retained subset")
            if not all(math.isfinite(item[field]) for field in ("matched_max_error","native_relative_rms","native_cosine","layout_relative_rms","layout_cosine")):
                raise ValueError("nonfinite numerical result")
            if row == -1 and (item["count"] or item["matched_max_error"] or item["native_relative_rms"] or item["native_cosine"] != 1):
                raise ValueError("all-zero row failed")
            checks[key] = item
        elif item["kind"] == "trial":
            round_id = item["round"]
            key = (row,level,round_id,arm)
            if key in trials or row < 0 or round_id not in range(-1,5) or item["repeats"] != 20:
                raise ValueError("invalid trial")
            if item["order"] != (ARMS.index(arm)-row-level-round_id-1) % len(ARMS):
                raise ValueError("arm order mismatch")
            if not all(math.isfinite(item[field]) and item[field] > 0 for field in (*METRICS,"submit_us")):
                raise ValueError("invalid interval")
            trials[key] = item
        else:
            raise ValueError("unexpected operator record")
    if len(checks) != (len(rows)+1)*3*6 or len(trials) != len(rows)*3*6*6:
        raise ValueError("incomplete layer evidence")
    for row in range(len(rows)):
        counts = [checks[row,l,"native"]["count"] for l in range(3)]
        if counts != sorted(counts, reverse=True):
            raise ValueError("increasing threshold retained more entries")
        for level in range(3):
            for arm in ARMS:
                for field in ("count","mass_fraction"):
                    if checks[row,level,arm][field] != checks[row,level,"native"][field]:
                        raise ValueError("unmatched activation subset")
    comparisons: dict[str, Contrast] = {}
    for layout,level in (("packed",1),("zigzag",2)):
        controls = {}
        for base in ("native",layout):
            for metric in METRICS:
                samples = []
                for row,position in enumerate(rows):
                    baseline = [trials[row,level,k,base][metric] for k in range(5)]
                    candidate = [trials[row,level,k,"threshold_"+layout][metric] for k in range(5)]
                    samples.append(paired(baseline,candidate,position["task"]))
                controls[f"{base}/{metric}"] = decide(samples)
        comparisons[layout] = {"controls": controls,
            "passed": all(v["decision"]=="PASS" for v in controls.values()),
            "failed": any(v["decision"]=="FAIL" for v in controls.values()),
            "median_density":median(checks[row,level,layout]["count"]/11008 for row in range(len(rows))),
            "median_retained_mass":median(checks[row,level,layout]["mass_fraction"] for row in range(len(rows))),
            "max_native_relative_rms":max(checks[row,level,"threshold_"+layout]["native_relative_rms"] for row in range(len(rows))),
            "max_layout_relative_rms":max(checks[row,level,"threshold_"+layout]["layout_relative_rms"] for row in range(len(rows))),
            "min_layout_cosine":min(checks[row,level,"threshold_"+layout]["layout_cosine"] for row in range(len(rows))),
            "min_native_cosine":min(checks[row,level,"threshold_"+layout]["native_cosine"] for row in range(len(rows)))}
    return {"layer":ident,"rows":len(rows),"checks":len(checks),"intervals":len(trials),"comparisons":comparisons}


def screen(manifest: Path, data: Path) -> dict[str, object]:
    """Apply the two-of-three-layer gate separately to each frozen layout."""
    source = json.loads(manifest.read_text())
    if [layer["layer"] for layer in source["layers"]] != [0,17,35]:
        raise ValueError("frozen layers changed")
    layers = [layer_result(layer,data/f"layer-{layer['layer']}.jsonl") for layer in source["layers"]]
    gates = {}
    for layout in ("packed","zigzag"):
        passed = sum(layer["comparisons"][layout]["passed"] for layer in layers)
        failed = sum(layer["comparisons"][layout]["failed"] for layer in layers)
        gates[layout] = "PASS" if passed >= 2 else "FAIL" if failed >= 2 else "INCONCLUSIVE"
    return {"operator":gates,"layers":layers,"model_quality":"unmeasured","whole_request_speed":"unmeasured"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("data",type=Path)
    args = parser.parse_args()
    print(json.dumps(screen(args.manifest,args.data),indent=2))
