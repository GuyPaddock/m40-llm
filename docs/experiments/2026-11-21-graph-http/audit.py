"""Recompute graph HTTP latency, CPU accounting and diagnostic GPU observations."""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import median


def cpu_ticks(path: Path) -> tuple[str, str, int]:
    """Read PID, process start and total user/system CPU ticks from proc stat."""
    pid, fields = path.read_text().rsplit(") ", 1)
    values = fields.split()
    return pid.split(" ", 1)[0], values[19], int(values[11]) + int(values[12])


def request_device_samples(directory: Path, labels: list[str]) -> dict[str, int | float | list[int]]:
    """Select board telemetry within each timed request's enclosing snapshots."""
    windows = []
    for label in labels:
        stamps = [
            datetime.fromisoformat((directory / f"{label}.{suffix}").read_text().splitlines()[0])
            for suffix in ("before", "after")
        ]
        if stamps[0].utcoffset() != stamps[1].utcoffset() or stamps[0] >= stamps[1]:
            raise ValueError(f"cannot align local NVML clock for {label}")
        windows.append(tuple(stamp.replace(tzinfo=None) for stamp in stamps))
    temperatures, clocks, busy, device_memory = [], [], [], []
    counts = [0] * len(windows)
    with (directory / "gpu.csv").open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            stamp = datetime.fromisoformat(row["timestamp"].replace("/", "-"))
            for index, (start, end) in enumerate(windows):
                if start <= stamp <= end:
                    counts[index] += 1
                    temperatures.append(int(row["temperature.gpu"]))
                    clocks.append(int(row["clocks.current.sm [MHz]"].split()[0]))
                    busy.append(int(row["utilization.gpu [%]"].split()[0]))
                    device_memory.append(int(row["memory.used [MiB]"].split()[0]))
    if not all(counts):
        raise ValueError("missing device samples in a timed request")
    return {"samples_per_request": counts, "temperature_c": [min(temperatures), max(temperatures)],
            "clocks_sm_mhz": sorted(set(clocks)), "median_board_gpu_busy_pct": median(busy),
            "board_memory_mib": [min(device_memory), max(device_memory)]}


def audit(directory: Path) -> dict[str, object]:
    """Check paired requests and reconstruct costs from raw host and server records."""
    def read_json(name: str) -> dict:
        return json.loads((directory / name).read_text())

    if not (directory / "after.txt").is_file():
        raise ValueError("missing post-teardown snapshot")
    preflight = (directory / "preflight.txt").read_text()
    clock = re.findall(r"^process_clock_ticks_per_second=(\d+)$", preflight, re.M)
    if len(clock) != 1:
        raise ValueError("missing process tick frequency")
    hz = int(clock[0])
    controls = json.loads((directory / "ids-0.json").read_text())
    candidates = json.loads((directory / "ids-1.json").read_text())
    if not controls or controls != candidates:
        raise ValueError("token traces differ")
    diagnostic = (directory / "diagnostic.log").read_text()
    setup = re.findall(r"capture_us=([\d.]+) instantiate_us=([\d.]+) nodes=(\d+) query_status=0", diagnostic)
    launches = list(map(float, re.findall(r"launch_cpu_us=([\d.]+)", diagnostic)))
    gpu = list(map(float, re.findall(r"replay gpu_elapsed_ms=([\d.]+)", diagnostic)))
    metrics = read_json("diagnostic-response.json")["metrics"]
    low = max(1, metrics["generated_tokens"]-1)
    high = metrics["prompt_tokens"] + metrics["generated_tokens"]
    if len(setup) != 1 or len(launches) != len(gpu) or not low <= len(gpu) <= high:
        raise ValueError("incomplete setup/replay diagnostics")
    capture, instantiate, nodes = setup[0]
    if not all(value > 0 for value in launches + gpu):
        raise ValueError("invalid diagnostic timing")
    timed = [json.loads(line) for line in (directory / "timings.jsonl").read_text().splitlines()]
    phase = "confirm" if (directory / "confirmation.json").exists() else "pair"
    raw = [r for r in timed if r["label"].startswith(phase+"-")]
    n = len(raw) // 2
    if n not in (5, 10, 20):
        raise ValueError("wrong trial count")
    expected = [f"{phase}-{i}-{arm}" for i in range(n) for arm in (i%2, 1-i%2)]
    if [r["label"] for r in raw] != expected:
        raise ValueError("incomplete pairs or order")
    summary = []
    per_token = []
    for arm in range(2):
        trace = read_json(f"trace-response-{arm}.json")
        rows = [r for r in raw if r["arm"] == arm]
        cpu_seconds = []
        for r in rows:
            label = r["label"]
            response = read_json(label+".json")
            m = r["metrics"]
            if response["output"] != trace["output"] or response["metrics"] != m or m["generated_tokens"] != len(controls):
                raise ValueError(f"mismatched request {label}")
            before = cpu_ticks(directory / (label+".cpu-before"))
            after = cpu_ticks(directory / (label+".cpu-after"))
            if before[:2] != after[:2] or after[2] < before[2]:
                raise ValueError(f"invalid CPU interval {label}")
            cpu_seconds.append((after[2]-before[2])/hz)
        pid = cpu_ticks(directory / f"trace-response-{arm}.cpu-after")[0]
        with (directory / "processes.csv").open() as file:
            peaks = [int(r[2].split()[0]) for r in csv.reader(file) if len(r)==3 and r[1].strip()==pid]
        token_ms = [r["metrics"]["decode_elapsed_ms"]/(len(controls)-1) for r in rows]
        per_token.append(token_ms)
        summary.append({"arm":arm,"http_s":median(r["http_seconds"] for r in rows),
                        "http_tokens_s":median(len(controls)/r["http_seconds"] for r in rows),
                        "generation_ttft_ms":median(r["metrics"]["prefill_elapsed_ms"] for r in rows),
                        "decode_ms_per_interval":median(token_ms),"process_cpu_s":median(cpu_seconds),
                        "process_cpu_seconds_by_trial":cpu_seconds,"isolated_trace_peak_mib":max(peaks),
                        "timed_device_samples":request_device_samples(directory, [r["label"] for r in rows])})
    saving = median(a-b for a,b in zip(*per_token, strict=True))
    return {"pairs":n,"ids":len(controls),"arms":summary,"median_paired_decode_saving_ms":saving,
            "capture_us":float(capture),"instantiate_us":float(instantiate),"nodes":int(nodes),
            "diagnostic_replays":len(gpu),"launch_cpu_median_us":median(launches),
            "replay_cuda_event_median_ms":median(gpu),"setup_only_break_even_intervals":
            (float(capture)+float(instantiate))/(1000*saving) if saving>0 else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
