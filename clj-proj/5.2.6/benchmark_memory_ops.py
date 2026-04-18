#!/usr/bin/env python3
import json
import multiprocessing as mp
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch_musa  # noqa: F401


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.6"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T101500Z")


def load_specs():
    with open(os.path.join(ROOT, "operator_specs.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name):
    return {"float16": torch.float16, "float32": torch.float32}[name]


def bytes_for_spec(spec):
    dtype_size = 2 if spec["dtype"] == "float16" else 4
    if spec["kind"] in ("copy", "slice"):
        numel = 1
        for d in spec["shape"]:
            numel *= d
        return numel * dtype_size
    if spec["kind"] == "cat":
        numel = 1
        for d in spec["shape_a"]:
            numel *= d
        return numel * dtype_size * 2
    return 0


def run_op(spec, device, scale=1.0):
    torch.musa.set_device(device)
    dtype = dtype_from_name(spec["dtype"])
    if spec["kind"] == "copy":
        shape = list(spec["shape"])
        shape[0] = max(1, int(shape[0] * scale))
        x = torch.randn(shape, device=f"musa:{device}", dtype=dtype)
        y = torch.empty_like(x)
        y.copy_(x)
        return
    if spec["kind"] == "slice":
        shape = list(spec["shape"])
        shape[1] = max(1, int(shape[1] * scale))
        x = torch.randn(shape, device=f"musa:{device}", dtype=dtype)
        _ = x[:, : max(1, shape[1] // 2), :, :]
        return
    if spec["kind"] == "cat":
        shape_a = list(spec["shape_a"])
        shape_b = list(spec["shape_b"])
        shape_a[0] = max(1, int(shape_a[0] * scale))
        shape_b[0] = max(1, int(shape_b[0] * scale))
        a = torch.randn(shape_a, device=f"musa:{device}", dtype=dtype)
        b = torch.randn(shape_b, device=f"musa:{device}", dtype=dtype)
        _ = torch.cat([a, b], dim=spec["dim"])
        return


def bench_one(spec, device, runs=5, warmups=2, scale=1.0):
    for _ in range(warmups):
        run_op(spec, device, scale=scale)
    torch.musa.synchronize(device)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        run_op(spec, device, scale=scale)
        torch.musa.synchronize(device)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)
    return {
        "timings_ms": timings,
        "avg_ms": sum(timings) / len(timings),
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def dual_worker(spec, device, queue):
    res = bench_one(spec, device=device, scale=0.5)
    queue.put({"device": f"musa:{device}", **res})


def dual_bench(spec):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=dual_worker, args=(spec, 0, queue)),
        ctx.Process(target=dual_worker, args=(spec, 1, queue)),
    ]
    wall_start = time.perf_counter()
    for p in procs:
        p.start()
    payloads = [queue.get(), queue.get()]
    for p in procs:
        p.join()
    wall_end = time.perf_counter()
    return {
        "wall_ms": (wall_end - wall_start) * 1000.0,
        "workers": sorted(payloads, key=lambda x: x["device"]),
        "effective_avg_ms": max(p["avg_ms"] for p in payloads),
    }


def main():
    os.makedirs(ARTIFACT, exist_ok=True)
    specs = load_specs()
    ops = []
    for spec in specs:
        single = bench_one(spec, device=0)
        dual = dual_bench(spec)
        ops.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "kind": spec["kind"],
                "llama_component": spec["llama_component"],
                "dtype": spec["dtype"],
                "bytes": bytes_for_spec(spec),
                "single_card": single,
                "dual_card": dual,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-MEM-OP-SPACE-TEST",
        "device_backend": "musa",
        "device_count": torch.musa.device_count(),
        "device_names": [torch.musa.get_device_name(i) for i in range(torch.musa.device_count())],
        "operators": ops,
    }
    with open(os.path.join(ARTIFACT, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
