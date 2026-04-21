from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def flops_for_matmul(shape: dict[str, int], world_size: int, partition_strategy: str) -> float:
    m = int(shape["m"])
    k = int(shape["k"])
    n = int(shape["n"])
    if partition_strategy == "sharded_local" and world_size > 1:
        m = max(1, m // world_size)
    return float(2 * m * k * n)


def estimate_operator_time_ms(request: dict) -> dict:
    operator = request["operator"]
    parallel = request["parallel_config"]
    topology = request["hardware_topology"]
    calibration = topology["calibration_override"]

    op_kind = str(operator.get("kind", "")).lower()
    world_size = int(parallel.get("world_size", 1))
    partition_strategy = str(parallel.get("partition_strategy", "replicated"))

    if op_kind == "matmul":
        flops = flops_for_matmul(operator["shape"], world_size, partition_strategy)
        gemm_tflops = float(calibration["gemm_tflops"])
        launch_overhead_ms = float(calibration.get("launch_overhead_ms", 0.0))
        predicted_ms = flops / (gemm_tflops * 1.0e12) * 1.0e3 + launch_overhead_ms
        return {
            "operator_kind": op_kind,
            "flops": flops,
            "world_size": world_size,
            "partition_strategy": partition_strategy,
            "gemm_tflops": gemm_tflops,
            "launch_overhead_ms": launch_overhead_ms,
            "predicted_time_ms": predicted_ms,
        }

    if op_kind in {"copy", "slice", "cat"}:
        bytes_count = int(operator["bytes"])
        bandwidth_gbps = float(calibration["memory_bandwidth_gbps"])
        alpha_ms = float(calibration.get("alpha_ms", 0.0))
        predicted_ms = alpha_ms + bytes_count / (bandwidth_gbps * 1.0e9) * 1.0e3
        return {
            "operator_kind": op_kind,
            "bytes": bytes_count,
            "world_size": world_size,
            "partition_strategy": partition_strategy,
            "memory_bandwidth_gbps": bandwidth_gbps,
            "alpha_ms": alpha_ms,
            "predicted_time_ms": predicted_ms,
        }

    if op_kind in {"send_recv", "broadcast", "all_reduce"}:
        bytes_count = int(operator["bytes"])
        beta_ms_per_byte = float(calibration["beta_ms_per_byte"])
        alpha_ms = float(calibration.get("alpha_ms", 0.0))
        predicted_ms = alpha_ms + beta_ms_per_byte * bytes_count
        return {
            "operator_kind": op_kind,
            "bytes": bytes_count,
            "world_size": world_size,
            "partition_strategy": partition_strategy,
            "alpha_ms": alpha_ms,
            "beta_ms_per_byte": beta_ms_per_byte,
            "predicted_time_ms": predicted_ms,
        }

    raise ValueError(f"unsupported operator kind: {op_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = load_json(args.request_json)
    estimate = estimate_operator_time_ms(request)

    report = {
        "created_at": _iso_now(),
        "request_path": str(Path(args.request_json).resolve()),
        "tool": "train-infer-estimation-release-2026-04-11/mvp_operator_app.py",
        "request": request,
        "estimate": estimate,
    }
    dump_json(Path(args.output_dir) / "report.json", report)


if __name__ == "__main__":
    main()
