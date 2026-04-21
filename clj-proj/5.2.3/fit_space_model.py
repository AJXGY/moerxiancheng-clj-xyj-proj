#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T100500Z")
TOOL_ROOT = os.path.join(
    os.path.dirname(ROOT), "train-infer-estimation-release-2026-04-11"
)
TOOL_ENTRY = os.path.join(TOOL_ROOT, "torch_operator_mvp.py")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def tflops(flops, ms):
    return flops / (ms / 1000.0) / 1e12


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def predictor_dir(op_id, scale):
    return os.path.join(ARTIFACT, "predictor", scale, op_id)


def run_operator_predictor(request_path, output_dir):
    cmd = [
        "python3",
        TOOL_ENTRY,
        "--request-json",
        request_path,
        "--output-dir",
        output_dir,
    ]
    completed = subprocess.run(
        cmd,
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "operator predictor failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def build_request(op, scale, reference_tflops, bench):
    world_size = 1 if scale == "single_card" else 2
    return {
        "operator": {
            "id": op["id"],
            "name": op["name"],
            "kind": op["kind"],
            "llama_component": op["llama_component"],
            "shape": op["shape"],
            "dtype": op["dtype"],
        },
        "parallel_config": {
            "mode": scale,
            "world_size": world_size,
            "partition_strategy": "replicated",
        },
        "hardware_topology": {
            "device_backend": bench["device_backend"],
            "device_count": bench["device_count"],
            "device_names": bench["device_names"],
            "physical_devices": list(range(world_size)),
            "calibration_override": {
                "gemm_tflops": reference_tflops,
                "launch_overhead_ms": 0.0,
            },
        },
    }


def estimate_with_tool(op, scale, reference_tflops, bench):
    out_dir = predictor_dir(op["id"], scale)
    os.makedirs(out_dir, exist_ok=True)
    request = build_request(op, scale, reference_tflops, bench)
    request_path = os.path.join(out_dir, "request.json")
    dump_json(request_path, request)
    run_operator_predictor(request_path, out_dir)
    report = load_json(os.path.join(out_dir, "report.json"))
    return report["estimate"]["predicted_time_ms"], os.path.join(out_dir, "report.json")


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]

    single_tputs = [tflops(op["flops"], op["single_card"]["avg_ms"]) for op in ops]
    dual_tputs = [tflops(op["flops"], op["dual_card"]["effective_avg_ms"]) for op in ops]
    single_model_tput = sum(single_tputs) / len(single_tputs)
    dual_model_tput = sum(dual_tputs) / len(dual_tputs)

    evaluated = []
    for idx, op in enumerate(ops):
        other_ops = [candidate for op_idx, candidate in enumerate(ops) if op_idx != idx]
        single_holdout_tput = sum(
            tflops(candidate["flops"], candidate["single_card"]["avg_ms"])
            for candidate in other_ops
        ) / len(other_ops)
        dual_holdout_tput = sum(
            tflops(candidate["flops"], candidate["dual_card"]["effective_avg_ms"])
            for candidate in other_ops
        ) / len(other_ops)

        single_pred, single_report = estimate_with_tool(
            op, "single_card", single_holdout_tput, bench
        )
        dual_pred, dual_report = estimate_with_tool(
            op, "dual_card", dual_holdout_tput, bench
        )

        single_real = op["single_card"]["avg_ms"]
        dual_real = op["dual_card"]["effective_avg_ms"]
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "shape": op["shape"],
                "flops": op["flops"],
                "point_role": "validation",
                "prediction_source": {
                    "tool": "train-infer-estimation-release-2026-04-11/torch_operator_mvp.py",
                    "single_card_report": single_report,
                    "dual_card_report": dual_report,
                },
                "single_card_reference_tflops": single_holdout_tput,
                "dual_card_reference_tflops": dual_holdout_tput,
                "single_card": {
                    "t_real_ms": single_real,
                    "t_sim_ms": single_pred,
                    "error_percent": error_percent(single_real, single_pred),
                },
                "dual_card": {
                    "t_real_ms": dual_real,
                    "t_sim_ms": dual_pred,
                    "error_percent": error_percent(dual_real, dual_pred),
                },
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMPUTE-OP-SPACE-TEST",
        "prediction_source": {
            "tool": "train-infer-estimation-release-2026-04-11/torch_operator_mvp.py",
            "request_fields": ["operator", "parallel_config", "hardware_topology"],
            "calibration_policy": "leave-one-out throughput reference passed via calibration_override",
        },
        "model_family": "operator tool with leave-one-out throughput calibration override",
        "single_card_model_tflops": single_model_tput,
        "dual_card_model_tflops": dual_model_tput,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["single_card"]["error_percent"] <= 20.0
            and op["dual_card"]["error_percent"] <= 20.0
            for op in evaluated
        ),
    }

    dump_json(os.path.join(ARTIFACT, "space_model_results.json"), payload)


if __name__ == "__main__":
    main()
