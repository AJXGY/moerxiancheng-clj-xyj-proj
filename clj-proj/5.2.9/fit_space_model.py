#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.9"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T113500Z")
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


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def solve_alpha_beta(points):
    (s1, t1), (s2, t2) = points
    beta = (t2 - t1) / (s2 - s1)
    alpha = t1 - beta * s1
    return alpha, beta


def build_model(operators, kind):
    kind_ops = [op for op in operators if op["kind"] == kind]
    if len(kind_ops) < 3:
        raise RuntimeError(f"Need at least three points for kind={kind}")
    kind_ops = sorted(kind_ops, key=lambda item: item["bytes"])
    alpha, beta = solve_alpha_beta(
        [
            (kind_ops[0]["bytes"], kind_ops[0]["real"]["avg_ms"]),
            (kind_ops[-1]["bytes"], kind_ops[-1]["real"]["avg_ms"]),
        ]
    )
    return {
        "alpha_ms": alpha,
        "beta_ms_per_byte": beta,
        "calibration_ids": [kind_ops[0]["id"], kind_ops[-1]["id"]],
    }


def predictor_dir(op_id):
    return os.path.join(ARTIFACT, "predictor", op_id)


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


def build_request(op, model, bench):
    return {
        "operator": {
            "id": op["id"],
            "name": op["name"],
            "kind": op["kind"],
            "dtype": op["dtype"],
            "bytes": op["bytes"],
        },
        "parallel_config": {
            "mode": "dual_card",
            "world_size": 2,
            "partition_strategy": "replicated",
        },
        "hardware_topology": {
            "device_backend": bench["device_backend"],
            "device_count": bench["device_count"],
            "device_names": bench["device_names"],
            "physical_devices": [0, 1],
            "communication_path": bench.get("communication_path"),
            "distributed_backend": bench.get("distributed_backend"),
            "calibration_override": {
                "alpha_ms": model["alpha_ms"],
                "beta_ms_per_byte": model["beta_ms_per_byte"],
            },
        },
    }


def estimate_with_tool(op, model, bench):
    out_dir = predictor_dir(op["id"])
    os.makedirs(out_dir, exist_ok=True)
    request = build_request(op, model, bench)
    request_path = os.path.join(out_dir, "request.json")
    dump_json(request_path, request)
    run_operator_predictor(request_path, out_dir)
    report = load_json(os.path.join(out_dir, "report.json"))
    return report["estimate"]["predicted_time_ms"], os.path.join(out_dir, "report.json")


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]

    per_kind = {}
    for kind in sorted({op["kind"] for op in ops}):
        model = build_model(ops, kind)
        per_kind[kind] = {
            **model,
            "description": f"Predictor uses alpha + beta * message_size for {kind}",
        }

    evaluated = []
    for op in ops:
        model = per_kind[op["kind"]]
        pred, report_path = estimate_with_tool(op, model, bench)
        real = op["real"]["avg_ms"]
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "kind": op["kind"],
                "bytes": op["bytes"],
                "point_role": "calibration" if op["id"] in model["calibration_ids"] else "validation",
                "prediction_source": {
                    "tool": "train-infer-estimation-release-2026-04-11/torch_operator_mvp.py",
                    "report": report_path,
                },
                "t_real_ms": real,
                "t_sim_ms": pred,
                "error_percent": error_percent(real, pred),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMM-OP-SPACE-TEST",
        "prediction_source": {
            "tool": "train-infer-estimation-release-2026-04-11/torch_operator_mvp.py",
            "request_fields": ["operator", "parallel_config", "hardware_topology"],
            "calibration_policy": "per-kind two-point calibration passed via alpha_ms + beta_ms_per_byte",
        },
        "model_family": "operator tool with per-kind communication calibration override",
        "per_kind_model": per_kind,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["error_percent"] <= 20.0 for op in evaluated if op["point_role"] == "validation"
        ),
    }

    dump_json(os.path.join(ARTIFACT, "space_model_results.json"), payload)


if __name__ == "__main__":
    main()
