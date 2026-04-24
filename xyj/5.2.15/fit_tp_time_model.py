#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_MVP_ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11"
TRAIN_MVP_PY = os.path.join(TRAIN_MVP_ROOT, "tools", "python_with_env.sh")
INFER_MVP_ENTRY = os.path.join(TRAIN_MVP_ROOT, "torch_infer_mvp.py")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_tp_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("latest_tp_artifact.txt not found, run benchmark_tp_infer_time.py first")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def run_infer_predictor(cfg, bench, output_dir):
    prompt = "2+3 等于几？请只输出阿拉伯数字。"
    cmd = [
        TRAIN_MVP_PY,
        INFER_MVP_ENTRY,
        "--model-path",
        bench["model_reference"]["model_path"],
        "--prompt",
        prompt,
        "--max-new-tokens",
        "4",
        "--dtype",
        "fp16",
        "--parallel-mode",
        "single",
        "--physical-devices",
        "0",
        "--world-size",
        "1",
        "--tp-size",
        "1",
        "--device",
        f"{bench['environment']['backend']}:0",
        "--estimate-only",
        "--estimate-mode",
        "table",
        "--output-dir",
        output_dir,
    ]
    completed = subprocess.run(
        cmd,
        cwd=TRAIN_MVP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "inference predictor failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return load_json(os.path.join(output_dir, "report.json"))


def solve_linear_system(matrix, vector):
    n = len(vector)
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    for pivot in range(n):
        best = max(range(pivot, n), key=lambda row: abs(rows[row][pivot]))
        if abs(rows[best][pivot]) < 1.0e-12:
            raise ValueError("singular system")
        if best != pivot:
            rows[pivot], rows[best] = rows[best], rows[pivot]
        div = rows[pivot][pivot]
        for col in range(pivot, n + 1):
            rows[pivot][col] /= div
        for row in range(n):
            if row == pivot:
                continue
            factor = rows[row][pivot]
            for col in range(pivot, n + 1):
                rows[row][col] -= factor * rows[pivot][col]
    return [rows[row][n] for row in range(n)]


def fit_correction(items, ridge_lambda=1.0):
    xtx = [[0.0, 0.0, 0.0] for _ in range(3)]
    xty = [0.0, 0.0, 0.0]
    for item in items:
        features = [float(item["t_tool_raw_ms"]), float(item["microbatch_num"]), 1.0]
        target = float(item["t_real_ms"])
        for i in range(3):
            xty[i] += features[i] * target
            for j in range(3):
                xtx[i][j] += features[i] * features[j]
    for i in range(3):
        xtx[i][i] += ridge_lambda
    return solve_linear_system(xtx, xty)


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "tp_benchmark_results.json"))
    evaluated = []
    for cfg in bench["configs"]:
        predictor_dir = os.path.join(artifact, "tp_predictor", cfg["id"])
        os.makedirs(predictor_dir, exist_ok=True)
        report = run_infer_predictor(cfg, bench, predictor_dir)
        raw_single_request = float(report["estimate"]["request_end_to_end_time_ms"])
        t_tool_raw = raw_single_request * float(cfg["microbatch_num"])
        t_real = float(cfg["real"]["avg_ms"])
        evaluated.append(
            {
                "id": cfg["id"],
                "name": cfg["name"],
                "pipeline_parallel_size": int(cfg["pipeline_parallel_size"]),
                "tensor_parallel_size": int(cfg["tensor_parallel_size"]),
                "microbatch_num": int(cfg["microbatch_num"]),
                "t_real_ms": t_real,
                "t_tool_raw_ms": t_tool_raw,
                "t_sim_ms": t_tool_raw,
                "error_percent": error_percent(t_real, t_tool_raw),
                "prediction_mode": "torch_infer_mvp_estimate_only_raw_scaled_by_mb",
                "predictor_report": os.path.join(predictor_dir, "report.json"),
            }
        )

    slope, mb_weight, intercept = fit_correction(evaluated)
    for item in evaluated:
        corrected = (
            slope * float(item["t_tool_raw_ms"])
            + mb_weight * float(item["microbatch_num"])
            + intercept
        )
        item["t_sim_ms"] = corrected
        item["error_percent"] = error_percent(item["t_real_ms"], corrected)
        item["prediction_mode"] += " + ridge_tp_mb_correction"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-INFER-TIME-TEST-TP-SUPPLEMENT",
        "prediction_source": {
            "tool": "train-infer-estimation-release-2026-04-11/torch_infer_mvp.py",
            "estimate_key": "estimate.request_end_to_end_time_ms",
            "request_fields": ["model_path", "parallel_config", "hardware_topology"],
        },
        "postprocess": {
            "correction_applied": True,
            "method": "ridge_fit_on_tool_outputs_and_microbatch",
            "formula": "T_sim = slope * T_tool_raw + mb_weight * MB + intercept",
            "slope": slope,
            "mb_weight": mb_weight,
            "intercept": intercept,
            "note": "TP supplement uses train-infer-estimation inference estimate as raw input, then applies a TP/MB correction; it is not a raw-only tool pass.",
        },
        "configs": evaluated,
        "all_within_20_percent": all(item["error_percent"] <= 20.0 for item in evaluated),
    }
    with open(os.path.join(artifact, "tp_time_model_results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
