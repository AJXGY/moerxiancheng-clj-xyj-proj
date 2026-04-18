#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.6"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T101500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def solve_alpha_beta(points):
    (s1, t1), (s2, t2) = points
    beta = (t2 - t1) / (s2 - s1)
    alpha = t1 - beta * s1
    return alpha, beta


def build_kind_model(ops, key):
    kind_ops = sorted((op for op in ops if op["kind"] == key), key=lambda item: item["bytes"])
    if len(kind_ops) < 3:
        raise RuntimeError(f"Need at least three points for kind={key}")
    single_alpha, single_beta = solve_alpha_beta(
        [
            (kind_ops[0]["bytes"], kind_ops[0]["single_card"]["avg_ms"]),
            (kind_ops[-1]["bytes"], kind_ops[-1]["single_card"]["avg_ms"]),
        ]
    )
    dual_alpha, dual_beta = solve_alpha_beta(
        [
            (kind_ops[0]["bytes"], kind_ops[0]["dual_card"]["effective_avg_ms"]),
            (kind_ops[-1]["bytes"], kind_ops[-1]["dual_card"]["effective_avg_ms"]),
        ]
    )
    calibration_ids = {kind_ops[0]["id"], kind_ops[-1]["id"]}
    return kind_ops, {
        "single_card": {
            "alpha_ms": single_alpha,
            "beta_ms_per_byte": single_beta,
        },
        "dual_card": {
            "alpha_ms": dual_alpha,
            "beta_ms_per_byte": dual_beta,
        },
        "calibration_ids": sorted(calibration_ids),
    }


def predict(model, bytes_count, scale):
    branch = model[scale]
    return branch["alpha_ms"] + branch["beta_ms_per_byte"] * bytes_count


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]
    kinds = sorted({op["kind"] for op in ops})
    per_kind = {}
    for kind in kinds:
        _, model = build_kind_model(ops, kind)
        per_kind[kind] = model

    all_single_betas = [per_kind[k]["single_card"]["beta_ms_per_byte"] for k in kinds]
    all_dual_betas = [per_kind[k]["dual_card"]["beta_ms_per_byte"] for k in kinds]

    evaluated = []
    for op in sorted(ops, key=lambda item: (item["kind"], item["bytes"])):
        model = per_kind[op["kind"]]
        point_role = "calibration" if op["id"] in model["calibration_ids"] else "validation"
        single_real = op["single_card"]["avg_ms"]
        dual_real = op["dual_card"]["effective_avg_ms"]
        single_pred = predict(model, op["bytes"], "single_card")
        dual_pred = predict(model, op["bytes"], "dual_card")
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "kind": op["kind"],
                "bytes": op["bytes"],
                "point_role": point_role,
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
        "task_id": "MTT-MEM-OP-SPACE-TEST",
        "model_family": "operator-level space model",
        "single_card_model_gbps": sum(1.0 / b for b in all_single_betas) / len(all_single_betas) / 1e6,
        "dual_card_model_gbps": sum(1.0 / b for b in all_dual_betas) / len(all_dual_betas) / 1e6,
        "per_kind_model": per_kind,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["single_card"]["error_percent"] <= 20.0 and op["dual_card"]["error_percent"] <= 20.0
            for op in evaluated
            if op["point_role"] == "validation"
        ),
    }
    with open(os.path.join(ARTIFACT, "space_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
