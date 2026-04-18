#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.9"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T113500Z")


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
    return kind_ops, {
        "alpha_ms": alpha,
        "beta_ms_per_byte": beta,
        "calibration_ids": [kind_ops[0]["id"], kind_ops[-1]["id"]],
    }


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    ops = bench["operators"]

    descriptions = {
        "send_recv": "T_sim = alpha + beta * message_size for send/recv round-trip with CPU staging",
        "broadcast": "T_sim = alpha + beta * message_size for broadcast with CPU staging",
        "all_reduce": "T_sim = alpha + beta * message_size for all_reduce with CPU staging",
    }
    per_kind = {}
    for kind in sorted({op["kind"] for op in ops}):
        _, model = build_model(ops, kind)
        per_kind[kind] = {
            **model,
            "description": descriptions.get(kind, f"T_sim = alpha + beta * message_size for {kind}"),
        }

    evaluated = []
    for op in ops:
        model = per_kind[op["kind"]]
        pred = model["alpha_ms"] + model["beta_ms_per_byte"] * op["bytes"]
        real = op["real"]["avg_ms"]
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "kind": op["kind"],
                "bytes": op["bytes"],
                "point_role": "calibration" if op["id"] in model["calibration_ids"] else "validation",
                "t_real_ms": real,
                "t_sim_ms": pred,
                "error_percent": error_percent(real, pred),
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMM-OP-SPACE-TEST",
        "model_family": "operator-level space model",
        "per_kind_model": per_kind,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["error_percent"] <= 20.0 for op in evaluated if op["point_role"] == "validation"
        ),
    }

    with open(os.path.join(ARTIFACT, "space_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
