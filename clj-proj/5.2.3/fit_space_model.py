#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T100500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tflops(flops, ms):
    return flops / (ms / 1000.0) / 1e12


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


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
        single_real = op["single_card"]["avg_ms"]
        dual_real = op["dual_card"]["effective_avg_ms"]
        single_pred = op["flops"] / (single_holdout_tput * 1e12) * 1000.0
        dual_pred = op["flops"] / (dual_holdout_tput * 1e12) * 1000.0
        evaluated.append(
            {
                "id": op["id"],
                "name": op["name"],
                "shape": op["shape"],
                "flops": op["flops"],
                "point_role": "validation",
                "single_card_reference_tflops": single_holdout_tput,
                "dual_card_reference_tflops": dual_holdout_tput,
                "single_card": {
                    "t_real_ms": single_real,
                    "t_sim_ms": single_pred,
                    "error_percent": error_percent(single_real, single_pred)
                },
                "dual_card": {
                    "t_real_ms": dual_real,
                    "t_sim_ms": dual_pred,
                    "error_percent": error_percent(dual_real, dual_pred)
                }
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-COMPUTE-OP-SPACE-TEST",
        "model_family": "leave-one-out throughput validation model",
        "single_card_model_tflops": single_model_tput,
        "dual_card_model_tflops": dual_model_tput,
        "operators": evaluated,
        "all_within_20_percent": all(
            op["single_card"]["error_percent"] <= 20.0 and op["dual_card"]["error_percent"] <= 20.0
            for op in evaluated
        )
    }

    with open(os.path.join(ARTIFACT, "space_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
