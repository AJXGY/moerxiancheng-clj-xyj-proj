#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("latest_artifact.txt not found, run benchmark_parallel_infer_time.py first")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def solve_linear_system_3x3(a, b):
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(3):
        pivot = col
        for r in range(col + 1, 3):
            if abs(m[r][col]) > abs(m[pivot][col]):
                pivot = r
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix while fitting model")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        div = m[col][col]
        for c in range(col, 4):
            m[col][c] /= div

        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for c in range(col, 4):
                m[r][c] -= factor * m[col][c]

    return [m[i][3] for i in range(3)]


def fit_coefficients(configs):
    # Linear model: T_sim = alpha + beta * pp + gamma * (1 / microbatch)
    xtx = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    xty = [0.0, 0.0, 0.0]

    for cfg in configs:
        x = [1.0, float(cfg["pipeline_parallel_size"]), 1.0 / float(cfg["microbatch_num"])]
        y = float(cfg["real"]["avg_ms"])
        for i in range(3):
            xty[i] += x[i] * y
            for j in range(3):
                xtx[i][j] += x[i] * x[j]

    alpha, beta, gamma = solve_linear_system_3x3(xtx, xty)
    return alpha, beta, gamma


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "benchmark_results.json"))
    configs = bench["configs"]

    alpha, beta, gamma = fit_coefficients(configs)

    evaluated = []
    for cfg in configs:
        pp = float(cfg["pipeline_parallel_size"])
        inv_mb = 1.0 / float(cfg["microbatch_num"])
        t_real = float(cfg["real"]["avg_ms"])
        t_sim = alpha + beta * pp + gamma * inv_mb
        evaluated.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "pipeline_parallel_size": cfg["pipeline_parallel_size"],
            "microbatch_num": cfg["microbatch_num"],
            "t_real_ms": t_real,
            "t_sim_ms": t_sim,
            "error_percent": error_percent(t_real, t_sim)
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-INFER-TIME-TEST",
        "model": {
            "formula": "T_sim = alpha + beta * pipeline_parallel_size + gamma * (1/microbatch_num)",
            "alpha_ms": alpha,
            "beta_ms_per_pp": beta,
            "gamma_ms": gamma
        },
        "configs": evaluated,
        "all_within_20_percent": all(item["error_percent"] <= 20.0 for item in evaluated)
    }

    with open(os.path.join(artifact, "time_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
