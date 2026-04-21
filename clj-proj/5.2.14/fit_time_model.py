#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_MVP_ROOT = os.path.join(
    os.path.dirname(ROOT), "train-infer-estimation-release-2026-04-11"
)
TRAIN_MVP_PY = os.path.join(TRAIN_MVP_ROOT, "tools", "python_with_env.sh")
TRAIN_MVP_ENTRY = os.path.join(TRAIN_MVP_ROOT, "torch_train_mvp.py")
DEFAULT_MODEL = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "latest_artifact.txt not found, run benchmark_parallel_train_time.py first"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def error_percent(real_ms, pred_ms):
    return abs(real_ms - pred_ms) / real_ms * 100.0


def load_model_config(model_path):
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_description(bench):
    model_reference = bench.get("model_reference", {})
    training_task = bench.get("training_task", {})
    model_path = model_reference.get("model_path", bench.get("model_path", DEFAULT_MODEL))
    model_cfg = load_model_config(model_path)
    execution_dtype = "float16"
    return {
        "name": "Meta-Llama-3.1-8B backbone training task",
        "train_workload": "llama_backbone_probe",
        "model_path": model_path,
        "train_samples_path": training_task.get("train_samples_path"),
        "max_seq_len": int(training_task.get("max_seq_len", 8)),
        "pipeline_split_index": int(training_task.get("pipeline_split_index", 16)),
        "dtype": execution_dtype,
        "hidden_size": int(model_cfg["hidden_size"]),
        "stage0_out_features": int(model_cfg["intermediate_size"]),
        "stage1_out_features": int(model_cfg["hidden_size"]),
        "sequence_hidden_tokens": int(training_task.get("max_seq_len", 8)),
        "description": (
            "Real Llama3.1-8B backbone training task with a lightweight classification head. "
            "PP=1 runs the full backbone on one device; PP=2 splits 32 decoder layers into "
            "two stages (16+16) and transfers activations through CPU staging between devices."
        ),
        "llama_reference": {
            "model_name": "Meta-Llama-3.1-8B",
            "num_hidden_layers": int(model_cfg["num_hidden_layers"]),
            "num_attention_heads": int(model_cfg["num_attention_heads"]),
            "num_key_value_heads": int(model_cfg["num_key_value_heads"]),
            "requested_dtype": str(model_cfg.get("torch_dtype") or "float16"),
            "execution_dtype": execution_dtype,
        },
    }


def build_hardware_topology(environment, cfg):
    device_count = int(environment.get("device_count", 0))
    physical_devices = list(range(max(1, min(device_count, cfg["pipeline_parallel_size"]))))
    return {
        "device_backend": environment.get("backend", "cpu"),
        "device_names": environment.get("device_names", []),
        "device_count": device_count,
        "physical_devices": physical_devices,
        "world_size": max(1, cfg["pipeline_parallel_size"]),
        "tp_size": 1,
        "topology": environment.get("topology", "local"),
        "interconnect": "cpu_staging",
        "nnodes": 1,
    }


def run_training_predictor(request_path, output_dir, device_backend):
    cmd = [
        TRAIN_MVP_PY,
        TRAIN_MVP_ENTRY,
        "--request-json",
        request_path,
        "--output-dir",
        output_dir,
        "--device",
        f"{device_backend}:0" if device_backend != "cpu" else "cpu:0",
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
            "training predictor failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "benchmark_results.json"))
    configs = bench["configs"]
    environment = bench["environment"]

    evaluated = []
    for cfg in configs:
        request = {
            "model": build_model_description(bench),
            "parallel_config": {
                "pipeline_parallel_size": int(cfg["pipeline_parallel_size"]),
                "microbatch_num": int(cfg["microbatch_num"]),
                "global_batch_size": int(cfg["global_batch_size"]),
                "dtype": cfg.get("dtype", "float16"),
            },
            "hardware_topology": build_hardware_topology(environment, cfg),
        }
        predictor_dir = os.path.join(artifact, "predictor", cfg["id"])
        os.makedirs(predictor_dir, exist_ok=True)
        request_path = os.path.join(predictor_dir, "request.json")
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request, f, ensure_ascii=False, indent=2)

        run_training_predictor(
            request_path=request_path,
            output_dir=predictor_dir,
            device_backend=environment.get("backend", "cpu"),
        )
        predictor_report = load_json(os.path.join(predictor_dir, "report.json"))
        t_sim = float(predictor_report["estimate"]["train_iteration_time_ms"])
        t_real = float(cfg["real"]["avg_ms"])
        evaluated.append(
            {
                "id": cfg["id"],
                "name": cfg["name"],
                "pipeline_parallel_size": cfg["pipeline_parallel_size"],
                "microbatch_num": cfg["microbatch_num"],
                "t_real_ms": t_real,
                "t_sim_ms": t_sim,
                "error_percent": error_percent(t_real, t_sim),
                "predictor_report": os.path.join(predictor_dir, "report.json"),
                "predictor_request": request_path,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": "MTT-PARALLEL-TRAIN-TIME-TEST",
        "prediction_source": {
            "tool": "train-infer-estimation-release-2026-04-11/torch_train_mvp.py",
            "request_fields": ["model", "parallel_config", "hardware_topology"],
        },
        "configs": evaluated,
        "all_within_20_percent": all(item["error_percent"] <= 20.0 for item in evaluated),
    }

    with open(os.path.join(artifact, "time_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
