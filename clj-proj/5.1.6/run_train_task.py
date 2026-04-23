#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"
SAMPLES_PATH = os.path.join(ROOT, "train_samples.jsonl")
TOOL_ROOT = os.path.join(os.path.dirname(ROOT), "train-infer-estimation-release-2026-04-11")

if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

from mvp_llama_train_runtime import LlamaTrainRuntime, _synchronize  # type: ignore


def detect_backend() -> dict[str, Any]:
    try:
        import torch_musa  # noqa: F401
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            count = int(torch.musa.device_count())
            return {
                "backend": "musa",
                "device_count": count,
                "device_names": [torch.musa.get_device_name(i) for i in range(count)],
            }
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            return {
                "backend": "cuda",
                "device_count": count,
                "device_names": [torch.cuda.get_device_name(i) for i in range(count)],
            }
    except Exception:
        pass

    return {"backend": "cpu", "device_count": 0, "device_names": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real 5.1.6 training task with shared MVP runtime")
    parser.add_argument("--mode", choices=["single", "dual"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--samples-path", default=SAMPLES_PATH)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--microbatch-num", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--split-index", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    return parser.parse_args()


def _module_state_to_cpu(module) -> dict[str, Any]:
    state = {}
    for key, value in module.state_dict().items():
        state[key] = value.detach().cpu()
    return state


def _trainable_modules(runtime: LlamaTrainRuntime) -> list[Any]:
    modules = []
    if hasattr(runtime, "head"):
        modules.append(runtime.head)
    if hasattr(runtime, "tp_heads"):
        modules.append(runtime.tp_heads)
    return modules


def _parameter_norm(modules: list[Any]) -> float:
    total = 0.0
    for module in modules:
        for param in module.parameters():
            total += float(param.detach().float().norm().item())
    return total


def _checkpoint_payload(runtime: LlamaTrainRuntime, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if hasattr(runtime, "head"):
        payload["head"] = _module_state_to_cpu(runtime.head)
    if hasattr(runtime, "tp_heads"):
        payload["tp_heads"] = [_module_state_to_cpu(module) for module in runtime.tp_heads]
    return payload


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = detect_backend()
    required_devices = 2 if args.mode == "dual" else 1
    if env["backend"] == "cpu":
        raise RuntimeError("No MUSA/CUDA accelerator available")
    if int(env["device_count"]) < required_devices:
        raise RuntimeError(
            f"Mode={args.mode} requires {required_devices} devices, found {env['device_count']}"
        )

    pp_size = 2 if args.mode == "dual" else 1
    runtime = LlamaTrainRuntime(
        model_path=args.model_path,
        samples_path=args.samples_path,
        device_backend=env["backend"],
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=1,
        max_seq_len=args.max_seq_len,
        split_index=args.split_index,
        lora_rank=args.lora_rank,
        adapter_only=False,
    )
    modules = _trainable_modules(runtime)

    timings_ms = []
    norm_trace = []
    for step in range(args.steps):
        if pp_size == 1:
            _synchronize(env["backend"], [0])
        else:
            _synchronize(env["backend"], [0, 1])
        started = time.perf_counter()
        runtime.train_iteration(
            microbatch_num=args.microbatch_num,
            global_batch_size=args.global_batch_size,
        )
        if pp_size == 1:
            _synchronize(env["backend"], [0])
        else:
            _synchronize(env["backend"], [0, 1])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings_ms.append(elapsed_ms)
        norm_trace.append(
            {
                "step": step + 1,
                "trainable_parameter_norm": _parameter_norm(modules),
                "elapsed_ms": elapsed_ms,
            }
        )

    checkpoint_path = output_dir / f"{args.mode}_adapter_checkpoint.pt"
    import torch

    torch.save(_checkpoint_payload(runtime, args.mode), checkpoint_path)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "success": True,
        "model_path": os.path.abspath(args.model_path),
        "samples_path": os.path.abspath(args.samples_path),
        "backend": env["backend"],
        "device_count": env["device_count"],
        "device_names": env["device_names"],
        "pipeline_parallel_size": pp_size,
        "microbatch_num": args.microbatch_num,
        "global_batch_size": args.global_batch_size,
        "steps": args.steps,
        "timings_ms": timings_ms,
        "avg_step_ms": sum(timings_ms) / len(timings_ms),
        "checkpoint_path": str(checkpoint_path),
        "parameter_norm_trace": norm_trace,
        "trainable_parameter_count": int(
            sum(param.numel() for module in modules for param in module.parameters())
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
