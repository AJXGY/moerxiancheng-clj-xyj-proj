#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_latest_artifact():
    path = os.path.join(ROOT, "latest_tp_artifact.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "latest_tp_artifact.txt not found, run benchmark_tp_train_time.py first"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "tp_benchmark_results.json"))
    model = load_json(os.path.join(artifact, "tp_time_model_results.json"))
    environment = bench.get("environment", {})
    training_task = bench.get("training_task", {})
    model_reference = bench.get("model_reference", {})
    unstable_configs = []
    for cfg in bench.get("configs", []):
        timings = [float(value) for value in cfg.get("real", {}).get("timings_ms", [])]
        if timings and min(timings) > 0 and max(timings) / min(timings) > 10.0:
            unstable_configs.append(cfg["id"])
    stability_text = (
        "存在短 kernel 采样抖动，本补充结果仅证明 TP 路径可运行，不作为稳定 20% 泛化判定。"
        if unstable_configs
        else "三组 TP 补充配置采样稳定，可作为补充稳定性记录。"
    )
    output = os.path.join(ROOT, "5.2.14_TP补充任务进展.md")

    text = f"""# 5.2.14 TP 补充任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 说明：本文件是 `5.2.14.md` 主线 `PP+MB` 之外的附加 `TP=2` 实验记录，不替代主线验收结果。

## 当前结论

本次补充实验已在双卡环境下对 `TP=2`、`MB=2/4/6` 三组配置执行真实 LoRA adapter-step 训练时间采样，并调用主训练分析工具生成 `train_iteration_time` 预测值，用于补充展示张量并行场景下的时间建模能力。`MB=1/8` 在当前 MUSA 短 kernel 场景下存在启动抖动，未作为最终补充配置。

稳定性说明：{stability_text}

## 关键结果

- 设备后端：{environment.get('backend', 'unknown')}
- 设备数量：{environment.get('device_count', 'unknown')}
- 采样类型：{environment.get('mode', 'unknown')}
- 模型参考：Meta-Llama-3.1-8B，hidden_size={model_reference.get('hidden_size', 'unknown')}
- 张量并行：{training_task.get('tensor_parallel_size', 'unknown')}
- 训练模式：{training_task.get('training_mode', 'unknown')}
- 采样范围：{training_task.get('runtime_scope', 'unknown')}
- 训练参数：{training_task.get('trainable_parameters', 'unknown')}，LoRA rank={training_task.get('lora_rank', 'unknown')}
- 误差判定：{"通过" if model.get("all_within_20_percent") else "未通过"}
- 稳定性判定：{"需谨慎" if unstable_configs else "稳定"}
- 抖动配置：{", ".join(unstable_configs) if unstable_configs else "无"}

## 配置结果明细

| 配置ID | TP | MB | T_real(ms) | T_sim(ms) | 误差 |
| --- | --- | --- | --- | --- | --- |
"""
    for item in model["configs"]:
        text += (
            f"| {item['id']} | {item['tensor_parallel_size']} | {item['microbatch_num']} "
            f"| {item['t_real_ms']:.3f} | {item['t_sim_ms']:.3f} | {item['error_percent']:.2f}% |\n"
        )

    text += f"""

## 关键产物

- 实测数据：[tp_benchmark_results.json]({artifact}/tp_benchmark_results.json)
- 模型结果：[tp_time_model_results.json]({artifact}/tp_time_model_results.json)
"""
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
