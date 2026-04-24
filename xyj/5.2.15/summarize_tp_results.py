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
        raise FileNotFoundError("latest_tp_artifact.txt not found")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "tp_benchmark_results.json"))
    model = load_json(os.path.join(artifact, "tp_time_model_results.json"))
    passed = bool(model.get("all_within_20_percent"))
    correction = model.get("postprocess", {})

    lines = [
        "# 5.2.15 TP 补充任务进展",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "- 说明：本文件是 `5.2.15.md` 主线 `PP+MB` 之外的附加 `TP=2` 推理实验记录，不替代主线验收结果。",
        "",
        "## 当前结论",
        "",
        (
            "本次补充实验已完成 `TP=2` 三组微批配置的真实推理型时间采样，并调用 "
            "`train-infer-estimation` 推理分析入口输出原始时间预测。"
        ),
        "",
        (
            "口径说明：当前 TP 补充结果为 `torch_infer_mvp.py` 原始预测输出再叠加 "
            "TP/MB 专项校正后的结果，不应表述为主工具原始裸预测直接达标。"
        ),
        "",
        f"- 判定结果：{'通过' if passed else '未通过'}",
        f"- 设备后端：{bench['environment']['backend']}",
        f"- 设备数量：{bench['environment']['device_count']}",
        "- 采样类型：real_llama_tp_inference_probe",
        "- 张量并行：2",
        "- 采样范围：llama_backbone_forward_with_tp_sharded_head",
        f"- 后处理公式：{correction.get('formula', 'none')}",
        "",
        "## 配置结果明细",
        "",
        "| 配置ID | TP | MB | T_real(ms) | T_tool_raw(ms) | T_sim(ms) | 误差 | 预测口径 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in model["configs"]:
        lines.append(
            "| {id} | {tp} | {mb} | {real:.3f} | {raw:.3f} | {sim:.3f} | {err:.2f}% | {mode} |".format(
                id=item["id"],
                tp=item["tensor_parallel_size"],
                mb=item["microbatch_num"],
                real=item["t_real_ms"],
                raw=item["t_tool_raw_ms"],
                sim=item["t_sim_ms"],
                err=item["error_percent"],
                mode=item["prediction_mode"],
            )
        )
    lines.extend(
        [
            "",
            "## 关键产物",
            "",
            f"- 实测数据：[tp_benchmark_results.json]({artifact}/tp_benchmark_results.json)",
            f"- 模型结果：[tp_time_model_results.json]({artifact}/tp_time_model_results.json)",
            f"- 预测请求目录：[tp_predictor]({artifact}/tp_predictor)",
            "",
            "## 如何复线",
            "",
            "```bash",
            f"cd {ROOT}",
            "bash run_5215_tp_suite.sh",
            "```",
        ]
    )
    with open(os.path.join(ROOT, "5.2.15_TP补充任务进展.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
