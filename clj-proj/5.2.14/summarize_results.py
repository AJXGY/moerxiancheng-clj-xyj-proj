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
        raise FileNotFoundError("latest_artifact.txt not found, run benchmark_parallel_train_time.py first")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    artifact = read_latest_artifact()
    bench = load_json(os.path.join(artifact, "benchmark_results.json"))
    model = load_json(os.path.join(artifact, "time_model_results.json"))
    environment = bench.get("environment", {})
    model_reference = bench.get("model_reference", {})
    training_task = bench.get("training_task", {})
    measurement_type = environment.get("mode", "unknown")
    is_synthetic = measurement_type == "synthetic_sample"
    passed = bool(model.get("all_within_20_percent"))
    c_status = "部分完成" if is_synthetic else "已完成"
    c_desc = (
        "当前为合成训练迭代采样（非直接实机实测），已完成至少 3 组并行配置采样"
        if is_synthetic else
        "已完成单机双卡至少 3 组组合的真实训练迭代时间实测（每组五次运行取平均），执行 Llama3.1-8B backbone 前向并更新 LoRA 风格低秩适配器参数"
    )
    conclusion = (
        "当前任务已通过。并行配置实测、训练时间维度预测与误差分析已完成，所有配置误差均控制在 20% 以内。"
        if passed and not is_synthetic
        else "当前任务已完成模型侧验证。并行配置采样、训练时间维度模型预测与误差分析已完成，所有配置误差均控制在 20% 以内；其中 T_real 为合成训练迭代采样。"
        if passed and is_synthetic
        else "当前任务尚未通过。并行配置实测、训练时间维度预测与误差分析已完成，但存在配置误差超过 20%，当前只能算完成了验证流程，不能算指标达标。"
    )
    f_status = "已完成" if passed else "未通过"
    f_desc = "所有配置误差均 ≤ 20%" if passed else "存在配置误差 > 20%，未满足验收阈值"
    postprocess = model.get("postprocess", {})
    correction_applied = bool(postprocess.get("correction_applied"))
    correction_text = (
        f"已对 train-infer-estimation 原始输出应用校正："
        f"T_sim = {postprocess.get('slope', 1.0):.6f} * T_tool_raw + "
        f"{postprocess.get('pp_weight', 0.0):.3f} * PP + {postprocess.get('intercept', 0.0):.3f}"
        if correction_applied
        else "未追加经验校正，T_sim 直接取自 train-infer-estimation 工具输出"
    )

    output = os.path.join(ROOT, "5.2.14任务进展.md")
    text = f"""# 5.2.14任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-PARALLEL-TRAIN-TIME-TEST
- 任务名称：摩尔线程架构并行配置下训练任务时间维度测试

## 当前结论

{conclusion}

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已完成性能建模环境与训练脚本准备 |
| B | 已完成 | 已准备 Llama3.1-8B 训练脚本，支持 PP=1/2 和 MB=1/2 多组并行配置 |
| C | {c_status} | {c_desc} |
| D | 已完成 | 已调用任务响应时间分析工具输出各配置 train_iteration_time 预测值；{correction_text} |
| E | 已完成 | 已计算并记录每组配置误差 |
| F | {f_status} | {f_desc} |

## 关键结果

- 设备后端：{environment.get('backend', 'unknown')}
- 设备数量：{environment.get('device_count', 'unknown')}
- 采样类型：{measurement_type}
- 并行规模：single_node_dual_gpu
- 模型参考：Meta-Llama-3.1-8B，hidden_size={model_reference.get('hidden_size', 'unknown')}，intermediate_size={model_reference.get('intermediate_size', 'unknown')}
- 训练任务：mode={training_task.get('training_mode', 'unknown')}，scope={training_task.get('runtime_scope', 'unknown')}，训练参数={training_task.get('trainable_parameters', 'unknown')}，LoRA rank={training_task.get('lora_rank', 'unknown')}
- dtype：执行 dtype=float16，请求 dtype={model_reference.get('requested_dtype', 'unknown')}
- 预测后处理：{correction_text}
- 判定结果：{"通过" if model['all_within_20_percent'] else "未通过"}

## 配置结果明细

| 配置ID | PP | MB | T_real(ms) | T_tool_raw(ms) | T_sim(ms) | 误差 | 预测模式 |
| --- | --- | --- | --- | --- | --- | --- | --- |
"""
    for item in model["configs"]:
        text += (
            f"| {item['id']} | {item['pipeline_parallel_size']} | {item['microbatch_num']} "
            f"| {item['t_real_ms']:.3f} | {item.get('t_tool_raw_ms', item['t_sim_ms']):.3f} | {item['t_sim_ms']:.3f} | {item['error_percent']:.2f}% | {item.get('prediction_mode', 'online_runtime_probe')} |\n"
        )

    text += f"""

## 关键产物

- 实测数据：[benchmark_results.json]({artifact}/benchmark_results.json)
- 模型结果：[time_model_results.json]({artifact}/time_model_results.json)
- 训练预测工具入口：[{model['prediction_source']['tool']}]({os.path.join(os.path.dirname(ROOT), 'train-infer-estimation-release-2026-04-11', 'torch_train_mvp.py')})
- 图表汇总：[5.2.14图表汇总.md]({ROOT}/5.2.14图表汇总.md)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 时间图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)

## 如何复线

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14
bash run_5214_suite.sh
```
"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
