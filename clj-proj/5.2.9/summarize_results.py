#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.9"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T113500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def status_line(model):
    return "通过" if model["all_within_20_percent"] else "未通过"


def render_result_rows(model):
    lines = []
    for op in model["operators"]:
        lines.append(
            f"| {op['id']} | {op['point_role']} | {op['t_real_ms']:.3f} | {op['t_sim_ms']:.3f} | {op['error_percent']:.2f}% |"
        )
    return "\n".join(lines)


def render_validation_rows(model):
    lines = []
    for op in model["operators"]:
        if op["point_role"] != "validation":
            continue
        lines.append(
            f"| {op['id']} | {op['kind']} | {op['bytes'] // (1024 * 1024)}MB | {op['t_real_ms']:.3f} | {op['t_sim_ms']:.3f} | {op['error_percent']:.2f}% |"
        )
    return "\n".join(lines)


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    output = os.path.join(ROOT, "5.2.9任务进展.md")
    text = f"""# 5.2.9任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-COMM-OP-SPACE-TEST
- 任务名称：摩尔线程架构通信密集型算子空间维度建模测试

## 当前结论

本次在 `MTT S3000` 双卡服务器上完成了通信密集型算子的空间维度建模验证。由于该卡型不属于官方标准 `MCCL` 支持范围，本实现采用了你允许的替代路径：`torch.distributed(gloo) + 双进程 + CPU staging + MUSA 设备缓冲区`，对 `Send/Recv`、`Broadcast` 与 `AllReduce` 做真实五次采样、建模和误差分析；预测时间 `T_sim` 由主分析工具的独立算子级预测入口输出。

最终判定：**{status_line(model)}**。本次采用“64MB/256MB 标定，中间规模验证”的方法，`0%` 误差点仅出现在标定端点，真正用于准确性判定的是中间规模验证点。

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已配置 MUSA 运行环境、双进程 gloo 通信库，并完成服务器联通与双卡可见性验证 |
| B | 已完成 | 已准备 `Send/Recv`、`Broadcast`、`AllReduce` 三类通信算子在多种消息规模下的测试数据 |
| C | 已完成 | 已在单机两卡规模下完成五次运行取平均值，得到 `T_real` |
| D | 已完成 | 已使用主分析工具的算子级空间维度模型对相同配置输出 `T_sim` |
| E | 已完成 | 已计算所有算子的误差值并记录 |
| F | {"已完成" if model["all_within_20_percent"] else "未完成"} | 判定标准为所有验证点误差均 ≤ 20%，本次结果为 **{status_line(model)}** |

## 环境与实现说明

- 设备后端：{bench["device_backend"]}
- 设备数量：{bench["device_count"]}
- 设备名称：{", ".join(bench["device_names"])}
- 分布式后端：{bench["distributed_backend"]}
- 通信路径：{bench["communication_path"]}

## 验证点结果

| 算子 | 类型 | 消息大小 | T_real(ms) | T_sim(ms) | 误差 |
| --- | --- | --- | ---: | ---: | ---: |
{render_validation_rows(model)}

## 全量结果（含标定点）

| 算子 | 点类型 | T_real(ms) | T_sim(ms) | 误差 |
| --- | --- | ---: | ---: | ---: |
{render_result_rows(model)}

## 关键产物

- 实测结果：[benchmark_results.json]({ARTIFACT}/benchmark_results.json)
- 建模结果：[space_model_results.json]({ARTIFACT}/space_model_results.json)
- 图表总览：[5.2.9图表汇总.md]({ROOT}/5.2.9图表汇总.md)
- 任务拓扑图：[topology.png]({ROOT}/charts/topology.png)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 耗时图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)

## 问题与取舍

- 已验证当前 `S3000` 环境无法直接以 `backend="mccl"` 完成 `c10d` 初始化，这部分问题保留在 [probe_mccl.py]({ROOT}/probe_mccl.py) 和 README 中。
- 为保证任务落地，本次采用 `gloo` 作为通信实现，满足“实现通信算子、完成建模与误差验证”的目标。
- 本任务的 `T_sim` 已切换为主分析工具 `train-infer-estimation-release-2026-04-11/torch_operator_mvp.py` 输出。
- 每类通信算子的两端消息规模用于构造工具的 `alpha_ms + beta_ms_per_byte` 标定输入，中间规模作为独立验证点。
- 报告中的 `0%` 误差点属于模型标定端点；主结果表已切换为“仅展示验证点”，全量结果另列。
- `256MB Send/Recv` 在五次采样中出现过一次明显慢样本，说明老内核与当前 runtime 组合下仍存在抖动，但平均值建模误差仍满足阈值要求。
- 如果后续必须换回平台原生 `MCCL`，更合适的环境是官方明确支持的卡型与 runtime 组合。

## 如何复线

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.9
bash run_529_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
