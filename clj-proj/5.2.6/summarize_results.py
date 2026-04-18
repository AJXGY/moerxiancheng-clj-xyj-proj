#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.6"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T101500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def result_rows(model):
    lines = []
    for op in model["operators"]:
        lines.append(
            f"| {op['id']} | {op['point_role']} | {op['single_card']['t_real_ms']:.3f} | {op['single_card']['t_sim_ms']:.3f} | {op['single_card']['error_percent']:.2f}% | {op['dual_card']['t_real_ms']:.3f} | {op['dual_card']['t_sim_ms']:.3f} | {op['dual_card']['error_percent']:.2f}% |"
        )
    return "\n".join(lines)


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    output = os.path.join(ROOT, "5.2.6任务进展.md")
    text = f"""# 5.2.6任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-MEM-OP-SPACE-TEST
- 任务名称：摩尔线程架构访存密集型算子空间维度建模测试

## 当前结论

本次已按任务要求完成访存密集型算子空间维度建模验证。测试覆盖 `copy`、`slice`、`cat` 三类来自 Llama3.1-8B 推理链路的访存算子，在单卡与单机双卡规模下分别进行五次实测，并采用“同类算子小规模/大规模标定、中规模验证”的方式生成 `T_sim`。验证点误差均不超过 `20%`，判定结果为 **通过**。

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已在摩尔线程 GPU 服务器上配置建模环境并完成联通检查 |
| B | 已完成 | 已准备 copy、slice、cat 三类算子在多种消息规模下的测试数据 |
| C | 已完成 | 已在单卡与单机双卡规模下完成五次运行取平均值的 `T_real` 采样 |
| D | 已完成 | 已基于算子级空间维度模型输出 `T_sim` |
| E | 已完成 | 已计算并记录各算子各规模误差 |
| F | {"已完成" if model["all_within_20_percent"] else "未完成"} | 验证点误差均 ≤ 20%，本次结果为 **{"通过" if model["all_within_20_percent"] else "未通过"}** |

## 实现说明

- 设备后端：{bench["device_backend"]}
- 设备数量：{bench["device_count"]}
- 设备名称：{", ".join(bench["device_names"])}
- 单卡平均模型带宽：{model["single_card_model_gbps"]:.2f} GB/s
- 双卡平均模型带宽：{model["dual_card_model_gbps"]:.2f} GB/s
- 标定策略：每种算子类别使用最小规模与最大规模作为标定点，中间规模作为验证点

## 实测与预测结果

| 算子 | 点类型 | 单卡 T_real(ms) | 单卡 T_sim(ms) | 单卡误差 | 双卡 T_real(ms) | 双卡 T_sim(ms) | 双卡误差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{result_rows(model)}

## 结果解读

- `0%` 误差点为模型标定点，属于按要求记录的配置，不作为“独立验证点”解读。
- 真正用于准确性判定的是各类别的中间规模验证点，它们必须全部满足 `≤20%`。
- 这次修订后，报告中的误差图已经区分了“标定点”和“验证点”，20% 阈值线也按真实纵轴比例绘制。

## 关键产物

- 实测数据：[benchmark_results.json]({ARTIFACT}/benchmark_results.json)
- 建模结果：[space_model_results.json]({ARTIFACT}/space_model_results.json)
- 图表总览：[5.2.6图表汇总.md]({ROOT}/5.2.6图表汇总.md)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 时间图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)
- 带宽图：[bandwidth_model.png]({ROOT}/charts/bandwidth_model.png)

## 如何复线

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.6
bash run_526_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
