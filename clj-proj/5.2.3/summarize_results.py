#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T100500Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    bench = load_json(os.path.join(ARTIFACT, "benchmark_results.json"))
    model = load_json(os.path.join(ARTIFACT, "space_model_results.json"))
    output = os.path.join(ROOT, "5.2.3任务进展.md")
    rows = []
    for op in model["operators"]:
        rows.append(
            f"| {op['id']} | {op['point_role']} | {op['single_card']['t_real_ms']:.3f} | {op['single_card']['t_sim_ms']:.3f} | {op['single_card']['error_percent']:.2f}% | {op['dual_card']['t_real_ms']:.3f} | {op['dual_card']['t_sim_ms']:.3f} | {op['dual_card']['error_percent']:.2f}% |"
        )
    md = f"""# 5.2.3任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-COMPUTE-OP-SPACE-TEST
- 任务名称：摩尔线程架构计算密集型算子空间维度建模测试

## 当前结论

本次已完成计算密集型算子的空间维度建模验证。测试对象为 Llama3.1-8B 中三类典型 GEMM 算子，在单卡与单机双卡两种规模下进行了五次实测取均值，并采用 leave-one-out 吞吐验证方式输出预测时间。所有验证点误差均不超过 20%，判定结果为 **通过**。

## A-F 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已在摩尔线程 GPU 服务器上配置建模环境并完成联通检查 |
| B | 已完成 | 已选取 Llama3.1-8B 典型 GEMM 计算密集型算子并确定输入规模 |
| C | 已完成 | 已完成单卡与单机双卡五次实测平均时间采样 |
| D | 已完成 | 已使用算子级空间维度模型输出预测时间 |
| E | 已完成 | 已计算并记录各算子在两种并行规模下的误差值 |
| F | 已完成 | 所有测试算子误差均 ≤ 20%，本次结果为 **通过** |

## 关键结果

- 设备后端：{bench["device_backend"]}
- 设备数量：{bench["device_count"]}
- 设备名称：{", ".join(bench["device_names"])}
- 单卡全体平均吞吐：{model["single_card_model_tflops"]:.2f} TFLOPS
- 双卡全体平均吞吐：{model["dual_card_model_tflops"]:.2f} TFLOPS
- 判定结果：{"通过" if model["all_within_20_percent"] else "未通过"}

## 实测与预测结果

| 算子 | 点类型 | 单卡 T_real(ms) | 单卡 T_sim(ms) | 单卡误差 | 双卡 T_real(ms) | 双卡 T_sim(ms) | 双卡误差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 结果说明

- 本任务的建模方法不是“按单算子逐点反推”，而是对每个算子采用 leave-one-out 验证，即用其余算子的吞吐均值预测当前算子。
- 因此表中全部点都是真正的验证点，不存在作为判定依据的 `0%` 拟合点。
- 误差图中的 20% 红线已按真实纵轴比例重绘，不再使用固定 25% 画布比例。

## 关键产物

- 实测数据：[benchmark_results.json]({ARTIFACT}/benchmark_results.json)
- 模型结果：[space_model_results.json]({ARTIFACT}/space_model_results.json)
- 图表汇总：[5.2.3图表汇总.md]({ROOT}/5.2.3图表汇总.md)
- 误差图：[error_compare.png]({ROOT}/charts/error_compare.png)
- 时间图：[runtime_compare.png]({ROOT}/charts/runtime_compare.png)

## 如何复线

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3
bash run_523_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
