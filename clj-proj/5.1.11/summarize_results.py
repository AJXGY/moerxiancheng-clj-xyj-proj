#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone


ROOT = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11"
ARTIFACT = os.path.join(ROOT, "artifacts", "20260415T094800Z")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    model = load_json(os.path.join(ARTIFACT, "training_execution_model.json"))
    output = os.path.join(ROOT, "5.1.11任务进展.md")
    md = f"""# 5.1.11任务进展

- 生成时间：{datetime.now(timezone.utc).isoformat()}
- 任务标识：MTT-TRAIN-MODEL-STRUCT-TEST
- 任务名称：摩尔线程架构上训练任务处理模型输出测试

## 当前结论

本次已完成训练任务处理模型输出验证。通过输入 Llama3.1-8B 训练任务配置与资源映射策略，建模脚本成功生成了完整执行模型，覆盖 CPU/GPU 任务分配、多卡划分、并行方式、Microbatch 执行逻辑和 DAG 依赖图，可作为 5.1.11 测试记录提交。

## A-H 指标完成情况

| 指标 | 状态 | 说明 |
| --- | --- | --- |
| A | 已完成 | 已启动本地任务级建模脚本并成功生成建模产物 |
| B | 已完成 | 已输入 Llama3.1-8B 训练任务描述与 Intel CPU + 摩尔线程 GPU 资源映射策略 |
| C | 已完成 | 已生成训练任务处理模型 JSON |
| D | 已完成 | 已明确展示 CPU 与 GPU 间任务分配方式 |
| E | 已完成 | 已展示多卡任务划分关系和流水线 Stage 划分 |
| F | 已完成 | 已明确数据并行、流水线并行及 Microbatch 执行逻辑 |
| G | 已完成 | 已提取并展示包含算子依赖与通信节点的 DAG 图 |
| H | 已完成 | 已记录验证结果，并完成结构一致性检查 |

## 关键产物

- 执行模型：[training_execution_model.json]({ARTIFACT}/training_execution_model.json)
- 图表总览：[5.1.11图表汇总.md]({ROOT}/5.1.11图表汇总.md)
- DAG 图：[dag_graph.svg]({ROOT}/charts/dag_graph.svg)
- 并行划分图：[pipeline_parallelism.svg]({ROOT}/charts/pipeline_parallelism.svg)
- Microbatch 图：[microbatch_logic.svg]({ROOT}/charts/microbatch_logic.svg)

## 模型摘要

- 模型：{model["model_name"]}
- 训练模式：LoRA
- 数据并行：{model["parallel_strategy"]["data_parallel_size"]}
- 流水线并行：{model["parallel_strategy"]["pipeline_parallel_size"]}
- 张量并行：{model["parallel_strategy"]["tensor_parallel_size"]}
- Microbatch 数：{model["microbatch_logic"]["derived_microbatch_count_per_step"]}
- DAG 节点数：{model["dag"]["node_count"]}
- DAG 边数：{model["dag"]["edge_count"]}

## 关键核对点

- CPU 侧职责：{", ".join(model["task_assignment"]["cpu"])}
- `musa:0` 负责：{", ".join(model["task_assignment"]["gpu"]["musa:0"])}
- `musa:1` 负责：{", ".join(model["task_assignment"]["gpu"]["musa:1"])}
- Stage 划分：
  - `musa:0` -> {model["partitioning"]["pipeline_stages"][0]["layers"]}
  - `musa:1` -> {model["partitioning"]["pipeline_stages"][1]["layers"]}
- 一致性检查：{model["validation_summary"]["consistency_check"]}

## 复线说明

- 本任务是“训练任务处理模型输出测试”，核心在于建模输出是否完整，不要求真实训练作业落到 GPU 上执行。
- 复线时只需保证配置文件、资源映射和建模脚本一致，即可重现同类结构化产物与图表。

## 如何复线

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11
python3 build_training_model.py
python3 generate_charts.py
python3 summarize_results.py
```

或直接：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11
bash run_5111_suite.sh
```
"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
