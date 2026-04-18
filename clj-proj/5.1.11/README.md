# 5.1.11 摩尔线程架构上训练任务处理模型输出测试

本目录用于完成 `MTT-TRAIN-MODEL-STRUCT-TEST`。

工程能力包括：

- 读取 Llama3.1-8B 训练任务配置与资源映射
- 构建训练任务处理模型 JSON
- 输出 CPU/GPU 任务分配、并行方式、microbatch 执行逻辑
- 生成 DAG 图、资源拓扑图、阶段划分图和完成情况图表
- 自动汇总 A-H 指标，输出 `5.1.11任务进展.md`

## 快速开始

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11
python3 build_training_model.py
python3 generate_charts.py
python3 summarize_results.py
```

也可以一键执行：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11
bash run_5111_suite.sh
```

## 主要文件

- `training_task_config.json`：训练任务描述
- `resource_mapping.json`：CPU + 摩尔线程 GPU 资源映射
- `build_training_model.py`：训练处理模型构建器
- `generate_charts.py`：生成 SVG 图表
- `summarize_results.py`：生成 `5.1.11任务进展.md`
- `charts/`：可视化图表输出目录
- `artifacts/`：最终产物目录
