# 5.2.3 摩尔线程架构计算密集型算子空间维度建模测试

本目录用于完成 `MTT-COMPUTE-OP-SPACE-TEST`。

交付内容包括：

- 计算密集型算子微基准脚本
- 单卡 / 单机双卡实测数据
- 空间维度预测模型与误差统计
- 600 dpi PNG 图表
- `5.2.3任务进展.md`

## 一键执行

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3
bash run_523_suite.sh
```

## 主要文件

- `operator_specs.json`：代表性算子描述
- `benchmark_compute_ops.py`：实测单卡 / 双卡算子耗时
- `fit_space_model.py`：拟合空间维度模型并计算误差
- `generate_charts.py`：生成 600 dpi PNG 图表
- `summarize_results.py`：输出 `5.2.3任务进展.md`
