# 5.2.14 摩尔线程架构并行配置下训练任务时间维度测试

本目录用于完成 `MTT-PARALLEL-TRAIN-TIME-TEST`。

工程能力包括：

- 基于 `Meta-Llama-3.1-8B` 真实隐藏层/中间层维度的两阶段训练探针
- 单机双卡训练并行配置组合描述
- 不同 `PP/MB` 组合下的训练迭代时间采样
- 时间维度模型拟合与误差统计
- 图表生成与 `5.2.14任务进展.md` 汇总

口径说明：

- 当存在可用 `MUSA/CUDA` 加速器时，脚本会执行训练时间探针实测。
- 当当前环境没有可用加速器时，脚本会退回到合成训练迭代采样，并在报告中明确标注为“模型侧验证”。
- 当前配置重点覆盖 `PP=1/2` 与 `MB=1/2/4`，满足题面对“至少 3 组组合”的要求。
- 此外可选执行 `TP=2` 的补充实验，作为张量并行场景下的附加验证；该补充实验不替代题面主线的 `PP+MB` 验收。
- 当前探针使用 `Meta-Llama-3.1-8B/config.json` 的 `hidden_size=4096` 与 `intermediate_size=14336` 构造两阶段训练微步。
- 该模型配置请求 dtype 为 `bfloat16`，但当前主机的 `MUSA` `bf16` GEMM 不可用，因此实际执行 dtype 自动回退为 `float16` 并在产物中记录。

## 快速开始

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14
bash run_5214_suite.sh
```

## 主要文件

- `parallel_configs.json`：并行配置组合描述
- `benchmark_parallel_train_time.py`：训练迭代时间采样
- `fit_time_model.py`：时间维度模型拟合与误差统计
- `generate_charts.py`：图表生成
- `summarize_results.py`：输出 `5.2.14任务进展.md`
- `artifacts/`：采样与建模产物
- `tp_parallel_configs.json`：TP 补充实验配置
- `benchmark_tp_train_time.py`：TP 补充训练时间采样
- `fit_tp_time_model.py`：TP 补充时间预测
- `summarize_tp_results.py`：输出 `5.2.14_TP补充任务进展.md`
