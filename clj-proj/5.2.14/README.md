# 5.2.14 摩尔线程架构并行配置下训练任务时间维度测试

本目录用于完成 `MTT-PARALLEL-TRAIN-TIME-TEST`。

工程能力包括：

- 基于 `Meta-Llama-3.1-8B` 真实 backbone 前向的 LoRA 风格低秩适配器训练
- 单机双卡训练并行配置组合描述
- 不同 `PP/MB` 组合下的训练迭代时间采样
- 时间维度模型拟合与误差统计
- 图表生成与 `5.2.14任务进展.md` 汇总

口径说明：

- 当存在可用 `MUSA/CUDA` 加速器时，脚本会执行训练时间探针实测。
- 当当前环境没有可用加速器时，脚本会退回到合成训练迭代采样，并在报告中明确标注为“模型侧验证”。
- 当前配置重点覆盖 `PP=1/2` 与 `MB=1/2/4`，满足题面对“至少 3 组组合”的要求。
- 此外可选执行 `TP=2` 的补充实验，作为张量并行场景下的附加验证；该补充实验不替代题面主线的 `PP+MB` 验收。
- 主线与补充线的训练执行和预测入口都统一挂在 `train-infer-estimation-release-2026-04-11/` 目录下，其中训练 runtime 复用 `mvp_llama_train_runtime.py`，预测入口复用 `torch_train_mvp.py`。
- 当前训练脚本不执行全量 8B 参数更新；它按 Llama3.1-8B `hidden_size=4096` 构造同形状 hidden features，并在 MUSA 上计量 LoRA adapter-step 的 A/B 低秩适配器更新，避免全量微调和重复 8B backbone 前向带来的长运行时间与显存压力。
- 预测阶段复用同一轮实测中的 primitive profile，避免每个配置重复加载 8B 模型；仍通过主训练分析工具生成 `train_iteration_time`。
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
- `train-infer-estimation-release-2026-04-11/mvp_llama_train_runtime.py`：共享训练 runtime
- `train-infer-estimation-release-2026-04-11/torch_train_mvp.py`：共享训练时间预测入口
- `tp_parallel_configs.json`：TP 补充实验配置
- `benchmark_tp_train_time.py`：TP 补充训练时间采样
- `fit_tp_time_model.py`：TP 补充时间预测
- `summarize_tp_results.py`：输出 `5.2.14_TP补充任务进展.md`
