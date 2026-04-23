# 5.2.15 摩尔线程架构并行配置下推理任务时间维度测试

本目录用于完成 `MTT-PARALLEL-INFER-TIME-TEST`。

交付内容包括：

- 单机双卡并行配置推理延迟基准脚本
- 不同流水线并行路数与微批次配置实测数据
- 时间维度模型拟合与误差统计
- 600 dpi PNG 图表
- `5.2.15任务进展.md`

## 一键执行

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/xyj/5.2.15
bash run_5215_suite.sh
```

单卡稳定测量模式：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/xyj/5.2.15
bash run_5215_suite.sh --single-only
```

## 一键实测

如需进行真实推理运行采集并把证据归档到 5.2.15：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/xyj/5.2.15
bash run_5215_real_suite.sh \
	--model-path /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B \
	--device-type musa \
	--single-device-ids 0 \
	--dual-device-ids 0,1
```

单卡实测模式：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/xyj/5.2.15
bash run_5215_real_suite.sh \
	--model-path /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B \
	--device-type musa \
	--single-device-ids 0 \
	--dual-device-ids 0,1 \
	--single-only
```

可选参数：

- `--dry-run`：仅做冒烟验证，不加载真实模型
- `--skip-model-build`：跳过 5.1.12 建模步骤，仅采集运行证据

## 主要文件

- `parallel_configs.json`：并行配置组合描述
- `benchmark_parallel_infer_time.py`：并行配置推理延迟基准
- `fit_time_model.py`：时间维度模型拟合与误差统计
- `generate_charts.py`：生成 600 dpi PNG 图表
- `summarize_results.py`：输出 `5.2.15任务进展.md`
- `run_5215_real_suite.sh`：一键实测并输出 `5.2.15实测结果.md`
