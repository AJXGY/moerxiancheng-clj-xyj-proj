# TP 补充总表

- 生成时间：2026-04-21T14:20:00+00:00
- 说明：本表用于汇总当前仓库中基于原始任务主线之外额外补充的张量并行（`TP`）验证结果。所有补充项均为附加证明，不替代原始 `md` 题面要求的主线验收记录。

## 总览

| 任务 | 主线要求 | TP 补充类型 | 当前状态 | 关键产物 |
| --- | --- | --- | --- | --- |
| `5.1.11` | 训练任务处理模型输出测试，主线展示 `DP/PP/Microbatch` | `TP=2` 建模输出补充 | 已完成 | [tp_supplement.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/tp_supplement/tp_supplement.md) |
| `5.2.14` | 训练任务时间维度测试，主线要求 `PP+MB` 实测与预测 | `TP=2` 真实训练时间补充实验 | 已完成 | [5.2.14_TP补充任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/5.2.14_TP补充任务进展.md) |

## 详细说明

### 5.1.11

- 主线仍以 [5.1.11任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/5.1.11任务进展.md) 为准。
- 补充项通过 [build_tp_supplement.py](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/build_tp_supplement.py) 生成一份 `TP=2` 的训练任务处理模型，覆盖：
  - 张量切片
  - AllReduce 通信节点
  - TP 场景下的 Microbatch 执行逻辑
- 关键补充产物：
  - [training_task_config_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/tp_supplement/training_task_config_tp2.json)
  - [training_execution_model_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/tp_supplement/training_execution_model_tp2.json)
  - [validation_report_tp2.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/tp_supplement/validation_report_tp2.json)

### 5.2.14

- 主线仍以 [5.2.14任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/5.2.14任务进展.md) 为准。
- 补充项通过 [run_5214_tp_suite.sh](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/run_5214_tp_suite.sh) 执行真实 `TP=2` 训练时间采样，并调用主训练分析工具输出 `train_iteration_time` 预测值。
- 本次真实补充实验产物目录：
  - [20260421T141521Z](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.14/artifacts/20260421T141521Z)
- 关键补充结果：
  - `cfg_tp2_mb1`: `T_real=98.745 ms`, `T_sim=101.123 ms`, `误差=2.41%`
  - `cfg_tp2_mb2`: `T_real=196.143 ms`, `T_sim=198.525 ms`, `误差=1.21%`
  - `cfg_tp2_mb4`: `T_real=392.258 ms`, `T_sim=392.384 ms`, `误差=0.03%`

## 当前不建议硬补 TP 的任务

- `5.1.5`：主线是推理运行适配性，不是并行建模或并行时间维度测试。
- `5.2.3`：主线是计算密集型算子空间维度建模，关注单卡/双卡算子尺度，不是 `PP/TP` 应用级并行。
- `5.2.6`：主线是访存密集型算子空间维度建模，同样不是应用级 `PP/TP` 任务。
- `5.2.9`：主线是通信算子建模，补充重点应是通信原语与后端实现，而不是额外堆一个 `TP` 应用实验。
