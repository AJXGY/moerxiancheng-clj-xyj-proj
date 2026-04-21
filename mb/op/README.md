# mb/op

`mb/op` 是一个算子级性能预测与实测对比工具，当前已经针对 `MTT S3000` 做过一轮校准。

## 当前状态

- 目标：将主要算子的预测误差压到 `20%` 以内
- 结果：当前 `GEMM`、`SDPA`、`RMSNorm`、`Softmax`、`ADD` 已全部满足该目标
- 适配环境：`torch 2.5.0` + `torch_musa 2.1.1` + `MUSA Runtime 4.2.0` + `muDNN v3000`

## 已知限制

- `MTT S3000` 上 `bfloat16` 的 GEMM 探测会触发 `invalid device function`
- 程序已内置自动回退逻辑：当 `bf16` 探测失败时，会自动切换到 `float16`
- 当前设备架构为 `mp 2.1`，`scaled_dot_product_attention` 不会走 FlashAttention 快路径，因此 `SDPA` 预测使用的是非 FlashAttention 经验模型

## 运行方式

先准备 MUSA 相关动态库路径：

```bash
export LD_LIBRARY_PATH=/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib:/home/o_mabin/.local/mudnn/mudnn/lib:/usr/local/musa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

再运行：

```bash
cd /home/o_mabin/moerxiancheng-clj-xyj-proj/mb/op
python3 main.py
```

默认会优先读取 `device_profiles.yaml`：

- 如果 YAML 中存在当前设备的历史硬件画像，会直接复用，不再重复跑 profiler
- 如果 YAML 不存在，或者设备信息对不上，会自动重新实测并写回 YAML
- 每次跑完指定算子后，最新经验参数会同步写回 YAML

只跑部分算子时可使用：

```bash
python3 main.py -t mm
python3 main.py -t sdpa rmsnorm
```

如果需要忽略缓存、强制重测硬件探针：

```bash
python3 main.py --force-remeasure
```

## YAML 缓存

文件位置：

- `mb/op/device_profiles.yaml`

当前会保存三类信息：

- 设备信息：设备名、能力、SM 数等
- 硬件探针结果：峰值算力、峰值带宽、调度开销、通信参数
- 校准信息：各算子的经验参数

经验参数会在每次实测后根据 `预测值/实测值` 的偏差做一次温和更新，并写回 YAML。后续同一设备再次运行时，会优先使用这套历史参数。

## 最近一次 S3000 校准结果

硬件探针输出：

- 峰值算力：`0.40 TFLOPS`
- 峰值带宽：`363.88 GB/s`
- 调度开销：`112.83 us`

误差结果：

- `GEMM`：`1.2% ~ 4.8%`
- `SDPA`：`1.0% ~ 9.5%`
- `RMSNorm`：`0.1% ~ 9.0%`
- `Softmax`：`2.6% ~ 12.6%`
- `ADD`：`0.1% ~ 3.3%`

## 本轮校准的关键调整

- `hardware_profiler.py`
  - 增加了设备信息采集
  - 增加了 `bf16 -> fp16/fp32` 的 GEMM 探测回退逻辑
  - 修复了不同设备类型下的同步行为

- `bench_core.py`
  - 将原先写死的 `cuda` 行为改为按 `musa/cuda` 自适应

- `predictor.py`
  - 去掉了固定的 `A100`/`108 SM` 假设
  - 为 `mp 2.1` 设备补充了非 FlashAttention 的 `SDPA` 经验模型
  - 重校准了 `RMSNorm`、`Softmax`、`ADD` 的经验参数

## 后续建议

- 如果后续更换 GPU 型号，不建议直接复用当前经验系数，最好重新跑一轮校准
- 如果 `torch_musa` 或 `muDNN` 版本升级，建议重新确认 `bf16 GEMM` 是否仍然需要回退
- 如果后续目标从“误差小于 20%”提高到“误差尽量小于 10%”，优先继续微调 `Softmax` 与 `SDPA`
