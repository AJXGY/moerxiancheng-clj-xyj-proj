# AI环境配置任务 Prompt

下面这段可以直接复制给 AI，帮助配置并启动 `train-infer-estimation-release-2026-04-11` 项目环境。

---

你现在要帮我配置并验证这个项目：

- 项目目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11`
- 目标：让项目的 Dashboard 可以正常启动，并能执行单机推理分析任务

请严格按下面要求执行，并把结果整理清楚。

## 1. 先准备运行环境

请优先检查并使用当前本机默认配置中的 Python：

```bash
/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/tools/python_with_env.sh
```

这个 Python 来自：

- `configs/dashboard_env.json`
- `mvp_dashboard.py` 默认配置

你需要确认：

1. 这个 Python 是否存在。
2. 这个环境里是否能导入并运行项目主脚本依赖。
3. 至少满足以下脚本可运行：
   - `mvp_dashboard.py`
   - `torch_infer_mvp.py`

重点检查：

- `torch`
- `transformers`
- `flask/http server` 相关标准依赖
- 项目内部模块是否能正常 import

如果这个 Python 不存在，或者依赖不完整，请：

1. 明确指出缺什么。
2. 给出安装命令。
3. 优先在现有环境补依赖，不要随意更换整套环境。

## 2. 模型与配置检查

请检查配置文件：

```bash
configs/dashboard_env.json
```

确认以下关键字段：

- `runner`
- `python_bin`
- `model_path`
- `parallel_mode`
- `physical_devices`
- `world_size`
- `tp_size`
- `warmup`
- `benchmark_repeat`
- `profile_repeat`

当前本机默认模型路径是：

```bash
/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B
```

请检查这个模型路径是否存在、是否可读。

如果不存在，请：

1. 明确指出当前路径不可用。
2. 要求我提供正确模型路径，或帮我把配置改到正确路径。
3. 不要假装模型已经就绪。

## 3. 启动 Dashboard

在项目根目录执行：

```bash
python3 mvp_dashboard.py
```

默认访问地址：

```bash
http://127.0.0.1:8123
```

如果需要，也可以识别这些环境变量：

- `MVP_DASHBOARD_CONFIG`
- `MVP_DASHBOARD_HOST`
- `MVP_DASHBOARD_PORT`

请帮我确认：

1. Dashboard 是否成功启动。
2. 监听地址和端口是什么。
3. 如果启动失败，失败日志是什么。

## 4. 单机分析任务怎么操作

启动后，请按项目当前逻辑说明单机模式下如何操作。

至少包括：

1. 如何选择 `runner`
   - `local_python`
   - `docker_run_image`
2. 如何填写：
   - `python_bin`
   - `model_path`
   - `warmup`
   - `benchmark_repeat`
   - `profile_repeat`
   - `parallel_mode`
   - `physical_devices`
   - `world_size`
   - `tp_size`
3. 如何点击并启动一次分析。
4. 分析结果输出目录会写到哪里。

请结合项目源码说明，不要只给泛泛建议。

根据当前项目代码，输出目录与运行状态相关文件需要重点说明：

- `dashboard_status.json`
- `report.json`
- `graph_viz/`

## 5. 如果要用托管 / 容器环境

这个项目支持托管环境和容器相关流程。

请检查并说明：

- `tools/dashboard_env.py`
- `notes/managed_env_execution_runbook.md`

如果采用托管环境，请给我标准执行顺序：

```bash
python3 tools/dashboard_env.py status --config configs/dashboard_env.json
python3 tools/dashboard_env.py prepare --config configs/dashboard_env.json
python3 tools/dashboard_env.py stop --config configs/dashboard_env.json
```

并说明：

1. 什么时候该 `prepare`
2. 什么时候该 `stop`
3. 如何确认容器/环境状态正常

## 6. 如果要跨机

如果这个项目以后要扩展成跨机，请提前说明需要满足：

1. 主机到远端免密 SSH
2. 远端 Python 环境可用
3. 远端代码路径尽量与主机一致
4. 远端模型路径可通过 `remote_model_path` 单独指定

并帮我检查配置里已有的相关字段：

- `remote_host`
- `remote_ssh_port`
- `remote_python_bin`
- `remote_model_path`
- `remote_physical_devices`

## 7. 你最终要输出什么

请最终给我以下结果：

### A. 环境检查结论

明确列出：

- Python 是否可用
- 依赖是否齐全
- 模型路径是否有效
- Dashboard 是否能启动

### B. 若有问题，给出修复命令

要给我可直接执行的命令，而不是只说“请安装依赖”。

### C. 单机启动步骤

给我一套可以直接复线的最短步骤。

### D. 可选的跨机说明

如果当前不做跨机，也要说明以后需要补哪些条件。

### E. 风险提示

如果你发现以下任一问题，请明确写出来：

- Python 路径不存在
- 模型路径不存在
- Dashboard 端口冲突
- Docker / GPU 环境未就绪
- 托管环境残留状态未清理

## 8. 执行原则

请遵守：

1. 先检查，再修改。
2. 不要假设模型路径一定存在。
3. 不要假设 Docker 一定可用。
4. 不要假设远端环境已经配好。
5. 如果你不能确认，就明确写“未验证”。
6. 输出要尽量面向实际操作，少空话。

---

## 本项目的已知默认信息

- Dashboard 入口：`mvp_dashboard.py`
- 后端默认配置文件：`configs/dashboard_env.json`
- 默认 Python：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/tools/python_with_env.sh`
- 默认监听地址：`127.0.0.1:8123`
- 默认 runner：`local_python`
- 可选 runner：`docker_run_image`

## 推荐先读的项目文件

- [docs/torch_mvp_dashboard.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/docs/torch_mvp_dashboard.md)
- [notes/managed_env_execution_runbook.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/notes/managed_env_execution_runbook.md)
- [configs/dashboard_env.json](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/configs/dashboard_env.json)
- [mvp_dashboard.py](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/mvp_dashboard.py)
- [torch_infer_mvp.py](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/torch_infer_mvp.py)

## 已完成任务参考材料

如果你需要同时理解我这边已经完成的摩尔线程相关验证任务，请把下面这些任务进展文档也一并纳入参考范围：

- [5.1.5任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.5/5.1.5任务进展.md)
- [5.1.11任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11/5.1.11任务进展.md)
- [5.2.3任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3/5.2.3任务进展.md)
- [5.2.6任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.6/5.2.6任务进展.md)
- [5.2.9任务进展.md](/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.9/5.2.9任务进展.md)

这些文档包含：

- 已验证的摩尔线程 GPU 环境与依赖组合
- 单卡、双卡、建模与通信测试的真实结果
- 已修正后的误差计算、标定点/验证点说明
- 当前项目在摩尔线程环境下可参考的实测口径

## 路径适配说明

这份 prompt 以当前机器用户 `o_mabin` 为准。

- 当前项目根目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11`
- 当前默认模型目录：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B`
- 当前默认 Python 优先尝试：`/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/tools/python_with_env.sh`
- 该包装器会自动补齐当前机器上的 `LD_LIBRARY_PATH`，再调用 `python3`

如果实际环境不是以上路径，请以“当前机器实际存在的路径”为准，不要继续沿用历史文档中的 `/home/o_zhanghui/...` 示例值。
