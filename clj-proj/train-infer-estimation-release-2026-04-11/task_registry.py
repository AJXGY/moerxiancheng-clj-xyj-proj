from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj")
MUSA_LD_PATH = (
    "/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu:"
    "/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread:"
    "/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib:"
    "/home/o_mabin/.local/mudnn/mudnn/lib:"
    "/usr/local/musa/lib"
)
DEFAULT_MODEL_PATH = "/home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/model/Meta-Llama-3.1-8B"


@dataclass(frozen=True)
class IntegratedTask:
    task_id: str
    name: str
    description: str
    project_dir: Path
    report_path: Path
    command: list[str]
    charts: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "project_dir": str(self.project_dir),
            "report_path": str(self.report_path),
            "report_exists": self.report_path.exists(),
            "charts": [
                {"path": str(path), "exists": path.exists(), "name": path.name}
                for path in self.charts
            ],
        }


def _bash(command: str) -> list[str]:
    return ["bash", "-lc", command]


def integrated_tasks() -> list[IntegratedTask]:
    return [
        IntegratedTask(
            task_id="5.1.5",
            name="5.1.5 推理任务运行测试",
            description="单卡/双卡推理运行验证，含真实输出与结果校验。",
            project_dir=PROJECT_ROOT / "5.1.5",
            report_path=PROJECT_ROOT / "5.1.5" / "5.1.5任务进展.md",
            command=_bash(
                "cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.5 && "
                f"export LD_LIBRARY_PATH={MUSA_LD_PATH}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}} && "
                "bash run_515_suite.sh "
                f"--model-path {DEFAULT_MODEL_PATH} --device-type musa --single-device-ids 0 --dual-device-ids 0,1"
            ),
            charts=(
                PROJECT_ROOT / "5.1.5" / "charts" / "status_af.svg",
                PROJECT_ROOT / "5.1.5" / "charts" / "runtime_compare.svg",
            ),
        ),
        IntegratedTask(
            task_id="5.1.11",
            name="5.1.11 训练任务处理模型输出测试",
            description="CPU/GPU 任务分配、并行策略、Microbatch 与 DAG 输出验证。",
            project_dir=PROJECT_ROOT / "5.1.11",
            report_path=PROJECT_ROOT / "5.1.11" / "5.1.11任务进展.md",
            command=_bash("cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.1.11 && bash run_5111_suite.sh"),
            charts=(
                PROJECT_ROOT / "5.1.11" / "charts" / "task_topology.svg",
                PROJECT_ROOT / "5.1.11" / "charts" / "dag_graph.svg",
            ),
        ),
        IntegratedTask(
            task_id="5.2.3",
            name="5.2.3 计算密集型算子空间维度建模测试",
            description="GEMM 类算子单卡/双卡实测与时空模型误差验证。",
            project_dir=PROJECT_ROOT / "5.2.3",
            report_path=PROJECT_ROOT / "5.2.3" / "5.2.3任务进展.md",
            command=_bash("cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.3 && bash run_523_suite.sh"),
            charts=(
                PROJECT_ROOT / "5.2.3" / "charts" / "error_compare.png",
                PROJECT_ROOT / "5.2.3" / "charts" / "runtime_compare.png",
            ),
        ),
        IntegratedTask(
            task_id="5.2.6",
            name="5.2.6 访存密集型算子空间维度建模测试",
            description="copy/slice/cat 多规模标定与验证点误差测试。",
            project_dir=PROJECT_ROOT / "5.2.6",
            report_path=PROJECT_ROOT / "5.2.6" / "5.2.6任务进展.md",
            command=_bash("cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.6 && bash run_526_suite.sh"),
            charts=(
                PROJECT_ROOT / "5.2.6" / "charts" / "error_compare.png",
                PROJECT_ROOT / "5.2.6" / "charts" / "runtime_compare.png",
            ),
        ),
        IntegratedTask(
            task_id="5.2.9",
            name="5.2.9 通信密集型算子空间维度建模测试",
            description="Send/Recv 与 AllReduce 双卡通信建模验证。",
            project_dir=PROJECT_ROOT / "5.2.9",
            report_path=PROJECT_ROOT / "5.2.9" / "5.2.9任务进展.md",
            command=_bash("cd /home/o_mabin/moerxiancheng-clj-xyj-proj/clj-proj/5.2.9 && bash run_529_suite.sh"),
            charts=(
                PROJECT_ROOT / "5.2.9" / "charts" / "error_compare.png",
                PROJECT_ROOT / "5.2.9" / "charts" / "runtime_compare.png",
            ),
        ),
    ]


def integrated_task_map() -> dict[str, IntegratedTask]:
    return {task.task_id: task for task in integrated_tasks()}

