import argparse
from dataclasses import dataclass

from hardware_profiler import HardwareProfiler
from bench_core import (
    ADDBenchmark,
    FFNBenchmark,
    MatmulBenchmark,
    RMSNormBenchmark,
    SDPABenchmark,
    SoftmaxBenchmark,
)
from predictor import (
    ADDKernel,
    FFNKernel,
    GEMMKernel,
    PredictorEngine,
    RMSNormKernel,
    SDPAKernel,
    SoftmaxKernel,
)


@dataclass(frozen=True)
class OperatorSpec:
    key: str
    title: str
    sweep_label: str
    sweep_values: list
    predictor_key: str
    build_predict_kwargs: callable
    build_benchmark: callable


OPERATOR_SPECS = {
    "mm": OperatorSpec(
        key="mm",
        title="校验 GEMM (矩阵乘法) | ( M, 4096, 14336 ) 扫描",
        sweep_label="M",
        sweep_values=[512, 2048, 8192, 16384],
        predictor_key="mm",
        build_predict_kwargs=lambda value: {"M": value, "K": 4096, "N": 14336},
        build_benchmark=lambda value: MatmulBenchmark(M=value, K=4096, N=14336),
    ),
    "sdpa": OperatorSpec(
        key="sdpa",
        title="校验 FlashAttention (SDPA) | (4, 32, N) 扫描",
        sweep_label="N",
        sweep_values=[512, 1024, 2048, 4096],
        predictor_key="sdpa",
        build_predict_kwargs=lambda value: {"B": 4, "H": 32, "S": value, "D": 128},
        build_benchmark=lambda value: SDPABenchmark(
            batch_size=4, num_heads=32, seq_len=value, head_dim=128
        ),
    ),
    "rmsnorm": OperatorSpec(
        key="rmsnorm",
        title="校验 RMSNorm | (4, N, 128) 扫描",
        sweep_label="N",
        sweep_values=[512, 1024, 2048, 4096],
        predictor_key="rmsnorm",
        build_predict_kwargs=lambda value: {"B": 4, "S": value, "H": 128},
        build_benchmark=lambda value: RMSNormBenchmark(
            batch_size=4, seq_len=value, hidden_size=128
        ),
    ),
    "softmax": OperatorSpec(
        key="softmax",
        title="校验 Softmax | (4, 32, N) 扫描",
        sweep_label="N",
        sweep_values=[512, 1024, 2048, 4096, 8192],
        predictor_key="softmax",
        build_predict_kwargs=lambda value: {"B": 4, "S": 32, "H": value},
        build_benchmark=lambda value: SoftmaxBenchmark(
            batch_size=4, num_heads=32, seq_len=value
        ),
    ),
    "add": OperatorSpec(
        key="add",
        title="校验 张量加法 | (16, N, 4096) 扫描",
        sweep_label="N",
        sweep_values=[512, 1024, 2048, 4096, 8192],
        predictor_key="add",
        build_predict_kwargs=lambda value: {"shape": (16, value, 4096)},
        build_benchmark=lambda value: ADDBenchmark(
            batch_size=16, seq_len=value, hidden_size=4096
        ),
    ),
}


KERNEL_REGISTRY = {
    "mm": GEMMKernel,
    "sdpa": SDPAKernel,
    "rmsnorm": RMSNormKernel,
    "softmax": SoftmaxKernel,
    "ffn": FFNKernel,
    "add": ADDKernel,
}


def _predict_operator(engine, spec, predict_kwargs):
    if spec.key == "add":
        return engine.predict_us(spec.predictor_key, *predict_kwargs["shape"])
    return engine.predict_us(spec.predictor_key, **predict_kwargs)


def run_operator(spec, engine, profiler):
    print(f"\n {spec.title}")
    print(f"{spec.sweep_label:<8} | {'预测(us)':<12} | {'实测(us)':<12} | {'误差 %':<8}")
    print("-" * 50)

    predicted_values = []
    actual_values = []

    for sweep_value in spec.sweep_values:
        predict_kwargs = spec.build_predict_kwargs(sweep_value)
        benchmark = spec.build_benchmark(sweep_value)
        predicted_us = _predict_operator(engine, spec, predict_kwargs)
        actual_us = benchmark.run() * 1e6
        error_pct = abs(predicted_us - actual_us) / actual_us * 100
        predicted_values.append(predicted_us)
        actual_values.append(actual_us)

        print(f"{sweep_value:<8} | {predicted_us:<12.2f} | {actual_us:<12.2f} | {error_pct:.1f}%")

    engine.update_kernel_from_measurements(spec.key, predicted_values, actual_values)


def build_parser():
    parser = argparse.ArgumentParser(description="LLM算子性能预测与实测对比引擎")
    parser.add_argument(
        "-t",
        "--tests",
        nargs="+",
        choices=list(OPERATOR_SPECS.keys()) + ["all"],
        default=["all"],
        help="指定要运行的测试项 (支持多选，默认运行所有)。",
    )
    parser.add_argument(
        "--force-remeasure",
        action="store_true",
        help="忽略 device_profiles.yaml 缓存，强制重新实测硬件探针。",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if "all" in args.tests:
        tasks_to_run = list(OPERATOR_SPECS.keys())
    else:
        tasks_to_run = list(dict.fromkeys(args.tests))

    print("正在初始化硬件探针并注册预测内核...")
    hw = HardwareProfiler().profile_hardware(force_remeasure=args.force_remeasure)
    engine = PredictorEngine(hw)

    for kernel_name, kernel_class in KERNEL_REGISTRY.items():
        engine.register_kernel(kernel_name, kernel_class)

    for task_name in tasks_to_run:
        run_operator(OPERATOR_SPECS[task_name], engine, hw)

    hw.set_calibration(engine.export_calibration())
    hw.save_profile()
    print(f"\n 已同步硬件profile和参数到 {hw.cache_path}")
    print("\n 所有指定测试项执行完毕。")


if __name__ == "__main__":
    main()
