import argparse
import torch
from hardware_profiler import HardwareProfiler
from bench_core import MatmulBenchmark, SDPABenchmark, RMSNormBenchmark, SoftmaxBenchmark, FFNBenchmark, ADDBenchmark
from predictor import PredictorEngine, GEMMKernel, SDPAKernel, RMSNormKernel, SoftmaxKernel, FFNKernel, ADDKernel

# ==========================================
# 测试任务封装区 (将每种算子的测试独立拆分)
# ==========================================

def test_matmul(engine):
    K, N = 4096, 14336
    print(f"\n 校验 GEMM (矩阵乘法) | M, K, N 参数扫描")
    print(f"{'M':<8} | {'预测(us)':<12} | {'实测(us)':<12} | {'误差 %':<8}")
    print("-" * 50)
    for M in [512, 2048, 8192, 16384]:
        pred = engine.predict_us("mm", M=M, K=K, N=N)
        real = MatmulBenchmark(M=M, K=K, N=N).run() * 1e6
        error = abs(pred - real) / real * 100
        print(f"{M:<8} | {pred:<12.2f} | {real:<12.2f} | {error:.1f}%")

def test_sdpa(engine):
    print(f"\n 校验 FlashAttention (SDPA) | Sequence Length 扫描")
    print(f"{'SeqLen':<8} | {'预测(us)':<12} | {'实测(us)':<12} | {'误差 %':<8}")
    print("-" * 50)
    for S in [512, 1024, 2048, 4096]:
        pred = engine.predict_us("sdpa", B=4, H=32, S=S, D=128)
        real = SDPABenchmark(batch_size=4, num_heads=32, seq_len=S, head_dim=128).run() * 1e6
        error = abs(pred - real) / real * 100
        print(f"{S:<8} | {pred:<12.2f} | {real:<12.2f} | {error:.1f}%")

def test_rmsnorm(engine):
    print(f"\n 校验 RMSNorm | Sequence Length 扫描")
    print(f"{'SeqLen':<8} | {'预测(us)':<12} | {'实测(us)':<12} | {'误差 %':<8}")
    print("-" * 50)
    for S in [512, 1024, 2048, 4096]:
        pred = engine.predict_us("rmsnorm", B=4, S=S, H=128)
        real = RMSNormBenchmark(batch_size=4, seq_len=S, hidden_size=128).run() * 1e6
        error = abs(pred - real) / real * 100
        print(f"{S:<8} | {pred:<12.2f} | {real:<12.2f} | {error:.1f}%")

def test_softmax(engine):
    print(f"\n 校验 Softmax | Sequence Length 扫描")
    print(f"{'SeqLen':<8} | {'预测(us)':<12} | {'实测(us)':<12} | {'误差 %':<8}")
    print("-" * 50)
    for S in [512, 1024, 2048, 4096, 8192]:
        pred = engine.predict_us("softmax", 4, 32, S)
        real = SoftmaxBenchmark(batch_size=4, num_heads=32, seq_len=S).run() * 1e6
        error = abs(pred - real) / real * 100
        print(f"{S:<8} | {pred:<12.2f} | {real:<12.2f} | {error:.1f}%")

def test_add(engine):
    print(f"\n 校验 张量加法 | Sequence Length 扫描")
    print(f"{'SeqLen':<8} | {'预测(us)':<12} | {'实测(us)':<12} | {'误差 %':<8}")
    print("-" * 50)
    for S in [512, 1024, 2048, 4096, 8192]:
        pred = engine.predict_us("add", 16, S, 4096)
        real = ADDBenchmark(batch_size=16, seq_len=S, hidden_size=4096).run() * 1e6
        error = abs(pred - real) / real * 100
        print(f"{S:<8} | {pred:<12.2f} | {real:<12.2f} | {error:.1f}%")

# ==========================================
# CLI 入口与调度中心
# ==========================================

def main():
    # 1. 配置命令行参数解析
    parser = argparse.ArgumentParser(description="LLM算子性能预测与实测对比引擎")
    
    # 注册支持的测试项映射字典
    TEST_MAP = {
        "mm": test_matmul,
        "sdpa": test_sdpa,
        "rmsnorm": test_rmsnorm,
        "softmax": test_softmax,
        "add": test_add
    }
    
    parser.add_argument(
        "-t", "--tests", 
        nargs="+", 
        choices=list(TEST_MAP.keys()) + ["all"], 
        default=["all"],
        help="指定要运行的测试项 (支持多选，默认运行所有)。"
    )
    
    args = parser.parse_args()

    # 2. 解析需要运行的任务列表
    if "all" in args.tests:
        tasks_to_run = list(TEST_MAP.keys())
    else:
        tasks_to_run = list(dict.fromkeys(args.tests))

    # 3. 初始化硬件环境与预测引擎 (延迟初始化，避免没跑任务却占显存)
    print("正在初始化硬件探针并注册预测内核...")
    hw = HardwareProfiler().profile_hardware()
    engine = PredictorEngine(hw)
    
    engine.register_kernel("mm", GEMMKernel)
    engine.register_kernel("sdpa", SDPAKernel)
    engine.register_kernel("rmsnorm", RMSNormKernel)
    engine.register_kernel("softmax", SoftmaxKernel)
    engine.register_kernel("ffn", FFNKernel)
    engine.register_kernel("add", ADDKernel)

    # 4. 循环调度执行任务
    for task_name in tasks_to_run:
        func = TEST_MAP[task_name]
        func(engine)
        
    print("\n 所有指定测试项执行完毕。")

if __name__ == "__main__":
    main()