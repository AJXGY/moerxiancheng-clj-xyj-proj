import os
import sys
import argparse
import torch
import torch.distributed as dist

# 导入底层硬件探针与引擎
from hardware_profiler import HardwareProfiler
from bench_core import AllReduceBenchmark, AllGatherBenchmark
from predictor import PredictorEngine, AllReduceKernel, AllGatherKernel

# ==========================================
# 1. 分布式环境初始化
# ==========================================
def setup_distributed():
    """
    检查并初始化分布式环境 (强制拦截单机运行)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(" 错误: 通信算子测试必须在分布式环境下运行！")
        print(" 请使用 torchrun 启动，例如: torchrun --nproc_per_node=2 test_comm.py -t all")
        sys.exit(1)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    if torch.__version__ >= "2.0.0":
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group(backend="nccl")
        
    return local_rank, world_size

# ==========================================
# 2. 算子测试任务区
# ==========================================

# 预设的张量扫描规模 (约 2MB, 10MB, 50MB, 100MB, 200MB, 500MB)
SIZE_LIST = [1_000_000, 5_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000]

def test_allreduce(engine, local_rank, world_size):
    if local_rank == 0:
        print(f"\n 校验 [AllReduce] | 规模扫描 (World Size = {world_size})")
        print(f"{'Size (MB)':<10} | {'预测 (us)':<12} | {'实测 (us)':<12} | {'误差 %':<8}")
        print("-" * 60)

    for num_elements in SIZE_LIST:
        D_bytes = num_elements * 2.0
        size_mb = D_bytes / (1024**2)

        pred_us = engine.predict_us("allreduce", D_bytes=D_bytes, N_gpus=world_size)
        bench = AllReduceBenchmark(num_elements=num_elements, device=f"cuda:{local_rank}")
        real_us = bench.run(min_run_time=0.5) * 1e6
        
        if local_rank == 0:
            error = abs(pred_us - real_us) / real_us * 100
            print(f"{size_mb:<10.2f} | {pred_us:<12.2f} | {real_us:<12.2f} | {error:.1f}%")


def test_allgather(engine, local_rank, world_size):
    if local_rank == 0:
        print(f"\n 校验 [AllGather] | 规模扫描 (World Size = {world_size})")
        print(f"{'Size (MB)':<10} | {'预测 (us)':<12} | {'实测 (us)':<12} | {'误差 %':<8}")
        print("-" * 60)

    for num_elements in SIZE_LIST:
        D_bytes = num_elements * 2.0
        size_mb = D_bytes / (1024**2)

        pred_us = engine.predict_us("allgather", D_bytes=D_bytes, N_gpus=world_size)
        bench = AllGatherBenchmark(full_elements=num_elements, device=f"cuda:{local_rank}")
        real_us = bench.run(min_run_time=0.5) * 1e6
        
        if local_rank == 0:
            error = abs(pred_us - real_us) / real_us * 100
            print(f"{size_mb:<10.2f} | {pred_us:<12.2f} | {real_us:<12.2f} | {error:.1f}%")

# ==========================================
# 3. CLI 入口与调度中心
# ==========================================
def main():
    # 1. 强制多进程环境初始化
    local_rank, world_size = setup_distributed()
    
    # 2. 调度器映射字典
    TEST_MAP = {
        "allreduce": test_allreduce,
        "allgather": test_allgather
    }

    # 为了避免多卡同时解析参数引发冲突，只用基础的解析逻辑
    parser = argparse.ArgumentParser(description="大模型分布式通信算子性能压测工具")
    parser.add_argument(
        "-t", "--tests", 
        nargs="+", 
        choices=list(TEST_MAP.keys()) + ["all"], 
        default=["all"],
        help="指定要运行的通信算子 (默认运行所有)。"
    )
    args = parser.parse_args()

    # 解析任务列表
    if "all" in args.tests:
        tasks_to_run = list(TEST_MAP.keys())
    else:
        tasks_to_run = list(dict.fromkeys(args.tests))

    if local_rank == 0:
        print("=" * 60)
        print(f"启动通信算子预测与实测引擎 (World Size: {world_size})")
        print(f"执行队列: {tasks_to_run}")
        print("=" * 60)

    # 3. 探针嗅探与预测引擎挂载
    hw = HardwareProfiler().profile_hardware(skip_comm=False)
    engine = PredictorEngine(hw)
    
    engine.register_kernel("allreduce", AllReduceKernel)
    engine.register_kernel("allgather", AllGatherKernel)

    # 4. 任务分发执行
    for task_name in tasks_to_run:
        func = TEST_MAP[task_name]
        # 把 world_size 和 local_rank 传给测试函数
        func(engine, local_rank, world_size)
        
    if local_rank == 0:
        print("\n所有指定通信测试执行完毕。")

    # 5. 优雅退出
    dist.destroy_process_group()

if __name__ == "__main__":
    main()