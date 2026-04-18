import os
import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark
import numpy as np

class HardwareProfiler:
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        
        # 1. 节点内计算与访存极限
        self.peak_tflops = 0.0
        self.peak_bw_gbps = 0.0
        self.kernel_overhead_us = 0.0
        
        # 2. 多卡通信极限 (默认赋理论值兜底，由探针动态覆盖)
        self.intra_node_bw_gbps = 250.0  
        self.inter_node_bw_gbps = 0.0
        self.nccl_latency_us = 10.0      

    def profile_hardware(self, skip_comm=True):
        """
        探针总入口
        :param skip_comm: 是否强制跳过通信探测 (单卡调试时极度有用)
        """

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if local_rank == 0:
            print(f"🔍 正在探测单卡计算与访存极限 ({torch.cuda.get_device_name(self.device)})...")
        
        # ==========================================
        # 阶段 1：单卡基础能力探测
        # ==========================================
        
        # 1. 测量 Launch Overhead (使用 benchmark)
        dummy = torch.zeros(1, device=self.device)
        t_overhead = benchmark.Timer(
            stmt="x + 1",
            globals={"x": dummy}
        ).blocked_autorange(min_run_time=0.5)
        self.kernel_overhead_us = t_overhead.median * 1e6

        # 2. 测量峰值算力 (GEMM)
        N = 8192
        A = torch.randn(N, N, dtype=self.dtype, device=self.device)
        B = torch.randn(N, N, dtype=self.dtype, device=self.device)
        flops = 2.0 * N**3
        
        t_gemm = benchmark.Timer(
            stmt="torch.matmul(A, B)",
            globals={"A": A, "B": B}
        ).blocked_autorange(min_run_time=1.0)
        self.peak_tflops = (flops / t_gemm.median) / 1e12

        # 3. 测量峰值带宽 (冲刷 L2 Cache)
        N_elem = 1024 * 1024 * 128
        X = torch.randn(N_elem, dtype=self.dtype, device=self.device)
        Y = torch.randn(N_elem, dtype=self.dtype, device=self.device)
        bytes_moved = N_elem * self.bytes_per_elem * 3
        
        t_bw = benchmark.Timer(
            stmt="X + Y",
            globals={"X": X, "Y": Y}
        ).blocked_autorange(min_run_time=1.0)
        self.peak_bw_gbps = (bytes_moved / t_bw.median) / 1e9
        
        if local_rank == 0:
            print(f"✅ 单卡探测完成:\n"
                  f"   峰值算力: {self.peak_tflops:.2f} TFLOPS\n"
                  f"   峰值带宽: {self.peak_bw_gbps:.2f} GB/s\n"
                  f"   调度开销: {self.kernel_overhead_us:.2f} us")

        # ==========================================
        # 阶段 2：多卡通信网络探测
        # ==========================================
        is_dist = dist.is_initialized()
        
        # 智能判定是否需要拉起通信探针
        if skip_comm or not is_dist:
            # if local_rank == 0:
            #     print("⚠️ [探针状态] 单进程模式或开启了 skip_comm，跳过 NCCL 网络探测，使用默认理论值。")
            return self

        world_size = dist.get_world_size()
        if world_size > 1:
            if local_rank == 0:
                print(f"🔍 正在探测多卡通信网络 (World Size = {world_size})...")
            self._profile_nccl_full()

        return self

    def _profile_nccl_full(self):
        """
        双轨通信压测：用极小包测延迟，用极大包测真实带宽
        """
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # ------------------------------------------
        # 轨 1：测纯延迟 (Latency) - 使用 1 个元素的极小张量
        # ------------------------------------------
        tiny_x = torch.ones(256, dtype=self.dtype, device=self.device)
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(tiny_x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()  # 确保所有卡同步起跑
        
        start_events_lat = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        end_events_lat = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
        
        for i in range(50):
            start_events_lat[i].record()
            dist.all_reduce(tiny_x, op=dist.ReduceOp.SUM)
            end_events_lat[i].record()
            
        torch.cuda.synchronize()
        dist.barrier()
        
        # 提取中位数延迟 (微秒)
        latencies_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events_lat, end_events_lat)]
        self.nccl_latency_us = np.median(latencies_us)

        # ------------------------------------------
        # 轨 2：测纯带宽 (Bandwidth) - 使用 100MB 级别大张量
        # ------------------------------------------
        # 50,000,000 * 2 bytes = 100 MB
        num_elements = 50_000_000 
        tensor_size_bytes = num_elements * self.bytes_per_elem
        huge_x = torch.randn(num_elements, dtype=self.dtype, device=self.device)
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(huge_x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()
        
        start_events_bw = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        end_events_bw = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        
        for i in range(10):
            start_events_bw[i].record()
            dist.all_reduce(huge_x, op=dist.ReduceOp.SUM)
            end_events_bw[i].record()
            
        torch.cuda.synchronize()
        dist.barrier()
        
        # 提取耗时并扣除延迟，计算纯净物理带宽
        times_s = [s.elapsed_time(e) / 1000.0 for s, e in zip(start_events_bw, end_events_bw)]
        median_time_s = np.median(times_s)
        
        # 理论单卡传输量: 2 * (N-1)/N * D
        bytes_moved = 2.0 * ((world_size - 1) / world_size) * tensor_size_bytes
        
        # 扣除纯延迟，反推真实带宽 (做个最大值保护，防止除以极小负数)
        pure_transfer_time_s = max(0.00001, median_time_s - (self.nccl_latency_us / 1e6))
        eff_bw_gbps = (bytes_moved / 1e9) / pure_transfer_time_s

        # ------------------------------------------
        # 智能赋值：判断当前属于机内拓扑还是跨机拓扑
        # ------------------------------------------
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        
        if local_world_size == world_size:
            # 都在同一台物理机上
            self.intra_node_bw_gbps = eff_bw_gbps
        else:
            # 跨机环境
            self.inter_node_bw_gbps = eff_bw_gbps

        if local_rank == 0:
            print(f"✅ 通信网络探测完成:\n"
                  f"   NCCL 握手发车延迟: {self.nccl_latency_us:.2f} us")
            if local_world_size == world_size:
                print(f"   机内 (Intra-node) 通信带宽: {self.intra_node_bw_gbps:.2f} GB/s")
            else:
                print(f"   跨机 (Inter-node) 通信带宽: {self.inter_node_bw_gbps:.2f} GB/s")