import torch
import torch.utils.benchmark as benchmark
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
class OperatorBenchmark:
    def __init__(self, name: str, description: str, device="cuda", dtype=torch.bfloat16):
        self.name = name
        self.description = description
        self.device = torch.device(device)
        self.dtype = dtype
        self.inputs = {}
        self.grad_output = None

    def prepare_inputs(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def _generate_auto_grad(self):
        with torch.no_grad():
            out = self.forward(**self.inputs)
        if isinstance(out, tuple): out = out[0]
        self.grad_output = torch.randn_like(out)

    def _forward_backward_wrapper(self, **kwargs):
        out = self.forward(**kwargs)
        if isinstance(out, tuple): out = out[0]
        out.backward(self.grad_output, retain_graph=True)

    def run(self, min_run_time=0.2, test_backward=False):
        torch.cuda.empty_cache()
        self.prepare_inputs()
        
        timer_fwd = benchmark.Timer(
            stmt="fn(**inputs)",
            globals={"fn": self.forward, "inputs": self.inputs},
            label=self.name,
            sub_label="Forward"
        )
        
        # 提取精确的中位数时间 (秒)
        measurement = timer_fwd.blocked_autorange(min_run_time=min_run_time)
        return measurement.median

class MatmulBenchmark(OperatorBenchmark):
    def __init__(self, M, K, N):
        super().__init__(name="aten::mm", description=f"M={M}, K={K}, N={N}")
        self.M, self.K, self.N = M, K, N

    def prepare_inputs(self):
        A = torch.randn(self.M, self.K, dtype=self.dtype, device=self.device)
        B_weight = torch.randn(self.N, self.K, dtype=self.dtype, device=self.device)
        self.inputs = {"a": A, "b": B_weight.t()}

    def forward(self, a, b):
        return torch.matmul(a, b)

import torch
import torch.nn.functional as F
import math

# ==========================================
# 🔥 计算密集型 (Compute-Bound) - 非传统 GEMM
# ==========================================

class SDPABenchmark(OperatorBenchmark):
    """
    算子 1: 融合注意力机制 (SDPA / FlashAttention)
    底层: 调用 FlashAttention 内核，极高的计算强度，彻底消除 O(N^2) 显存读写
    """
    def __init__(self, batch_size, num_heads, seq_len, head_dim):
        super().__init__(
            name="aten::scaled_dot_product_attention", 
            description=f"B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}"
        )
        self.B, self.H, self.S, self.D = batch_size, num_heads, seq_len, head_dim

    def prepare_inputs(self):
        shape = (self.B, self.H, self.S, self.D)
        self.inputs = {
            "query": torch.randn(*shape, dtype=self.dtype, device=self.device),
            "key": torch.randn(*shape, dtype=self.dtype, device=self.device),
            "value": torch.randn(*shape, dtype=self.dtype, device=self.device)
        }

    def forward(self, query, key, value):
        # 默认不加 mask，PyTorch 会自动分发给 FlashAttention
        return F.scaled_dot_product_attention(query, key, value)

class Conv1dBenchmark(OperatorBenchmark):
    """
    算子 2: 一维卷积 (常用于 Mamba/SSM 架构)
    底层: 调用 cuDNN 卷积内核
    """
    def __init__(self, batch_size, channels, seq_len, kernel_size):
        super().__init__(
            name="aten::conv1d", 
            description=f"B={batch_size}, C={channels}, L={seq_len}, K={kernel_size}"
        )
        self.B, self.C, self.L, self.K = batch_size, channels, seq_len, kernel_size
        self.conv = torch.nn.Conv1d(
            in_channels=channels, out_channels=channels, 
            kernel_size=kernel_size, padding=kernel_size//2, 
            device=self.device, dtype=self.dtype
        )

    def prepare_inputs(self):
        # Conv1d 要求输入格式为 [Batch, Channels, Length]
        self.inputs = {
            "x": torch.randn(self.B, self.C, self.L, dtype=self.dtype, device=self.device)
        }

    def forward(self, x):
        return self.conv(x)

class FFNBenchmark(OperatorBenchmark):

    def __init__(self, M, K, N):
        super().__init__(name="aten::ffn_mm", description=f"M={M}, K={K}, N={N}")
        self.M, self.K, self.N = M, K, N

    def prepare_inputs(self):
        A = torch.randn(self.M, self.K, dtype=self.dtype, device=self.device)
        B_weight = torch.randn(self.N, self.K, dtype=self.dtype, device=self.device)
        self.inputs = {"a": A, "b": B_weight.t()}

    def forward(self, a, b):
        return torch.nn.functional.linear(a, b)

# ==========================================
# 🧊 访存密集型 (Memory-Bound)
# ==========================================

class RMSNormBenchmark(OperatorBenchmark):
    """算子 3: RMSNorm (LLaMA 标准归一化)"""
    def __init__(self, batch_size, seq_len, hidden_size):
        super().__init__(name="aten::rmsnorm (Manual)", description=f"B={batch_size}, S={seq_len}, H={hidden_size}")
        self.B, self.S, self.H = batch_size, seq_len, hidden_size
        self.weight = torch.nn.Parameter(torch.ones(self.H, device=self.device, dtype=self.dtype))

    def prepare_inputs(self):
        self.inputs = {"x": torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=self.device)}

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + 1e-6))

class SiLUBenchmark(OperatorBenchmark):
    """算子 4: SiLU 激活函数"""
    def __init__(self, batch_size, seq_len, hidden_size):
        super().__init__(name="aten::silu", description=f"B={batch_size}, S={seq_len}, H={hidden_size}")
        self.B, self.S, self.H = batch_size, seq_len, hidden_size

    def prepare_inputs(self):
        self.inputs = {"x": torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=self.device)}

    def forward(self, x):
        return F.silu(x, inplace=True)

class SoftmaxBenchmark(OperatorBenchmark):
    def __init__(self, batch_size, num_heads, seq_len):
        super().__init__(name="aten::_softmax", description=f"B={batch_size}, H={num_heads}, S={seq_len}")
        self.B, self.H_heads, self.S = batch_size, num_heads, seq_len

    def prepare_inputs(self):
        self.inputs = {"x": torch.randn(self.B, self.H_heads, self.S, dtype=self.dtype, device=self.device)}

    def forward(self, x):
        return F.softmax(x, dim=-1)

class ADDBenchmark(OperatorBenchmark):
    """算子 5: 张量加法 (常见于残差连接)"""
    def __init__(self, batch_size, seq_len, hidden_size):
        super().__init__(name="aten::add", description=f"B={batch_size}, S={seq_len}, H={hidden_size}")
        self.B, self.S, self.H = batch_size, seq_len, hidden_size

    def prepare_inputs(self):
        self.inputs = {
            "x": torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=self.device),
            "y": torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=self.device)
        }
        self.cache_flusher = torch.empty(int(50 * 1024 * 1024 / 4), dtype=torch.float32, device=self.device)
    def forward(self, x, y):
        # self.cache_flusher.zero_()
        return x + y

class AllReduceBenchmark(OperatorBenchmark):
    """
    算子 5: AllReduce (多卡数据同步)
    底层: 调用 NCCL 库。时间受限于 NVLink / InfiniBand 带宽以及节点拓扑。
    """
    def __init__(self, num_elements, device="cuda", dtype=torch.bfloat16):
        # 将元素数量转换为大致的 MB 大小，方便在 description 中查看
        size_mb = (num_elements * 2) / (1024 ** 2)  # BF16 占 2 字节
        super().__init__(
            name="nccl::all_reduce", 
            description=f"Elements={num_elements} ({size_mb:.2f} MB)",
            device=device,
            dtype=dtype
        )
        self.num_elements = num_elements

    def prepare_inputs(self):
        # 强制拦截：通信算子必须依赖分布式环境
        if not dist.is_initialized():
            raise RuntimeError(
                "AllReduceBenchmark 必须在分布式环境中运行！\n"
                "请先调用 torch.distributed.init_process_group()，"
                "或者使用 torchrun 启动脚本。"
            )
        

        self.inputs = {
            "x": torch.randn(self.num_elements, dtype=self.dtype, device=self.device)
        }

    def forward(self, x):
        # in-place (就地) 操作进行跨卡求和同步
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
        
    def run(self, min_run_time=0.2, test_backward=False):
        """
        重写 run 方法：为了保证测试准确，NCCL 压测前后建议加入分布式屏障 (Barrier)
        """
        torch.cuda.empty_cache()
        self.prepare_inputs()
        
        # 保证所有卡都在同一起跑线
        dist.barrier()
        
        timer_fwd = benchmark.Timer(
            stmt="fn(**inputs)",
            globals={"fn": self.forward, "inputs": self.inputs},
            label=self.name,
            sub_label="Forward"
        )
        
        # 提取精确的中位数时间
        measurement = timer_fwd.blocked_autorange(min_run_time=min_run_time)
        
        # 测试结束后再次同步
        dist.barrier()
        
        return measurement.median


class AllGatherBenchmark(OperatorBenchmark):
    """
    算子: AllGather (多卡碎片收集)
    """
    def __init__(self, full_elements, device="cuda", dtype=torch.bfloat16):
        # full_elements 是指拼凑完成后的张量总元素数
        size_mb = (full_elements * 2) / (1024 ** 2)
        super().__init__(
            name="nccl::all_gather", 
            description=f"Full Elements={full_elements} ({size_mb:.2f} MB)",
            device=device,
            dtype=dtype
        )
        self.full_elements = full_elements

    def prepare_inputs(self):
        if not dist.is_initialized():
            raise RuntimeError("AllGather 需要分布式环境！")
            
        world_size = dist.get_world_size()
        
        # 必须能被卡数整除
        assert self.full_elements % world_size == 0, "Full elements must be divisible by world_size"
        chunk_size = self.full_elements // world_size
        
        # 1. 本卡的碎片输入 (比如只占 1/8 的数据)
        self.input_tensor = torch.randn(chunk_size, dtype=self.dtype, device=self.device)
        
        # 2. 提前开辟好一块连续的完整内存，用来接收所有人发来的碎片
        self.output_tensor = torch.empty(self.full_elements, dtype=self.dtype, device=self.device)
        
        self.inputs = {
            "out_tensor": self.output_tensor,
            "in_tensor": self.input_tensor
        }

    def forward(self, out_tensor, in_tensor):
        # 直接在连续内存上操作
        dist.all_gather_into_tensor(out_tensor, in_tensor)
        return out_tensor
        
    def run(self, min_run_time=0.2, test_backward=False):
        torch.cuda.empty_cache()
        self.prepare_inputs()
        dist.barrier()
        
        timer_fwd = benchmark.Timer(
            stmt="fn(**inputs)",
            globals={"fn": self.forward, "inputs": self.inputs},
            label=self.name,
            sub_label="Forward"
        )
        
        measurement = timer_fwd.blocked_autorange(min_run_time=min_run_time)
        dist.barrier()
        return measurement.median
