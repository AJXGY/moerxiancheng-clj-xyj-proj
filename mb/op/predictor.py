import math

class PredictorEngine:
    """算子时间预测主引擎：解耦硬件 specs 和算子逻辑"""
    def __init__(self, hw_profile):
        self.peak_tflops = hw_profile.peak_tflops
        self.peak_bw_gbps = hw_profile.peak_bw_gbps
        self.overhead_us = hw_profile.kernel_overhead_us
        self.num_sms = 108  # A100 标准，若是其他卡可从 hw_profile 扩展获取
        self.kernels = {}
        self.hw=hw_profile
    def register_kernel(self, name, kernel_class):
        self.kernels[name] = kernel_class(self)

    def predict_us(self, name, *args, **kwargs):
        if name not in self.kernels:
            raise ValueError(f"算子 '{name}' 未注册！")
        return self.kernels[name].predict_us(*args, **kwargs)

class BaseKernel:
    def __init__(self, engine:PredictorEngine):
        self.engine = engine
    def predict_us(self, **kwargs):
        raise NotImplementedError

class GEMMKernel(BaseKernel):
    """矩阵乘法 (MM/Linear) 预测逻辑"""
    def predict_us(self, M, K, N):
        tile_m, tile_n = 128, 128
        
        # 1. 访存预测 (Memory Bound)
        bytes_acc = (M * K + K * N + M * N) * 2.0
        t_mem_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.9 * 1e9)) * 1e6
        
        # 2. 算力预测 (Compute Bound + Wave Quantization)
        grid_m, grid_n = math.ceil(M / tile_m), math.ceil(N / tile_n)
        total_blocks = grid_m * grid_n
        waves = math.ceil(total_blocks / self.engine.num_sms)
        
        flops_per_block = 2.0 * tile_m * tile_n * K
        t_wave_us = (flops_per_block / (self.engine.peak_tflops / self.engine.num_sms * 1e12)) * 1e6
        
        return max(waves * t_wave_us, t_mem_us) + self.engine.overhead_us

class SDPAKernel(BaseKernel):
    """
    修正版 FlashAttention (SDPA) 预测逻辑
    考虑到 S 维度的并行切分
    """
    def predict_us(self, B, H, S, D):
        bytes_acc = 4 * (B * H * S * D) * 2.0
        t_mem_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.9 * 1e9)) * 1e6
        

        tile_s = 128 
        num_s_blocks = math.ceil(S / tile_s)
        
        # 总任务数 = Batch * Heads * S_blocks
        total_tasks = B * H * num_s_blocks
        waves = math.ceil(total_tasks / self.engine.num_sms)
        
        # 总 FLOPs = 4 * B * H * S^2 * D
        # 分摊到每个 task
        flops_per_task = (4.0 * B * H * (S**2) * D) / total_tasks 
        
        # 每一个 SM 的理论产出
        tflops_per_sm = self.engine.peak_tflops / self.engine.num_sms 
        
        t_task_us = (flops_per_task / (tflops_per_sm * 1e12)) * 1e6
        t_comp_us = waves * t_task_us  / 0.8  # 考虑 Tensor Core 的效率损失
        
        # 3. 结果合并
        # 注意：SDPA 随着 S 增加，计算量按 S^2 增长，而访存按 S 增长，所以 S 大了必然是 Compute Bound
        return max(t_comp_us, t_mem_us) + self.engine.overhead_us



class FFNKernel(BaseKernel):
    def predict_us(self, B, M, N):
        bytes_acc = (B * M + B * N + M * N) * 2.0
        t_mem_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.9 * 1e9)) * 1e6
        
        flops = 2.0 * B * M * N
        t_comp_us = (flops / (self.engine.peak_tflops * 1e12)) * 1e6
        
        return max(t_comp_us, t_mem_us) + self.engine.overhead_us




class RMSNormKernel(BaseKernel):
    def __init__(self, engine):
        super().__init__(engine)
        # 校准值：算子在 GPU 上的物理拉起延迟 (可以通过 M=1 时的实测值扣除 overhead 得到)
        # 在 A100 上，复杂的 Norm 算子通常有 30-40us 的固定成本
        self.latency_floor_us = 35.0 

    def predict_us(self, B, S, H):
        # 1. 理论计算量
        num_elements = B * S * H
        bytes_acc = num_elements * 2 * 2 # BF16, 1读1写
        
        # 2. 显存带宽耗时 (HBM)
        t_hbm_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.9 * 1e9)) * 1e6
        
        t_core_us = self.latency_floor_us + t_hbm_us
        
        return t_core_us + self.engine.overhead_us

class SoftmaxKernel(BaseKernel):
    def predict_us(self, B, S, H):
        # Bytes: 读输入 X, 写输出 Y 
        bytes_acc = (B * S * H * 2.0) * 2.0 
        
        # FLOPs: 平方、求和、除法、乘法，平均每个元素约 4 个 FLOPs
        flops = 4.0 * B * S * H 
        
        # 朴素 Roofline 公式
        t_compute_us = (flops / (self.engine.peak_tflops * 1e12)) * 1e6
        t_memory_us = (bytes_acc / (self.engine.peak_bw_gbps * 1e9)) * 1e6
        
        return max(t_compute_us, t_memory_us) + self.engine.overhead_us


class ADDKernel(BaseKernel):
    """
    张量加法 (Element-wise Addition)
    残差连接算子
    """
    def __init__(self, engine):
        super().__init__(engine)


    def predict_us(self, *shape):
        """
        shape: 传入张量的维度，比如 (B, S, H)
        """
        # 计算元素总数
        import math
        num_elements = math.prod(shape)
        
        # 1 次加法 FLOP
        flops = num_elements * 1.0

        bytes_acc = num_elements * 3.0 * 2.0  # BF16占2字节
        eff_bw_gbps = self.engine.peak_bw_gbps 
        
        # 3. 耗时计算
        t_memory_us = (bytes_acc / (eff_bw_gbps * 1e9)) * 1e6
        t_compute_us = (flops / (self.engine.peak_tflops * 1e12)) * 1e6
        
        # 访存主导
        t_core_us = max(t_compute_us, t_memory_us)
        
        return t_core_us + self.engine.overhead_us

class AllReduceKernel(BaseKernel):
    """
    基于 Ring-AllReduce / NCCL 算法的通信时间预测器
    """
    def predict_us(self, D_bytes, N_gpus):
        """
        D_bytes: 需要 AllReduce 的张量总大小 (字节)
        N_gpus: 参与通信的 GPU 数量 (World Size)
        """
        if N_gpus <= 1:
            return 0.0
            
        # 搬运的数据量 
        # 这里算的是单卡需要发送和接收的总数据量
        bytes_moved = 2.0 * ((N_gpus - 1) / N_gpus) * D_bytes
        
        eff_bw = self.engine.hw.intra_node_bw_gbps

        # 时间 = 数据量 / 带宽 + 延迟
        t_comm_us = (bytes_moved / (eff_bw * 1e9)) * 1e6
        
        return t_comm_us + self.engine.hw.nccl_latency_us

class AllGatherKernel(BaseKernel):
    """
    AllGather 通信算子预测器 (常用于 FSDP 权重收集 / 序列并行)
    """
    def predict_us(self, D_bytes, N_gpus):
        if N_gpus <= 1:
            return 0.0
            
        # 1. 真实搬运的数据量 (去掉了 2.0 倍数)
        # 这里的 D_bytes 是指 Gather 完毕后完整张量的总大小
        bytes_moved = ((N_gpus - 1) / N_gpus) * D_bytes
        
        eff_bw = self.engine.hw.intra_node_bw_gbps *0.9

            
        # 3. 耗时计算 (微秒)
        t_comm_us = (bytes_moved / (eff_bw * 1e9)) * 1e6
        
        return t_comm_us + self.engine.hw.nccl_latency_us