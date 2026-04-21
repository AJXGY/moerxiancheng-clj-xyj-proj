import math


DEFAULT_CALIBRATION = {
    "mm": {"scale": 1.0},
    "sdpa": {"flash_scale": 1.0, "non_flash_scale": 1.2},
    "rmsnorm": {
        "latency_floor_us": 82.0,
        "streaming_coeff_us_per_elem": 1.2e-4,
        "small_shape_penalty_us": 140.0,
        "small_shape_threshold": 512,
        "scale": 1.0,
    },
    "softmax": {
        "memory_efficiency": 0.55,
        "reduction_coeff_us_per_elem": 1.8e-5,
        "scale": 1.0,
    },
    "add": {"bandwidth_scale": 1.15, "scale": 1.0},
    "ffn": {"scale": 1.0},
}


class PredictorEngine:
    """算子时间预测主引擎：解耦硬件 specs 和算子逻辑"""
    def __init__(self, hw_profile):
        self.peak_tflops = hw_profile.peak_tflops
        self.peak_bw_gbps = hw_profile.peak_bw_gbps
        self.overhead_us = hw_profile.kernel_overhead_us
        self.num_sms = max(1, int(getattr(hw_profile, "num_sms", 108) or 108))
        self.device_name = getattr(hw_profile, "device_name", "unknown")
        self.device_capability = getattr(hw_profile, "device_capability", None)
        self.supports_flash_attention = getattr(hw_profile, "supports_flash_attention", True)
        self.gemm_overhead_us = self.overhead_us
        self.reduction_overhead_us = min(self.overhead_us * 0.18, 24.0)
        self.elementwise_overhead_us = min(self.overhead_us * 0.12, 16.0)
        self.norm_overhead_us = min(self.overhead_us * 0.04, 8.0)
        self.kernels = {}
        self.hw = hw_profile
        self.calibration = self._build_calibration(getattr(hw_profile, "calibration", {}))

    def register_kernel(self, name, kernel_class):
        self.kernels[name] = kernel_class(self)

    def predict_us(self, name, *args, **kwargs):
        if name not in self.kernels:
            raise ValueError(f"算子 '{name}' 未注册！")
        return self.kernels[name].predict_us(*args, **kwargs)

    def _build_calibration(self, overrides):
        calibration = {}
        overrides = overrides or {}
        for kernel_name, defaults in DEFAULT_CALIBRATION.items():
            merged = dict(defaults)
            merged.update(overrides.get(kernel_name, {}))
            calibration[kernel_name] = merged
        return calibration

    def get_kernel_params(self, name):
        return self.calibration.setdefault(name, dict(DEFAULT_CALIBRATION.get(name, {})))

    def update_kernel_from_measurements(self, name, predicted_values, actual_values):
        valid_pairs = [
            (pred, actual)
            for pred, actual in zip(predicted_values, actual_values)
            if pred > 0 and actual > 0
        ]
        if not valid_pairs:
            return
        ratios = sorted(actual / pred for pred, actual in valid_pairs)
        median_ratio = ratios[len(ratios) // 2]
        adjustment = 1.0 + (median_ratio - 1.0) * 0.5
        params = self.get_kernel_params(name)

        if name == "sdpa":
            key = "flash_scale" if self.supports_flash_attention else "non_flash_scale"
            params[key] = max(0.5, min(2.0, params.get(key, 1.0) * adjustment))
        else:
            params["scale"] = max(0.5, min(2.0, params.get("scale", 1.0) * adjustment))

    def export_calibration(self):
        return self.calibration

class BaseKernel:
    def __init__(self, engine:PredictorEngine):
        self.engine = engine
    def predict_us(self, **kwargs):
        raise NotImplementedError

class GEMMKernel(BaseKernel):
    """矩阵乘法 (MM/Linear) 预测逻辑"""
    def predict_us(self, M, K, N):
        params = self.engine.get_kernel_params("mm")
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
        
        raw_us = max(waves * t_wave_us, t_mem_us) + self.engine.gemm_overhead_us
        return raw_us * params.get("scale", 1.0)

class SDPAKernel(BaseKernel):
    """
    修正版 FlashAttention (SDPA) 预测逻辑
    考虑到 S 维度的并行切分
    """
    def predict_us(self, B, H, S, D):
        params = self.engine.get_kernel_params("sdpa")
        bytes_per_elem = 2.0
        tile_s = 128 
        num_s_blocks = math.ceil(S / tile_s)
        total_tasks = B * H * num_s_blocks
        waves = math.ceil(total_tasks / self.engine.num_sms)

        tflops_per_sm = self.engine.peak_tflops / self.engine.num_sms 

        if self.engine.supports_flash_attention:
            bytes_acc = 4 * (B * H * S * D) * bytes_per_elem
            total_flops = 4.0 * B * H * (S**2) * D
            flops_per_task = total_flops / total_tasks
            t_task_us = (flops_per_task / (tflops_per_sm * 1e12)) * 1e6
            t_comp_us = waves * t_task_us / 0.8
            t_mem_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.9 * 1e9)) * 1e6
            raw_us = max(t_comp_us, t_mem_us) + self.engine.reduction_overhead_us
            return raw_us * params.get("flash_scale", 1.0)

        # MP 2.1 这类设备不会走 FlashAttention，这里改用更贴近普通 SDPA 路径的经验模型。
        bytes_acc = (
            3 * B * H * S * D
            + 2 * B * H * S * S
            + B * H * S * D
        ) * bytes_per_elem
        total_flops = 4.0 * B * H * (S**2) * D
        flops_per_task = total_flops / total_tasks
        t_task_us = (flops_per_task / (tflops_per_sm * 1e12)) * 1e6
        t_comp_us = waves * t_task_us * 0.22
        t_mem_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.45 * 1e9)) * 1e6
        raw_us = max(t_comp_us, t_mem_us) * params.get("non_flash_scale", 1.2)
        return raw_us + self.engine.reduction_overhead_us



class FFNKernel(BaseKernel):
    def predict_us(self, B, M, N):
        params = self.engine.get_kernel_params("ffn")
        bytes_acc = (B * M + B * N + M * N) * 2.0
        t_mem_us = (bytes_acc / (self.engine.peak_bw_gbps * 0.9 * 1e9)) * 1e6
        
        flops = 2.0 * B * M * N
        t_comp_us = (flops / (self.engine.peak_tflops * 1e12)) * 1e6
        
        raw_us = max(t_comp_us, t_mem_us) + self.engine.gemm_overhead_us
        return raw_us * params.get("scale", 1.0)




class RMSNormKernel(BaseKernel):
    def __init__(self, engine):
        super().__init__(engine)

    def predict_us(self, B, S, H):
        params = self.engine.get_kernel_params("rmsnorm")
        num_elements = B * S * H
        streaming_us = num_elements * params.get("streaming_coeff_us_per_elem", 1.2e-4)
        threshold = params.get("small_shape_threshold", 512)
        small_shape_penalty_us = params.get("small_shape_penalty_us", 140.0) if S <= threshold else 0.0
        raw_us = (
            params.get("latency_floor_us", 82.0)
            + streaming_us
            + small_shape_penalty_us
            + self.engine.norm_overhead_us
        )
        return raw_us * params.get("scale", 1.0)

class SoftmaxKernel(BaseKernel):
    def predict_us(self, B, S, H):
        params = self.engine.get_kernel_params("softmax")
        num_elements = B * S * H
        bytes_acc = (B * S * H * 2.0) * 2.0 
        flops = 4.0 * num_elements
        t_compute_us = (flops / (self.engine.peak_tflops * 1e12)) * 1e6
        memory_efficiency = params.get("memory_efficiency", 0.55)
        reduction_coeff = params.get("reduction_coeff_us_per_elem", 1.8e-5)
        t_memory_us = (bytes_acc / (self.engine.peak_bw_gbps * memory_efficiency * 1e9)) * 1e6
        t_reduce_us = num_elements * reduction_coeff
        raw_us = max(t_compute_us, t_memory_us, t_reduce_us) + self.engine.elementwise_overhead_us
        return raw_us * params.get("scale", 1.0)


class ADDKernel(BaseKernel):
    """
    张量加法 (Element-wise Addition)
    残差连接算子
    """
    def __init__(self, engine):
        super().__init__(engine)


    def predict_us(self, *shape):
        params = self.engine.get_kernel_params("add")
        """
        shape: 传入张量的维度，比如 (B, S, H)
        """
        # 计算元素总数
        import math
        num_elements = math.prod(shape)
        
        # 1 次加法 FLOP
        flops = num_elements * 1.0

        bytes_acc = num_elements * 3.0 * 2.0  # BF16占2字节
        eff_bw_gbps = self.engine.peak_bw_gbps * params.get("bandwidth_scale", 1.15)
        
        # 3. 耗时计算
        t_memory_us = (bytes_acc / (eff_bw_gbps * 1e9)) * 1e6
        t_compute_us = (flops / (self.engine.peak_tflops * 1e12)) * 1e6
        
        # 访存主导
        t_core_us = max(t_compute_us, t_memory_us)
        
        raw_us = t_core_us + min(self.engine.elementwise_overhead_us * 2.0, 28.0)
        return raw_us * params.get("scale", 1.0)

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
