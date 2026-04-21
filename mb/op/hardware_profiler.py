import os
from copy import deepcopy
from datetime import datetime, timezone
import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark
import numpy as np
import yaml


def _dtype_to_name(dtype):
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    return mapping.get(dtype, str(dtype))


def _name_to_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping.get(name, torch.float16)

class HardwareProfiler:
    def __init__(self, device="musa", dtype=torch.bfloat16, cache_path=None):
        self.device = torch.device(device)
        self.dtype = dtype
        self.bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        self.device_name = str(self.device)
        self.num_sms = 108
        self.device_capability = None
        self.supports_flash_attention = True
        self.cache_path = cache_path or os.path.join(
            os.path.dirname(__file__), "device_profiles.yaml"
        )
        self.profile_source = "uninitialized"
        self.measured_at = None
        self.calibration = {}
        
        # 1. 节点内计算与访存极限
        self.peak_tflops = 0.0
        self.peak_bw_gbps = 0.0
        self.kernel_overhead_us = 0.0
        
        # 2. 多卡通信极限 (默认赋理论值兜底，由探针动态覆盖)
        self.intra_node_bw_gbps = 250.0  
        self.inter_node_bw_gbps = 0.0
        self.nccl_latency_us = 10.0      

    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _device_signature(self):
        capability = list(self.device_capability) if self.device_capability is not None else None
        return {
            "device_type": self.device.type,
            "device_name": self.device_name,
            "device_capability": capability,
            "num_sms": self.num_sms,
        }

    def _load_cache_doc(self):
        if not os.path.exists(self.cache_path):
            return {"version": 1, "profiles": []}
        with open(self.cache_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        data.setdefault("version", 1)
        data.setdefault("profiles", [])
        return data

    def _write_cache_doc(self, data):
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)

    def _matches_record(self, record):
        device_info = record.get("device", {})
        if device_info.get("device_type") != self.device.type:
            return False
        if device_info.get("device_name") != self.device_name:
            return False
        record_capability = device_info.get("device_capability")
        if self.device_capability is not None and record_capability is not None:
            if list(self.device_capability) != list(record_capability):
                return False
        return True

    def _find_cached_record(self):
        data = self._load_cache_doc()
        for record in data.get("profiles", []):
            if self._matches_record(record):
                return data, record
        return data, None

    def _apply_record(self, record):
        hardware = record.get("hardware", {})
        self.dtype = _name_to_dtype(hardware.get("dtype", "float16"))
        self.bytes_per_elem = int(hardware.get("bytes_per_elem", self.bytes_per_elem))
        self.peak_tflops = float(hardware.get("peak_tflops", self.peak_tflops))
        self.peak_bw_gbps = float(hardware.get("peak_bw_gbps", self.peak_bw_gbps))
        self.kernel_overhead_us = float(
            hardware.get("kernel_overhead_us", self.kernel_overhead_us)
        )
        self.intra_node_bw_gbps = float(
            hardware.get("intra_node_bw_gbps", self.intra_node_bw_gbps)
        )
        self.inter_node_bw_gbps = float(
            hardware.get("inter_node_bw_gbps", self.inter_node_bw_gbps)
        )
        self.nccl_latency_us = float(hardware.get("nccl_latency_us", self.nccl_latency_us))
        self.measured_at = hardware.get("measured_at")
        self.calibration = deepcopy(record.get("calibration", {}))
        self.profile_source = "cache"

    def _build_record(self):
        return {
            "device": self._device_signature(),
            "hardware": {
                "dtype": _dtype_to_name(self.dtype),
                "bytes_per_elem": self.bytes_per_elem,
                "peak_tflops": float(self.peak_tflops),
                "peak_bw_gbps": float(self.peak_bw_gbps),
                "kernel_overhead_us": float(self.kernel_overhead_us),
                "intra_node_bw_gbps": float(self.intra_node_bw_gbps),
                "inter_node_bw_gbps": float(self.inter_node_bw_gbps),
                "nccl_latency_us": float(self.nccl_latency_us),
                "measured_at": self.measured_at or self._now_iso(),
            },
            "calibration": deepcopy(self.calibration),
        }

    def save_profile(self):
        data, record = self._find_cached_record()
        new_record = self._build_record()
        if record is None:
            data["profiles"].append(new_record)
        else:
            profiles = data.get("profiles", [])
            for idx, existing in enumerate(profiles):
                if self._matches_record(existing):
                    profiles[idx] = new_record
                    break
        self._write_cache_doc(data)

    def set_calibration(self, calibration):
        self.calibration = deepcopy(calibration)

    def _sync_device(self):
        if self.device.type == "musa":
            torch.musa.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    def _device_name(self):
        try:
            if self.device.type == "musa":
                return torch.musa.get_device_name(self.device)
            if self.device.type == "cuda":
                return torch.cuda.get_device_name(self.device)
        except RuntimeError as exc:
            return f"{self.device} (unavailable: {exc})"
        return str(self.device)

    def _load_device_metadata(self):
        self.device_name = self._device_name()
        try:
            if self.device.type == "musa":
                props = torch.musa.get_device_properties(self.device)
                self.num_sms = int(
                    getattr(props, "multi_processor_count", self.num_sms) or self.num_sms
                )
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                if major is not None and minor is not None:
                    self.device_capability = (int(major), int(minor))
                    self.supports_flash_attention = self.device_capability >= (2, 2)
            elif self.device.type == "cuda":
                props = torch.cuda.get_device_properties(self.device)
                self.num_sms = int(
                    getattr(props, "multi_processor_count", self.num_sms) or self.num_sms
                )
                self.device_capability = (int(props.major), int(props.minor))
                self.supports_flash_attention = True
        except RuntimeError:
            pass

    def _time_stmt(self, stmt, globals_dict, min_run_time):
        measurement = benchmark.Timer(stmt=stmt, globals=globals_dict).blocked_autorange(
            min_run_time=min_run_time
        )
        self._sync_device()
        return measurement

    def _measure_peak_tflops(self, local_rank):
        attempts = [
            (self.dtype, 4096),
            (torch.float16, 4096),
            (torch.float16, 2048),
            (torch.float32, 2048),
            (torch.float32, 1024),
        ]
        last_error = None

        for dtype, n in attempts:
            try:
                A = torch.randn(n, n, dtype=dtype, device=self.device)
                B = torch.randn(n, n, dtype=dtype, device=self.device)
                flops = 2.0 * n**3
                t_gemm = self._time_stmt(
                    stmt="torch.matmul(A, B)",
                    globals_dict={"A": A, "B": B, "torch": torch},
                    min_run_time=0.5,
                )
                self.peak_tflops = (flops / t_gemm.median) / 1e12
                self.dtype = dtype
                self.bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
                if local_rank == 0 and dtype != attempts[0][0]:
                    print(
                        f"⚠️ GEMM 探测回退到 dtype={dtype}, N={n}，"
                        "以避开当前设备不支持的内核路径。"
                    )
                return
            except RuntimeError as exc:
                last_error = exc
                if local_rank == 0:
                    print(f"⚠️ GEMM 探测失败 (dtype={dtype}, N={n}): {exc}")

        raise RuntimeError(f"无法完成 GEMM 探测，最后一次错误: {last_error}") from last_error

    def _measure_peak_bandwidth(self):
        n_elem = 1024 * 1024 * 64
        for scale in (1, 2, 4):
            try:
                cur_n_elem = n_elem // scale
                X = torch.randn(cur_n_elem, dtype=self.dtype, device=self.device)
                Y = torch.randn(cur_n_elem, dtype=self.dtype, device=self.device)
                bytes_moved = cur_n_elem * self.bytes_per_elem * 3
                t_bw = self._time_stmt(
                    stmt="X + Y",
                    globals_dict={"X": X, "Y": Y},
                    min_run_time=0.5,
                )
                self.peak_bw_gbps = (bytes_moved / t_bw.median) / 1e9
                return
            except RuntimeError:
                continue
        raise RuntimeError("无法完成显存带宽探测，请检查设备或当前 dtype 是否可用。")
        
    def profile_hardware(self, skip_comm=True, force_remeasure=False):
        """
        探针总入口
        :param skip_comm: 是否强制跳过通信探测 (单卡调试时极度有用)
        """

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._load_device_metadata()

        if not force_remeasure:
            _, cached_record = self._find_cached_record()
            if cached_record is not None:
                self._apply_record(cached_record)
                if local_rank == 0:
                    print(
                        f"📦 命中硬件画像缓存 ({self.device_name})，"
                        f"直接复用 {self.measured_at or '历史'} 的探针结果。"
                    )
                return self
        
        if local_rank == 0:
            print(f"🔍 正在探测单卡计算与访存极限 ({self.device_name})...")
        
        # ==========================================
        # 阶段 1：单卡基础能力探测
        # ==========================================
        
        # 1. 测量 Launch Overhead (使用 benchmark)
        dummy = torch.zeros(1, device=self.device)
        t_overhead = self._time_stmt(
            stmt="x + 1",
            globals_dict={"x": dummy},
            min_run_time=0.3,
        )
        self.kernel_overhead_us = t_overhead.median * 1e6

        # 2. 测量峰值算力 (GEMM)
        self._measure_peak_tflops(local_rank)

        # 3. 测量峰值带宽 (冲刷 L2 Cache)
        self._measure_peak_bandwidth()
        
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
            self.measured_at = self._now_iso()
            self.profile_source = "measured"
            self.save_profile()
            return self

        world_size = dist.get_world_size()
        if world_size > 1:
            if local_rank == 0:
                print(f"🔍 正在探测多卡通信网络 (World Size = {world_size})...")
            self._profile_nccl_full()

        self.measured_at = self._now_iso()
        self.profile_source = "measured"
        self.save_profile()
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
        torch.musa.synchronize()
        dist.barrier()  # 确保所有卡同步起跑
        
        start_events_lat = [torch.musa.Event(enable_timing=True) for _ in range(50)]
        end_events_lat = [torch.musa.Event(enable_timing=True) for _ in range(50)]
        
        for i in range(50):
            start_events_lat[i].record()
            dist.all_reduce(tiny_x, op=dist.ReduceOp.SUM)
            end_events_lat[i].record()
            
        torch.musa.synchronize()
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
        torch.musa.synchronize()
        dist.barrier()
        
        start_events_bw = [torch.musa.Event(enable_timing=True) for _ in range(10)]
        end_events_bw = [torch.musa.Event(enable_timing=True) for _ in range(10)]
        
        for i in range(10):
            start_events_bw[i].record()
            dist.all_reduce(huge_x, op=dist.ReduceOp.SUM)
            end_events_bw[i].record()
            
        torch.musa.synchronize()
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
