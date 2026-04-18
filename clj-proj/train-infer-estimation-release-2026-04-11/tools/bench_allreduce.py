from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from mvp_backend import default_device_string, distributed_backend_for_device, set_device, synchronize


def main() -> None:
    backend = default_device_string().split(":", 1)[0]
    dist.init_process_group(distributed_backend_for_device(backend))
    rank = dist.get_rank()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    set_device(local_rank, backend)

    results = []
    for num_bytes in [8192, 73728, 262144, 1048576]:
        numel = max(num_bytes // 2, 1)
        tensor = torch.ones(numel, device=backend, dtype=torch.bfloat16)
        for _ in range(5):
            dist.barrier()
            synchronize(backend)
            dist.all_reduce(tensor)
            synchronize(backend)

        local_samples = []
        for _ in range(20):
            dist.barrier()
            synchronize(backend)
            start = time.perf_counter()
            dist.all_reduce(tensor)
            synchronize(backend)
            local_samples.append((time.perf_counter() - start) * 1.0e3)

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_samples)
        if rank == 0:
            worst_case_samples = [max(items) for items in zip(*gathered)]
            results.append(
                {
                    "bytes": num_bytes,
                    "mean_ms": sum(worst_case_samples) / len(worst_case_samples),
                    "min_ms": min(worst_case_samples),
                    "max_ms": max(worst_case_samples),
                }
            )

    if rank == 0:
        output_path = Path("/output/bench.json")
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
