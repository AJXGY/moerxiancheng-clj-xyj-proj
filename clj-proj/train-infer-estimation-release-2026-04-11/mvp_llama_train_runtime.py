from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.llama.modeling_llama import create_causal_mask


def _device_str(backend: str, device_id: int) -> str:
    return "cpu" if backend == "cpu" else f"{backend}:{device_id}"


def _synchronize(backend: str, device_ids: list[int]) -> None:
    if backend == "musa" and hasattr(torch, "musa"):
        for device_id in device_ids:
            torch.musa.synchronize(device_id)
    elif backend == "cuda":
        for device_id in device_ids:
            torch.cuda.synchronize(device_id)


def _stable_summary(vals: list[float], runs: int, warmups: int) -> dict:
    median_val = statistics.median(vals)
    stable_cutoff = median_val * 0.8
    stable_vals = [value for value in vals if value >= stable_cutoff]
    if not stable_vals:
        stable_vals = list(vals)
    return {
        "profile_kind": "online_llama_backbone_probe",
        "timings_ms": vals,
        "avg_ms": sum(stable_vals) / len(stable_vals),
        "median_ms": median_val,
        "min_ms": min(vals),
        "max_ms": max(vals),
        "runs": runs,
        "warmups": warmups,
        "stable_cutoff_ms": stable_cutoff,
        "stable_timings_ms": stable_vals,
        "stable_count": len(stable_vals),
    }


def _gather_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_indices = attention_mask.long().sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_indices]


def _load_samples(samples_path: str) -> list[dict]:
    samples: list[dict] = []
    for line in Path(samples_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        samples.append(json.loads(line))
    if not samples:
        raise RuntimeError(f"No samples found in {samples_path}")
    return samples


@dataclass
class LlamaTrainRuntime:
    model_path: str
    samples_path: str
    device_backend: str
    pipeline_parallel_size: int
    tensor_parallel_size: int = 1
    max_seq_len: int = 8
    split_index: int = 16

    def __post_init__(self) -> None:
        self.device0 = _device_str(self.device_backend, 0)
        self.device1 = _device_str(self.device_backend, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.samples = self._encode_samples(_load_samples(self.samples_path))
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device_backend != "cpu" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.num_labels = max(int(item["label"]) for item in self.samples) + 1

        if self.pipeline_parallel_size <= 1 and self.tensor_parallel_size <= 1:
            self.model.to(self.device0)
            self.head = torch.nn.Linear(
                self.model.config.hidden_size,
                self.num_labels,
                bias=True,
                device=self.device0,
                dtype=torch.float16 if self.device_backend != "cpu" else torch.float32,
            )
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-3)
        elif self.pipeline_parallel_size > 1:
            self.model.embed_tokens.to(self.device0)
            for index, layer in enumerate(self.model.layers):
                layer.to(self.device0 if index < self.split_index else self.device1)
            self.model.norm.to(self.device1)
            self.head = torch.nn.Linear(
                self.model.config.hidden_size,
                self.num_labels,
                bias=True,
                device=self.device1,
                dtype=torch.float16 if self.device_backend != "cpu" else torch.float32,
            )
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-3)
        else:
            self.model.to(self.device0)
            shard_out_features = max(1, (self.num_labels + self.tensor_parallel_size - 1) // self.tensor_parallel_size)
            self.tp_heads = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        self.model.config.hidden_size,
                        shard_out_features,
                        bias=True,
                        device=self.device0 if shard_index == 0 else self.device1,
                        dtype=torch.float16 if self.device_backend != "cpu" else torch.float32,
                    )
                    for shard_index in range(self.tensor_parallel_size)
                ]
            )
            self.optimizer = torch.optim.SGD(self.tp_heads.parameters(), lr=1e-3)

    def _encode_samples(self, samples: list[dict]) -> list[dict]:
        encoded_samples = []
        for item in samples:
            encoded = self.tokenizer(
                str(item["text"]),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
            )
            encoded_samples.append(
                {
                    "input_ids": encoded["input_ids"][0].cpu(),
                    "attention_mask": encoded["attention_mask"][0].cpu(),
                    "label": int(item["label"]),
                }
            )
        return encoded_samples

    def _batch(self, microbatch_index: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = (microbatch_index * batch_size) % len(self.samples)
        picked = [self.samples[(start + offset) % len(self.samples)] for offset in range(batch_size)]
        input_ids = torch.stack([item["input_ids"] for item in picked], dim=0).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in picked], dim=0).to(device)
        labels = torch.tensor([item["label"] for item in picked], device=device, dtype=torch.long)
        return input_ids, attention_mask, labels

    def _run_pp1_microbatch(self, microbatch_index: int, batch_size: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self._batch(microbatch_index, batch_size, self.device0)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            pooled = _gather_last_token(outputs.last_hidden_state, attention_mask).detach()
        logits = self.head(pooled)
        return torch.nn.functional.cross_entropy(logits.float(), labels)

    def _run_pp2_microbatch(self, microbatch_index: int, batch_size: int) -> torch.Tensor:
        input_ids0, attention_mask0, _ = self._batch(microbatch_index, batch_size, self.device0)
        with torch.no_grad():
            hidden_states = self.model.embed_tokens(input_ids0)
            position_ids0 = torch.arange(hidden_states.shape[1], device=self.device0).unsqueeze(0)
            causal_mask0 = create_causal_mask(
                self.model.config,
                hidden_states,
                attention_mask0,
                past_key_values=None,
                position_ids=position_ids0,
            )
            position_embeddings0 = self.model.rotary_emb(hidden_states, position_ids=position_ids0)
            for layer_index in range(self.split_index):
                hidden_states = self.model.layers[layer_index](
                    hidden_states,
                    attention_mask=causal_mask0,
                    position_embeddings=position_embeddings0,
                    position_ids=position_ids0,
                    past_key_values=None,
                    use_cache=False,
                )

            hidden_states1 = hidden_states.to("cpu").to(self.device1)
            attention_mask1 = attention_mask0.to("cpu").to(self.device1)
            position_ids1 = torch.arange(hidden_states1.shape[1], device=self.device1).unsqueeze(0)
            causal_mask1 = create_causal_mask(
                self.model.config,
                hidden_states1,
                attention_mask1,
                past_key_values=None,
                position_ids=position_ids1,
            )
            position_embeddings1 = self.model.rotary_emb(hidden_states1, position_ids=position_ids1)
            for layer_index in range(self.split_index, self.model.config.num_hidden_layers):
                hidden_states1 = self.model.layers[layer_index](
                    hidden_states1,
                    attention_mask=causal_mask1,
                    position_embeddings=position_embeddings1,
                    position_ids=position_ids1,
                    past_key_values=None,
                    use_cache=False,
                )
            hidden_states1 = self.model.norm(hidden_states1)
            pooled = _gather_last_token(hidden_states1, attention_mask1).detach()

        _, _, labels1 = self._batch(microbatch_index, batch_size, self.device1)
        logits = self.head(pooled)
        return torch.nn.functional.cross_entropy(logits.float(), labels1)

    def _run_tp2_microbatch(self, microbatch_index: int, batch_size: int) -> torch.Tensor:
        input_ids, attention_mask, labels = self._batch(microbatch_index, batch_size, self.device0)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            pooled = _gather_last_token(outputs.last_hidden_state, attention_mask).detach()

        pooled_rank0 = pooled
        pooled_rank1 = pooled.to("cpu").to(self.device1)
        shard_logits = [
            self.tp_heads[0](pooled_rank0),
            self.tp_heads[1](pooled_rank1).to("cpu").to(self.device0),
        ]
        logits = torch.cat(shard_logits, dim=-1)[:, : self.num_labels]
        return torch.nn.functional.cross_entropy(logits.float(), labels)

    def train_iteration(self, microbatch_num: int, global_batch_size: int) -> None:
        batch_size = max(1, int(global_batch_size) // max(1, int(microbatch_num)))
        self.optimizer.zero_grad(set_to_none=True)
        for microbatch_index in range(int(microbatch_num)):
            if self.pipeline_parallel_size <= 1 and self.tensor_parallel_size <= 1:
                loss = self._run_pp1_microbatch(microbatch_index, batch_size)
            elif self.pipeline_parallel_size <= 1:
                loss = self._run_tp2_microbatch(microbatch_index, batch_size)
            else:
                loss = self._run_pp2_microbatch(microbatch_index, batch_size)
            (loss / float(microbatch_num)).backward()
        self.optimizer.step()
        if self.pipeline_parallel_size <= 1 and self.tensor_parallel_size <= 1:
            _synchronize(self.device_backend, [0])
        elif self.pipeline_parallel_size <= 1:
            _synchronize(self.device_backend, [0, 1])
        else:
            _synchronize(self.device_backend, [0, 1])


def benchmark_llama_backbone_probe(
    model_path: str,
    samples_path: str,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
    microbatch_num: int,
    global_batch_size: int,
    device_backend: str,
    runs: int,
    warmups: int,
    max_seq_len: int = 8,
    split_index: int = 16,
) -> dict:
    runtime = LlamaTrainRuntime(
        model_path=model_path,
        samples_path=samples_path,
        device_backend=device_backend,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        max_seq_len=max_seq_len,
        split_index=split_index,
    )
    return benchmark_runtime(runtime, microbatch_num, global_batch_size, runs, warmups)


def benchmark_runtime(
    runtime: LlamaTrainRuntime,
    microbatch_num: int,
    global_batch_size: int,
    runs: int,
    warmups: int,
) -> dict:
    for _ in range(warmups):
        runtime.train_iteration(microbatch_num=microbatch_num, global_batch_size=global_batch_size)

    timings_ms: list[float] = []
    for _ in range(runs):
        if runtime.pipeline_parallel_size <= 1 and runtime.tensor_parallel_size <= 1:
            _synchronize(runtime.device_backend, [0])
        else:
            _synchronize(runtime.device_backend, [0, 1])
        start = time.perf_counter()
        runtime.train_iteration(microbatch_num=microbatch_num, global_batch_size=global_batch_size)
        if runtime.pipeline_parallel_size <= 1 and runtime.tensor_parallel_size <= 1:
            _synchronize(runtime.device_backend, [0])
        else:
            _synchronize(runtime.device_backend, [0, 1])
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    return _stable_summary(timings_ms, runs=runs, warmups=warmups)
