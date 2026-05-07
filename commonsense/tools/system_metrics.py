"""Tiny instrumentation helper for short-horizon SFT runs.

Writes a single sys_metrics.json file capturing steady-state step time,
parameter footprint and peak GPU memory. Designed to be imported from the
three LIFT trainers (finetune_blocktt.py, finetune_lora.py, finetune_sft.py).
"""
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch


class SysMon:
    WARMUP_STEPS = 100  # steps to discard before computing median

    def __init__(self, out_dir, method: str, rank: Optional[int], base_params: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out = self.out_dir / "sys_metrics.json"
        self.method = method
        self.rank = rank
        self.base_params = base_params
        self.step_times: list[float] = []
        self.start_wall = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_step(self, dt: float) -> None:
        self.step_times.append(float(dt))

    def dump(self, model: torch.nn.Module, extra: Optional[dict[str, Any]] = None) -> None:
        if len(self.step_times) > self.WARMUP_STEPS + 10:
            warm = self.step_times[self.WARMUP_STEPS:]
        else:
            warm = self.step_times
        median_step = statistics.median(warm) if warm else None

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        data: dict[str, Any] = {
            "method": self.method,
            "rank": self.rank,
            "base_params": self.base_params,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100.0 * trainable / self.base_params,
            "stored_extra_pct": 100.0 * max(0, total - self.base_params) / self.base_params,
            "steps_recorded": len(self.step_times),
            "median_step_s": median_step,
            "total_wall_s": time.time() - self.start_wall,
            "peak_alloc_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
            ),
        }
        if extra:
            data.update(extra)
        self.out.write_text(json.dumps(data, indent=2))
