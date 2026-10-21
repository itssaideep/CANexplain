"""Computational resource profiler for detection models."""

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np


class ResourceProfiler:
    """Measures inference latency, peak memory, and model serialization footprint."""

    @staticmethod
    def profile_inference(
        model_score_fn: Callable[[np.ndarray], np.ndarray],
        windows: np.ndarray,
        num_warmup: int = 5,
    ) -> dict[str, float]:
        """Profiles inference latency per window in milliseconds."""
        n_windows = len(windows)
        if n_windows == 0:
            return {"latency_ms_per_window": 0.0, "total_time_sec": 0.0}

        # Warmup
        if num_warmup > 0 and n_windows >= num_warmup:
            _ = model_score_fn(windows[:num_warmup])

        t0 = time.perf_counter()
        _ = model_score_fn(windows)
        total_time = time.perf_counter() - t0

        latency_ms = (total_time / n_windows) * 1000.0

        return {
            "latency_ms_per_window": float(latency_ms),
            "total_inference_time_sec": float(total_time),
            "num_windows_evaluated": int(n_windows),
        }

    @staticmethod
    def profile_training(
        train_fn: Callable[[], Any],
    ) -> tuple[Any, dict[str, float]]:
        """Profiles wall-clock training duration and peak memory during model fitting."""
        tracemalloc.start()
        t0 = time.perf_counter()

        result = train_fn()

        train_time = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = {
            "training_time_sec": float(train_time),
            "peak_memory_mb": float(peak_bytes / (1024 * 1024)),
        }
        return result, metrics
