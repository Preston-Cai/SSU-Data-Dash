"""Shared CatBoost device policy for fresh CPU and GPU installations."""

from functools import lru_cache
import os
import warnings

from catboost.utils import get_gpu_device_count


@lru_cache(maxsize=None)
def _resolve(device, threads):
    if device not in {"AUTO", "CPU", "GPU"}:
        raise ValueError("SSU_DEVICE must be auto, cpu, or gpu")
    if threads < 1:
        raise ValueError("SSU_THREADS must be positive")
    if device == "AUTO":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                device = "GPU" if get_gpu_device_count() > 0 else "CPU"
        except Exception:
            device = "CPU"
    params = {"task_type": device, "thread_count": threads}
    if device == "GPU":
        params["devices"] = "0"
    return params


def catboost_runtime():
    """Return a copy; explicit GPU errors propagate rather than hiding failures."""
    return dict(_resolve(os.environ.get("SSU_DEVICE", "auto").upper(),
                         int(os.environ.get("SSU_THREADS", min(8, os.cpu_count() or 1)))))
