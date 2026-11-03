"""Bounded, process-local reuse of deterministic feature work.

These objects never leave this process. Fitted text transforms and history
lookups are shared read-only; callers copy cached DataFrames before use.
Prediction arrays have their separate persistent cache in experiment_cache.
"""
from collections import OrderedDict
import os

from experiment_cache import dataframe_digest


_LIMITS = {"history": 2, "history_features": 4, "text_fit": 4}
_CACHES = {name: OrderedDict() for name in _LIMITS}


def enabled():
    return os.environ.get("SSU_DISABLE_CACHE", "0") != "1"


def callable_key(function):
    """Identity is sufficient for loaded code in a process-local cache."""
    return getattr(function, "__func__", function)


def frame_key(frame):
    return dataframe_digest(frame)


def get(namespace, key):
    if not enabled():
        return None
    cache = _CACHES[namespace]
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def put(namespace, key, value):
    if not enabled():
        return
    cache = _CACHES[namespace]
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _LIMITS[namespace]:
        cache.popitem(last=False)


def clear():
    for cache in _CACHES.values():
        cache.clear()
