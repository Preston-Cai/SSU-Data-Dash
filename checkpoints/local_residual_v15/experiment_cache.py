"""Content-addressed, process-persistent caches for numeric experiment results.

Each entry is one atomically replaced NPZ file. JSON metadata, array layouts,
and checksums are verified on read; stale or damaged files are cache misses.
No Python objects or pickle payloads are loaded.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import warnings
import zipfile

import numpy as np
import pandas as pd


_FORMAT_VERSION = 1
_META_NAME = "__cache_metadata__"
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_KEY = re.compile(r"[0-9a-f]{64}\Z")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (Path, pd.Period)):
        return str(value)
    raise TypeError(f"Cannot fingerprint value of type {type(value).__name__}")


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False,
                      default=_json_default).encode("utf-8")


def _check_namespace(namespace: str) -> None:
    if not isinstance(namespace, str) or not _NAMESPACE.fullmatch(namespace):
        raise ValueError("Cache namespace must contain only letters, digits, dots, underscores, or hyphens")


def _dtype_spec(dtype) -> dict:
    spec = {"dtype": str(dtype)}
    if isinstance(dtype, pd.CategoricalDtype):
        spec.update(categories=dtype.categories.tolist(), ordered=dtype.ordered)
    return spec


def dataframe_digest(frame: pd.DataFrame) -> str:
    """Hash schema, index, order, and values, including labels if supplied.

    Pass only the frames actually consumed by a computation. Callers are
    responsible for adding model/feature source fingerprints to their config.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    index_dtypes = ([_dtype_spec(level.dtype) for level in frame.index.levels]
                    if isinstance(frame.index, pd.MultiIndex)
                    else [_dtype_spec(frame.index.dtype)])
    schema = {
        "shape": list(frame.shape), "columns": frame.columns.tolist(),
        "column_names": list(frame.columns.names),
        "column_index_type": type(frame.columns).__name__,
        "dtypes": [_dtype_spec(dtype) for dtype in frame.dtypes],
        "index_type": type(frame.index).__name__, "index_names": list(frame.index.names),
        "index_dtypes": index_dtypes,
    }
    digest = hashlib.sha256(_canonical_json(schema))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def stable_key(namespace: str, config: dict, frames: list[pd.DataFrame]) -> str:
    """Create a deterministic key from configuration and ordered input frames."""
    _check_namespace(namespace)
    payload = {"cache_format": _FORMAT_VERSION, "namespace": namespace,
               "config": config, "frames": [dataframe_digest(frame) for frame in frames]}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def source_digest(paths) -> str:
    """Hash source contents in the supplied order; paths themselves are ignored."""
    digest = hashlib.sha256()
    for path in paths:
        contents = Path(path).read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _array_checksum(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _is_numeric(array: np.ndarray) -> bool:
    return (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_))


class ArrayCache:
    """Persist numeric arrays under outputs/cache (or SSU_CACHE_DIR).

    SSU_DISABLE_CACHE=1 disables both reads and writes, including for instances
    created before the environment variable was set. Missing/corrupt entries
    return None. Invalid paths or unsupported input arrays raise ValueError.
    """

    def __init__(self, root=None):
        configured = os.environ.get("SSU_CACHE_DIR")
        self.root = Path(root if root is not None else
                         configured if configured else Path(__file__).parent / "outputs" / "cache")

    @staticmethod
    def _disabled() -> bool:
        return os.environ.get("SSU_DISABLE_CACHE", "0") == "1"

    def _path(self, namespace: str, key: str) -> Path:
        _check_namespace(namespace)
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("Cache key must be a lowercase SHA-256 hex digest")
        return self.root / namespace / f"{key}.npz"

    def load(self, namespace: str, key: str) -> dict[str, np.ndarray] | None:
        path = self._path(namespace, key)
        if self._disabled():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    return None
                encoded = archive[_META_NAME]
                if encoded.dtype != np.uint8 or encoded.ndim != 1:
                    return None
                envelope = json.loads(encoded.tobytes().decode("utf-8"))
                payload = envelope["payload"]
                if hashlib.sha256(_canonical_json(payload)).hexdigest() != envelope["sha256"]:
                    return None
                if (payload["version"] != _FORMAT_VERSION or payload["namespace"] != namespace
                        or payload["key"] != key or not isinstance(payload["metadata"], dict)):
                    return None
                specs = payload["arrays"]
                if not isinstance(specs, dict) or not specs:
                    return None
                if set(archive.files) != set(specs) | {_META_NAME}:
                    return None
                result = {}
                for name, spec in specs.items():
                    if not _ARRAY_NAME.fullmatch(name):
                        return None
                    array = archive[name]
                    if (not _is_numeric(array) or list(array.shape) != spec["shape"]
                            or array.dtype.str != spec["dtype"]
                            or _array_checksum(array) != spec["sha256"]):
                        return None
                    result[name] = array
                return result
        except (OSError, ValueError, TypeError, KeyError, EOFError, AttributeError, zipfile.BadZipFile):
            return None

    def save(self, namespace: str, key: str, arrays: dict,
             metadata: dict | None = None) -> None:
        path = self._path(namespace, key)
        if self._disabled():
            return
        if not isinstance(arrays, dict) or not arrays:
            raise ValueError("Cache entries require a nonempty dictionary of numeric arrays")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Cache metadata must be a dictionary")
        checked, specs = {}, {}
        for name, value in arrays.items():
            if not isinstance(name, str) or not _ARRAY_NAME.fullmatch(name):
                raise ValueError(f"Invalid cache array name: {name!r}")
            array = np.asarray(value)
            if not _is_numeric(array):
                raise ValueError(f"Cache array {name!r} must have numeric or boolean dtype")
            array = np.ascontiguousarray(array)
            checked[name] = array
            specs[name] = {"shape": list(array.shape), "dtype": array.dtype.str,
                           "sha256": _array_checksum(array)}
        payload = {"version": _FORMAT_VERSION, "namespace": namespace, "key": key,
                   "arrays": specs, "metadata": metadata if metadata is not None else {}}
        envelope = {"payload": payload, "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest()}
        checked[_META_NAME] = np.frombuffer(_canonical_json(envelope), dtype=np.uint8)
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{key}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, **checked)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    # Windows readers hold a short-lived handle that can block
                    # replacement. Never remove the old entry to work around it.
                    if attempt == 4:
                        raise
                    time.sleep(0.01 * (2 ** attempt))
            temporary = None
        except OSError as error:
            # A full disk or transient file lock must not discard an expensive
            # completed experiment; the result can still be used in memory.
            warnings.warn(f"Could not persist cache entry {path.name}: {error}", RuntimeWarning)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
