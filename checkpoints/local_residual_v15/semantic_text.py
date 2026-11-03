"""Frozen BGE remark embeddings, persisted independently of model training.

Only cache misses import/load the optional neural encoder. No prices, dates or
fitted vocabulary enter these embeddings, so the same vectors can be reused
across temporal folds. Long remarks are chunked instead of silently truncated.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Iterable

import numpy as np


MODEL_ID = "BAAI/bge-large-en-v1.5"
# Official repository revision, published 2024-02-21, before all validation sales.
# https://huggingface.co/BAAI/bge-large-en-v1.5/tree/d4aa6901d3a41ba39fb536a557fa166f842b0e09
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
EMBEDDING_DIM = 1024
MAX_TOKENS = 512
CHUNK_OVERLAP = 32
CACHE_PATH = Path(__file__).resolve().parent / "outputs" / "cache" / "semantic_text.sqlite"
_ENCODERS = {}
_SQL_BATCH = 400
_ENCODE_BLOCK = 256


def embedding_metadata() -> dict:
    """Stable, JSON-serializable identity for downstream model/cache keys."""
    return {
        "format_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float32",
        "normalize_embeddings": True,
        "prompt": "",
        "text_preprocessing": "none",
        "max_tokens": MAX_TOKENS,
        "long_text": "character_offset_chunks_token_weighted_mean_l2",
        "chunk_overlap_tokens": CHUNK_OVERLAP,
    }


def model_fingerprint() -> str:
    """Return the immutable encoder/configuration fingerprint."""
    payload = json.dumps(embedding_metadata(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_dependencies() -> None:
    """Check imports without loading weights, before an expensive training run."""
    missing = [name for name in ("sentence_transformers", "torch", "transformers")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Semantic remark embeddings need optional packages: " + ", ".join(missing)
            + ". Install them with: python -m pip install -r requirements-semantic.txt. "
            "Then run python prepare_semantic_text.py to build the reusable cache."
        )
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception as error:
        raise RuntimeError(
            "The semantic encoder dependencies could not be imported. Install compatible "
            "versions with: python -m pip install -r requirements-semantic.txt. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error


def _load_encoder(device: str | None):
    requested = device or os.environ.get("SSU_SEMANTIC_DEVICE") or None
    key = requested or "auto"
    if key not in _ENCODERS:
        check_dependencies()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Could not import the semantic encoder. Install compatible dependencies with "
                "python -m pip install -r requirements-semantic.txt. "
                f"Original error: {error}"
            ) from error
        print(f"Semantic text: loading {MODEL_ID} on {key} (first use may download weights)",
              flush=True)
        encoder = SentenceTransformer(
            MODEL_ID, revision=MODEL_REVISION, device=requested,
            trust_remote_code=False, model_kwargs={"use_safetensors": True},
        )
        encoder.max_seq_length = MAX_TOKENS
        encoder.float()
        encoder.eval()
        encoder.requires_grad_(False)
        if encoder.get_embedding_dimension() != EMBEDDING_DIM:
            raise RuntimeError("The pinned semantic encoder returned an unexpected dimension.")
        _ENCODERS[key] = encoder
    return _ENCODERS[key]


def release_encoder() -> None:
    """Release neural weights/GPU memory before the CatBoost training stage."""
    _ENCODERS.clear()
    gc.collect()
    # Cache-only calls must not import torch merely to release an unused model.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _chunks(texts, encoder):
    """Keep every character-bearing token, including tokens beyond BGE's limit."""
    tokenizer = encoder.tokenizer
    limit = MAX_TOKENS - tokenizer.num_special_tokens_to_add(pair=False)
    tokenized = tokenizer(texts, add_special_tokens=False, truncation=False,
                          return_offsets_mapping=True, verbose=False)
    chunks, owners, weights = [], [], []
    for owner, (text, ids, offsets) in enumerate(zip(
            texts, tokenized["input_ids"], tokenized["offset_mapping"])):
        if len(ids) <= limit:
            chunks.append(text)
            owners.append(owner)
            weights.append(max(len(ids), 1))
            continue
        start = 0
        while start < len(ids):
            end = min(start + limit, len(ids))
            # A chunk beginning in the middle of a word can re-tokenize to a
            # slightly different length; check it before the encoder truncates.
            while True:
                chunk = text[offsets[start][0]:offsets[end - 1][1]]
                count = len(tokenizer(chunk, add_special_tokens=False,
                                      truncation=False, verbose=False)["input_ids"])
                if count <= limit:
                    break
                end -= 1
                if end <= start:
                    raise RuntimeError("Could not split a long remark within the token limit.")
            chunks.append(chunk)
            owners.append(owner)
            weights.append(end - start if start == 0 else max(end - start - CHUNK_OVERLAP, 1))
            if end == len(ids):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks, np.asarray(owners), np.asarray(weights, dtype=np.float32)


def _encode(texts, *, batch_size, device):
    encoder = _load_encoder(device)
    chunks, owners, weights = _chunks(texts, encoder)
    vectors = np.asarray(encoder.encode(
        chunks, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False, prompt="",
    ), dtype=np.float32)
    if vectors.shape != (len(chunks), EMBEDDING_DIM) or not np.isfinite(vectors).all():
        raise RuntimeError("Semantic encoder returned invalid vectors; nothing was cached.")
    pooled = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    np.add.at(pooled, owners, vectors * weights[:, None])
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise RuntimeError("Semantic encoder returned zero/invalid vectors; nothing was cached.")
    return pooled / norms


def cache_path() -> Path:
    """Locate the shared numeric cache; respect the experiment cache override."""
    configured = os.environ.get("SSU_CACHE_DIR")
    return Path(configured).expanduser() / "semantic_text.sqlite" if configured else CACHE_PATH


def _open_cache():
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("""CREATE TABLE IF NOT EXISTS embeddings (
        model_key TEXT NOT NULL,
        text_sha256 TEXT NOT NULL,
        vector BLOB NOT NULL,
        vector_sha256 TEXT NOT NULL,
        PRIMARY KEY (model_key, text_sha256)
    ) WITHOUT ROWID""")
    connection.commit()
    return connection


def _read_vector(blob, checksum):
    if not isinstance(blob, bytes) or len(blob) != EMBEDDING_DIM * 4:
        return None
    if hashlib.sha256(blob).hexdigest() != checksum:
        return None
    vector = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
        return None
    return vector


def get_embeddings(texts: Iterable[str], *, batch_size: int = 32,
                   device: str | None = None) -> np.ndarray:
    """Return finite float32 Nx1024 vectors in input order, reusing exact-text cache.

    Pass strings (use fillna("") at dataframe boundaries). Changing case, spacing,
    negation or any other character creates a different cache entry. Neither the
    sentence encoder nor its vocabulary is fitted to the competition dataset.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of strings, not one string")
    texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError('Every remark must be a string; replace missing values with "" first.')
    output = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    if not texts:
        return output
    positions, unique = {}, {}
    for index, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        positions.setdefault(digest, []).append(index)
        unique[digest] = text
    del texts
    model_key = model_fingerprint()
    missing = []
    connection = _open_cache()
    try:
        keys = list(unique)
        for start in range(0, len(keys), _SQL_BATCH):
            group = keys[start:start + _SQL_BATCH]
            placeholders = ",".join("?" for _ in group)
            rows = connection.execute(
                "SELECT text_sha256, vector, vector_sha256 FROM embeddings "
                f"WHERE model_key=? AND text_sha256 IN ({placeholders})", [model_key, *group])
            hits = set()
            for digest, blob, checksum in rows:
                vector = _read_vector(blob, checksum)
                if vector is not None:
                    output[positions[digest]] = vector
                    hits.add(digest)
            missing.extend(digest for digest in group if digest not in hits)
        if missing:
            print(f"Semantic text: {len(keys) - len(missing):,} cached, "
                  f"{len(missing):,} new unique remarks", flush=True)
        for start in range(0, len(missing), _ENCODE_BLOCK):
            group = missing[start:start + _ENCODE_BLOCK]
            vectors = _encode([unique[digest] for digest in group],
                              batch_size=batch_size, device=device)
            records = []
            for digest, vector in zip(group, vectors):
                blob = np.asarray(vector, dtype="<f4").tobytes()
                records.append((model_key, digest, blob, hashlib.sha256(blob).hexdigest()))
                output[positions[digest]] = vector
            # Commit a resumable block; never hold a writer transaction while encoding.
            with connection:
                connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                                       records)
            print(f"Semantic text: encoded {min(start + len(group), len(missing)):,} / "
                  f"{len(missing):,} new remarks", flush=True)
    finally:
        connection.close()
    return output
