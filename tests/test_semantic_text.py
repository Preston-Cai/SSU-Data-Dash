"""Exercise text cache without optional neural packages or network access."""
import hashlib
import sqlite3

import numpy as np
import pytest

import semantic_text


@pytest.fixture
def fake_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("SSU_CACHE_DIR", str(tmp_path))
    encoded = []

    def encode(texts, **kwargs):
        encoded.extend(texts)
        values = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            vector = np.random.default_rng(seed).normal(size=semantic_text.EMBEDDING_DIM).astype("float32")
            values.append(vector / np.linalg.norm(vector))
        return np.asarray(values)

    monkeypatch.setattr(semantic_text, "_encode", encode)
    return encoded


def test_exact_text_reuse_and_order(fake_encoding):
    a = semantic_text.get_embeddings(["needs renovation", "renovated", "needs renovation"])
    assert fake_encoding == ["needs renovation", "renovated"]
    b = semantic_text.get_embeddings(["renovated", "needs renovation"])
    np.testing.assert_array_equal(b, a[[1, 0]])
    assert len(fake_encoding) == 2
    assert a.dtype == np.float32 and a.shape == (3, 1024)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1, atol=1e-6)


def test_corrupt_cache_recomputed(fake_encoding):
    expected = semantic_text.get_embeddings(["original kitchen"])
    with sqlite3.connect(semantic_text.cache_path()) as connection:
        connection.execute("UPDATE embeddings SET vector=?", (b"broken",))
    actual = semantic_text.get_embeddings(["original kitchen"])
    np.testing.assert_array_equal(actual, expected)
    assert fake_encoding == ["original kitchen", "original kitchen"]


def test_encoder_revision_invalidates_text_cache(fake_encoding, monkeypatch):
    semantic_text.get_embeddings(["updated roof"])
    monkeypatch.setattr(semantic_text, "MODEL_REVISION", "different-revision")
    semantic_text.get_embeddings(["updated roof"])
    assert fake_encoding == ["updated roof", "updated roof"]
