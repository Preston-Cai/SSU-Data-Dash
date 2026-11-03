"""Cache correctness checks on tiny arrays, without model training."""
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pandas as pd
import pytest

import experiment_cache
from experiment_cache import ArrayCache, dataframe_digest, source_digest, stable_key


@pytest.fixture(autouse=True)
def enabled_cache(monkeypatch):
    monkeypatch.delenv("SSU_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("SSU_CACHE_DIR", raising=False)


def frame():
    return pd.DataFrame({"sold_price": [500_000.0, 600_000.0],
                         "remarks": ["renovated", "needs work"],
                         "date_sold": pd.to_datetime(["2025-01-01", "2025-02-01"])},
                        index=pd.Index([10, 20], name="listing"))


def test_key_covers_configuration_input_contents_schema_and_order():
    original = frame()
    config = {"cutoff": "2025-03-01", "depth": 8}
    key = stable_key("prediction", config, [original])
    assert key == stable_key("prediction", dict(reversed(list(config.items()))), [original.copy()])
    variants = []
    for column, value in [("sold_price", 500_001.0), ("remarks", "brand new wording"),
                          ("date_sold", pd.Timestamp("2025-01-02"))]:
        changed = original.copy()
        changed.loc[10, column] = value
        variants.append(changed)
    variants.extend([original.iloc[::-1], original.reset_index(drop=True),
                     original.rename(columns={"remarks": "description"}),
                     original.astype({"sold_price": "int64"}),
                     original.rename_axis("another_index"),
                     original[list(reversed(original.columns))]])
    assert all(stable_key("prediction", config, [variant]) != key for variant in variants)
    assert stable_key("prediction", {**config, "cutoff": "2025-04-01"}, [original]) != key
    assert stable_key("prediction", {**config, "depth": 9}, [original]) != key
    assert stable_key("other_namespace", config, [original]) != key
    assert stable_key("prediction", config, [original, original.iloc[:1]]) != stable_key(
        "prediction", config, [original.iloc[:1], original])


def test_categorical_schema_is_part_of_digest():
    original = pd.DataFrame({"kind": pd.Categorical(["a", "b"], categories=["a", "b"])})
    changed = original.copy()
    changed["kind"] = changed["kind"].cat.add_categories(["unused"])
    assert dataframe_digest(original) != dataframe_digest(changed)


def test_cross_instance_reuse_and_noncontiguous_arrays(tmp_path):
    key = stable_key("prediction", {}, [frame()])
    arrays = {"predictions": np.array([500_000.0, np.nan]),
              "features": np.arange(24).reshape(4, 6)[:, ::2],
              "mask": np.array([True, False])}
    ArrayCache(tmp_path).save("prediction", key, arrays, {"rows": 2})
    loaded = ArrayCache(tmp_path).load("prediction", key)
    assert loaded is not None
    assert loaded.keys() == arrays.keys()
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)
        assert loaded[name].dtype == expected.dtype
    assert ArrayCache(tmp_path).load("prediction", "a" * 64) is None


@pytest.mark.parametrize("damage", ["garbage", "truncated", "array", "metadata", "extra_array"])
def test_corrupt_entry_is_a_miss(tmp_path, damage):
    cache = ArrayCache(tmp_path)
    key = stable_key("prediction", {}, [])
    cache.save("prediction", key, {"predictions": np.arange(5.0)})
    path = tmp_path / "prediction" / f"{key}.npz"
    if damage == "garbage":
        path.write_bytes(b"not a zip file")
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if damage == "array":
            arrays["predictions"][0] += 1
        elif damage == "metadata":
            envelope = json.loads(arrays["__cache_metadata__"].tobytes())
            envelope["payload"]["metadata"]["changed"] = True
            arrays["__cache_metadata__"] = np.frombuffer(json.dumps(envelope).encode(), dtype=np.uint8)
        else:
            arrays["unexpected"] = np.arange(2)
        np.savez(path, **arrays)
    assert cache.load("prediction", key) is None


def test_namespace_and_key_cannot_escape_root(tmp_path):
    cache = ArrayCache(tmp_path)
    for namespace in ["../outside", "..", "C:\\outside", "/outside", "a/b"]:
        with pytest.raises(ValueError):
            cache.load(namespace, "a" * 64)
    with pytest.raises(ValueError):
        cache.load("prediction", "../outside")
    with pytest.raises(ValueError):
        cache.save("prediction", "a" * 64, {"../outside": np.arange(2)})
    with pytest.raises(ValueError):
        cache.save("prediction", "a" * 64, {"predictions": np.array([object()])})


def test_environment_configuration_and_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("SSU_CACHE_DIR", str(tmp_path))
    cache = ArrayCache()
    key = stable_key("prediction", {}, [])
    cache.save("prediction", key, {"predictions": np.arange(3)})
    assert cache.root == tmp_path
    monkeypatch.setenv("SSU_DISABLE_CACHE", "1")
    assert cache.load("prediction", key) is None
    cache.save("prediction", "b" * 64, {"predictions": np.arange(3)})
    assert not (tmp_path / "prediction" / f"{'b' * 64}.npz").exists()
    monkeypatch.delenv("SSU_DISABLE_CACHE")
    assert cache.load("prediction", key) is not None


def test_failed_replace_preserves_previous_entry_and_cleans_temporary(tmp_path, monkeypatch):
    cache = ArrayCache(tmp_path)
    key = stable_key("prediction", {}, [])
    cache.save("prediction", key, {"predictions": np.array([1.0])})

    def fail_replace(*args):
        raise OSError("simulated file lock")

    monkeypatch.setattr(experiment_cache.os, "replace", fail_replace)
    with pytest.warns(RuntimeWarning, match="Could not persist"):
        cache.save("prediction", key, {"predictions": np.array([2.0])})
    np.testing.assert_array_equal(cache.load("prediction", key)["predictions"], [1.0])
    assert list((tmp_path / "prediction").iterdir()) == [tmp_path / "prediction" / f"{key}.npz"]


def test_concurrent_writers_publish_only_complete_entries(tmp_path):
    key = stable_key("prediction", {}, [])

    def write_and_read(value):
        cache = ArrayCache(tmp_path)
        cache.save("prediction", key, {"first": np.full(128, value), "second": np.full((16, 8), value)})
        loaded = cache.load("prediction", key)
        # Concurrent replacement can temporarily be locked on Windows; a miss
        # is safe, whereas returning mixed or partially written arrays is not.
        if loaded is not None:
            assert np.all(loaded["first"] == loaded["first"][0])
            assert np.all(loaded["second"] == loaded["first"][0])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_and_read, range(16)))
    assert ArrayCache(tmp_path).load("prediction", key) is not None
    assert len(list((tmp_path / "prediction").iterdir())) == 1


def test_source_digest_changes_only_with_order_or_contents(tmp_path):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("x = 1\n")
    second.write_text("x = 2\n")
    key = source_digest([first, second])
    assert key == source_digest([first, second])
    assert key != source_digest([second, first])
    second.write_text("x = 3\n")
    assert key != source_digest([first, second])
