"""Tests for the isolated PyTorch MLP workflow."""

import numpy as np
import pandas as pd
import pytest

import data
from mlp_attempts import exec, features, models


def make_listings(n, seed=0):
    rng = np.random.default_rng(seed)
    sold = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 516, n), unit="D"
    )
    price = rng.uniform(4e5, 2e6, n).round(-2)
    return pd.DataFrame({
        "ml_num": [f"SYN-{i:07d}" for i in range(n)],
        "sold_price": price,
        "list_price": (price * rng.uniform(0.9, 1.1, n)).round(-2),
        "date_listed": sold - pd.to_timedelta(rng.integers(1, 60, n), unit="D"),
        "date_sold": sold,
        "house_type_name": rng.choice(["Condo Apt", "Detached"], n),
        "house_style": rng.choice(["Apartment", "Two Storey"], n),
        "bedrooms": rng.integers(1, 5, n),
        "washrooms": rng.integers(1, 4, n),
        "parking_total": rng.integers(0, 3, n),
        "garage": 0.0,
        "house_area": rng.choice(["600-699", "1200"], n),
        "lot_front": np.nan,
        "lot_depth": np.nan,
        "municipality": "Toronto",
        "community": "Testville",
        "latitude": 43.7 + rng.normal(0, 0.05, n),
        "longitude": -79.4 + rng.normal(0, 0.05, n),
    })


def test_test_workflow_does_not_refit_models(tmp_path, monkeypatch):
    listings = make_listings(200)
    monkeypatch.setattr(data, "load", lambda name: listings)
    monkeypatch.setattr(exec, "ROOT", tmp_path)

    class AlreadyTrainedModel:
        epochs_completed = 3

        def fit(self, train):
            raise AssertionError("test workflow must not refit a checkpoint model")

        def predict(self, homes):
            return np.full(len(homes), 750_000.0)

    for track in exec.TRACKS:
        (tmp_path / f"{track}.pt").write_bytes(b"placeholder")
    monkeypatch.setattr(
        exec.models,
        "make_model",
        lambda track, checkpoint: AlreadyTrainedModel(),
    )
    exec.test(tmp_path)

    predictions = pd.read_csv(tmp_path / "outputs" / "validation_predictions.csv")
    assert (predictions["predicted_price"] == 750_000).all()


def test_checkpoint_path_helpers(tmp_path):
    older = tmp_path / "off_market_20260101-120000-000000.pt"
    newer = tmp_path / "off_market_20260102-120000-000000.pt"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")

    assert exec.latest_checkpoint_path("off_market", tmp_path) == newer
    assert exec.infer_checkpoint_track(newer) == "off_market"
    with pytest.raises(ValueError, match="Cannot infer track"):
        exec.infer_checkpoint_track("mystery.pt")


def test_resolves_two_explicit_test_checkpoints(tmp_path):
    off_market = tmp_path / "off_market_20260101-120000-000000.pt"
    list_aware = tmp_path / "list_aware_20260101-120000-000000.pt"
    off_market.write_bytes(b"off-market")
    list_aware.write_bytes(b"list-aware")

    assert exec.resolve_test_checkpoints(checkpoint_paths=[off_market, list_aware]) == {
        "off_market": off_market,
        "list_aware": list_aware,
    }
    with pytest.raises(ValueError, match="Expected track off_market"):
        exec.resolve_test_checkpoints(checkpoint_paths=[list_aware, off_market])


def test_checkpoint_restores_config_and_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    from mlp_attempts.mlp import OffMarketMLP
    from mlp_attempts.utils import TrainConfig

    listings = make_listings(80)
    config = TrainConfig(
        num_epochs=1,
        learning_rate=0.002,
        device="cpu",
        target_transform="log",
    )
    trained = OffMarketMLP(
        config=config, validation_fraction=0, batch_size=32
    ).fit(listings)
    homes = data.hide_answers(listings.iloc[:10], "off_market")
    expected = trained.predict(homes)

    checkpoint = tmp_path / "off_market.pt"
    trained.save_checkpoint(checkpoint)
    restored = models.make_model("off_market", checkpoint=checkpoint)

    assert type(restored.model) is type(trained.model)
    assert restored.config.learning_rate == pytest.approx(0.002)
    assert restored.preprocessor.columns_ == trained.preprocessor.columns_
    assert torch.equal(
        restored.model.state_dict()["0.weight"],
        trained.model.state_dict()["0.weight"],
    )
    np.testing.assert_allclose(restored.predict(homes), expected)


def test_log_price_encoding_round_trips_to_dollars():
    pytest.importorskip("torch")
    from mlp_attempts.utils import decode_prices, encode_prices

    prices = np.array([400_000.0, 1_250_000.0, 3_000_000.0])
    encoded = encode_prices(prices, target_transform="log")
    np.testing.assert_allclose(
        decode_prices(encoded, target_transform="log"), prices
    )


def test_preprocessor_supports_one_hot_and_id_encodings():
    torch = pytest.importorskip("torch")
    from mlp_attempts.mlp import TabularPreprocessor

    training = pd.DataFrame({
        "house_type_name": ["Condo", "Detached"],
        "house_style": ["Apartment", "Two Storey"],
        "municipality": ["M1", "M2"],
        "community": ["C1", "C2"],
        "bedrooms": [1, 3],
    })
    preprocessor = TabularPreprocessor(
        one_hot_columns=["house_type_name", "house_style"]
    ).fit(training)
    encoded = preprocessor.transform(training)

    assert encoded.shape == (2, 7)
    torch.testing.assert_close(encoded[0, :4], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(encoded[:, 4], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(encoded[:, 5], torch.tensor([1.0, 2.0]))


def test_embedding_network_trains_category_vectors():
    torch = pytest.importorskip("torch")
    from mlp_attempts.mlp import EmbeddedPriceNetwork

    specs = [
        {"column": "municipality", "input_index": 1, "num_embeddings": 3, "embedding_dim": 2},
        {"column": "community", "input_index": 2, "num_embeddings": 4, "embedding_dim": 3},
    ]
    network = EmbeddedPriceNetwork(input_size=3, embedding_specs=specs)
    outputs = network(torch.tensor([[0.5, 1.0, 1.0], [-0.5, 2.0, 3.0]]))
    outputs.sum().backward()

    assert outputs.shape == (2, 1)
    assert network.embeddings["municipality"].weight.grad is not None
    assert network.embeddings["community"].weight.grad is not None


def test_mlp_features_do_not_duplicate_property_type():
    encoded = features.off_market_mlp_features(make_listings(4))

    assert "house_type_name" not in encoded.columns
    assert all(f"is_{kind}" in encoded.columns for kind in features.PROPERTY_TYPES)
    assert {"house_style", "municipality", "community"}.issubset(encoded.columns)
