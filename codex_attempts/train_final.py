"""Train frozen finalist ensembles and save timestamped isolated artifacts."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from datetime import datetime
from pathlib import Path

import cloudpickle
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

EXPERIMENTS = HERE / "experiments"
RECENT = runpy.run_path(str(EXPERIMENTS / "recency_encoding_experiment.py"), run_name="final_recent")
TE = runpy.run_path(str(EXPERIMENTS / "target_encoding_experiment.py"), run_name="final_te")
LGB = runpy.run_path(str(EXPERIMENTS / "lightgbm_experiment.py"), run_name="final_lgb")
TEXT = runpy.run_path(str(EXPERIMENTS / "linear_text_experiment.py"), run_name="final_text")
PREDICT = runpy.run_path(str(HERE / "predict_artifact.py"), run_name="final_predict")

WeightedTargetEncoder = RECENT["WeightedTargetEncoder"]
TargetEncoder = TE["TargetEncoder"]
FeatureBuilder = LGB["FeatureBuilder"]
SparseFeatures = TEXT["SparseFeatures"]
target = LGB["target"]
text_target = TEXT["target"]
predict_checkpoint = PREDICT["predict"]

ARTIFACTS = HERE / "artifacts"
CUTOFF = pd.Timestamp("2026-06-01")


def fit_lgb_component(train, track, prior, iterations, objective, learning_rate, recent_half_life=None):
    builder = FeatureBuilder(track, use_remarks=True)
    frame = builder.fit_transform(train)
    regular = TargetEncoder(track, prior)
    frame = pd.concat([frame, regular.fit_transform(train)], axis=1)
    recent = None
    if recent_half_life is not None:
        recent = WeightedTargetEncoder(track, CUTOFF, recent_half_life, prior=5.0)
        frame = pd.concat([frame, recent.fit_transform(train)], axis=1)
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=31,
        min_child_samples=80,
        colsample_bytree=0.9,
        subsample=0.85,
        subsample_freq=1,
        reg_lambda=2.0,
        random_state=0,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        frame,
        target(train, track, residual=True),
        categorical_feature=list(builder.categories),
    )
    return {
        "kind": "lightgbm",
        "builder": builder,
        "regular_encoder": regular,
        "recent_encoder": recent,
        "model": model,
    }


def fit_xgb_component(train):
    track = "off_market"
    builder = FeatureBuilder(track, use_remarks=True)
    frame = builder.fit_transform(train)
    regular = TargetEncoder(track, 5.0)
    recent = WeightedTargetEncoder(track, CUTOFF, 4.0, prior=5.0)
    frame = pd.concat([
        frame,
        regular.fit_transform(train),
        recent.fit_transform(train),
    ], axis=1)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=868,
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        tree_method="hist",
        max_bin=256,
        enable_categorical=True,
        max_cat_to_onehot=20,
        max_cat_threshold=64,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(frame, target(train, track, residual=True), verbose=False)
    return {
        "kind": "xgboost",
        "builder": builder,
        "regular_encoder": regular,
        "recent_encoder": recent,
        "model": model,
    }


def fit_text_component(train, track, alpha):
    builder = SparseFeatures(track, "word", max_features=20_000, min_df=10)
    frame = builder.fit_transform(train)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=2000)
    model.fit(frame, text_target(train, track))
    return {"kind": "text", "builder": builder, "model": model}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    listings = data.load("listings")

    off_text_train = listings[listings.date_sold >= CUTOFF - pd.DateOffset(months=36)].copy()
    list_train = off_text_train.copy()
    print("Training off-market recency LightGBM...", flush=True)
    off_recent = fit_lgb_component(
        listings, "off_market", prior=5.0, iterations=628,
        objective="regression", learning_rate=0.04, recent_half_life=24.0,
    )
    print("Training off-market text Ridge...", flush=True)
    off_text = fit_text_component(off_text_train, "off_market", alpha=1.0)
    print("Training off-market XGBoost...", flush=True)
    off_xgb = fit_xgb_component(listings)

    print("Training list-aware regular LightGBM...", flush=True)
    list_regular = fit_lgb_component(
        list_train, "list_aware", prior=5.0, iterations=127,
        objective="regression_l1", learning_rate=0.05,
    )
    print("Training list-aware recency LightGBM...", flush=True)
    list_recent = fit_lgb_component(
        list_train, "list_aware", prior=100.0, iterations=198,
        objective="regression_l1", learning_rate=0.04, recent_half_life=4.0,
    )
    print("Training list-aware text Ridge...", flush=True)
    list_text = fit_text_component(list_train, "list_aware", alpha=50.0)

    checkpoint = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "training_cutoff": str(CUTOFF.date()),
        "rules": {
            "uses_external_avm": False,
            "off_market_uses_list_price": False,
            "uses_after_sale_features": False,
        },
        "validation": {
            "off_market": {"mape": 9.8465, "mdape": 7.2759},
            "list_aware": {"mape": 5.2525, "mdape": 4.0810},
            "weight_tuning_months": ["2025-09", "2025-10", "2025-11"],
            "official_months": ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"],
        },
        "tracks": {
            "off_market": {
                "components": [off_recent, off_text, off_xgb],
                "weights": [0.15, 0.20, 0.65],
                "calibration": 0.9772745745864185,
            },
            "list_aware": {
                "components": [list_regular, list_recent, list_text],
                "weights": [0.35, 0.35, 0.30],
                "calibration": 0.9865499704372777,
            },
        },
    }
    # Name the artifact at save time, not at the start of a potentially long fit.
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ARTIFACTS / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "ensemble_checkpoint.pkl"
    with checkpoint_path.open("wb") as handle:
        cloudpickle.dump(checkpoint, handle)

    # Native component checkpoints are included for inspection/portability.
    off_xgb["model"].save_model(run_dir / "off_market_xgboost.json")
    off_recent["model"].booster_.save_model(run_dir / "off_market_recency_lightgbm.txt")
    list_regular["model"].booster_.save_model(run_dir / "list_aware_regular_lightgbm.txt")
    list_recent["model"].booster_.save_model(run_dir / "list_aware_recency_lightgbm.txt")
    joblib.dump(off_text["model"], run_dir / "off_market_text_ridge.joblib")
    joblib.dump(list_text["model"], run_dir / "list_aware_text_ridge.joblib")

    for track in ("off_market", "list_aware"):
        homes = data.load(track)
        predicted = predict_checkpoint(checkpoint, homes, track)
        if not np.isfinite(predicted).all() or (predicted <= 0).any():
            raise RuntimeError(f"Invalid {track} predictions")
        pd.DataFrame({"ml_num": homes.ml_num, "predicted_price": np.round(predicted)}).to_csv(
            run_dir / f"submission_{track}.csv", index=False
        )

    files = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "created_at": checkpoint["created_at"],
        "checkpoint": checkpoint_path.name,
        "validation": checkpoint["validation"],
        "weights": {track: checkpoint["tracks"][track]["weights"] for track in checkpoint["tracks"]},
        "calibration": {track: checkpoint["tracks"][track]["calibration"] for track in checkpoint["tracks"]},
        "files": files,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved finalist checkpoint and predictions to {run_dir}")


if __name__ == "__main__":
    main()
