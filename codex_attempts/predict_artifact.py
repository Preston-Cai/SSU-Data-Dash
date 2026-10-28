"""Reload a saved isolated ensemble checkpoint and make predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402


def tree_prediction(component: dict, homes: pd.DataFrame, track: str) -> np.ndarray:
    visible = data.hide_answers(homes, track)
    frame = component["builder"].transform(visible)
    frame = pd.concat([frame, component["regular_encoder"].transform(visible)], axis=1)
    if component.get("recent_encoder") is not None:
        frame = pd.concat([frame, component["recent_encoder"].transform(visible)], axis=1)
    raw = component["model"].predict(frame)
    if track == "list_aware":
        return visible.list_price.to_numpy(float) * np.exp(raw)
    return np.exp(raw)


def text_prediction(component: dict, homes: pd.DataFrame, track: str) -> np.ndarray:
    visible = data.hide_answers(homes, track)
    raw = component["model"].predict(component["builder"].transform(visible))
    if track == "list_aware":
        return visible.list_price.to_numpy(float) * np.exp(raw)
    return np.exp(raw)


def predict(checkpoint: dict, homes: pd.DataFrame, track: str) -> np.ndarray:
    track_model = checkpoint["tracks"][track]
    predictions = []
    for component in track_model["components"]:
        if component["kind"] == "text":
            predictions.append(text_prediction(component, homes, track))
        else:
            predictions.append(tree_prediction(component, homes, track))
    weights = np.asarray(track_model["weights"], dtype=float)
    blended = np.exp(sum(weight * np.log(values) for weight, values in zip(weights, predictions)))
    return blended * float(track_model["calibration"])


def load_checkpoint(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return cloudpickle.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    output_dir = args.output_dir or args.checkpoint.parent / "reloaded_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    for track in ("off_market", "list_aware"):
        homes = data.load(track)
        predicted = predict(checkpoint, homes, track)
        pd.DataFrame({"ml_num": homes.ml_num, "predicted_price": np.round(predicted)}).to_csv(
            output_dir / f"submission_{track}.csv", index=False
        )
    print(f"Wrote reloaded predictions to {output_dir}")


if __name__ == "__main__":
    main()

