"""Custom checkpointed MLP workflow.

From the repository folder:

    python -m mlp_attempts.exec train --epochs 10
    python -m mlp_attempts.exec train --resume --epochs 10
    python -m mlp_attempts.exec test
    python -m mlp_attempts.exec report

The starter ``run.py`` remains unchanged. This workflow trains each model once
on sales before the validation window and keeps it frozen for every validation
month.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import data
import run as starter
from mlp_expriments import models


TRACKS = ["off_market", "list_aware"]
VALIDATION_MONTHS = pd.period_range("2025-12", "2026-05", freq="M")
TRAINING_CUTOFF = VALIDATION_MONTHS[0].start_time

# ---------------------------------------------------------------------------
# EDIT TRAINING SETTINGS HERE
# ---------------------------------------------------------------------------
# Shared defaults apply to both models. Put only track-specific differences in
# TRACK_CONFIG. A command-line --epochs value overrides num_epochs for that run.
TRAIN_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-8,
    # "log" trains on log(sold_price) but predictions/metrics remain in dollars.
    "target_transform": "log",
    # Any optimizer class exposed by torch.optim, such as Adam, AdamW, or SGD.
    "optimizer": "SGD",
    # Examples: {"betas": (0.9, 0.99)} for Adam, or {"momentum": 0.9} for SGD.
    "optimizer_kwargs": {},
    "validation_fraction": 0.1,
    "batch_size": 1024,
    "random_seed": 0,
    # Trainable vector widths for the two high-cardinality categories.
    "embedding_dims": {"municipality": 4, "community": 16},
}

TRACK_CONFIG = {
    "off_market": {},
    "list_aware": {},
}


def config_for(track: str, num_epochs: int | None = None) -> dict:
    """Return shared settings plus one track's overrides."""
    config = {**TRAIN_CONFIG, **TRACK_CONFIG[track]}
    if num_epochs is not None:
        config["num_epochs"] = num_epochs
    return config


def apply_config(model, config: dict) -> None:
    """Apply editable settings to a fresh or checkpoint-loaded model."""
    model.config.num_epochs = int(config["num_epochs"])
    model.config.learning_rate = float(config["learning_rate"])
    model.config.weight_decay = float(config["weight_decay"])
    requested_target_transform = str(config["target_transform"])
    if (
        model.model is not None
        and requested_target_transform != model.config.target_transform
    ):
        raise ValueError(
            "Target transform cannot change when resuming. Start a fresh training "
            "run without --resume to switch between raw and log prices."
        )
    model.config.target_transform = requested_target_transform
    model.config.optimizer_name = str(config["optimizer"])
    model.config.optimizer_kwargs = dict(config["optimizer_kwargs"])
    model.validation_fraction = float(config["validation_fraction"])
    model.batch_size = int(config["batch_size"])
    model.random_seed = int(config["random_seed"])
    requested_embedding_dims = dict(config["embedding_dims"])
    if (
        model.model is not None
        and requested_embedding_dims
        and not getattr(model.model, "embedding_specs", [])
    ):
        raise ValueError(
            "This checkpoint predates categorical embeddings. Start a fresh "
            "training run without --resume to train embeddings."
        )
    if model.model is not None and requested_embedding_dims != model.embedding_dims:
        raise ValueError(
            "Embedding dimensions cannot change when resuming. Start a fresh "
            "training run to use the edited embedding_dims."
        )
    model.embedding_dims = requested_embedding_dims

    # Resumed optimizers already have parameter groups. Keep them synchronized
    # so editing learning_rate or weight_decay here takes effect on resume too.
    if model.config.optimizer is not None:
        current_name = type(model.config.optimizer).__name__
        if current_name != model.config.optimizer_name:
            from mlp_expriments.utils import make_optimizer

            print(
                f"Changing optimizer from {current_name} to "
                f"{model.config.optimizer_name}; optimizer state will restart."
            )
            model.config.optimizer = make_optimizer(model.model, model.config)
        else:
            optimizer_settings = {
                **model.config.optimizer_kwargs,
                "lr": model.config.learning_rate,
                "weight_decay": model.config.weight_decay,
            }
            model.config.optimizer.defaults.update(optimizer_settings)
            for group in model.config.optimizer.param_groups:
                group.update(optimizer_settings)


def checkpoint_directory(checkpoint_dir: str | Path | None = None) -> Path:
    return Path(checkpoint_dir) if checkpoint_dir is not None else ROOT / "checkpoints"


def timestamped_checkpoint_path(
    track: str,
    checkpoint_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    return checkpoint_directory(checkpoint_dir) / f"{track}_{timestamp}.pt"


def latest_checkpoint_path(
    track: str, checkpoint_dir: str | Path | None = None
) -> Path | None:
    """Return the newest timestamped checkpoint, with legacy-name fallback."""
    directory = checkpoint_directory(checkpoint_dir)
    candidates = list(directory.glob(f"{track}_*.pt")) if directory.exists() else []
    if candidates:
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    legacy = directory / f"{track}.pt"
    return legacy if legacy.exists() else None


def infer_checkpoint_track(path: str | Path) -> str:
    """Infer and validate a track from a checkpoint filename."""
    name = Path(path).name
    for track in TRACKS:
        if name == f"{track}.pt" or name.startswith(f"{track}_"):
            return track
    raise ValueError(
        f"Cannot infer track from checkpoint filename: {name}. Expected a name "
        "starting with off_market_ or list_aware_."
    )


def train(
    checkpoint_dir: str | Path | None = None,
    tracks: list[str] | None = None,
    num_epochs: int | None = None,
    resume: bool = False,
    resume_from: str | Path | None = None,
) -> bool:
    """Train selected tracks, checkpointing on completion or interruption."""
    tracks = list(tracks or TRACKS)
    listings = data.load("listings")
    training_rows = listings[listings["date_sold"] < TRAINING_CUTOFF]
    stop_message = (
        "Press Q to save and stop."
        if sys.platform == "win32"
        else "Press Ctrl+C to save and stop."
    )
    print(
        f"Training on {len(training_rows):,} sales before "
        f"{TRAINING_CUTOFF:%Y-%m-%d}. {stop_message}"
    )

    exact_resume_path = Path(resume_from) if resume_from is not None else None
    if exact_resume_path is not None:
        if not exact_resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {exact_resume_path}")
        exact_track = infer_checkpoint_track(exact_resume_path)
        if tracks != [exact_track]:
            raise ValueError(
                f"{exact_resume_path.name} is a {exact_track} checkpoint; "
                f"requested tracks were {tracks}."
            )
        if checkpoint_dir is None:
            checkpoint_dir = exact_resume_path.parent

    for track in tracks:
        source_path = exact_resume_path
        if source_path is None and resume:
            source_path = latest_checkpoint_path(track, checkpoint_dir)
        resume_track = source_path is not None
        if resume_track:
            model = models.make_model(track, checkpoint=source_path)
            action = f"Resuming after epoch {model.epochs_completed}"
        else:
            model = models.make_model(track)
            action = "Starting (no checkpoint found)" if resume else "Starting"

        def next_checkpoint_path(track=track):
            return timestamped_checkpoint_path(track, checkpoint_dir)

        config = config_for(track, num_epochs)
        apply_config(model, config)
        effective_config = {
            **config,
            "track": track,
            "device": model.config.device,
            "target_scale": model.config.target_scale,
            "resume": resume_track,
            "resume_from": str(source_path) if source_path is not None else None,
            "checkpoint_directory": str(checkpoint_directory(checkpoint_dir)),
        }
        print(f"Effective training config for {track}:")
        print(json.dumps(effective_config, indent=2, sort_keys=True, default=str))
        print(
            f"{action} {track} training for {config['num_epochs']} epoch(s) "
            f"(lr={config['learning_rate']}, batch={config['batch_size']})."
        )
        try:
            model.fit(
                training_rows,
                checkpoint_path=next_checkpoint_path,
                resume=resume_track,
            )
        except KeyboardInterrupt:
            print("Stopped training. The latest model is safe in its checkpoint.")
            return False
        print(f"Latest {track} checkpoint: {model.last_checkpoint_path}")
    return True


def resolve_test_checkpoints(
    checkpoint_dir: str | Path | None = None,
    checkpoint_paths: list[str | Path] | None = None,
) -> dict[str, Path]:
    """Resolve either two explicit checkpoints or the newest for each track."""
    if checkpoint_paths is not None:
        if len(checkpoint_paths) != len(TRACKS):
            raise ValueError("Provide exactly two checkpoints: off-market, then list-aware.")
        resolved = {}
        for expected_track, supplied_path in zip(TRACKS, checkpoint_paths):
            path = Path(supplied_path)
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {path}")
            actual_track = infer_checkpoint_track(path)
            if actual_track != expected_track:
                raise ValueError(
                    f"Expected track {expected_track}, but {path.name} "
                    f"is a {actual_track} checkpoint."
                )
            resolved[expected_track] = path
        return resolved

    resolved = {}
    for track in TRACKS:
        path = latest_checkpoint_path(track, checkpoint_dir)
        if path is None:
            raise FileNotFoundError(
                f"No {track} checkpoint exists in {checkpoint_directory(checkpoint_dir)}. "
                "Run `python -m mlp_attempts.exec train` first."
            )
        resolved[track] = path
    return resolved


def test(
    checkpoint_dir: str | Path | None = None,
    checkpoint_paths: list[str | Path] | None = None,
) -> None:
    """Test two selected frozen checkpoint models without fitting again."""
    trained_models = {}
    selected_checkpoints = resolve_test_checkpoints(checkpoint_dir, checkpoint_paths)
    for track, path in selected_checkpoints.items():
        trained_models[track] = models.make_model(track, checkpoint=path)
        print(
            f"Loaded {path.name} after "
            f"{trained_models[track].epochs_completed} epoch(s)."
        )

    listings = data.load("listings")
    results = []
    for month in VALIDATION_MONTHS:
        homes = listings[listings["date_sold"].dt.to_period("M") == month]
        for track in TRACKS:
            predicted = trained_models[track].predict(data.hide_answers(homes, track))
            results.append(
                pd.DataFrame(
                    {
                        "ml_num": homes["ml_num"],
                        "track": track,
                        "month": str(month),
                        "house_type_name": homes["house_type_name"],
                        "sold_price": homes["sold_price"],
                        "predicted_price": np.round(predicted),
                    }
                )
            )
        print(f"{month}: predicted {len(homes):,} homes with frozen checkpoints")

    path = ROOT / "outputs" / "validation_predictions.csv"
    path.parent.mkdir(exist_ok=True)
    pd.concat(results).to_csv(path, index=False)
    print(f"Wrote {path.relative_to(ROOT)}")


def report() -> None:
    """Generate the starter report from this workflow's test predictions."""
    original_root = starter.ROOT
    try:
        starter.ROOT = ROOT
        starter.report()
    finally:
        starter.ROOT = original_root


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and checkpoint models")
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override TRAIN_CONFIG['num_epochs'] for this invocation",
    )
    train_parser.add_argument(
        "--track", choices=["all", *TRACKS], default="all", help="track to train"
    )
    train_parser.add_argument("--checkpoint-dir", type=Path, default=None)
    resume_group = train_parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="continue existing checkpoints and start any missing track",
    )
    resume_group.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="continue from this exact checkpoint file",
    )

    test_parser = subparsers.add_parser("test", help="test frozen checkpoints")
    test_source = test_parser.add_mutually_exclusive_group()
    test_source.add_argument("--checkpoint-dir", type=Path, default=None)
    test_source.add_argument(
        "--checkpoints",
        type=Path,
        nargs=2,
        metavar=("OFF_MARKET", "LIST_AWARE"),
        help="exact off-market and list-aware checkpoint files, in that order",
    )
    subparsers.add_parser("report", help="report metrics from test predictions")

    args = parser.parse_args(argv)
    if args.command == "train":
        if args.epochs is not None and args.epochs < 1:
            parser.error("--epochs must be at least 1")
        if args.epochs is None and TRAIN_CONFIG["num_epochs"] < 1:
            parser.error("TRAIN_CONFIG['num_epochs'] must be at least 1")
        tracks = TRACKS if args.track == "all" else [args.track]
        if args.resume_from is not None:
            try:
                checkpoint_track = infer_checkpoint_track(args.resume_from)
            except ValueError as error:
                parser.error(str(error))
            if args.track == "all":
                tracks = [checkpoint_track]
            elif args.track != checkpoint_track:
                parser.error(
                    f"--track {args.track} does not match "
                    f"{args.resume_from.name}"
                )
        return train(
            args.checkpoint_dir,
            tracks,
            args.epochs,
            args.resume,
            args.resume_from,
        )
    if args.command == "test":
        return test(args.checkpoint_dir, args.checkpoints)
    return report()


if __name__ == "__main__":
    try:
        result = main()
    except FileNotFoundError as error:
        sys.exit(str(error))
    sys.exit(1 if result is False else 0)
