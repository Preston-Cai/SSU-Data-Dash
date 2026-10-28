"""PyTorch data loading, training, and evaluation helpers for price regression."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def _keyboard_stop_requested() -> bool:
    """Return true when Q is pressed in an interactive Windows console.

    This avoids Ctrl+C, which some native BLAS/Fortran runtimes intercept and
    abort before Python can save a checkpoint.
    """
    if sys.platform != "win32":
        return False
    import msvcrt

    while msvcrt.kbhit():
        if msvcrt.getwch().lower() == "q":
            return True
    return False


class FeatureDataset(Dataset):
    """A tensor dataset that also supports prediction data without targets."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor | None = None):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.targets is None:
            return self.inputs[index]
        return self.inputs[index], self.targets[index]


def make_dataloader(
    X,
    targets,
    preprocessor,
    *,
    fit_preprocessor: bool = False,
    batch_size: int = 1024,
    shuffle: bool = False,
    target_scale: float = 1_000_000.0,
    target_transform: str = "identity",
) -> DataLoader:
    """Transform feature rows and return a regression/prediction DataLoader."""
    inputs = preprocessor.fit_transform(X) if fit_preprocessor else preprocessor.transform(X)
    target_tensor = None
    if targets is not None:
        values = encode_prices(
            np.asarray(targets, dtype=np.float32),
            target_scale=target_scale,
            target_transform=target_transform,
        )
        target_tensor = torch.as_tensor(values, dtype=torch.float32)
    dataset = FeatureDataset(inputs, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_train_validation_loaders(
    X,
    targets,
    preprocessor,
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
    batch_size: int = 1024,
    target_scale: float = 1_000_000.0,
    target_transform: str = "identity",
    fit_preprocessor: bool = True,
) -> tuple[DataLoader, DataLoader | None]:
    """Split rows, fit preprocessing on training rows, and build both loaders."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if len(X) != len(targets):
        raise ValueError("X and targets must have the same number of rows.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    validation_size = int(round(len(X) * validation_fraction))
    if validation_fraction > 0 and len(X) > 1:
        validation_size = min(max(validation_size, 1), len(X) - 1)
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    X_train = X.iloc[train_indices]
    y_array = np.asarray(targets)
    train_loader = make_dataloader(
        X_train,
        y_array[train_indices],
        preprocessor,
        fit_preprocessor=fit_preprocessor,
        batch_size=batch_size,
        shuffle=True,
        target_scale=target_scale,
        target_transform=target_transform,
    )
    if validation_size == 0:
        return train_loader, None

    validation_loader = make_dataloader(
        X.iloc[validation_indices],
        y_array[validation_indices],
        preprocessor,
        batch_size=batch_size,
        target_scale=target_scale,
        target_transform=target_transform,
    )
    return train_loader, validation_loader


@dataclass
class TrainConfig:
    """Configuration and accumulated state for one regression training run."""

    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    criterion_instance: nn.Module = field(default_factory=nn.MSELoss)
    optimizer_name: str = "Adam"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer: optim.Optimizer | None = None
    scheduler: Any = None
    validation_dataloader: DataLoader | None = None
    validate_freq: int = 1
    train_log_freq: int = 1
    target_scale: float = 1_000_000.0
    target_transform: str = "identity"
    best_validation_loss: float = float("inf")
    best_validation_mape: float = float("inf")


def encode_prices(
    prices: np.ndarray,
    *,
    target_scale: float = 1_000_000.0,
    target_transform: str = "identity",
) -> np.ndarray:
    """Encode dollar prices into the values optimized by the network."""
    if target_transform == "identity":
        return prices / target_scale
    if target_transform == "log":
        if np.any(prices <= 0):
            raise ValueError("Log-price training requires every price to be positive.")
        return np.log(prices)
    raise ValueError(f"Unknown target transform: {target_transform}")


def decode_prices(
    values: np.ndarray,
    *,
    target_scale: float = 1_000_000.0,
    target_transform: str = "identity",
) -> np.ndarray:
    """Convert model target values back into dollar prices."""
    if target_transform == "identity":
        return values * target_scale
    if target_transform == "log":
        return np.exp(values)
    raise ValueError(f"Unknown target transform: {target_transform}")


@dataclass
class LoadedCheckpoint:
    """Objects reconstructed from a :class:`Checkpointer` checkpoint."""

    model: nn.Module
    config: TrainConfig
    extra: dict[str, Any]
    saved_at: str | None = None


class Checkpointer:
    """Save and restore a model, its class, and its training configuration.

    Checkpoints contain import paths and state dictionaries instead of pickled
    Python objects. This keeps them compatible with PyTorch's safer
    ``weights_only=True`` loader.
    """

    _RUNTIME_CONFIG_FIELDS = {
        "criterion_instance",
        "optimizer",
        "scheduler",
        "validation_dataloader",
    }

    @staticmethod
    def _class_path(value: object) -> str:
        cls = value if isinstance(value, type) else type(value)
        return f"{cls.__module__}:{cls.__qualname__}"

    @staticmethod
    def _load_class(path: str):
        module_name, qualname = path.split(":", 1)
        # Checkpoints created before the files were organized under
        # ``mlp_attempts`` store classes as ``mlp:ClassName``.
        module_name = {
            "mlp": "mlp_attempts.mlp",
            "utils": "mlp_attempts.utils",
        }.get(module_name, module_name)
        value = importlib.import_module(module_name)
        for part in qualname.split("."):
            value = getattr(value, part)
        return value

    @classmethod
    def save(
        cls,
        path: str | Path,
        model: nn.Module,
        config: TrainConfig,
        *,
        model_kwargs: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_values = {
            item.name: getattr(config, item.name)
            for item in fields(config)
            if item.name not in cls._RUNTIME_CONFIG_FIELDS
        }
        optimizer = None
        if config.optimizer is not None:
            optimizer = {
                "class": cls._class_path(config.optimizer),
                "defaults": dict(config.optimizer.defaults),
                "state_dict": config.optimizer.state_dict(),
            }
        torch.save(
            {
                "format_version": 1,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "model_class": cls._class_path(model),
                "model_kwargs": model_kwargs or {},
                "model_state_dict": model.state_dict(),
                "config": config_values,
                "criterion_class": cls._class_path(config.criterion_instance),
                "criterion_state_dict": config.criterion_instance.state_dict(),
                "optimizer": optimizer,
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> LoadedCheckpoint:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError(f"Unsupported checkpoint format in {path}.")

        model_class = cls._load_class(checkpoint["model_class"])
        model = model_class(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])

        criterion_class = cls._load_class(checkpoint["criterion_class"])
        criterion = criterion_class()
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        config = TrainConfig(**checkpoint["config"], criterion_instance=criterion)

        # Put parameters on their training device before optimizer state is
        # restored, so momentum/variance tensors follow the same device.
        if str(config.device).startswith("cuda") and not torch.cuda.is_available():
            config.device = "cpu"
        model.to(config.device)

        optimizer_checkpoint = checkpoint.get("optimizer")
        if optimizer_checkpoint is not None:
            optimizer_class = cls._load_class(optimizer_checkpoint["class"])
            config.optimizer = optimizer_class(
                model.parameters(), **optimizer_checkpoint["defaults"]
            )
            config.optimizer.load_state_dict(optimizer_checkpoint["state_dict"])
        return LoadedCheckpoint(
            model=model,
            config=config,
            extra=checkpoint["extra"],
            saved_at=checkpoint.get("saved_at"),
        )


def calculate_errors(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return the project regression errors, expressed as percentages."""
    signed = 100.0 * (predictions - targets) / targets
    absolute = np.abs(signed)
    return {
        "mape": float(np.mean(absolute)),
        "mdape": float(np.median(absolute)),
        "median_signed_error": float(np.median(signed)),
    }


def _log_evaluation(name: str, epoch_num: int | None, metrics: dict[str, float]) -> None:
    epoch = f" for epoch {epoch_num}" if epoch_num is not None else ""
    print(
        f"{name} result{epoch}: loss {metrics['loss']:.6f}, "
        f"MAPE {metrics['mape']:.2f}%, MdAPE {metrics['mdape']:.2f}%, "
        f"median signed error {metrics['median_signed_error']:+.2f}%"
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_instance: nn.Module | None = None,
    *,
    device: str = "cpu",
    target_scale: float = 1_000_000.0,
    target_transform: str = "identity",
    name: str = "Evaluation",
    epoch_num: int | None = None,
) -> dict[str, float]:
    """Evaluate labeled data using the same price errors used by validation."""
    criterion = criterion_instance or nn.MSELoss()
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, batch_targets in dataloader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            outputs = model(inputs).squeeze(-1)
            total_loss += criterion(outputs, batch_targets).item() * len(inputs)
            num_samples += len(inputs)
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    if was_training:
        model.train()
    if num_samples == 0:
        raise ValueError("Cannot evaluate an empty dataloader.")

    predicted_prices = decode_prices(
        np.concatenate(predictions),
        target_scale=target_scale,
        target_transform=target_transform,
    )
    sold_prices = decode_prices(
        np.concatenate(targets),
        target_scale=target_scale,
        target_transform=target_transform,
    )
    metrics = {
        "loss": total_loss / num_samples,
        **calculate_errors(predicted_prices, sold_prices),
    }
    _log_evaluation(name, epoch_num, metrics)
    return metrics


def validate(model: nn.Module, cfg: TrainConfig, epoch_num: int) -> dict[str, float]:
    """Run and log validation using the shared evaluation helper."""
    if cfg.validation_dataloader is None:
        raise ValueError("validation_dataloader must be provided to validate.")
    metrics = evaluate(
        model,
        cfg.validation_dataloader,
        cfg.criterion_instance,
        device=cfg.device,
        target_scale=cfg.target_scale,
        target_transform=cfg.target_transform,
        name="Validation",
        epoch_num=epoch_num,
    )
    cfg.best_validation_loss = min(cfg.best_validation_loss, metrics["loss"])
    cfg.best_validation_mape = min(cfg.best_validation_mape, metrics["mape"])
    return metrics


def evaluate_test(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: TrainConfig | None = None,
) -> dict[str, float]:
    """Run the same evaluation and logging path on a labeled test dataset."""
    cfg = cfg or TrainConfig()
    return evaluate(
        model,
        dataloader,
        cfg.criterion_instance,
        device=cfg.device,
        target_scale=cfg.target_scale,
        target_transform=cfg.target_transform,
        name="Test",
    )


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    """Construct the optimizer selected by ``cfg.optimizer_name``."""
    optimizer_class = getattr(optim, cfg.optimizer_name, None)
    if optimizer_class is None or not isinstance(optimizer_class, type):
        raise ValueError(f"Unknown torch optimizer: {cfg.optimizer_name}")
    kwargs = {
        **cfg.optimizer_kwargs,
        "lr": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
    }
    return optimizer_class(model.parameters(), **kwargs)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: TrainConfig,
    *,
    start_epoch: int = 0,
    epoch_callback: Callable[[int], None] | None = None,
) -> list[dict[str, float]]:
    """Train a scalar price regressor and optionally validate each epoch.

    ``start_epoch`` only affects epoch numbering. ``cfg.num_epochs`` is the
    number of epochs to run in this invocation, which makes resumed runs easy
    to reason about.
    """
    model.to(cfg.device)
    if cfg.optimizer is None:
        cfg.optimizer = make_optimizer(model, cfg)

    history = []
    for epoch_num in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_samples = 0
        for inputs, targets in dataloader:
            if _keyboard_stop_requested():
                print("Q pressed; stopping training and saving the checkpoint...")
                raise KeyboardInterrupt
            inputs = inputs.to(cfg.device)
            targets = targets.to(cfg.device)
            cfg.optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)
            loss = cfg.criterion_instance(outputs, targets)
            loss.backward()
            cfg.optimizer.step()
            total_loss += loss.item() * len(inputs)
            num_samples += len(inputs)

        epoch_metrics = {"epoch": float(epoch_num), "train_loss": total_loss / num_samples}
        if epoch_num % cfg.train_log_freq == 0:
            print(f"Training result for epoch {epoch_num}: loss {epoch_metrics['train_loss']:.6f}")
        if cfg.validation_dataloader is not None and epoch_num % cfg.validate_freq == 0:
            validation_metrics = validate(model, cfg, epoch_num)
            epoch_metrics.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        if cfg.scheduler is not None:
            cfg.scheduler.step()
        history.append(epoch_metrics)
        if epoch_callback is not None:
            epoch_callback(epoch_num)
    return history


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: str = "cpu",
    target_scale: float = 1_000_000.0,
    target_transform: str = "identity",
) -> np.ndarray:
    """Predict dollar prices from an unlabeled dataloader."""
    was_training = model.training
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs.to(device)).squeeze(-1)
            predictions.append(outputs.cpu().numpy())
    if was_training:
        model.train()
    if not predictions:
        return np.array([], dtype=float)
    return decode_prices(
        np.concatenate(predictions),
        target_scale=target_scale,
        target_transform=target_transform,
    )


def save_checkpoint(path: str | Path, model: nn.Module, cfg: TrainConfig) -> None:
    """Backward-compatible shortcut for saving a reconstructable checkpoint."""
    Checkpointer.save(path, model, cfg)
