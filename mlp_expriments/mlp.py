"""Simple PyTorch MLP price models for the two competition tracks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mlp_expriments import features as mlp_features
from mlp_expriments import utils


class TabularPreprocessor:
    """Fit a stable numeric tensor layout to a feature table or feature dict.

    Numeric columns are median-imputed and standardized. Selected categorical
    columns expand to one binary input per known category. Other categorical
    columns receive integer ids from 1; 0 is reserved for missing or unseen
    values.
    """

    def __init__(
        self,
        *,
        one_hot_columns: list[str] | None = None,
        embedding_columns: list[str] | None = None,
        scale_category_ids: bool = False,
    ):
        self.columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.one_hot_columns_: list[str] = list(one_hot_columns or [])
        self.embedding_columns_: list[str] = list(embedding_columns or [])
        self.scale_category_ids_ = scale_category_ids
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_means_: dict[str, float] = {}
        self.numeric_stds_: dict[str, float] = {}
        self.category_maps_: dict[str, dict[str, int]] = {}
        self.fitted_ = False

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, Mapping):
            return pd.DataFrame(dict(X))
        raise TypeError("X must be a pandas DataFrame or a mapping of feature columns.")

    @staticmethod
    def _category_values(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("<MISSING>")

    def fit(self, X) -> "TabularPreprocessor":
        frame = self._as_frame(X)
        self.columns_ = frame.columns.tolist()
        self.categorical_columns_ = [
            column
            for column in self.columns_
            if (
                pd.api.types.is_object_dtype(frame[column])
                or isinstance(frame[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(frame[column])
            )
        ]
        self.numeric_columns_ = [
            column for column in self.columns_ if column not in self.categorical_columns_
        ]

        for column in self.numeric_columns_:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.numeric_medians_[column] = median
            self.numeric_means_[column] = mean
            self.numeric_stds_[column] = std if np.isfinite(std) and std > 0 else 1.0

        for column in self.categorical_columns_:
            values = self._category_values(frame[column])
            categories = sorted(value for value in values.unique() if value != "<MISSING>")
            self.category_maps_[column] = {
                category: index for index, category in enumerate(categories, start=1)
            }

        self.fitted_ = True
        return self

    def transform(self, X) -> torch.Tensor:
        if not self.fitted_:
            raise RuntimeError("Fit the preprocessor before calling transform().")
        frame = self._as_frame(X)
        missing = [column for column in self.columns_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        columns = []
        for column in self.columns_:
            if column in self.category_maps_:
                mapping = self.category_maps_[column]
                values = self._category_values(frame[column]).map(mapping).fillna(0)
                if column in self.one_hot_columns_:
                    category_ids = values.to_numpy(dtype=np.int64)
                    columns.extend(
                        (category_ids == category_id).astype(np.float32)
                        for category_id in range(1, len(mapping) + 1)
                    )
                    continue
                array = values.to_numpy(dtype=np.float32)
                if self.scale_category_ids_:
                    array = array / max(len(mapping), 1)
            else:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(
                    self.numeric_medians_[column]
                )
                array = (
                    (values.to_numpy(dtype=np.float32) - self.numeric_means_[column])
                    / self.numeric_stds_[column]
                )
            columns.append(array)

        matrix = np.column_stack(columns).astype(np.float32, copy=False)
        return torch.from_numpy(matrix)

    def fit_transform(self, X) -> torch.Tensor:
        return self.fit(X).transform(X)

    @property
    def input_size(self) -> int:
        if not self.fitted_:
            raise RuntimeError("Fit the preprocessor before reading input_size.")
        id_columns = [
            column
            for column in self.categorical_columns_
            if column not in self.one_hot_columns_
        ]
        one_hot_size = sum(
            len(self.category_maps_[column]) for column in self.one_hot_columns_
        )
        return len(self.numeric_columns_) + len(id_columns) + one_hot_size

    def embedding_specs(self, embedding_dims: Mapping[str, int]) -> list[dict]:
        """Describe where categorical IDs occur in the transformed tensor."""
        if not self.fitted_:
            raise RuntimeError("Fit the preprocessor before reading embedding specs.")
        specs = []
        output_index = 0
        for column in self.columns_:
            if column in self.one_hot_columns_:
                output_index += len(self.category_maps_[column])
                continue
            if column in self.embedding_columns_:
                specs.append(
                    {
                        "column": column,
                        "input_index": output_index,
                        "num_embeddings": len(self.category_maps_[column]) + 1,
                        "embedding_dim": int(embedding_dims[column]),
                    }
                )
            output_index += 1
        return specs

    def state_dict(self) -> dict:
        """Return the fitted preprocessing values needed at prediction time."""
        return {
            "columns": self.columns_,
            "numeric_columns": self.numeric_columns_,
            "categorical_columns": self.categorical_columns_,
            "one_hot_columns": self.one_hot_columns_,
            "embedding_columns": self.embedding_columns_,
            "scale_category_ids": self.scale_category_ids_,
            "numeric_medians": self.numeric_medians_,
            "numeric_means": self.numeric_means_,
            "numeric_stds": self.numeric_stds_,
            "category_maps": self.category_maps_,
            "fitted": self.fitted_,
        }

    def load_state_dict(self, state: dict) -> None:
        self.columns_ = list(state["columns"])
        self.numeric_columns_ = list(state["numeric_columns"])
        self.categorical_columns_ = list(state["categorical_columns"])
        # Old checkpoints keep their original scaled-id encoding and width.
        self.one_hot_columns_ = list(state.get("one_hot_columns", []))
        self.embedding_columns_ = list(state.get("embedding_columns", []))
        self.scale_category_ids_ = bool(state.get("scale_category_ids", True))
        self.numeric_medians_ = dict(state["numeric_medians"])
        self.numeric_means_ = dict(state["numeric_means"])
        self.numeric_stds_ = dict(state["numeric_stds"])
        self.category_maps_ = {
            column: dict(mapping) for column, mapping in state["category_maps"].items()
        }
        self.fitted_ = bool(state["fitted"])


def build_model(input_size: int) -> nn.Sequential:
    """Build the requested input_size -> 50 -> 1 price regressor."""
    return nn.Sequential(
        OrderedDict(
            [
                ("input", nn.Linear(input_size, 128)),
                ("bn1", nn.BatchNorm1d(128)),
                ("activation1", nn.ReLU()),
                ("layer1", nn.Linear(128, 64)),
                ("bn2", nn.BatchNorm1d(64)),
                ("activation2", nn.ReLU()),
                ("layer2", nn.Linear(64, 32)),
                ("bn3", nn.BatchNorm1d(32)),
                ("activation3", nn.ReLU()),
                ("output", nn.Linear(32, 1)),
            ]
        )
    )


class EmbeddedPriceNetwork(nn.Sequential):
    """Dense price network with optional trainable categorical embeddings."""

    def __init__(self, input_size: int, embedding_specs: list[dict] | None = None):
        self.embedding_specs = list(embedding_specs or [])
        dense_input_size = input_size - len(self.embedding_specs) + sum(
            spec["embedding_dim"] for spec in self.embedding_specs
        )
        dense_layers = list(build_model(dense_input_size).children())
        super().__init__(*dense_layers)
        self._dense_module_names = [str(index) for index in range(len(dense_layers))]
        if self.embedding_specs:
            self.embeddings = nn.ModuleDict(
                {
                    spec["column"]: nn.Embedding(
                        spec["num_embeddings"],
                        spec["embedding_dim"],
                        padding_idx=0,
                    )
                    for spec in self.embedding_specs
                }
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.embedding_specs:
            embedded_indices = {spec["input_index"] for spec in self.embedding_specs}
            dense_indices = [
                index for index in range(inputs.shape[1]) if index not in embedded_indices
            ]
            parts = [inputs[:, dense_indices]]
            parts.extend(
                self.embeddings[spec["column"]](
                    inputs[:, spec["input_index"]].long()
                )
                for spec in self.embedding_specs
            )
            inputs = torch.cat(parts, dim=1)

        # Embedding modules are registered after the dense Sequential modules,
        # so explicitly run only the dense layers here.
        for name in self._dense_module_names:
            inputs = self._modules[name](inputs)
        return inputs


class OffMarketNetwork(EmbeddedPriceNetwork):
    pass


class ListAwareNetwork(EmbeddedPriceNetwork):
    pass


class MLPPriceModel:
    """Adapter exposing the project's standard ``fit``/``predict`` interface."""

    def __init__(
        self,
        make_features,
        network_class,
        *,
        config: utils.TrainConfig | None = None,
        validation_fraction: float = 0.1,
        batch_size: int = 1024,
        random_seed: int = 0,
        embedding_dims: Mapping[str, int] | None = None,
        checkpoint: str | Path | None = None,
    ):
        self.make_features = make_features
        self.network_class = network_class
        self.config = config or utils.TrainConfig()
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.embedding_dims = dict(
            embedding_dims or {"municipality": 4, "community": 16}
        )
        self.preprocessor = TabularPreprocessor(
            one_hot_columns=mlp_features.MLP_ONE_HOT_COLUMNS,
            embedding_columns=mlp_features.MLP_ID_COLUMNS,
        )
        self.model: nn.Module | None = None
        self.epochs_completed = 0
        self.last_checkpoint_path: Path | None = None
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def fit(
        self,
        train: pd.DataFrame,
        *,
        checkpoint_path: str | Path | Callable[[], str | Path] | None = None,
        resume: bool = False,
    ) -> "MLPPriceModel":
        """Train the model and checkpoint on completion or interruption.

        When ``resume`` is true, the model must have been initialized from a
        checkpoint. Its network, optimizer, and fitted preprocessor are reused.
        A Ctrl+C checkpoint is also written before the interrupt is re-raised.
        """
        if "sold_price" not in train:
            raise ValueError("Training data must include sold_price.")
        if resume and self.model is None:
            raise RuntimeError("Load a checkpoint before resuming training.")
        torch.manual_seed(self.random_seed)
        X = self.make_features(train)
        train_loader, validation_loader = utils.make_train_validation_loaders(
            X,
            train["sold_price"],
            self.preprocessor,
            validation_fraction=self.validation_fraction,
            seed=self.random_seed,
            batch_size=self.batch_size,
            target_scale=self.config.target_scale,
            target_transform=self.config.target_transform,
            fit_preprocessor=not resume,
        )
        if not resume:
            embedding_specs = self.preprocessor.embedding_specs(self.embedding_dims)
            self.model = self.network_class(
                self.preprocessor.input_size,
                embedding_specs=embedding_specs,
            )
            self.epochs_completed = 0
            # A fresh optimizer must point at this newly constructed network.
            self.config = replace(
                self.config,
                optimizer=None,
                validation_dataloader=validation_loader,
                best_validation_loss=float("inf"),
                best_validation_mape=float("inf"),
            )
        else:
            self.config.validation_dataloader = validation_loader

        def save_training_checkpoint() -> Path | None:
            if checkpoint_path is None:
                return None
            destination = checkpoint_path() if callable(checkpoint_path) else checkpoint_path
            destination = Path(destination)
            self.save_checkpoint(destination)
            self.last_checkpoint_path = destination
            return destination

        def checkpoint_epoch(epoch_num: int) -> None:
            self.epochs_completed = epoch_num

        try:
            utils.train(
                self.model,
                train_loader,
                self.config,
                start_epoch=self.epochs_completed,
                epoch_callback=checkpoint_epoch,
            )
            save_training_checkpoint()
        except KeyboardInterrupt:
            saved_path = save_training_checkpoint()
            if saved_path is not None:
                print(f"Training interrupted; saved {saved_path}")
            raise
        return self

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a fully usable predictor, including preprocessing and config."""
        if self.model is None:
            raise RuntimeError("Fit the model before saving a checkpoint.")
        utils.Checkpointer.save(
            path,
            self.model,
            self.config,
            model_kwargs={
                "input_size": self.preprocessor.input_size,
                "embedding_specs": self.preprocessor.embedding_specs(self.embedding_dims),
            },
            extra={
                "preprocessor": self.preprocessor.state_dict(),
                "validation_fraction": self.validation_fraction,
                "batch_size": self.batch_size,
                "random_seed": self.random_seed,
                "epochs_completed": self.epochs_completed,
                "embedding_dims": self.embedding_dims,
            },
        )

    def load_checkpoint(self, path: str | Path) -> "MLPPriceModel":
        """Load the network class, weights, preprocessing, and config at init time."""
        loaded = utils.Checkpointer.load(path)
        expected_class = self.network_class
        if not isinstance(loaded.model, expected_class):
            raise TypeError(
                f"Checkpoint contains {type(loaded.model).__name__}, "
                f"not {expected_class.__name__}."
            )
        self.model = loaded.model
        self.config = loaded.config
        self.preprocessor.load_state_dict(loaded.extra["preprocessor"])
        self.validation_fraction = float(
            loaded.extra.get("validation_fraction", self.validation_fraction)
        )
        self.batch_size = int(loaded.extra.get("batch_size", self.batch_size))
        self.random_seed = int(loaded.extra.get("random_seed", self.random_seed))
        self.epochs_completed = int(loaded.extra.get("epochs_completed", 0))
        self.embedding_dims = dict(
            loaded.extra.get("embedding_dims", self.embedding_dims)
        )
        return self

    def predict(self, homes: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Fit the model before calling predict().")
        loader = utils.make_dataloader(
            self.make_features(homes),
            None,
            self.preprocessor,
            batch_size=self.batch_size,
            target_transform=self.config.target_transform,
        )
        predictions = utils.predict(
            self.model,
            loader,
            device=self.config.device,
            target_scale=self.config.target_scale,
            target_transform=self.config.target_transform,
        )
        return np.maximum(predictions, 1.0)

    def evaluate(self, test: pd.DataFrame) -> dict[str, float]:
        """Evaluate a labeled validation/test frame through the shared helper."""
        if self.model is None:
            raise RuntimeError("Fit the model before calling evaluate().")
        if "sold_price" not in test:
            raise ValueError("Evaluation data must include sold_price.")
        loader = utils.make_dataloader(
            self.make_features(test),
            test["sold_price"],
            self.preprocessor,
            batch_size=self.batch_size,
            target_scale=self.config.target_scale,
            target_transform=self.config.target_transform,
        )
        return utils.evaluate_test(self.model, loader, self.config)


class OffMarketMLP(MLPPriceModel):
    def __init__(self, **kwargs):
        super().__init__(mlp_features.off_market_mlp_features, OffMarketNetwork, **kwargs)


class ListAwareMLP(MLPPriceModel):
    def __init__(self, **kwargs):
        super().__init__(mlp_features.list_aware_mlp_features, ListAwareNetwork, **kwargs)
