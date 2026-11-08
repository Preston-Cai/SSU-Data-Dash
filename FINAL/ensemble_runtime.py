"""Reuse exact ensemble member predictions across folds and process restarts.

Fitting is deferred until predict(), when both the training set and the query
are known. A different query still needs a fitted model unless its predictions
are cached. No serialized Python objects or trained model weights are loaded.
"""
import ast
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
from runtime_config import catboost_runtime

from experiment_cache import ArrayCache, dataframe_digest, stable_key

ROOT = Path(__file__).resolve().parent


def base_signature():
    """Fingerprint learners, feature code and libraries, excluding corrections.

    Changing the semantic/text correction does not invalidate the expensive
    price learners. Changing their implementations or features does.
    """
    source = ast.parse((ROOT / "models.py").read_text(encoding="utf-8"))
    names = {"make_baseline_model", "SegmentedFeatureModel", "FeatureModel",
             "CatBoostFeatureModel", "SPATIAL_K", "CALIBRATION_FACTORS"}
    selected = []
    for node in source.body:
        assigned = {x.id for x in getattr(node, "targets", []) if isinstance(x, ast.Name)}
        if getattr(node, "name", None) in names or assigned & names:
            selected.append(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.ClassDef) and node.name == "DiverseEnsemble":
            selected.extend(ast.dump(method, include_attributes=False) for method in node.body
                            if getattr(method, "name", None) == "_make_member")
    digest = hashlib.sha256("\n".join(selected).encode())
    for name in ["features.py", "history_features.py", "runtime_config.py", "feature_cache.py", "ensemble_runtime.py", "experiment_cache.py"]:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    for package in ["numpy", "pandas", "scikit-learn", "catboost"]:
        digest.update(f"{package}={importlib.metadata.version(package)}".encode())
    digest.update(repr(sorted(catboost_runtime().items())).encode())
    return digest.hexdigest()


class CachedEnsemble:
    """Same v10 price ensemble, with a persistent cache for each member."""

    def __init__(self, track):
        from models import DiverseEnsemble
        self.track = track
        self.ensemble = DiverseEnsemble(track)
        self.cache = ArrayCache()

    def fit(self, train):
        self.train = train.copy(deep=True)
        self.train_digest = dataframe_digest(self.train)
        self.signature = base_signature()
        return self

    def predict(self, homes):
        query_digest = dataframe_digest(homes)
        self.component_predictions = {}
        self.cache_hit = True
        log_prediction = np.zeros(len(homes), dtype=float)
        for name, weight, params in self.ensemble.member_specs:
            config = {"schema": 1, "track": self.track, "member": name, "params": params,
                      "source": self.signature, "train": self.train_digest, "query": query_digest}
            key = stable_key("ensemble_member", config, [])
            saved = self.cache.load("ensemble_member", key)
            prediction = None if saved is None else saved.get("prediction")
            valid = (prediction is not None and prediction.shape == (len(homes),)
                     and np.all(np.isfinite(prediction) & (prediction > 0)))
            print(f"Ensemble {self.track}: {name}, weight={weight:.0%}, "
                  f"{'disk cache hit' if valid else 'training'}", flush=True)
            if not valid:
                self.cache_hit = False
                model = self.ensemble._make_member(name, params)
                model.fit(self.train)
                prediction = np.asarray(model.predict(homes), dtype=float)
                if prediction.shape != (len(homes),) or not np.all(np.isfinite(prediction) & (prediction > 0)):
                    raise ValueError(f"Invalid prices from ensemble member {name}")
                self.cache.save("ensemble_member", key, {"prediction": prediction}, metadata={
                    "track": self.track, "member": name, "train_n": len(self.train),
                    "train_max_sale": str(self.train["date_sold"].max()), "query_n": len(homes),
                    "source": self.signature,
                })
                del model
            self.component_predictions[name] = prediction
            log_prediction += weight * np.log(prediction)
        result = np.exp(log_prediction)
        self.component_predictions["uncorrected_ensemble"] = result.copy()
        return result
