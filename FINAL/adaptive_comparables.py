"""V14 trust weights learned only from earlier walk-forward predictions."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import features
from ensemble_runtime import CachedEnsemble
from comparable_model import ComparableEnsemble, BLEND
from specialist_comparables import forecast

TRUST_PARAMS = dict(max_iter=100, max_leaf_nodes=7, min_samples_leaf=250,
                    learning_rate=.05, l2_regularization=10., early_stopping=False,
                    random_state=83)
PRIOR_STRENGTH = .05 ** 2


def usable(stats):
    return (np.isfinite(stats["price"]) & (stats["price"] > 0) & (stats["count"] >= 3)
            & np.isfinite(stats["spread"]) & (stats["spread"] <= .30)
            & np.isfinite(stats["nearest"]) & (stats["nearest"] <= 5.))


def trust_features(homes, base, stats, track):
    """Explicit numeric whitelist; query outcomes can never become predictors."""
    use = usable(stats)
    delta = np.clip(np.log(np.where(use, stats["price"], base) / base), -.40, .40)
    X = pd.DataFrame({"delta": delta, "absolute_delta": abs(delta),
                      "log_base_price": np.log(base), "count": stats["count"],
                      "spread": stats["spread"], "nearest": stats["nearest"], "age": stats["age"],
                      "size": features.square_feet(homes.house_area).to_numpy(float),
                      "bedrooms": homes.bedrooms.to_numpy(float),
                      "washrooms": homes.washrooms.to_numpy(float)})
    for kind in features.PROPERTY_TYPES:
        X[f"type_{kind}"] = homes.house_type_name.eq(kind).to_numpy(dtype=int)
    if track == "list_aware":
        asking = homes.list_price.to_numpy(float)
        if not np.all(np.isfinite(asking) & (asking > 0)):
            raise ValueError("Trust model requires positive asking prices")
        X["base_relative_to_asking"] = np.log(base / asking)
    return X, delta, use


class AdaptiveComparableEnsemble:
    def __init__(self, track):
        self.track = track

    def fit(self, train):
        if train.ml_num.duplicated().any():
            raise ValueError("Adaptive blend requires unique listing IDs")
        self.train = train.copy(deep=True)
        self.trust_model = None
        self.blend_fold_audit = []
        blocks, targets, weights = [], [], []
        months = pd.period_range(end=train.date_sold.max().to_period("M"), periods=6, freq="M")
        for month in months:
            past = train[train.date_sold < month.start_time]
            held = train[train.date_sold.dt.to_period("M") == month]
            if past.empty or held.empty:
                continue
            print(f"Adaptive blend {self.track}: historical holdout {month}", flush=True)
            homes = data.hide_answers(held, self.track)
            base = CachedEnsemble(self.track).fit(past).predict(homes)
            stats = forecast(self.track, past, homes)
            X, delta, use = trust_features(homes, base, stats, self.track)
            residual = np.clip(np.log(held.sold_price.to_numpy(float) / base), -.5, .5)
            # Minimize (residual - w*delta)^2 + lambda*(w-prior)^2.
            # This avoids unstable residual/delta labels near zero disagreement.
            importance = delta ** 2 + PRIOR_STRENGTH
            target = (delta * residual + PRIOR_STRENGTH * BLEND[self.track]) / importance
            blocks.append(X.loc[use])
            targets.append(target[use]); weights.append(importance[use])
            self.blend_fold_audit.append(dict(month=str(month), train_max_sale=str(past.date_sold.max()),
                train_n=len(past), holdout_n=len(held), usable_n=int(use.sum())))
        if blocks and sum(map(len, blocks)) >= 500:
            X = pd.concat(blocks, ignore_index=True)
            y = np.concatenate(targets); weight = np.concatenate(weights)
            self.trust_model = HistGradientBoostingRegressor(**TRUST_PARAMS)
            self.trust_model.fit(X, y, sample_weight=weight / weight.mean())
            print(f"Adaptive blend {self.track}: learned from {len(X):,} historical OOF homes", flush=True)
        else:
            print(f"Adaptive blend {self.track}: insufficient OOF evidence; fixed-weight fallback", flush=True)
        self.v13 = ComparableEnsemble(self.track).fit(self.train)
        return self

    def predict(self, homes):
        v13 = self.v13.predict(homes)
        base = self.v13.component_predictions["uncorrected_ensemble"]
        stats = forecast(self.track, self.train, homes)
        self.specialist_stats = stats
        X, delta, use = trust_features(homes, base, stats, self.track)
        weight = np.full(len(homes), BLEND[self.track])
        if self.trust_model is not None and len(homes):
            weight = np.clip(self.trust_model.predict(X), 0., .75)
        weight = np.where(use, weight, 0.)
        prediction = np.where(use, base * np.exp(weight * delta), v13)
        fixed = np.where(use, base * np.exp(BLEND[self.track] * delta), v13)
        self.component_predictions = dict(self.v13.component_predictions)
        self.component_predictions["v13_prediction"] = v13
        self.component_predictions["specialist_comparables"] = np.where(use, stats["price"], v13)
        self.component_predictions["specialist_fixed_blend"] = fixed
        self.comparable_count = stats["count"]
        self.comparable_spread = stats["spread"]
        self.comparable_weight = weight
        print(f"Adaptive blend {self.track}: {use.sum():,}/{len(homes):,} eligible, "
              f"mean specialist weight {weight[use].mean() if use.any() else 0.:.1%}", flush=True)
        return prediction


class FixedSpecialistEnsemble(AdaptiveComparableEnsemble):
    """The reported v14 fixed blend, without fitting an adaptive trust model."""

    def fit(self, train):
        self.train = train.copy(deep=True)
        self.trust_model = None
        self.blend_fold_audit = []
        self.v13 = ComparableEnsemble(self.track).fit(self.train)
        return self
