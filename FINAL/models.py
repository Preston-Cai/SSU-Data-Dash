"""V15 model routing, segmented learners, and the base price ensemble."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import features
from history_features import HistoricalFeaturesMixin
from text_correction import TextCorrectedEnsemble
from runtime_config import catboost_runtime
import data


SPATIAL_K = 20
CALIBRATION_FACTORS = {"off_market": 1.005, "list_aware": 0.994}
DEFAULT_PROFILE = "local_residual"


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"Unknown track: {track}")
    from ensemble_runtime import CachedEnsemble
    profile = DEFAULT_PROFILE
    if profile in {"local_residual", "adaptive_comparables"} and track == "list_aware":
        from adaptive_comparables import FixedSpecialistEnsemble
        return FixedSpecialistEnsemble(track)
    if profile == "local_residual":
        from local_residual_model import LocalResidualEnsemble
        return LocalResidualEnsemble(track)
    if profile == "adaptive_comparables":
        from adaptive_comparables import AdaptiveComparableEnsemble
        return AdaptiveComparableEnsemble(track)
    if profile == "comparables":
        from comparable_model import ComparableEnsemble
        return ComparableEnsemble(track)
    if profile in {"semantic", "text_corrected", "v11"}:
        return (TextCorrectedEnsemble(track, semantic=profile == "semantic")
                if track == "off_market" else CachedEnsemble(track))
    if profile == "ensemble":
        return CachedEnsemble(track)
    if profile != "baseline":
        raise ValueError("MODEL_PROFILE must be local_residual, adaptive_comparables, comparables, semantic, v11, text_corrected, ensemble, or baseline")
    return make_baseline_model(track)


def make_baseline_model(track):
    """The retained v9 off-market model and improved list-aware model."""
    if track == "list_aware":
        # return ListPriceBaseline()
        return CatBoostFeatureModel(
            features.list_aware_features,
            CALIBRATION_FACTORS[track],
            target="list_residual",
            iterations=1000,
            text_components=64,
        )
    return SegmentedFeatureModel(track)


class SegmentedFeatureModel:
    """Fit separate learners for ground-oriented and condo homes in either track."""

    GROUPS = {
        "ground_oriented": ["Detached", "Semi-Detached", "Freehold Townhouse"],
        "condo_style": ["Condo Apt", "Condo Townhouse", "Co-Op Apt"],
    }

    def __init__(self, track="off_market", **model_params):
        self.track = track
        self.model_params = model_params

    def fit(self, train):
        self.models = {}
        for name, property_types in self.GROUPS.items():
            subset = train[train["house_type_name"].isin(property_types)]
            if subset.empty:
                raise ValueError(f"No training homes for segment {name}")
            params = dict(iterations=2400 if self.track == "off_market" else 1000,
                          text_components=64)
            params.update(self.model_params)
            print(f"Training {self.track} / {name}: {len(subset):,} homes", flush=True)
            self.models[name] = CatBoostFeatureModel(
                features.off_market_features if self.track == "off_market" else features.list_aware_features,
                CALIBRATION_FACTORS[self.track],
                target="price" if self.track == "off_market" else "list_residual",
                **params,
            ).fit(subset, history_train=train)
        return self

    def predict(self, homes):
        predictions = np.full(len(homes), np.nan)
        types = homes["house_type_name"].to_numpy()
        for name, property_types in self.GROUPS.items():
            rows = np.flatnonzero(np.isin(types, property_types))
            if len(rows):
                predictions[rows] = self.models[name].predict(homes.iloc[rows])
        if np.isnan(predictions).any():
            raise ValueError("No segment model for one or more property types")
        return predictions


class DiverseEnsemble:
    """Blend independently trained models in log-price space.

    Weights are fixed before validation, rather than fitted on the reported
    validation labels. All learners use the complete supplied training fold.
    """

    def __init__(self, track):
        self.track = track
        self.member_specs = [
            ("baseline", 0.4, {}),
            ("shallow", 0.2, dict(depth=6, iterations=4800, learning_rate=0.035,
                                  l2_leaf_reg=8.0, random_seed=17, text_components=128)),
            ("deep", 0.2, dict(depth=10, iterations=2800, learning_rate=0.035,
                               l2_leaf_reg=12.0, random_seed=29, text_components=128)),
            ("alternate_groups", 0.2, dict(depth=8, iterations=3200, learning_rate=0.04,
                                           l2_leaf_reg=8.0, random_seed=43, text_components=128)),
        ]

    def _make_member(self, name, params):
        if name == "baseline":
            return make_baseline_model(self.track)
        segmented = (self.track == "off_market") != (name == "alternate_groups")
        if segmented:
            return SegmentedFeatureModel(self.track, **params)
        return CatBoostFeatureModel(
            features.off_market_features if self.track == "off_market" else features.list_aware_features,
            CALIBRATION_FACTORS[self.track],
            target="price" if self.track == "off_market" else "list_residual", **params,
        )

    def fit(self, train):
        self.models = []
        for name, weight, params in self.member_specs:
            print(f"Ensemble {self.track}: {name}, weight={weight:.0%}", flush=True)
            model = self._make_member(name, params)
            model.fit(train)
            self.models.append((name, weight, model))
        return self

    def predict(self, homes):
        self.component_predictions = {}
        log_prediction = np.zeros(len(homes))
        for name, weight, model in self.models:
            prediction = np.asarray(model.predict(homes), dtype=float)
            if not np.all(np.isfinite(prediction) & (prediction > 0)):
                raise ValueError(f"Invalid prices from ensemble member {name}")
            self.component_predictions[name] = prediction
            log_prediction += weight * np.log(prediction)
        return np.exp(log_prediction)


class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self   # nothing to learn

    def predict(self, homes):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: the median sold price of the k nearest earlier sales of
    the same property type.

    "Nearest" means closest by latitude and longitude. "Earlier" means sold before the
    home was listed, so the model never looks into the future, even when the
    training table holds later sales. It makes no adjustment for size, condition or
    for prices moving since those sales: improving on that is your job.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        # Newest sales first, so when several sales are equally near (a condo
        # building shares one location), the most recent ones are picked.
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:   # no earlier sale of this type: use every type
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]

        # Distance in degrees. In Toronto a degree of longitude is only 0.72 times as
        # long as a degree of latitude (0.72 = cosine of 43.7 degrees north).
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        nearest = np.argsort(distance, kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel(HistoricalFeaturesMixin):
    """A machine-learning model (gradient-boosted trees) on the features in features.py.

    It learns log(price) rather than price, so a 10% miss on a cheap condo counts
    as much as a 10% miss on an expensive house. Not used until you switch it on
    in make_model().
    """

    def __init__(self, make_features, calibration_factor=1.0, target="price"):
        self.make_features = make_features
        self.calibration_factor = calibration_factor
        self.target = target
        self.model = HistGradientBoostingRegressor(random_state=0)
        self.min_history = 5
        self.spatial_k = SPATIAL_K

    def fit(self, train, history_train=None):
        if history_train is None:
            history_train = train
        X = self.make_features(train)
        self._build_history(history_train)
        X = self._add_history_features(
            X, train["date_listed"], train["community"], train["house_type_name"],
            train["latitude"], train["longitude"]
        )
        self.numeric_columns = [column for column in X.columns if is_numeric_dtype(X[column])]
        self.categorical_columns = [column for column in X.columns if column not in self.numeric_columns]
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.categorical_columns:
            categorical = X[self.categorical_columns].astype("string").fillna("UNKNOWN")
            self.encoder.fit(categorical)
        if self.target == "list_residual":
            y = np.log(train["sold_price"] / train["list_price"])
        else:
            y = np.log(train["sold_price"])
        self.model.fit(self._transform(X), y)
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        X = self._add_history_features(
            X, homes["date_listed"], homes["community"], homes["house_type_name"],
            homes["latitude"], homes["longitude"]
        )
        prediction = np.exp(self.model.predict(self._transform(X)))
        if self.target == "list_residual":
            prediction = homes["list_price"].to_numpy(dtype=float) * prediction
        return self.calibration_factor * prediction

    def _transform(self, X):
        numeric = X.reindex(columns=self.numeric_columns).to_numpy(dtype=float, na_value=np.nan)
        if not self.categorical_columns:
            return numeric
        categorical = X.reindex(columns=self.categorical_columns).astype("string").fillna("UNKNOWN")
        encoded = self.encoder.transform(categorical)
        return np.hstack([numeric, encoded])


class CatBoostFeatureModel(FeatureModel):
    """CatBoost version of FeatureModel using the same leakage-safe features.

    Unlike the histogram model, CatBoost receives community, house style, and
    other categories directly.  This lets it learn category effects and their
    interactions with size, location, and comparable-sale evidence without a
    manually one-hot-encoded representation.
    """

    def __init__(
        self,
        make_features,
        calibration_factor=1.0,
        target="price",
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        text_components=0,
        random_seed=0,
    ):
        super().__init__(make_features, calibration_factor, target)
        self.text_components = text_components
        self.model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            **catboost_runtime(),
            verbose=100,
            allow_writing_files=False,
        )

    @staticmethod
    def _catboost_table(X):
        """Return CatBoost-ready values and the native categorical columns."""
        table = X.copy()
        categorical_columns = [
            column for column in table.columns if not is_numeric_dtype(table[column])
        ]
        for column in categorical_columns:
            table[column] = table[column].astype("string").fillna("UNKNOWN").astype(str)
        return table, categorical_columns

    def _build_history(self, train):
        """Reuse immutable lookups for members with identical historical sales."""
        import feature_cache

        # Full content includes dates and outcomes, so a changed training fold
        # cannot borrow history from a different cutoff or dataset.
        self._history_cache_key = (
            type(self),
            feature_cache.callable_key(FeatureModel._build_history),
            feature_cache.callable_key(features.square_feet),
            feature_cache.callable_key(self._make_lookup),
            feature_cache.callable_key(self._make_spatial_lookup),
            feature_cache.frame_key(train),
        )
        cached = feature_cache.get("history", self._history_cache_key)
        if cached is None:
            super()._build_history(train)
            feature_cache.put("history", self._history_cache_key,
                              (self.history, self.spatial_history))
        else:
            self.history, self.spatial_history = cached

    def _history_features(self, homes):
        """Reuse the exact pre-text feature table across ensemble members."""
        import feature_cache

        X = self.make_features(homes)
        key = (
            self._history_cache_key, self.target, self.min_history, self.spatial_k,
            feature_cache.callable_key(self.make_features),
            feature_cache.callable_key(self._add_history_features),
            feature_cache.callable_key(self._query_history),
            feature_cache.callable_key(self._query_spatial),
            feature_cache.frame_key(homes), feature_cache.frame_key(X),
        )
        cached = feature_cache.get("history_features", key)
        if cached is not None:
            return cached.copy(deep=True)
        X = self._add_history_features(
            X, homes["date_listed"], homes["community"], homes["house_type_name"],
            homes["latitude"], homes["longitude"],
        )
        feature_cache.put("history_features", key, X.copy(deep=True))
        return X

    def _add_text_features(self, X, homes, fit=False):
        """Learn vocabulary and text components using this fold's training homes only."""
        if not self.text_components:
            return X
        remarks = homes["remarks"].fillna("").astype(str)
        if fit:
            import feature_cache

            # An independently fitted 64-component SVD is not the first 64
            # columns of a 128-component fit. Keep those cache entries separate.
            text_key = (
                feature_cache.callable_key(self._add_text_features),
                self.text_components, feature_cache.frame_key(remarks.to_frame()),
            )
            cached = feature_cache.get("text_fit", text_key)
            if cached is not None:
                self.text_vectorizer, self.text_svd, dense_text = cached
                text_table = pd.DataFrame(
                    dense_text, index=X.index,
                    columns=[f"remarks_component_{i:02d}" for i in range(dense_text.shape[1])],
                )
                return pd.concat([X, text_table], axis=1)
            # Keep negations and word pairs: 'needs updating' carries a different
            # signal from 'updated'. Exclude numeric tokens, including prices.
            self.text_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=10, max_df=0.98,
                max_features=16000, sublinear_tf=True,
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", dtype=np.float32,
            )
            sparse_text = self.text_vectorizer.fit_transform(remarks)
            components = min(self.text_components, sparse_text.shape[1] - 1,
                             sparse_text.shape[0] - 1)
            if components < 1:
                raise ValueError("Not enough training text for SVD text features")
            self.text_svd = TruncatedSVD(n_components=components, random_state=0)
            dense_text = self.text_svd.fit_transform(sparse_text)
            feature_cache.put("text_fit", text_key,
                              (self.text_vectorizer, self.text_svd, dense_text))
        else:
            dense_text = self.text_svd.transform(self.text_vectorizer.transform(remarks))
        text_table = pd.DataFrame(
            dense_text, index=X.index,
            columns=[f"remarks_component_{i:02d}" for i in range(dense_text.shape[1])],
        )
        return pd.concat([X, text_table], axis=1)

    def fit(self, train, history_train=None):
        if history_train is None:
            history_train = train
        self._build_history(history_train)
        X = self._history_features(train)
        X = self._add_text_features(X, train, fit=True)
        X, self.categorical_columns = self._catboost_table(X)
        if self.target == "list_residual":
            y = np.log(train["sold_price"] / train["list_price"])
        else:
            y = np.log(train["sold_price"])
        self.model.fit(X, y, cat_features=self.categorical_columns)
        return self

    def predict(self, homes):
        X = self._history_features(homes)
        X = self._add_text_features(X, homes)
        X, _ = self._catboost_table(X)
        prediction = np.exp(self.model.predict(X))
        if self.target == "list_residual":
            prediction = homes["list_price"].to_numpy(dtype=float) * prediction
        return self.calibration_factor * prediction
