"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.linear_model import Ridge

import features
import data


SPATIAL_K = 20
CALIBRATION_FACTORS = {"off_market": 1.005, "list_aware": 0.994}
DEFAULT_PROFILE = "comparables"


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track not in {"off_market", "list_aware"}:
        raise ValueError(f"Unknown track: {track}")
    from ensemble_runtime import CachedEnsemble
    profile = os.environ.get("MODEL_PROFILE", DEFAULT_PROFILE)
    if profile == "comparables":
        from comparable_model import ComparableEnsemble
        return ComparableEnsemble(track)
    if profile in {"semantic", "text_corrected", "v11"}:
        return (TextCorrectedEnsemble(track, semantic=profile == "semantic")
                if track == "off_market" else CachedEnsemble(track))
    if profile == "ensemble":
        return CachedEnsemble(track)
    if profile != "baseline":
        raise ValueError("MODEL_PROFILE must be comparables, semantic, v11, text_corrected, ensemble, or baseline")
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


class TextCorrectedEnsemble:
    """Learn textual corrections from strictly earlier walk-forward errors.

    Six inner monthly folds supply residual labels. Each inner ensemble is
    trained before its held-out sale month; neither its learner nor its text
    vocabulary sees those held-out homes. The correction itself is fitted only
    on these historical residuals, never on the outer validation month.
    """

    def __init__(self, track, semantic=False):
        self.track = track
        self.semantic = semantic
        self.correction = None
        self.semantic_correction = None

    def _semantic_features(self, remarks, fit=False):
        from semantic_text import get_embeddings, release_encoder
        try:
            embeddings = get_embeddings(remarks)
        finally:
            # The encoder and CatBoost share one GPU; release it between stages.
            release_encoder()
        if fit:
            self.semantic_center = embeddings.mean(axis=0)
        return 3.0 * (embeddings - self.semantic_center)

    def fit(self, train):
        from ensemble_runtime import CachedEnsemble
        if self.semantic:
            from semantic_text import check_dependencies
            check_dependencies()
        if train["ml_num"].duplicated().any():
            raise ValueError("Text correction requires unique listing IDs")
        last_month = train["date_sold"].max().to_period("M")
        correction_rows, residuals = [], []
        self.inner_fold_audit = []
        for month in pd.period_range(end=last_month, periods=6, freq="M"):
            past = train[train["date_sold"] < month.start_time]
            held = train[train["date_sold"].dt.to_period("M") == month]
            if past.empty or held.empty:
                continue
            print(f"Text correction {self.track}: inner holdout {month}", flush=True)
            inner = CachedEnsemble(self.track).fit(past)
            prediction = inner.predict(data.hide_answers(held, self.track))
            cache_hit = inner.cache_hit
            del inner
            correction_rows.append(held)
            residuals.append(np.log(held["sold_price"].to_numpy(dtype=float) / prediction))
            self.inner_fold_audit.append({
                "month": str(month), "train_max_sale": str(past["date_sold"].max()),
                "train_n": len(past), "holdout_n": len(held), "cached": cache_hit,
            })

        self.correction = None
        self.semantic_correction = None
        if correction_rows and sum(map(len, correction_rows)) >= 100:
            held = pd.concat(correction_rows)
            y = np.clip(np.concatenate(residuals), -0.35, 0.35)
            # Focus on variation explained by text, not a global price shift.
            y = y - y.mean()
            self.correction = make_pipeline(
                FeatureUnion([
                    ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=5,
                        max_df=0.995, max_features=30000, sublinear_tf=True)),
                    ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                        min_df=5, max_features=30000, sublinear_tf=True)),
                ], transformer_weights={"words": 1.0, "characters": 0.5}),
                Ridge(alpha=20.0, fit_intercept=False, solver="lsqr"),
            )
            self.correction.fit(held["remarks"].fillna("").astype(str), y)
            if self.semantic:
                from scipy.sparse import csr_matrix, hstack
                remarks = held["remarks"].fillna("").astype(str)
                lexical = self.correction[0].transform(remarks)
                semantic = self._semantic_features(remarks, fit=True)
                # A separate challenger leaves the v11 forecast available for
                # an exact paired comparison on the same base predictions.
                self.semantic_correction = Ridge(alpha=20.0, fit_intercept=False, solver="lsqr")
                self.semantic_correction.fit(hstack([lexical, csr_matrix(semantic)], format="csr"), y)
            print(f"Text correction {self.track}: fitted on {len(held):,} historical OOF errors", flush=True)
        print(f"Text correction {self.track}: final ensemble uses all {len(train):,} homes", flush=True)
        self.base_model = CachedEnsemble(self.track).fit(train)
        return self

    def predict(self, homes):
        base = self.base_model.predict(homes)
        self.component_predictions = dict(self.base_model.component_predictions)
        self.component_predictions["uncorrected_ensemble"] = base.copy()
        adjustment = np.zeros(len(homes))
        if self.correction is not None and len(homes):
            # Fixed shrinkage and cap were chosen before outer validation.
            adjustment = np.clip(0.5 * self.correction.predict(
                homes["remarks"].fillna("").astype(str)), -0.10, 0.10)
        self.component_predictions["v11_prediction"] = base * np.exp(adjustment)
        if self.semantic_correction is not None and len(homes):
            from scipy.sparse import csr_matrix, hstack
            remarks = homes["remarks"].fillna("").astype(str)
            lexical = self.correction[0].transform(remarks)
            semantic = self._semantic_features(remarks)
            adjustment = np.clip(0.5 * self.semantic_correction.predict(
                hstack([lexical, csr_matrix(semantic)], format="csr")), -0.10, 0.10)
        self.text_log_adjustment = adjustment
        return base * np.exp(adjustment)


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


class FeatureModel:
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

    def _build_history(self, train):
        """Store prior-sale price-per-square-foot lookup tables.

        Each query uses only sales strictly before its listing date. The current
        sale and later sales therefore cannot contribute to its own feature.
        """
        sqft = features.square_feet(train["house_area"])
        valid = sqft.notna() & (sqft > 0) & train["sold_price"].notna() & (train["sold_price"] > 0)
        source = pd.DataFrame({
            "date": train.loc[valid, "date_sold"],
            "value": np.log(train.loc[valid, "sold_price"] / sqft.loc[valid]),
            "community": train.loc[valid, "community"].astype("string").fillna("UNKNOWN"),
            "property_type": train.loc[valid, "house_type_name"].astype("string").fillna("UNKNOWN"),
            "latitude": train.loc[valid, "latitude"],
            "longitude": train.loc[valid, "longitude"],
            "square_feet": sqft.loc[valid],
        })
        source["community_type"] = source["community"] + "|" + source["property_type"]
        self.history = {}
        for level in ["community_type", "community", "property_type"]:
            self.history[level] = self._make_lookup(source, level)
        self.history["global"] = self._make_lookup(source.assign(global_key="ALL"), "global_key")
        # A city-wide, trailing price-per-square-foot index.  It is calculated
        # using sales strictly before each sale date, so it is safe to use when
        # adjusting an older comparable to a home's listing date.
        global_table = self.history["global"]["ALL"]
        positions = np.searchsorted(global_table["dates"], source["date"].to_numpy(dtype="datetime64[ns]"), side="left")
        starts = np.searchsorted(
            global_table["dates"],
            source["date"].to_numpy(dtype="datetime64[ns]") - np.timedelta64(90, "D"),
            side="left",
        )
        market_counts = positions - starts
        market_level = np.full(len(source), np.nan)
        enough_history = market_counts >= 10
        market_level[enough_history] = (
            global_table["prefix_sum"][positions[enough_history]]
            - global_table["prefix_sum"][starts[enough_history]]
        ) / market_counts[enough_history]
        source["market_log_price_per_sqft"] = market_level
        self.spatial_history = self._make_spatial_lookup(source)

    @staticmethod
    def _make_lookup(source, key_column):
        lookup = {}
        for key, group in source.groupby(key_column, sort=False):
            group = group.sort_values("date")
            dates = group["date"].to_numpy(dtype="datetime64[ns]")
            values = group["value"].to_numpy(dtype=float)
            lookup[str(key)] = {
                "dates": dates,
                "prefix_sum": np.concatenate(([0.0], np.cumsum(values))),
                "prefix_count": np.arange(len(values) + 1, dtype=float),
            }
        return lookup

    @staticmethod
    def _make_spatial_lookup(source):
        lookup = {}
        spatial = source.dropna(subset=["latitude", "longitude"])
        for key, group in spatial.groupby("community_type", sort=False):
            group = group.sort_values("date")
            lookup[str(key)] = {
                "dates": group["date"].to_numpy(dtype="datetime64[ns]"),
                "values": group["value"].to_numpy(dtype=float),
                "latitude": group["latitude"].to_numpy(dtype=float),
                "longitude": group["longitude"].to_numpy(dtype=float),
                "square_feet": group["square_feet"].to_numpy(dtype=float),
                "market_log_price_per_sqft": group["market_log_price_per_sqft"].to_numpy(dtype=float),
            }
        return lookup

    def _query_history(self, level, keys, cutoffs, window_days=None, minimum_count=None):
        values = np.full(len(cutoffs), np.nan)
        counts = np.zeros(len(cutoffs))
        if minimum_count is None:
            minimum_count = self.min_history
        keys = pd.Series(keys, index=range(len(cutoffs))).astype("string").fillna("UNKNOWN")
        for key in keys.unique():
            rows = np.flatnonzero(keys.to_numpy() == key)
            table = self.history[level].get(str(key))
            if table is None:
                continue
            positions = np.searchsorted(table["dates"], cutoffs[rows], side="left")
            if window_days is None:
                starts = np.zeros(len(rows), dtype=int)
            else:
                starts = np.searchsorted(
                    table["dates"],
                    cutoffs[rows] - np.timedelta64(window_days, "D"),
                    side="left",
                )
            sums = table["prefix_sum"][positions] - table["prefix_sum"][starts]
            prior_counts = table["prefix_count"][positions] - table["prefix_count"][starts]
            has_history = prior_counts >= minimum_count
            values[rows[has_history]] = sums[has_history] / prior_counts[has_history]
            counts[rows[has_history]] = prior_counts[has_history]
        return values, counts

    def _query_spatial(self, keys, cutoffs, latitudes, longitudes, square_feet, market_levels):
        values = np.full(len(cutoffs), np.nan)
        time_adjusted_values = np.full(len(cutoffs), np.nan)
        size_matched_values = np.full(len(cutoffs), np.nan)
        counts = np.zeros(len(cutoffs))
        nearest_distances = np.full(len(cutoffs), np.nan)
        keys = pd.Series(keys, index=range(len(cutoffs))).astype("string").fillna("UNKNOWN")
        for row, key in enumerate(keys):
            table = self.spatial_history.get(str(key))
            if table is None or pd.isna(latitudes[row]) or pd.isna(longitudes[row]):
                continue
            recent = (
                (table["dates"] < cutoffs[row])
                & (table["dates"] >= cutoffs[row] - np.timedelta64(365, "D"))
            )
            if recent.sum() < 5:
                continue
            dx = (table["longitude"][recent] - longitudes[row]) * 0.72
            dy = table["latitude"][recent] - latitudes[row]
            distance = np.sqrt(dx * dx + dy * dy)
            k = min(self.spatial_k, len(distance))
            nearest = np.argpartition(distance, k - 1)[:k]
            nearest_distance = distance[nearest]
            weights = 1 / (nearest_distance + 1e-4)
            values[row] = np.average(table["values"][recent][nearest], weights=weights)
            # Translate each comparable's price-per-square-foot to the
            # city-wide price level at the listing date.  Rows without a
            # reliable historical index are excluded from this extra feature,
            # while the original spatial feature remains available as fallback.
            comparable_market = table["market_log_price_per_sqft"][recent][nearest]
            usable_market = (
                np.isfinite(comparable_market) & np.isfinite(market_levels[row])
                if market_levels is not None
                else np.zeros(len(comparable_market), dtype=bool)
            )
            if usable_market.any():
                adjusted_values = (
                    table["values"][recent][nearest][usable_market]
                    + market_levels[row]
                    - comparable_market[usable_market]
                )
                time_adjusted_values[row] = np.average(
                    adjusted_values,
                    weights=weights[usable_market],
                )
            if np.isfinite(square_feet[row]) and square_feet[row] > 0:
                size_penalty = 0.01 * np.abs(
                    np.log(table["square_feet"][recent] / square_feet[row])
                )
                matched_distance = distance + size_penalty
                matched_nearest = np.argpartition(matched_distance, k - 1)[:k]
                matched_weights = 1 / (matched_distance[matched_nearest] + 1e-4)
                size_matched_values[row] = np.average(
                    table["values"][recent][matched_nearest],
                    weights=matched_weights,
                )
            counts[row] = len(distance)
            nearest_distances[row] = nearest_distance.min()
        return values, time_adjusted_values, size_matched_values, counts, nearest_distances

    def _add_history_features(self, X, listing_dates, communities, property_types, latitudes, longitudes):
        homes = X.copy()
        cutoffs = pd.to_datetime(listing_dates).to_numpy(dtype="datetime64[ns]")
        community_keys = pd.Series(communities, index=homes.index).astype("string").fillna("UNKNOWN").reset_index(drop=True)
        type_keys = pd.Series(property_types, index=homes.index).astype("string").fillna("UNKNOWN").reset_index(drop=True)
        keys_by_level = {
            "community_type": community_keys + "|" + type_keys,
            "community": community_keys,
            "property_type": type_keys,
            "global": pd.Series("ALL", index=range(len(homes)), dtype="string"),
        }
        result = np.full(len(homes), np.nan)
        counts = np.zeros(len(homes))
        recent_result = np.full(len(homes), np.nan)
        recent_counts = np.zeros(len(homes))
        for level in ["community_type", "community", "property_type", "global"]:
            values, prior_counts = self._query_history(level, keys_by_level[level].to_numpy(), cutoffs)
            choose = np.isnan(result) & ~np.isnan(values)
            result[choose] = values[choose]
            counts[choose] = prior_counts[choose]
            recent_values, recent_prior_counts = self._query_history(
                level,
                keys_by_level[level].to_numpy(),
                cutoffs,
                window_days=365,
                minimum_count=3,
            )
            recent_choose = np.isnan(recent_result) & ~np.isnan(recent_values)
            recent_result[recent_choose] = recent_values[recent_choose]
            recent_counts[recent_choose] = recent_prior_counts[recent_choose]

            # Keep each hierarchy level separately as well. This lets the
            # gradient-boosted model learn how to blend specific and broad
            # comparable-sale evidence instead of relying only on the fallback.
            homes[f"prior_log_price_per_sqft_{level}"] = values
            homes[f"prior_sales_count_{level}"] = prior_counts
            homes[f"recent_log_price_per_sqft_{level}"] = recent_values
            homes[f"recent_sales_count_{level}"] = recent_prior_counts
        homes["prior_log_price_per_sqft"] = result
        homes["prior_sales_count"] = counts
        homes["recent_log_price_per_sqft"] = recent_result
        homes["recent_sales_count"] = recent_counts
        market_levels = None
        if self.target != "list_residual":
            market_levels, _ = self._query_history(
                "global",
                keys_by_level["global"].to_numpy(),
                cutoffs,
                window_days=90,
                minimum_count=10,
            )
            homes["global_recent_90d_log_price_per_sqft"] = market_levels
        spatial_values, time_adjusted_values, size_matched_values, spatial_counts, nearest_distances = self._query_spatial(
            community_keys + "|" + type_keys,
            cutoffs,
            pd.Series(latitudes, index=homes.index).to_numpy(dtype=float),
            pd.Series(longitudes, index=homes.index).to_numpy(dtype=float),
            homes["square_feet"].to_numpy(dtype=float),
            market_levels,
        )
        homes["spatial_recent_log_price_per_sqft"] = spatial_values
        # Size matching helped the off-market task, but list-aware homes already
        # expose a strong size-and-price signal through their list price.
        if self.target != "list_residual":
            homes["spatial_time_adjusted_log_price_per_sqft"] = time_adjusted_values
            homes["spatial_size_matched_log_price_per_sqft"] = size_matched_values
        else:
            # Asking price relative to historical value exposes pricing strategy
            # directly. Missing historical evidence remains missing.
            size = pd.to_numeric(homes["square_feet"], errors="coerce")
            log_size = np.log(size.where(size > 0))
            asking = pd.to_numeric(homes["list_price"], errors="coerce")
            log_asking = np.log(asking.where(asking > 0))
            for column in ["spatial_recent_log_price_per_sqft", "recent_log_price_per_sqft", "prior_log_price_per_sqft"]:
                homes[f"log_asking_relative_to_{column}"] = log_asking - log_size - homes[column]
        homes["spatial_recent_sales_count"] = spatial_counts
        homes["nearest_comparable_distance"] = nearest_distances
        return homes

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
            task_type="GPU",
            devices="0",
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
