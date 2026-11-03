"""EDIT ME: the models. Every model has the same two steps:

    model.fit(train)        train = past sales, including sold_price and date_sold
    model.predict(homes)    homes = homes to value, with only the columns of a test file;
                            returns one predicted sold price per home

make_model() picks the model for each track. The two starting models are baselines:
deliberately simple, so you can measure how much your own ideas add.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

import features


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        # return ListPriceBaseline()
        return FeatureModel(features.list_aware_features)   # try this instead
    # return NearestSalesBaseline(k=10)
    return FeatureModel(features.off_market_features)       # try this instead


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

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)
        self.min_history = 5

    def fit(self, train):
        X = self.make_features(train)
        self._build_history(train)
        X = self._add_history_features(
            X, train["date_listed"], train["community"], train["house_type_name"]
        )
        self.numeric_columns = [column for column in X.columns if is_numeric_dtype(X[column])]
        self.categorical_columns = [column for column in X.columns if column not in self.numeric_columns]
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.categorical_columns:
            categorical = X[self.categorical_columns].astype("string").fillna("UNKNOWN")
            self.encoder.fit(categorical)
        self.model.fit(self._transform(X), np.log(train["sold_price"]))
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        X = self._add_history_features(
            X, homes["date_listed"], homes["community"], homes["house_type_name"]
        )
        return np.exp(self.model.predict(self._transform(X)))

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
        })
        source["community_type"] = source["community"] + "|" + source["property_type"]
        self.history = {}
        for level in ["community_type", "community", "property_type"]:
            self.history[level] = self._make_lookup(source, level)
        self.history["global"] = self._make_lookup(source.assign(global_key="ALL"), "global_key")

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

    def _add_history_features(self, X, listing_dates, communities, property_types):
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
        return homes

    def _transform(self, X):
        numeric = X.reindex(columns=self.numeric_columns).to_numpy(dtype=float, na_value=np.nan)
        if not self.categorical_columns:
            return numeric
        categorical = X.reindex(columns=self.categorical_columns).astype("string").fillna("UNKNOWN")
        encoded = self.encoder.transform(categorical)
        return np.hstack([numeric, encoded])
