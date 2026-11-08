"""V15: recent, local OOF-error evidence alongside retained v14 forecasts."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import data
import features
from ensemble_runtime import CachedEnsemble
from adaptive_comparables import AdaptiveComparableEnsemble, trust_features
from specialist_comparables import forecast

HISTORY_MONTHS = 9
FIT_MONTHS = 6
HEAD_PARAMS = dict(iterations=700, depth=5, learning_rate=.03, l2_leaf_reg=20.,
                   loss_function="RMSE", random_seed=97, task_type="CPU", thread_count=4,
                   verbose=200, allow_writing_files=False)
HEAD_WEIGHT = .35


def local_error_features(history, homes):
    """Aggregate historical OOF residuals available strictly before listing.

    history has one sale per listing ID and an OOF error, log(actual/base).
    Query outcomes are never read. Each window also excludes the query ID.
    """
    history = history.reset_index(drop=True); homes = homes.reset_index(drop=True)
    types = history.house_type_name.fillna("UNKNOWN").astype(str)
    communities = history.community.fillna("UNKNOWN").astype(str)
    groups = {}
    for level, key in [("local", types + "|" + communities), ("type", types)]:
        groups[level] = pd.DataFrame({"key": key}).groupby("key", sort=False).indices
    sold = history.date_sold.to_numpy(dtype="datetime64[ns]")
    error = history.oof_error.to_numpy(float)
    ids = history.ml_num.to_numpy()
    xy = history[["latitude", "longitude"]].to_numpy(float)
    size = features.square_feet(history.house_area).to_numpy(float)
    qsize = features.square_feet(homes.house_area).to_numpy(float)
    output = {f"error_{level}_{days}_{stat}": np.full(len(homes), np.nan if stat != "count" else 0.)
              for level in ["local", "type"] for days in [90, 365] for stat in ["mean", "median", "count"]}
    output.update(neighbor_error=np.full(len(homes), np.nan), neighbor_count=np.zeros(len(homes)),
                  neighbor_spread=np.full(len(homes), np.nan))
    for i, home in homes.iterrows():
        if pd.isna(home.date_listed):
            continue
        cutoff = np.datetime64(home.date_listed, 'ns')
        if np.isnat(cutoff):
            continue
        kind = str(home.house_type_name) if pd.notna(home.house_type_name) else "UNKNOWN"
        community = str(home.community) if pd.notna(home.community) else "UNKNOWN"
        for level, key in [("local", kind + "|" + community), ("type", kind)]:
            candidates = np.asarray(groups[level].get(key, []), dtype=int)
            for days in [90, 365]:
                j = candidates[(sold[candidates] < cutoff)
                               & (sold[candidates] >= cutoff - np.timedelta64(days, 'D'))
                               & (ids[candidates] != home.ml_num) & np.isfinite(error[candidates])]
                output[f"error_{level}_{days}_count"][i] = len(j)
                if len(j) >= 5:
                    output[f"error_{level}_{days}_mean"][i] = error[j].mean()
                    output[f"error_{level}_{days}_median"][i] = np.median(error[j])
                if level != "local" or days != 365 or len(j) < 5:
                    continue
                qxy = np.array([home.latitude, home.longitude], dtype=float)
                j = j[np.isfinite(xy[j]).all(axis=1)]
                if not np.isfinite(qxy).all() or len(j) < 5:
                    continue
                distance = np.linalg.norm((xy[j] - qxy) * [111.2, 111.2 * .72], axis=1)
                age = (cutoff - sold[j]) / np.timedelta64(1, 'D')
                with np.errstate(divide='ignore', invalid='ignore'):
                    size_gap = np.abs(np.log(size[j] / qsize[i]))
                size_gap = np.where(np.isfinite(size_gap), size_gap, 1.)
                score = distance + 2 * size_gap + .5 * age / 365
                chosen = np.lexsort((ids[j].astype(str), score))[:20]
                w = np.exp(-distance[chosen] / 2 - age[chosen] / 180 - size_gap[chosen])
                if w.sum() <= 0:
                    continue
                values = error[j[chosen]]
                mean = np.average(values, weights=w)
                output['neighbor_error'][i] = mean
                output['neighbor_count'][i] = len(chosen)
                output['neighbor_spread'][i] = np.sqrt(np.average((values - mean)**2, weights=w))
    return pd.DataFrame(output)


def head_features(homes, base, stats, history):
    X, _, _ = trust_features(homes, base, stats, 'off_market')
    observable = features.off_market_features(data.hide_answers(homes, 'off_market')).reset_index(drop=True)
    # Observable house context can explain shared errors even when the two
    # price forecasts agree; the trust-only model cannot make that correction.
    for col in ['community', 'municipality', 'house_style', 'lot_front', 'lot_depth',
                'parking', 'remarks_renovated', 'remarks_needs_work']:
        X[col] = observable[col].to_numpy()
    return pd.concat([X, local_error_features(history, homes)], axis=1)


class LocalResidualEnsemble:
    def __init__(self, track='off_market'):
        if track != 'off_market':
            raise ValueError('The local-residual experiment is off-market only')
        self.track = track

    def fit(self, train):
        if train.ml_num.duplicated().any():
            raise ValueError('Local correction requires unique listing IDs')
        self.train = train.copy(deep=True)
        self.head = None
        self.local_fold_audit = []
        periods = pd.period_range(end=train.date_sold.max().to_period('M'), periods=HISTORY_MONTHS, freq='M')
        history_blocks = []; training_blocks = []; targets = []
        # Three warmup months supply past-error evidence for the first head fold.
        for month in periods:
            past = train[train.date_sold < month.start_time]
            held = train[train.date_sold.dt.to_period('M') == month]
            if past.empty or held.empty:
                continue
            print(f'Local correction: historical holdout {month}', flush=True)
            homes = data.hide_answers(held, 'off_market')
            base = CachedEnsemble('off_market').fit(past).predict(homes)
            residual = np.clip(np.log(held.sold_price.to_numpy(float) / base), -.5, .5)
            used = month in periods[-FIT_MONTHS:] and bool(history_blocks)
            if used:
                earlier_errors = pd.concat(history_blocks, ignore_index=True)
                stats = forecast('off_market', past, homes)
                training_blocks.append(head_features(homes, base, stats, earlier_errors))
                targets.append(residual)
            self.local_fold_audit.append(dict(month=str(month), train_max_sale=str(past.date_sold.max()),
                  train_n=len(past), holdout_n=len(held), used_for_head=used))
            # Append after building this month's head features: its labels cannot
            # enter any feature of its own month, even for later-listed homes.
            history_blocks.append(held.assign(oof_error=residual))
        self.error_history = (pd.concat(history_blocks, ignore_index=True) if history_blocks
                              else train.iloc[:0].assign(oof_error=pd.Series(dtype=float)))
        if training_blocks and sum(map(len, training_blocks)) >= 500:
            X = pd.concat(training_blocks, ignore_index=True)
            categorical = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
            for c in categorical:
                X[c] = X[c].astype('string').fillna('UNKNOWN').astype(str)
            self.categorical = categorical
            self.head = CatBoostRegressor(**HEAD_PARAMS)
            self.head.fit(X, np.concatenate(targets), cat_features=categorical)
            print(f'Local correction: fitted CPU head on {len(X):,} historical OOF homes', flush=True)
        self.v14 = AdaptiveComparableEnsemble('off_market').fit(train)
        return self

    def predict(self, homes):
        retained = self.v14.predict(homes)
        self.component_predictions = dict(self.v14.component_predictions)
        self.component_predictions['v14_prediction'] = retained.copy()
        base = self.component_predictions['uncorrected_ensemble']
        candidate = retained.copy(); active = np.zeros(len(homes), dtype=bool)
        if self.head is not None and len(homes):
            # v14 just computed these exact rows. Reuse them rather than hashing
            # the full training data and loading the same disk arrays again.
            stats = getattr(self.v14, 'specialist_stats', None)
            if stats is None:
                stats = forecast('off_market', self.train, homes)
            X = head_features(homes, base, stats, self.error_history)
            for c in self.categorical:
                X[c] = X[c].astype('string').fillna('UNKNOWN').astype(str)
            correction = np.clip(self.head.predict(X), -.25, .25)
            active = X.error_local_365_count.to_numpy() >= 5
            candidate = np.where(active, base * np.exp(correction), retained)
        self.component_predictions['local_corrected_base'] = candidate
        self.local_correction_weight = HEAD_WEIGHT * active
        self.comparable_count = self.v14.comparable_count
        self.comparable_spread = self.v14.comparable_spread
        self.comparable_weight = self.v14.comparable_weight
        self.blend_fold_audit = self.v14.blend_fold_audit
        print(f'Local correction: active for {active.sum():,}/{len(homes):,} homes', flush=True)
        return retained * np.exp(self.local_correction_weight * np.clip(np.log(candidate / retained), -.25, .25))
