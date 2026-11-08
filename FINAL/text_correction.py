"""Retained v11/v12 experiments; not used by the default v15 profile."""

import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import data


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


