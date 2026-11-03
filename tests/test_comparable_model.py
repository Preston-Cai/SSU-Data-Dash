import numpy as np
import pandas as pd
import pytest

import comparable_model as cm
import data
from test_starter import make_listings


def history(n=160):
    frame = make_listings(n).reset_index(drop=True)
    frame['date_sold'] = pd.date_range('2024-01-01', periods=n, freq='3D')
    frame['date_listed'] = frame.date_sold - pd.Timedelta(days=10)
    frame['latitude'] = 43.7
    frame['longitude'] = -79.4
    frame['remarks'] = ['original kitchen needs updating' if i % 2 else
                        'renovated kitchen large garden' for i in range(n)]
    return frame


def test_retrieval_excludes_self_same_day_and_future():
    ref = history(30).assign(house_type_name='Detached')
    query = ref.iloc[[10]].copy()
    query['date_listed'] = ref.loc[10, 'date_sold']
    pairs = cm.retrieve_pairs(ref, query)
    comps = ref.iloc[pairs['c']]
    assert len(comps) == cm.K
    assert (comps.date_sold < query.date_listed.iloc[0]).all()
    assert not comps.ml_num.eq(query.ml_num.iloc[0]).any()
    # Even if its date is earlier, the subject cannot be its own comparable.
    ref.loc[10, 'date_sold'] = ref.loc[0, 'date_sold']
    assert 10 not in cm.retrieve_pairs(ref, query)['c']
    changed = query.assign(sold_price=1e12, days_on_market=99999,
                           external_avm_estimate=1e12, date_sold=pd.Timestamp('2099-01-01'))
    np.testing.assert_array_equal(cm.retrieve_pairs(ref, changed)['c'],
                                  cm.retrieve_pairs(ref, query)['c'])


@pytest.mark.parametrize('track', ['off_market', 'list_aware'])
def test_native_catboost_smoke_and_no_query_outcome_features(track, monkeypatch):
    monkeypatch.setattr(cm, 'PARAMS', dict(iterations=3, depth=3, loss_function='RMSE',
                        random_seed=0, task_type='CPU', verbose=False, allow_writing_files=False,
                        thread_count=2))
    train = history()
    learner = cm.AdjustedComparables(track).fit(train)
    homes = train.iloc[:8].copy().assign(ml_num=[f'query-{i}' for i in range(8)],
             date_listed=train.date_sold.max() + pd.Timedelta(days=5))
    prediction = learner.predict(data.hide_answers(homes, track))
    assert (prediction['count'] >= 3).all()
    assert np.isfinite(prediction['price']).all() and (prediction['price'] > 0).all()
    altered = homes.assign(sold_price=1e12, date_sold=pd.Timestamp('2099-01-01'),
                           days_on_market=99999, external_avm_estimate=1e12)
    if track == 'off_market':
        altered['list_price'] = 1e12
    repeated = learner.predict(altered)
    np.testing.assert_array_equal(prediction['price'], repeated['price'])
    no_geo = homes.assign(latitude=np.nan)
    assert np.isnan(learner.predict(no_geo)['price']).all()


def test_pair_target_and_equal_subject_weight(monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, **kwargs): pass
        def fit(self, X, y, **kwargs):
            captured.update(X=X, y=y, **kwargs)

    monkeypatch.setattr(cm, 'CatBoostRegressor', Recorder)
    train = history()
    learner = cm.AdjustedComparables('list_aware').fit(train)
    pairs = cm.retrieve_pairs(train, train)
    ratio = np.log(train.sold_price.to_numpy() / train.list_price.to_numpy())
    np.testing.assert_allclose(captured['y'], ratio[pairs['q']] - ratio[pairs['c']], atol=1e-14)
    totals = np.bincount(pairs['q'], weights=captured['sample_weight'], minlength=len(train))
    np.testing.assert_allclose(totals[totals > 0], 1)
    assert not any('sold_price' in col or 'external_avm' in col or 'days_on_market' in col
                   for col in captured['X'])


def test_blend_fallback_and_persistent_prediction_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('SSU_CACHE_DIR', str(tmp_path))
    monkeypatch.delenv('SSU_DISABLE_CACHE', raising=False)
    monkeypatch.setattr(cm, 'base_signature', lambda: 'test')
    calls = []

    class Base:
        def __init__(self, track): pass
        def fit(self, train): return self
        def predict(self, homes):
            self.component_predictions = {'uncorrected_ensemble': np.full(len(homes), 100.)}
            return np.full(len(homes), 100.)

    class Pair:
        def __init__(self, track): pass
        def fit(self, train): calls.append(True); return self
        def predict(self, homes):
            return dict(price=np.array([120., 120., np.nan, 120.]), count=np.array([8, 2, 0, 8]),
                        spread=np.array([.1, .1, np.nan, .5]), nearest=np.array([1., 1., np.nan, 1.]))

    monkeypatch.setattr(cm, 'CachedEnsemble', Base)
    monkeypatch.setattr(cm, 'AdjustedComparables', Pair)
    train = history(); homes = data.hide_answers(train.iloc[:4], 'off_market')
    model = cm.ComparableEnsemble('off_market').fit(train)
    result = model.predict(homes)
    assert result[0] == pytest.approx(100 * 1.2 ** .3)
    np.testing.assert_array_equal(result[1:], [100., 100., 100.])
    repeated = cm.ComparableEnsemble('off_market').fit(train).predict(homes)
    np.testing.assert_array_equal(result, repeated)
    assert len(calls) == 1
