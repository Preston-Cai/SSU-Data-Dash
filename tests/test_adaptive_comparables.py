import numpy as np
import pandas as pd
import pytest
import data
import models
import adaptive_comparables as ac
import specialist_comparables as sc
from test_comparable_model import history


@pytest.mark.parametrize('family', ['ground', 'condo'])
def test_specialist_retrieval_temporal_and_type_safety(family):
    ref = history(50)
    kind = 'Detached' if family == 'ground' else 'Condo Apt'
    ref['house_type_name'] = kind
    query = ref.iloc[[30]].copy()
    query['date_listed'] = ref.loc[30, 'date_sold']
    pairs = sc.retrieve(ref, query, family)
    assert len(pairs['c']) == (8 if family == 'ground' else 12)
    assert (ref.iloc[pairs['c']].date_sold < query.date_listed.iloc[0]).all()
    assert 30 not in pairs['c']
    altered = query.assign(sold_price=1e12, date_sold=pd.Timestamp('2099-01-01'),
                           list_price=1e12, external_avm_estimate=1e12)
    np.testing.assert_array_equal(sc.retrieve(ref, altered, family)['c'], pairs['c'])
    ref.loc[pairs['c'][0], 'house_type_name'] = 'Other'
    assert pairs['c'][0] not in sc.retrieve(ref, query, family)['c']


@pytest.mark.parametrize('track,family', [('off_market', 'ground'), ('list_aware', 'condo')])
def test_specialist_tiny_cpu_fit(track, family, monkeypatch):
    monkeypatch.setattr(sc, 'PARAMS', dict(iterations=3, depth=3, loss_function='RMSE',
                         task_type='CPU', thread_count=2, verbose=False, allow_writing_files=False))
    train = history(100).assign(house_type_name='Detached' if family=='ground' else 'Condo Apt')
    model = sc.Specialist(track, family).fit(train)
    homes = train.iloc[:4].assign(ml_num=['new1','new2','new3','new4'],
                                  date_listed=train.date_sold.max()+pd.Timedelta(days=2))
    result = model.predict(data.hide_answers(homes, track))
    assert np.isfinite(result['price']).all() and (result['price'] > 0).all()
    assert (result['count'] >= 3).all()
    altered = homes.assign(sold_price=1e12, date_sold=pd.Timestamp('2099-01-01'),
                           days_on_market=12345, external_avm_estimate=1e12)
    if track=='off_market': altered['list_price']=1e12
    np.testing.assert_array_equal(model.predict(altered)['price'], result['price'])


def test_adaptive_training_boundaries_and_fallback(monkeypatch):
    train = history(720)
    months = pd.period_range('2025-01', periods=8, freq='M')
    train['date_sold'] = [months[i//90].start_time+pd.Timedelta(days=15) for i in range(720)]
    train['date_listed'] = train.date_sold-pd.Timedelta(days=5)
    dates = train.set_index('ml_num').date_sold.to_dict()
    seen = []; captured = {}

    def fake_forecast(track, past, homes):
        assert not set(past.ml_num) & set(homes.ml_num)
        known = [dates[x] for x in homes.ml_num if x in dates]
        if known:
            assert past.date_sold.max() < min(known).to_period('M').start_time
            seen.append(past.date_sold.max())
        n = len(homes)
        result = dict(price=np.full(n, 110.), count=np.full(n, 8), spread=np.full(n, .1),
                      nearest=np.ones(n), age=np.full(n, 100.))
        if not known:
            result['count'][0] = 0
        return result

    class Base:
        def __init__(self, track): pass
        def fit(self, past): return self
        def predict(self, homes):
            assert 'sold_price' not in homes
            return np.full(len(homes),100.)

    class V13:
        def __init__(self, track): pass
        def fit(self, past): return self
        def predict(self, homes):
            self.component_predictions={'uncorrected_ensemble':np.full(len(homes),100.)}
            return np.full(len(homes),103.)

    class Trust:
        def __init__(self, **kwargs): pass
        def fit(self,X,y,sample_weight): captured.update(X=X.copy(),y=y.copy(),weight=sample_weight.copy())
        def predict(self,X): return np.full(len(X),2.)  # must be constrained

    monkeypatch.setattr(ac,'forecast',fake_forecast)
    monkeypatch.setattr(ac,'CachedEnsemble',Base)
    monkeypatch.setattr(ac,'ComparableEnsemble',V13)
    monkeypatch.setattr(ac,'HistGradientBoostingRegressor',Trust)
    model=ac.AdaptiveComparableEnsemble('off_market').fit(train)
    assert len(seen)==6 and len(captured['X'])==540
    assert np.isfinite(captured['y']).all() and np.isfinite(captured['weight']).all()
    for row in model.blend_fold_audit:
        assert pd.Timestamp(row['train_max_sale']) < pd.Period(row['month']).start_time
    homes = data.hide_answers(train.iloc[:3].assign(ml_num=['new1','new2','new3']), 'off_market')
    prediction=model.predict(homes)
    assert prediction[0]==103.  # weak specialist falls back to v13, not plain ensemble
    np.testing.assert_allclose(prediction[1:],100*1.1**.75)
    assert model.comparable_weight.max()==.75
    assert 'v13_prediction' in model.component_predictions
    assert 'specialist_fixed_blend' in model.component_predictions
    assert not any('sold_price' in c or 'date_sold' in c or 'avm' in c for c in captured['X'])


def test_trust_feature_whitelist():
    homes=history(10)
    stats=dict(price=np.full(10,1e6),count=np.full(10,8),spread=np.full(10,.1),
               nearest=np.ones(10),age=np.full(10,100.))
    X,_,_=ac.trust_features(homes,np.full(10,1e6),stats,'off_market')
    changed=homes.assign(sold_price=1e12,list_price=1e12,external_avm_estimate=1e12,
                         date_sold=pd.Timestamp('2099-01-01'),days_on_market=99999)
    Y,_,_=ac.trust_features(changed,np.full(10,1e6),stats,'off_market')
    pd.testing.assert_frame_equal(X,Y)


def test_fixed_specialists_reproduce_report_component_without_inner_fits(monkeypatch):
    class V13:
        def __init__(self, track): pass
        def fit(self, train): return self
        def predict(self, homes):
            self.component_predictions={'uncorrected_ensemble':np.full(len(homes),100.)}
            return np.full(len(homes),103.)
    def forecast(track,train,homes):
        return dict(price=np.array([110.,120.,np.nan]),count=np.array([8,2,0]),
                    spread=np.array([.1,.1,np.nan]),nearest=np.array([1.,1.,np.nan]),
                    age=np.full(3,100.))
    def forbidden(*args,**kwargs): raise AssertionError('Fixed blend must not train historical folds')
    monkeypatch.setattr(ac,'ComparableEnsemble',V13)
    monkeypatch.setattr(ac,'forecast',forecast)
    monkeypatch.setattr(ac,'CachedEnsemble',forbidden)
    model=ac.FixedSpecialistEnsemble('list_aware').fit(history(20))
    result=model.predict(history(3))
    np.testing.assert_array_equal(result,model.component_predictions['specialist_fixed_blend'])
    np.testing.assert_allclose(result,[100*1.1**.2,103.,103.])
    assert model.blend_fold_audit==[] and model.trust_model is None
