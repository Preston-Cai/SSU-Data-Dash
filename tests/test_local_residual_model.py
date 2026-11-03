import numpy as np
import pandas as pd
import data
import local_residual_model as lr
from test_comparable_model import history


def test_local_errors_exclude_self_same_day_and_future():
    ref = history(30).assign(house_type_name='Detached', oof_error=np.linspace(-.1,.1,30))
    homes = ref.iloc[[0]].copy()
    homes['date_listed'] = ref.loc[10,'date_sold']
    before = lr.local_error_features(ref, homes)
    assert before.error_local_365_count.iloc[0] == 9
    changed = ref.copy()
    changed.loc[10:, 'oof_error'] = 1e6
    changed.loc[0,'oof_error'] = 1e6
    pd.testing.assert_frame_equal(lr.local_error_features(changed,homes), before)
    changed_query=homes.assign(sold_price=1e12,list_price=1e12,external_avm_estimate=1e12,
                                date_sold=pd.Timestamp('2099-01-01'))
    pd.testing.assert_frame_equal(lr.local_error_features(ref,changed_query), before)
    missing = homes.assign(date_listed=pd.NaT)
    assert lr.local_error_features(ref,missing).error_local_365_count.iloc[0]==0


def test_temporal_head_and_cpu_api(monkeypatch):
    train=history(900)
    months=pd.period_range('2025-01',periods=10,freq='M')
    train['date_sold']=[months[i//90].start_time+pd.Timedelta(days=15) for i in range(900)]
    train['date_listed']=train.date_sold-pd.Timedelta(days=10)
    dates=train.set_index('ml_num').date_sold.to_dict(); fits=[]
    monkeypatch.setattr(lr,'HEAD_PARAMS',dict(iterations=3,depth=3,task_type='CPU',
                        loss_function='RMSE',thread_count=2,verbose=False,allow_writing_files=False))

    class Base:
        def __init__(self,track): assert track=='off_market'
        def fit(self,past): self.past=past;fits.append(past);return self
        def predict(self,homes):
            assert 'sold_price' not in homes
            assert self.past.date_sold.max()<min(dates[x] for x in homes.ml_num).to_period('M').start_time
            return np.full(len(homes),1e6)

    def stats(track,past,homes):
        assert not set(past.ml_num)&set(homes.ml_num)
        n=len(homes)
        return dict(price=np.full(n,1.03e6),count=np.full(n,8),spread=np.full(n,.1),
                     nearest=np.ones(n),age=np.full(n,100.))

    class V14:
        def __init__(self,track): pass
        def fit(self,past): self.blend_fold_audit=[];return self
        def predict(self,homes):
            n=len(homes);self.component_predictions={'uncorrected_ensemble':np.full(n,1e6)}
            self.comparable_count=np.full(n,8);self.comparable_spread=np.full(n,.1)
            self.comparable_weight=np.full(n,.3)
            return np.full(n,1.02e6)

    original=lr.head_features; seen=[]
    def checked(homes,base,stats,earlier):
        known=[dates[x] for x in homes.ml_num if x in dates]
        if known:
            assert earlier.date_sold.max()<min(known).to_period('M').start_time
            assert not set(homes.ml_num)&set(earlier.ml_num)
            seen.append(len(homes))
        return original(homes,base,stats,earlier)

    monkeypatch.setattr(lr,'CachedEnsemble',Base)
    monkeypatch.setattr(lr,'forecast',stats)
    monkeypatch.setattr(lr,'AdaptiveComparableEnsemble',V14)
    monkeypatch.setattr(lr,'head_features',checked)
    model=lr.LocalResidualEnsemble().fit(train)
    assert len(fits)==9 and len(seen)==6 and sum(seen)==540
    assert model.head is not None
    homes=data.hide_answers(train.iloc[:2].assign(ml_num=['new1','new2'],
                          date_listed=pd.Timestamp('2025-11-01')), 'off_market')
    result=model.predict(homes)
    assert np.isfinite(result).all() and (result>0).all()
    assert (model.local_correction_weight==.35).all()
    result=model.predict(homes.assign(community='unknown place'))
    np.testing.assert_array_equal(result,[1.02e6,1.02e6])
    assert 'v14_prediction' in model.component_predictions
