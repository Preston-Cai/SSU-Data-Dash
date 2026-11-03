# V15: local error correction off-market; fixed specialist blend list-aware

```powershell
$env:MODEL_PROFILE = "local_residual"
python run.py validate
python run.py report
```

No new dependencies. The default is local_residual; an existing PowerShell
MODEL_PROFILE setting still overrides it. Full validation remains user-run.

## List-aware selection

The saved v14 report's `member_specialist_fixed_blend` scored 3.843% MdAPE and
4.903% MAPE, compared with 3.857% / 4.906% for its learned blend and 3.854% /
4.920% for v13. The selected list-aware model is now that fixed blend: 20%
specialist comparables and 80% original ensemble in log-price space, with the
same eligibility checks, disagreement cap and v13 fallback as the reported
component. It skips the historical trust-model folds. Selection is based on an
existing validation component; this is not a new independent validation score.

Both local_residual and adaptive_comparables profiles now use this selection
for list-aware. The comparables profile still reproduces v13 in both tracks.

## Off-market experiment

V14 only changes the balance between two price forecasts. The new challenger
can also learn corrections when those forecasts share an error. It builds a
history of the original ensemble's out-of-fold errors, log(actual/base), using
nine earlier monthly holdouts. Every base forecast is produced by a model
trained strictly before that holdout month's start.

The first three months warm up the error history. The following six train a
small CPU CatBoost residual model (700 trees, depth 5, learning rate .03,
regularization 20, RMSE loss). Inputs include base/specialist disagreement,
comparable quality, observable house context, and prior-error summaries:

- 90-day and 365-day mean/median errors and counts by property type and by
  community plus type.
- A weighted average and dispersion from up to 20 prior error observations
  selected by distance, size and recency within the community/type.

Each query can use only reference sales strictly before its listing date,
with its own listing ID excluded. Historical training features are built
before appending any outcomes from their month. No outer-month errors enter
that month's fit. Later walk-forward folds may use earlier evaluated months,
following the established temporal protocol. The original ensemble's residual
labels are clipped to +/-0.5 in log space; this applies only to the new head.

The residual model produces a corrected original-ensemble price. Where at least
five eligible local error observations exist, the final challenger blends 35%
of this price with 65% of v14 in log space. Head corrections and the final
candidate/v14 log difference are capped at +/-0.25. Otherwise it uses v14.
These settings are fixed before this experiment's validation.

Compare `predicted_price` with `member_v14_prediction`; the report also includes
`member_local_corrected_base` and all existing v13/specialist components.
The CSV records local_correction_weight. local_correction_audit.json records
the historical training cutoffs and which months trained the residual head.

## Runtime and preservation

Original ensemble, v13 and specialist prediction caches remain valid. The new
head needs three older base-ensemble monthly cutoffs beyond v14's six-month
history. With the v14 caches populated, this can add up to 21 old off-market
CatBoost fits on the first run (three cutoffs times seven member fits).
Subsequent runs reuse those forecasts. The local head and v14 trust model
retrain cheaply on CPU each run; the experiment adds no new specialist fits
where matching cached forecasts exist. Feature or data changes still invalidate
affected caches. No runtime or accuracy improvement is confirmed until tested.

Use `$env:MODEL_PROFILE = "adaptive_comparables"` for retained v14 off-market
plus the selected fixed list-aware blend, without the new local correction.
Existing checkpoints and saved validation outputs are untouched.
