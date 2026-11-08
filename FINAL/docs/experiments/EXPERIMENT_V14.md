# V14: specialist comparables with learned trust

```powershell
$env:MODEL_PROFILE = "adaptive_comparables"
python run.py validate
python run.py report
```

This profile is now the default when MODEL_PROFILE is unset. An existing shell
setting takes precedence, so explicitly select it. No new dependencies are
needed. Full training and validation are left to the user.

## Specialists

Each track trains separate ground-oriented and condo-style pair learners.
Property type still has to match exactly within each family. Ground-oriented
homes use up to eight comparables and rank by distance, size, rooms, style,
lot-area difference, basic condition flags and recency. Condo-style homes use
up to twelve, with more emphasis on size and proximity. Coordinates are a
proximity proxy; they do not establish that two units share a building.

Both keep the strict sale-before-listing-date rule and 730-day lookback. Same
community is preferred; municipality fallback is used with fewer than three
eligible local sales. Each family independently fits its text vocabulary and
32-component SVD on its outer training rows. Size and lot-area log differences
are added. Each pair learner has 2,400 trees, depth 8 and RMSE loss. Comparable
outcomes remain external anchors, never subject feature inputs.

## Learning trust without using the outer labels

For each outer fold, the preceding six sale months supply out-of-fold forecasts.
For each historical month, both the base ensemble and specialists train on
sales strictly before that month's start, then predict its held-out homes.
Only those historical errors train the trust model. The current outer month's
outcomes and in-sample price predictions never train that month's trust model.
Previously evaluated months may become historical training data in later folds,
as in the existing walk-forward protocol.

Trust inputs include property type, size, rooms, base predicted value,
base/specialist disagreement, comparable count, dispersion, distance and age.
List-aware also uses base price relative to the known asking price. Inputs are
explicitly whitelisted; subject sold prices and after-sale fields are excluded.

Let r = log(actual/base), clipped to +/-0.5, and d = log(specialist/base), clipped
to +/-0.4. The training objective is `(r - w*d)^2 + 0.0025*(w - prior)^2`.
Equivalently, the trust regressor uses target
`(d*r + 0.0025*prior)/(d*d + 0.0025)` with importance `d*d + 0.0025`.
This avoids dividing directly by tiny prediction disagreements. Priors remain
0.30 off-market and 0.20 list-aware. The CPU trust learner has 100 boosting
iterations, seven leaves, at least 250 rows per leaf, regularization 10, and no
random early-stopping split. Predicted weights are constrained to [0, 0.75].

Specialists require at least three comparable sales, nearest distance at most
5 km and adjusted log-price spread at most 0.30. Weak specialist evidence falls
back to the exact v13 prediction. With fewer than 500 usable historical OOF
rows, the specialist uses the fixed prior weight. Settings were fixed before
this validation; no automatic tuning against its results is performed.

## Reading the report

- `predicted_price`: the adaptive specialist challenger.
- `member_v13_prediction`: the preserved successful v13 on the same homes.
- `member_specialist_fixed_blend`: specialists with fixed 30%/20% weights.
- `member_specialist_comparables`: standalone specialists, falling back to v13
  on homes that fail the evidence checks.
- `member_uncorrected_ensemble`: the original ensemble.
- `member_adjusted_comparables`: the original v13 pooled comparable learner.

The prediction CSV records specialist count, spread and actual blend weight.
`blend_audit.json` in each run snapshot records the six historical training
cutoffs and sample counts. The learned blend is an experiment, not a confirmed
improvement. If the fixed specialist blend improves while the adaptive blend
does not, that distinction is visible without another validation run.

## Runtime and restoration

V13's source and prediction-cache keys are unchanged. Specialists have a
separate disk cache for each family, track, training set and query set. Changing
only adaptive_comparables.py reuses all expensive price forecasts. A cold
six-month validation needs up to 48 specialist fits (12 unique cutoffs, two
families, two tracks). Existing historical base forecasts are reused where
available; some historical list-aware cutoffs may still need training. The
first run can be substantially slower than v13. Subsequent runs reuse completed
specialist work, including after a restart. The trust model itself is refitted
on CPU for each fold; it never requires another nested trust-model validation.

Checkpoints are untouched. To use the retained v13:

```powershell
$env:MODEL_PROFILE = "comparables"
```
