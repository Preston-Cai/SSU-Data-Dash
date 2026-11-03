# V13: learned comparable-sale adjustments

Run from the project root:

```powershell
$env:MODEL_PROFILE = "comparables"
python run.py validate
python run.py report
```

This is now the default when MODEL_PROFILE is unset. An existing PowerShell
setting overrides it, so explicitly select `comparables`. No new dependencies
or neural encoder downloads are required. V11 and exp_v12(failed) are untouched.

The challenger learns from pairs of homes: a subject and up to eight earlier
sales of the same property type. Comparable sales must precede the subject's
listing date and be within the preceding 730 days. Same-community sales are
preferred, with municipality fallback when fewer than three are available.
Distance, size, bedroom/bathroom differences and recency determine selection.
Self-listings and same-day/future sales are excluded.

The adjustment model receives the subject's features, differences from the
comparable, locations/styles, comparable age/distance, and 32 fold-fitted
TF-IDF/SVD text components and their differences. Known comparable outcomes
anchor reconstruction outside the learner, rather than entering its inputs. The
vocabulary is fitted only on the outer training fold; no validation vocabulary
or outcomes are fitted. After-sale fields are excluded from subject features.

Off-market learns log(subject sale price / comparable sale price). List-aware
learns the difference in log(sale/list) ratios, then reconstructs a price using
the subject's known asking price. Each subject has equal total training weight
regardless of how many comparables it has. Each track adds one 2,400-tree
CatBoost learner per month, with fixed settings and no outer-label tuning.

Adjusted comparable prices are combined using a distance/recency-weighted
median. The experiment blends this with the existing ensemble in log-price
space: 30% comparable weight off-market and 20% list-aware. Homes fall back
entirely to the baseline if there are fewer than three comparables, the nearest
is over 5 km away, or adjusted log-price dispersion exceeds 0.30. Before
blending, the comparable-to-baseline log difference is capped at +/-0.40.
These settings are fixed before validation, not selected from its results.

Report columns:

- `predicted_price`: the new blended challenger.
- `member_uncorrected_ensemble`: the retained baseline on the identical homes.
- `member_adjusted_comparables`: standalone comparable prediction, falling back
  to baseline on homes that fail the evidence checks.

The prediction CSV also records `comparable_count`, `comparable_weight` and
`comparable_spread`, so performance can be evaluated on active homes separately.

Existing base prediction cache keys are preserved. The challenger's predictions
are cached separately by source, training data, track and query data. A cold
run adds 12 full pair-model fits across six months; it does not repeat the six
historical text-correction folds. The first pair training can still be sizable
(up to eight rows per historical subject); no runtime or accuracy gain is
guaranteed. There is no automatic selection based on validation labels.

To return to the retained ensemble, set `$env:MODEL_PROFILE = "ensemble"`.
Full validation is left to the user; local tests cover temporal exclusion,
feature leakage, native CatBoost API compatibility on tiny CPU data, fallback,
cache reuse and baseline preservation.
