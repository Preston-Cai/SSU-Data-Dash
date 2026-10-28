# Complete experiment handoff

This document is the source of truth for the isolated experiments in
`codex_attempts/`. It is deliberately self-contained so that a person or a
new Codex chat can understand the work without access to the conversation that
created it.

For an executable bridge from this explanation to the exact implementation,
open `CODE_WALKTHROUGH.ipynb` in this directory. It displays definitions from
the real source files and reproduces the final blend weights and calibration.

## One-minute explanation

The task is to predict the selling price of synthetic Toronto homes. There are
two separate versions of the problem:

1. **Off-market:** estimate a home's value without seeing its asking price.
   This is harder and counts twice as much in the competition.
2. **List-aware:** predict the selling price after the asking price is known.
   This is easier because the asking price is already a strong estimate.

The original user models were already much better than the starter baselines,
but the off-market model still had a median error of 11.19%. The central idea of
this experiment was that a home's value depends on more than its bedrooms and
property type. It also depends on what similar nearby homes sold for, how
recently they sold, the current market period, and information in the listing
remarks.

Instead of trusting one model, the final solution combines three models that
make different mistakes. The result reduced off-market median error from 11.19%
to 7.28%, and list-aware median error from 4.48% to 4.08%.

## Competition and data context

The complete rules are in the repository's `handout.md`. The important facts
for this experiment are:

- Training data contains 150,000 synthetic sales from January 2021 through May
  2026.
- The hidden test homes sold from June through September 2026.
- Validation must imitate real use: train on earlier sales and predict later
  sales. A random split would leak future market information.
- Off-market predictions are forbidden from using `list_price`.
- `external_avm_estimate` is unavailable in the test data, so this experiment
  does not use it on either track.
- Fields only known after a sale, such as `sold_price` and `days_on_market`, are
  never model inputs.
- The location data is synthetic, so real census or map data would not line up.
  No external data was used here.

The competition's reported metric is **MdAPE**, the median absolute percentage
error. An MdAPE of 7% means that the typical prediction is about 7% away from the
actual selling price. **MAPE** is the mean absolute percentage error; it reacts
more strongly to bad outliers. Lower is better for both.

## Starting point and final outcome

The user's best existing December 2025-May 2026 walk-forward result before this
experiment was:

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|---:|
| off-market | 10,069 | 13.87% | 11.19% | -1.26% |
| list-aware | 10,069 | 5.62% | 4.48% | +0.62% |

The isolated final ensemble obtained:

| Track | MAPE (95% interval) | MdAPE (95% interval) | Median signed error |
|---|---:|---:|---:|
| off-market | **9.85%** (9.65 to 10.03) | **7.28%** (7.11 to 7.46) | -0.86% |
| list-aware | **5.25%** (5.16 to 5.35) | **4.08%** (4.00 to 4.18) | -0.69% |

That is a relative reduction of approximately 29.0% in off-market MAPE, 35.0%
in off-market MdAPE, 6.5% in list-aware MAPE, and 8.9% in list-aware MdAPE.
The off-market improvement is the main success because that track is harder and
worth twice as many accuracy points.

The exact row-level predictions, monthly scores, bootstrap intervals, and signed
errors are stored in:

- `results/final_walk_forward_predictions.csv`
- `results/final_walk_forward_report.json`

## Validation design and an important honesty note

The project uses walk-forward validation. To predict December 2025, the model is
trained only on sales before December. To predict January 2026, it is retrained
on sales before January, and so on through May 2026. This reproduces what will
happen when a fitted model predicts future homes.

Development was split conceptually into two periods:

- **September-November 2025:** screening, model selection, blend-weight
  selection, and calibration.
- **December 2025-May 2026:** the handout's official walk-forward validation
  period.

The final blend weights and calibration constants were calculated from
September-November, not optimized on December-May. However, several finalist
components and an earlier blend were also evaluated on December-May while the
experiment continued. Therefore December-May is an honest time-forward
validation set, but it should not be described as a perfectly untouched final
test set. The competition's held-back June-September homes are the true unseen
test. This distinction matters when discussing expected generalization.

For every validation month, all neighborhood statistics and models were rebuilt
using only rows before that month's cutoff. Target-derived features for a home
never include that home's sale price, and validation targets never enter the
training features.

## Experimental process, in order

### 1. Isolate the work

All new scripts, result files, models, and submissions were placed under
`codex_attempts/`. The starter workflow and the user's existing custom model
were not used as scratch space. This makes the alternative approach easy to
audit or remove and prevents accidental contamination of the old files.

### 2. Build a stronger general feature set

The raw data needs translation before a model can use it well. The experiments
created features in four groups:

- **Physical home features:** the midpoint of square-footage ranges, log floor
  area, lot area, log lot area, bathrooms per bedroom, area per bedroom, area
  per bathroom, parking, garage, and a condominium indicator.
- **Location features:** municipality, community, latitude, longitude, rounded
  coordinate cells at multiple resolutions, and interactions such as community
  plus property type or house style.
- **Time features:** listing year/month and cyclical month indicators. These
  help represent market changes and seasonality.
- **Remarks features:** flags for renovation, condition, luxury, as-is wording,
  transit, basement, and related concepts. Separate text models also used the
  full remarks instead of only flags.

For list-aware predictions, the feature builders may also use list price and
features derived from it. The off-market feature builder does not receive list
price at all.

### 3. Use the right prediction target

Home prices are right-skewed, so directly minimizing errors in dollars gives
expensive homes too much influence. The off-market models instead predict
`log(sold_price)`. This makes the learning problem closer to percentage error
and guarantees positive prices after applying `exp`.

For the list-aware track, the models predict
`log(sold_price / list_price)`. In simple language, the model starts from the
asking price and learns the percentage adjustment needed to reach the likely
selling price. This is more stable than relearning the entire price from
scratch.

### 4. Establish boosted-tree and text baselines

LightGBM was tested with multiple history windows, L1/L2 objectives, leaf counts,
and feature subsets. On November 2025 screening data, ordinary LightGBM reached
8.86% off-market MdAPE and 4.38% list-aware MdAPE.

A separate Ridge regression read word-level TF-IDF features from remarks and
structured text tokens. By itself it was weaker off-market (9.30% MdAPE) but
strong list-aware (about 4.18%). Its value later was diversity: its errors were
not identical to the tree models' errors.

### 5. Add leakage-safe local price history

This was the most important feature-engineering improvement.

For each community, community-property combination, geographic cell, and exact
coordinate group, the code summarizes earlier sales using:

- typical log sale price;
- typical log price per square foot; and
- for list-aware models, the typical sold-to-list ratio.

Rare groups are noisy, so their value is smoothed toward the overall market
average. The `prior` parameter controls how strongly a small group is pulled
toward that average. Training rows use leave-one-out calculations so a row
cannot simply reveal its own target to the model. When predicting a validation
or test row, all statistics come from the allowed training history.

Adding these target encodings reduced the November off-market screening MdAPE
from 8.86% to 7.75%. It helped far more than merely changing tree libraries,
which is why feature engineering—not model branding—is the main story.

### 6. Make local history sensitive to recency

A sale from last month is usually more relevant than a sale several years ago.
Recency-weighted encoders therefore give older observations exponentially less
weight. Several half-lives were tested. On the November screen, this produced a
small additional off-market improvement to about 7.69% MdAPE.

Both ordinary and recency-weighted encodings were retained because they answer
slightly different questions: the ordinary version estimates a stable local
level, while the recency version follows changing market conditions.

### 7. Test other model families and valuation layers

The experiments did not assume that boosted trees were automatically best:

- **XGBoost** worked well with the same engineered and historical features. Its
  November off-market result was 7.62% MdAPE, and its mistakes were sufficiently
  different from LightGBM to make it useful in a blend.
- **CatBoost** was tested because its ordered boosting is designed for
  categorical variables. It was slow and reached 10.34% off-market MdAPE, so it
  was not promoted. Its list-aware result was competitive at 4.17%.
- **Direct comparable sales** selected nearby, recent, physically similar
  properties and combined their prices. The best version reached only 11.83%
  off-market MdAPE. The exact estimator was rejected, but the idea motivated the
  successful local-history features.
- **Local-anchor residual models** first estimated a neighborhood anchor and
  then asked a tree to predict the difference. Their best screen was about 7.85%
  MdAPE, worse than the direct log-price trees.
- **Character TF-IDF, longer text histories, sample weighting, alternative list
  adjustments, and several stacking variants** were also tested. None improved
  the prevalidation results enough to justify more complexity.

Negative results were kept in `results/` rather than hidden. They explain why
the final solution looks the way it does.

### 8. Blend models with different strengths

An ensemble helps when its components make different errors. Predictions are
combined with a weighted geometric mean, which is the natural average for
log-price models. A small multiplicative calibration then corrects systematic
over- or underprediction. Weights and calibration were selected using the
September-November prevalidation predictions.

The frozen final off-market ensemble is:

- 15% recency-aware LightGBM;
- 20% word TF-IDF Ridge;
- 65% XGBoost; and
- a final multiplicative calibration of `0.9772745745864185`.

The frozen final list-aware ensemble is:

- 35% ordinary target-encoded LightGBM;
- 35% recency-aware LightGBM;
- 30% word TF-IDF Ridge; and
- a final multiplicative calibration of `0.9865499704372777`.

The text model remains in both ensembles even when it is weaker alone because it
adds information and error diversity not captured by the trees.

### 9. Run the exact six-month walk-forward validation

Every component was retrained separately for each month from December 2025
through May 2026. The final aggregate scores are reported above. Off-market
monthly MdAPE ranged from 7.05% to 7.67%; list-aware monthly MdAPE ranged from
3.87% to 4.37%. The relatively stable monthly results are encouraging, although
May is slightly worse and future market drift remains a risk.

### 10. Train once for the real test and save everything

`train_final.py` trains the frozen ensemble once using every permitted listing
through May 2026. It then predicts the June-September test homes and saves:

- one full `cloudpickle` checkpoint containing fitted feature builders,
  historical encoders, models, blend weights, and calibration;
- native LightGBM and XGBoost model files for inspection;
- serialized Ridge models;
- the two submission CSV files; and
- a manifest with hashes, validation metadata, and the actual save timestamp.

`predict_artifact.py` reloads the full checkpoint and predicts without training.
The reload was tested and reproduced both saved submissions byte-for-byte. Both
submission files also pass the starter `check_submission.py` checks.

## Compact experiment evidence

These are November 2025 screening results, not the official six-month scores.
They show how the reasoning evolved. Because different configurations were used,
the table is directional rather than a perfectly controlled ablation study.

| Approach | Off-market MdAPE | List-aware MdAPE | Decision |
|---|---:|---:|---|
| Ordinary LightGBM features | 8.86% | 4.38% | Keep as foundation |
| Word TF-IDF Ridge | 9.30% | 4.18% | Keep for text and diversity |
| Leakage-safe target encodings | 7.75% | 4.35% | Major off-market gain; keep |
| Recency-weighted encodings | 7.69% | 4.36% | Small complementary gain; keep |
| XGBoost with encoded features | 7.62% | 4.34% | Best tree screen; keep |
| CatBoost | 10.34% | 4.17% | Reject off-market; no blend gain |
| Direct comparable estimator | 11.83% | not promoted | Reject estimator, retain concept |
| Local-anchor residual model | 7.85% | not promoted | Reject; simpler target worked better |

The raw CSV files in `results/` contain every screened configuration, rather
than only the best row.

## Research and how it affected the experiments

Research was used to choose experiments, not to borrow data.

1. The CatBoost paper's ordered boosting and target-statistic ideas motivated
   testing CatBoost and carefully constructed leave-one-out categorical history:
   <https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html>
2. Work on location, time, and physical similarity in property assessment
   motivated geographic cells, property interactions, recency weighting, and
   local comparable features:
   <https://researchexchange.iaao.org/jptaa/vol14/iss2/1/>
3. Research treating comparable selection as a data-driven valuation problem
   motivated the explicit comparable and local-anchor experiments:
   <https://ideas.repec.org/a/spr/anresc/v64y2020i2d10.1007_s00168-019-00959-2.html>

The papers influenced the architecture, but no outside records or pretrained
property model were added to the data.

## Reproducing the work

Run commands from the repository root in PowerShell. The experiment used the
project's `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r "codex_attempts\requirements-attempts.txt"

# Rebuild the final six-month component predictions and report. This is slow.
.\.venv\Scripts\python.exe "codex_attempts\validate.py"
.\.venv\Scripts\python.exe run.py report

# Train the frozen final models once and create a timestamped artifact.
.\.venv\Scripts\python.exe "codex_attempts\train_final.py"

# Predict from an already trained artifact; this does not retrain.
.\.venv\Scripts\python.exe "codex_attempts\predict_artifact.py" `
  "codex_attempts\artifacts\TIMESTAMP\ensemble_checkpoint.pkl"
```

The finalized artifact from this run is
`artifacts/20261028-000439-310846/`. The most important files are:

- `ensemble_checkpoint.pkl`: the complete reloadable ensemble;
- `manifest.json`: hashes, weights, calibration, validation summary, and time;
- `submission_off_market.csv`: final off-market test predictions; and
- `submission_list_aware.csv`: final list-aware test predictions.

Artifact directory names are created when saving begins after fitting, not when
training starts. This avoids the earlier timestamp bug where a long training
run appeared to have been saved at its start time.

## Code map

| File | Purpose |
|---|---|
| `experiments/experiment.py` | Shared parsing, base features, metrics, and early CatBoost experiment |
| `experiments/lightgbm_experiment.py` | Base LightGBM screening and reusable feature builder |
| `experiments/linear_text_experiment.py` | TF-IDF plus Ridge text models |
| `experiments/target_encoding_experiment.py` | Smoothed leave-one-out local history features |
| `experiments/recency_encoding_experiment.py` | Exponentially recency-weighted history features |
| `experiments/comparables_experiment.py` | Direct comparable-sale estimator |
| `experiments/anchor_residual_experiment.py` | Local anchor plus residual model tests |
| `experiments/catboost_focused.py` | Focused CatBoost screen |
| `experiments/xgboost_experiment.py` | XGBoost screen |
| `experiments/ensemble_prevalidation.py` | September-October component predictions |
| `experiments/xgboost_prevalidation.py` | September-October XGBoost predictions |
| `experiments/blend_analysis.py`, `experiments/stack_analysis.py` | Blend and stacking investigations |
| `experiments/ensemble_walk_forward.py` | Six-month LightGBM/text component walk-forward |
| `experiments/xgboost_walk_forward.py` | Six-month XGBoost walk-forward |
| `finalize_results.py` | Apply frozen blend and write the final validation report |
| `train_final.py` | Train once on all permitted history and save artifact/submissions |
| `predict_artifact.py` | Load an artifact and predict without retraining |
| `validate.py` | Recompute walk-forward validation and export the starter report CSV |
| `select_final_blend.py` | Reproduce the final weights and multiplicative calibration |
| `docs/CODE_WALKTHROUGH.ipynb` | Explain and execute the important implementation pieces |

Other scripts under `experiments/` are retained targeted experiments. Their
names and corresponding result CSVs indicate the hypothesis they tested.

## Limits, risks, and sensible next work

- June-September remains unseen, so validation improvement is evidence, not a
  guarantee of the hidden-test result.
- The final ensemble is computationally heavier and operationally more complex
  than one tree model.
- The text remarks are synthetic and may not generalize like real agent-written
  remarks.
- Rare communities and property combinations still have weak local evidence;
  smoothing reduces but does not eliminate this problem.
- May's off-market error was higher than earlier validation months, suggesting
  market drift should be watched.
- December-May was inspected during development, as disclosed above. Future
  changes should be chosen on a new prevalidation window or nested time splits,
  not repeatedly tuned against the reported six-month result.
- The next most valuable work is likely segmented error analysis by price band,
  property type, community frequency, and month, followed by uncertainty
  intervals or conformal prediction. These items matter to the judging rubric
  and were not fully developed in this experiment.

## What a future Codex chat should say in plain language

If asked to explain this work, start with the problem and the two tracks. Then
explain that the main improvement was teaching the model about earlier local
sales without letting it see the future. Describe the ensemble as three
independent opinions: structured-property/local-history trees, a second boosted
tree implementation, and a text reader. Explain that weaker standalone models
can still improve an average when they make different mistakes. Mention the
failed comparable and CatBoost experiments to show that alternatives were
measured rather than assumed. Finally, state the validation caveat and the exact
9.85%/7.28% off-market and 5.25%/4.08% list-aware MAPE/MdAPE results.

Do not claim that AVM, future sales, hidden targets, outside property data, or
list price were used by the off-market model. Do not describe December-May as a
perfectly untouched test set. Do not say the gain came only from switching model
families: the largest measured gain came from leakage-safe local-history feature
engineering.
