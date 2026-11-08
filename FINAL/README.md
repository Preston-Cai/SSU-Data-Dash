# DATADASH — v15 home-price prediction

Predict synthetic Toronto home prices in two tracks: off-market (no asking
price) and list-aware (asking price available). Read [handout.md](handout.md)
and [rubric.md](rubric.md) for the competition requirements.

## Examiner quick start

Use the `jiang-attempts` branch. Python 3.10–3.13 is recommended with the pinned
requirements. Run from the repository root:

```sh
python -m pip install -r requirements.txt
python run.py validate
python run.py report
python run.py predict
python run.py check
```

The default profile is `local_residual` (v15). The dataset downloads on first
use; set `DATA_DIR` to a local dataset folder to avoid downloading it. V15 does
not require PyTorch or the optional semantic-text dependencies.

CatBoost automatically uses GPU when one is detected, otherwise CPU. To force
CPU in PowerShell:

```powershell
$env:SSU_DEVICE = "cpu"
$env:SSU_THREADS = "8"
python run.py validate
```

On macOS/Linux: `SSU_DEVICE=cpu SSU_THREADS=8 python run.py validate`.
Explicit GPU selection is configured in `runtime_config.py` and requires a
working CUDA installation. CPU runs are supported, but full v15 validation performs many historical fits and can be
slow. CPU/GPU training can produce different predictions; the saved scores are
historical GPU results, not a promise of identical CPU scores.

## Inspect results without training

The [saved v15 report](checkpoints/local_residual_v15/MODEL_REPORT.md) and
[row-level predictions](checkpoints/local_residual_v15/validation_predictions.csv)
are included in Git. Saved December 2025–May 2026 results:

| Track | MdAPE | MAPE |
|---|---:|---:|
| Off-market | 7.280% | 9.740% |
| List-aware | 3.843% | 4.903% |

These are the preserved original run, not a rerun of the reorganized code.
Some model choices used inspected validation results, as documented in the
[experiment history](docs/experiments/EXPERIMENT_V15.md).

## Code map

| File | Responsibility |
|---|---|
| `run.py` | Validation, reports, submissions, source snapshots |
| `data.py` | Data loading and hiding unavailable fields |
| `features.py` | Observable property and listing-text features |
| `history_features.py` | Sale-before-listing neighborhood and spatial statistics |
| `models.py` | Profile selection, CatBoost learners, base ensemble |
| `comparable_model.py` | Learned adjustments to earlier comparable sales |
| `specialist_comparables.py` | Separate house/condo comparable learners |
| `adaptive_comparables.py` | Historical trust model and fixed list-aware blend |
| `local_residual_model.py` | V15 local out-of-fold error correction |
| `runtime_config.py` | CPU/GPU selection and tree-training thread budget |
| `ensemble_runtime.py` | Persistent base-ensemble prediction reuse |
| `experiment_cache.py`, `feature_cache.py` | Disk arrays and bounded in-process feature reuse |
| `text_correction.py` | Retained v11/v12 alternatives; not used by v15 |
| `tests/` | Temporal boundaries, caches, model routing and CPU checks |
| `docs/experiments/` | V12–v15 experiment explanations |
| `checkpoints/` | Preserved historical source and result snapshots |

## What the cache does for an examiner

`outputs/cache/` is ignored by Git. A fresh GitHub clone has **no populated
prediction cache**. Saved checkpoint CSVs are evidence, not executable model
weights and not automatic cache entries.

During the first run, completed component predictions are cached. Identical
later requests and interrupted runs can reuse them. Cache keys include data,
source, model settings, libraries and the CPU/GPU runtime policy. Different
cutoffs, homes, environments or code may require fitting again. Changes in this
cleanup invalidate earlier cache signatures.

Use `SSU_CACHE_DIR` to choose a persistent local cache folder. Copying a cache
from another machine helps only when its keys match; a GPU cache is not silently
used as evidence of a fresh CPU fit. These caches contain predictions, not
portable fitted models. Predicting new hidden homes generally requires training.

## Validation and development

Validation trains before each sale month. Historical neighborhood, comparable
and local-error features use sales strictly before the query listing date.
Off-market hides list price; both tracks exclude unavailable after-sale fields.

Run the small regression suite with `python -m pytest tests -q`. Full training
and six-month validation remain separate, expensive commands.

Old checkpoints and optional experiments are retained for auditability. The
active v15 path does not run semantic text encoding. Model selection is fixed
to v15 in `models.py`; shell environment variables cannot switch profiles.
