# SSU Data Dash: predict Toronto home prices

## Our Final submission model

| Purpose | Link |
|---|---|
| Understand the final model and its source code | [`gbt_ensemble_experiments/README.md`](gbt_ensemble_experiments/README.md) |
| Reproduce and verify the submitted CSVs | [`SUBMISSION_README.md`](SUBMISSION_README.md) |
| Read the final four-page report | [`report/report.pdf`](report/report.pdf) |

The final model is the ensemble implemented in `gbt_ensemble_experiments/`,
not the starter implementation in `models.py` and `features.py`. All setup,
prediction, and verification instructions needed to reproduce the submitted
CSVs from a fresh clone are kept in `SUBMISSION_README.md`.

`python run.py validate` and `python run.py predict` are wired to that final
ensemble. Prediction writes the two official CSVs in `submissions/`. The
starter modules are retained only as the original baseline/reference
implementation. `run.py predict` reuses a matching fitted checkpoint when one
exists and automatically trains the final ensemble when one does not, as
detailed in `SUBMISSION_README.md`.

## Instructions
Start with the case handout, [`handout.md`](handout.md), and the judging rubric, [`rubric.md`](rubric.md).

Two tracks:
- **Off-market:** value a home with no list price.
- **List-aware:** predict a listed home's sale price, knowing its list price.

Train on 150,000 sales (January 2021 to May 2026). Predict homes that sold June to September 2026.

**Data:** [huggingface.co/datasets/Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).
All of it is synthetic.

## Setup

Python 3.10 or newer (3.11+ on macOS 27):

```
pip install -r gbt_ensemble_experiments/requirements.txt
```

The root `requirements.txt` is retained for the starter baseline only.

The data downloads itself on first use. No Python? Use the
[Colab notebook](https://colab.research.google.com/github/erwinzhang7/SSU-Data-Dash/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate  # final GBT walk-forward validation (slow)
python run.py report    # results table and figure
python run.py predict   # train if needed; write final GBT submission files
python run.py check     # check both CSVs
```

On Windows, use `py` if `python` is not found.

## Retained starter files

- **`features.py`**: starter feature examples; not used by the final model.
- **`models.py`**: starter baselines; not used by final prediction. They remain available for
  comparison through `python run.py validate-starter`.

## Validation

`python run.py validate` predicts each month with the final ensemble using sales
from before that month, as in the real test. Use `python run.py validate
--reuse-results` to export the saved walk-forward predictions without refitting.
The explicitly named `validate-starter` command runs the retained baselines. A random split lets
the model see the future and makes your score look better than it is. Judges compare your
validation error with your hidden-test error.

## Submission

One PDF report (template: `report/report.tex`, 4 pages max) that links to:
1. your video on YouTube, 5 minutes max (unlisted is fine, not private);
2. your GitHub repository (public, or shared with the organisers), holding your code and two files:

```
submissions/submission_off_market.csv    one row per home in test_off_market
submissions/submission_list_aware.csv    one row per home in test_list_aware

ml_num,predicted_price
SYN-0123456,815000
```

`predict` writes both; run `check` before you hand in. Deadline: see the case email.

The prepared final files are in `submissions/`. Follow
[`SUBMISSION_README.md`](SUBMISSION_README.md) to reproduce and verify them.

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- Neighbourhoods, coordinates and streets are synthetic, so map data will not line up.
- No real listing or sales records.
- Do not try to recover the hidden test prices.
