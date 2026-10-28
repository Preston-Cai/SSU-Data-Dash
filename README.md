# SSU Data Dash: predict Toronto home prices

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
pip install -r requirements.txt
```

The data downloads itself on first use. No Python? Use the
[Colab notebook](https://colab.research.google.com/github/erwinzhang7/SSU-Data-Dash/blob/main/notebooks/quickstart.ipynb).
Prefer R? See `r/baseline.R`.

## Commands

```
python run.py validate   # test on Dec 2025 to May 2026
python run.py report     # results table and a figure
python run.py predict    # write both submission files
python run.py check      # check them
```

The custom neural-model experiments are collected under
[`mlp_attempts/`](mlp_attempts/README.md), separate from the starter runner and
the selected Codex ensemble:

```
pip install -r mlp_attempts/requirements.txt
python -m mlp_attempts.exec train --epochs 10
python -m mlp_attempts.exec test
python -m mlp_attempts.exec report
```

Every checkpoint save gets its actual save time in a new filename, such as
`mlp_attempts/checkpoints/off_market_20261027-153000-123456.pt`. Testing and `--resume`
automatically select the newest checkpoint for each track. Resume an exact file with:

```
python -m mlp_attempts.exec train --resume-from mlp_attempts/checkpoints/off_market_20261027-153000-123456.pt --epochs 10
```

Test two exact model checkpoints by supplying off-market first and list-aware second:

```
python -m mlp_attempts.exec test --checkpoints mlp_attempts/checkpoints/off_market_TIMESTAMP.pt mlp_attempts/checkpoints/list_aware_TIMESTAMP.pt
```
On Windows, press Q during training to save and stop cleanly; some native math
libraries abort on Ctrl+C before Python can save. On other platforms, use Ctrl+C.
Continue later with:

```
python -m mlp_attempts.exec train --resume --epochs 10
```

`--epochs` is the number of epochs to run now, including when resuming. Train one track with
`--track off_market` or `--track list_aware`.

On Windows, use `py` if `python` is not found.

## What to edit

- **`features.py`**: the features.
- **`models.py`**: the models. Two simple baselines to beat: list price (list-aware) and the
  10 nearest earlier sales (off-market). `FeatureModel` (gradient-boosted trees) is ready to switch on.

## Validation

`validate` predicts each month from the sales before it, as in the real test. A random split lets
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

## Rules

- Outside data only if it covers May 2026 or earlier. List every source with its URL.
- Neighbourhoods, coordinates and streets are synthetic, so map data will not line up.
- No real listing or sales records.
- Do not try to recover the hidden test prices.
