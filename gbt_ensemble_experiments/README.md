This folder contains the isolated model-development work for the SSU Data Dash.
The project's existing starter and custom-model files remain outside this
folder.

## Start here

Choose the document that matches what you need:

| Need | Read |
|---|---|
| Understand the approach without much ML background | [`docs/HOW_WE_BUILT_THE_MODEL.md`](docs/HOW_WE_BUILT_THE_MODEL.md) |
| Connect the important ideas to executable source code | [`docs/CODE_WALKTHROUGH.ipynb`](docs/CODE_WALKTHROUGH.ipynb) |
| Run, retrain, or locate the selected model | [`FINAL_MODEL.md`](FINAL_MODEL.md) |
| Review every experiment and technical decision | [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md) |

## Folder layout

```text
gbt_ensemble_experiments/
├── README.md                 this navigation page
├── FINAL_MODEL.md            short operational instructions
├── train_final.py            train the selected ensemble once
├── predict_artifact.py       predict from a saved checkpoint
├── validate.py               walk-forward validation and starter CSV export
├── select_final_blend.py     reproduce weights and calibration from Sep-Nov
├── finalize_results.py       build the frozen validation blend/report
├── requirements-attempts.txt isolated Python dependencies
├── docs/                     accessible article and technical report
├── experiments/              exploratory and walk-forward scripts
├── results/                  validation predictions and experiment tables
└── artifacts/                timestamped fitted models and submissions
```

## Main commands

Run commands from the repository root in PowerShell:

```powershell
# Train the final selected model and save a new timestamped checkpoint.
.\.venv\Scripts\python.exe "gbt_ensemble_experiments\train_final.py"

# Predict with the existing checkpoint without retraining.
.\.venv\Scripts\python.exe "gbt_ensemble_experiments\predict_artifact.py" `
  "gbt_ensemble_experiments\artifacts\20261028-000439-310846\ensemble_checkpoint.pkl"

# Rebuild the six-month component predictions and final validation report (slow).
.\.venv\Scripts\python.exe "gbt_ensemble_experiments\validate.py"

# Then generate the normal starter report from those predictions.
.\.venv\Scripts\python.exe run.py report
```

If validation has already been computed and you only need to restore the CSV
expected by `run.py report`, use:

```powershell
.\.venv\Scripts\python.exe "gbt_ensemble_experiments\validate.py" --reuse-results
.\.venv\Scripts\python.exe run.py report
```

The experiments obey the handout constraints: validation always predicts the
future from the past, off-market never uses list price, and neither track uses
the external AVM or after-sale fields.
