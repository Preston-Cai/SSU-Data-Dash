This folder contains the isolated model-development work for the SSU Data Dash.
The project's existing starter and custom-model files remain outside this
folder.

## Start here

Choose the document that matches what you need:

| Need | Read |
|---|---|
| Understand the approach without much ML background | [`docs/HOW_WE_BUILT_THE_MODEL.md`](docs/HOW_WE_BUILT_THE_MODEL.md) |
| Connect the important ideas to executable source code | [`docs/CODE_WALKTHROUGH.ipynb`](docs/WALKTHROUGH.ipynb) |
| Reproduce the submitted CSVs and validation results | [`../SUBMISSION_README.md`](../SUBMISSION_README.md) |
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
├── requirements.txt          final-model Python dependencies
├── docs/                     accessible article and technical report
├── experiments/              exploratory and walk-forward scripts
├── results/                  validation predictions and experiment tables
└── artifacts/                timestamped fitted models and submissions
```

## Reproducing the submitted CSVs and validation results

Use the single authoritative guide at
[`../SUBMISSION_README.md`](../SUBMISSION_README.md). Checkpoints are generated
locally and are not present in a fresh clone; the documented `run.py predict`
command trains the selected model when needed and writes both submission files.
The same guide contains the commands for rebuilding the six-month walk-forward
validation results and report outputs.

The experiments obey the handout constraints: validation always predicts the
future from the past, off-market never uses list price, and neither track uses
the external AVM or after-sale fields.
