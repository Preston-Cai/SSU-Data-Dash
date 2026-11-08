# Final submission handoff

The final submission files are ready in `submissions/`:

```text
submissions/submission_off_market.csv   3,974 rows
submissions/submission_list_aware.csv   4,064 rows
```

Both files have exactly these columns:

```text
ml_num,predicted_price
```

They were generated from:

```text
gbt_ensemble_experiments/artifacts/20261107-210017-299182/ensemble_checkpoint.pkl
```

## Verify the files

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -c "from check_submission import check_all; raise SystemExit(0 if check_all('submissions') else 1)"
```

Current result:

```text
submission_off_market.csv: OK
submission_list_aware.csv: OK
```

## Recreate predictions without retraining

```powershell
.\.venv\Scripts\python.exe gbt_ensemble_experiments\predict_artifact.py `
  gbt_ensemble_experiments\artifacts\20261107-210017-299182\ensemble_checkpoint.pkl `
  --output-dir submissions
```

This loads the saved models and does not refit them.

## Retrain only if necessary

```powershell
.\.venv\Scripts\python.exe gbt_ensemble_experiments\train_final.py
```

Retraining creates a new timestamped artifact. Run prediction and checking
again after retraining.

## Conservative validation reference

The documented walk-forward benchmark uses December 2025 through May 2026:

| Track | MdAPE | MAPE |
|---|---:|---:|
| Off-market | 7.27% | 9.81% |
| List-aware | 3.98% | 5.19% |

Each validation month used only sales before that month. The hidden submission
period is June through September 2026.

The report is `report/report.pdf`. Model instructions are in
`gbt_ensemble_experiments/FINAL_MODEL.md`.
