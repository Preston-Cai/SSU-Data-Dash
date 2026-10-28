# Final selected model: quick instructions

Use this file when you only need to train, load, or run the selected model.
The longer experimental history is in `docs/EXPERIMENT_REPORT.md`.

## Selected model

The final model is a separate ensemble for each competition track:

- **Off-market:** 15% recency-aware LightGBM, 20% word TF-IDF Ridge, and 65%
  XGBoost, followed by calibration `0.9772745745864185`.
- **List-aware:** 35% target-encoded LightGBM, 35% recency-aware LightGBM, and
  30% word TF-IDF Ridge, followed by calibration `0.9865499704372777`.

The percentages are geometric-blend weights. Do not substitute one of the
individual component files for the full ensemble.

Validated performance over December 2025-May 2026:

| Track | MAPE | MdAPE |
|---|---:|---:|
| off-market | 9.85% | 7.28% |
| list-aware | 5.25% | 4.08% |

## Current saved model

The complete fitted model is:

```text
codex_attempts/artifacts/20261028-000439-310846/ensemble_checkpoint.pkl
```

This checkpoint contains the fitted feature builders, historical encoders,
component models, blend weights, and calibration. The native `.txt`, `.json`,
and `.joblib` files beside it are inspection copies; use
`ensemble_checkpoint.pkl` for prediction.

The two predictions produced by that checkpoint are in the same directory:

```text
submission_off_market.csv
submission_list_aware.csv
```

## Retrain the final model

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe "codex_attempts\train_final.py"
```

This trains the selected configuration once on all permitted sales through May
2026. It creates a new save-time-stamped directory under:

```text
codex_attempts/artifacts/
```

The new directory contains `ensemble_checkpoint.pkl`, native component models,
the manifest, and both submission CSV files. Retraining does not overwrite the
existing artifact.

If the isolated dependencies are missing, install them first:

```powershell
.\.venv\Scripts\python.exe -m pip install -r "codex_attempts\requirements-attempts.txt"
```

## Predict without retraining

Pass the exact checkpoint file, not its directory:

```powershell
.\.venv\Scripts\python.exe "codex_attempts\predict_artifact.py" `
  "codex_attempts\artifacts\20261028-000439-310846\ensemble_checkpoint.pkl"
```

By default this writes both CSV files into a `reloaded_predictions` folder next
to the checkpoint. To choose another destination:

```powershell
.\.venv\Scripts\python.exe "codex_attempts\predict_artifact.py" `
  "codex_attempts\artifacts\20261028-000439-310846\ensemble_checkpoint.pkl" `
  --output-dir "codex_attempts\my_predictions"
```

## Check the saved submissions

```powershell
.\.venv\Scripts\python.exe -c "from check_submission import check_all; raise SystemExit(0 if check_all(r'codex_attempts\artifacts\20261028-000439-310846') else 1)"
```

Both submissions in the current artifact have already passed this check, and a
checkpoint reload reproduced them byte-for-byte.

## Validate and use the starter report

Recompute the complete December 2025-May 2026 walk-forward validation. This is
slow because each component is refitted before every validation month:

```powershell
.\.venv\Scripts\python.exe "codex_attempts\validate.py"
.\.venv\Scripts\python.exe run.py report
```

To use the stored final validation predictions without refitting anything:

```powershell
.\.venv\Scripts\python.exe "codex_attempts\validate.py" --reuse-results
.\.venv\Scripts\python.exe run.py report
```

Both paths write the selected ensemble's predictions to
`outputs/validation_predictions.csv`, which is the exact location and schema
read by the provided `run.py report` function.

## Important rules

- Use `train_final.py` to retrain the selected model.
- Use `predict_artifact.py` to load it without retraining.
- Do not use list price in the off-market track.
- Do not add the external AVM or after-sale fields.
- Do not use the experimental screening scripts as the final model entry point.
