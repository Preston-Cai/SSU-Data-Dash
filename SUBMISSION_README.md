# Final submission handoff

The final four-page report is [`report/report.pdf`](report/report.pdf). It links
to the presentation video and this GitHub repository.

The final submission files are ready in `submissions/`:

```text
submissions/submission_off_market.csv   3,974 rows
submissions/submission_list_aware.csv   4,064 rows
```

Both files have exactly these columns:

```text
ml_num,predicted_price
```

The current CSVs were most recently regenerated from the clean reproduced
checkpoint:

```text
gbt_ensemble_experiments/artifacts/20261108-123158-453037/ensemble_checkpoint.pkl
```

Its predictions are byte-for-byte identical to the originally selected
`20261107-201132-343475` checkpoint.

The fitted checkpoint is intentionally excluded from Git because it is a large
generated artifact. The two submission CSVs themselves are tracked. `run.py
predict` reuses the selected checkpoint when present and automatically trains
the configured final model when no matching checkpoint exists.

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

## Recreate validation outputs

```powershell
.\.venv\Scripts\python.exe run.py validate
.\.venv\Scripts\python.exe run.py report
```

The validation command runs the final GBT ensemble and is slow. To export the
already-computed walk-forward predictions without refitting, use
`run.py validate --reuse-results`.

## Reproduce predictions from a fresh clone

Install the final-model dependencies, then generate and check the CSVs:

```powershell
.\.venv\Scripts\python.exe -m pip install -r gbt_ensemble_experiments\requirements.txt
.\.venv\Scripts\python.exe run.py predict
.\.venv\Scripts\python.exe run.py check
```

`run.py predict` loads an existing checkpoint only if its validation metadata,
weights, and calibration match the selected 3.98% MdAPE / 5.19% MAPE
configuration. Otherwise, it trains that configuration, saves a timestamped
checkpoint, and then generates the CSVs.

## Regenerate CSVs from the existing checkpoint

```powershell
.\.venv\Scripts\python.exe run.py predict
```

When the selected checkpoint exists, this command skips training and loads it
directly. It never uses the retained starter models in `models.py`.

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
