# Final submission handoff

| Purpose | Link |
|---|---|
| Understand the final model and its source code | [`gbt_ensemble_experiments/README.md`](gbt_ensemble_experiments/README.md) |
| Read the final four-page report | [`report/report.pdf`](report/report.pdf) |
| Inspect the off-market submission | [`submissions/submission_off_market.csv`](submissions/submission_off_market.csv) |
| Inspect the list-aware submission | [`submissions/submission_list_aware.csv`](submissions/submission_list_aware.csv) |

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
generated artifact. **A fresh clone therefore contains no fitted checkpoint.**
Running `python run.py predict` detects that absence, trains the selected final
model, saves a new local checkpoint, and writes both submission CSVs. If a
matching local checkpoint already exists, the same command reuses it instead.
The two submitted CSVs themselves are tracked in Git.

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

The validation command runs the final GBT ensemble and is **slow**. To export the
already-computed walk-forward predictions without refitting, use
`run.py validate --reuse-results`.

## Reproduce the submission from a fresh clone

Install the final-model dependencies, then generate and check the CSVs:

```powershell
.\.venv\Scripts\python.exe -m pip install -r gbt_ensemble_experiments\requirements.txt
.\.venv\Scripts\python.exe run.py predict
.\.venv\Scripts\python.exe run.py check
```

Because checkpoints are not committed, `run.py predict` trains the selected
model on its first run in a fresh clone. It loads an existing local checkpoint
only when its validation metadata, weights, and calibration match the selected
7.27% off-market / 3.98% list-aware MdAPE configuration. It then generates both
CSVs.

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

The model overview is in `gbt_ensemble_experiments/README.md`; this document is
the authoritative reproduction and submission-verification guide.
