# Local residual v15 — retained successful version

Exact source snapshot and predictions from the completed local_residual validation. No training or validation was rerun when creating this checkpoint.

| Track | Homes | MdAPE | MAPE | Median signed error |
|---|---:|---:|---:|---:|
| list_aware | 10,069 | 3.843% | 4.903% | -0.306% |
| off_market | 10,069 | 7.280% | 9.740% | -0.101% |

Off-market retains the v14 adaptive comparable ensemble with the v15 local OOF-error correction blend. List-aware uses the v14 fixed specialist blend, with 20% specialist weight and its original eligibility/fallback rules.

Original ensemble calibration multipliers remain 1.005 off-market and 0.994 list-aware. They apply in validation and prediction; no removal or recalibration experiment has been applied.

Restore the saved Python and requirements files to the project root. In PowerShell set `$env:MODEL_PROFILE = "local_residual"`, then run `python run.py validate` and `python run.py report`. Run from the project root to use the existing dataset and prediction caches.

This checkpoint saves source and results, not trained model weights. Large reusable caches and downloaded encoder files remain in their existing locations. GPU retraining may vary slightly. Working source is unchanged.

## Next-session note

The discussed next candidate is adaptive calibration of the final blended prediction, learned from strictly earlier out-of-fold errors and evaluated alongside this baseline. No implementation, validation, scheduled run, or multiplier change was started for that idea. Preserve this checkpoint before testing it.