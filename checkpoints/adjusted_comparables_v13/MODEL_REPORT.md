# Adjusted comparables v13 — retained successful experiment

Exact source snapshot and predictions from the completed comparables-profile validation. No model training or validation was rerun when saving this checkpoint.

| Track | Homes | MdAPE | MAPE | Median signed error |
|---|---:|---:|---:|---:|
| list_aware | 10,069 | 3.854% | 4.920% | -0.336% |
| off_market | 10,069 | 7.385% | 9.885% | -0.018% |

The retained model blends the uncorrected ensemble with learned comparable-sale adjustments: 30% off-market and 20% list-aware when comparable evidence passes the fixed checks. The saved CSV includes the baseline and standalone comparable predictions for paired analysis.

Restore the Python and requirements files to the project root. In PowerShell select `$env:MODEL_PROFILE = "comparables"`, then use `python run.py validate` and `python run.py report`. Run from the project root so dataset and cache locations remain correct.

This is a source-and-results checkpoint, not a trained-weight archive. Reusable caches remain in outputs/cache and were not duplicated. Retraining on GPU may produce small numerical differences. The working project remains unchanged.