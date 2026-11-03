# Diverse ensemble v10

Walk-forward validation: December 2025 through May 2026; 10,069 homes per track.

| Track | MdAPE | MAPE | Median signed error |
|---|---:|---:|---:|
| Off-market | 7.69% | 10.10% | +0.27% |
| List-aware | 3.92% | 4.96% | -0.31% |

Four members combined in log-price space: baseline 40%, shallow 20%, deep 20%, alternate grouping 20%. All use RMSE training, and both tracks include learned listing-text features. List-aware also includes asking-price-to-comparable-value features.

Source files were checked against the run snapshot matching validation_predictions.csv. This is a source-and-results checkpoint; trained model weights are not stored. Validation predictions include all four members.

To restore, copy source files to the project root. Run `python run.py validate` followed by `python run.py report`, with MODEL_PROFILE unset or set to ensemble. GPU training may produce small numerical differences between runs.
