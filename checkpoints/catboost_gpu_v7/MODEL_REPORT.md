# CatBoost GPU checkpoint (v7)

Validated with walk-forward predictions from December 2025 through May 2026.
Every validation month was predicted using only sales before that month.

| Track | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|
| Off-market | 10.85% | 8.31% | +0.31% |
| List-aware | 5.08% | 3.98% | -0.27% |

## Model

- CatBoost on GPU (`task_type="GPU"`, CUDA device `0`)
- 1,200 trees, depth 8, learning rate 0.05, L2 regularization 5
- Native categorical handling for community, house style, and other categorical features
- Leakage-safe historical price-per-square-foot and spatial-comparable features
- Time-adjusted spatial-comparable feature used only for off-market predictions
- List-aware model predicts the log sold-price/list-price residual

This checkpoint preserves the best validated CatBoost model before testing a higher tree count.
