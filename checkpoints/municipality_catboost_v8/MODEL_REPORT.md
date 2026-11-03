# Municipality + CatBoost checkpoint (v8)

Validated with walk-forward predictions from December 2025 through May 2026.
Each month was valued using only sales from before that month.

| Track | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|
| Off-market | 10.74% | 8.14% | +0.31% |
| List-aware | 5.08% | 3.98% | -0.28% |

## Selected model

- GPU CatBoost with native categorical features and leakage-safe historical and spatial-comparable features.
- Off-market: 2,400 trees; includes municipality and the time-adjusted spatial-comparable feature.
- List-aware: 1,000 trees; includes municipality, predicts the log sold-price/list-price residual, and excludes the time-adjusted spatial feature.
- `garage` remains excluded because an earlier validation found it unhelpful.

## Next session

Start with an error analysis: identify the largest absolute percentage errors by property type, community, price band, and recurring listing-text patterns. Use those findings to propose narrowly targeted features or adjustments, then validate each independently.
