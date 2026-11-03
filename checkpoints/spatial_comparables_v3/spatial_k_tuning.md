# Spatial comparable `k` tuning

The spatial feature uses the nearest recent comparable sales from the same community and property type. Each candidate was evaluated with the complete December 2025 through May 2026 walk-forward validation.

| k | Off-market MAPE | Off-market MdAPE | List-aware MAPE | List-aware MdAPE |
|---:|---:|---:|---:|---:|
| 5 | 12.32% | 9.37% | 5.50% | 4.37% |
| 10 | 12.22% | 9.30% | 5.47% | 4.34% |
| 20 | **12.18%** | **9.25%** | **5.47%** | **4.34%** |
| 40 | 12.17% | 9.27% | 5.48% | 4.36% |

`k=20` is retained because off-market MdAPE is the primary metric and off-market counts double. The live model default remains `SPATIAL_K = 20`.
