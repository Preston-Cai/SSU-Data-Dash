# Corrected walk-forward failure analysis

Source: `gbt_ensemble_experiments/results/final_walk_forward_predictions.csv`
after the past-only encoder fix. Values below are MdAPE.

| Segment | Off-market | List-aware | Validation homes per track |
|---|---:|---:|---:|
| Highest sold-price decile | 13.42% | 5.41% | 1,007 |
| Semi-detached | 9.56% | 5.86% | 1,034 |
| Communities with 200--999 earlier sales | 7.74% | 4.27% | 4,364 |
| Communities with at least 1,000 earlier sales | 6.96% | 3.92% | 5,697 |
| May 2026 | 7.63% | 4.32% | 2,291 |

The 50--199 earlier-sales community band had only eight validation homes. Its
error should not be presented as a stable general finding.

Interpretation: expensive and semi-detached homes appear to have more difficult
price variation, while communities with less prior history provide weaker local
signals. May's higher errors are consistent with market drift. These are
plausible interpretations, not causal conclusions.
