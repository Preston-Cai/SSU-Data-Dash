# Validation report

Walk-forward validation, 2025-12 to 2026-05: each month was predicted by models trained only on the sales before it.
Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---|---|---|---|
| off_market | 10,069 | 10.10% (9.91 to 10.28) | 7.72% (7.52 to 7.90) | +0.28% |
| list_aware | 10,069 | 4.97% (4.89 to 5.05) | 3.91% (3.82 to 4.01) | -0.31% |

MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.
A median signed error above 0 means the model tends to predict too high.
The intervals treat homes as independent. Homes sold in the same month share the same
market swings, so the real uncertainty is somewhat wider.

![Error by price range](figures/error_by_price_range.png)
