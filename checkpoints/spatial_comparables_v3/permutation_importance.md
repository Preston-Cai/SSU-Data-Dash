# Grouped permutation importance

Future holdout month: 2026-05
Each feature group was shuffled 5 times after training on earlier sales.
Importance is the increase in MdAPE, in percentage points; larger positive values mean the model relied more on that group.

## off_market

Baseline MdAPE: **9.46%**

| Feature group | MdAPE after shuffle | Importance (pp) | Std. dev. (pp) |
|---|---:|---:|---:|
| square_feet | 25.09% | +15.63 | 0.34 |
| lot_front | 16.41% | +6.95 | 0.28 |
| spatial_comparables | 16.21% | +6.75 | 0.30 |
| washrooms | 13.47% | +4.00 | 0.21 |
| historical_comparables | 11.70% | +2.23 | 0.24 |
| parking | 11.26% | +1.80 | 0.07 |
| community | 9.95% | +0.49 | 0.11 |
| recent_log_price_per_sqft_community | 9.93% | +0.47 | 0.09 |
| prior_sales_count_property_type | 9.75% | +0.29 | 0.04 |
| lot_depth | 9.75% | +0.28 | 0.09 |
| recent_sales_count_property_type | 9.55% | +0.09 | 0.05 |
| prior_log_price_per_sqft_community | 9.53% | +0.07 | 0.05 |
| prior_log_price_per_sqft_community_type | 9.48% | +0.02 | 0.03 |
| recent_log_price_per_sqft_community_type | 9.47% | +0.01 | 0.03 |
| prior_sales_count_community_type | 9.46% | +0.00 | 0.01 |
| prior_log_price_per_sqft_global | 9.46% | +0.00 | 0.00 |
| prior_sales_count_global | 9.46% | +0.00 | 0.00 |
| house_style | 9.46% | +0.00 | 0.00 |
| recent_log_price_per_sqft_property_type | 9.46% | -0.00 | 0.01 |
| months | 9.46% | -0.00 | 0.02 |
| recent_sales_count_global | 9.46% | -0.01 | 0.02 |
| prior_sales_count_community | 9.46% | -0.01 | 0.01 |
| prior_log_price_per_sqft_property_type | 9.45% | -0.01 | 0.02 |
| recent_log_price_per_sqft_global | 9.45% | -0.01 | 0.02 |
| latitude | 9.44% | -0.02 | 0.03 |
| longitude | 9.44% | -0.03 | 0.05 |
| bedrooms | 9.43% | -0.03 | 0.03 |
| property_type | 9.43% | -0.03 | 0.03 |
| recent_sales_count_community | 9.41% | -0.05 | 0.01 |
| recent_sales_count_community_type | 9.40% | -0.06 | 0.04 |

## list_aware

Baseline MdAPE: **4.50%**

| Feature group | MdAPE after shuffle | Importance (pp) | Std. dev. (pp) |
|---|---:|---:|---:|
| list_price | 40.13% | +35.62 | 0.35 |
| square_feet | 5.95% | +1.45 | 0.02 |
| spatial_comparables | 5.50% | +1.00 | 0.05 |
| lot_front | 5.02% | +0.52 | 0.06 |
| washrooms | 4.91% | +0.41 | 0.06 |
| prior_sales_count_property_type | 4.82% | +0.32 | 0.10 |
| lot_depth | 4.74% | +0.24 | 0.03 |
| community | 4.72% | +0.22 | 0.03 |
| parking | 4.67% | +0.17 | 0.00 |
| historical_comparables | 4.63% | +0.13 | 0.05 |
| property_type | 4.57% | +0.07 | 0.01 |
| prior_log_price_per_sqft_community | 4.56% | +0.06 | 0.02 |
| prior_log_price_per_sqft_community_type | 4.55% | +0.05 | 0.03 |
| recent_log_price_per_sqft_property_type | 4.55% | +0.05 | 0.02 |
| recent_log_price_per_sqft_community | 4.55% | +0.04 | 0.04 |
| latitude | 4.54% | +0.03 | 0.00 |
| recent_sales_count_property_type | 4.53% | +0.03 | 0.01 |
| recent_sales_count_community_type | 4.52% | +0.02 | 0.01 |
| months | 4.52% | +0.02 | 0.01 |
| recent_sales_count_global | 4.52% | +0.02 | 0.00 |
| longitude | 4.52% | +0.02 | 0.02 |
| recent_sales_count_community | 4.51% | +0.01 | 0.02 |
| recent_log_price_per_sqft_community_type | 4.51% | +0.01 | 0.00 |
| recent_log_price_per_sqft_global | 4.51% | +0.01 | 0.01 |
| prior_log_price_per_sqft_global | 4.51% | +0.01 | 0.01 |
| bedrooms | 4.50% | +0.00 | 0.00 |
| prior_sales_count_community | 4.50% | +0.00 | 0.00 |
| prior_sales_count_global | 4.50% | +0.00 | 0.00 |
| house_style | 4.50% | +0.00 | 0.00 |
| prior_sales_count_community_type | 4.50% | -0.00 | 0.01 |
| prior_log_price_per_sqft_property_type | 4.49% | -0.01 | 0.01 |

