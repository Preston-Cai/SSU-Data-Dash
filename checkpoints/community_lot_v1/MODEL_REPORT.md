# Model checkpoint: community + lot dimensions

## Model

This checkpoint uses the gradient-boosted price model with:

- property size, bedrooms, washrooms, parking, latitude and longitude;
- listing month and property type indicators;
- one-hot encoded `community`;
- numeric `lot_front` and `lot_depth` features.

`lot_area` is intentionally not included. The model keeps the two dimensions separately so it can learn that lot width and depth may have different effects within a community.

## Validation

Walk-forward validation covers December 2025 through May 2026. Each month was predicted using only sales from earlier dates.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|---:|
| Off-market | 10,069 | 14.89% | **11.70%** | +0.53% |
| List-aware | 10,069 | 5.62% | **4.48%** | +0.62% |

The off-market MdAPE improved from 12.00% with community alone to 11.70% after adding the two lot dimensions.

## Interpretation

The improvement suggests that lot dimensions provide useful information after accounting for community. Keeping width and depth separately also allows the tree model to learn interactions between lot shape and community rather than forcing every lot into one area measurement.

## Reproduce

From the project root:

```powershell
python run.py validate
python run.py report
```

The copied validation predictions and report artifacts are included in this checkpoint folder.
