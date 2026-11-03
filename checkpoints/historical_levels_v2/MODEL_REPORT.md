# Model checkpoint: parallel historical comparable levels

## Model

This checkpoint includes:

- one-hot encoded `community` and `house_style`;
- `lot_front` and `lot_depth`;
- all-history and recent 365-day comparable price-per-square-foot features;
- separate historical estimates for community + property type, community only, property type only, and the global market;
- fallback estimates and prior-sale counts.

All historical features use only sales strictly before each home’s listing date.

## Walk-forward validation

Validation covers December 2025 through May 2026.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|---:|
| Off-market | 10,069 | **13.43%** | **10.66%** | -0.57% |
| List-aware | 10,069 | **5.55%** | **4.42%** | +0.43% |

The expanded historical levels improved off-market MdAPE from 10.70% to 10.66% and off-market MAPE from 13.45% to 13.43%.

## Reproduce

```powershell
python run.py validate
python run.py report
```

