# Model checkpoint: track-specific spatial comparables

## Model

This checkpoint uses separate feature sets for the two tasks:

- **Off-market:** community, property characteristics, structured listing-text features, historical comparable-sale features, geographic spatial comparables, and a size-matched spatial comparable feature.
- **List-aware:** the same core model plus list-price residual prediction, but without the size-matched spatial comparable feature because it slightly worsened that track.

Both tracks use their own small multiplicative calibration factor selected from walk-forward validation.

## Walk-forward validation

Validation covers December 2025 through May 2026, predicting each month with only earlier sales.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|---:|
| Off-market | 10,069 | **11.58%** | **8.81%** | -0.11% |
| List-aware | 10,069 | **5.23%** | **4.09%** | -0.17% |

## Why features differ by track

The size-matched spatial comparable feature improved off-market MdAPE from 8.88% to 8.81%, but increased list-aware MdAPE from 4.09% to 4.11%. It is therefore retained only for off-market predictions.

## Reproduce

```powershell
python run.py validate
python run.py report
```
