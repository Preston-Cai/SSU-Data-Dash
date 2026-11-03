# Model checkpoint: spatial comparable sales

## Model

This checkpoint includes community and house-style encoding, lot dimensions, historical comparable price features, and recent spatial comparables. Spatial features use earlier sales from the same community and property type, restricted to the prior 365 days and weighted by geographic distance.

## Walk-forward validation

Validation covers December 2025 through May 2026.

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|---:|
| Off-market | 10,069 | **12.18%** | **9.25%** | -0.53% |
| List-aware | 10,069 | **5.47%** | **4.34%** | +0.42% |

This is the strongest validated model version so far.

