# Model checkpoint: structured listing text

## Model

This checkpoint combines the spatial-comparable model with structured signals derived from listing remarks: word count, renovation/upgrade language, needs-work/as-is language, outdoor-space language, and transit language.

## Walk-forward validation

| Track | Homes | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|---:|
| Off-market | 10,069 | **11.63%** | **8.88%** | -0.14% |
| List-aware | 10,069 | **5.36%** | **4.23%** | -0.15% |

The structured-text signals improved off-market MdAPE from 9.21% to 8.88%.
