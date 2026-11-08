# Five-minute slide script

Target runtime: 4 minutes 45 seconds to 4 minutes 55 seconds.

## Slide 1 — Predicting Toronto home sale prices (0:00–0:20)

We built a home price estimator for two situations. In the off-market track,
we value a home without seeing the asking price. In the list-aware track, the
asking price is available. Our goal was to improve accuracy without using any
information that would not have existed on the prediction date.

## Slide 2 — Two valuation questions (0:20–0:55)

These tracks answer different questions. Off-market valuation is useful before
a home is listed, so the model can use the property, location, date, and
remarks, but never list price. List-aware valuation estimates the closing price
after listing and can use the seller's asking price. That makes it easier, but
it also means the two validation scores cannot be treated as the same task.

## Slide 3 — The final ensemble (0:55–1:50)

The largest improvement came from teaching the model about earlier local
sales. We created historical price and price-per-square-foot summaries for
communities, property types, styles, and geographic cells. A second version
gave more weight to recent transactions so the model could react to market
drift. We combined those signals with physical home features, location,
calendar variables, and listing language. Off-market uses XGBoost, a
recency-aware LightGBM model, and a TF-IDF Ridge text model. List-aware relies
mainly on historical sold-to-list ratios for the community and property type,
supported by two LightGBM components and an independent off-market value
estimate. Each track uses a geometric blend because the models predict in
log-price space and make different errors.

## Slide 4 — Model selection through experiments (1:50–2:40)

We did not choose the most complicated model. Preston tested a dense PyTorch
MLP with standardized numeric inputs and learned location embeddings. It
reached 11.19 percent off-market and 5.63 percent list-aware MdAPE. Its feature
set did not include remarks or the richer historical local-price signals that
helped the tree ensemble. Jiang developed a CatBoost ensemble with a residual
text correction. It reported a slightly better list-aware development score,
but off-market was 7.67 percent and a cold six-month validation required 114
CatBoost fits. We selected the GBT ensemble as the practical model to present
and reproduce.

## Slide 5 — Walk-forward validation (2:40–3:25)

We used walk-forward validation instead of a random split. For each month from
December 2025 through May 2026, every model trained only on sales from before
that month and then predicted that month. Historical encodings followed the
same rule within training, and homes sold on the same date could not reveal
their targets to one another. We chose blend weights and calibration using
September through November 2025. We did inspect model variants during
development, so the hidden June through September test remains the truly unseen
evaluation.

## Slide 6 — Validation results (3:25–4:10)

Across 10,069 validation homes per track, the final off-market model reached
7.27 percent MdAPE and 9.81 percent MAPE. The list-aware model reached 3.98
percent MdAPE and 5.19 percent MAPE. For context, the starter methods scored
20.8 percent off-market and 7.3 percent list-aware MdAPE. The improvement is
largest off-market, which is also the harder track and counts twice in the
competition.

## Slide 7 — Where the model struggles (4:10–4:55)

The errors are not evenly distributed. The highest price decile has 13.42
percent off-market MdAPE, almost twice the overall result. Semi-detached homes
are also harder. Communities with fewer earlier sales perform worse than those
with at least one thousand, which is consistent with weaker local evidence.
May is worse than the earlier validation months, suggesting sensitivity to
market drift. These patterns are associations, not causal claims. A responsible
use of this model would flag expensive, rare, or low-history homes and attach a
wider uncertainty range instead of presenting every estimate with the same
confidence.
