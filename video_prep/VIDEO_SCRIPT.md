Opening — 0:00–0:20

Preston: We built a home-price estimator for two situations: off-market, where we cannot see the asking price, and list-aware, where we can.

Lorance: We explored different approaches, including fifteen versions of a CatBoost workflow, before choosing our final ensemble.

## Final model and features — 0:20–1:25

Preston: Our final model combines home characteristics, location, listing remarks, and local sale history.

Preston: We use size, lot, rooms, parking, property type, and ratios such as bathrooms per bedroom.

Preston: Community, coordinates, map cells, and calendar features capture local differences and market changes.

Preston: Our key feature summarizes what similar nearby homes had already sold for, with a second version emphasizing recent sales.

Preston: LightGBM learns patterns in those features, XGBoost supplies another off-market prediction, and a TF–IDF Ridge model learns from listing remarks.

Preston: We combine predictions geometrically, then apply a small calibration.

Preston: Off-market weights are thirty-two percent LightGBM, nineteen percent text, and forty-nine percent XGBoost.

Preston: The list-aware blend uses six percent ordinary-history LightGBM, twenty-five percent recency-aware LightGBM, thirteen percent independent off-market value, and fifty-six percent historical community sold-to-list median.

Preston: Off-market models predict log price; list-aware learners model the adjustment from asking price.

## Experiments and the CatBoost decision — 1:25–2:50

Lorance: We tried fifteen versions of the CatBoost approach before giving up on it for this submission because the training workflow became too heavy.

Lorance: It grew into an ensemble with historical price features and text corrections.

Lorance: By version eleven, we were testing corrections learned from earlier prediction errors; version twelve added frozen semantic representations of listing remarks.

Lorance: That semantic experiment did not improve completed off-market validation.

Lorance: Version thirteen learned adjustments between each home and up to eight earlier comparable sales.

Lorance: Version fourteen separated houses and condos into specialists and learned when to trust their predictions.

Lorance: Version fifteen added local error correction using recent errors from similar homes, with limits on correction size and fallback predictions.

Lorance: The cost came from repeatedly training the underlying ensembles across historical monthly cutoffs, not just fitting one correction model.

Lorance: Even after caching improvements, the documented cold version-twelve validation required one hundred fourteen CatBoost fits.

Lorance: We stopped pursuing that complexity; version fifteen has no confirmed accuracy improvement in the experiment notes.

Preston: We also tested a PyTorch neural network: its median percentage errors were eleven point one nine off-market and five point six three list-aware.

Preston: It lacked text and richer historical features, so this comparison reflects both the model and its available information.

## Validation and results — 2:50–4:00

Preston: We tested future sales with walk-forward validation, training before December twenty twenty-five, predicting December, then moving forward monthly through May twenty twenty-six.

Preston: Every model and neighbourhood table was rebuilt at each cutoff.

Preston: Training features used only strictly earlier sales, and same-day homes could not reveal their prices to each other.

Preston: Blend weights and calibration were selected on September through November, before the six reported months.

Preston: Across ten thousand sixty-nine validation sales, median absolute percentage error was seven point two seven percent off-market and three point nine eight percent list-aware.

Lorance: That means half the predictions missed by less than those percentages.

Preston: Mean absolute percentage errors were nine point eight one and five point one nine percent, respectively.

## Limitations and close — 4:00–4:50

Preston: The most expensive ten percent of homes were hardest: median errors reached thirteen point four two percent off-market and five point three two percent list-aware.

Preston: Communities with more historical sales performed better, while May was the weakest month for both tracks.

Lorance: Limited comparables and market drift are plausible explanations, not proven causes.

Lorance: June through September remains the unseen test, so validation is evidence, not a guarantee.

Preston: Next, we would add uncertainty intervals and use nested time splits for further tuning.

Lorance: Our lesson: useful historical information, honest timing, and manageable training matter more than adding complexity.