# Five-minute video script

## Opening — 0:00–0:20

- “We built a home-price estimator for two situations.”
- “Off-market means estimating without an asking price, while list-aware means the asking price is available.”
- “I’ll explain what we built, why we made the main choices, how we tested it, and where it still fails.”

## What we built — 0:20–1:30

- “The model uses area, lot dimensions, rooms, parking, property type, style, useful ratios, municipality, community, map cells, and calendar features.”
- “A TF–IDF text model reads the listing remarks, including clues about renovation, condition, views, and transit.”
- “Our most important features summarize earlier sales from similar homes nearby.”
- “One version emphasizes recent sales.”
- “For list-aware homes, we also calculate historical sold-to-list ratios.”
- “The final prediction is an ensemble: LightGBM learns the structured patterns, XGBoost supplies another off-market estimate, and Ridge contributes a text-based view.”
- “We combine their prices with a weighted geometric average and a small calibration.”
- “Off-market uses thirty-two percent LightGBM, nineteen percent text, and forty-nine percent XGBoost.”
- “List-aware combines ordinary and recent-history LightGBM, an independent off-market value estimate, and a community-plus-property-type sold-to-list anchor.”
- “Off-market predicts log price; list-aware predicts the log adjustment from asking price to sale price.”

## Why — 1:30–2:25

- “Percentage error is our metric, so log targets are a natural choice.”
- “Asking price is already a useful starting point, so list-aware predicts how sale price usually differs from it.”
- “Local history matters because a location label alone does not tell us what similar homes actually sold for.”
- “Recency weighting helps when the market changes.”
- “We kept the text model because an ensemble improves when its members make different mistakes.”
- “We tried several alternatives that did not earn a place in the final workflow.”
- “A PyTorch MLP reached eleven point one nine percent off-market MdAPE and five point six three percent list-aware MdAPE.”
- “Its inputs were thinner: basic numeric fields, category embeddings, and list price, but not the remarks or strict historical local-price features that helped the trees.”
- “We also tried CatBoost-heavy workflows, direct comparables, text variations, residual corrections, weighting, and stacking.”
- “They performed worse in prevalidation or added too much training time for an unreliable gain.”
- “The main lesson was that better-timed information mattered more than using the fanciest model.”

## How we validated it — 2:25–3:35

- “We used walk-forward validation because a random split would let the model learn from the future.”
- “To predict December twenty twenty-five, we trained only on sales before December.”
- “Then we moved the cutoff forward one month at a time through May twenty twenty-six.”
- “At every cutoff, we rebuilt the models and local-history tables.”
- “Validation sale prices never entered the features.”
- “The timing rule also applies inside training: each training home uses only sales with an earlier sale date.”
- “Homes sold on the same day are processed as a batch, so they cannot reveal their prices to one another.”
- “Across ten thousand and sixty-nine validation homes, off-market MdAPE was seven point two seven percent.”
- “The conservative list-aware validation result was three point nine eight percent.”
- “That means half of predictions missed by less than those percentages.”
- “The corresponding MAPE values were nine point eight one percent and five point one nine percent.”
- “These are the conservative scores we use when describing validation performance.”

## Where it fails — 3:35–4:35

- “The most expensive ten percent of homes were hardest, with median errors of about thirteen point four percent off-market and five point three percent list-aware.”
- “Among property types with enough examples, semi-detached homes were hardest, at about nine point six and five point eight percent.”
- “Homes in communities with fewer comparable sales also performed worse than homes in high-history communities.”
- “Sparse-history groups had errors around seven point seven percent off-market and four point two percent list-aware, higher than the most common communities.”
- “May was the weakest validation month for both tracks, at about seven point six percent off-market and four point three percent list-aware.”
- “These patterns suggest that unusual homes have fewer truly similar examples and rare groups push the model toward a broad average.”
- “Off-market is harder because the asking-price signal is missing, and market drift can make older history less representative.”
- “Those are interpretations of the observed errors, not claims that one validation run proves the cause.”

## Close — 4:35–5:00

- “June through September is still the hidden test, so validation is evidence rather than a guarantee.”
- “The next improvement I would make is segment-level uncertainty and nested time splits for future tuning.”
- “The main takeaway is to combine different models, use local history, and enforce what was actually knowable on each sale date.”
