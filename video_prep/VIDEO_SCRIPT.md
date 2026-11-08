# Complete five-minute video script

Each bullet is one natural spoken beat. Pause briefly between sections and let
the visual do some of the work; do not read the headings aloud.

## Opening — 0:00–0:15

- “We built a home-price estimator for two different situations.”
- “In the off-market track, we estimate value without seeing the asking price.”
- “In the list-aware track, the asking price is available.”
- “I’ll show what we built, why we chose it, how we tested it on future sales,
  and where it still makes mistakes.”

## What we built — 0:15–1:25

- “We started with facts about the home: its size, lot, bedrooms, bathrooms,
  parking, style, and property type.”
- “We turned some of those into more useful relationships, like area per
  bedroom and bathrooms per bedroom.”
- “We also used municipality, community, latitude and longitude, and rounded
  map cells, because nearby homes often share a price level.”
- “Calendar features let the model represent seasonality and changes in the
  market over time.”
- “Listing remarks provide another kind of information, such as renovation,
  condition, basement, transit, or luxury language.”
- “Our most useful feature idea was local sale history.”
- “For each home, we summarize what similar homes in its community, property
  group, and geographic area had already sold for.”
- “One version estimates the stable local price level, and another gives more
  weight to recent sales.”
- “For the list-aware track, we also learn how similar homes sold relative to
  their asking prices.”
- “Three models turn those features into a prediction.”
- “LightGBM learns nonlinear patterns from the home and local-history features.”
- “XGBoost gives us a second tree-based opinion for off-market homes.”
- “A TF–IDF Ridge model reads the listing remarks and makes different kinds of
  errors from the trees.”
- “We combine their prices with a weighted geometric average and a small
  calibration.”
- “The off-market blend is thirty-two percent LightGBM, nineteen percent text,
  and forty-nine percent XGBoost.”
- “The list-aware blend is six percent ordinary-history LightGBM, twenty-five
  percent recency-aware LightGBM, thirteen percent independent off-market value,
  and fifty-six percent historical community sold-to-list median.”
- “That specialized anchor helps us decide when an asking price disagrees with
  the home's underlying value or with normal sale ratios in its community.”

## Why — 1:25–2:40

- “Off-market models predict log sale price, because our metric cares about
  percentage error rather than just error in dollars.”
- “The list-aware models predict the percentage adjustment from asking price
  to sale price, which is a more stable problem than rebuilding the full price
  from scratch.”
- “We kept local history because location labels alone do not tell us what
  comparable homes were actually selling for.”
- “We kept recency weighting because an old sale is usually less relevant than
  a recent one.”
- “We kept the text model even though it was not always strongest by itself,
  because an ensemble improves when its members make different mistakes.”
- “We also tested approaches that did not earn a place in the final model.”
- “Our custom PyTorch neural network reached an eleven point one nine percent
  off-market MdAPE and a five point six three percent list-aware MdAPE.”
- “Its features were much thinner: basic size, rooms, parking and location,
  plus property and style indicators, municipality and community embeddings,
  lot dimensions, and list price only for the list-aware model.”
- “It did not have listing text, the richer ratios and interactions, or the
  historical local-price features that drove the tree models.”
- “So it had to infer market value mostly from raw coordinates and category
  identities.”
- “A dense neural network also has a less helpful built-in structure than
  boosted trees for this mixture of numeric and categorical table data.”
- “That explains why this particular MLP was weaker; it does not mean every
  possible neural network would fail.”
- “CatBoost was slow and weak off-market.”
- “A direct comparable-sales model sounded intuitive, but it was less accurate
  than giving comparable-history features to boosted trees.”
- “A local-anchor residual model, character-level text, longer text windows,
  sample weighting, and stacking also failed to improve prevalidation enough
  to justify their complexity.”
- “The main gain came from better, properly timed information, not from picking
  the fanciest model name.”

## How we validated it — 2:40–3:35

- “We used walk-forward validation because a random split would let the model
  learn from the future.”
- “To predict December 2025, we trained only on sales before December.”
- “Then we moved the cutoff forward, retrained everything, and predicted each
  month through May 2026.”
- “Every model and every neighbourhood table was rebuilt at each cutoff.”
- “Validation sale prices never entered the features.”
- “We enforced the same timing rule inside the training data.”
- “Every training home’s local-price features use sales dated strictly earlier
  than that home’s sale.”
- “Homes sold on the same day are processed together, so they cannot reveal
  their answers to one another.”
- “We selected blend weights and calibration on September through November,
  before the six reported validation months.”
- “Across ten thousand and sixty-nine later sales, off-market MdAPE was seven
  point two seven percent.”
- “List-aware MdAPE was three point nine eight percent.”
- “That means half of the predictions missed by less than those percentages.”
- “The corresponding mean errors were nine point eight one percent off-market
  and five point one nine percent list-aware.”

## Where it fails — 3:35–4:35

- “The error chart shows that the most expensive ten percent of homes were the
  hardest price group.”
- “Their median error was thirteen point four two percent off-market and five
  point three two percent list-aware.”
- “Among property types with at least one hundred validation homes,
  semi-detached homes were hardest.”
- “Their median errors were nine point five six percent off-market and five
  point eight four percent list-aware.”
- “The amount of local history also matters.”
- “For communities with two hundred to nine hundred and ninety-nine earlier
  sales, off-market MdAPE was seven point seven four percent.”
- “For communities with at least one thousand earlier sales, it fell to six
  point nine six percent.”
- “The list-aware comparison was four point two zero versus three point eight
  one percent.”
- “There were only eight homes in the even sparser fifty-to-one-hundred-and-
  ninety-nine group, so we do not make a broad claim from that tiny sample.”
- “May 2026 was the weakest month for both tracks, at seven point six three
  percent off-market and four point two seven percent list-aware.”
- “Our interpretation is that unusual expensive homes and rare local groups
  have fewer truly comparable earlier sales.”
- “May’s result also suggests market drift that recency weighting cannot fully
  remove.”
- “These are explanations for measured patterns, not proof of their causes.”

## Close — 4:35–5:00

- “June through September is the truly unseen hidden test, so our validation is
  evidence, not a guarantee.”
- “Rare groups remain uncertain, and this ensemble is more expensive to train
  than one simple model.”
- “Our next step would be segment-level uncertainty intervals and nested time
  splits for any further tuning.”
- “The main lesson was to combine different models with useful local history,
  while enforcing what was actually knowable on every sale date.”
