# Five-minute project walkthrough

This is a demonstration of understanding, not a script for reading the report.
Speak conversationally, show the actual artifacts named below, and use the four
questions as visible chapter titles. Replace `[NEW]` only from the corrected
validation output before recording.

## 0:00–0:15 — Opening

**On screen:** One example home, then the two prediction tracks.

“We built a home-price estimator for two situations. Off-market means estimating
without an asking price. List-aware means the asking price is available. I’ll
show what we built, why we made those choices, how we tested it on the future,
and where it still gets things wrong.”

Do not show the report title page or begin with metric definitions.

## 0:15–1:25 — What we built

**Demonstrate:** Start with a home card. Add feature groups one at a time as you
explain them.

Explain in your own words:

1. **The home itself:** size, lot, bedrooms, bathrooms, parking, style, and
   property type. We also create useful ratios such as area per bedroom.
2. **Where and when:** municipality, community, rounded map cells, and calendar
   variables that let the model represent seasonality and market movement.
3. **What the listing says:** a text model reads remarks; structured flags also
   capture ideas such as renovation, condition, basement, transit, and luxury.
4. **What had already sold nearby:** for community/type/style and geographic
   groups, summarize earlier log prices and price per square foot. A second
   version gives recent sales more influence. For list-aware homes, include
   earlier sold-to-list ratios.

Then draw three model boxes feeding one prediction:

- LightGBM learns nonlinear combinations of the structured and local-history
  features.
- XGBoost supplies another tree-based opinion for off-market homes.
- TF–IDF Ridge reads listing language and makes different mistakes.
- A weighted geometric average combines them, followed by a small calibration.

Mention the actual weights briefly: off-market is 32% LightGBM, 19% text, and
49% XGBoost. List-aware is 6% ordinary-history LightGBM, 25% recency-aware
LightGBM, 13% independent off-market value anchor, and 56% historical community
median sold-to-list anchor.

## 1:25–2:35 — Why we built it this way

**Demonstrate:** Show a compact experiment funnel: idea → measured result →
keep/reject. Avoid reading a long table.

Main reasoning:

- Predict log price off-market because percentage errors matter more than raw
  dollar errors. List-aware predicts the log sold-to-list adjustment because
  asking price is already a strong anchor.
- Local history was the strongest idea: location labels alone do not say what
  similar homes were actually selling for.
- Recency weighting addresses market drift.
- Text stays in the blend even when weaker alone because diversity can improve
  an ensemble.

Approaches tried but not promoted:

- A custom PyTorch MLP reached 11.19% off-market MdAPE and 5.63% list-aware
  MdAPE, worse than the corrected ensemble's 7.27% and 3.98%. Its inputs were
  comparatively thin: basic size/room/parking/location fields, one-hot property
  type and style, municipality/community embeddings, lot dimensions, and list
  price for the list-aware track. It lacked remarks, engineered ratios and
  interactions, and—most importantly—strictly historical local-price features.
  The network therefore had to infer market level from category IDs and raw
  coordinates. A dense MLP also supplies a weaker inductive bias than boosted
  trees for this mixed tabular data. This is an evidence-based explanation, not
  proof that all neural networks would fail.
- CatBoost was slow and weak off-market.
- A direct comparable-sales estimator was intuitive but less accurate than
  letting boosted trees learn from comparable-history features.
- A local-anchor residual model did not beat the simpler log-price target.
- Character TF–IDF, longer text windows, sample weighting, alternative
  list-price adjustments, and stacking added complexity without reliable
  prevalidation improvement.

Use one sentence to make the lesson clear: “The gain came mainly from better,
properly timed information—not from choosing the fanciest model name.”

## 2:35–3:35 — How we validated it

**Demonstrate:** Animate a timeline with six steps: Dec 2025 through May 2026.
At each step, everything left of the cutoff is training and one month is test.

Explain:

- A random split would let the model learn a future market and is forbidden.
- Before each month, every component and every neighbourhood table is rebuilt.
- Validation sold prices never enter features.
- The timing rule also applies inside training: each training home can use only
  sales with an earlier `date_sold`. Same-day sales are treated as a batch and
  cannot see one another.
- September–November 2025 selected blend weights and calibration; the six
  reported months evaluate the frozen blend.

**On screen:** Generated results table, not numbers typed into a slide.

Say: “Across 10,069 later sales, off-market validation MdAPE was **7.27%** and
list-aware MdAPE was **3.98%**. In plain language, half of predictions missed
by less than those percentages.” Optionally give MAPE as a secondary tail-error
measure, but keep MdAPE central because that is the competition metric.

## 3:35–4:35 — Where it fails

**Demonstrate:** Show the generated price-decile chart, then a compact breakdown
by property type and community-history count. Point to the pattern; do not just
state that errors exist.

Fill these evidence statements after the corrected run:

- Highest-error price band: top sold-price decile (13.42% off-market and 5.32%
  list-aware MdAPE).
- Hardest sufficiently represented property type: semi-detached (9.56%
  off-market and 5.84% list-aware MdAPE).
- Error for homes in sparse-history communities versus common communities:
  7.74% versus 6.96% off-market and 4.20% versus 3.81% list-aware when comparing
  communities with 200--999 earlier sales against those with 1,000 or more.
  Do not headline the 50--199 band because it contains only eight homes.
- Weakest validation month for both tracks: May 2026 (7.63% off-market and
  4.27% list-aware MdAPE).

Explain likely causes only after showing the evidence:

- Unusual high/low-priced homes have fewer truly similar examples.
- Rare property/community combinations force encodings toward the broad market
  average, losing local detail.
- Off-market is harder because the seller's price signal is unavailable.
- Month-to-month deterioration can indicate drift that historical weighting
  cannot fully anticipate.
- Remarks help only when they contain meaningful condition information.

Distinguish measured facts (“this segment had the highest MdAPE”) from an
interpretation (“likely because comparable evidence is sparse”).

## 4:35–5:00 — Honest close

**On screen:** Three limitations and one next step.

“June through September is the truly unseen hidden test, so validation is
evidence, not a guarantee. Rare groups remain uncertain, and the ensemble is
more expensive than one model. We would next add segment-level uncertainty and
use nested time splits for further tuning. The main lesson was to combine
different models with local history while enforcing what was actually knowable
on each sale date.”

## Recording checklist

- Use visuals or code/output for each of the four required questions.
- Say “strictly earlier sales,” never “leave one out.”
- Do not claim off-market uses list price, AVM data, outside property data, or
  future sales.
- Do not call December–May a perfectly untouched test set.
- Add team names and links before recording.
- Rehearse to 4:40–4:55, leaving a buffer under five minutes.
