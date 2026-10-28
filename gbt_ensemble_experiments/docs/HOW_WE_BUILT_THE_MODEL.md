# How we improved the home-price model

This is the short, non-specialist explanation of the experiment. It is written
for a computer science student who understands code and data, but does not need
much machine-learning background.

## The problem we were solving

We needed to predict how much a home would sell for. There were two versions of
the question.

In the **off-market** version, we only know facts about the home, such as its
location, type, size, and description. We do not know its asking price. In the
**list-aware** version, we also know the asking price. The second problem is
easier because the seller has already given us a strong clue.

There was also an important time rule: when predicting a sale, the model could
only learn from older sales. Letting it see later sales would be like studying
the answer key before an exam. It would look accurate during development but
would fail when asked to predict the future.

The user's existing models were already good. Their typical error was 11.19%
off-market and 4.48% list-aware. The goal was not merely to replace their neural
network with a fashionable algorithm. The goal was to understand what useful
information the model was missing.

## How I approached it

I treated the work like a sequence of questions rather than one large model
search.

First, I asked: **what would a human real-estate appraiser look at?** They would
consider the home's physical details, but they would also look at nearby sales,
recent changes in the market, similar property types, and the wording of the
listing. That became the experimental roadmap.

Second, I kept the new work in a separate `codex_attempts` folder. This meant
I could try aggressive changes without damaging the user's existing workflow.
It also made unsuccessful experiments useful: their code and results could be
kept as evidence instead of being mixed into the final model.

Third, I used older months to choose ideas and later months to check whether
they worked on the future. This was slower than a random train/test split because
the models had to be retrained for each month, but it matched the real task.

## Step 1: describe each home more clearly

Raw data is not always presented in the most useful form. For example, house
area could be a range, and lot dimensions were stored separately. I converted
these into simpler numerical descriptions: approximate floor area, lot area,
area per bedroom, bathrooms per bedroom, and related ratios.

I also represented location in several ways. A neighborhood name is helpful,
but nearby coordinates can carry more detail. I divided latitude and longitude
into small geographic cells and combined neighborhood with property type and
style. These combinations let the model learn ideas such as “condominiums in
this neighborhood behave differently from detached homes in the same area.”

Time was included because Toronto's synthetic housing market changes from year
to year and from season to season. Finally, I extracted simple clues from the
remarks, such as whether a listing mentioned renovation, luxury, transit, or an
as-is condition.

These features gave boosted-tree models a strong starting point. A boosted tree
can be thought of as a large collection of small decision rules. One rule might
adjust for home size, another for neighborhood, and another for a particular
combination of property type and date.

## Step 2: teach the model about earlier local sales

This was the most important improvement.

The original property fields tell us what a home is, but not directly what its
local market is worth. I created historical summaries such as the typical prior
sale price in a community, the typical price per square foot for that property
type, and—for the list-aware task—the usual relationship between asking and
selling price.

There is a danger here. If the summary for a home includes that home's own sale
price, we have accidentally handed the answer to the model. The code therefore
uses leave-one-out calculations while training and only older sales while
predicting. Small groups are also pulled toward the citywide average, because
one or two unusual sales should not define an entire neighborhood.

This one idea improved the off-market model more than simply switching between
model libraries. That was a useful lesson: better information often matters
more than a more complicated algorithm.

## Step 3: make recent sales matter more

A sale from last month usually describes the current market better than a sale
from four years ago. I added a second version of the local summaries where older
sales gradually receive less weight.

The ordinary history and recent history were both useful. The ordinary version
is more stable; the recent version reacts faster to market changes. Keeping both
gave the final system two different views of the same neighborhood.

## Step 4: try genuinely different approaches

I tested LightGBM, XGBoost, CatBoost, text regression, and a hand-built comparable
sales method. These names are less important than the different ways they look
at the problem.

The tree models were strongest on structured data and local history. A separate
text model read the listing remarks and structured facts as words. It was not
the best off-market model by itself, but it noticed patterns the trees missed.

The direct comparable-sales method tried to act like an appraiser by selecting
nearby, recent, physically similar homes. Surprisingly, it was clearly worse
than the learned models. CatBoost was also slower and weaker off-market than
expected. I did not keep these approaches just because they sounded sensible or
were popular. Their ideas influenced later features, but their predictions were
left out of the final system.

This was the general decision rule throughout the experiment: keep an idea when
it improves later-month predictions or adds useful diversity; otherwise record
the result and move on.

## Step 5: combine models that make different mistakes

The final system is an ensemble, which simply means that several models vote on
the price.

For off-market homes, most of the vote comes from XGBoost, with smaller votes
from a recency-aware LightGBM model and the text model. For list-aware homes, two
LightGBM models—one stable and one recency-aware—are combined with the text
model.

Why keep the text model if it is weaker on its own? Imagine asking three people
to estimate a home's value. Two use spreadsheets and one reads the listing
carefully. The reader might be less accurate on average, but can still improve
the group when the spreadsheets overlook something. An ensemble benefits from
different mistakes, not just individually high scores.

The combination uses a geometric average because the models learn prices in log
space. In plain language, they focus on percentage differences rather than raw
dollar differences. That prevents very expensive homes from dominating the
training objective.

## What the result means

The final validation results were:

| Track | Earlier typical error | Final typical error |
|---|---:|---:|
| off-market | 11.19% | **7.28%** |
| list-aware | 4.48% | **4.08%** |

“Typical error” here means MdAPE: the median percentage difference between the
prediction and actual selling price. The off-market reduction was about 35%
relative to the earlier model, which was the main success. The list-aware gain
was smaller because the asking price already made that problem difficult to
improve much further.

The results were measured by repeatedly training on the past and predicting the
next month from December 2025 through May 2026. Model weights were chosen using
September through November. However, several finalist models were inspected on
the December-May period during development, so it is not correct to call that a
perfectly untouched test set. The competition's hidden June-September sales are
the real final test.

## What I would do next

I would first study where the model still fails: very expensive homes, rare
property types, neighborhoods with few prior sales, or particular months. That
would tell us whether the next improvement should be another feature, better
handling of rare categories, or a way to follow market changes faster.

I would also add uncertainty estimates. A price prediction such as `$900,000`
looks more confident than it really is. Saying “roughly `$840,000` to `$970,000`”
would be more useful to a buyer or seller, especially for unusual homes.

## Where to go from here

If you only want to run or retrain the selected model, read
`../FINAL_MODEL.md`.
It contains the checkpoint location and exact commands.

If you want every technical detail, experiment file, validation caveat, and
research source, continue with `EXPERIMENT_REPORT.md` in this directory.

If you want to connect this story to the exact feature, encoding, blending, and
calibration code, open `CODE_WALKTHROUGH.ipynb` in this directory.
