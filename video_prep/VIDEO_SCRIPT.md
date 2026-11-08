# Five-minute conversational video script

The dialogue below is meant to sound like two people explaining the project,
not reading a report. The bracketed production notes are not spoken. They tell
a video-generation or editing agent exactly what to show while the speakers
talk.

**Visual style:** clean documentary presentation; white background, dark navy
text, blue for off-market, orange for list-aware, simple icons, and synthetic
homes only. Do not show real maps or invent data. Use the exact values written
in the annotations.

## 0:00–0:35 — Opening and introductions

**[VISUAL]** Title card: “Predicting Toronto Home Sale Prices”. Subtitle:
“Off-market and list-aware valuation”. Show two lower-thirds: “Preston — model
and validation” and “Lorance — experiments and interpretation”. Then animate a
split screen: off-market has a crossed-out asking-price tag; list-aware has a
visible asking-price tag.

Preston: Hi, I’m Preston.

Lorance: And I’m Lorance. So, our project is basically a home-price estimator,
but there are two versions of the question. One is: what is this home worth
before it is listed? That is our off-market track. The other is: now that we can
see the asking price, what will it probably sell for? That is list-aware.

Preston: And we were pretty careful about one thing from the beginning. The
model can only use information that would actually have been available at the
time we make the prediction.

## 0:35–1:30 — What the model sees

**[VISUAL]** Start with one synthetic house card. Animate four groups around it:
“home”: area, lot, rooms, parking, type, style; “location”: municipality,
community, coordinates, map cells; “time”: listing date and season; “remarks”:
renovation, condition, views, transit, luxury, and schools. Add small ratio
chips: “area per bedroom” and “bathrooms per bedroom”.

Preston: So, I’ll start with the home itself. We give it the physical details,
where it is, when it was listed, and what the listing actually says. We also
make a few ratios, because “two bathrooms” means something different in a very
small home than in a huge one.

**[VISUAL]** Draw three arrows from earlier synthetic sale cards to the current
home. Label them “strictly earlier sales”. Show chips: “log price”, “price per
square foot”, “community and type”, “nearby geography”, “recent weighting”, and
“sold-to-list ratio”.

Preston: But honestly, the feature family that mattered most was local history.
We look at similar homes that sold earlier. So the model is not just saying,
“this community is expensive.” It is asking, “what were comparable homes here
actually doing recently?”

Lorance: And that is where the two tracks start to differ. List-aware can also
look at how earlier homes sold relative to their asking prices. Off-market never
gets that current asking-price signal.

## 1:30–2:15 — How the ensemble works

**[VISUAL]** Animate three boxes into a “geometric blend”: “LightGBM — structured
features and history”, “XGBoost — second off-market estimate”, and “TF–IDF Ridge
— listing remarks”. For list-aware, add “independent value anchor” and
“community + property-type sold-to-list anchor”. End on “calibrated prediction”.

Lorance: We did not want one model making every decision. LightGBM is good at
mixing the structured features and the local history. XGBoost gives us another
off-market opinion. And the text model notices things in the remarks that a
tree model may not notice in the same way.

Preston: Then we blend those opinions. We predict log price off-market. For
list-aware, we predict how much the sale price moves above or below the asking
price. That makes the target more stable, and the final geometric blend keeps
the models on the same percentage scale.

## 2:15–2:55 — What we tried and rejected

**[VISUAL]** Show an experiment funnel. Cards entering: “PyTorch MLP”,
“CatBoost-heavy workflow”, “direct comparables”, “text variants”, “residual
corrections”, “weighting”, and “stacking”. Cards leaving are labelled “not
promoted”; the final GBT ensemble continues forward.

Lorance: We also tried a PyTorch MLP. It sounded promising, but the result was
eleven point one nine percent off-market MdAPE and five point six three percent
list-aware. Its inputs were thinner, especially because it did not have the
remarks and the strict local-history features.

Preston: We tried heavier CatBoost workflows and direct comparable models too.
Some were interesting, but they were either weaker in prevalidation or took a
lot longer without a dependable gain. So the lesson was not “use the fanciest
model.” It was more like: get the timing and the information right first.

## 2:55–3:45 — Walk-forward validation

**[VISUAL]** Timeline with six blocks: Dec 2025, Jan 2026, Feb 2026, Mar 2026,
Apr 2026, May 2026. Animate a cutoff moving right. Blue area before each cutoff
is “training”; orange block is “test”.

Preston: For validation, we wanted to imitate the real future test. So, for
December, we trained on sales before December. Then we moved the cutoff forward
one month and did it again, all the way through May.

Lorance: And this rule did not stop at the outer split. Inside the training
data, a home could only use sales from earlier dates. Homes sold on the same day
were processed together, so they could not leak their prices to one another.

**[VISUAL]** Score card: “10,069 homes per track”; large cards “OFF-MARKET
MdAPE: 7.27%” and “LIST-AWARE MdAPE: 3.98%”; small line “MAPE: 9.81% / 5.19%”.

Preston: Across ten thousand and sixty-nine homes, off-market MdAPE was seven
point two seven percent, and list-aware was three point nine eight percent.

Lorance: So, in plain language, half of the predictions missed by less than
those percentages. The mean errors were nine point eight one and five point one
nine percent.

## 3:45–4:35 — Where it fails

**[VISUAL]** Display `figures/error_by_price_range.png`. Highlight the rightmost
decile. Then show cards: “top price decile: 13.42% off-market, 5.32%
list-aware”; “semi-detached: 9.56%, 5.84%”; “sparse local history: about 7.7%,
4.2%”; “May 2026: 7.63%, 4.27%”. Add “associations, not proven causes”.

Preston: The model is not equally good for every home. The most expensive ten
percent were the hardest. Semi-detached homes were also the hardest sufficiently
represented property type.

Lorance: Homes in communities with fewer comparable sales did worse too, and
May was the weakest month for both tracks.

Preston: That makes sense as a likely explanation: unusual homes and rare local
groups give us fewer close examples. Off-market is harder because we lose the
asking-price clue, and an older sale may not describe a changing market.

Lorance: But those are interpretations of the pattern, not proof of the cause.

## 4:35–5:00 — Closing

**[VISUAL]** Closing card with three takeaways: “local history”, “strict timing”,
and “measure difficult segments”. Then show “Hidden test: June–September 2026”.

Lorance: June through September is still hidden, so validation is evidence, not
a guarantee.

Preston: If we continued, we would add uncertainty by segment and use nested
time splits for future tuning. The main takeaway is simple: combine different
models, use local history, and only use what was knowable on the prediction
date. Thanks for watching.
