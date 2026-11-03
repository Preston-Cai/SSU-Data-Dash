# Semantic text experiment after v11

Current default: both tracks use the uncorrected cached ensemble. The semantic
challenger did not improve off-market MdAPE in its completed validation.
Use `$env:MODEL_PROFILE = "ensemble"` in an existing PowerShell session to
override any earlier `semantic` setting. The experiment below remains available
explicitly; all speed improvements and saved caches are retained.

The preserved checkpoint is `checkpoints/text_corrected_ensemble_v11(ultra slow)`.
It keeps off-market text correction (7.674% MdAPE) and the stronger list-aware
uncorrected ensemble (3.912% MdAPE). Its selected predictions come from the
completed run; its original predictions are also included. It contains source
and results, not trained weights.

## Run on this Windows / RTX 4060 machine

The optional semantic packages are not installed yet. Install them once:

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-semantic.txt
$env:MODEL_PROFILE = "semantic"
python run.py validate
python run.py report
```

The CUDA wheel is available from the [official PyTorch index](https://download.pytorch.org/whl/cu128/torch/).
The first semantic run downloads about 1.34 GB of encoder weights. Validation
prepares the required remarks before starting price training; this preparation
is resumable. `python run.py prepare_text` runs just that optional preparation.
For smaller encoding batches, use `python prepare_semantic_text.py --batch-size 8`.
`SSU_SEMANTIC_DEVICE=cpu` forces CPU encoding if needed; it will be slower.

## What changed

List-aware keeps the best uncorrected ensemble. Off-market fits two corrections
on the same six months of strictly earlier out-of-fold errors: the v11 lexical
correction and a challenger that adds frozen, contextual BGE remark vectors.
The challenger uses centered vectors with scale 3, Ridge alpha 20, correction
shrinkage 0.5, and a log-adjustment cap of 0.10, fixed before validation.
It does not tune settings against the outer validation labels.

The [pinned BGE revision](https://huggingface.co/BAAI/bge-large-en-v1.5/commit/d4aa6901d3a41ba39fb536a557fa166f842b0e09)
predates the validation period. Its encoder is frozen; only the correction,
lexical vocabulary and semantic centering are fitted on historical residuals.
Long remarks are chunked so their ending is retained. Text is processed locally
after the weights download. This is an experiment, not a confirmed improvement.

The report includes `member_v11_prediction` for off-market, using exactly the
same underlying price forecasts as the challenger. Compare it with
`predicted_price` and `member_uncorrected_ensemble`. The new comparison avoids
confounding a text change with the small variation in GPU retraining.

## Speed and cache behavior

- Each ensemble member's full-precision predictions persist in `outputs/cache`.
  Identical training/query data can be reused by outer and inner folds and after
  a restart. A completed member is saved before the next one starts.
- Deterministic history features and identical TF-IDF/SVD fits are shared in
  bounded memory. Independent 64- and 128-component fits remain separate.
- Frozen text vectors persist in SQLite by exact text and encoder revision.
  Changing correction settings does not require re-encoding the remarks.
- Training rows, tree counts and ensemble weights are unchanged. List-aware no
  longer trains six historical correction folds that worsened its score.
- A cold six-month run needs 114 individual CatBoost fits instead of the old
  approximately 204: 12 unique off-market cutoffs times 7 fits, plus 6 list-aware
  cutoffs times 5 fits. Feature reuse is additional savings. This is a workload
  count, not a measured runtime guarantee, and semantic preparation adds cost.

The first run must populate the new cache. Old CSVs did not record training-data
fingerprints, so they are not silently treated as verified cache entries.
Unchanged subsequent runs reuse the price predictions; feature/data/model
changes invalidate the affected prediction keys. Changing the semantic
correction alone retains the base prediction cache. GPU reruns without a cache
can still differ slightly. `SSU_CACHE_DIR` changes the cache location.
`SSU_DISABLE_CACHE=1` bypasses price/feature reuse for debugging; frozen semantic
text remains cached. Cache files contain no executable pickled models.

For the v11 architecture with the speed improvements and no neural dependencies:

```powershell
$env:MODEL_PROFILE = "v11"
python run.py validate
python run.py report
```

`python run.py predict` writes test predictions and reuses matching cached
history. A new training cutoff or new query still requires price fitting once;
saved prediction arrays are not saved model weights. Hidden test scores remain
unavailable locally. Run `python run.py check` to check submission structure.

Full training, encoder inference, and validation are left to the user. Local
checks exercise cache correctness, temporal boundaries and routing with mocks.
