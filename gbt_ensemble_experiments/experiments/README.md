# Experiment scripts

These files document model ideas that were screened before selecting the final
ensemble. They are not the normal training entry point.

## Final validation components

- `ensemble_walk_forward.py`: rebuilds the LightGBM and text predictions for
  every official validation month.
- `xgboost_walk_forward.py`: rebuilds the XGBoost predictions for those months.
- After both finish, run `../finalize_results.py` to apply the frozen weights and
  write the final report under `../results/`.

## Main reusable building blocks

- `experiment.py`: shared raw feature functions and metrics.
- `lightgbm_experiment.py`: structured feature builder and LightGBM helpers.
- `linear_text_experiment.py`: TF-IDF/Ridge text feature builder.
- `target_encoding_experiment.py`: leakage-safe local-history encodings.
- `recency_encoding_experiment.py`: time-weighted local-history encodings.

## Alternative ideas retained for evidence

The other scripts cover CatBoost, direct comparable sales, local anchors,
sample weighting, list-price strategies, extended text, XGBoost screening,
stacking, and blend analysis. Their outputs are stored in `../results/`.

To train the selected final model, use `../train_final.py`, not a script in this
folder.
