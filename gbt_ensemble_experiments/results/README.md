# Results directory

The two files that summarize the selected validation result are:

- `final_walk_forward_report.json`: overall and monthly metrics.
- `final_walk_forward_predictions.csv`: row-level validation predictions.

The final blend selection is recorded in:

- `final_blend_selection.json`: chosen components, weights, and calibration;
- `final_blend_weight_search.csv`: every 0.05-grid candidate considered.

Files containing `screen` are earlier November screening experiments. Files
containing `prevalidation` were used for September-November model/blend choices.
The remaining walk-forward files are component predictions used to construct the
final blend. See `../docs/EXPERIMENT_REPORT.md` for interpretation.
