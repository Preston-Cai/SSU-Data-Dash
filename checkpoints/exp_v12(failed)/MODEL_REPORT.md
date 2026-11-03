# exp_v12(failed)

Archived semantic-text experiment from the completed validation run, with its exact saved source and predictions. No training or validation was rerun to create this checkpoint.

| Track / variant | MdAPE | MAPE |
|---|---:|---:|
| list_aware / predicted_price | 3.914% | 4.965% |
| list_aware / member_uncorrected_ensemble | 3.914% | 4.965% |
| off_market / predicted_price | 7.716% | 10.098% |
| off_market / member_uncorrected_ensemble | 7.682% | 10.105% |
| off_market / member_v11_prediction | 7.688% | 10.096% |

Marked failed because the off-market semantic correction worsened MdAPE against the uncorrected ensemble in the same run. The older lexical correction also did not improve MdAPE in this run. List-aware already uses the uncorrected ensemble.

The working project was rolled back to MODEL_PROFILE=ensemble for both tracks; its caching and feature-reuse improvements were retained. Saving this checkpoint does not change that working configuration. The source at the root of this checkpoint reproduces the failed experiment with MODEL_PROFILE=semantic.

To reproduce, restore these Python and requirements files to the project root, select MODEL_PROFILE=semantic, and run validation there. These are source and result artifacts, not trained model weights. Large reusable caches and downloaded encoder weights remain in their existing locations and are not duplicated here.
