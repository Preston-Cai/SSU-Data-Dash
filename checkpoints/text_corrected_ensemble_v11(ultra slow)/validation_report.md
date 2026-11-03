# V11: off-market text correction; list-aware uncorrected ensemble

Selected from the completed run: no new model training was performed. List-aware prices use the saved uncorrected ensemble column, rounded to dollars.

| Track | MAPE | MdAPE | Median signed error |
|---|---:|---:|---:|
| off_market | 10.09% | 7.67% | +0.23% |
| list_aware | 4.97% | 3.91% | -0.32% |

This checkpoint preserves the slow implementation with in-memory historical prediction caching. Source code routes off-market through text correction and list-aware through the uncorrected ensemble. The MODEL_PROFILE default and text_corrected profile reproduce this selection; ensemble skips off-market correction too. GPU retraining may vary slightly.

Source-and-results checkpoint only; trained model weights are not saved. Restore the source files to the project root to reproduce. original_validation_predictions.csv preserves the original experiment with correction in both tracks.