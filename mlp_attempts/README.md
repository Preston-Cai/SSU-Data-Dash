# MLP experiments

This folder contains the complete custom PyTorch MLP workflow. It is separate
from both the starter commands and `codex_attempts/`.

## Folder layout

```text
mlp_attempts/
├── exec.py          training, checkpoint testing, and reporting entry point
├── mlp.py           preprocessing, embeddings, networks, and model wrappers
├── utils.py         training loop, optimizer setup, metrics, and checkpointer
├── explore.py       exploratory data plots
├── checkpoints/     timestamped PyTorch checkpoints
├── outputs/         frozen-checkpoint validation predictions
├── reports/         MLP validation reports and LaTeX table
└── figures/         validation and exploratory figures
```

The provided `features.py` and `models.py` at the repository root remain in
their original starter state. This workflow imports its own MLP-specific
`features.py` and `models.py` from this directory. Old checkpoints that stored
the former `mlp:ClassName` import path are remapped by the checkpointer and
remain loadable after this reorganization.

Install the project dependencies before running the workflow:

```powershell
pip install -r requirements.txt
pip install -r mlp_attempts/requirements.txt
```

## Edit the training configuration

The main editable settings are near the top of `exec.py` in `TRAIN_CONFIG` and
`TRACK_CONFIG`. They include:

- epochs, learning rate, weight decay, and batch size;
- optimizer name and optimizer-specific arguments;
- raw-price versus log-price target;
- validation fraction and random seed; and
- trainable embedding sizes for municipality and community.

## Train

Run from the repository root:

```powershell
python -m mlp_attempts.exec train --epochs 10
```

Train only one track:

```powershell
python -m mlp_attempts.exec train --track off_market --epochs 10
python -m mlp_attempts.exec train --track list_aware --epochs 10
```

On Windows, press **Q** to save a checkpoint and stop cleanly. Native numerical
libraries can intercept Ctrl+C before Python saves.

## Resume training

Resume the newest checkpoint for each requested track:

```powershell
python -m mlp_attempts.exec train --resume --epochs 10
```

Resume one exact checkpoint:

```powershell
python -m mlp_attempts.exec train `
  --resume-from "mlp_attempts\checkpoints\off_market_TIMESTAMP.pt" `
  --epochs 10
```

`--epochs` means additional epochs to run during this invocation.

## Test frozen checkpoints without retraining

Use the newest checkpoint for both tracks:

```powershell
python -m mlp_attempts.exec test
```

Or provide exact off-market and list-aware checkpoints, in that order:

```powershell
python -m mlp_attempts.exec test --checkpoints `
  "mlp_attempts\checkpoints\off_market_TIMESTAMP.pt" `
  "mlp_attempts\checkpoints\list_aware_TIMESTAMP.pt"
```

The same fitted models are reused for every validation month. Testing does not
call `fit` and therefore does not retrain six times.

## Generate the MLP report

After `test` writes `outputs/validation_predictions.csv`:

```powershell
python -m mlp_attempts.exec report
```

The report and figure are written inside this folder under `reports/` and
`figures/`, rather than mixing with starter or Codex outputs.

## Current saved work

The existing timestamped checkpoints were preserved under `checkpoints/`:

- `off_market_Adam_20261027-174427-540832.pt`
- `list_aware_Adam_20261027-174427-540832.pt`
- `list_aware_SGD_20261027-184149-749032.pt`

The existing validation predictions, reports, and exploration figures were also
moved here unchanged.
