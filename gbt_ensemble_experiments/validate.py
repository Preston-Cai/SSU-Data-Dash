"""Validate the selected ensemble and export starter-compatible predictions.

Full recomputation (slow):

    python gbt_ensemble_experiments/validate.py

Export the already-computed validation predictions without retraining:

    python gbt_ensemble_experiments/validate.py --reuse-results

After either command, the normal starter report works unchanged:

    python run.py report
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RESULTS = HERE / "results"
FINAL_PREDICTIONS = RESULTS / "final_walk_forward_predictions.csv"
STARTER_PREDICTIONS = PROJECT_ROOT / "outputs" / "validation_predictions.csv"
STARTER_COLUMNS = [
    "ml_num",
    "track",
    "month",
    "house_type_name",
    "sold_price",
    "predicted_price",
]


def run_script(relative_path: str) -> None:
    path = HERE / relative_path
    print(f"Running {path.relative_to(PROJECT_ROOT)}...", flush=True)
    subprocess.run([sys.executable, "-u", str(path)], cwd=PROJECT_ROOT, check=True)


def recompute_validation() -> None:
    """Rebuild every monthly component and apply the frozen final blend."""
    run_script("experiments/ensemble_walk_forward.py")
    run_script("experiments/xgboost_walk_forward.py")
    run_script("finalize_results.py")


def export_for_starter_report() -> Path:
    """Write the six-month predictions in the schema expected by run.py report."""
    if not FINAL_PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"{FINAL_PREDICTIONS} does not exist. Run this command without "
            "--reuse-results to recompute validation first."
        )
    predictions = pd.read_csv(FINAL_PREDICTIONS)
    missing = [column for column in STARTER_COLUMNS if column not in predictions]
    if missing:
        raise ValueError(f"Final validation predictions are missing columns: {missing}")

    exported = predictions[STARTER_COLUMNS].copy()
    exported["predicted_price"] = np.round(exported["predicted_price"])
    exported = exported.sort_values(["month", "track", "ml_num"], kind="stable")
    STARTER_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(STARTER_PREDICTIONS, index=False)
    return STARTER_PREDICTIONS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="skip model fitting and export the existing final validation CSV",
    )
    args = parser.parse_args(argv)

    if not args.reuse_results:
        recompute_validation()
    path = export_for_starter_report()
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}", flush=True)
    print("Now run: python run.py report", flush=True)


if __name__ == "__main__":
    main()
