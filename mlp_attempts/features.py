"""Feature construction used only by the PyTorch MLP experiments."""

from __future__ import annotations

import pandas as pd


PROPERTY_TYPES = [
    "Condo Apt",
    "Detached",
    "Semi-Detached",
    "Condo Townhouse",
    "Freehold Townhouse",
    "Co-Op Apt",
]
MLP_ONE_HOT_COLUMNS = ["house_style"]
MLP_ID_COLUMNS = ["municipality", "community"]
MLP_CATEGORICAL_COLUMNS = MLP_ONE_HOT_COLUMNS + MLP_ID_COLUMNS


def square_feet(house_area):
    """Convert an area value or range to one numeric midpoint."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def base_off_market_features(homes):
    """Build the original numeric starter inputs inside the MLP package."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["months"] = (
        homes["date_listed"] - pd.Timestamp("2021-01-01")
    ).dt.days / 30.44
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def off_market_mlp_features(homes):
    """Return mixed numeric/categorical inputs for the off-market MLP."""
    X = base_off_market_features(homes)
    X["garage"] = homes["garage"]
    X["lot_front"] = homes["lot_front"]
    X["lot_depth"] = homes["lot_depth"]
    for column in MLP_CATEGORICAL_COLUMNS:
        X[column] = homes[column]
    return X


def list_aware_mlp_features(homes):
    """Return the MLP inputs available when list price is known."""
    X = off_market_mlp_features(homes)
    X["list_price"] = homes["list_price"]
    return X
