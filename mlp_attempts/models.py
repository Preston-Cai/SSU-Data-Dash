"""Model factory used only by the custom PyTorch MLP workflow."""

from __future__ import annotations

def make_model(track: str, checkpoint=None):
    """Create a fresh MLP or restore one completely from a checkpoint."""
    # Keep PyTorch optional for users running only the provided starter code or
    # inspecting CLI/checkpoint helpers.
    from mlp_attempts.mlp import ListAwareMLP, OffMarketMLP

    if track == "list_aware":
        return ListAwareMLP(checkpoint=checkpoint)
    if track == "off_market":
        return OffMarketMLP(checkpoint=checkpoint)
    raise ValueError(f"Unknown track: {track}")
