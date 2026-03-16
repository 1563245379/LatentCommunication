import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LatentStopClassifier(nn.Module):
    """Binary classifier predicting whether to stop latent autoregressive generation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch, hidden_dim] or [hidden_dim] — last latent vector.
        Returns:
            stop_prob: [batch, 1] or [1] — probability of stopping (after sigmoid).
        """
        logits = self.classifier(hidden_state)
        return torch.sigmoid(logits)

    def predict(self, hidden_state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return boolean tensor indicating whether to stop.

        Args:
            hidden_state: [batch, hidden_dim] or [hidden_dim].
            threshold: decision boundary (default 0.5).
        Returns:
            should_stop: boolean tensor.
        """
        return self.forward(hidden_state).squeeze(-1) >= threshold


def save_classifier(
    classifier: LatentStopClassifier,
    path: str,
    hidden_dim: int,
) -> None:
    """Save classifier checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "hidden_dim": hidden_dim,
        },
        path,
    )
    print(f"[LatentStopClassifier] Saved to {path}")


def load_classifier(
    path: str,
    device: Optional[torch.device] = None,
) -> LatentStopClassifier:
    """Load classifier from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hidden_dim = checkpoint["hidden_dim"]
    classifier = LatentStopClassifier(hidden_dim)
    classifier.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        classifier = classifier.to(device)
    classifier.eval()
    print(f"[LatentStopClassifier] Loaded from {path} (hidden_dim={hidden_dim})")
    return classifier
