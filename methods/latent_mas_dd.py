"""
LatentMAS-DD: Data-Driven alignment variant.

Replaces the static parameter-space alignment (W_out -> W_in) with a
data-driven ridge regression alignment trained on actual hidden-state pairs:
    h_L[t] -> h_K[t+1]   (last hidden state -> target layer, temporal shift)

The execution flow is identical to LatentMASMethod; only the alignment
matrix is swapped with the pre-trained one loaded from disk.
"""

import argparse
from typing import Dict, List, Tuple, Optional

import torch

from models import ModelWrapper
from methods.latent_mas import LatentMASMethod


class LatentMASDDMethod(LatentMASMethod):
    """LatentMAS with Data-Driven alignment matrix (ridge regression)."""

    def __init__(
        self,
        model: ModelWrapper,
        *,
        alignment_path: str,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__(
            model,
            latent_steps=latent_steps,
            judger_max_new_tokens=judger_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            args=args,
        )
        # Override the alignment matrix with the pre-trained data-driven one
        self._load_dd_alignment(model, alignment_path)

    def _load_dd_alignment(self, model: ModelWrapper, alignment_path: str):
        """Load pre-trained alignment matrix and override ModelWrapper's matrix."""
        checkpoint = torch.load(alignment_path, map_location="cpu", weights_only=True)
        W_align = checkpoint["W_align"].float()
        target_norm = checkpoint["target_norm"].float()

        source_model = model.model
        target_device = model.device

        W_align = W_align.to(target_device)
        target_norm = target_norm.to(target_device)

        key = id(source_model)
        model._latent_realign_matrices[key] = (W_align, target_norm)

        print(f"[LatentMAS-DD] Loaded data-driven alignment matrix from {alignment_path}")
        print(f"  Matrix shape: {W_align.shape}, Target norm: {target_norm.item():.4f}")

    def _filter_latent_past_kv(self, past_kv: Optional[Tuple]) -> Tuple[Optional[Tuple], int]:
        """Only keep KV cache entries from the autoregressive latent steps."""
        from models import _past_length
        original_len = _past_length(past_kv)
        truncated = self._truncate_past(past_kv, self.latent_steps)
        offset = original_len - self.latent_steps if original_len > self.latent_steps else 0
        return truncated, offset