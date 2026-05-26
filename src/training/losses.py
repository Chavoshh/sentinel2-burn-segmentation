"""Loss functions for binary segmentation.

We combine BCE (binary cross-entropy) with Dice loss:
- BCE gives sharp per-pixel gradients but can be dominated by the majority class
  when classes are imbalanced (our train set is ~9.8:1 non-burn:burn)
- Dice directly optimizes overlap (IoU-like), making it robust to imbalance,
  but it has flat gradients when the prediction or ground truth is all-negative
- The weighted sum gets the best of both: BCE keeps gradients flowing, Dice
  ensures the model values predicting the minority class correctly.

Reference: this combination is the de-facto baseline in modern semantic
segmentation (e.g., nnU-Net uses Dice + CE).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Soft Dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    where P is predicted probabilities (after sigmoid) and G is the ground truth.

    The epsilon is critical: without it, samples with zero positive ground-truth
    pixels would have undefined loss. With it, those samples cleanly contribute
    a loss of ~1 when the model is wrong and ~0 when the model correctly
    predicts zero everywhere.

    Args:
        smooth: Numerical stability term. Default 1.0 is standard.
                Larger values make the loss less sharp; smaller values risk
                division-by-zero on all-negative samples.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1, H, W) raw model output; targets: (B, 1, H, W) in {0, 1}
        probs = torch.sigmoid(logits)
        # Flatten per-sample so we compute Dice per-sample then average
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        # Loss is 1 - Dice, averaged across the batch
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """Weighted combination of BCE-with-logits and Dice loss.

    Default weights (0.5, 0.5) are a sensible baseline. Try (0.4, 0.6) if
    Dice should dominate (more emphasis on the minority class), or (0.7, 0.3)
    if BCE should dominate (more emphasis on per-pixel calibration).

    Args:
        bce_weight: Weight for the BCE component (default 0.5).
        dice_weight: Weight for the Dice component (default 0.5).
        dice_smooth: Smoothing constant for Dice (default 1.0).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns a dict with the components so we can log them separately."""
        bce_val = self.bce(logits, targets)
        dice_val = self.dice(logits, targets)
        total = self.bce_weight * bce_val + self.dice_weight * dice_val
        return {
            "loss": total,
            "loss_bce": bce_val.detach(),
            "loss_dice": dice_val.detach(),
        }


if __name__ == "__main__":
    # Smoke test: verify the loss runs and produces sensible numbers
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    loss_fn = BCEDiceLoss()

    # Case 1: random logits vs random binary target → loss ~0.5-1.0
    logits = torch.randn(B, 1, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)
    out = loss_fn(logits, target)

    print("Case 1 — random logits vs random target:")
    print(f"  total: {out['loss'].item():.4f}")
    print(f"  bce:   {out['loss_bce'].item():.4f}")
    print(f"  dice:  {out['loss_dice'].item():.4f}")

    out["loss"].backward()
    print(f"  Grad on logits: max abs = {logits.grad.abs().max().item():.4f}")

    # Case 2: near-perfect predictions → loss should be very small
    target2 = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32)
    perfect_logits = target2 * 20 - 10  # Large positive where target=1, large negative where target=0
    out2 = loss_fn(perfect_logits, target2)
    print(f"\nCase 2 — near-perfect predictions:")
    print(f"  total: {out2['loss'].item():.6f}  (should be near 0)")

    # Case 3: all-zero target — verify Dice's edge case handling
    target_zero = torch.zeros(B, 1, H, W)
    logits_neg = torch.full((B, 1, H, W), -10.0)  # Predict ~0 everywhere
    out3 = loss_fn(logits_neg, target_zero)
    print(f"\nCase 3 — all-zero target, all-negative predictions:")
    print(f"  total: {out3['loss'].item():.6f}  (should be near 0, not NaN)")