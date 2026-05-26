"""Metrics for binary segmentation training.

We use torchmetrics for two reasons:
1. Correctness — handles edge cases (zero-positive batches, etc.) properly
2. Distributed/batched aggregation — accumulates state across batches without
   bias, so per-epoch metrics aren't just averages of per-batch metrics
   (which would over-weight small batches).
"""

from __future__ import annotations

import torch
import torchmetrics
import torchmetrics.classification as tmc


def build_metrics(prefix: str = "") -> torchmetrics.MetricCollection:
    """Build the standard metric collection for binary burn segmentation.

    Args:
        prefix: Optional prefix for metric names (e.g., "val/" or "train/").
                Useful when you want to log train and val metrics separately.

    Returns:
        A MetricCollection that can be updated batch-by-batch and computed
        once per epoch. Usage:

            metrics = build_metrics(prefix="val/")
            metrics.to(device)
            for batch in val_loader:
                ...
                metrics.update(preds, targets)
            results = metrics.compute()  # dict of {"val/iou": ..., "val/f1": ..., ...}
            metrics.reset()
    """
    # threshold=0.5 means: pixel is "burn" if predicted probability > 0.5
    # All metrics expect predictions as probabilities in [0, 1], not logits.
    metric_kwargs = dict(task="binary", threshold=0.5)

    return torchmetrics.MetricCollection({
    "iou":       tmc.BinaryJaccardIndex(threshold=0.5),
    "f1":        tmc.BinaryF1Score(threshold=0.5),
    "precision": tmc.BinaryPrecision(threshold=0.5),
    "recall":    tmc.BinaryRecall(threshold=0.5),
}, prefix=prefix)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw model logits to probabilities for metric computation."""
    return torch.sigmoid(logits)


if __name__ == "__main__":
    # Smoke test: verify metrics produce sensible numbers on known inputs
    torch.manual_seed(0)
    metrics = build_metrics(prefix="test/")

    # Case 1: perfect predictions → all metrics should be ~1.0
    target = torch.randint(0, 2, (4, 1, 64, 64))
    perfect_probs = target.float()
    metrics.update(perfect_probs, target)
    print("Perfect predictions:")
    for name, val in metrics.compute().items():
        print(f"  {name}: {val.item():.4f}")
    metrics.reset()

    # Case 2: all-negative predictions on imbalanced target (~10% positive)
    # Mimics our actual class distribution
    target = (torch.rand(4, 1, 64, 64) < 0.1).int()
    zero_probs = torch.zeros_like(target).float()
    metrics.update(zero_probs, target)
    print("\nAll-zero predictions on ~10% positive target:")
    for name, val in metrics.compute().items():
        print(f"  {name}: {val.item():.4f}")
    print("  (IoU and F1 should be 0; recall should be 0; precision is undefined → 0 by convention)")