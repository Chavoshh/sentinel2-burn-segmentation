"""Evaluate a trained burn-segmentation checkpoint on a held-out split.

Usage:
    uv run python scripts/evaluate.py --checkpoint models/<run-id>/best.pt
    uv run python scripts/evaluate.py --checkpoint models/<run-id>/best.pt --split val
"""

from __future__ import annotations

# IMPORTANT: numpy must be imported before torch on Windows.
import numpy as np  # noqa: F401

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BurnDataset  # noqa: E402
from src.models.unet import build_unet  # noqa: E402
from src.training.metrics import build_metrics, logits_to_probs  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a model and its training config from a checkpoint file.

    Returns:
        (model, config_dict) - model is on device, in eval mode.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = build_unet(
        encoder_name=config["encoder"],
        encoder_weights=None,  # don't re-download ImageNet weights, we'll overwrite with the checkpoint
        in_channels=config["in_channels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    logger.info(f"Model:       U-Net + {config['encoder']}")
    logger.info(f"Trained for: {ckpt['epoch']} epochs")
    logger.info(f"Val metrics at this checkpoint: {ckpt['val_metrics']}")
    return model, config


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> dict[str, float | int]:
    """Run inference on a loader, return aggregated metrics + confusion matrix."""
    metrics = build_metrics(prefix=f"{split_name}/").to(device)
    metrics.reset()

    # Pixel-level confusion matrix counters
    tp = fp = fn = tn = 0
    total_burn_pixels = 0
    total_pixels = 0

    pbar = tqdm(loader, desc=f"Evaluating on {split_name}")
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1)  # (B, 1, H, W)

        logits = model(images)
        probs = logits_to_probs(logits)
        preds = (probs > 0.5).int()
        targets = masks.int()

        metrics.update(probs, targets)

        # Confusion matrix accumulation
        tp += int(((preds == 1) & (targets == 1)).sum().item())
        fp += int(((preds == 1) & (targets == 0)).sum().item())
        fn += int(((preds == 0) & (targets == 1)).sum().item())
        tn += int(((preds == 0) & (targets == 0)).sum().item())
        total_burn_pixels += int(targets.sum().item())
        total_pixels += int(targets.numel())

    results = {k: float(v.item()) for k, v in metrics.compute().items()}
    results["confusion_matrix/tp"] = tp
    results["confusion_matrix/fp"] = fp
    results["confusion_matrix/fn"] = fn
    results["confusion_matrix/tn"] = tn
    results["total_burn_pixels"] = total_burn_pixels
    results["total_pixels"] = total_pixels
    results["burn_fraction"] = total_burn_pixels / total_pixels if total_pixels else 0.0
    return results


def print_results_table(results: dict[str, float | int], split_name: str) -> None:
    """Pretty-print the evaluation results."""
    print()
    print("=" * 60)
    print(f"  Evaluation results on {split_name.upper()} split")
    print("=" * 60)

    # Headline metrics
    print(f"\n  IoU (Jaccard):    {results[f'{split_name}/iou']:.4f}")
    print(f"  F1 (Dice):        {results[f'{split_name}/f1']:.4f}")
    print(f"  Precision:        {results[f'{split_name}/precision']:.4f}")
    print(f"  Recall:           {results[f'{split_name}/recall']:.4f}")

    # Confusion matrix
    tp, fp, fn, tn = (
        results["confusion_matrix/tp"],
        results["confusion_matrix/fp"],
        results["confusion_matrix/fn"],
        results["confusion_matrix/tn"],
    )
    total = tp + fp + fn + tn
    print(f"\n  Pixel-level confusion matrix ({total:,} total pixels):")
    print(f"                  Predicted")
    print(f"                  no-burn      burn")
    print(f"    True no-burn  {tn:>10,}  {fp:>10,}")
    print(f"    True burn     {fn:>10,}  {tp:>10,}")

    # Sanity checks
    burn_recall = tp / (tp + fn) if (tp + fn) else 0
    burn_precision = tp / (tp + fp) if (tp + fp) else 0
    print(f"\n  Burn pixels in ground truth: {results['total_burn_pixels']:,} "
          f"({100*results['burn_fraction']:.2f}% of total)")
    print(f"  Sanity check: recall from CM = {burn_recall:.4f} "
          f"(should match metric: {results[f'{split_name}/recall']:.4f})")
    print(f"  Sanity check: precision from CM = {burn_precision:.4f} "
          f"(should match metric: {results[f'{split_name}/precision']:.4f})")
    print("=" * 60)


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate a trained burn-segmentation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a .pt checkpoint file (e.g. models/<run-id>/best.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
    parser.add_argument("--patches-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "patches")
    parser.add_argument("--splits-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Where to save results JSON (default: docs/<split>_results.json)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, train_config = load_model_from_checkpoint(args.checkpoint, device)

    dataset = BurnDataset(
        patches_dir=args.patches_dir,
        splits_csv=args.splits_csv,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    logger.info(f"{args.split.title()} dataset: {len(dataset)} samples, {len(loader)} batches")

    results = evaluate(model, loader, device, args.split)
    print_results_table(results, args.split)

    # Save to JSON
    output_path = args.output_json or (PROJECT_ROOT / "docs" / f"{args.split}_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(dataset),
        "results": results,
        "train_config": train_config,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())