"""Per-sample IoU distribution analysis on val and test splits.

Helps explain why aggregate IoU might differ between val and test by showing
the underlying distribution of per-patch performance.

Usage:
    uv run python scripts/analyze_per_sample.py --checkpoint models/<run-id>/best.pt
"""

from __future__ import annotations

# IMPORTANT: numpy must be imported before torch on Windows.
import numpy as np  # noqa: F401

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BurnDataset  # noqa: E402
from src.models.unet import build_unet  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_unet(
        encoder_name=config["encoder"],
        encoder_weights=None,
        in_channels=config["in_channels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def per_sample_metrics(
    model: torch.nn.Module,
    dataset: BurnDataset,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    """Compute IoU, F1, precision, recall for each individual sample.

    Returns a DataFrame with one row per patch.
    """
    records = []
    # batch_size=1 so we process one patch at a time and can identify it
    for idx in tqdm(range(len(dataset)), desc=f"Per-sample on {split_name}"):
        image, mask = dataset[idx]
        sample_name = dataset.sample_names[idx]

        image = image.unsqueeze(0).to(device)  # (1, 6, H, W)
        target = (mask > 0.5).int().to(device)  # (H, W)

        logits = model(image)
        pred = (torch.sigmoid(logits).squeeze() > 0.5).int()  # (H, W)

        # Per-patch confusion matrix
        tp = int(((pred == 1) & (target == 1)).sum().item())
        fp = int(((pred == 1) & (target == 0)).sum().item())
        fn = int(((pred == 0) & (target == 1)).sum().item())
        tn = int(((pred == 0) & (target == 0)).sum().item())

        # Per-patch metrics (handling degenerate cases for patches with zero burn)
        n_burn = tp + fn
        n_pred_burn = tp + fp

        # IoU is well-defined unless both prediction and truth are all-zero
        iou_denom = tp + fp + fn
        iou = tp / iou_denom if iou_denom > 0 else float("nan")  # nan for all-no-burn patches
        precision = tp / n_pred_burn if n_pred_burn > 0 else float("nan")
        recall = tp / n_burn if n_burn > 0 else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

        records.append({
            "sample_name": sample_name,
            "burn_fraction": n_burn / target.numel(),
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "burn_pixels_gt": n_burn,
            "burn_pixels_pred": n_pred_burn,
        })

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, split_name: str) -> None:
    """Pretty-print per-sample summary stats."""
    print()
    print("=" * 70)
    print(f"  Per-sample IoU distribution on {split_name.upper()} ({len(df)} samples)")
    print("=" * 70)

    iou_valid = df["iou"].dropna()
    print(f"  Samples with computable IoU: {len(iou_valid)} (NaN = all-no-burn patches)")
    print(f"\n  Per-sample IoU statistics:")
    print(f"    min:    {iou_valid.min():.4f}")
    print(f"    25%:    {iou_valid.quantile(0.25):.4f}")
    print(f"    median: {iou_valid.median():.4f}")
    print(f"    mean:   {iou_valid.mean():.4f}  (per-sample mean, different from aggregate)")
    print(f"    75%:    {iou_valid.quantile(0.75):.4f}")
    print(f"    max:    {iou_valid.max():.4f}")

    # Bucket by burn fraction to see where the model struggles
    df = df.copy()
    df["burn_bucket"] = pd.cut(
        df["burn_fraction"] * 100,
        bins=[-0.01, 0.5, 5, 20, 100],
        labels=["<0.5%", "0.5-5%", "5-20%", "20-100%"],
    )
    print(f"\n  IoU by burn coverage bucket:")
    bucket_stats = df.groupby("burn_bucket", observed=True)["iou"].agg(["count", "mean", "median"])
    print(bucket_stats.to_string())

    # Worst 5 samples
    worst = df.dropna(subset=["iou"]).nsmallest(5, "iou")[["sample_name", "burn_fraction", "iou"]]
    print(f"\n  5 worst samples (lowest IoU):")
    print(worst.to_string(index=False))


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--patches-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "patches")
    parser.add_argument("--splits-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    logger.info("Model loaded.")

    for split in ["val", "test"]:
        dataset = BurnDataset(
            patches_dir=args.patches_dir,
            splits_csv=args.splits_csv,
            split=split,
        )
        df = per_sample_metrics(model, dataset, device, split)
        summarize(df, split)

        out_path = args.output_dir / f"{split}_per_sample.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved per-sample results: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())