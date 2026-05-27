"""Generate the final results.

Selects representative test samples (spanning the IoU range) and renders them
as a multi-panel grid: RGB | SWIR | Ground Truth | Prediction | Error map.

Honest sample selection: we pick top, median, and bottom performers (excluding
near-empty patches) rather than cherry-picking the best ones.

Usage:
    uv run python scripts/make_results_figure.py --checkpoint models/<run-id>/best.pt
"""

from __future__ import annotations

# IMPORTANT: numpy must be imported before torch on Windows.
import numpy as np

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BurnDataset  # noqa: E402
from src.models.unet import build_unet  # noqa: E402

logger = logging.getLogger(__name__)


# --- Band indices (within the 6-band pre-processed cache, NOT the original 12-band)
# Our cache stores bands in this order: B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR-1), B12(SWIR-2)
CACHED_BLUE = 0
CACHED_GREEN = 1
CACHED_RED = 2
CACHED_NIR = 3
CACHED_SWIR1 = 4
CACHED_SWIR2 = 5


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


def stretch_for_display(img: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Percentile-clip and rescale an HxWxC image to [0,1] for display."""
    img = img.astype(np.float32)
    out = np.zeros_like(img)
    for c in range(img.shape[-1]):
        channel = img[..., c]
        lo = np.percentile(channel, lo_pct)
        hi = np.percentile(channel, hi_pct)
        if hi > lo:
            out[..., c] = np.clip((channel - lo) / (hi - lo), 0, 1)
    return out


def make_rgb(image_chw: np.ndarray) -> np.ndarray:
    """Extract RGB from a (6, H, W) cached image."""
    rgb = image_chw[[CACHED_RED, CACHED_GREEN, CACHED_BLUE], :, :]
    rgb = np.transpose(rgb, (1, 2, 0))
    return stretch_for_display(rgb)


def make_swir(image_chw: np.ndarray) -> np.ndarray:
    """Extract SWIR-2/NIR/Red false-color composite."""
    swir = image_chw[[CACHED_SWIR2, CACHED_NIR, CACHED_RED], :, :]
    swir = np.transpose(swir, (1, 2, 0))
    return stretch_for_display(swir)


def select_diverse_samples(per_sample_csv: Path, n_per_bucket: int = 2) -> list[dict]:
    """Pick a balanced set of test samples spanning the IoU range.

    Strategy: among patches with meaningful burn (>0.5% coverage), pick:
    - Top N (best performers — what the model does well)
    - Median N (typical performance)
    - Bottom N (honest limitations)
    """
    df = pd.read_csv(per_sample_csv)
    # Exclude near-empty AND near-total burn patches; both are uninformative
    # for showing what the model does spatially.
    df = df[(df["burn_fraction"] >= 0.005) & (df["burn_fraction"] <= 0.80)].dropna(subset=["iou"]).copy()
    df = df.sort_values("iou", ascending=False).reset_index(drop=True)

    n = len(df)
    if n < n_per_bucket * 3:
        logger.warning(f"Only {n} samples meet criteria; figure may have fewer rows than requested.")

    # Top, middle, bottom indices
    top_idx = list(range(0, n_per_bucket))
    mid_start = (n - n_per_bucket) // 2
    mid_idx = list(range(mid_start, mid_start + n_per_bucket))
    bot_idx = list(range(n - n_per_bucket, n))

    selected = []
    for label, indices in [("strong", top_idx), ("median", mid_idx), ("hard", bot_idx)]:
        for i in indices:
            row = df.iloc[i]
            selected.append({
                "sample_name": row["sample_name"],
                "burn_fraction": row["burn_fraction"],
                "iou": row["iou"],
                "bucket": label,
            })
    return selected


@torch.no_grad()
def infer_one(model: torch.nn.Module, dataset: BurnDataset, sample_name: str, device: torch.device):
    """Run inference on one sample and return (image_chw, mask_2d, pred_2d, prob_2d)."""
    idx = dataset.sample_names.index(sample_name)
    image, mask = dataset[idx]
    image_chw = image.numpy()  # (6, H, W)
    mask_np = mask.numpy()  # (H, W)

    logits = model(image.unsqueeze(0).to(device))
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)
    pred = (prob > 0.5).astype(np.uint8)
    return image_chw, mask_np.astype(np.uint8), pred, prob


def make_error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Build an RGB error map:
    - White: true negative (no burn, correctly predicted)
    - Green: true positive (burn correctly detected)
    - Red:   false positive (predicted burn, was no burn)
    - Blue:  false negative (missed burn)
    """
    h, w = gt.shape
    error = np.ones((h, w, 3), dtype=np.float32)  # start white

    tp_mask = (gt == 1) & (pred == 1)
    fp_mask = (gt == 0) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)

    # True positive: light green
    error[tp_mask] = [0.4, 0.85, 0.4]
    # False positive: red
    error[fp_mask] = [0.95, 0.3, 0.3]
    # False negative: blue
    error[fn_mask] = [0.2, 0.5, 0.95]
    return error


def render_figure(
    samples: list[dict],
    model: torch.nn.Module,
    dataset: BurnDataset,
    device: torch.device,
    output_path: Path,
) -> None:
    """Generate the multi-panel results figure."""
    n_rows = len(samples)
    n_cols = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), facecolor="white")
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    col_titles = [
        "Post-fire RGB",
        "False-color SWIR\n(B12-B08-B04)",
        "Ground truth",
        "Model prediction",
        "Error map",
    ]

    # Mask colormap: black/white for binary masks
    binary_cmap = ListedColormap(["#e0e0e0", "#c0392b"])  # darker bg so small burns are visible

    for row, sample in enumerate(samples):
        image_chw, gt, pred, prob = infer_one(model, dataset, sample["sample_name"], device)
        rgb = make_rgb(image_chw)
        swir = make_swir(image_chw)
        error = make_error_map(gt, pred)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(swir)
        axes[row, 2].imshow(gt, cmap=binary_cmap, vmin=0, vmax=1)
        axes[row, 3].imshow(pred, cmap=binary_cmap, vmin=0, vmax=1)
        axes[row, 4].imshow(error)

        # Row label on the leftmost panel (rotated y-label)
        bucket_label = {
            "strong": "Strong",
            "median": "Median",
            "hard": "Hard",
        }[sample["bucket"]]
        axes[row, 0].set_ylabel(
            f"{bucket_label}\nIoU = {sample['iou']:.2f}\nBurn = {sample['burn_fraction']*100:.1f}%",
            fontsize=11,
            rotation=0,
            labelpad=45,
            va="center",
        )

        # Per-row info (only in first column to avoid clutter)
        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, fontsize=12)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#cccccc")

    # Legend for the error map
    legend_text = (
        "Error map legend:  "
        r"$\bf{green}$ = true positive, "
        r"$\bf{red}$ = false positive (over-prediction), "
        r"$\bf{blue}$ = false negative (missed burn)"
    )
    fig.suptitle(
        "Test-set predictions across difficulty levels",
        fontsize=15, y=0.995,
    )
    fig.text(0.5, 0.005, legend_text, ha="center", fontsize=9, color="#555555")

    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved figure: {output_path}")
    plt.close(fig)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--per-sample-csv", type=Path,
        default=PROJECT_ROOT / "docs" / "test_per_sample.csv",
    )
    parser.add_argument("--patches-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "patches")
    parser.add_argument("--splits-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "results_grid.png")
    parser.add_argument("--n-per-bucket", type=int, default=2,
                        help="How many samples to pick per bucket (strong/median/hard). Total rows = 3 × n.")
    args = parser.parse_args()

    if not args.per_sample_csv.exists():
        logger.error(f"Per-sample CSV not found: {args.per_sample_csv}")
        logger.error("Run scripts/analyze_per_sample.py first to generate it.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    dataset = BurnDataset(
        patches_dir=args.patches_dir,
        splits_csv=args.splits_csv,
        split="test",
    )

    selected = select_diverse_samples(args.per_sample_csv, n_per_bucket=args.n_per_bucket)
    logger.info(f"Selected {len(selected)} samples for the figure:")
    for s in selected:
        logger.info(f"  [{s['bucket']:>6}] {s['sample_name']}  IoU={s['iou']:.3f}  burn={s['burn_fraction']*100:.2f}%")

    render_figure(selected, model, dataset, device, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())