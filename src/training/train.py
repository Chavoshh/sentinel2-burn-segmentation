"""End-to-end training for the burned-area segmentation model.

This module exposes a single `train(config)` function that runs a full experiment:
- Builds model, loss, optimizer, scheduler
- Trains for N epochs with W&B logging
- Saves checkpoints (best by val IoU, plus latest)
- Logs sample predictions visually every K epochs

Designed to be called from scripts/train.py or directly from a notebook.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from src.data.dataset import BurnDataset
from src.models.unet import build_unet, count_parameters
from src.training.losses import BCEDiceLoss
from src.training.metrics import build_metrics, logits_to_probs

logger = logging.getLogger(__name__)


# --- Config ------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters and settings for a training run.

    Keep this dataclass small and stable; new flags should have sensible defaults
    so old configs keep working.
    """

    # Data
    patches_dir: Path = Path("data/processed/patches")
    splits_csv: Path = Path("data/processed/splits.csv")
    batch_size: int = 4
    num_workers: int = 0   # measured fastest on Windows; see Step 7 benchmark

    # Model
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 6

    # Optimization
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Scheduler (cosine annealing)
    use_scheduler: bool = True
    min_lr: float = 1e-6

    # Logging / checkpointing
    wandb_entity: str = "chavosh-personal"
    wandb_project: str = "sentinel2-burn-segmentation"
    run_name: str | None = None  # if None, W&B auto-generates one
    log_predictions_every: int = 5     # log sample predictions every N epochs
    n_pred_samples: int = 4            # how many val samples to visualize
    checkpoints_dir: Path = Path("models")

    # Reproducibility
    seed: int = 42

    # Tags help filter runs in the W&B UI
    tags: list[str] = field(default_factory=lambda: ["baseline"])


# --- Utility -----------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, numpy, and PyTorch for reproducibility.

    Note: full determinism on CUDA also requires
    `torch.backends.cudnn.deterministic = True`, which costs speed. We don't
    enable that here; the seed is enough for "approximately reproducible" runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = BurnDataset(
        patches_dir=config.patches_dir,
        splits_csv=config.splits_csv,
        split="train",
    )
    val_ds = BurnDataset(
        patches_dir=config.patches_dir,
        splits_csv=config.splits_csv,
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,   # drop the last partial batch for cleaner training stats
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# --- Per-epoch loops ---------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    metrics: MetricCollection,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """One epoch of training. Returns a dict of mean training metrics."""
    model.train()
    metrics.reset()

    epoch_loss = 0.0
    epoch_bce = 0.0
    epoch_dice = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1)   # (B, 1, H, W)

        optimizer.zero_grad()
        logits = model(images)
        losses = loss_fn(logits, masks)
        losses["loss"].backward()
        optimizer.step()

        # Accumulate losses
        epoch_loss += losses["loss"].item()
        epoch_bce += losses["loss_bce"].item()
        epoch_dice += losses["loss_dice"].item()
        n_batches += 1

        # Update metrics (use no_grad to avoid leaking compute into the graph)
        with torch.no_grad():
            probs = logits_to_probs(logits)
            metrics.update(probs, masks.int())

        pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

    epoch_results = {k: v.item() for k, v in metrics.compute().items()}
    epoch_results["train/loss"] = epoch_loss / n_batches
    epoch_results["train/loss_bce"] = epoch_bce / n_batches
    epoch_results["train/loss_dice"] = epoch_dice / n_batches
    return epoch_results


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: BCEDiceLoss,
    metrics: MetricCollection,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """One full pass through validation. Returns a dict of mean val metrics."""
    model.eval()
    metrics.reset()

    epoch_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1)

        logits = model(images)
        losses = loss_fn(logits, masks)

        epoch_loss += losses["loss"].item()
        n_batches += 1

        probs = logits_to_probs(logits)
        metrics.update(probs, masks.int())

    epoch_results = {k: v.item() for k, v in metrics.compute().items()}
    epoch_results["val/loss"] = epoch_loss / n_batches
    return epoch_results


# --- Visual prediction logging -----------------------------------------------

@torch.no_grad()
def log_sample_predictions(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    n_samples: int,
) -> None:
    """Log a few prediction visualizations to W&B for visual progress tracking."""
    model.eval()

    # Grab one batch from the val loader (deterministic since shuffle=False)
    images, masks = next(iter(val_loader))
    images = images[:n_samples].to(device)
    masks = masks[:n_samples].cpu().numpy()

    logits = model(images)
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()   # (N, H, W)
    preds = (probs > 0.5).astype(np.uint8)

    # Recover a displayable RGB by selecting the first 3 channels (Blue, Green, Red)
    # of the input and reordering to (Red, Green, Blue) for human viewing.
    images_cpu = images.cpu().numpy()  # (N, 6, H, W)
    rgb = images_cpu[:, [2, 1, 0], :, :]  # take Red(idx 2), Green(idx 1), Blue(idx 0)
    rgb = np.transpose(rgb, (0, 2, 3, 1))   # (N, H, W, 3)
    rgb = np.clip(rgb, 0, 1)                # already normalized, but be safe

    # Build a wandb.Image for each sample with class masks overlaid
    wb_images = []
    class_labels = {0: "no-burn", 1: "burn"}
    for i in range(n_samples):
        wb_images.append(
            wandb.Image(
                rgb[i],
                masks={
                    "prediction":   {"mask_data": preds[i],         "class_labels": class_labels},
                    "ground_truth": {"mask_data": masks[i].astype(np.uint8), "class_labels": class_labels},
                },
            )
        )

    wandb.log({"val/predictions": wb_images, "epoch": epoch})


# --- Checkpointing -----------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, float],
    config: TrainConfig,
    path: Path,
) -> None:
    """Save a checkpoint including model state, optimizer state, and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": asdict(config),
        },
        path,
    )


# --- Main entry point --------------------------------------------------------

def train(config: TrainConfig) -> dict[str, float]:
    """Run a full training experiment."""
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU:    {torch.cuda.get_device_name(0)}")

    # --- W&B init ---
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.run_name,
        tags=config.tags,
        config=asdict(config),
    )
    logger.info(f"W&B run: {run.url}")

    # --- Build everything ---
    train_loader, val_loader = build_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = build_unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
    ).to(device)
    trainable, _ = count_parameters(model)
    logger.info(f"Model: U-Net + {config.encoder} ({trainable:,} trainable params)")

    loss_fn = BCEDiceLoss(
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr,
        )
        if config.use_scheduler else None
    )

    train_metrics = build_metrics(prefix="train/").to(device)
    val_metrics = build_metrics(prefix="val/").to(device)

    # --- Checkpoint paths ---
    run_dir = config.checkpoints_dir / run.id
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    # --- Main training loop ---
    best_val_iou = -1.0
    for epoch in range(1, config.epochs + 1):
        train_results = train_one_epoch(
            model, train_loader, loss_fn, optimizer, train_metrics, device, epoch,
        )
        val_results = validate(
            model, val_loader, loss_fn, val_metrics, device, epoch,
        )

        # Console summary
        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss {train_results['train/loss']:.4f}  "
            f"val_loss {val_results['val/loss']:.4f}  "
            f"val_iou {val_results['val/iou']:.4f}  "
            f"val_f1 {val_results['val/f1']:.4f}"
        )

        # W&B logging
        log_data = {**train_results, **val_results, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        wandb.log(log_data)

        # Visual predictions every N epochs (and at epoch 1 and last epoch)
        if epoch == 1 or epoch == config.epochs or epoch % config.log_predictions_every == 0:
            log_sample_predictions(model, val_loader, device, epoch, config.n_pred_samples)

        # Save 'last' checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, val_results, config, last_path)

        # Save 'best' if val IoU improved
        if val_results["val/iou"] > best_val_iou:
            best_val_iou = val_results["val/iou"]
            save_checkpoint(model, optimizer, epoch, val_results, config, best_path)
            wandb.run.summary["best_val_iou"] = best_val_iou
            wandb.run.summary["best_epoch"] = epoch
            logger.info(f"  → New best val IoU: {best_val_iou:.4f}, saved {best_path.name}")

        if scheduler is not None:
            scheduler.step()

    logger.info(f"Training complete. Best val IoU: {best_val_iou:.4f}")
    logger.info(f"Checkpoints saved to: {run_dir}")
    wandb.finish()
    return val_results