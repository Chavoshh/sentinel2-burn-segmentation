"""U-Net model factory for burned-area segmentation.

We use segmentation-models-pytorch (smp) for the architecture. smp provides
production-quality U-Net implementations with swappable encoders, ImageNet
pretraining support, and automatic handling of non-RGB input channels.

Why a factory function instead of a class:
- We want to swap encoders cheaply (e.g. ResNet-34 vs MobileNet for comparison)
- smp's UnetEncoder/UnetDecoder are already well-tested; no reason to subclass
- A simple function call is easier to call from configs and CLI scripts
"""

from __future__ import annotations

from typing import Literal

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# Encoder presets we've validated for this project.
# When adding a new encoder, run the smoke test in this file to confirm it fits in VRAM.
SUPPORTED_ENCODERS = (
    "resnet18",       # lightest, ~11M params
    "resnet34",       # default, ~21M params
    "resnet50",       # may OOM on 4GB VRAM
    "efficientnet-b0",
    "mobilenet_v2",
)


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: Literal["imagenet"] | None = "imagenet",
    in_channels: int = 6,
    classes: int = 1,
) -> nn.Module:
    """Build a U-Net segmentation model.

    Args:
        encoder_name: One of SUPPORTED_ENCODERS. ResNet-34 is the default baseline.
        encoder_weights: "imagenet" for pretrained encoder, None for random init.
        in_channels: Number of input bands. Default 6 = our (RGB + NIR + SWIR-1 + SWIR-2).
                     smp handles non-3 channel inputs by adapting the first conv layer:
                     RGB weights copied for channels 0-2, averaged for channels 3+.
        classes: Number of output classes. 1 for binary segmentation (burn / no-burn).

    Returns:
        An nn.Module that takes (B, C, H, W) → (B, classes, H, W). No final activation;
        use BCEWithLogitsLoss (binary) or CrossEntropyLoss (multi-class).
    """
    if encoder_name not in SUPPORTED_ENCODERS:
        raise ValueError(
            f"Unsupported encoder: {encoder_name!r}. "
            f"Choose from {SUPPORTED_ENCODERS}."
        )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Smoke test: build the model, run one forward pass, print param count.
    model = build_unet()
    trainable, total = count_parameters(model)

    x = torch.randn(2, 6, 512, 512)
    with torch.no_grad():
        y = model(x)

    print(f"Model:               {type(model).__name__} with {model.encoder._get_name()} encoder")
    print(f"Input shape:         {tuple(x.shape)}")
    print(f"Output shape:        {tuple(y.shape)}")
    print(f"Output range:        [{y.min():.3f}, {y.max():.3f}]  (raw logits, no sigmoid)")
    print(f"Trainable params:    {trainable:,}")
    print(f"Total params:        {total:,}")