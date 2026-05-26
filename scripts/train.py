"""Entry point for training the burned-area segmentation model.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/baseline.yaml
    uv run python scripts/train.py --config configs/baseline.yaml --epochs 5
    uv run python scripts/train.py --epochs 3 --run-name smoke_test --tags smoke quick
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

import yaml

# Make 'src' importable when this script is run from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import TrainConfig, train  # noqa: E402


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_yaml_config(path: Path) -> dict:
    """Load a YAML config file. Returns an empty dict if path is None."""
    if path is None:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the burn segmentation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (e.g. configs/baseline.yaml). "
             "Any fields here override TrainConfig defaults; CLI flags override the YAML.",
    )

    # CLI overrides for the fields you'd most often want to tweak.
    # Anything else can be edited in the YAML.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None, dest="run_name")
    parser.add_argument("--tags", type=str, nargs="*", default=None,
                        help="Tags for the W&B run, space-separated.")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Compose a TrainConfig from defaults + YAML + CLI overrides (in that order)."""
    yaml_data = load_yaml_config(args.config)

    # Start from defaults
    config_kwargs: dict = {}

    # Apply YAML values, but only for fields TrainConfig actually has
    valid_fields = {f.name for f in fields(TrainConfig)}
    for key, value in yaml_data.items():
        if key not in valid_fields:
            logging.warning(f"Unknown config key in YAML: {key!r} (ignored)")
            continue
        # Convert path-like strings to Path objects
        if key in {"patches_dir", "splits_csv", "checkpoints_dir"} and value is not None:
            value = Path(value)
        config_kwargs[key] = value

    # Apply CLI overrides (only non-None values)
    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "run_name": args.run_name,
        "tags": args.tags,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config_kwargs[key] = value

    return TrainConfig(**config_kwargs)


def main() -> int:
    setup_logging()
    args = parse_args()

    config = build_config(args)
    logging.info(f"Loaded config from: {args.config}")
    logging.info(f"Run name:           {config.run_name or '(auto)'}")
    logging.info(f"Epochs:             {config.epochs}")
    logging.info(f"Tags:               {config.tags}")

    try:
        final_metrics = train(config)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user.")
        return 130   # standard exit code for SIGINT
    except Exception:
        logging.exception("Training failed.")
        return 1

    logging.info("Final metrics:")
    for key in sorted(final_metrics):
        logging.info(f"  {key}: {final_metrics[key]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())