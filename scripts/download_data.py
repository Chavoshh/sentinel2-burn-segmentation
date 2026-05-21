"""
Download the CaBuAr (California Burned Areas) dataset from Hugging Face.

The dataset provides Sentinel-2 pre/post-fire imagery with binary burn masks
for California wildfires (2015-2022), organized into 5 cross-validation folds.

Reference:
    Cambrin, D. R., Colomba, L., & Garza, P. (2023).
    CaBuAr: California Burned Areas dataset for delineation.
    IEEE Geoscience and Remote Sensing Magazine.
    Dataset: https://huggingface.co/datasets/DarthReca/california_burned_areas
    License: CC-BY-NC 4.0

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --output-dir data/raw --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# --- Configuration -----------------------------------------------------------

REPO_ID = "DarthReca/california_burned_areas"
REPO_TYPE = "dataset"

# We download only the patched, post-fire HDF5 files (smaller and pre-tiled).
# The 'raw' folder contains the full 5490x5490 scenes; we skip those for now.
# The CaBuAr repo offers raw and normalized versions, at full or patched resolution.
# We download the RAW patched version (recommended by the authors) so we can apply
# our own normalization. The repo path 'normalized/pacthed/' has an upstream typo
# we do not use; the path we want is 'raw/patched/'.
ALLOW_PATTERNS = [
    "raw/patched/512x512.hdf5",     # main training data (post-fire + masks, 512x512)
    "raw/patched/chabud_test.h5",   # ChaBuD challenge test set (for benchmarking)
    "metadata.parquet",             # scene-level metadata: coordinates, CRS, timestamps
    "README.md",                    # dataset documentation
]

DEFAULT_OUTPUT_DIR = Path("data/raw/cabuar")


# --- Logging setup -----------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger with a clean format for CLI scripts."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("download_data")


# --- Main download logic -----------------------------------------------------

def download_cabuar(
    output_dir: Path,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Download the CaBuAr dataset (patched, post-fire HDF5 files only).

    Args:
        output_dir: Where to store the downloaded files.
        dry_run: If True, only print what would be downloaded.
        logger: Optional logger; one is created if not provided.

    Returns:
        Path to the directory containing the downloaded files.
    """
    log = logger or setup_logging()

    log.info("Repository:    %s (%s)", REPO_ID, REPO_TYPE)
    log.info("Output dir:    %s", output_dir.resolve())
    log.info("Patterns:      %s", ALLOW_PATTERNS)
    log.info("Dry run:       %s", dry_run)

    if dry_run:
        log.info("[DRY RUN] Skipping actual download. Done.")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting download... (this may take a few minutes)")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(output_dir),
        allow_patterns=ALLOW_PATTERNS,
    )

    log.info("Download complete. Local path: %s", local_path)
    return Path(local_path)


def summarize_download(output_dir: Path, logger: logging.Logger) -> None:
    """Print a small summary of what was downloaded."""
    hdf5_files = sorted(list(output_dir.rglob("*.hdf5")) + list(output_dir.rglob("*.h5")))
    if not hdf5_files:
        logger.warning("No HDF5 files found in %s", output_dir)
        return

    total_bytes = sum(f.stat().st_size for f in hdf5_files)
    total_mb = total_bytes / (1024 * 1024)

    logger.info("---- Summary ----")
    logger.info("HDF5 files:    %d", len(hdf5_files))
    logger.info("Total size:    %.1f MB", total_mb)
    for f in hdf5_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info("  %s  (%.1f MB)", f.relative_to(output_dir), size_mb)


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the CaBuAr dataset from Hugging Face."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to store downloaded files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without doing it.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(verbose=args.verbose)

    try:
        local_path = download_cabuar(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            logger=logger,
        )
        if not args.dry_run:
            summarize_download(local_path, logger)
    except Exception as exc:
        logger.exception("Download failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())