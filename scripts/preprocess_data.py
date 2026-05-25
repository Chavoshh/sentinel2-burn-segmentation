"""Pre-process the CaBuAr HDF5 file into per-patch .npy caches.

Why: HDF5 + BZip2 decompression is the data-loading bottleneck (~2.7 img/s).
Caching to uncompressed .npy gets us 10-20x faster training data loading.

Usage:
    uv run python scripts/preprocess_data.py
    uv run python scripts/preprocess_data.py --output-dir data/processed/patches --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers BZip2 codec
import numpy as np
from tqdm import tqdm

# Project root so we can resolve paths regardless of where the script is invoked
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DEFAULT_BANDS  # noqa: E402

DEFAULT_HDF5 = PROJECT_ROOT / "data" / "raw" / "cabuar" / "raw" / "patched" / "512x512.hdf5"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "patches"
DEFAULT_STATS = PROJECT_ROOT / "data" / "processed" / "band_stats.json"


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("preprocess")


def load_p99(stats_path: Path, bands: tuple[int, ...]) -> np.ndarray:
    """Load per-band p99 normalization constants for the selected bands."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    idx_to_p99 = {info["band_idx"]: info["p99"] for info in stats["stats"].values()}
    return np.array([idx_to_p99[b] for b in bands], dtype=np.float32)


def process_one_sample(
    hdf5_handle: h5py.File,
    sample_name: str,
    bands: tuple[int, ...],
    p99: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one sample, select bands, normalize, return (image, mask)."""
    sample = hdf5_handle[sample_name]
    post_fire = sample["post_fire"][:]                   # (512, 512, 12), uint16
    mask = np.squeeze(sample["mask"][:]).astype(np.uint8)  # (512, 512), uint8

    image = post_fire[..., list(bands)].astype(np.float32)   # (512, 512, 6), float32
    image = np.clip(image, 0.0, p99)
    image = image / p99                                       # → [0, 1]
    # Transpose to (C, H, W) so the loader doesn't have to
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # (6, 512, 512)

    return image, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--bands", nargs="+", type=int, default=list(DEFAULT_BANDS),
                        help="Band indices to extract (default: 6 bands)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done, but don't write files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files (default: skip)")
    args = parser.parse_args()

    log = setup_logging()
    bands = tuple(args.bands)

    log.info("HDF5 input:    %s", args.hdf5)
    log.info("Output dir:    %s", args.output_dir)
    log.info("Stats file:    %s", args.stats)
    log.info("Bands:         %s", bands)
    log.info("Dry run:       %s", args.dry_run)

    if not args.hdf5.exists():
        log.error("HDF5 file not found: %s", args.hdf5)
        return 1
    if not args.stats.exists():
        log.error("Band stats not found: %s (run the data exploration notebook first)", args.stats)
        return 1

    p99 = load_p99(args.stats, bands)
    log.info("Normalization p99 per band: %s", p99.round(0).astype(int).tolist())

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_failed = 0
    total_bytes = 0

    with h5py.File(args.hdf5, "r") as f:
        sample_names = list(f.keys())
        log.info("Processing %d samples...", len(sample_names))

        for name in tqdm(sample_names, disable=args.dry_run):
            img_path = args.output_dir / f"{name}_img.npy"
            mask_path = args.output_dir / f"{name}_mask.npy"

            if not args.force and img_path.exists() and mask_path.exists():
                n_skipped += 1
                continue

            try:
                image, mask = process_one_sample(f, name, bands, p99)
            except Exception as e:  # noqa: BLE001
                log.exception("Failed on %s: %s", name, e)
                n_failed += 1
                continue

            if not args.dry_run:
                np.save(img_path, image)
                np.save(mask_path, mask)
                total_bytes += img_path.stat().st_size + mask_path.stat().st_size
                n_written += 1

    log.info("---- Summary ----")
    log.info("Written:    %d", n_written)
    log.info("Skipped:    %d (already existed; use --force to overwrite)", n_skipped)
    log.info("Failed:     %d", n_failed)
    if total_bytes:
        log.info("Total size: %.1f MB", total_bytes / (1024 * 1024))

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())