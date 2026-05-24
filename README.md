# Sentinel-2 Burned Area Segmentation

Deep learning segmentation of burned areas from Sentinel-2 satellite imagery using PyTorch.

**Status:** In active development

## Overview

This project trains a U-Net model to identify burn scars in multispectral Sentinel-2 imagery. The goal is to demonstrate an end-to-end remote-sensing ML workflow: from raw satellite data to a trained, evaluated, and reproducible segmentation model.

## Roadmap

- [x] Environment and project setup
- [x] Dataset acquisition and exploration
- [ ] Geographic train/val/test split
- [ ] Baseline U-Net training
- [ ] Evaluation and visualization
- [ ] Model improvements (loss tuning, augmentation, SWIR bands)
- [ ] Documentation and reproducibility

## Tech Stack

- **Language:** Python 3.11
- **Package manager:** [uv](https://github.com/astral-sh/uv)
- **Deep learning:** PyTorch
- **Geospatial:** rasterio, GDAL
- **Experiment tracking:** Weights & Biases

## Author

**Chavosh Almassian** | Remote Sensing & Geoinformatics (M.Sc., KIT)
[GitHub](https://github.com/Chavoshh) · [LinkedIn](https://www.linkedin.com/in/chavosh-almassian-81a05216a/)

---

*This is a portfolio project demonstrating GeoAI / remote-sensing ML capabilities.*