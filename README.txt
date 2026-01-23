# CLTT BigRed — Contrastive Learning Through Time for Egocentric Windows (WIP)

## Overview
CLTT BigRed trains a **SimCLR-style contrastive model** on **egocentric video frame windows**, leveraging **temporal adjacency** to form positive pairs (“contrastive learning through time”) in addition to standard augmentations. The codebase is designed to be **HPC/Slurm-friendly** and supports reproducible experiments via a clean CLI, checkpointing, and optional Weights & Biases logging.

## Key Features
- **Sliding-window dataset** for egocentric frames (single-view and paired-view modes)
- **Paired-window temporal positives** via configurable time offsets (`--two-view-offset`)
- **Reproducible training**: checkpoints, resume support, configurable runs
- **Slurm job-array template** for sweeps on HPC clusters
- Optional **object-focus transform** that blurs backgrounds using placeholder bounding boxes (easy to swap for real detector outputs)

## Repository Structure
- `train_simclr.py` — main training script (CLI, checkpoints, wandb, resume)
- `utils/EgocentricWindowDataset_new.py` — dataset for frame windows and paired temporal views
- `slurm/run_simclr_array.sh` — Slurm job array template
- `requirements.txt` — Python dependencies

---

## Setup

### 1) Create and activate a Python environment
```bash
python -m venv .venv
source .venv/bin/activate
