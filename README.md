Contrastive Learning Through Time for Egocentric Windows (WIP)
Overview

CLTT  trains a SimCLR-style contrastive model on egocentric video frame windows, leveraging temporal adjacency to form positive pairs (“contrastive learning through time”) alongside standard augmentations. The codebase is designed to be HPC/Slurm-friendly and supports reproducible experiments via a CLI, checkpointing, and optional Weights & Biases logging.

Key Features

Sliding-window dataset for egocentric frames (single-view and paired-view modes)

Paired-window temporal positives via configurable time offsets (--two-view-offset)

Reproducible training: checkpoints, resume support, configurable runs

Slurm job-array template for sweeps on HPC clusters

Optional object-focus transform that blurs backgrounds using placeholder bounding boxes (swap-in real detector outputs)

Repository Structure

train_simclr.py — main training script (CLI, checkpoints, wandb, resume)

utils/EgocentricWindowDataset_new.py — dataset for frame windows and paired temporal views

slurm/run_simclr_array.sh — Slurm job array template

requirements.txt — Python dependencies

Setup
1) Create and activate a Python environment
python -m venv .venv
source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt

3) Data format (extracted frames)

Organize extracted frames by clip:

root_dir/
  clip_001/
    frame_0001.jpg
    frame_0002.jpg
    ...
  clip_002/
    frame_0001.jpg
    ...

Usage
Local run (single node)
python train_simclr.py \
  --root-dir /path/to/frames \
  --window-size 5 \
  --frame-step 10 \
  --batch-size 64 \
  --epochs 25 \
  --checkpoint-dir checkpoints/run1

Paired-window mode (temporal positives with offset)

Creates two temporally offset views of a window: (view0, view1).

python train_simclr.py \
  --root-dir /path/to/frames \
  --window-size 5 \
  --frame-step 10 \
  --two-view-offset 1 \
  --batch-size 64 \
  --epochs 25

Enable object-focus mode (placeholder boxes)
python train_simclr.py \
  --root-dir /path/to/frames \
  --use-object-focus \
  ...

Slurm job array (HPC)

Edit slurm/run_simclr_array.sh (paths, resources, hyperparams)

Submit:

sbatch slurm/run_simclr_array.sh

Implementation Notes
Dataset windows

Single-view mode returns a stacked tensor shaped like [W, C, H, W] (W = window size)

Two-view mode returns (view0, view1) where each view is a stacked window

Object-focus transform

Current implementation uses a central dummy box (get_bboxes_dummy()).

Replace dummy boxes with real detector outputs to study object-centric temporal consistency.

Checkpointing

Checkpoints are saved as simclr_epoch_XXX.pth and simclr_final.pth.

Resume options:

--resume loads the latest checkpoint from --checkpoint-dir

--checkpoint-path loads a specific checkpoint file

W&B logging (optional)

Enable logging with --wandb and related flags.

🚧 Work in Progress

Status: This project is actively under development. Results and APIs may change as experiments evolve.

Planned improvements:

Replace dummy bounding boxes with detector-based object regions

Add systematic evaluation (e.g., linear probing / retrieval) and reporting

Expand ablations (augmentations, offsets, temperature, batch size, window size)

📚 Cite / References

If you use this repository for academic work, please cite:

SimCLR: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.

CLTT / Contrastive Learning Through Time: (Add the specific paper citation you are following here.)

BibTeX
@inproceedings{chen2020simclr,
  title     = {A Simple Framework for Contrastive Learning of Visual Representations},
  author    = {Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@misc{karanam2026clttbigred,
  title        = {CLTT BigRed: Contrastive Learning Through Time for Egocentric Windows (Work in Progress)},
  author       = {Dheeraj Karanam},
  year         = {2026},
  howpublished = {\\url{https://github.com/Dheeraj31104/cltt_bigred}},
  note         = {Work in progress}
}

Quick tip (so GitHub renders it nicely)

Make sure the file name is exactly README.md and the content is not indented with 4 leading spaces on every line (that turns it into a code block).

