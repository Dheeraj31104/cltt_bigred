# Contrastive Learning Through Time for Egocentric Windows (CLTT)

**Status:** Work in Progress 🚧

CLTT trains a SimCLR-style contrastive model on egocentric video **frame windows**, leveraging **temporal adjacency** to form positive pairs—a setting we refer to as *contrastive learning through time*. In addition to standard data augmentations, temporally offset windows are treated as positives to encourage temporal consistency in learned representations.

The codebase is designed to be **HPC/Slurm-friendly**, supports **reproducible experiments**, and provides a clean **CLI interface** with checkpointing and optional Weights & Biases logging.

---

## Key Features

* **Sliding-window egocentric dataset**

  * Supports single-view and paired-view (temporal positive) modes
* **Temporal contrastive pairs**

  * Configurable temporal offsets via `--two-view-offset`
* **Reproducible training**

  * Checkpointing, resume support, and configurable experiment runs
* **HPC-ready**

  * Slurm job-array template for large-scale sweeps
* **Optional object-focused augmentation**

  * Background blurring using placeholder bounding boxes (easily replaceable with real detector outputs)

---

## Repository Structure

```
.
├── train_simclr.py                 # Main training script (CLI, checkpoints, W&B, resume)
├── utils/
│   └── EgocentricWindowDataset_new.py  # Dataset for frame windows and temporal pairs
├── slurm/
│   └── run_simclr_array.sh          # Slurm job-array template
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Setup

### 1. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Data format (extracted frames)

Organize extracted frames by clip directory:

```
root_dir/
├── clip_001/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── clip_002/
│   ├── frame_0001.jpg
│   └── ...
└── ...
```

---

## Usage

### Local run (single node)

```bash
python train_simclr.py \
  --root-dir /path/to/frames \
  --window-size 5 \
  --frame-step 10 \
  --batch-size 64 \
  --epochs 25 \
  --checkpoint-dir checkpoints/run1
```

### Paired-window mode (temporal positives)

Creates two temporally offset views of the same window: `(view0, view1)`.

```bash
python train_simclr.py \
  --root-dir /path/to/frames \
  --window-size 5 \
  --frame-step 10 \
  --two-view-offset 1 \
  --batch-size 64 \
  --epochs 25
```

### Enable object-focus mode (placeholder bounding boxes)

```bash
python train_simclr.py \
  --root-dir /path/to/frames \
  --use-object-focus \
  ...
```

### Slurm job array (HPC)

1. Edit `slurm/run_simclr_array.sh` to configure paths, resources, and hyperparameters.
2. Submit the job array:

```bash
sbatch slurm/run_simclr_array.sh
```

---

## Implementation Notes

### Dataset windows

* **Single-view mode** returns a stacked tensor of shape:

  ```
  [W, C, H, W]
  ```

  where `W` is the temporal window size.

* **Two-view mode** returns a tuple `(view0, view1)`, where each element is a stacked temporal window.

### Object-focus transform

* The current implementation uses a **dummy central bounding box** via `get_bboxes_dummy()`.
* This is intended as a placeholder and can be replaced with real detector outputs to study **object-centric temporal consistency**.

### Checkpointing and resume

* Checkpoints are saved as:

  * `simclr_epoch_XXX.pth`
  * `simclr_final.pth`

Resume options:

* `--resume` loads the latest checkpoint from `--checkpoint-dir`
* `--checkpoint-path` loads a specific checkpoint file

### Weights & Biases logging (optional)

Enable experiment tracking with `--wandb` and related flags.

---

## Work in Progress

This project is under active development. APIs, results, and defaults may change as experiments evolve.

Planned improvements:

* Replace dummy bounding boxes with detector-based object regions
* Add systematic evaluation (e.g., linear probing, retrieval benchmarks)
* Expand ablation studies (augmentations, temporal offsets, temperature, batch size, window size)

---

## References

If you use this repository for academic work, please cite the following:

**SimCLR**
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML.

**Contrastive Learning Through Time (CLTT)**
[https://openreview.net/forum?id=HTCRs8taN8](https://openreview.net/forum?id=HTCRs8taN8)

### BibTeX

```bibtex
@inproceedings{chen2020simclr,
  title     = {A Simple Framework for Contrastive Learning of Visual Representations},
  author    = {Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2020}
}

@misc{karanam2026clttbigred,
  title        = {CLTT BigRed: Contrastive Learning Through Time for Egocentric Windows},
  author       = {Dheeraj Karanam},
  year         = {2026},
  howpublished = {\url{https://github.com/Dheeraj31104/cltt_bigred}},
  note         = {Work in progress}
}
```
