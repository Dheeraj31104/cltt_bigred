CLTT BigRed Project

Overview
- Train SimCLR on egocentric frame windows with a Slurm-friendly CLI.
- Includes a dataset helper for sliding windows and optional paired-window views.
- Optional object-focus transform blurs backgrounds using dummy bounding boxes.

Repository Layout
- train_simclr.py: main training script (CLI, checkpoints, wandb, resume)
- utils/EgocentricWindowDataset_new.py: dataset for frame windows and paired views
- slurm/run_simclr_array.sh: Slurm job array template
- requirements.txt: Python dependencies

Setup
- Create and activate a Python environment.
- Install dependencies:
  pip install -r requirements.txt
- Ensure you have a directory of extracted frames, organized by clip:
  root_dir/
    clip_001/
      frame_0001.jpg
      frame_0002.jpg
      ...
    clip_002/
      ...

Usage
- Local run (single node):
  python train_simclr.py \
    --root-dir /path/to/frames \
    --window-size 5 \
    --frame-step 10 \
    --batch-size 64 \
    --epochs 25 \
    --checkpoint-dir checkpoints/run1

- Paired-window mode (two views offset in time):
  python train_simclr.py \
    --root-dir /path/to/frames \
    --window-size 5 \
    --frame-step 10 \
    --two-view-offset 1 \
    --batch-size 64 \
    --epochs 25

- Enable object-focus (dummy boxes):
  python train_simclr.py --use-object-focus ...

- Slurm job array:
  Edit slurm/run_simclr_array.sh and submit with:
  sbatch slurm/run_simclr_array.sh

Notes
- Dataset windows:
  - Single-view mode returns a stacked tensor [W, C, H, W].
  - Two-view mode returns (view0, view1) with each view stacked.
- Object-focus currently uses a central dummy box; swap get_bboxes_dummy()
  with real detector outputs for true object regions.
- Checkpoints are saved as simclr_epoch_XXX.pth and simclr_final.pth.
- Resume behavior:
  --resume loads the latest checkpoint in --checkpoint-dir.
  --checkpoint-path loads a specific checkpoint file.
- Wandb is optional; enable with --wandb and related flags.
