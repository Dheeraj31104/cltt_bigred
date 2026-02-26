# Temporal Dataset + Loss Notes

Last updated: February 25, 2026

## 1) Current Dataset Design (Single-Window Only)

File: `utils/EgocentricWindowDataset_new.py`

The dataset now uses only single-window temporal sampling.

Two temporal controls:

1. `frame_step`: gap (in frames) between consecutive frames inside one window.
2. `window_stride`: gap (in frames) between starts of consecutive windows.

Window indexing:

`window(t) = [t, t + frame_step, t + 2*frame_step, ..., t + (window_size-1)*frame_step]`

Next window starts at:

`t_next = t + window_stride`

Notes:

1. `window_stride=1` gives dense overlapping windows.
2. Larger `window_stride` reduces overlap between neighboring windows.
3. macOS sidecar files like `._*.jpg` are ignored during indexing.

## 2) Current Loss Design

File: `losses/paired_cosine_tt.py`
Class: `TemporalAllPairsTTLoss`

Input shape:

`z: [B, T, D]`

Meaning:

1. `B` = number of windows in batch
2. `T` = frames per window
3. `D` = embedding dim

Positive/negative definition:

1. Positives: all other frames from the same window.
2. Negatives: all frames from other windows.
3. Self-pair is excluded.

Loss type:

Multi-positive InfoNCE over temporal frames in each window.

## 3) Matrix Representation (Example B=2, T=3)

Flatten order:

`[a1, a2, a3, b1, b2, b3]`

Positive mask `P`:

```text
0 1 1 0 0 0
1 0 1 0 0 0
1 1 0 0 0 0
0 0 0 0 1 1
0 0 0 1 0 1
0 0 0 1 1 0
```

Final objective:

Average negative log-probability over entries where `P=1`.

## 4) Training CLI Controls

File: `train_simclr.py`

Main temporal controls:

1. `--window-size`
2. `--frame-step`
3. `--window-stride`

Example:

```bash
python train_simclr.py \
  --root-dir '/Volumes/Dheeraj Ext/cropped_images' \
  --window-size 5 \
  --frame-step 10 \
  --window-stride 30 \
  --batch-size 64 \
  --epochs 25
```

## 5) Visualization Script

File: `visualize_temporal_dataset.py`

Modes:

1. `--mode sample` for one window.
2. `--mode batch` for a batch grid.

Important batch sampling options:

1. `--batch-sampling contiguous` (neighboring windows; often visually similar)
2. `--batch-sampling random` (more diverse windows/clips)

Sample visualization:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
/opt/anaconda3/envs/ds/bin/python visualize_temporal_dataset.py \
  --mode sample \
  --root-dir '/Volumes/Dheeraj Ext/cropped_images' \
  --window-size 5 \
  --frame-step 10 \
  --window-stride 30 \
  --sample-idx 0 \
  --save-path outputs/temporal_window_0.png
```

Batch visualization (random, better diversity):

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp \
/opt/anaconda3/envs/ds/bin/python visualize_temporal_dataset.py \
  --mode batch \
  --root-dir '/Volumes/Dheeraj Ext/cropped_images' \
  --window-size 5 \
  --frame-step 10 \
  --window-stride 30 \
  --batch-size 16 \
  --batch-sampling random \
  --seed 7 \
  --save-path outputs/temporal_batch_random_16.png
```

## 6) Why Repeated Pictures Were Seen Earlier

With `contiguous` batch sampling and low `window_stride`, consecutive rows came from adjacent starts, so they overlapped heavily and looked repeated.

Ways to reduce visual repetition:

1. Increase `window_stride`.
2. Increase `frame_step`.
3. Use `--batch-sampling random` for inspection.

## 7) Slurm Settings

File: `slurm/run_simclr_array.sh`

Added dataset control:

`WINDOW_STRIDE=1`

This now passes to training via:

`--window-stride "$WINDOW_STRIDE"`

## 8) Environment Notes

In this workspace:

1. `/opt/anaconda3/envs/ds/bin/python` had both `torch` and `cv2`.
2. Default `python` had runtime issues importing `torch`.
3. Matplotlib cache permissions were handled with:
   `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp`

## 9) Update Cadence

I will append this file in future turns whenever we make meaningful dataset/loss/training/visualization changes.

Suggested cadence:

1. After each feature change.
2. After each bug fix.
3. After each workflow/command update.

## 10) Current Training Command (Agreed Config)

Use this for actual training with the current temporal setup:

```bash
/opt/anaconda3/envs/ds/bin/python /Users/dheerajkaranam/Projects/CLTT_new/cltt_bigred/train_simclr.py \
  --root-dir '/Volumes/Dheeraj Ext/cropped_images' \
  --window-size 5 \
  --frame-step 2 \
  --window-stride 10 \
  --batch-size 16 \
  --epochs 25 \
  --num-workers 4 \
  --checkpoint-dir /Users/dheerajkaranam/Projects/CLTT_new/cltt_bigred/checkpoints/ws5_fs2_wstr10 \
  --csv-log \
  --csv-log-path /Users/dheerajkaranam/Projects/CLTT_new/cltt_bigred/checkpoints/ws5_fs2_wstr10/metrics.csv
```

Reminder:

1. `--batch-sampling random` applies only to `visualize_temporal_dataset.py`, not `train_simclr.py`.

## 11) CSV Logging Format (Latest)

`train_simclr.py` now writes two CSV files when `--csv-log` is enabled.

Training CSV (epoch-level only):

Header:

`epoch,loss,learning_rate,decay_rate`

Meaning:

1. `epoch`: epoch index
2. `loss`: mean training loss for that epoch
3. `learning_rate`: scheduler LR at epoch end
4. `decay_rate`: `learning_rate / initial_lr`

Evaluation CSV (separate file):

Header:

`epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc`

Defaults:

1. If `--csv-log-path` is `/path/metrics.csv`, eval CSV defaults to `/path/metrics_eval.csv`.
2. You can override eval path with `--eval-csv-log-path`.
