#!/usr/bin/env python3
"""
Sweep multiple SimCLR checkpoints through COIL-20 / COIL-100 linear evaluation.

Usage examples
--------------
# Evaluate every checkpoint in a directory:
python eval_coil_sweep.py \
    --checkpoint-dir ./checkpoints/ws5_fs2_wstr10 \
    --coil-dir ./data/coil-20 \
    --output-csv ./results/coil_sweep.csv

# Evaluate specific checkpoint files:
python eval_coil_sweep.py \
    --checkpoints ./checkpoints/run_a/simclr_epoch_010.pth \
                  ./checkpoints/run_a/simclr_epoch_050.pth \
                  ./checkpoints/run_a/simclr_final.pth \
    --coil-dir ./data/coil-20 \
    --output-csv ./results/coil_sweep.csv

# Include a random-init baseline (no checkpoint loaded):
python eval_coil_sweep.py \
    --checkpoint-dir ./checkpoints/ws5_fs2_wstr10 \
    --coil-dir ./data/coil-20 \
    --random-init-baseline \
    --output-csv ./results/coil_sweep.csv
"""

import argparse
import csv
import os
from typing import List, Optional

import torch
import torch.nn as nn

from evaluation.coil_linear_eval import linear_eval_on_coil
from models.simclr_resnet import SimCLRResNet18


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def resolve_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """Return all .pth files in checkpoint_dir sorted by epoch number."""
    files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth")
    ]

    def _sort_key(path: str) -> int:
        base = os.path.basename(path)
        # epoch checkpoints: simclr_epoch_NNN.pth
        if "epoch_" in base:
            try:
                return int(base.split("epoch_")[1].replace(".pth", ""))
            except (IndexError, ValueError):
                pass
        # final checkpoint goes last
        if "final" in base:
            return 10**9
        return -1

    return sorted(files, key=_sort_key)


def load_encoder(
    checkpoint_path: Optional[str],
    device: torch.device,
    proj_dim: int = 128,
) -> tuple:
    """
    Load SimCLRResNet18 from a checkpoint file.
    Returns (encoder, feature_dim, label) where label is a short string
    identifying the checkpoint.

    If checkpoint_path is None, returns a randomly initialised encoder
    (used as baseline).
    """
    model = SimCLRResNet18(proj_dim=proj_dim)

    if checkpoint_path is None:
        label = "random_init"
        print("[sweep] Loading random-init baseline (no checkpoint).")
    else:
        label = os.path.basename(checkpoint_path).replace(".pth", "")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Support both plain state dicts and our full checkpoint format.
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # Recover proj_dim saved in checkpoint if present
            proj_dim = ckpt.get("proj_dim", proj_dim)
            # Re-create model with correct proj_dim if needed
            if proj_dim != model.projection_head[-1].out_features:
                model = SimCLRResNet18(proj_dim=proj_dim)
            model.load_state_dict(state_dict)
            epoch = ckpt.get("epoch", "?")
            print(f"[sweep] Loaded checkpoint: {label} (epoch={epoch})")
        else:
            # Assume raw state dict
            model.load_state_dict(ckpt)
            print(f"[sweep] Loaded raw state dict: {label}")

    model = model.to(device)
    return model.encoder, model.feature_dim, label


def print_results_table(rows: List[dict]) -> None:
    """Pretty-print a comparison table to stdout."""
    if not rows:
        return
    header = ["checkpoint", "train_acc", "val_acc", "test_acc", "num_classes"]
    col_w = [max(len(h), max(len(str(r.get(h, ""))) for r in rows)) + 2 for h in header]

    def fmt_row(values):
        return "  ".join(str(v).ljust(w) for v, w in zip(values, col_w))

    sep = "-" * sum(c + 2 for c in col_w)
    print("\n" + sep)
    print(fmt_row(header))
    print(sep)
    for r in rows:
        print(fmt_row([
            r.get("checkpoint", ""),
            f"{r.get('train_acc', 0):.4f}",
            f"{r.get('val_acc', 0):.4f}",
            f"{r.get('test_acc', 0):.4f}",
            r.get("num_classes", ""),
        ]))
    print(sep + "\n")


def save_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] Results saved to {path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep SimCLR checkpoints through COIL linear evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint source (one of the two must be provided)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to scan for all .pth checkpoint files.",
    )
    src.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="PATH",
        help="Explicit list of checkpoint file paths to evaluate.",
    )

    # COIL dataset
    parser.add_argument(
        "--coil-dir",
        type=str,
        required=True,
        help="Root directory of the COIL dataset (flat or ImageFolder layout).",
    )

    # Linear eval settings
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-epochs", type=int, default=20,
                        help="Epochs to train the linear head.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="SGD learning rate for the linear head.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.70,
                        help="Fraction of COIL data used for linear head training.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction used for validation (remainder = test).")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches per split (useful for quick checks).")
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection head output dim (must match checkpoint).")

    # Baseline
    parser.add_argument(
        "--random-init-baseline",
        action="store_true",
        help="Also evaluate a randomly initialised encoder as a lower-bound baseline.",
    )

    # Output
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save results CSV. If omitted, results are only printed.",
    )

    # Misc
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda | mps | cpu | auto")

    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[sweep] Device: {device}")

    # Collect checkpoint paths
    if args.checkpoint_dir is not None:
        checkpoint_paths = find_checkpoints(args.checkpoint_dir)
        if not checkpoint_paths:
            raise RuntimeError(
                f"No .pth files found in {args.checkpoint_dir}"
            )
        print(f"[sweep] Found {len(checkpoint_paths)} checkpoint(s) in {args.checkpoint_dir}")
    else:
        checkpoint_paths = args.checkpoints

    # Optionally prepend a random-init baseline (None = no checkpoint loaded)
    if args.random_init_baseline:
        checkpoint_paths = [None] + list(checkpoint_paths)

    rows = []
    for ckpt_path in checkpoint_paths:
        encoder, feature_dim, label = load_encoder(
            checkpoint_path=ckpt_path,
            device=device,
            proj_dim=args.proj_dim,
        )

        metrics = linear_eval_on_coil(
            encoder=encoder,
            feature_dim=feature_dim,
            device=device,
            coil_dir=args.coil_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_epochs=args.eval_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            max_batches=args.max_batches,
        )

        row = {"checkpoint": label, **metrics}
        rows.append(row)

        print(
            f"  [{label}]  train={metrics['train_acc']:.4f}  "
            f"val={metrics['val_acc']:.4f}  "
            f"test={metrics['test_acc']:.4f}  "
            f"classes={metrics['num_classes']}"
        )

    print_results_table(rows)

    if args.output_csv:
        save_csv(rows, args.output_csv)


if __name__ == "__main__":
    main()
