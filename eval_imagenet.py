#!/usr/bin/env python3
"""Standalone Tiny ImageNet linear evaluation on saved SimCLR checkpoints.

Two modes:
  --checkpoint   : evaluate a single checkpoint
  --checkpoint-dir : sweep all simclr_epoch_*.pth files in a directory,
                     evaluating each encoder epoch and writing a CSV summary
"""

import argparse
import csv
import os
import re

import torch

from evaluation.imagenet_linear_eval import linear_eval_on_imagenet
from models.simclr_resnet import SimCLRResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear eval on Tiny ImageNet from SimCLR checkpoint(s).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to a single .pth checkpoint file.")
    group.add_argument("--checkpoint-dir", type=str, help="Directory of simclr_epoch_*.pth files to sweep.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to tiny-imagenet-200 (must have train/ and val/).")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (64 for Tiny ImageNet).")
    parser.add_argument("--eval-epochs", type=int, default=30, help="Epochs to train the linear head per encoder checkpoint.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Cap batches per pass (for quick testing).")
    parser.add_argument("--out-csv", type=str, default=None, help="Path to write sweep results CSV (sweep mode only).")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device)
    proj_dim = ckpt.get("proj_dim", 128)
    model = SimCLRResNet18(proj_dim=proj_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", -1)
    return model, epoch


def run_eval(model, epoch, args, device) -> dict:
    metrics = linear_eval_on_imagenet(
        encoder=model.encoder,
        feature_dim=model.feature_dim,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_epochs=args.eval_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_dir=args.data_dir,
        train_fraction=args.train_fraction,
        max_batches=args.max_batches,
    )
    metrics["encoder_epoch"] = epoch
    return metrics


def print_results(metrics: dict) -> None:
    ep = metrics.get("encoder_epoch", "?")
    print(f"\n=== Encoder epoch {ep} ===")
    print(f"  Train acc: {metrics['train_acc']:.4f}  top-5: {metrics['train_top5_acc']:.4f}")
    print(f"  Val   acc: {metrics['val_acc']:.4f}  top-5: {metrics['val_top5_acc']:.4f}")
    print(f"  Test  acc: {metrics['test_acc']:.4f}  top-5: {metrics['test_top5_acc']:.4f}")


def find_epoch_checkpoints(ckpt_dir: str) -> list:
    pattern = re.compile(r"simclr_epoch_(\d+)\.pth$")
    ckpts = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(ckpt_dir, fname)))
    return sorted(ckpts, key=lambda x: x[0])


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.checkpoint:
        # Single checkpoint mode
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model, epoch = load_model(args.checkpoint, device)
        print(f"Loaded checkpoint (encoder epoch {epoch}): {args.checkpoint}")
        metrics = run_eval(model, epoch, args, device)
        print_results(metrics)
        print(f"\n  Classes: {metrics['num_classes']}  "
              f"Train: {metrics['train_size']}  Val: {metrics['val_size']}  Test: {metrics['test_size']}")

    else:
        # Sweep mode: eval every simclr_epoch_*.pth in checkpoint-dir
        ckpts = find_epoch_checkpoints(args.checkpoint_dir)
        if not ckpts:
            raise RuntimeError(f"No simclr_epoch_*.pth files found in {args.checkpoint_dir}")

        print(f"Found {len(ckpts)} checkpoints to evaluate in {args.checkpoint_dir}")

        out_csv = args.out_csv or os.path.join(args.checkpoint_dir, "imagenet_eval_sweep.csv")
        csv_fields = ["encoder_epoch", "train_acc", "train_top5_acc",
                      "val_acc", "val_top5_acc", "test_acc", "test_top5_acc",
                      "train_loss", "val_loss", "test_loss", "num_classes"]

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()

            for encoder_epoch, ckpt_path in ckpts:
                print(f"\n[{encoder_epoch}/{ckpts[-1][0]}] Evaluating {os.path.basename(ckpt_path)} ...")
                model, _ = load_model(ckpt_path, device)
                metrics = run_eval(model, encoder_epoch, args, device)
                print_results(metrics)
                writer.writerow(metrics)
                f.flush()

        print(f"\nSweep complete. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
