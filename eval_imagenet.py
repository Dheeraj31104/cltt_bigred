#!/usr/bin/env python3
"""Standalone Tiny ImageNet linear evaluation on a saved SimCLR checkpoint."""

import argparse
import os

import torch

from evaluation.imagenet_linear_eval import linear_eval_on_imagenet
from models.simclr_resnet import SimCLRResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear eval on Tiny ImageNet from a SimCLR checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to tiny-imagenet-200 (must have train/ and val/).")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (64 for Tiny ImageNet).")
    parser.add_argument("--eval-epochs", type=int, default=30, help="Epochs to train the linear head.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Cap batches per pass (for quick testing).")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    proj_dim = ckpt.get("proj_dim", 128)

    model = SimCLRResNet18(proj_dim=proj_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}): {args.checkpoint}")

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

    print("\n=== Linear Eval Results ===")
    print(f"  Classes       : {metrics['num_classes']}")
    print(f"  Train samples : {metrics['train_size']}  |  Val: {metrics['val_size']}  |  Test: {metrics['test_size']}")
    print(f"  Train acc     : {metrics['train_acc']:.4f}  top-5: {metrics['train_top5_acc']:.4f}")
    print(f"  Val   acc     : {metrics['val_acc']:.4f}  top-5: {metrics['val_top5_acc']:.4f}")
    print(f"  Test  acc     : {metrics['test_acc']:.4f}  top-5: {metrics['test_top5_acc']:.4f}")


if __name__ == "__main__":
    main()
