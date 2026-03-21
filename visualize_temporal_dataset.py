#!/usr/bin/env python3
"""Visualize one temporal window sample from EgocentricWindowDataset."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

from utils.EgocentricWindowDataset_new import EgocentricWindowDataset


def build_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one temporal window sample.")
    parser.add_argument(
        "--mode",
        type=str,
        default="sample",
        choices=["sample", "batch"],
        help="Visualization mode: one sample window or a full dataloader batch.",
    )
    parser.add_argument("--root-dir", type=str, required=True, help="Root folder with frame subfolders.")
    parser.add_argument("--window-size", type=int, default=5, help="Frames per window.")
    parser.add_argument("--frame-step", type=int, default=10, help="Stride in frames inside a window.")
    parser.add_argument("--window-stride", type=int, default=1, help="Stride in frames between consecutive window starts.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of window sample to visualize.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used when --mode batch.")
    parser.add_argument("--batch-idx", type=int, default=0, help="Batch index used when --mode batch.")
    parser.add_argument(
        "--batch-sampling",
        type=str,
        default="contiguous",
        choices=["contiguous", "random"],
        help="How to pick samples for a visualized batch.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --batch-sampling random.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize/crop size used before visualization.")
    parser.add_argument("--use-object-focus", action="store_true", help="Enable object-focus preprocessing.")
    parser.add_argument("--max-windows-per-clip", type=int, default=0, help="Cap windows per clip (0 keeps all).")
    parser.add_argument(
        "--single-window-short-clips",
        dest="single_window_short_clips",
        action="store_true",
        default=True,
        help="Collapse short clips to one representative window.",
    )
    parser.add_argument(
        "--allow-multiple-short-windows",
        dest="single_window_short_clips",
        action="store_false",
        help="Keep all candidate windows for short clips.",
    )
    parser.add_argument(
        "--short-clip-window-threshold",
        type=int,
        default=2,
        help="Short-clip threshold used when collapsing short clips.",
    )
    parser.add_argument("--save-path", type=str, default="", help="If set, save figure to this path.")
    return parser.parse_args()


def _to_image(frame: torch.Tensor):
    return frame.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _visualize_sample_window(dataset: EgocentricWindowDataset, sample_idx: int, save_path: str) -> None:
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample-idx must be in [0, {len(dataset)-1}], got {sample_idx}")

    window = dataset[sample_idx]  # [T, C, H, W]
    frame_paths = dataset.samples[sample_idx]
    t_steps = window.size(0)

    print(f"dataset_size={len(dataset)}")
    print(f"sample_idx={sample_idx}")
    print(f"window_shape={tuple(window.shape)}")
    print("frame_paths:")
    for p in frame_paths:
        print(f"  {p}")

    fig, axes = plt.subplots(1, t_steps, figsize=(3 * t_steps, 3))
    if t_steps == 1:
        axes = [axes]

    for i in range(t_steps):
        axes[i].imshow(_to_image(window[i]))
        axes[i].set_title(f"t{i}")
        axes[i].axis("off")

    fig.suptitle(f"Temporal window idx={sample_idx}, T={t_steps}", fontsize=12)
    fig.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved_figure={save_path}")
    else:
        plt.show()


def _visualize_batch_windows(
    dataset: EgocentricWindowDataset,
    batch_size: int,
    batch_idx: int,
    batch_sampling: str,
    seed: int,
    save_path: str,
) -> None:
    if batch_size < 1:
        raise ValueError(f"batch-size must be >= 1, got {batch_size}")
    if batch_idx < 0:
        raise ValueError(f"batch-idx must be >= 0, got {batch_idx}")

    n = len(dataset)
    if n == 0:
        raise RuntimeError("No samples available in dataset.")

    if batch_sampling == "contiguous":
        sample_start = batch_idx * batch_size
        sample_end_exclusive = min(sample_start + batch_size, n)
        if sample_start >= n:
            max_batch_idx = (n - 1) // batch_size
            raise IndexError(f"batch-idx must be in [0, {max_batch_idx}], got {batch_idx}")
        sample_indices = list(range(sample_start, sample_end_exclusive))
    else:
        if batch_idx != 0:
            print("[INFO] batch-idx is ignored when batch-sampling=random.")
        rng = np.random.default_rng(seed)
        choose_n = min(batch_size, n)
        sample_indices = sorted(rng.choice(n, size=choose_n, replace=False).tolist())

    windows = torch.stack([dataset[i] for i in sample_indices], dim=0)
    bsz, t_steps, _, _, _ = windows.shape

    print(f"dataset_size={len(dataset)}")
    print(f"batch_idx={batch_idx}")
    print(f"batch_sampling={batch_sampling}")
    print(f"batch_shape={tuple(windows.shape)}")
    print(f"sample_indices={sample_indices}")
    print("batch_frame_paths:")
    for sample_i in sample_indices:
        print(f"  sample_idx={sample_i}")
        for p in dataset.samples[sample_i]:
            print(f"    {p}")

    fig, axes = plt.subplots(bsz, t_steps, figsize=(3 * t_steps, 2.5 * bsz))
    if bsz == 1 and t_steps == 1:
        axes = [[axes]]
    elif bsz == 1:
        axes = [axes]
    elif t_steps == 1:
        axes = [[ax] for ax in axes]

    for b in range(bsz):
        sample_i = sample_indices[b]
        for t in range(t_steps):
            ax = axes[b][t]
            ax.imshow(_to_image(windows[b, t]))
            ax.axis("off")
            if b == 0:
                ax.set_title(f"t{t}")
            if t == 0:
                ax.set_ylabel(f"s{sample_i}", rotation=0, labelpad=25, va="center")

    fig.suptitle(f"Temporal batch idx={batch_idx}, B={bsz}, T={t_steps}", fontsize=12)
    fig.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved_figure={save_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    transform = build_transforms(args.image_size)

    dataset = EgocentricWindowDataset(
        root_dir=args.root_dir,
        window_size=args.window_size,
        frame_step=args.frame_step,
        window_stride=args.window_stride,
        transform=transform,
        return_stack=True,
        verbose=False,
        use_object_focus=args.use_object_focus,
        max_windows_per_clip=args.max_windows_per_clip,
        single_window_short_clips=args.single_window_short_clips,
        short_clip_window_threshold=args.short_clip_window_threshold,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found in {args.root_dir}. Try reducing window-size/frame-step or relaxing sampling limits."
        )

    if args.mode == "sample":
        _visualize_sample_window(dataset=dataset, sample_idx=args.sample_idx, save_path=args.save_path)
    else:
        _visualize_batch_windows(
            dataset=dataset,
            batch_size=args.batch_size,
            batch_idx=args.batch_idx,
            batch_sampling=args.batch_sampling,
            seed=args.seed,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    main()
