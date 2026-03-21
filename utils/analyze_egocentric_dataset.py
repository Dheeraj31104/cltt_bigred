#!/usr/bin/env python3
"""Audit egocentric frame folders and report window-count reductions."""

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from typing import List


VALID_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class ClipStats:
    path: str
    frames: int
    max_start_exclusive: int
    dense_windows: int
    final_windows: int


def _starts_count(max_start_exclusive: int, stride: int) -> int:
    if max_start_exclusive <= 0:
        return 0
    return (max_start_exclusive + stride - 1) // stride


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why egocentric sample count is low.")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing frame folders.")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--max-windows-per-clip", type=int, default=0)
    parser.add_argument(
        "--single-window-short-clips",
        dest="single_window_short_clips",
        action="store_true",
        default=True,
        help="Collapse clips with <= short threshold windows to one window.",
    )
    parser.add_argument(
        "--allow-multiple-short-windows",
        dest="single_window_short_clips",
        action="store_false",
        help="Do not collapse short clips.",
    )
    parser.add_argument("--short-clip-window-threshold", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10, help="Show top-K clips by frame count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.window_size < 2:
        raise ValueError("window_size must be >= 2")
    if args.frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if args.window_stride < 1:
        raise ValueError("window_stride must be >= 1")
    if args.max_windows_per_clip < 0:
        raise ValueError("max_windows_per_clip must be >= 0")
    if args.short_clip_window_threshold < 1:
        raise ValueError("short_clip_window_threshold must be >= 1")

    root_dir = os.path.abspath(args.root_dir)
    min_len = 1 + (args.window_size - 1) * args.frame_step

    total_dirs = 0
    dirs_with_any_files = 0
    dirs_with_valid_images = 0

    valid_images_total = 0
    unsupported_files_total = 0
    ext_hist = Counter()

    clips_too_short = 0
    frames_too_short = 0
    clips_eligible = 0
    frames_eligible = 0

    windows_before_short_and_cap = 0
    windows_after_short = 0
    windows_final = 0

    short_collapsed_clips = 0
    windows_removed_by_short_collapse = 0
    capped_clips = 0
    windows_removed_by_cap = 0

    stride1_windows_if_no_short_no_cap = 0
    clip_stats: List[ClipStats] = []

    for dirpath, _, filenames in os.walk(root_dir):
        total_dirs += 1
        non_hidden = [f for f in filenames if not f.startswith(".")]
        if not non_hidden:
            continue

        dirs_with_any_files += 1
        valid = []
        for f in non_hidden:
            ext = os.path.splitext(f)[1]
            ext_hist[ext.lower()] += 1
            if ext in VALID_EXTS:
                valid.append(f)
            else:
                unsupported_files_total += 1

        if not valid:
            continue

        dirs_with_valid_images += 1
        valid = sorted(valid)
        L = len(valid)
        valid_images_total += L

        if L < min_len:
            clips_too_short += 1
            frames_too_short += L
            continue

        clips_eligible += 1
        frames_eligible += L

        max_start_exclusive = L - (args.window_size - 1) * args.frame_step
        dense_count = _starts_count(max_start_exclusive, args.window_stride)
        stride1_windows_if_no_short_no_cap += max_start_exclusive
        windows_before_short_and_cap += dense_count

        after_short = dense_count
        if args.single_window_short_clips and dense_count <= args.short_clip_window_threshold and dense_count > 0:
            short_collapsed_clips += 1
            windows_removed_by_short_collapse += dense_count - 1
            after_short = 1
        windows_after_short += after_short

        final_count = after_short
        if args.max_windows_per_clip > 0 and final_count > args.max_windows_per_clip:
            capped_clips += 1
            windows_removed_by_cap += final_count - args.max_windows_per_clip
            final_count = args.max_windows_per_clip
        windows_final += final_count

        clip_stats.append(
            ClipStats(
                path=dirpath,
                frames=L,
                max_start_exclusive=max_start_exclusive,
                dense_windows=dense_count,
                final_windows=final_count,
            )
        )

    print("=== Egocentric Dataset Audit ===")
    print(f"root_dir: {root_dir}")
    print(f"window_size={args.window_size} frame_step={args.frame_step} window_stride={args.window_stride}")
    print(
        "single_window_short_clips={} short_clip_window_threshold={} max_windows_per_clip={}".format(
            args.single_window_short_clips,
            args.short_clip_window_threshold,
            args.max_windows_per_clip,
        )
    )
    print()
    print(f"dirs_traversed: {total_dirs}")
    print(f"dirs_with_any_files: {dirs_with_any_files}")
    print(f"dirs_with_valid_images: {dirs_with_valid_images}")
    print(f"valid_images_total: {valid_images_total}")
    print(f"unsupported_non_hidden_files: {unsupported_files_total}")
    print()
    print(f"min_frames_needed_per_clip: {min_len}")
    print(f"eligible_clips: {clips_eligible}")
    print(f"too_short_clips: {clips_too_short}")
    print(f"frames_in_eligible_clips: {frames_eligible}")
    print(f"frames_in_too_short_clips: {frames_too_short}")
    print()
    print(f"windows_before_short_and_cap: {windows_before_short_and_cap}")
    print(f"windows_after_short_collapse: {windows_after_short}")
    print(f"windows_final: {windows_final}")
    print(f"short_collapsed_clips: {short_collapsed_clips}")
    print(f"windows_removed_by_short_collapse: {windows_removed_by_short_collapse}")
    print(f"capped_clips: {capped_clips}")
    print(f"windows_removed_by_cap: {windows_removed_by_cap}")
    print()
    print(f"what_if_stride_1_no_short_no_cap: {stride1_windows_if_no_short_no_cap}")
    for stride in (1, 2, 5, 10, 20, 30):
        windows = sum(_starts_count(c.max_start_exclusive, stride) for c in clip_stats)
        print(f"what_if_stride_{stride}_no_short_no_cap: {windows}")

    print()
    print("Top clips by frame count:")
    for c in sorted(clip_stats, key=lambda x: x.frames, reverse=True)[: max(0, args.top_k)]:
        print(
            f"  frames={c.frames:6d} dense_windows={c.dense_windows:6d} final_windows={c.final_windows:6d} path={c.path}"
        )

    print()
    print("Top file extensions encountered:")
    for ext, count in ext_hist.most_common(12):
        print(f"  {ext or '<no_ext>'}: {count}")


if __name__ == "__main__":
    main()
