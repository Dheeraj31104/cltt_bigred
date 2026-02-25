import os
from typing import List, Sequence, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import cv2
import numpy as np


def focus_objects_pil(pil_img: Image.Image, blur_ksize=(31, 31)) -> Image.Image:
    """
    Take a PIL RGB image -> return a new PIL RGB image where
    object regions are sharp and background is blurred.

    Uses a central fallback box as the focus region. Replace `bboxes` below
    with detector outputs to focus on actual objects.
    """
    # PIL RGB -> numpy BGR (for cv2)
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape

    # Fallback focus region (center box) when detector boxes are unavailable.
    bboxes = [(int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))]

    if not bboxes:
        # If no boxes, just return original image
        return pil_img

    # Build a binary mask: 1 inside object boxes, 0 outside
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w - 1, int(x2))
        y2 = min(h - 1, int(y2))
        mask[y1:y2 + 1, x1:x2 + 1] = 1

    # Blur the entire image
    blurred = cv2.GaussianBlur(img_bgr, blur_ksize, 0)

    # Expand mask to 3 channels
    mask_3c = np.repeat(mask[:, :, None], 3, axis=2)

    # Combine: keep original in object regions, blurred in background
    focused_bgr = np.where(mask_3c == 1, img_bgr, blurred)

    # Convert back to PIL RGB
    focused_rgb = cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2RGB)
    focused_pil = Image.fromarray(focused_rgb)
    return focused_pil




class EgocentricWindowDataset(Dataset):
    """
    Builds sliding *strided* windows of frames from extracted video frames.
    Each sample is a single window of size `window_size`:
        [t, t + frame_step, t + 2*frame_step, ..., t + (window_size-1)*frame_step]
    Successive windows start every `window_stride` frames.
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int = 3,
        frame_step: int = 1,
        window_stride: int = 1,
        transform: Union[None, T.Compose] = None,
        return_stack: bool = True,
        verbose: bool = True,
        use_object_focus: bool = False,
        max_windows_per_clip: int = 0,
        single_window_short_clips: bool = True,
        short_clip_window_threshold: int = 2,
    ):
        """
        Args:
            root_dir: Root directory containing all frame folders (e.g. 'frames_out').
            window_size: Number of frames in each sample (>= 2).
            frame_step: Step in *frames* between consecutive frames in a window.
                        frame_step = 1  -> adjacent frames
                        frame_step = 30 -> ~1s apart if FPS=30
            window_stride: Step in *frames* between starts of consecutive windows.
                        window_stride = 1  -> dense overlapping windows
                        window_stride = 30 -> roughly one new window per second at 30 FPS
            transform: Transform applied to each frame.
            return_stack:
                True  -> returns tensor [W, C, H, W]
                False -> returns tuple of length W.
            use_object_focus:
                If True, each frame is passed through focus_objects_pil
                before transforms.
            max_windows_per_clip:
                Hard cap for windows from one clip.
                0 means "no cap" (use all valid sliding windows).
            single_window_short_clips:
                If True, clips that only yield a small number of candidate windows
                are collapsed to one representative window.
            short_clip_window_threshold:
                "Small number" threshold used when single_window_short_clips=True.
                Example: threshold=2 means clips that would produce 1 or 2 windows
                now produce exactly 1 window.
        """
        if window_size < 2:
            raise ValueError("window_size must be >= 2 for adjacent/strided frames")
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if window_stride < 1:
            raise ValueError("window_stride must be >= 1")
        if max_windows_per_clip < 0:
            raise ValueError("max_windows_per_clip must be >= 0")
        if short_clip_window_threshold < 1:
            raise ValueError("short_clip_window_threshold must be >= 1")

        self.root_dir = os.path.abspath(root_dir)
        self.window_size = window_size
        self.frame_step = frame_step
        self.window_stride = window_stride
        self.transform = transform
        self.return_stack = return_stack
        self.verbose = verbose
        self.use_object_focus = use_object_focus
        self.max_windows_per_clip = max_windows_per_clip
        self.single_window_short_clips = single_window_short_clips
        self.short_clip_window_threshold = short_clip_window_threshold

        self.samples: List[List[str]] = []
        self._build_index()

        if self.verbose:
            print(f"[EgocentricWindowDataset] root_dir       = {self.root_dir}")
            print(f"[EgocentricWindowDataset] window_size   = {self.window_size}")
            print(f"[EgocentricWindowDataset] frame_step    = {self.frame_step}")
            print(f"[EgocentricWindowDataset] window_stride = {self.window_stride}")
            print(f"[EgocentricWindowDataset] max_win_clip  = {self.max_windows_per_clip}")
            print(f"[EgocentricWindowDataset] short->single = {self.single_window_short_clips}")
            print(f"[EgocentricWindowDataset] short_thresh  = {self.short_clip_window_threshold}")
            print(f"[EgocentricWindowDataset] total samples = {len(self.samples)}")

    def _select_start_indices(self, max_start_exclusive: int) -> List[int]:
        starts = list(range(0, max_start_exclusive, self.window_stride))
        if not starts:
            return starts

        if self.single_window_short_clips and len(starts) <= self.short_clip_window_threshold:
            return [starts[len(starts) // 2]]

        if self.max_windows_per_clip > 0 and len(starts) > self.max_windows_per_clip:
            if self.max_windows_per_clip == 1:
                return [starts[len(starts) // 2]]
            keep_idx = np.linspace(
                0,
                len(starts) - 1,
                num=self.max_windows_per_clip,
                dtype=int,
            )
            starts = [starts[i] for i in keep_idx]

        return starts

    def _build_index(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

        num_clips = 0
        num_frames_total = 0

        for dirpath, _, filenames in os.walk(self.root_dir):
            image_files = [
                f for f in filenames
                if os.path.splitext(f)[1] in valid_exts and not f.startswith(".")
            ]
            if not image_files:
                continue

            # Sort frames in temporal order
            image_files = sorted(image_files)
            full_paths = [os.path.join(dirpath, f) for f in image_files]
            L = len(full_paths)

            # For one window, last index is t + (window_size-1)*frame_step.
            min_len = 1 + (self.window_size - 1) * self.frame_step

            if L < min_len:
                continue  # not enough frames in this clip

            num_clips += 1
            num_frames_total += L

            max_start_exclusive = L - (self.window_size - 1) * self.frame_step
            for start in self._select_start_indices(max_start_exclusive):
                window_paths = [
                    full_paths[start + i * self.frame_step]
                    for i in range(self.window_size)
                ]
                self.samples.append(window_paths)

        if self.verbose:
            print(f"[EgocentricWindowDataset] clips with enough frames: {num_clips}")
            print(f"[EgocentricWindowDataset] total frames in those clips: {num_frames_total}")
            if len(self.samples) == 0:
                print(
                    f"[WARN] No samples found in {self.root_dir}. "
                    f"Check that frames exist, or reduce window_size/frame_step/window_stride, "
                    f"or relax short-clip / max-window limits."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def shape(self):
        sample = self[0]
        return (len(self),) + sample.shape

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")

        if self.use_object_focus:
            img = focus_objects_pil(img)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        return img

    def __getitem__(self, idx: int):
        paths: Sequence[str] = self.samples[idx]
        frames = [self._load_image(p) for p in paths]

        if self.return_stack:
            # [W, C, H, W]
            return torch.stack(frames, dim=0)
        else:
            return tuple(frames)
