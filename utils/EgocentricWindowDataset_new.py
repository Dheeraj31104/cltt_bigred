import os
from typing import List, Sequence, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import cv2
import numpy as np


def get_bboxes_dummy(img_w: int, img_h: int):
    """
    Returns one central box as 'object region'.

    This is a placeholder. Later you can replace this with real detector
    bounding boxes (e.g., from YOLO / DETR) in (x1, y1, x2, y2) format.
    """
    x1 = int(img_w * 0.25)
    x2 = int(img_w * 0.75)
    y1 = int(img_h * 0.25)
    y2 = int(img_h * 0.75)
    return [(x1, y1, x2, y2)]


def focus_objects_pil(pil_img: Image.Image, blur_ksize=(31, 31)) -> Image.Image:
    """
    Take a PIL RGB image -> return a new PIL RGB image where
    object regions are sharp and background is blurred.

    Currently uses get_bboxes_dummy (central box). Swap that out with a real
    detector later to focus on actual objects.
    """
    # PIL RGB -> numpy BGR (for cv2)
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape

    # Get bounding boxes for objects (dummy for now)
    bboxes = get_bboxes_dummy(w, h)

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

    If two_view_offset == 0 (default):
        Each sample is a single window of size `window_size`:
            [t, t + frame_step, t + 2*frame_step, ..., t + (window_size-1)*frame_step]

    If two_view_offset > 0:
        Each sample is a *pair* of windows (view0, view1):

            view0: [t,
                    t + frame_step,
                    ...,
                    t + (window_size-1)*frame_step]

            view1: [t + two_view_offset,
                    t + two_view_offset + frame_step,
                    ...,
                    t + two_view_offset + (window_size-1)*frame_step]

        with all indices kept within the same clip.
    """

    def __init__(
        self,
        root_dir: str,
        window_size: int = 3,
        frame_step: int = 1,
        transform: Union[None, T.Compose] = None,
        return_stack: bool = True,
        verbose: bool = True,
        use_object_focus: bool = False,
        two_view_offset: int = 0,  # <-- NEW: offset (in frames) between view0 and view1
    ):
        """
        Args:
            root_dir: Root directory containing all frame folders (e.g. 'frames_out').
            window_size: Number of frames in each sample (>= 2).
            frame_step: Step in *frames* between consecutive frames in a window.
                        frame_step = 1  -> adjacent frames
                        frame_step = 30 -> ~1s apart if FPS=30
            transform: Transform applied to each frame.
            return_stack:
                True  -> returns tensor [W, C, H, W]
                False -> returns tuple of length W.
            use_object_focus:
                If True, each frame is passed through focus_objects_pil
                before transforms.
            two_view_offset:
                If 0 -> return a *single* window per sample (original behavior).
                If >0 -> return a *pair* of windows (view0, view1) offset in time
                         by `two_view_offset` frames.
        """
        if window_size < 2:
            raise ValueError("window_size must be >= 2 for adjacent/strided frames")
        if frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if two_view_offset < 0:
            raise ValueError("two_view_offset must be >= 0")

        self.root_dir = os.path.abspath(root_dir)
        self.window_size = window_size
        self.frame_step = frame_step
        self.transform = transform
        self.return_stack = return_stack
        self.verbose = verbose
        self.use_object_focus = use_object_focus
        self.two_view_offset = two_view_offset

        # Each item in self.samples is:
        #   - if two_view_offset == 0: List[str]  (single window paths)
        #   - if two_view_offset > 0: Tuple[List[str], List[str]]  (view0_paths, view1_paths)
        self.samples: List[Union[List[str], tuple]] = []
        self._build_index()

        if self.verbose:
            print(f"[EgocentricWindowDataset] root_dir       = {self.root_dir}")
            print(f"[EgocentricWindowDataset] window_size   = {self.window_size}")
            print(f"[EgocentricWindowDataset] frame_step    = {self.frame_step}")
            print(f"[EgocentricWindowDataset] two_view_offs = {self.two_view_offset}")
            print(f"[EgocentricWindowDataset] total samples = {len(self.samples)}")

    def _build_index(self):
        valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

        num_clips = 0
        num_frames_total = 0

        for dirpath, _, filenames in os.walk(self.root_dir):
            image_files = [
                f for f in filenames
                if os.path.splitext(f)[1] in valid_exts
            ]
            if not image_files:
                continue

            # Sort frames in temporal order
            image_files = sorted(image_files)
            full_paths = [os.path.join(dirpath, f) for f in image_files]
            L = len(full_paths)

            # For a *single* window, last index is:
            #   t + (window_size-1)*frame_step
            # For a *pair* of windows (view0 + offset + view1), last index is:
            #   t + two_view_offset + (window_size-1)*frame_step
            if self.two_view_offset == 0:
                min_len = 1 + (self.window_size - 1) * self.frame_step
            else:
                min_len = 1 + self.two_view_offset + (self.window_size - 1) * self.frame_step

            if L < min_len:
                continue  # not enough frames in this clip

            num_clips += 1
            num_frames_total += L

            if self.two_view_offset == 0:
                # Original behavior: one window per sample
                max_start = L - (self.window_size - 1) * self.frame_step
                for start in range(0, max_start):
                    window_paths = [
                        full_paths[start + i * self.frame_step]
                        for i in range(self.window_size)
                    ]
                    self.samples.append(window_paths)
            else:
                # New behavior: pair of windows (view0, view1)
                # Need to ensure all indices for both views are valid.
                # view0 last index: start + (W-1)*frame_step
                # view1 last index: start + two_view_offset + (W-1)*frame_step
                max_start = L - (self.two_view_offset + (self.window_size - 1) * self.frame_step)
                for start in range(0, max_start):
                    # view 0
                    view0_paths = [
                        full_paths[start + i * self.frame_step]
                        for i in range(self.window_size)
                    ]
                    # view 1 (shifted by two_view_offset frames)
                    view1_paths = [
                        full_paths[start + self.two_view_offset + i * self.frame_step]
                        for i in range(self.window_size)
                    ]
                    self.samples.append((view0_paths, view1_paths))

        if self.verbose:
            print(f"[EgocentricWindowDataset] clips with enough frames: {num_clips}")
            print(f"[EgocentricWindowDataset] total frames in those clips: {num_frames_total}")
            if len(self.samples) == 0:
                print(
                    f"[WARN] No samples found in {self.root_dir}. "
                    f"Check that frames exist, or reduce window_size/frame_step/two_view_offset."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def shape(self):
        sample = self[0]
        if isinstance(sample, torch.Tensor):
            return (len(self),) + sample.shape
        else:
            # when returning (win_v1, win_v2)
            win_t, _ = sample
            return (len(self),) + win_t.shape

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
        sample = self.samples[idx]

        # --- single-view mode (original behavior) ---
        if self.two_view_offset == 0:
            paths: Sequence[str] = sample
            frames = [self._load_image(p) for p in paths]

            if self.return_stack:
                # [W, C, H, W]
                return torch.stack(frames, dim=0)
            else:
                return tuple(frames)

        # --- two-view mode: return (view0, view1) ---
        else:
            view0_paths, view1_paths = sample
            frames_v0 = [self._load_image(p) for p in view0_paths]
            frames_v1 = [self._load_image(p) for p in view1_paths]

            if self.return_stack:
                win_v0 = torch.stack(frames_v0, dim=0)  # [W, C, H, W]
                win_v1 = torch.stack(frames_v1, dim=0)  # [W, C, H, W]
                return win_v0, win_v1
            else:
                return tuple(frames_v0), tuple(frames_v1)
