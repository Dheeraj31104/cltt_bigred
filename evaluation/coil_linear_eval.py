"""COIL-20 / COIL-100 linear evaluation utilities for frozen-encoder SimCLR.

Supports two directory layouts:
  1. Flat COIL layout  : obj{class}__{angle}.png  (e.g. obj3__45.png)
  2. ImageFolder layout: one subfolder per class containing image files

COIL-20  -> 20 classes, 72 poses each  -> 1 440 images total (grayscale)
COIL-100 -> 100 classes, 72 poses each -> 7 200 images total (colour)
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------

class COILDataset(Dataset):
    """
    Dataset for COIL-20 and COIL-100.

    Flat layout  (preferred):
        <root>/obj1__0.png, obj1__5.png, ..., obj20__355.png

    ImageFolder layout (fallback):
        <root>/class_a/img1.png
        <root>/class_b/img2.png
        ...
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        # ---- try flat COIL naming: obj{id}__{angle}.ext ----
        coil_re = re.compile(r"^obj(\d+)__\d+\.(png|jpg|jpeg)$", re.IGNORECASE)
        try:
            entries = os.listdir(self.root_dir)
        except FileNotFoundError:
            raise FileNotFoundError(f"COIL root directory not found: {self.root_dir}")

        flat_matches = [
            (f, m) for f in entries if (m := coil_re.match(f))
        ]

        if flat_matches:
            class_ids = sorted({int(m.group(1)) for _, m in flat_matches})
            self.classes = [str(c) for c in class_ids]
            self.class_to_idx = {s: i for i, s in enumerate(self.classes)}
            for fname, m in sorted(flat_matches):
                label = self.class_to_idx[str(int(m.group(1)))]
                self.samples.append((os.path.join(self.root_dir, fname), label))
            return

        # ---- fallback: ImageFolder layout ----
        valid_exts = {".png", ".jpg", ".jpeg"}
        subdirs = sorted(
            d for d in entries
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith(".")
        )
        if not subdirs:
            raise RuntimeError(
                f"No COIL-style image files and no subfolders found in {self.root_dir}.\n"
                f"Expected either flat COIL naming (obj{{id}}__{{angle}}.png) "
                f"or one subfolder per class."
            )
        self.classes = subdirs
        self.class_to_idx = {c: i for i, c in enumerate(subdirs)}
        for cls_name in subdirs:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # COIL-20 is grayscale → force RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)


# ---------------------------------------------------------------------------
#  Transforms
# ---------------------------------------------------------------------------

def build_coil_transforms(image_size: int) -> T.Compose:
    """Resize to model input size, then apply ImageNet normalisation."""
    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
#  Encoder freeze / restore helpers  (mirrors cifar_linear_eval.py)
# ---------------------------------------------------------------------------

def _freeze_encoder(encoder: nn.Module) -> List[bool]:
    states = [p.requires_grad for p in encoder.parameters()]
    for p in encoder.parameters():
        p.requires_grad_(False)
    return states


def _restore_encoder(encoder: nn.Module, states: List[bool]) -> None:
    for p, s in zip(encoder.parameters(), states):
        p.requires_grad_(s)


# ---------------------------------------------------------------------------
#  Evaluation loop helper
# ---------------------------------------------------------------------------

def _eval_loader(
    encoder: nn.Module,
    linear_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: Optional[int],
) -> Tuple[float, float, int]:
    loss_sum = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = linear_head(encoder(x))
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            num_batches += 1

    avg_loss = loss_sum / max(1, num_batches)
    acc = correct / max(1, total)
    return avg_loss, acc, num_batches


# ---------------------------------------------------------------------------
#  Main linear eval function
# ---------------------------------------------------------------------------

def linear_eval_on_coil(
    encoder: nn.Module,
    feature_dim: int,
    device: torch.device,
    coil_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    eval_epochs: int = 20,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    split_seed: int = 42,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Freeze encoder, train a linear head on COIL features, return metrics dict.

    The dataset is split into train / val / test based on train_fraction and
    val_fraction (test = remainder). Splits are random but reproducible via
    split_seed.

    Returns:
        {train_loss, train_acc, val_loss, val_acc, test_loss, test_acc,
         num_classes, train_size, val_size, test_size, train_fraction,
         val_fraction}
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0")

    transform = build_coil_transforms(image_size)
    dataset = COILDataset(root_dir=coil_dir, transform=transform)
    num_classes = dataset.num_classes
    n = len(dataset)

    train_len = max(1, int(n * train_fraction))
    val_len = max(1, int(n * val_fraction))
    test_len = max(1, n - train_len - val_len)
    # Adjust if rounding leaves a gap
    train_len = n - val_len - test_len

    gen = torch.Generator().manual_seed(split_seed)
    train_split, val_split, test_split = random_split(
        dataset, [train_len, val_len, test_len], generator=gen
    )

    print(
        "[coil-eval] Split: train={} val={} test={} | classes={}".format(
            train_len, val_len, test_len, num_classes
        )
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_split, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_split, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_split, shuffle=False, **loader_kwargs)

    linear_head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(
        linear_head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Freeze encoder and switch to eval mode
    was_training = encoder.training
    grad_states = _freeze_encoder(encoder)
    encoder.eval()

    train_loss = 0.0
    train_acc = 0.0
    for epoch in range(max(1, eval_epochs)):
        linear_head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for b_idx, (x, y) in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                feats = encoder(x)
            logits = linear_head(feats)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            num_batches += 1

        train_loss = running_loss / max(1, num_batches)
        train_acc = correct / max(1, total)

    linear_head.eval()
    val_loss, val_acc, val_batches = _eval_loader(
        encoder, linear_head, val_loader, device, criterion, max_batches
    )
    test_loss, test_acc, test_batches = _eval_loader(
        encoder, linear_head, test_loader, device, criterion, max_batches
    )

    # Restore encoder to its original state
    _restore_encoder(encoder, grad_states)
    if was_training:
        encoder.train()

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_classes": num_classes,
        "train_size": train_len,
        "val_size": val_len,
        "test_size": test_len,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
    }
