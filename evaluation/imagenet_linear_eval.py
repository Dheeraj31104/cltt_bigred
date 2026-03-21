"""ImageNet-style linear evaluation for frozen-encoder SimCLR.

Works with any ImageFolder-compatible dataset:
  - Tiny ImageNet (200 classes, 64x64)
  - ImageNet-100  (100 classes, 224x224)
  - Full ImageNet-1k (1000 classes, 224x224)

Expected directory layout
--------------------------
<data_dir>/
    train/
        <class_id>/
            *.JPEG  (or *.jpg / *.png)
    val/
        <class_id>/
            *.JPEG

Note: Tiny ImageNet's val folder ships as a flat layout with a
val_annotations.txt file.  Run the helper script below once to
reorganise it into the standard per-class sub-folder structure:

    python -c "
    import os, shutil
    data = 'tiny-imagenet-200'
    ann  = open(f'{data}/val/val_annotations.txt').readlines()
    for line in ann:
        fname, cls = line.split()[0], line.split()[1]
        os.makedirs(f'{data}/val/{cls}', exist_ok=True)
        shutil.move(f'{data}/val/images/{fname}', f'{data}/val/{cls}/{fname}')
    "
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from evaluation.cifar_linear_eval import _freeze_encoder, _restore_encoder, _topk_correct


def build_imagenet_transforms(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _eval_loader(
    encoder: nn.Module,
    linear_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: Optional[int],
) -> Tuple[float, float, float, int]:
    eval_loss = 0.0
    eval_correct = 0
    eval_top5_correct = 0
    eval_total = 0
    eval_batches = 0

    with torch.no_grad():
        for b_idx, (x, y) in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = encoder(x)
            logits = linear_head(feats)
            loss = criterion(logits, y)

            eval_loss += loss.item()
            eval_batches += 1
            eval_correct += (logits.argmax(dim=1) == y).sum().item()
            eval_top5_correct += _topk_correct(logits, y, k=5)
            eval_total += y.size(0)

    return (
        eval_loss / max(1, eval_batches),
        eval_correct / max(1, eval_total),
        eval_top5_correct / max(1, eval_total),
        eval_batches,
    )


def linear_eval_on_imagenet(
    encoder: nn.Module,
    feature_dim: int,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    eval_epochs: int,
    lr: float,
    weight_decay: float,
    data_dir: str,
    train_fraction: float = 0.9,
    split_seed: int = 42,
    max_batches: Optional[int] = None,
) -> dict:
    transform = build_imagenet_transforms(image_size)

    try:
        train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
        test_ds  = datasets.ImageFolder(root=f"{data_dir}/val",   transform=transform)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find ImageNet train/val folders under {data_dir}. "
            "Ensure the directory contains train/ and val/ sub-folders in ImageFolder format."
        ) from exc

    num_classes = len(train_ds.classes)

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    train_len = int(len(train_ds) * train_fraction)
    train_len = max(1, min(train_len, len(train_ds) - 1))
    val_len = len(train_ds) - train_len
    split_generator = torch.Generator().manual_seed(split_seed)
    train_split, val_split = random_split(train_ds, [train_len, val_len], generator=split_generator)

    print(
        f"[linear-eval] ImageNet split: train={len(train_split)} val={len(val_split)} "
        f"test={len(test_ds)} classes={num_classes} (train_fraction={train_fraction:.2f})"
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=device.type == "cuda")

    linear_head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer   = torch.optim.SGD(linear_head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion   = nn.CrossEntropyLoss()

    was_training = encoder.training
    grad_states  = _freeze_encoder(encoder)
    encoder.eval()

    train_loss = 0.0
    train_acc = 0.0
    train_top5_acc = 0.0
    for _ in range(max(1, eval_epochs)):
        linear_head.train()
        running_loss = 0.0
        correct = 0
        top5_correct = 0
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
            loss   = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches  += 1
            correct      += (logits.argmax(dim=1) == y).sum().item()
            top5_correct += _topk_correct(logits, y, k=5)
            total        += y.size(0)

        train_loss     = running_loss / max(1, num_batches)
        train_acc      = correct      / max(1, total)
        train_top5_acc = top5_correct / max(1, total)

    linear_head.eval()
    val_loss,  val_acc,  val_top5_acc,  val_batches  = _eval_loader(
        encoder, linear_head, val_loader,  device, criterion, max_batches)
    test_loss, test_acc, test_top5_acc, test_batches = _eval_loader(
        encoder, linear_head, test_loader, device, criterion, max_batches)

    _restore_encoder(encoder, grad_states)
    if was_training:
        encoder.train()

    return {
        "train_loss":      train_loss,
        "train_acc":       train_acc,
        "train_top5_acc":  train_top5_acc,
        "val_loss":        val_loss,
        "val_acc":         val_acc,
        "val_top5_acc":    val_top5_acc,
        "test_loss":       test_loss,
        "test_acc":        test_acc,
        "test_top5_acc":   test_top5_acc,
        "num_classes":     num_classes,
        "train_fraction":  train_fraction,
        "train_size":      train_len,
        "val_size":        val_len,
        "test_size":       len(test_ds),
        "val_batches":     val_batches,
        "test_batches":    test_batches,
    }
