"""CIFAR linear evaluation utilities for frozen-encoder SimCLR."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def _cifar_stats(name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], int]:
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = 10
    elif name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported CIFAR dataset: {name}")
    return mean, std, num_classes


def build_cifar_transforms(image_size: int, dataset_name: str) -> T.Compose:
    mean, std, _ = _cifar_stats(dataset_name)
    return T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def _freeze_encoder(encoder: nn.Module) -> List[bool]:
    states = [p.requires_grad for p in encoder.parameters()]
    for p in encoder.parameters():
        p.requires_grad_(False)
    return states


def _restore_encoder(encoder: nn.Module, states: List[bool]) -> None:
    for param, state in zip(encoder.parameters(), states):
        param.requires_grad_(state)


def _eval_loader(
    encoder: nn.Module,
    linear_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: Optional[int],
) -> Tuple[float, float, int]:
    eval_loss = 0.0
    eval_correct = 0
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
            eval_total += y.size(0)

    eval_loss = eval_loss / max(1, eval_batches)
    eval_acc = eval_correct / max(1, eval_total)
    return eval_loss, eval_acc, eval_batches


def linear_eval_on_cifar(
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
    dataset_name: str = "cifar10",
    download: bool = False,
    train_fraction: float = 0.7,
    split_seed: int = 42,
    max_batches: Optional[int] = None,
) -> dict:
    _, _, num_classes = _cifar_stats(dataset_name)
    transform = build_cifar_transforms(image_size=image_size, dataset_name=dataset_name)
    dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    try:
        train_ds = dataset_cls(root=data_dir, train=True, transform=transform, download=download)
        test_ds = dataset_cls(root=data_dir, train=False, transform=transform, download=download)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load {dataset_name} from {data_dir}. "
            f"Set --cifar-download to fetch it if needed."
        ) from exc

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    train_len = int(len(train_ds) * train_fraction)
    train_len = max(1, min(train_len, len(train_ds) - 1))
    val_len = len(train_ds) - train_len
    split_generator = torch.Generator().manual_seed(split_seed)
    train_split, val_split = random_split(train_ds, [train_len, val_len], generator=split_generator)

    print(
        "[linear-eval] CIFAR split: train={train} val={val} test={test} (train_fraction={frac:.2f})".format(
            train=len(train_split),
            val=len(val_split),
            test=len(test_ds),
            frac=train_fraction,
        )
    )

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    linear_head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(linear_head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    was_training = encoder.training
    grad_states = _freeze_encoder(encoder)
    encoder.eval()

    train_loss = 0.0
    train_acc = 0.0
    for _ in range(max(1, eval_epochs)):
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
            num_batches += 1
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / max(1, num_batches)
        train_acc = correct / max(1, total)

    linear_head.eval()
    val_loss, val_acc, val_batches = _eval_loader(
        encoder=encoder,
        linear_head=linear_head,
        loader=val_loader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
    )
    test_loss, test_acc, test_batches = _eval_loader(
        encoder=encoder,
        linear_head=linear_head,
        loader=test_loader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
    )

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
        "train_fraction": train_fraction,
        "train_size": train_len,
        "val_size": val_len,
        "test_size": len(test_ds),
        "val_batches": val_batches,
        "test_batches": test_batches,
    }
