#!/usr/bin/env python3
"""
Train SimCLR on egocentric frame windows with Slurm-friendly CLI arguments.
The script mirrors the notebook logic but adds argparse, checkpointing/resume,
sub-batch gradient accumulation, and max-batches-per-epoch trimming so it can
run cleanly inside sbatch or job arrays.
"""

import argparse
import os
import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, models
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from utils.EgocentricWindowDataset_new import EgocentricWindowDataset

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None


# -----------------------
#    Model + Loss
# -----------------------


class SimCLRResNet18(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()

        base = models.resnet18(weights=None)
        num_ftrs = base.fc.in_features  # 512 for ResNet-18
        base.fc = nn.Identity()

        self.encoder = base
        self.feature_dim = num_ftrs
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),   # bias unchanged (default True)
            nn.BatchNorm1d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, proj_dim),   # bias unchanged (default True)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)             # (B, 512)
        z = self.projection_head(h)     # (B, proj_dim)
        return h, z

class PairedCosineTTLoss(nn.Module):
    """
    Buffer contains consecutive positive pairs:
      (0,1), (2,3), ..., (N-2, N-1)

    For each anchor i:
      numerator: exp(sim(i, pos(i))/tau)
      denominator: sum_{k != i and k != pos(i)} exp(sim(i,k)/tau)
    """
    def __init__(self, temperature: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (N, D) where N is even and ordered in positive pairs.
        returns: scalar loss averaged over N anchors
        """
        if z.dim() != 2:
            raise ValueError(f"Expected z to be (N, D), got {tuple(z.shape)}")
        N, _ = z.shape
        if N % 2 != 0:
            raise ValueError(f"N must be even (pairs), got N={N}")

        # cosine similarity via normalized dot product
        z = F.normalize(z, dim=1, eps=self.eps)
        sim = (z @ z.T) / self.temperature  # (N, N)

        # pos index mapping for consecutive pairs: 0<->1, 2<->3, ...
        idx = torch.arange(N, device=z.device)
        pos_idx = idx ^ 1  # flips last bit: even->odd, odd->even

        # positive logits: sim[i, pos(i)]
        pos_logits = sim[idx, pos_idx]  # (N,)

        # build mask for negatives: exclude self and exclude positive
        neg_mask = torch.ones((N, N), dtype=torch.bool, device=z.device)
        neg_mask.fill_diagonal_(False)                 # exclude self
        neg_mask[idx, pos_idx] = False                 # exclude positive

        # collect negatives per anchor: (N, N-2)
        neg_logits = sim[neg_mask].view(N, N - 2)

        # loss per anchor: -( pos - logsumexp(negs) )
        loss_per = -(pos_logits - torch.logsumexp(neg_logits, dim=1))
        return loss_per.mean()



# -----------------------
#    Utilities
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def build_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size),
            T.centerCrop(image_size),
            T.ToTensor(),
        ]
    )


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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    num_batches = 0
    with torch.no_grad():
        for b_idx, (x, y) in enumerate(test_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = encoder(x)
            logits = linear_head(feats)
            loss = criterion(logits, y)

            test_loss += loss.item()
            num_batches += 1
            test_correct += (logits.argmax(dim=1) == y).sum().item()
            test_total += y.size(0)

    test_loss = test_loss / max(1, num_batches)
    test_acc = test_correct / max(1, test_total)

    if was_training:
        encoder.train()

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_classes": num_classes,
    }


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    ckpts = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("simclr_epoch_")]
    if not ckpts:
        return None

    def _epoch_num(path: str) -> int:
        base = os.path.basename(path)
        num_str = base.replace("simclr_epoch_", "").replace(".pth", "")
        try:
            return int(num_str)
        except ValueError:
            return -1

    ckpts = sorted(ckpts, key=_epoch_num)
    return ckpts[-1]


def _prepare_views(
    batch: Union[torch.Tensor, Sequence[torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes the batch output from the dataset into (view0, view1).
    Supports:
        - Tensor shape [B, W, C, H, Wimg] -> uses first/last frame as v0/v1.
        - Tuple/list of two tensors from EgocentricWindowDataset two-view mode.
    """
    if isinstance(batch, (list, tuple)):
        v0, v1 = batch
        # If dataset returns window stacks, select first/last frame.
        if v0.dim() == 5:
            v0 = v0[:, 0]
            v1 = v1[:, -1]
        return v0.to(device), v1.to(device)

    batch = batch.to(device)
    return batch[:, 0], batch[:, -1]


def _init_wandb(args: argparse.Namespace):
    """
    Initialize wandb if requested and available.
    Returns wandb.run or None.
    """
    if not getattr(args, "wandb", False):
        return None
    if wandb is None:
        print("[WARN] wandb requested but not installed; continuing without logging.")
        return None

    tags = args.wandb_tags.split(",") if args.wandb_tags else []
    tags = [t for t in tags if t]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=tags,
        mode=args.wandb_mode,
        config=vars(args),
    )
    return run


# -----------------------
#    Training
# -----------------------
def train_simclr_from_buffer(
    dataset,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    sub_batch_size: Optional[int] = None,
    lr: float = 1e-3,
    temperature: float = 0.5,
    proj_dim: int = 128,
    image_size: int = 224,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Optional[str] = None,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    max_batches_per_epoch: Optional[int] = None,
    num_workers: int = 4,
    linear_eval_every: int = 0,
    linear_eval_epochs: int = 5,
    linear_eval_batch_size: int = 256,
    linear_eval_lr: float = 0.1,
    linear_eval_weight_decay: float = 0.0,
    linear_eval_max_batches: Optional[int] = None,
    cifar_dataset: str = "cifar10",
    cifar_data_dir: str = "data/cifar",
    cifar_download: bool = False,
    wandb_run=None,
) -> Tuple[nn.Module, dict]:
    """
    Train SimCLR (ResNet50 + projection head) on a temporal buffer dataset.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SimCLRResNet18(proj_dim=proj_dim).to(device)
    criterion = PairedCosineTTLoss(temperature=temperature)

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": lr},
            {"params": model.projection_head.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    start_epoch = 1
    history = {"epoch_loss": []}
    scheduler_state_dict = None
    global_step = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if resume:
        if checkpoint_path is None and checkpoint_dir is not None:
            checkpoint_path = _find_latest_checkpoint(checkpoint_dir)

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            history = ckpt.get("history", {"epoch_loss": []})
            scheduler_state_dict = ckpt.get("scheduler_state_dict")
            global_step = ckpt.get("global_step", 0)
        else:
            print("[WARN] resume enabled but checkpoint not found; starting fresh.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        last_epoch=start_epoch - 2,  # step() will move to start_epoch-1
    )
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for b_idx, batch in enumerate(pbar):
            if max_batches_per_epoch is not None and b_idx >= max_batches_per_epoch:
                break

            v0, v1 = _prepare_views(batch, device)
            bsz = v0.size(0)

            grad_norm = None

            if sub_batch_size is None or sub_batch_size >= bsz:
                x = torch.cat([v0, v1], dim=0)
                _, z = model(x)
                z1, z2 = torch.chunk(z, 2, dim=0)  
                           

                # interleave: [z1_0, z2_0, z1_1, z2_1, ...] -> (2B, D)
                z_pairs = torch.stack([z1, z2], dim=1).reshape(-1, z1.size(1))

                loss = criterion(z_pairs)  # <-- loss expects single tensor

                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm).item()
                optimizer.step()

                batch_loss_value = loss.item()
            else:
                optimizer.zero_grad()
                total_loss = 0.0
                for start in range(0, bsz, sub_batch_size):
                    end = min(start + sub_batch_size, bsz)
                    sub_v0 = v0[start:end]
                    sub_v1 = v1[start:end]
                    sub_b = sub_v0.size(0)

                    x = torch.cat([sub_v0, sub_v1], dim=0)
                    _, z = model(x)
                    z1, z2 = torch.chunk(z, 2, dim=0)

                    z_pairs = torch.stack([z1, z2], dim=1).reshape(-1, z1.size(1))
                    loss_sub = criterion(z_pairs)

                    loss_scaled = loss_sub * (sub_b / bsz)
                    loss_scaled.backward()
                    total_loss += loss_sub.item() * (sub_b / bsz)

                if max_grad_norm is not None:
                    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm).item()
                optimizer.step()

                batch_loss_value = total_loss

            running_loss += batch_loss_value
            num_batches += 1
            global_step += 1
            pbar.set_postfix(loss=batch_loss_value)

            if wandb_run is not None:
                log_payload = {
                    "train/loss_batch": batch_loss_value,
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }
                if grad_norm is not None:
                    log_payload["train/grad_norm"] = grad_norm
                wandb_run.log(log_payload, step=global_step)

        scheduler.step()
        epoch_loss = running_loss / max(1, num_batches)
        history["epoch_loss"].append(epoch_loss)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{epochs} - mean loss: {epoch_loss:.4f} "
            f"- lr: {lr_now:.2e} - batches: {num_batches}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss_epoch": epoch_loss,
                    "train/lr": lr_now,
                    "train/batches": num_batches,
                    "train/epoch": epoch,
                },
                step=epoch,
            )

        if linear_eval_every and epoch % linear_eval_every == 0:
            eval_metrics = linear_eval_on_cifar(
                encoder=model.encoder,
                feature_dim=model.feature_dim,
                device=device,
                image_size=image_size,
                batch_size=linear_eval_batch_size,
                num_workers=num_workers,
                eval_epochs=linear_eval_epochs,
                lr=linear_eval_lr,
                weight_decay=linear_eval_weight_decay,
                data_dir=cifar_data_dir,
                dataset_name=cifar_dataset,
                download=cifar_download,
                max_batches=linear_eval_max_batches,
            )
            eval_metrics["epoch"] = epoch
            history.setdefault("linear_eval", []).append(eval_metrics)
            print(
                "Linear eval @ epoch {epoch}: train_acc={train_acc:.4f}, "
                "test_acc={test_acc:.4f}, train_loss={train_loss:.4f}, "
                "test_loss={test_loss:.4f}".format(**eval_metrics)
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "linear_eval/train_loss": eval_metrics["train_loss"],
                        "linear_eval/train_acc": eval_metrics["train_acc"],
                        "linear_eval/test_loss": eval_metrics["test_loss"],
                        "linear_eval/test_acc": eval_metrics["test_acc"],
                        "linear_eval/epoch": epoch,
                    },
                    step=epoch,
                )

        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, f"simclr_epoch_{epoch:03d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "proj_dim": proj_dim,
                    "backbone": "resnet18",
                    "history": history,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    if checkpoint_dir is not None:
        final_path = os.path.join(checkpoint_dir, "simclr_final.pth")
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "proj_dim": proj_dim,
                "backbone": "resnet18",
                "history": history,
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
            },
            final_path,
        )
        print(f"Saved final model: {final_path}")

    return model, history


# -----------------------
#    CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR training for egocentric windows (Slurm-friendly).")
    # Data
    parser.add_argument("--root-dir", type=str, default="Images_ego", help="Root folder with frame subfolders.")
    parser.add_argument("--window-size", type=int, default=5, help="Frames per window.")
    parser.add_argument("--frame-step", type=int, default=10, help="Stride in frames inside a window.")
    parser.add_argument("--two-view-offset", type=int, default=0, help="If >0, dataset returns (view0, view1) windows.")
    parser.add_argument("--use-object-focus", action="store_true", help="Blur background, keep object regions sharp.")
    parser.add_argument("--image-size", type=int, default=224, help="Final image side length after cropping.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")

    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sub-batch-size", type=int, default=None, help="Optional grad accumulation chunk size.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None, help="Trim batches per epoch for faster sweeps.")

    # Linear eval on CIFAR
    parser.add_argument("--linear-eval-every", type=int, default=0, help="Run linear eval every N epochs (0 disables).")
    parser.add_argument("--linear-eval-epochs", type=int, default=5, help="Epochs for the linear classifier.")
    parser.add_argument("--linear-eval-batch-size", type=int, default=256)
    parser.add_argument("--linear-eval-lr", type=float, default=0.1)
    parser.add_argument("--linear-eval-weight-decay", type=float, default=0.0)
    parser.add_argument("--linear-eval-max-batches", type=int, default=None, help="Trim batches in linear eval train/test.")
    parser.add_argument("--cifar-dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--cifar-data-dir", type=str, default="data/cifar")
    parser.add_argument("--cifar-download", action="store_true", help="Download CIFAR if missing.")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint-dir.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Resume from an explicit checkpoint path.")

    # Misc
    parser.add_argument("--device", type=str, default="auto", help="cuda|cpu|mps|auto")
    parser.add_argument("--seed", type=int, default=42)

    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="simclr-ego", help="wandb project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/org (optional).")
    parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name.")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated wandb tags.")
    parser.add_argument("--wandb-mode", type=str, default="online", help="wandb mode: online|offline|disabled.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    wandb_run = _init_wandb(args)

    transform = build_transforms(image_size=args.image_size)

    dataset = EgocentricWindowDataset(
        root_dir=args.root_dir,
        window_size=args.window_size,
        frame_step=args.frame_step,
        transform=transform,    
        return_stack=True,
        verbose=True,
        use_object_focus=args.use_object_focus,
        two_view_offset=args.two_view_offset,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.root_dir}. Check path or window parameters.")

    train_simclr_from_buffer(
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sub_batch_size=args.sub_batch_size,
        lr=args.lr,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        image_size=args.image_size,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        max_batches_per_epoch=args.max_batches_per_epoch,
        num_workers=args.num_workers,
        linear_eval_every=args.linear_eval_every,
        linear_eval_epochs=args.linear_eval_epochs,
        linear_eval_batch_size=args.linear_eval_batch_size,
        linear_eval_lr=args.linear_eval_lr,
        linear_eval_weight_decay=args.linear_eval_weight_decay,
        linear_eval_max_batches=args.linear_eval_max_batches,
        cifar_dataset=args.cifar_dataset,
        cifar_data_dir=args.cifar_data_dir,
        cifar_download=args.cifar_download,
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()
