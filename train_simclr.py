#!/usr/bin/env python3
"""
Train SimCLR on egocentric frame windows with Slurm-friendly CLI arguments.
The script mirrors the notebook logic but adds argparse, checkpointing/resume,
sub-batch gradient accumulation, and max-batches-per-epoch trimming so it can
run cleanly inside sbatch or job arrays.
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.cifar_linear_eval import linear_eval_on_cifar
from losses.paired_cosine_tt import PairedCosineTTLoss
from models.simclr_resnet import SimCLRResNet18
from utils.EgocentricWindowDataset_new import EgocentricWindowDataset


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
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    ckpts = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("simclr_epoch_")
    ]
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
        if v0.dim() == 5:
            v0 = v0[:, 0]
            v1 = v1[:, -1]
        return v0.to(device), v1.to(device)

    batch = batch.to(device)
    return batch[:, 0], batch[:, -1]


def _coerce_metric_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _infer_epoch(metrics: Mapping[str, object]) -> Optional[int]:
    for key in ("train/epoch", "linear_eval/epoch", "epoch"):
        if key in metrics:
            try:
                return int(metrics[key])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
    return None


class CSVLogger:
    def __init__(self, path: str, append: bool = True) -> None:
        self.path = path
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_exists = os.path.isfile(path)
        file_has_content = file_exists and os.path.getsize(path) > 0
        mode = "a" if append else "w"
        self._file = open(path, mode, newline="")
        self._writer = csv.writer(self._file)
        if not append or not file_has_content:
            self._writer.writerow(["step", "epoch", "key", "value"])
            self._file.flush()

    def log(self, metrics: Mapping[str, object], step: Optional[int] = None, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = _infer_epoch(metrics)
        for key, value in metrics.items():
            self._writer.writerow([step, epoch, key, _coerce_metric_value(value)])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _resolve_resume_checkpoint(args: argparse.Namespace) -> Optional[str]:
    if not getattr(args, "resume", False):
        return None

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None and args.checkpoint_dir is not None:
        checkpoint_path = _find_latest_checkpoint(args.checkpoint_dir)

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        return None
    return checkpoint_path


def _init_csv_logger(args: argparse.Namespace) -> Optional[CSVLogger]:
    if not getattr(args, "csv_log", False):
        return None

    log_path = args.csv_log_path
    if not log_path:
        if args.checkpoint_dir:
            log_path = os.path.join(args.checkpoint_dir, "metrics.csv")
        else:
            log_path = os.path.join("logs", "simclr_metrics.csv")

    resume_ckpt = _resolve_resume_checkpoint(args)
    append = resume_ckpt is not None
    if not append and os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
        print(f"[INFO] Fresh run detected; overwriting existing CSV at {log_path}")
    elif append and not os.path.isfile(log_path):
        print(f"[WARN] Resume requested but CSV log not found; creating new file at {log_path}")

    print(f"[INFO] CSV logging to {log_path}")
    return CSVLogger(log_path, append=append)


# -----------------------
#    Config + Trainer
# -----------------------
@dataclass
class LinearEvalConfig:
    every: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    max_batches: Optional[int]
    train_fraction: float
    split_seed: int
    dataset: str
    data_dir: str
    download: bool


class SimCLRTrainer:
    """Object-oriented trainer for SimCLR with optional CIFAR linear eval."""

    def __init__(
        self,
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
        max_grad_norm: Optional[float] = 1.0,
        checkpoint_dir: Optional[str] = None,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
        max_batches_per_epoch: Optional[int] = None,
        num_workers: int = 4,
        linear_eval: Optional[LinearEvalConfig] = None,
        csv_logger: Optional[CSVLogger] = None,
    ) -> None:
        self.dataset = dataset
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.sub_batch_size = sub_batch_size
        self.lr = lr
        self.temperature = temperature
        self.proj_dim = proj_dim
        self.image_size = image_size
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.checkpoint_path = checkpoint_path
        self.max_batches_per_epoch = max_batches_per_epoch
        self.num_workers = num_workers
        self.linear_eval = linear_eval
        self.csv_logger = csv_logger

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.model = SimCLRResNet18(proj_dim=self.proj_dim).to(self.device)
        self.criterion = PairedCosineTTLoss(temperature=self.temperature)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": self.lr},
                {"params": self.model.projection_head.parameters(), "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )

        self.start_epoch = 1
        self.history = {"epoch_loss": []}
        self.scheduler_state_dict = None
        self.global_step = 0

        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._maybe_resume()
        self.scheduler = self._build_scheduler()
        if self.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(self.scheduler_state_dict)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            last_epoch=self.start_epoch - 2,
        )

    def _maybe_resume(self) -> None:
        if not self.resume:
            return

        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None and self.checkpoint_dir is not None:
            checkpoint_path = _find_latest_checkpoint(self.checkpoint_dir)

        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            print("[WARN] resume enabled but checkpoint not found; starting fresh.")
            return

        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.history = ckpt.get("history", {"epoch_loss": []})
        self.scheduler_state_dict = ckpt.get("scheduler_state_dict")
        self.global_step = ckpt.get("global_step", 0)

    def _train_batch(self, v0: torch.Tensor, v1: torch.Tensor) -> Tuple[float, Optional[float]]:
        bsz = v0.size(0)
        grad_norm = None

        if self.sub_batch_size is None or self.sub_batch_size >= bsz:
            x = torch.cat([v0, v1], dim=0)
            _, z = self.model(x)
            z1, z2 = torch.chunk(z, 2, dim=0)

            z_pairs = torch.stack([z1, z2], dim=1).reshape(-1, z1.size(1))
            loss = self.criterion(z_pairs)

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
            self.optimizer.step()

            return loss.item(), grad_norm

        self.optimizer.zero_grad()
        total_loss = 0.0
        for start in range(0, bsz, self.sub_batch_size):
            end = min(start + self.sub_batch_size, bsz)
            sub_v0 = v0[start:end]
            sub_v1 = v1[start:end]
            sub_b = sub_v0.size(0)

            x = torch.cat([sub_v0, sub_v1], dim=0)
            _, z = self.model(x)
            z1, z2 = torch.chunk(z, 2, dim=0)

            z_pairs = torch.stack([z1, z2], dim=1).reshape(-1, z1.size(1))
            loss_sub = self.criterion(z_pairs)

            loss_scaled = loss_sub * (sub_b / bsz)
            loss_scaled.backward()
            total_loss += loss_sub.item() * (sub_b / bsz)

        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
        self.optimizer.step()

        return total_loss, grad_norm

    def _train_epoch(self, epoch: int) -> Tuple[float, int]:
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
        for b_idx, batch in enumerate(pbar):
            if self.max_batches_per_epoch is not None and b_idx >= self.max_batches_per_epoch:
                break

            v0, v1 = _prepare_views(batch, self.device)
            batch_loss_value, grad_norm = self._train_batch(v0, v1)

            running_loss += batch_loss_value
            num_batches += 1
            self.global_step += 1
            pbar.set_postfix(loss=batch_loss_value)

            if self.csv_logger is not None:
                log_payload = {
                    "train/loss_batch": batch_loss_value,
                    "train/epoch": epoch,
                    "train/global_step": self.global_step,
                }
                if grad_norm is not None:
                    log_payload["train/grad_norm"] = grad_norm
                self.csv_logger.log(log_payload, step=self.global_step, epoch=epoch)

        epoch_loss = running_loss / max(1, num_batches)
        return epoch_loss, num_batches

    def _run_linear_eval(self, epoch: int) -> None:
        if self.linear_eval is None:
            return
        if not self.linear_eval.every or epoch % self.linear_eval.every != 0:
            return

        eval_metrics = linear_eval_on_cifar(
            encoder=self.model.encoder,
            feature_dim=self.model.feature_dim,
            device=self.device,
            image_size=self.image_size,
            batch_size=self.linear_eval.batch_size,
            num_workers=self.num_workers,
            eval_epochs=self.linear_eval.epochs,
            lr=self.linear_eval.lr,
            weight_decay=self.linear_eval.weight_decay,
            data_dir=self.linear_eval.data_dir,
            dataset_name=self.linear_eval.dataset,
            download=self.linear_eval.download,
            train_fraction=self.linear_eval.train_fraction,
            split_seed=self.linear_eval.split_seed,
            max_batches=self.linear_eval.max_batches,
        )
        eval_metrics["epoch"] = epoch
        self.history.setdefault("linear_eval", []).append(eval_metrics)

        print(
            "Linear eval @ epoch {epoch}: train_acc={train_acc:.4f}, "
            "val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, "
            "train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            "test_loss={test_loss:.4f}".format(**eval_metrics)
        )

        if self.csv_logger is not None:
            self.csv_logger.log(
                {
                    "linear_eval/train_loss": eval_metrics["train_loss"],
                    "linear_eval/train_acc": eval_metrics["train_acc"],
                    "linear_eval/val_loss": eval_metrics["val_loss"],
                    "linear_eval/val_acc": eval_metrics["val_acc"],
                    "linear_eval/test_loss": eval_metrics["test_loss"],
                    "linear_eval/test_acc": eval_metrics["test_acc"],
                    "linear_eval/epoch": epoch,
                },
                step=epoch,
                epoch=epoch,
            )

    def _save_checkpoint(self, epoch: int) -> None:
        if self.checkpoint_dir is None:
            return

        ckpt_path = os.path.join(self.checkpoint_dir, f"simclr_epoch_{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "proj_dim": self.proj_dim,
                "backbone": "resnet18",
                "history": self.history,
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    def _save_final(self) -> None:
        if self.checkpoint_dir is None:
            return

        final_path = os.path.join(self.checkpoint_dir, "simclr_final.pth")
        torch.save(
            {
                "epoch": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "proj_dim": self.proj_dim,
                "backbone": "resnet18",
                "history": self.history,
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            final_path,
        )
        print(f"Saved final model: {final_path}")

    def fit(self) -> Tuple[nn.Module, dict]:
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss, num_batches = self._train_epoch(epoch)

            self.scheduler.step()
            self.history["epoch_loss"].append(epoch_loss)

            lr_now = self.scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{self.epochs} - mean loss: {epoch_loss:.4f} "
                f"- lr: {lr_now:.2e} - batches: {num_batches}"
            )

            if self.csv_logger is not None:
                self.csv_logger.log(
                    {
                        "train/loss_epoch": epoch_loss,
                        "train/lr": lr_now,
                        "train/batches": num_batches,
                        "train/epoch": epoch,
                    },
                    step=epoch,
                    epoch=epoch,
                )

            self._run_linear_eval(epoch)
            self._save_checkpoint(epoch)

        self._save_final()
        return self.model, self.history


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
    max_grad_norm: Optional[float] = 1.0,
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
    linear_eval_train_fraction: float = 0.7,
    linear_eval_split_seed: int = 42,
    cifar_dataset: str = "cifar10",
    cifar_data_dir: str = "data/cifar",
    cifar_download: bool = False,
    csv_logger: Optional[CSVLogger] = None,
) -> Tuple[nn.Module, dict]:
    linear_eval_cfg = LinearEvalConfig(
        every=linear_eval_every,
        epochs=linear_eval_epochs,
        batch_size=linear_eval_batch_size,
        lr=linear_eval_lr,
        weight_decay=linear_eval_weight_decay,
        max_batches=linear_eval_max_batches,
        train_fraction=linear_eval_train_fraction,
        split_seed=linear_eval_split_seed,
        dataset=cifar_dataset,
        data_dir=cifar_data_dir,
        download=cifar_download,
    )

    trainer = SimCLRTrainer(
        dataset=dataset,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        sub_batch_size=sub_batch_size,
        lr=lr,
        temperature=temperature,
        proj_dim=proj_dim,
        image_size=image_size,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        checkpoint_path=checkpoint_path,
        max_batches_per_epoch=max_batches_per_epoch,
        num_workers=num_workers,
        linear_eval=linear_eval_cfg,
        csv_logger=csv_logger,
    )
    return trainer.fit()


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
    parser.add_argument(
        "--linear-eval-train-fraction",
        type=float,
        default=0.7,
        help="Fraction of CIFAR train split used to fit the linear head (rest is validation).",
    )
    parser.add_argument(
        "--linear-eval-split-seed",
        type=int,
        default=42,
        help="Seed for the CIFAR train/validation split used during linear eval.",
    )
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

    # CSV logging
    parser.add_argument("--csv-log", action="store_true", help="Enable CSV logging.")
    parser.add_argument(
        "--csv-log-path",
        type=str,
        default=None,
        help="Path to CSV log file (defaults to checkpoint dir if set).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    csv_logger = _init_csv_logger(args)

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

    try:
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
            linear_eval_train_fraction=args.linear_eval_train_fraction,
            linear_eval_split_seed=args.linear_eval_split_seed,
            cifar_dataset=args.cifar_dataset,
            cifar_data_dir=args.cifar_data_dir,
            cifar_download=args.cifar_download,
            csv_logger=csv_logger,
        )
    finally:
        if csv_logger is not None:
            csv_logger.close()


if __name__ == "__main__":
    main()
