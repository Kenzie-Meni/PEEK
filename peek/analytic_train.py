"""
Training utilities for Imagenette fine-tuning used by PEEK studies.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import build_imagenette_model, set_backbone_trainable
from .utils.paths import repo_path


@dataclass
class ImagenetteFineTuneConfig:
    model_name: str = "vgg16"
    data_root: str = "../datasets/imagenette2"
    checkpoint_out: str = "weights/vgg16_imagenette_best.pt"
    last_checkpoint_out: str = "weights/vgg16_imagenette_last.pt"
    history_out: str = "analysis/vgg16_imagenette/train_history.csv"
    batch_size: int = 64
    epochs: int = 100
    head_warmup_epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.10
    dropout: float = 0.2
    early_stop_patience: int = 10
    grad_clip_norm: float = 1.0
    random_seed: int = 1337
    num_workers: int = 4
    amp: bool = True
    freeze_backbone_during_warmup: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imagenette_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def imagenette_val_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _make_loaders(config: ImagenetteFineTuneConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    data_root = repo_path(config.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected Imagenette train/val folders under {data_root}, but they were not found."
        )

    train_set = datasets.ImageFolder(str(train_dir), transform=imagenette_train_transform())
    val_set = datasets.ImageFolder(str(val_dir), transform=imagenette_val_transform())

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_set.classes


@torch.no_grad()
def evaluate_imagenette_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


def train_imagenette_model(
    config: Optional[ImagenetteFineTuneConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Fine-tune an ImageNet-pretrained classification model on Imagenette2.
    """
    cfg = config or ImagenetteFineTuneConfig()
    set_seed(cfg.random_seed)

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, class_names = _make_loaders(cfg)

    model = build_imagenette_model(
        cfg.model_name,
        num_classes=len(class_names),
        pretrained=True,
        dropout=cfg.dropout,
    ).to(device_obj)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.scheduler_min_lr,
    )

    use_amp = bool(cfg.amp and device_obj.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    checkpoint_out = repo_path(cfg.checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_out = repo_path(cfg.last_checkpoint_out)
    last_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    history_out = repo_path(cfg.history_out)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    config_out = history_out.with_suffix(".json")

    for epoch in range(cfg.epochs):
        warmup_active = epoch < cfg.head_warmup_epochs and cfg.freeze_backbone_during_warmup
        set_backbone_trainable(model, cfg.model_name, not warmup_active)
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        start_time = time.time()

        for images, targets in train_loader:
            images = images.to(device_obj, non_blocking=True)
            targets = targets.to(device_obj, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == targets).sum().item())
            total_examples += batch_size

        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)
        val_metrics = evaluate_imagenette_model(model, val_loader, criterion, device_obj)
        scheduler.step(val_metrics["loss"])

        epoch_row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(time.time() - start_time),
            "warmup_backbone_frozen": float(1 if warmup_active else 0),
        }
        history.append(epoch_row)

        torch.save(model.state_dict(), last_checkpoint_out)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_out)
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"[{cfg.model_name} epoch {epoch + 1:03d}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if epochs_without_improvement >= cfg.early_stop_patience:
            if verbose:
                print(
                    f"Early stopping after {epoch + 1} epochs "
                    f"(no val_loss improvement for {cfg.early_stop_patience} epochs)."
                )
            break

    with history_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    config_out.write_text(json.dumps(asdict(cfg), indent=2))

    return {
        "best_checkpoint": checkpoint_out,
        "last_checkpoint": last_checkpoint_out,
        "history_csv": history_out,
        "config_json": config_out,
    }


def launch_imagenette_training_tmux(
    config: Optional[ImagenetteFineTuneConfig] = None,
    *,
    session_name: Optional[str] = None,
    env_python: Optional[Union[str, Path]] = None,
    workdir: Optional[Union[str, Path]] = None,
) -> dict[str, str]:
    """
    Launch an Imagenette fine-tuning job in a detached tmux session.
    """
    cfg = config or ImagenetteFineTuneConfig()
    workdir_p = repo_path(workdir or ".")
    python_exec = str(env_python or "/home/rwhite/.local/share/mamba/envs/yolo26/bin/python")
    session = session_name or f"peek_{cfg.model_name}_imagenette"

    run_dir = repo_path(Path(cfg.history_out).parent)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f"{session}_config.json"
    log_path = run_dir / f"{session}.log"
    config_path.write_text(json.dumps(asdict(cfg), indent=2))

    run_code = (
        "import json; "
        "from pathlib import Path; "
        "from peek.analytic_train import ImagenetteFineTuneConfig, train_imagenette_model; "
        f"cfg = ImagenetteFineTuneConfig(**json.loads(Path(r'{config_path}').read_text())); "
        "train_imagenette_model(cfg)"
    )
    tmux_cmd = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        f"cd {workdir_p} && {python_exec} -c \"{run_code}\" |& tee {log_path}",
    ]
    subprocess.run(tmux_cmd, check=True)

    return {
        "session_name": session,
        "config_json": str(config_path),
        "log_file": str(log_path),
        "attach_command": f"tmux attach -t {session}",
    }


def train_vgg16_dense_head_imagenette(
    config: Optional[ImagenetteFineTuneConfig] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
) -> dict[str, Path]:
    cfg = config or ImagenetteFineTuneConfig()
    cfg.model_name = "vgg16"
    return train_imagenette_model(cfg, device=device, verbose=verbose)


def launch_vgg16_dense_head_training_tmux(
    config: Optional[ImagenetteFineTuneConfig] = None,
    *,
    session_name: str = "peek_vgg16_imagenette",
    env_python: Optional[Union[str, Path]] = None,
    workdir: Optional[Union[str, Path]] = None,
) -> dict[str, str]:
    cfg = config or ImagenetteFineTuneConfig()
    cfg.model_name = "vgg16"
    return launch_imagenette_training_tmux(
        cfg,
        session_name=session_name,
        env_python=env_python,
        workdir=workdir,
    )
