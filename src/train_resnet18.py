from __future__ import annotations

# ============================================================
# 1) Imports
# ============================================================
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# 2) Constants / configuration
# ============================================================
SEED = 42

EXPECTED_CLASSES = ["bacterial", "fungal"]
EXPECTED_CLASS_TO_IDX = {"bacterial": 0, "fungal": 1}

BATCH_SIZE = 32
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-4

IMAGE_SIZE = 224
RESIZE_SIZE = 256

# Light fine-tuning setup:
# - layer4 gets a smaller LR (more stable updates on pretrained backbone)
# - fc gets a higher LR (new classifier needs faster learning)
LR_LAYER4 = 1e-5
LR_FC = 1e-4
WEIGHT_DECAY = 1e-4


# ============================================================
# 3) Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 4) Path helpers
# ============================================================
def get_paths() -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "split"

    return {
        "project_root": project_root,
        "data_root": data_root,
        "train": data_root / "train",
        "val": data_root / "val",
        "test": data_root / "test",
        "models_dir": project_root / "outputs" / "models",
        "logs_dir": project_root / "outputs" / "logs",
        "best_ckpt": project_root / "outputs" / "models" / "best_resnet18.pth",
        "last_ckpt": project_root / "outputs" / "models" / "last_resnet18.pth",
        "history_csv": project_root / "outputs" / "logs" / "train_history.csv",
        "test_metrics_txt": project_root / "outputs" / "logs" / "test_metrics.txt",
    }


# ============================================================
# 5) Dataset validation helpers
# ============================================================
def validate_dataset_folders(paths: Dict[str, Path]) -> None:
    for split_name in ("train", "val", "test"):
        split_path = paths[split_name]

        if not split_path.exists():
            raise FileNotFoundError(f"Missing split folder: {split_path}")
        if not split_path.is_dir():
            raise NotADirectoryError(f"Split path is not a directory: {split_path}")

        class_dirs = sorted([p.name for p in split_path.iterdir() if p.is_dir()])
        if class_dirs != EXPECTED_CLASSES:
            raise ValueError(
                f"{split_name} classes must be exactly {EXPECTED_CLASSES}, found {class_dirs}"
            )

        for class_name in EXPECTED_CLASSES:
            class_path = split_path / class_name
            if not class_path.exists():
                raise FileNotFoundError(f"Missing class folder: {class_path}")


# ============================================================
# 6) Transforms
# ============================================================
def build_transforms() -> Dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training transform: mild, explainable augmentation to improve robustness.
    train_tfms = transforms.Compose(
        [
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    # Validation/Test transform: deterministic pipeline for stable evaluation.
    # Keep this exactly aligned with src/plot_training.py inference transform.
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


# ============================================================
# 7) Dataset and DataLoader builders
# ============================================================
def build_datasets(
    paths: Dict[str, Path], tfms: Dict[str, transforms.Compose]
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_ds = datasets.ImageFolder(root=str(paths["train"]), transform=tfms["train"])
    val_ds = datasets.ImageFolder(root=str(paths["val"]), transform=tfms["val"])
    test_ds = datasets.ImageFolder(root=str(paths["test"]), transform=tfms["test"])
    return train_ds, val_ds, test_ds


def validate_imagefolder_mappings(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    test_ds: datasets.ImageFolder,
) -> None:
    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if len(dataset) == 0:
            raise ValueError(f"{split_name} dataset is empty.")

        if dataset.classes != EXPECTED_CLASSES:
            raise ValueError(
                f"{split_name} classes mismatch. Expected {EXPECTED_CLASSES}, got {dataset.classes}"
            )

        if dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
            raise ValueError(
                f"{split_name} class_to_idx mismatch. "
                f"Expected {EXPECTED_CLASS_TO_IDX}, got {dataset.class_to_idx}"
            )


def build_dataloaders(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    test_ds: datasets.ImageFolder,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ============================================================
# 8) Dataset overview / sanity checks
# ============================================================
def get_split_counts(dataset: datasets.ImageFolder, class_to_idx: Dict[str, int]) -> Dict[str, int]:
    counts = Counter(dataset.targets)
    return {class_name: counts[class_idx] for class_name, class_idx in class_to_idx.items()}


def print_dataset_overview(
    train_ds: datasets.ImageFolder, val_ds: datasets.ImageFolder, test_ds: datasets.ImageFolder
) -> None:
    print("\nDataset overview")
    print(f"train size: {len(train_ds)}")
    print(f"val size:   {len(val_ds)}")
    print(f"test size:  {len(test_ds)}")
    print(f"class_to_idx: {train_ds.class_to_idx}")


def validate_sample_batch(train_loader: DataLoader, device: torch.device) -> None:
    images, labels = next(iter(train_loader))
    print("\nSample batch check")
    print(f"images shape: {tuple(images.shape)}")
    print(f"labels shape: {tuple(labels.shape)}")
    unique_labels = sorted(set(labels.tolist()))
    print(f"unique labels in sample batch: {unique_labels}")

    expected_shape_suffix = (3, IMAGE_SIZE, IMAGE_SIZE)
    if images.ndim != 4 or images.shape[1:] != expected_shape_suffix:
        raise ValueError(f"Unexpected image batch shape: {tuple(images.shape)}")

    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Sample batch labels must be in {{0,1}}, got {unique_labels}")

    # Quick device move sanity check.
    _ = images.to(device)
    _ = labels.to(device)


# ============================================================
# 9) Model builder
# ============================================================
def build_model(device: torch.device) -> nn.Module:
    # Load pretrained ResNet18.
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # Light fine-tuning:
    # unfreeze only layer4 + final fc layer.
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)
    return model


def print_model_trainability(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable parameters: {trainable}")
    print(f"frozen parameters:    {frozen}")


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    # Different learning rates for layer4 and fc.
    return optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": LR_LAYER4},
            {"params": model.fc.parameters(), "lr": LR_FC},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def get_class_weights(train_counts: Dict[str, int], class_to_idx: Dict[str, int]) -> torch.Tensor:
    # Weighted loss helps because fungal class is the majority class.
    num_classes = len(class_to_idx)
    total = sum(train_counts.values())
    weights = [0.0] * num_classes
    for class_name, class_idx in class_to_idx.items():
        class_count = train_counts[class_name]
        weights[class_idx] = total / (num_classes * class_count)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# 10) Metrics helpers
# ============================================================
def compute_confusion_matrix(
    y_true: List[int], y_pred: List[int], num_classes: int
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def compute_classification_metrics(
    cm: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"per_class": {}}
    total = int(cm.sum())
    correct = int(np.trace(cm))
    overall_accuracy = (correct / total) if total > 0 else 0.0
    metrics["overall_accuracy"] = overall_accuracy

    f1_values: List[float] = []
    for idx, class_name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["per_class"][class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(cm[idx, :].sum()),
        }
        f1_values.append(f1)

    metrics["macro_f1"] = float(np.mean(f1_values)) if f1_values else 0.0
    return metrics


# ============================================================
# 11) Train function
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


# ============================================================
# 12) Validation/evaluation function
# ============================================================
@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if collect_predictions:
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc, all_targets, all_preds


# ============================================================
# 13) Checkpoint and logging helpers
# ============================================================
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    class_to_idx: Dict[str, int],
    train_counts: Dict[str, int],
    val_counts: Dict[str, int],
    hyperparameters: Dict[str, Any],
    best_val_loss: float,
    best_val_acc: float,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "class_to_idx": class_to_idx,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "hyperparameters": hyperparameters,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
    }
    torch.save(checkpoint, path)


def save_history_csv(path: Path, history: List[Dict[str, float]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "lr_layer4",
        "lr_fc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_test_metrics_txt(
    path: Path,
    test_loss: float,
    test_acc: float,
    cm: np.ndarray,
    class_names: List[str],
    metrics: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("Test Evaluation (Best Checkpoint)")
    lines.append("=" * 50)
    lines.append(f"test_loss: {test_loss:.6f}")
    lines.append(f"test_accuracy: {test_acc:.6f}")
    lines.append(f"overall_accuracy: {metrics['overall_accuracy']:.6f}")
    lines.append(f"macro_f1: {metrics['macro_f1']:.6f}")
    lines.append("")
    lines.append("Confusion Matrix (rows=true, cols=pred)")
    lines.append("Classes: " + ", ".join(class_names))
    lines.append(str(cm))
    lines.append("")
    lines.append("Per-class metrics")
    for class_name in class_names:
        row = metrics["per_class"][class_name]
        lines.append(
            f"{class_name}: "
            f"precision={row['precision']:.6f}, "
            f"recall={row['recall']:.6f}, "
            f"f1={row['f1_score']:.6f}, "
            f"support={row['support']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 14) Main training pipeline
# ============================================================
def main() -> None:
    set_seed(SEED)
    paths = get_paths()

    # Required folder checks and output folder creation.
    validate_dataset_folders(paths)
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)

    # Build datasets/loaders.
    tfms = build_transforms()
    train_ds, val_ds, test_ds = build_datasets(paths, tfms)
    validate_imagefolder_mappings(train_ds, val_ds, test_ds)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, BATCH_SIZE
    )

    # Dataset sanity output.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print_dataset_overview(train_ds, val_ds, test_ds)
    validate_sample_batch(train_loader, device)

    train_counts = get_split_counts(train_ds, train_ds.class_to_idx)
    val_counts = get_split_counts(val_ds, val_ds.class_to_idx)
    print(f"train counts: {train_counts}")
    print(f"val counts:   {val_counts}")

    # Build model, optimizer, weighted loss.
    model = build_model(device)
    print_model_trainability(model)

    class_weights = get_class_weights(train_counts, train_ds.class_to_idx).to(device)
    print(f"class weights: {class_weights.detach().cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model)

    # Scheduler lowers LR when validation loss stops improving.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    hyperparameters: Dict[str, Any] = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "image_size": IMAGE_SIZE,
        "resize_size": RESIZE_SIZE,
        "lr_layer4": LR_LAYER4,
        "lr_fc": LR_FC,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "trainable_layers": ["layer4", "fc"],
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau(val_loss)",
        "loss": "CrossEntropyLoss(weighted)",
        "class_weights": class_weights.detach().cpu().tolist(),
    }

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nStarting training")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_one_epoch(
            model, val_loader, criterion, device, collect_predictions=False
        )

        # Step scheduler with validation loss.
        # We track LR before/after step to print updates explicitly.
        prev_lr_layer4 = float(optimizer.param_groups[0]["lr"])
        prev_lr_fc = float(optimizer.param_groups[1]["lr"])
        scheduler.step(val_loss)
        new_lr_layer4 = float(optimizer.param_groups[0]["lr"])
        new_lr_fc = float(optimizer.param_groups[1]["lr"])

        if (new_lr_layer4 != prev_lr_layer4) or (new_lr_fc != prev_lr_fc):
            print(
                "Learning rate updated -> "
                f"layer4: {prev_lr_layer4:.2e} -> {new_lr_layer4:.2e}, "
                f"fc: {prev_lr_fc:.2e} -> {new_lr_fc:.2e}"
            )

        current_lr_layer4 = new_lr_layer4
        current_lr_fc = new_lr_fc

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "lr_layer4": round(current_lr_layer4, 10),
                "lr_fc": round(current_lr_fc, 10),
            }
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"lr_layer4: {current_lr_layer4:.2e} | lr_fc: {current_lr_fc:.2e}"
        )

        # Best checkpoint is selected by validation loss for stable early-stopping behavior.
        if val_loss < (best_val_loss - EARLY_STOPPING_MIN_DELTA):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=paths["best_ckpt"],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                class_to_idx=train_ds.class_to_idx,
                train_counts=train_counts,
                val_counts=val_counts,
                hyperparameters=hyperparameters,
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
            )
        else:
            epochs_without_improvement += 1

        # Simple early stopping based on validation loss.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no val_loss improvement for {EARLY_STOPPING_PATIENCE} epochs)."
            )
            break

    # Save last checkpoint and training history CSV.
    final_epoch = history[-1]["epoch"] if history else 0
    save_checkpoint(
        path=paths["last_ckpt"],
        model=model,
        optimizer=optimizer,
        epoch=int(final_epoch),
        class_to_idx=train_ds.class_to_idx,
        train_counts=train_counts,
        val_counts=val_counts,
        hyperparameters=hyperparameters,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
    )
    save_history_csv(paths["history_csv"], history)

    # ============================================================
    # 15) Final test evaluation with best checkpoint
    # ============================================================
    best_checkpoint = torch.load(paths["best_ckpt"], map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate_one_epoch(
        model, test_loader, criterion, device, collect_predictions=True
    )

    class_names = [name for name, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    metrics = compute_classification_metrics(cm, class_names)

    save_test_metrics_txt(
        path=paths["test_metrics_txt"],
        test_loss=test_loss,
        test_acc=test_acc,
        cm=cm,
        class_names=class_names,
        metrics=metrics,
    )

    # ============================================================
    # 16) Summary prints
    # ============================================================
    print("\nTraining complete")
    print(f"best checkpoint: {paths['best_ckpt']}")
    print(f"last checkpoint: {paths['last_ckpt']}")
    print(f"history csv: {paths['history_csv']}")
    print(f"test metrics: {paths['test_metrics_txt']}")
    print(f"best epoch (by val_loss): {best_epoch}")
    print(f"best val loss: {best_val_loss:.6f}")
    print(f"best val accuracy: {best_val_acc:.6f}")
    print(f"test loss (best ckpt): {test_loss:.6f}")
    print(f"test accuracy (best ckpt): {test_acc:.6f}")
    print(f"macro F1 (test): {metrics['macro_f1']:.6f}")
    print(f"final class_to_idx: {train_ds.class_to_idx}")


if __name__ == "__main__":
    main()
