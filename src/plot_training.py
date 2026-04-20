from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


EXPECTED_CLASSES = ["bacterial", "fungal"]
EXPECTED_CLASS_TO_IDX = {"bacterial": 0, "fungal": 1}
REQUIRED_HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
]


def get_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "history_csv": root / "outputs" / "logs" / "train_history.csv",
        "best_checkpoint": root / "outputs" / "models" / "best_resnet18.pth",
        "test_dir": root / "data" / "split" / "test",
        "plots_dir": root / "outputs" / "plots",
        "training_curves_png": root / "outputs" / "plots" / "training_curves.png",
        "loss_curve_png": root / "outputs" / "plots" / "loss_curve.png",
        "accuracy_curve_png": root / "outputs" / "plots" / "accuracy_curve.png",
        "confusion_matrix_png": root / "outputs" / "plots" / "confusion_matrix.png",
    }


def validate_inputs(paths: Dict[str, Path]) -> None:
    if not paths["history_csv"].exists():
        raise FileNotFoundError(f"Missing file: {paths['history_csv']}")
    if not paths["best_checkpoint"].exists():
        raise FileNotFoundError(f"Missing file: {paths['best_checkpoint']}")
    if not paths["test_dir"].exists():
        raise FileNotFoundError(f"Missing folder: {paths['test_dir']}")
    if not paths["test_dir"].is_dir():
        raise NotADirectoryError(f"Test path is not a folder: {paths['test_dir']}")

    class_folders = sorted([p.name for p in paths["test_dir"].iterdir() if p.is_dir()])
    if class_folders != EXPECTED_CLASSES:
        raise ValueError(
            f"Test class folders must be exactly {EXPECTED_CLASSES}, found {class_folders}"
        )

    paths["plots_dir"].mkdir(parents=True, exist_ok=True)


def load_training_history(history_csv_path: Path) -> pd.DataFrame:
    history = pd.read_csv(history_csv_path)
    missing_columns = [col for col in REQUIRED_HISTORY_COLUMNS if col not in history.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {history_csv_path}: {missing_columns}. "
            f"Required: {REQUIRED_HISTORY_COLUMNS}"
        )
    return history


def plot_training_curves(history: pd.DataFrame, paths: Dict[str, Path]) -> None:
    # Main combined dashboard: loss + accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: loss plot.
    axes[0].plot(history["epoch"], history["train_loss"], color="blue", label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], color="red", label="Validation Loss")
    axes[0].set_title("Loss During Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True)
    axes[0].legend()

    # Right: accuracy plot.
    axes[1].plot(
        history["epoch"], history["train_accuracy"], color="blue", label="Train Accuracy"
    )
    axes[1].plot(
        history["epoch"], history["val_accuracy"], color="red", label="Validation Accuracy"
    )
    axes[1].set_title("Accuracy During Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(paths["training_curves_png"], dpi=300)

    # Optional single plots for report usage.
    fig_loss, ax_loss = plt.subplots(figsize=(6, 5))
    ax_loss.plot(history["epoch"], history["train_loss"], color="blue", label="Train Loss")
    ax_loss.plot(history["epoch"], history["val_loss"], color="red", label="Validation Loss")
    ax_loss.set_title("Loss During Training")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.grid(True)
    ax_loss.legend()
    fig_loss.tight_layout()
    fig_loss.savefig(paths["loss_curve_png"], dpi=300)

    fig_acc, ax_acc = plt.subplots(figsize=(6, 5))
    ax_acc.plot(history["epoch"], history["train_accuracy"], color="blue", label="Train Accuracy")
    ax_acc.plot(history["epoch"], history["val_accuracy"], color="red", label="Validation Accuracy")
    ax_acc.set_title("Accuracy During Training")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True)
    ax_acc.legend()
    fig_acc.tight_layout()
    fig_acc.savefig(paths["accuracy_curve_png"], dpi=300)

    # Display the main training figure as requested.
    plt.figure(fig.number)
    plt.show()


def build_eval_model(device: torch.device) -> nn.Module:
    # Recreate ResNet18 with 2-class output head.
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.to(device)
    model.eval()
    return model


def build_test_loader(test_dir: Path) -> Tuple[DataLoader, datasets.ImageFolder]:
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_transforms)
    if test_dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
        raise ValueError(
            f"class_to_idx mismatch. Expected {EXPECTED_CLASS_TO_IDX}, got {test_dataset.class_to_idx}"
        )
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, np.ndarray, List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    test_accuracy = correct / total if total > 0 else 0.0

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return test_accuracy, cm, y_true, y_pred


def plot_confusion_matrix(cm: np.ndarray, paths: Dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    class_labels = EXPECTED_CLASSES
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    # Write values inside each confusion matrix cell.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > (cm.max() / 2 if cm.max() > 0 else 0) else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(paths["confusion_matrix_png"], dpi=300)
    plt.show()


def main() -> None:
    paths = get_paths()

    print("Running startup checks...")
    validate_inputs(paths)
    print("Dataset validation passed.")

    history = load_training_history(paths["history_csv"])
    print(f"Loaded training history: {paths['history_csv']}")

    plot_training_curves(history, paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, test_dataset = build_test_loader(paths["test_dir"])
    print(f"class_to_idx mapping: {test_dataset.class_to_idx}")

    model = build_eval_model(device)
    checkpoint = torch.load(paths["best_checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {paths['best_checkpoint']}")

    test_accuracy, confusion_matrix, _, _ = evaluate_on_test(model, test_loader, device)
    print(f"Final test accuracy: {test_accuracy:.6f}")
    print("Confusion matrix values:")
    print(confusion_matrix)

    plot_confusion_matrix(confusion_matrix, paths)

    print("Plots saved successfully:")
    print(f"- {paths['training_curves_png']}")
    print(f"- {paths['loss_curve_png']}")
    print(f"- {paths['accuracy_curve_png']}")
    print(f"- {paths['confusion_matrix_png']}")


if __name__ == "__main__":
    main()
