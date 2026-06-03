"""
utils/evaluate.py
Evaluation helpers — metrics, confusion matrix, comparison table.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall":    recall_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_training_history(histories: dict, save_path: str = None):
    """
    histories: {"LSTM": {val_acc: [...], ...}, "BERT": {...}, ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, h in histories.items():
        if "val_acc" in h:
            axes[0].plot(h["val_acc"], label=name, marker="o")
        if "val_loss" in h:
            axes[1].plot(h["val_loss"], label=name, marker="o")

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def model_comparison_table(results: dict):
    """
    results: {"LSTM": {accuracy, f1, ...}, "BERT": {...}, "RoBERTa": {...}}
    Prints a formatted comparison table.
    """
    header = f"{'Model':<12} {'Accuracy':>10} {'F1':>8} {'Precision':>11} {'Recall':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        print(f"{name:<12} {m['accuracy']:>10.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>11.4f} {m['recall']:>8.4f}")
    print("=" * len(header) + "\n")
