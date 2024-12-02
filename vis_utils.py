import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import auc, confusion_matrix, roc_curve


class Visualization:
    def __init__(self, save_dir: str):
        """
        Initialize visualization utilities.

        Args:
            save_dir: Directory to save plots and metrics
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-pastel")
        self.fig_size = (8, 6)
        self.dpi = 300
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 14, "font.weight": "bold"})

    def plot_train_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_accuracies: List[float],
        sigma: float = 2.0,
    ):
        """Plot training metrics with Gaussian smoothing."""
        plt.figure(figsize=self.fig_size)
        plt.style.use("seaborn-v0_8-pastel")

        ax1 = plt.gca()
        ax2 = ax1.twinx()

        train_losses_smooth = gaussian_filter1d(train_losses, sigma=sigma)
        val_losses_smooth = gaussian_filter1d(val_losses, sigma=sigma)
        val_accuracies_smooth = gaussian_filter1d(val_accuracies, sigma=sigma)

        # Plot smoothed data
        line1 = ax1.plot(train_losses_smooth, label="Train Loss", color="tab:blue", linestyle="-", alpha=0.6)
        line2 = ax1.plot(
            val_losses_smooth, label="Validation Loss", color="tab:red", linestyle="--", alpha=0.6
        )
        line3 = ax2.plot(
            val_accuracies_smooth, label="Validation Accuracy", color="tab:green", linestyle="-", alpha=0.6
        )

        ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Loss", fontsize=12, fontweight="bold", color="tab:blue", alpha=0.6)
        ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold", color="tab:green", alpha=0.6)
        plt.title("Train Metrics", fontsize=14, fontweight="bold", pad=20)

        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curves.png"), dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        plt.style.use("seaborn-v0_8-pastel")

        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            cmap="coolwarm",
            annot=True,
            square=True,  # Make cells square
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 8, "weight": "bold", "family": "DejaVu Serif"},
        )

        plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"), dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def plot_roc_curves(self, fpr: Dict, tpr: Dict, roc_auc: Dict, n_classes: int):
        """Plot ROC curves for each class."""
        plt.figure(figsize=(8, 6))
        plt.style.use("seaborn-v0_8-pastel")

        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})", alpha=0.7
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        plt.title("Receiver Operating Characteristic (ROC) Curves")
        plt.legend(loc="lower right", fontsize="small")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "roc_curves.png"), dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def save_classification_metrics(
        self, precision: float, recall: float, f1: float, class_names: List[str], class_metrics: Dict
    ):
        """Save classification metrics to CSV."""
        class_data = []
        for i, class_name in enumerate(class_names):
            class_data.append(
                {
                    "Class": class_name,
                    "Precision": class_metrics["precision"][i],
                    "Recall": class_metrics["recall"][i],
                    "F1-Score": class_metrics["f1"][i],
                }
            )
        df_class = pd.DataFrame(class_data)
        df_class.to_csv(os.path.join(self.save_dir, "per_class_metrics.csv"), index=False)


def plot_train_history(
    save_dir: str,
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    precision: float,
    recall: float,
    f1: float,
    class_metrics: Dict,
):
    """
    Wrapper function to create all visualizations and save metrics.
    """
    vis = Visualization(save_dir)
    vis.plot_train_curves(train_losses, val_losses, val_accuracies, sigma=2)
    vis.plot_confusion_matrix(y_true, y_pred, class_names)

    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(class_names)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    vis.plot_roc_curves(fpr, tpr, roc_auc, n_classes)
    vis.save_classification_metrics(precision, recall, f1, class_names, class_metrics)


def plot_original_spectra(file_path):
    """
    Plot original 18 spectra with 0.5 offset between each spectrum.
    Args:
        file_path(str): Path to the CSV file containing the original spectra.
    """
    df = pd.read_csv(file_path)
    wavenumbers = df.columns[1:].astype(float).values  # Skip the Name column
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:18]
    plt.figure(figsize=(6, 13))
    plt.style.use("ggplot")

    for i, row in df.iterrows():
        spectrum = row.values[1:]  # Skip the Name column
        offset = i * 0.7
        plt.plot(wavenumbers, spectrum + offset, label=row["Name"], color=colors[i])

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized Intensity (a.u.)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/original_spectra.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_augmented_spectra(file_path, n_samples=20):
    """
    Plot randomly selected augmented spectra for the first lipid type.

    Args:
        file_path(str): Path to the CSV file containing the augmented spectra.
        n_samples(int): Number of augmented spectra to plot.
    """
    df = pd.read_csv(file_path)

    first_lipid = df.iloc[0]["name"]  # get the fisrt lipid name
    wavenumbers = df.columns[1:].astype(float).values  # Skip the name column

    lipid_data = df[df["name"] == first_lipid]

    # Select random samples
    random_indices = np.random.choice(len(lipid_data), size=n_samples, replace=False)
    selected_data = lipid_data.iloc[random_indices]

    plt.figure(figsize=(6, 13))
    plt.style.use("ggplot")
    base_color = plt.cm.tab20(0)

    for i, idx in enumerate(random_indices):
        spectrum = selected_data.iloc[i, 1:].values  # Skip the name column
        offset = i * 0.5
        plt.plot(wavenumbers, spectrum + offset, color=base_color, label=f"Sample{i + 1}", alpha=0.7)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Normalized Intensity (a.u.)")
    plt.title(f"Augmented Spectra for {first_lipid}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/augmented_spectra.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    orig_file_path = "dataset/library.csv"
    aug_file_path = "dataset/data_aug.csv"
    plot_original_spectra(orig_file_path)
    plot_augmented_spectra(aug_file_path)
