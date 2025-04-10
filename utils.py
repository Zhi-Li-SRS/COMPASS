import random
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rampy as rp
import torch
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from sklearn.manifold import TSNE

from dataload import load_data


def airpls_baseline(y: np.ndarray, lam: float = 1e3, niter: int = 15, tol: float = 1e-3):
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) baseline correction"""
    y = y.copy()
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            break

        max_residual = np.max(np.abs(dn))
        if max_residual < tol:
            break

        w_new = np.exp(2 * (d / dn.mean()))
        w = np.where(d < 0, w_new, 0)
        w = np.clip(w, 1e-6, 1e6)

    return y - z, z


def modpoly_baseline(spectrum, poly_order=2, max_iter=50, tol=1e-2):
    """
    Modified polynomial baseline correction.
    Iteratively fits polynomial to points likely to be baseline.

    Args:
        spectrum: Input spectrum array
        poly_order: Order of polynomial to fit
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        corrected: Baseline-corrected spectrum
        baseline: Estimated baseline
    """
    x = np.arange(len(spectrum))
    y = spectrum.copy()
    x_norm = (x - x.min()) / (x.max() - x.min())
    window_size = len(x) // 10
    if window_size % 2 == 0:
        window_size += 1
    y_smooth = np.convolve(y, np.ones(window_size) / window_size, mode="valid")
    y_smooth = np.pad(y_smooth, (window_size // 2, window_size // 2), mode="edge")
    coeff = np.polyfit(x_norm, y_smooth, poly_order)
    baseline = np.polyval(coeff, x_norm)

    prev_chisq = float("inf")
    for _ in range(max_iter):
        residuals = y - baseline

        weights = 1.0 / (1.0 + np.exp(2.0 * residuals / np.std(residuals)))
        weights = np.maximum(weights, 0.1)

        # Weighted polynomial fit
        coeff = np.polyfit(x_norm, y, poly_order, w=weights)
        baseline = np.polyval(coeff, x_norm)

        # Check convergence
        chisq = np.sum(weights * (y - baseline) ** 2)
        if abs(chisq - prev_chisq) < tol:
            break
        prev_chisq = chisq

    corrected = y - baseline
    corrected = np.maximum(corrected, 0)

    return corrected, baseline


def normalize_spectrum(spectrum: np.ndarray):
    """Normalize spectrum to max intensity"""
    if np.all(spectrum == 0):
        return spectrum
    return spectrum / np.max(spectrum) if np.max(spectrum) != 0 else spectrum


def smooth_spectrum(spectrum: np.ndarray, lamda=5):
    """Smooth spectrum using Whittaker smoother"""
    if np.all(spectrum == 0):
        return spectrum
    smoothed = rp.smooth(np.arange(len(spectrum)), spectrum, method="whittaker", Lambda=lamda)
    return rp.smooth(np.arange(len(spectrum)), smoothed, method="whittaker", Lambda=lamda / 2)



def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_tsne(
    data: np.ndarray,
    labels: None,
    figsize: tuple = (8, 6),
    point_size: int = 20,
    alpha: float = 0.6,
    title: str = "t-SNE Embedding",
    save_path: str = None,
    random_state: int = 42,
):
    """
    Create and plot a 2D UMAP embedding of high-dimensional data.
    """

    reducer = TSNE(n_components=2, random_state=random_state)

    embedding = reducer.fit_transform(data)
    plt.figure(figsize=figsize)
    plt.style.use("seaborn-v0_8-pastel")
    plt.rcParams.update({"font.size": 14, "font.family": "Arial", "font.weight": "bold"})
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[: len(unique_labels)]

        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0], embedding[mask, 1], c=[colors[i]], label=label, s=point_size, alpha=alpha
            )
        plt.legend(loc="best", prop={"size": 10, "weight": "bold", "family": "Arial"})
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=point_size, alpha=alpha)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE 1", fontsize=14, fontweight="bold")
    plt.ylabel("t-SNE 2", fontsize=14, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class GradCAM:
    """Grad-CAM implementation for 1D spectral data"""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM for the input spectrum

        Args:
            input_tensor: Input spectrum tensor of shape (1, 1, n_features)
            target_class: Target class index. If None, uses the predicted class

        Returns:
            cam: Normalized Grad-CAM of shape (n_features,)
        """
        input_length = input_tensor.shape[-1]
        input_tensor.requires_grad_(True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()  # the highest predicted class index

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1  # one-hot encoding of the target class

        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.detach().cpu().numpy()[0]  # (C, L)
        features = self.features.detach().cpu().numpy()[0]  # (C, L)

        weights = np.mean(gradients, axis=1)  # (C,)

        cam = np.sum(weights[:, np.newaxis] * features, axis=0)  # (L,)
        cam = np.maximum(cam, 0)

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        x_original = np.linspace(0, 1, input_length)
        x_cam = np.linspace(0, 1, len(cam))

        f = interp1d(x_cam, cam, kind="cubic", bounds_error=False, fill_value=0)
        cam = f(x_original)
        cam = np.clip(cam, 0, 0.999)
        return cam

    def __del__(self):
        self.remove_hooks()


if __name__ == "__main__":
    file_path = "Raman_dataset/train_data.csv"
    df = pd.read_csv(file_path)
    labels = df["name"].values
    features = df.drop("name", axis=1).values
    plot_tsne(features, labels=labels, save_path="figures/tsne_aug.png")
