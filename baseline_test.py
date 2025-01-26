from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import sparse
from scipy.sparse.linalg import spsolve


def airpls_baseline(
    y: np.ndarray, lam: float = 1e3, niter: int = 15, tol: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) baseline correction
    Reference: Zhang ZM, Chen S, Liang YZ. Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst. 2010.

    Parameters:
        y: Input spectrum
        lam: Smoothness parameter (larger = smoother)
        niter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        corrected: Baseline corrected spectrum
        baseline: Estimated baseline
    """
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


def load_and_process_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load HSI image and preprocess"""
    image = tifffile.imread(image_path)
    image = np.flip(image, axis=0)  # Flip image to match reference
    background_mask = np.all(image == 0, axis=0)
    return image, background_mask


def random_sample_spectra(image: np.ndarray, background_mask, n_samples: int = 20) -> np.ndarray:
    """Randomly sample pixel spectra"""
    N, height, width = image.shape
    pixels = image.reshape(N, -1).T
    valid_indices = np.where(~background_mask.flatten())[0]

    n_samples = min(n_samples, len(valid_indices))
    if n_samples == 0:
        raise ValueError("No valid pixels found for sampling")

    selected_indices = np.random.choice(valid_indices, n_samples, replace=False)
    return pixels[selected_indices]


def plot_comparison(spectra: np.ndarray, wavenumbers: np.ndarray, method_name: str):
    """Plot spectra with baselines"""
    plt.figure(figsize=(15, 25))

    for i in range(20):
        ax = plt.subplot(10, 2, i + 1)
        raw = spectra[i]

        # Apply baseline correction
        corrected, baseline = airpls_baseline(raw)

        # Plotting
        ax.plot(wavenumbers, raw, "k-", lw=1, alpha=0.7, label="Raw")
        ax.plot(wavenumbers, baseline, "r--", lw=1, alpha=0.9, label="Baseline")
        ax.plot(wavenumbers, corrected, "b-", lw=1, alpha=0.7, label="Corrected")

        ax.set_xlim(wavenumbers[0], wavenumbers[-1])
        ax.set_ylabel("Intensity")
        if i % 2 == 0:
            ax.legend(loc="upper right", fontsize=6)
        if i >= 18:
            ax.set_xlabel("Wavenumber (cm⁻¹)")

    plt.suptitle(f"Baseline Correction Comparison - {method_name}", y=0.92)
    plt.tight_layout()
    plt.savefig(f"baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    path = "HSI_data/Result of 0116Plin1-2-sweep-830-860-100.tif"
    wavenumbers = np.linspace(2000, 2300, 97)

    image, bg_mask = load_and_process_image(path)
    sampled_spectra = random_sample_spectra(image, bg_mask, n_samples=20)
    plot_comparison(sampled_spectra, wavenumbers, method_name="airPLS")
