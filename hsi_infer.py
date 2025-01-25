import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rampy as rp
import tifffile
import torch
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from model import LipidNet


class HSIPredictor:
    def __init__(self, model_path: str, library_path: str, target: List[str]):
        """
        Initialize HSI predictor

        Args:
            model_path: Path to pretrained model weights
            library_path: Path to library CSV with reference spectra
            target_subtypes: List of subtypes to predict
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target = target
        self.prob_thresh = 0.75
        self.colors = {
            "d7-glucose": "#FF0000",  # Red
            "d2-fructose": "#00FF00",  # Green
            "d-tyrosine": "#0000FF",  # Blue
            "d-methionine": "#FF00FF",  # Magenta
            "d-leucine": "#FFD700",  # Gold
        }
        # Load pretrained model
        self.model = self._load_model(model_path)

        # Load reference data
        self.ref = self._load_references(library_path)

        self.wavenumbers = self.ref.columns[1:].astype(float).values
        self.target_indices = self._get_target_indices()

        self.original_spectra = None
        self.img_wavenumbers = None

    def _load_model(self, model_path: str):
        """Load pretrained model"""
        model = LipidNet(num_classes=18)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model.to(self.device)

    def _load_references(self, library_path: str):
        """Load reference spectra"""
        return pd.read_csv(library_path)

    def _get_target_indices(self):
        """Get indices of target subtypes in original model output"""
        name_to_idx = {name: idx for idx, name in enumerate(self.ref["name"].unique())}
        return {name: name_to_idx[name] for name in self.target}

    def baseline_als(self, y: np.ndarray, lam=10**4, p=0.05, niter=15):
        """Asymmetric Least Squares baseline correction"""
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        y = np.asarray(y)

        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        return y - z

    def normalize_spectrum(self, spectrum: np.ndarray):
        """Normalize spectrum to max intensity"""
        if np.all(spectrum == 0):
            return spectrum

        max_val = np.max(np.abs(spectrum))
        if max_val != 0:
            return spectrum / max_val
        return spectrum

    def smooth_spectrum(self, spectrum: np.ndarray, lamda=0.5):
        """Smooth spectrum using Whittaker smoother"""
        if np.all(spectrum == 0):
            return spectrum

        smoothed = rp.smooth(np.arange(len(spectrum)), spectrum, method="whittaker", Lambda=lamda)
        smoothed = rp.smooth(np.arange(len(spectrum)), smoothed, method="whittaker", Lambda=lamda / 2)
        return smoothed

    def preprocess_spectrum(self, spectrum: np.ndarray):
        """Preprocess a single spectrum"""
        if np.all(spectrum == 0):
            return spectrum

        # spectrum = self.baseline_als(spectrum)
        # spectrum = np.maximum(spectrum, 0)
        spectrum = self.normalize_spectrum(spectrum)
        # spectrum = self.smooth_spectrum(spectrum)

        return spectrum

    def interpolate_spectrum(self, spectrum: np.ndarray, orig_wavenumbers: np.ndarray):
        """Interpolate spectrum to match reference wavenumbers"""
        if np.all(spectrum == 0):
            return np.zeros(len(self.wavenumbers))

        f = interp1d(
            orig_wavenumbers,
            spectrum,
            kind="cubic",
            bounds_error=False,
            fill_value=(spectrum[0], spectrum[-1]),
        )
        return f(self.wavenumbers)

    def predict_hsi(self, image_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        image = tifffile.imread(image_path)
        image = np.flip(image, axis=0)  # Flip image to match reference

        assert len(image.shape) == 3, "Input image must be an image stack"
        N, height, width = image.shape
        batch_size = 4096
        self.img_wavenumbers = np.linspace(self.wavenumbers[0], self.wavenumbers[-1], N)
        self.original_spectra = image.reshape(N, -1).T

        background_mask = np.all(self.original_spectra == 0, axis=1)
        background_mask = background_mask.reshape(height, width)

        spectra = self.original_spectra.copy()
        processed_spectra = np.zeros((len(self.wavenumbers), height * width))
        predictions = []
        spectra_by_type = {name: [] for name in self.target}

        print("Processing spectra...")
        total_batches = (height * width + batch_size - 1) // batch_size

        for i in tqdm(range(0, height * width, batch_size), total=total_batches, desc="Batch Progress"):
            end_idx = min(i + batch_size, height * width)
            batch = spectra[i:end_idx]
            batch = np.array([self.preprocess_spectrum(s) for s in batch])
            processed_batch = np.array([self.interpolate_spectrum(s, self.img_wavenumbers) for s in batch])
            processed_spectra[:, i : i + len(processed_batch)] = processed_batch.T
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(processed_batch).to(self.device)
                batch_tensor = batch_tensor.unsqueeze(1)
                output = self.model(batch_tensor)  # (batch_size, num_classes)
                target_mask = torch.zeros_like(output, dtype=torch.bool)
                for _, idx in self.target_indices.items():
                    target_mask[:, idx] = True
                output = torch.where(target_mask, output, torch.tensor(-1e9).to(self.device))
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        for name, model_idx in self.target_indices.items():
            pred_probs = predictions[:, model_idx]
            print(f"{name}:")
            print(f"  Max prob: {np.max(pred_probs):.4f}")
            print(f"  Mean prob: {np.mean(pred_probs):.4f}")
            print(f"  Pixels > threshold: {np.sum(pred_probs > self.prob_thresh)}")

        prediction_maps = {}

        merged_map = np.zeros((height, width, 3))

        for name, model_idx in self.target_indices.items():
            prob_map = predictions[:, model_idx].reshape(height, width)
            prob_map = np.where(prob_map > self.prob_thresh, prob_map, 0)
            prob_map[background_mask] = 0
            prediction_maps[name] = prob_map

            # Save colored map
            color = (
                np.array(tuple(int(self.colors[name].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))) / 255.0
            )
            colored_map = np.zeros((height, width, 3))
            for i in range(3):
                colored_map[:, :, i] = prob_map * color[i]

            plt.figure(figsize=(10, 10))
            im = plt.imshow(colored_map)
            plt.title(f"{name} Mapping")
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"{name}_colored.png"), dpi=300, bbox_inches="tight")
            plt.close()

            merged_map += colored_map

            # Save grayscale TIFF
            tifffile.imwrite(os.path.join(output_dir, f"{name}_pred.tif"), (prob_map * 255).astype(np.uint8))

            prob_values = predictions[:, model_idx].copy()
            prob_values[background_mask.flatten()] = 0
            mask = prob_values > self.prob_thresh
            if np.any(mask):
                spectra_by_type[name] = np.where(mask)[0].tolist()
            else:
                spectra_by_type[name] = []

        # Save merged image
        merged_map = np.clip(merged_map, 0, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(merged_map)
        plt.title("Merged Mapping")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, "merged.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot spectral comparisons with matching colors
        self._plot_spectral_comparisons(spectra_by_type, output_dir)
        print("All visualizations saved!")

    def _plot_spectral_comparisons(self, spectra_by_type: Dict[str, List[np.ndarray]], output_dir: str):
        """Plot predicted spectra vs references"""
        n_types = len(self.target)
        fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
        plt.style.use("seaborn-v0_8-pastel")

        for idx, subtype in enumerate(self.target):
            ax = axes[idx] if n_types > 1 else axes
            color = self.colors[subtype]
            ref_spectrum = self.ref[self.ref["name"] == subtype].iloc[0, 1:].values
            ax.plot(self.wavenumbers, ref_spectrum, "k-", label=f"{subtype}", linewidth=2)

            if spectra_by_type[subtype]:
                pred_spectra = []
                for pixel_idx in spectra_by_type[subtype]:
                    raw_spectrum = self.original_spectra[pixel_idx]
                    # processed = self.baseline_als(raw_spectrum)
                    # processed = self.normalize_spectrum(processed)
                    # processed = self.smooth_spectrum(processed)
                    interpolated = self.interpolate_spectrum(raw_spectrum, self.img_wavenumbers)
                    pred_spectra.append(interpolated)

                pred_spectra = np.array(pred_spectra)
                mean_spectrum = np.mean(pred_spectra, axis=0)
                processed = self.baseline_als(mean_spectrum)
                processed = self.normalize_spectrum(processed)
                mean_spectrum = self.smooth_spectrum(processed)

                std_spectrum = np.std(pred_spectra, axis=0)

                ax.plot(self.wavenumbers, mean_spectrum, color=color, label=f"predicted", linewidth=2)
                # ax.fill_between(
                #     self.wavenumbers,
                #     mean_spectrum - std_spectrum,
                #     mean_spectrum + std_spectrum,
                #     color=color,
                #     alpha=0.2,
                # )

            ax.set_title(subtype)
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Normalized Intensity")
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spectral_comparisons.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="HSI Prediction")
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/best_model.pth", help="Path to pretrained model"
    )
    parser.add_argument(
        "--library_path", type=str, default="Raman_dataset/library.csv", help="Path to library CSV"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="HSI_data/Result of 0116Plin1-2-sweep-830-860-100.tif",
        help="Path to HSI image stack",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predicted_results/hsi_plin1_predict",
        help="Directory to save results",
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["d2-fructose", "d-tyrosine", "d-methionine", "d-leucine"],
        help="List of target subtypes to predict",
    )

    args = parser.parse_args()

    predictor = HSIPredictor(model_path=args.model_path, library_path=args.library_path, target=args.target)

    predictor.predict_hsi(args.image_path, args.output_dir)


if __name__ == "__main__":
    main()
