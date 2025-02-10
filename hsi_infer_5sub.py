import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

from model import LipidNet
from utils import modpoly_baseline, normalize_spectrum, smooth_spectrum


class HSIPredictor5Sub:
    def __init__(self, model_path: str, library_path: str, target: List[str]):
        """
        Initialize HSI predictor for 5 subtypes.
        """
        self.device = torch.device("cpu")
        self.target = target
        self.prob_thresh = 0.9
        self.colors = {
            "d7-glucose": "#FF0000",  # Red
            "d2-fructose": "#00FF00",  # Green
            "d-tyrosine": "#0000FF",  # Blue
            "d-methionine": "#FF00FF",  # Magenta
            "d-leucine": "#FFD700",  # Gold
        }
        # Load pretrained model (trained on 5 subtypes)
        self.model = self._load_model(model_path)
        # Load reference data
        self.ref = pd.read_csv(library_path)
        self.wavenumbers = self.ref.columns[1:].astype(float).values
        # Map target names to indices (assumes model output order matches self.target list)
        self.target_indices = {name: i for i, name in enumerate(self.target)}

        self.original_spectra = None
        self.img_wavenumbers = None

    def _load_model(self, model_path: str):
        """Load pretrained model"""
        model = LipidNet(num_classes=5)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model.to(self.device)

    def preprocess_spectrum(self, spectrum: np.ndarray):
        if np.all(spectrum == 0):
            return spectrum
        return normalize_spectrum(spectrum)

    def interpolate_spectrum(self, spectrum: np.ndarray, orig_wavenumbers: np.ndarray):
        if np.all(spectrum == 0):
            return np.zeros(len(self.wavenumbers))
        f = interp1d(orig_wavenumbers, spectrum, kind="cubic", bounds_error=False, fill_value=0)
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

        # Create background mask
        background_mask = np.all(self.original_spectra == 0, axis=1).reshape(height, width)
        spectra = self.original_spectra.copy()
        predictions = []
        spectra_by_type = {name: [] for name in self.target}

        print("Processing spectra...")
        total_batches = (height * width + batch_size - 1) // batch_size
        for i in tqdm(range(0, height * width, batch_size), total=total_batches, desc="Batch Progress"):
            end_idx = min(i + batch_size, height * width)
            batch = np.array([self.preprocess_spectrum(s) for s in spectra[i:end_idx]])
            processed_batch = np.array([self.interpolate_spectrum(s, self.img_wavenumbers) for s in batch])
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(processed_batch).to(self.device).unsqueeze(1)
                output = self.model(batch_tensor)  # (batch_size, 5)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
        self.predictions = np.concatenate(predictions, axis=0)

        merged_map = np.zeros((height, width, 3))
        for name, model_idx in self.target_indices.items():
            prob_map = self.predictions[:, model_idx].reshape(height, width)
            prob_map = np.where(prob_map > self.prob_thresh, prob_map, 0)
            prob_map[background_mask] = 0

            # Save colored map
            color = np.array([int(self.colors[name].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
            colored_map = np.zeros((height, width, 3))
            for i in range(3):
                colored_map[:, :, i] = prob_map * color[i]
            plt.figure(figsize=(10, 10))
            plt.imshow(colored_map)
            plt.title(f"{name} Mapping")
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"{name}_colored.png"), dpi=300, bbox_inches="tight")
            plt.close()

            merged_map += colored_map
            tifffile.imwrite(os.path.join(output_dir, f"{name}_pred.tif"), (prob_map * 255).astype(np.uint8))

            prob_values = self.predictions[:, model_idx].copy()
            prob_values[background_mask.flatten()] = 0
            if np.any(prob_values > self.prob_thresh):
                spectra_by_type[name] = np.where(prob_values > self.prob_thresh)[0].tolist()
            else:
                spectra_by_type[name] = []

        merged_map = np.clip(merged_map, 0, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(merged_map)
        plt.title("Merged Mapping")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, "merged.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print("Saving high probability spectra...")
        spectra_data = {"wavenumber": self.wavenumbers}
        for name, model_idx in self.target_indices.items():
            prob_values = self.predictions[:, model_idx].copy()
            prob_values[background_mask.flatten()] = 0
            mask = prob_values > self.prob_thresh
            print(f"\n{name}:")
            print(f"Pixels above {self.prob_thresh} probability: {np.sum(mask)}")
            if np.any(mask):
                selected_spec = self.original_spectra[mask]
                interpolated_spec = [
                    self.interpolate_spectrum(s, self.img_wavenumbers) for s in selected_spec
                ]
                avg_spec = np.mean(interpolated_spec, axis=0)
                avg_spec = normalize_spectrum(avg_spec)
                spectra_data[f"{name}_avg"] = avg_spec
                spectra_data[f"{name}_count"] = len(selected_spec)
            else:
                spectra_data[f"{name}_avg"] = np.zeros(len(self.wavenumbers))
                spectra_data[f"{name}_count"] = 0
        df = pd.DataFrame(spectra_data)
        csv_path = os.path.join(output_dir, "avg_spectra.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved spectra to {csv_path}")

        self._plot_spectral_comparisons(spectra_by_type, output_dir)
        print("All visualizations saved!")

    def _plot_spectral_comparisons(self, spectra_by_type: Dict[str, List[int]], output_dir: str):
        n_types = len(self.target)
        fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
        plt.style.use("seaborn-v0_8-pastel")
        plt.rcParams.update({"font.size": 14, "font.family": "Arial", "font.weight": "bold"})
        for idx, subtype in enumerate(self.target):
            ax = axes[idx] if n_types > 1 else axes
            color = self.colors[subtype]
            # Plot reference spectrum (assumes first matching row contains reference)
            ref_spectrum = self.ref[self.ref["name"] == subtype].iloc[0, 1:].values
            ax.plot(
                self.wavenumbers,
                ref_spectrum,
                color="#404040",
                label=f"{subtype} ref",
                linewidth=2,
                alpha=0.8,
            )
            pred_spec = None
            if spectra_by_type[subtype]:
                selected_spec = [
                    self.interpolate_spectrum(self.original_spectra[i], self.img_wavenumbers)
                    for i in spectra_by_type[subtype]
                ]
                pred_spec = np.mean(selected_spec, axis=0)
                pred_spec = normalize_spectrum(pred_spec)
            if pred_spec is not None:
                corrected, baseline = modpoly_baseline(pred_spec)
                smoothed = smooth_spectrum(normalize_spectrum(corrected))
                ax.plot(self.wavenumbers, pred_spec, color=color, linestyle="--", label="Raw")
                ax.plot(self.wavenumbers, baseline, color="#FF6B6B", linestyle="-.", label="Baseline")
                ax.plot(self.wavenumbers, smoothed, color=color, label="Corrected", linewidth=2)
            ax.set_title(subtype, fontweight="bold")
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Normalized Intensity")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spectral_comparisons.png"), dpi=300, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="HSI Prediction for 5 Subtypes")
    parser.add_argument(
        "--model_path", type=str, default="checkpoints_5sub/best_model.pth", help="Path to pretrained model"
    )
    parser.add_argument(
        "--library_path", type=str, default="Raman_dataset/library.csv", help="Path to library CSV"
    )
    parser.add_argument(
        "--image_path", type=str, default="HSI_data/1-Wt_FB.tif", help="Path to HSI image stack"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predicted_results/hsi_5sub_prediction/wt_fb",
        help="Directory to save results",
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["d2-fructose", "d7-glucose", "d-tyrosine", "d-methionine", "d-leucine"],
        help="List of 5 target subtypes",
    )
    args = parser.parse_args()

    predictor = HSIPredictor5Sub(
        model_path=args.model_path, library_path=args.library_path, target=args.target
    )
    predictor.predict_hsi(args.image_path, args.output_dir)


if __name__ == "__main__":
    main()
