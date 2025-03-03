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
from scipy.ndimage import minimum_filter
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from model import LipidNet
from utils import *


class HSIPredictor:
    def __init__(self, model_path: str, library_path: str, target: List[str]):
        """
        Initialize HSI predictor

        Args:
            model_path: Path to pretrained model weights
            library_path: Path to library CSV with reference spectra
            target_subtypes: List of subtypes to predict
        """
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target = target

        # Set thresholds for each target subtype
        self.prob_thresholds = {
            "d7-glucose": 0.95,
            "d2-fructose": 0.6,
            "d-tyrosine": 0.9,
            "d-methionine": 0.6,
            "d-leucine": 0.6,
        }

        # Set default thresholds
        self.default_prob_thresh = 0.7

        self.bg_thresh = 0.9
        self.colors = {
            "d7-glucose": "#FF00FF",  # Magenta
            "d2-fructose": "#FF3300",  # Bright red-orange
            "d-tyrosine": "#DAA520",  # Goldenrod
            "d-methionine": "#0066FF",  # Bright royal blue
            "d-leucine": "#33CC00",  # Bright green
            "background": "#808080",  # Gray
        }

        self.color_alpha = {
            "d7-glucose": 0.5,
            "d2-fructose": 1.0,
            "d-tyrosine": 0.8,
            "d-methionine": 1.0,
            "d-leucine": 1.0,
            "background": 1.0,
        }
        # Load pretrained model
        self.model, self.class_names = self._load_model(model_path)

        if isinstance(self.class_names, np.ndarray):
            self.class_names = self.class_names.tolist()
        # Get Background class index
        self.bg_idx = self.class_names.index("background") if "background" in self.class_names else None
        if self.bg_idx is None:
            print("Warning: Background class not found in model classes. Background filtering disabled")
        else:
            print(f"Background class index: {self.bg_idx}")

        # Load reference data
        self.ref = self._load_references(library_path)
        self.wavenumbers = self.ref.columns[1:].astype(float).values

        self.target_indices = (
            self._get_target_indices()
        )  # {'d2-fructose': 0, 'd-tyrosine': 1, 'd-methionine': 2, 'd-leucine': 3}

        self.valid_class_indices = list(self.target_indices.values())
        if self.bg_idx is not None:
            self.valid_class_indices.append(self.bg_idx)

        self.original_spectra = None
        self.img_wavenumbers = None

    def _load_model(self, model_path: str):
        """Load pretrained model"""
        ckpt = torch.load(model_path, map_location=self.device)

        class_names = ckpt.get("class_names", [])
        if len(class_names) == 0:
            raise ValueError("Class names not found in checkpoint")

        model = LipidNet(num_classes=len(class_names))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model.to(self.device), class_names

    def _load_references(self, library_path: str):
        """Load reference spectra"""
        return pd.read_csv(library_path)

    def _get_target_indices(self):
        """Get indices of target subtypes in original model output"""
        target_indices = {}
        for name in self.target:
            if name in self.class_names:
                target_indices[name] = self.class_names.index(name)
            else:
                print(f"Warning: Target {name} not fount in model classes")

        return target_indices

    def preprocess_spectrum(self, spectrum: np.ndarray):
        """Preprocess a single spectrum"""
        if np.all(spectrum == 0):
            return spectrum

        # spectrum = self.baseline_als(spectrum)
        # spectrum = np.maximum(spectrum, 0)
        spectrum = normalize_spectrum(spectrum)
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
        """
        Predict spatial molecular distribution in HSI with background filtering

        Args:
            image_path: Path to HSI image stack
            output_dir: Directory to save results
        """

        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess image
        image = tifffile.imread(image_path)
        image = np.flip(image, axis=0)  # Flip image to match reference

        assert len(image.shape) == 3, "Input image must be an image stack"
        N, height, width = image.shape

        # Set batch size to speed up the prediction
        batch_size = 4096

        # Create wavenumber array for the image
        self.img_wavenumbers = np.linspace(self.wavenumbers[0], self.wavenumbers[-1], N)
        self.original_spectra = image.reshape(N, -1).T

        # Create mask for background pixels (all zeros)
        zero_mask = np.all(self.original_spectra == 0, axis=1)
        zero_mask = zero_mask.reshape(height, width)

        spectra = self.original_spectra.copy()

        # Initialize arrays for results
        predictions = []
        spectra_by_type = {name: [] for name in self.target}
        bg_probs = np.zeros(height * width)

        # Process images in batches
        print("Processing spectra...")
        total_batches = (height * width + batch_size - 1) // batch_size

        for i in tqdm(range(0, height * width, batch_size), total=total_batches, desc="Batch Progress"):
            end_idx = min(i + batch_size, height * width)  # Ensure we don't go out of bounds
            batch = spectra[i:end_idx]

            # Preprocess batch
            batch = np.array([self.preprocess_spectrum(s) for s in batch])
            processed_batch = np.array([self.interpolate_spectrum(s, self.img_wavenumbers) for s in batch])

            # Run inference
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(processed_batch).to(self.device)
                batch_tensor = batch_tensor.unsqueeze(1)
                output = self.model(batch_tensor)  # (batch_size, num_classes)

                # Only consider the target and background
                restricted_output = output[:, self.valid_class_indices]
                restricted_probs = torch.softmax(restricted_output, dim=1)

                probs = torch.zeros_like(output)
                for idx, valid_idx in enumerate(self.valid_class_indices):
                    probs[:, valid_idx] = restricted_probs[:, idx]

                # Save background probabilities if available
                if self.bg_idx is not None:
                    bg_probs[i:end_idx] = probs[:, self.bg_idx].cpu().numpy()
                predictions.append(probs.cpu().numpy())

        self.predictions = np.concatenate(predictions, axis=0)

        # Create background spectra mask
        if self.bg_idx is not None:
            bg_mask = bg_probs > self.bg_thresh
            bg_mask = bg_mask.reshape(height, width)

            # Save background probability map
            bg_prob_map = bg_probs.reshape(height, width)
            plt.figure(figsize=(10, 10))
            plt.imshow(bg_prob_map, cmap="gray")
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, "background_prob.png"), dpi=300, bbox_inches="tight")
            plt.close()

            # Save background mask
            bg_mask_img = bg_mask.astype(np.uint8) * 255
            tifffile.imwrite(os.path.join(output_dir, "backgroun_mask.tif"), bg_mask_img)

            print(f"Detected {np.sum(bg_mask)} background pixels")

        else:
            bg_mask = zero_mask
            print("No background class detected, using zero pixels as background")

        # Combine the spectra mask and all zero mask
        combined_mask = np.logical_or(zero_mask, bg_mask)
        combined_mask_reshaped = combined_mask.flatten()

        # Initialize prediction maps for each target
        prediction_maps = {}
        merged_map = np.zeros((height, width, 3))

        # Track the pixel counts for each molecule
        pixel_counts = {}

        for name, model_idx in self.target_indices.items():
            # Get the molecule specific thereshold
            mol_threshold = self.prob_thresholds.get(name, self.default_prob_thresh)
            print(f"Using threshold {mol_threshold} for {name}")

            prob_map = self.predictions[:, model_idx].reshape(height, width)
            prob_map = np.where(prob_map > mol_threshold, prob_map, 0)
            prob_map[combined_mask] = 0
            prediction_maps[name] = prob_map

            # Count pixels above threshold
            pixel_counts[name] = np.sum(prob_map > 0)

            # Save colored map
            color = (
                np.array(tuple(int(self.colors[name].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))) / 255.0
            )

            # Apply the molecule-specific alpha value
            alpha = self.color_alpha[name]

            colored_map = np.zeros((height, width, 3))
            for i in range(3):
                if name != "d7-glucose" and name != "d-tyrosine":
                    colored_map[:, :, i] = prob_map * color[i] * 1.5
                else:
                    colored_map[:, :, i] = prob_map * color[i] * alpha

            plt.figure(figsize=(10, 10))
            im = plt.imshow(colored_map)
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"{name}_colored.png"), dpi=300, bbox_inches="tight")
            plt.close()

            merged_map += colored_map

            # Save grayscale TIFF
            tifffile.imwrite(os.path.join(output_dir, f"{name}_pred.tif"), (prob_map * 255).astype(np.uint8))

            prob_values = self.predictions[:, model_idx].copy()
            prob_values[combined_mask_reshaped] = 0

        # Save merged image
        merged_map = np.clip(merged_map, 0, 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(merged_map)
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, "merged.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print("Saving high probability spectra...")
        spectra_data = {"wavenumber": self.wavenumbers}
        valid_predictions = {}
        for name, model_idx in self.target_indices.items():
            prob_values = self.predictions[:, model_idx].copy()
            prob_values[combined_mask_reshaped] = 0
            mol_threshold = self.prob_thresholds.get(name, self.default_prob_thresh)
            mask = prob_values > mol_threshold
            print(f"\n{name}:")
            print(f"Pixels above {mol_threshold} probability: {np.sum(mask)}")

            if np.any(mask):
                selected_spec = self.original_spectra[mask]
                interpolated_spec = [
                    self.interpolate_spectrum(spec, self.img_wavenumbers) for spec in selected_spec
                ]
                avg_spec = np.mean(interpolated_spec, axis=0)
                avg_spec = normalize_spectrum(avg_spec)
                spectra_data[f"{name}_avg"] = avg_spec
                spectra_data[f"{name}_count"] = len(selected_spec)
                valid_predictions[name] = True
                print(f"✓ Saved average spectrum ({len(selected_spec)} pixels)")
            else:
                valid_predictions[name] = False
                print(f"✗ No valid prediction above 0.9")

        # Save to CSV
        df = pd.DataFrame(spectra_data)
        csv_path = os.path.join(output_dir, "avg_spectra.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved spectra to {csv_path}")

        # Create color legend image
        self._create_color_legend(output_dir, pixel_counts)
        print("All visualizations saved!")

        # # Plot spectral comparisons with matching colors
        # self._plot_spectral_comparisons(spectra_by_type, output_dir, use_csv=False)
        # print("All visualizations saved!")

    def _create_color_legend(self, output_dir: str, pixel_counts: Dict[str, int]):
        """Create a color legend image showing molecule colors and pixel counts"""
        plt.figure(figsize=(10, 6))
        plt.style.use("seaborn-v0_8-deep")

        y_positions = np.arange(len(self.target))
        colors = [self.colors[name] for name in self.target]

        plt.barh(y_positions, [1] * len(self.target), color=colors, height=0.5)

        # Add labels with pixel counts
        labels = [f"{name} ({pixel_counts.get(name, 0)} pixels)" for name in self.target]
        plt.yticks(y_positions, labels)

        plt.title("Molecule Color Legend")
        plt.xlim(0, 1.5)
        plt.axis("off")
        for i, name in enumerate(self.target):
            plt.text(1.1, i, self.colors[name], va="center")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "color_legend.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_spectral_comparisons(
        self, spectra_by_type: Dict[str, List[np.ndarray]], output_dir: str, use_csv=False
    ):
        """Plot predicted spectra vs references"""
        n_types = len(self.target)
        fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
        plt.style.use("seaborn-v0_8-pastel")
        plt.rcParams.update({"font.size": 14, "font.family": "Arial", "font.weight": "bold"})
        spectra_df = None
        if use_csv:
            csv_path = os.path.join(output_dir, "avg_spectra.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"No saved spectra found at {csv_path}")
            spectra_df = pd.read_csv(csv_path)

        for idx, subtype in enumerate(self.target):
            ax = axes[idx] if n_types > 1 else axes
            color = self.colors[subtype]

            # Plot reference spectrum
            ref_spectrum = self.ref[self.ref["name"] == subtype].iloc[0, 1:].values
            ax.plot(
                self.wavenumbers, ref_spectrum, color="#404040", label=f"{subtype}", linewidth=2, alpha=0.8
            )

            pred_spec = None
            if use_csv and spectra_df is not None:
                if f"{subtype}_avg" in spectra_df.columns:
                    pred_spec = spectra_df[f"{subtype}_avg"].values
            else:
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
    parser = argparse.ArgumentParser(description="HSI Prediction")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints_with_bg/best_model.pth",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--library_path", type=str, default="Raman_dataset/library.csv", help="Path to library CSV"
    )
    parser.add_argument(
        "--image_path", type=str, default="HSI_data/1-Wt_FB.tif", help="Path to HSI image stack"
    )
    parser.add_argument(
        "--output_dir", type=str, default="predicted_results/hsi_wt_with_bg", help="Directory to save results"
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["d2-fructose", "d7-glucose", "d-tyrosine", "d-methionine", "d-leucine"],
        help="List of target subtypes to predict",
    )

    args = parser.parse_args()

    predictor = HSIPredictor(model_path=args.model_path, library_path=args.library_path, target=args.target)

    predictor.predict_hsi(args.image_path, args.output_dir)
    # predictor._plot_spectral_comparisons(None, args.output_dir, use_csv=True)


if __name__ == "__main__":
    main()
