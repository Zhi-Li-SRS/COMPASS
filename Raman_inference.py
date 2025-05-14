import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from dataload import load_data
from model import *
from utils import *


class Infer:
    """Inference using pretrained model."""

    def __init__(self, args):
        self.args = args
        set_seed(42)
        self.device = torch.device("cpu")

        os.makedirs(self.args.output_dir, exist_ok=True)

        self.model = None
        self.features = None
        self.labels = None
        self.class_names = None
        self.wavenumbers = None
        self.load_model()
        
        # For brain_clusters.csv that doesn't have class labels
        if self.args.reference_data_path:
            self.load_reference_data()
        else:
            self.load_data()
            
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 14, "font.weight": "bold"})

    def load_model(self):
        """Load pretrained model."""
        print("Loading model...")
        self.model = COMPASS(num_classes=self.args.num_classes)
        checkpoint = torch.load(self.args.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
    def load_reference_data(self):
        """Load reference data with proper class names from training dataset."""
        print(f"Loading reference data from {self.args.reference_data_path}...")
        _, _, self.class_names = load_data(self.args.reference_data_path)
        
        # Load brain clusters data
        print(f"Loading clusters data from {self.args.data_path}...")
        df = pd.read_csv(self.args.data_path)
        
        # First column is cluster ID, rest are wavenumbers and intensity values
        self.wavenumbers = df.columns[1:].astype(float).values
        self.features = df.iloc[:, 1:].values
        
        # No real labels for clusters, assign placeholder zeros
        self.labels = np.zeros(len(df), dtype=int)
        
    def load_data(self):
        """Load spectral data."""
        print("Loading data...")
        self.features, self.labels, self.class_names = load_data(self.args.data_path)

        df = pd.read_csv(self.args.data_path)
        self.wavenumbers = df.columns[1:].astype(float).values

    def load_custom_data(self, data_path):
        """Load custom spectra data from a CSV file."""
        print(f"Loading custom data from {data_path}...")
        df = pd.read_csv(data_path)
        wavenumbers = df.columns[1:].astype(float).values
        spectra = df.iloc[:, 1:].values
        cluster_names = df.iloc[:, 0].values
    
        if len(wavenumbers) != len(self.wavenumbers):
            print(f"Warning: Wavenumber mismatch. Model: {len(self.wavenumbers)}, Input: {len(wavenumbers)}")
            interpolated_spectra = []
            for spectrum in spectra:
                interp_func = interp1d(wavenumbers, spectrum, bounds_error=False, fill_value="extrapolate")
                interpolated_spectra.append(interp_func(self.wavenumbers))
            spectra = np.array(interpolated_spectra)
        
        return spectra, cluster_names, wavenumbers

    def extract_features(self) -> Tuple[np.array, np.array]:
        """Extract the latent features before the classifier."""
        print("Extracting features...")
        features_list = []
        dataset = TensorDataset(torch.FloatTensor(self.features), torch.LongTensor(self.labels))
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        with torch.no_grad():
            for batch, target in tqdm(data_loader, desc="Extracting features"):
                if len(batch.shape) == 2:
                    batch = batch.unsqueeze(1)
                features = self.model.get_features(batch)
                features_list.append(features.cpu().numpy())

        return np.concatenate(features_list)

    def extract_features_from_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Extract features from provided spectra."""
        print("Extracting features from provided spectra...")
        features_list = []
        
        spectra_tensor = torch.FloatTensor(spectra)
        dataset = TensorDataset(spectra_tensor)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        with torch.no_grad():
            for (batch,) in tqdm(data_loader, desc="Extracting features"):
                if len(batch.shape) == 2:
                    batch = batch.unsqueeze(1)
                features = self.model.get_features(batch)
                features_list.append(features.cpu().numpy())

        return np.concatenate(features_list)

    def plot_tsne_comparison(self, features: np.ndarray):
        """Plot t-SNE visualization comparing original spectra and learned features"""
        tsne = TSNE(n_components=2, random_state=42)
        original_tsne = tsne.fit_transform(self.features)
        features_tsne = tsne.fit_transform(features)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        plt.style.use("seaborn-v0_8-pastel")

        unique_labels = np.unique(self.labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        scatter_handles = []
        legend_labels = []
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            class_name = self.class_names[label]
            scatter1 = ax1.scatter(
                original_tsne[mask, 0],
                original_tsne[mask, 1],
                c=[colors[i]],
                alpha=0.6,
                label=self.class_names[label],
            )
            # Plot for learned features
            scatter2 = ax2.scatter(
                features_tsne[mask, 0],
                features_tsne[mask, 1],
                c=[colors[i]],
                alpha=0.6,
                label=self.class_names[label],
            )
            scatter_handles.append(scatter1)
            legend_labels.append(class_name)

        ax1.set_title("Original Data t-SNE", fontsize=12, fontweight="bold")
        ax2.set_title("Learned Embedding t-SNE", fontsize=12, fontweight="bold")

        # Add single legend to figure
        fig.legend(
            handles=list(scatter_handles),
            labels=list(legend_labels),
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            fontsize=10,
            borderaxespad=0,
        )

        ax1.axis("off")
        ax2.axis("off")
        plt.tight_layout()

        save_path = os.path.join(self.args.output_dir, "tsne_comparison.png")
        plt.savefig(save_path, dpi=self.args.fig_dpi, bbox_inches="tight", pad_inches=0.5)
        plt.close()

        print(f"t-SNE visualization saved to {save_path}")

    def plot_gradcam(self):
        """Visualize Grad-CAM activation maps overlaid with original spectra"""
        # Initialize Grad-CAM with the final conv layer
        print(f"Features shape: {self.features.shape}")
        print(f"Wavenumbers shape: {self.wavenumbers.shape}")
        grad_cam = GradCAM(self.model, self.model.layer4)

        max_samples = min(30, len(self.features))
        samples_per_row = 6
        rows = (max_samples + samples_per_row - 1) // samples_per_row
        cols = samples_per_row

        fig_width = 16
        fig_height = 9 * (rows / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        plt.style.use("seaborn-v0_8-pastel")
        axs = axs.flatten() if rows > 1 else [axs]

        self.model.train()  # Temporarily set to train mode to compute gradients

        for i in tqdm(range(max_samples), desc="Processing Spectra"):
            spectrum = self.features[i]
            input_tensor = torch.FloatTensor(spectrum).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Generate Grad-CAM
            cam = grad_cam.generate_cam(input_tensor)
            assert len(cam) == len(spectrum), f"CAM length {len(cam)} != spectrum length {len(spectrum)}"
            # Plot original spectrum
            axs[i].plot(self.wavenumbers, spectrum, color="gray", linewidth=1, alpha=0.7)

            # Overlay Grad-CAM heatmap
            cmap = plt.cm.Reds
            for j in range(len(spectrum) - 1):
                axs[i].fill_between(
                    [self.wavenumbers[j], self.wavenumbers[j + 1]],
                    [spectrum[j], spectrum[j + 1]],
                    color=cmap(cam[j]),
                    alpha=cam[j],
                )

            axs[i].set_title(f"{self.class_names[self.labels[i]]}", fontsize=8)
            axs[i].set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=6)
            axs[i].set_ylabel("Normalized Intensity (a.u.)", fontsize=6)
            axs[i].tick_params(axis="both", which="major", labelsize=6)

        self.model.eval()  # Reset model to eval mode
        grad_cam.remove_hooks()

        # Hide unused subplots
        for i in range(max_samples, len(axs)):
            axs[i].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.args.output_dir, "gradcam.png")
        plt.savefig(save_path, dpi=self.args.fig_dpi, bbox_inches="tight")
        plt.close()

        print(f"Grad-CAM visualization saved to {save_path}")

    def calculate_similarity(self, spectra: np.ndarray, cluster_names: np.ndarray):
        """
        Calculate similarity between input spectra and library classes using model probabilities.
        
        Args:
            spectra (np.ndarray): Input spectra
            cluster_names (np.ndarray): Names of the input spectra clusters
        
        """
        print("Calculating similarity (probabilities) to library classes...")
        
        prediction_result = self.predict(spectra)
        probabilities = prediction_result["probabilities"]
        similarity_results = []
        
        for i, probs in enumerate(probabilities):
            similarities = {}
            similarities['cluster_id'] = cluster_names[i]
            
            min_val = np.min(probs)
            max_val = np.max(probs)
            if max_val > min_val: 
                normalized_probs = (probs - min_val) / (max_val - min_val)
            else:
                normalized_probs = probs
            
            for j, class_name in enumerate(self.class_names):
                similarities[class_name] = normalized_probs[j]
                
            similarity_results.append(similarities)
        
        result_df = pd.DataFrame(similarity_results)
        return result_df

    def predict(self, spectra: np.ndarray):
        """
        Predict classes for input spectra.

        Args:
            spectra (np.ndarray): Input spectra of shape (n_samples, n_features)
        """

        if len(spectra.shape) == 1:
            spectra = spectra[np.newaxis, :]

        spectra_tensor = torch.FloatTensor(spectra)
        if len(spectra_tensor.shape) == 2:
            spectra_tensor = spectra_tensor.unsqueeze(1)

        dataset = TensorDataset(spectra_tensor)
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        predictions = []
        probabilities = []

        with torch.no_grad():
            for (batch,) in tqdm(data_loader, desc="Predicting"):
                # Get model predictions
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)

                predictions.append(outputs.argmax(dim=1))
                probabilities.append(probs)

        # Concatenate batches
        predictions = torch.cat(predictions).cpu().numpy()
        probabilities = torch.cat(probabilities).cpu().numpy()

        result = {
            "predicted_classes": predictions,
            "class_names": [self.class_names[i] for i in predictions],
            "probabilities": probabilities,
        }
        return result


def main():
    """Main function to run inference"""
    parser = argparse.ArgumentParser(description="LipidNet Inference")
    parser.add_argument(
        "--model_path", type=str, default="checkpoints_with_bg/best_model.pth", help="Path to saved model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="brain_clusters.csv",
        help="Path to original spectra data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predicted_results",
        help="Directory to save results",
    )
    parser.add_argument("--num_classes", type=int, default=19, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--fig_dpi", type=int, default=300, help="DPI for saved figures")
    parser.add_argument(
        "--clusters_path",
        type=str,
        default="brain_clusters.csv",
        help="Path to brain clusters CSV file",
    )
    parser.add_argument(
        "--similarity_output",
        type=str,
        default="cluster_similarity.csv",
        help="Output file for similarity scores",
    )
    parser.add_argument(
        "--reference_data_path",
        type=str,
        default="Raman_dataset/raw/CD_lipid_library.csv",
        help="Path to reference data with class names",
    )

    args = parser.parse_args()

    inference = Infer(args)
    
    if args.clusters_path:
        spectra, cluster_names, _ = inference.load_custom_data(args.clusters_path)
        similarity_df = inference.calculate_similarity(spectra, cluster_names)
        
        output_path = os.path.join(args.output_dir, args.similarity_output)
        similarity_df.to_csv(output_path, index=False)
        print(f"Similarity scores saved to {output_path}")
            
    else:
        # Original functionality
        features = inference.extract_features()
        inference.plot_tsne_comparison(features)
        # inference.plot_gradcam()


if __name__ == "__main__":
    main()
