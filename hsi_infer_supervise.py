import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm

from model import *
from utils import normalize_spectrum, set_seed

class HSIPredictor:
    def __init__(self, model_path: str, library_path: str, target: List[str] = None, exclude: List[str] = None):
        """
        Args:
            model_path: Path to pretrained model weights
            library_path: Path to library CSV with reference spectra
            target: List of subtypes to predict (if None, predict all non-background classes)
            exclude: List of subtypes to exclude from prediction
        """
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set default probability threshold for classification
        self.default_prob_thresh = 0.6
        
        # Set class-specific thresholds if needed
        self.prob_thresholds = {}
        
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
        
        self.non_bg_class_names = [name for i, name in enumerate(self.class_names) if i != self.bg_idx]
        print(f"Found {len(self.non_bg_class_names)} non-background classes: {self.non_bg_class_names}")
        
        # Processs if we need to exlude some molecules
        self.exclude = exclude or []
        for name in self.exclude:
            if name not in self.class_names:
                print(f"Warning: Exclude target '{name}' not found in model classes")
            else:
                print(f"Excluding target '{name}' from prediction")
                
        # Set target classes for prediction (if None, use all non-background classes except excluded ones)
        if target is None:
            self.target = [name for name in self.non_bg_class_names if name not in self.exclude]
            print(f"Using all non-bg classes except excluded ones")
        else:
            self.target = [name for name in target if name in self.class_names and 
                          name not in self.exclude and 
                          self.class_names.index(name) != self.bg_idx]
            print(f"Using specified target classes except excluded ones")
        
        print(f"Final target classes for prediction: {self.target}")
        
        # Load reference data
        self.ref = self._load_references(library_path)
        self.wavenumbers = self.ref.columns[1:].astype(float).values
        
        # Get indices of target classes in model output
        self.target_indices = self._get_target_indices()
        print(f"Target indices: {self.target_indices}")
        
        self.original_spectra = None
        self.img_wavenumbers = None

    def _load_model(self, model_path: str):
        """Load pretrained model"""
        print(f"Loading model from {model_path}...")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        # Get class names from checkpoint
        class_names = ckpt.get("class_names", [])
        if len(class_names) == 0:
            raise ValueError("Class names not found in checkpoint")
        
        # Create model with correct number of classes
        model = COMPASS(num_classes=len(class_names))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Loaded model with {len(class_names)} classes")
        
        return model.to(self.device), class_names

    def _load_references(self, library_path: str):
        """Load reference spectra"""
        print(f"Loading reference spectra from {library_path}...")
        return pd.read_csv(library_path)

    def _get_target_indices(self):
        """Get indices of target subtypes in original model output (excluding background)"""
        target_indices = {}
        for name in self.target:
            if name in self.class_names:
                class_idx = self.class_names.index(name)
                # Skip background class
                if class_idx != self.bg_idx:
                    target_indices[name] = class_idx
        
        return target_indices

    def preprocess_spectrum(self, spectrum: np.ndarray):
        """Preprocess a single spectrum"""
        if np.all(spectrum == 0):
            return spectrum
        spectrum = normalize_spectrum(spectrum)
        
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
        Predict spatial molecular distribution in HSI and save as TIF files

        Args:
            image_path: Path to HSI image stack
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess image
        print(f"Loading image from {image_path}...")
        image = tifffile.imread(image_path)
        image = np.flip(image, axis=0)  # Flip image to match reference

        assert len(image.shape) == 3, "Input image must be an image stack"
        N, height, width = image.shape
        print(f"Image shape: {image.shape}")

        batch_size = 4096
        
        # Create wavenumber array for the image
        self.img_wavenumbers = np.linspace(self.wavenumbers[0], self.wavenumbers[-1], N)
        self.original_spectra = image.reshape(N, -1).T

        # Create mask for background pixels (all zeros)
        zero_mask = np.all(self.original_spectra == 0, axis=1)
        zero_mask = zero_mask.reshape(height, width)
        print(f"Found {np.sum(zero_mask)} zero pixels")

        spectra = self.original_spectra.copy()
        
        # Initialize array for predictions
        all_probs = np.zeros((height * width, len(self.class_names)))

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
                batch_tensor = batch_tensor.unsqueeze(1)  # Add channel dimension
                output = self.model(batch_tensor)  # (batch_size, num_classes)
                
                probs = torch.softmax(output, dim=1).cpu().numpy()
                
                if self.exclude is not None:
                    exclude_indices = [self.class_names.index(name) for name in self.exclude if name in self.class_names]
                    for idx in exclude_indices:
                        probs[:, idx] = 0
                    
                    row_sums = probs.sum(axis=1, keepdims=True)
                    non_zero_rows = row_sums.flatten() > 0
                    if np.any(non_zero_rows):
                        probs[non_zero_rows, :] = probs[non_zero_rows, :] / row_sums[non_zero_rows]
                   
                all_probs[i:end_idx] = probs

        # Create background mask if background class exists
        if self.bg_idx is not None:
            # Apply background threshold
            bg_prob_threshold = self.prob_thresholds.get("background", 0.8)
            print(f"Using threshold {bg_prob_threshold} for background")
            
            bg_probs = all_probs[:, self.bg_idx]
            bg_mask = bg_probs > bg_prob_threshold
            bg_mask = bg_mask.reshape(height, width)
            
            # Combine zero mask and background mask
            combined_mask = np.logical_or(zero_mask, bg_mask)
            print(f"Detected {np.sum(bg_mask)} background pixels")
            print(f"Total masked pixels: {np.sum(combined_mask)}")
        else:
            combined_mask = zero_mask
            print("No background class detected, using zero pixels as background")
            
        combined_mask_flat = combined_mask.flatten()

        print(f"Creating prediction maps for {len(self.target_indices)} target classes...")
        
        pixel_counts = {}
        
        # Process each target class
        for name, model_idx in self.target_indices.items():
            # Get the class-specific threshold or use default
            threshold = self.prob_thresholds.get(name, self.default_prob_thresh)
            if name == "D7-glucose_lipid":
                threshold = 0.9
            print(f"Using threshold {threshold} for {name}")
            
            # Get probabilities for this class
            prob_values = all_probs[:, model_idx].copy()
            
            # Apply mask and threshold
            prob_values[combined_mask_flat] = 0
            prob_map = prob_values.reshape(height, width)
            thresholded_map = np.where(prob_map > threshold, prob_map, 0)
            
            # Count pixels above threshold
            pixel_count = np.sum(thresholded_map > 0)
            pixel_counts[name] = pixel_count
            print(f"{name}: {pixel_count} pixels above threshold {threshold}")
            
            # Save as TIF file (scaled to 0-255 for visualization)
            tif_path = os.path.join(output_dir, f"{name}_pred.tif")
            tifffile.imwrite(tif_path, (thresholded_map * 255).astype(np.uint8))
            print(f"Saved {name} prediction map to {tif_path}")
        
        # Print summary
        print("\nPrediction summary:")
        for name, count in pixel_counts.items():
            print(f"{name}: {count} pixels")
        
        print(f"\nAll TIF files saved to {output_dir}")


def main():
    """Main function to run HSI prediction"""
    parser = argparse.ArgumentParser(description="HSI Prediction")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints_lipids_and_protein/best_model.pth",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--library_path", 
        type=str, 
        default="Raman_dataset/molecules_9/train_reference.csv", 
        help="Path to library CSV"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        default="HSI_data/1-Wt_FB.tif", 
        help="Path to HSI image stack"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="prediction_results/1-Wt", 
        help="Directory to save results"
    )
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=None,
        help="List of target classes to predict (if none, all non-background classes will be predicted)",
    )
    
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="List of classes to exclude from prediction",
    )

    args = parser.parse_args()

    predictor = HSIPredictor(
        model_path=args.model_path, 
        library_path=args.library_path, 
        target=args.target,
        exclude=args.exclude
    )

    predictor.predict_hsi(args.image_path, args.output_dir)


if __name__ == "__main__":
    main()