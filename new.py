from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM implementation for 1D spectral data"""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        # Register hooks
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks
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
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)

        # Generate Grad-CAM
        gradients = self.gradients.detach().cpu().numpy()[0]  # (C, L)
        features = self.features.detach().cpu().numpy()[0]  # (C, L)

        # Calculate importance weights
        weights = np.mean(gradients, axis=1)  # (C,)

        # Compute weighted combination of forward activation maps
        cam = np.sum(weights[:, np.newaxis] * features, axis=0)  # (L,)
        cam = np.maximum(cam, 0)  # ReLU

        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)

        return cam

    def __del__(self):
        self.remove_hooks()


class Infer:
    # ... (existing code) ...

    def visualize_gradcam(self, sample_idx: int = 0):
        """Visualize Grad-CAM for a single spectrum with detailed analysis"""
        self.logger.info("Generating detailed Grad-CAM visualization...")

        # Initialize Grad-CAM with layer3 as target
        grad_cam = GradCAM(self.model, self.model.layer3)

        # Get a sample spectrum
        spectrum = torch.FloatTensor(self.features[sample_idx]).unsqueeze(0).unsqueeze(0)
        label = self.labels[sample_idx]

        # Generate Grad-CAM
        cam = grad_cam.generate_cam(spectrum, target_class=label)

        # Interpolate CAM to match spectrum length if needed
        if len(cam) != len(self.features[sample_idx]):
            x_old = np.linspace(0, 1, len(cam))
            x_new = np.linspace(0, 1, len(self.features[sample_idx]))
            cam = np.interp(x_new, x_old, cam)

        # Apply thresholding to highlight important regions
        threshold = 0.5  # Adjust this threshold as needed
        cam_threshold = np.where(cam > threshold, cam, 0)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 0.5])
        plt.style.use("default")

        # Plot original spectrum with highlighted regions
        spectrum_data = self.features[sample_idx]
        ax1.plot(self.wavenumbers, spectrum_data, "k-", linewidth=1.5, label="Spectrum")

        # Find peaks and highlight important regions
        important_regions = np.where(cam_threshold > 0)[0]
        if len(important_regions) > 0:
            for start, end in self._get_continuous_regions(important_regions):
                ax1.axvspan(
                    self.wavenumbers[start],
                    self.wavenumbers[end],
                    alpha=0.3,
                    color="red",
                    label="Important Region" if start == important_regions[0] else None,
                )

        # Plot Grad-CAM values separately
        ax2.plot(self.wavenumbers, cam, "r-", linewidth=1.5, label="Grad-CAM")
        ax2.fill_between(self.wavenumbers, cam, alpha=0.3, color="red")
        ax2.axhline(y=threshold, color="gray", linestyle="--", label="Threshold")

        # Customize plots
        ax1.set_title(f"Class: {self.class_names[label]}", fontsize=12, pad=10)
        ax1.set_ylabel("Intensity (a.u.)", fontsize=10)
        ax1.legend(fontsize=10)

        ax2.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=10)
        ax2.set_ylabel("Grad-CAM Score", fontsize=10)
        ax2.legend(fontsize=10)

        # Add grid
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.args.output_dir, "gradcam_detailed.png")
        plt.savefig(save_path, dpi=self.args.fig_dpi, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Detailed Grad-CAM visualization saved to {save_path}")

    def _get_continuous_regions(self, indices):
        """Helper function to find continuous regions in array indices"""
        regions = []
        start = indices[0]
        prev = indices[0]

        for idx in indices[1:]:
            if idx - prev > 1:
                regions.append((start, prev))
                start = idx
            prev = idx

        regions.append((start, prev))
        return regions
