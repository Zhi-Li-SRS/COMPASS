import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RamanClassifier(nn.Module):
    def __init__(self, model, base_channels=8, num_classes:int=25):
        super(RamanClassifier, self).__init__()

        self.encoder = model.encoder
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Latent space Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(True),
            nn.Linear(base_channels * 4, num_classes)
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        latent = self.encoder(x)
        pooled_latent = self.global_pool(latent)  # Shape: (N, latent_dim, 1)
        classification = self.classifier(pooled_latent).detach()  # Shape: (N, num_classes)

        return classification
