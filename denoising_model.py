import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# class Conv1d(nn.Module):
#     """Basic Conv1d block"""
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self,x):
#         return self.relu(self.conv(x))
#
# class ResBlock1D(nn.Module):
#     """Residual block for 1D signals"""
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super().__init__()
#         self.conv1 = Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.downsample = downsample
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         return self.relu(out)
    
class RamanDenoise(nn.Module):
    def __init__(self, input_channels=1, base_channels=8):
        super(RamanDenoise, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(base_channels, input_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    # def _make_layer(self, block, out_channels, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.in_channels != out_channels:
    #         downsample = nn.Sequential(
    #             nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False)
    #         )
    #
    #     layers = []
    #     layers.append(block(self.in_channels, out_channels, stride, downsample))
    #     self.in_channels = out_channels
    #     for _ in range(1, blocks):
    #         layers.append(block(out_channels, out_channels))
    #
    #     return nn.Sequential(*layers)

    def forward(self, x,):
        latent = self.encoder(x)  # Extract features
        # pooled_latent = self.global_pool(latent)  # Shape: (N, latent_dim, 1)
        # classification = self.classifier(pooled_latent).detach()  # Shape: (N, num_classes)
        reconstructed = self.decoder(latent) # Shape: (N, input_channels, L)
        return reconstructed

class RamanClassifier(nn.Module):
    def __init__(self, denoiser, base_channels=8, num_classes:int=25):
        super(RamanClassifier, self).__init__()

        self.encoder = denoiser.encoder
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


