import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    """Basic Conv1d block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock1D(nn.Module):
    """Residual block for 1D signals"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv1d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Attention(nn.Module):
    """Attention module"""

    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LipidNet(nn.Module):
    """
    LipidNet model for lipid classification.
    """

    def __init__(self, input_channels=1, base_channels=8, num_classes=5):
        super().__init__()
        self.in_channels = base_channels

        self.conv1 = Conv1d(input_channels, base_channels, kernel_size=7, stride=1, padding=3)

        self.layer1 = self._make_layer(ResBlock1D, out_channels=base_channels, blocks=2)
        self.layer2 = self._make_layer(ResBlock1D, out_channels=base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(ResBlock1D, out_channels=base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(ResBlock1D, out_channels=base_channels * 8, blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, num_classes),
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # input shape: (B, 1, features)
        x = self.conv1(x)  # (B, 8, features)

        # ResNet blocks
        x = self.layer1(x)  # (B, 8, features)
        x = self.layer2(x)  # (B, 16, features/2)
        x = self.layer3(x)  # (B, 32, features/4)
        x = self.layer4(x)  # (B, 64, features/8)

        # # Attention
        # x = x.permute(0, 2, 1)  # (B, features, 32)
        # x, _ = self.attention(x)  # (B, features, 32)
        # x = x.permute(0, 2, 1)  # (B, 32, features)
        x = self.avg_pool(x)  # (B, 64, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.dropout(x)
        x = self.classifier(x)  # (B, num_classes)
        return x

    def get_features(self, x):
        """Extract features before classification layer"""
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        return x.view(x.size(0), -1)

    # def get_attention_weights(self, x):
    #     """Get attention weights for visualization"""
    #     x = self.conv1(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)

    #     x = x.permute(0, 2, 1)
    #     _, attn = self.attention(x)  # attn shape: (B, num_heads, features, features)
    #     return attn
