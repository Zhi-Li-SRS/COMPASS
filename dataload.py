import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RamanDataset(Dataset):
    """Custom Dataset for loading spectral data"""

    def __init__(self, data, labels=None, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[idx]
            return x, y

        return x


def load_data(csv_path):
    """Load and preprocess data from CSV file"""
    df = pd.read_csv(csv_path)

    names = df["name"].unique()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    labels = df["name"].map(name_to_idx).values
    features = df.drop("name", axis=1).values

    return features, labels, names


def create_dataloaders(train_path, val_path, batch_size=32, transform=None):
    """Create train and validation data loaders from separate files

    Args:
        train_path (str): Path to training data CSV
        val_path (str): Path to validation data CSV
        batch_size (int): Batch size for dataloaders
        transform (callable): Optional transform to be applied on the data
    """
    train_features, train_labels, train_names = load_data(train_path)
    train_dataset = RamanDataset(train_features, train_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_features, val_labels, val_names = load_data(val_path)

    if set(train_names) != set(val_names):
        raise ValueError("Training and validation datasets have different classes!")

    val_dataset = RamanDataset(val_features, val_labels, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_names
