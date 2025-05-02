import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataload import load_data
from scipy.interpolate import interp1d
from skimage import io
from preprocess import normalize

class RandomForestClassifier:
    def __init__(self, data_path, background_path, train_path, target_path=None):
        # Load image and background
        self.data_path = data_path
        self.image = io.imread(self.data_path)
        background_df = pd.read_csv(background_path)
        background = background_df.to_numpy()[:, 0]
        interp = interp1d(np.linspace(0, 1, len(background)), background)
        background = interp(np.linspace(0, 1, self.image.shape[-1]))
        self.background = normalize(np.flip(background))  # Get normalized background array

        # Load training_data and targets
        self.train_data, self.target_data, self.names = load_data(train_path, target_path)

    def normalize