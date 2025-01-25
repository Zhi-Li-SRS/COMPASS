import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def normalize(array, max=1, min=0):
    """
    Normalize array by minimum and maximum values
    """
    min_val = np.min(array)
    max_val = np.max(array)

    if np.all(array == 0):
        norm = array
    else:
        norm = ((array - min_val) / (max_val - min_val)) * (max - min) + min
    return norm


def preprocess_data(input_file, output_file):
    """
    Preprocess the spectral data file by transposing and reformatting.
    """
    df = pd.read_csv(input_file, index_col=0)

    df_transposed = df.transpose()

    df_transposed.to_csv(output_file)
    return df_transposed


def augment_data(
    spectrum, wavenumbers, background, n_augment=400, noise_level=0.3, bg_level=1.1, bg_scale=1, max_shift=15
):
    """
    Augment a single spectrum by adding noise and random shifts.
    Ensures all values are positive and within reasonable range for Raman spectra.
    """
    augmented_spectra = []
    original_max = np.max(spectrum)  # Get original spectrum's max valu

    for _ in range(n_augment):
        bg_mult = np.random.uniform(bg_scale * noise_level * original_max, bg_level * original_max)
        curr_background = bg_mult * background

        noise = np.random.normal(0, noise_level * original_max, len(spectrum))
        noisy_spectrum = spectrum + noise
        noisy_spectrum = np.maximum(noisy_spectrum, 0)

        shift = np.random.uniform(-max_shift / 2, max_shift / 2)  # Reduced shift range
        shifted_wavenumbers = wavenumbers + shift

        interp_func = interp1d(
            shifted_wavenumbers,
            noisy_spectrum,
            kind="cubic",
            bounds_error=False,
            fill_value=(spectrum[0], spectrum[-1]),
        )
        aug_spectrum = interp_func(wavenumbers)
        aug_spectrum = aug_spectrum + curr_background
        aug_spectrum = aug_spectrum / np.max(aug_spectrum)
        aug_spectrum = np.maximum(aug_spectrum, 0)
        augmented_spectra.append(aug_spectrum)

    return np.array(augmented_spectra)


def create_augmented_dataset(input_file, output_file, background_file, n_augment=400):
    """
    Create augmented dataset from original spectra.
    """
    df = pd.read_csv(input_file)
    names = df.iloc[:, 0]  # Get the names of the lipids
    wavenumbers = df.columns[1:].astype(float).values
    spectra = df.iloc[:, 1:].values

    background_df = pd.read_csv(background_file)
    background = background_df.to_numpy()[:, 0]
    interp = interp1d(np.linspace(0, 1, len(background)), background)
    background = interp(np.linspace(0, 1, spectra.shape[1]))
    background = normalize(np.flip(background))  # Get normalized background array

    augmented_data = []

    for idx, name in enumerate(names):
        spectrum = spectra[idx]
        aug_spectra = augment_data(spectrum, wavenumbers, background, n_augment=n_augment)

        spectra_df = pd.DataFrame(aug_spectra, columns=wavenumbers)
        spectra_df.insert(0, "name", name)
        augmented_data.append(spectra_df)

    df_augmented = pd.concat(augmented_data, ignore_index=True)
    df_augmented.to_csv(output_file, index=False)


if __name__ == "__main__":

    df_augmented = create_augmented_dataset(
        "Raman_dataset/library.csv", "Raman_dataset/val_data.csv", "background/CD_HSI_76.csv"
    )

    print("Data preprocessing and augmentation completed!")
