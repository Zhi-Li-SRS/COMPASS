import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import skewnorm

def normalize(array, max=1, min=0, eps=1e-8):
    """
    Normalize array by minimum and maximum values
    """
    min_val = np.min(array)
    max_val = np.max(array)

    if np.all(array==0):
        raise Exception('All values in array are zero.')
    else:
        norm = ((array - min_val)/(max_val - min_val + eps))*(max - min) + min
    return norm



def preprocess_data(input_file, output_file):
    """
    Preprocess the spectral data file by transposing and reformatting.
    """
    df = pd.read_csv(input_file, index_col=0)

    df_transposed = df.transpose()

    df_transposed.to_csv(output_file)
    return df_transposed


def augment_data(spectrum, wavenumbers, background, n_augment=100, noise_level=0.1, bg_level=0.5, max_shift=15):
    """
    Augment a single spectrum by adding noise and random shifts.
    Ensures all values are positive and within reasonable range for Raman spectra.
    """
    augmented_spectra = []
    original_max = np.max(spectrum)  # Get original spectrum's max value

    bg_mult = np.sort(np.maximum(np.random.normal(bg_level, 2 * noise_level, n_augment), 2 * noise_level))
    # background = np.outer(bg_mult, background)
    noise_mult = np.sort(np.maximum(np.abs(skewnorm.rvs(-4, loc=noise_level, scale =0.01, size=n_augment)), 0.5 * noise_level))

    # noise = np.random.normal(0, noise_mult * original_max, len(spectrum))

    for i in range(n_augment):
        noise = np.random.normal(0, noise_mult[i] * original_max, len(spectrum))
        noisy_spectrum = spectrum + noise
        # noisy_spectrum = np.maximum(noisy_spectrum, 0)

        shift = np.random.uniform(-max_shift / 2, max_shift / 2)  # Reduced shift range
        shifted_wavenumbers = wavenumbers + shift

        interp_func = interp1d(
            shifted_wavenumbers, noisy_spectrum, kind="cubic", bounds_error=False, fill_value=0
        )
        aug_spectrum = interp_func(wavenumbers)

        background_2 = bg_mult[i] * background


        # aug_spectrum = np.maximum(aug_spectrum, 0)
        aug_spectrum = aug_spectrum + 1e-6
        aug_spectrum = aug_spectrum / np.max(aug_spectrum)
        aug_spectrum = aug_spectrum + background_2
        augmented_spectra.append(aug_spectrum)

    return np.array(augmented_spectra)


def create_augmented_dataset(input_file, output_file, background_file, n_augment=100):
    """
    Create augmented dataset from original spectra.
    """
    df = pd.read_csv(input_file)
    names = df.iloc[:, 0]  # Get the names of the lipids
    wavenumbers = df.columns[1:].astype(float).values
    spectra = df.iloc[:, 1:].values

    background_df = pd.read_csv(background_file)
    background = background_df.to_numpy()[:,0]
    interp = interp1d(np.linspace(0,1, len(background)), background)
    background = interp(np.linspace(0,1, spectra.shape[1]))
    background = normalize(np.flip(background)) # Get normalized background array
    

    augmented_data = []

    for idx, name in enumerate(names):
        spectrum = spectra[idx]
        aug_spectra = augment_data(spectrum, wavenumbers, background, n_augment=n_augment)

        spectra_df = pd.DataFrame(aug_spectra, columns=wavenumbers)
        spectra_df.insert(0, "name", name)
        augmented_data.append(spectra_df)

    df_augmented = pd.concat(augmented_data, ignore_index=True)
    df_augmented.to_csv(output_file, index=False)
    return df_augmented

import matplotlib.pyplot as plt
if __name__ == "__main__":
    classification_path = "dataset/classification_"
    denoising_path = "dataset/denoising_"
    background_path = "background/water_HSI_76.csv"

    classification_df_augmented = create_augmented_dataset(
        input_file=classification_path + "library.csv",
        output_file=classification_path + "val_data.csv",
        background_file="background/water_HSI_76.csv"
    )

    denoising_df_augmented = create_augmented_dataset(
        input_file=denoising_path + "library.csv",
        output_file=denoising_path + "val_data.csv",
        background_file="background/water_HSI_76.csv"
    )
    # lipid = df_augmented.iloc[105:131, 1:].T
    # plt.plot(lipid)
    # plt.show()

    print("Data preprocessing and augmentation completed!")
