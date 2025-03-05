import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import skewnorm

def normalize(array, max=1, min=0, eps=1e-8, axis=None):
    """
    Normalize array by minimum and maximum values
    """
    min_val = np.min(array)
    max_val = np.max(array)
    diff = max_val - min_val

    if axis is None:
        if np.all(array==0):
            raise Exception('All values in array are zero.')
        else:
            norm = ((array - min_val)/(max_val - min_val + eps))*(max - min) + min
    else:
        norm = array.copy()
        eps = eps * np.ones_like(array)
        idx = np.where(diff != 0)[0]
        norm[idx] = (((array[idx].T - min_val[idx]) / (diff[idx]+eps[idx])) * (max - min) + min).T

    return norm


# def normalizebyvalue(array, max_val, min_val, max=1, min=0, eps=1e-8, axis=None):
#     diff = max_val - min_val
#     if axis is None:
#         if np.all(diff == 0):
#             norm = array
#         else:
#             norm = ((array - min_val) / (diff+eps)) * (max - min) + min
#     else:
#         norm = array.copy()
#         eps = eps * np.ones_like(array)
#         idx = np.where(diff != 0)[0]
#         norm[idx] = (((array[idx].T - min_val[idx]) / (diff[idx]+eps[idx])) * (max - min) + min).T
#
#     return norm



def preprocess_data(input_file, output_file):
    """
    Preprocess the spectral data file by transposing and reformatting.
    """
    df = pd.read_csv(input_file, index_col=0)

    df_transposed = df.transpose()

    df_transposed.to_csv(output_file)
    return df_transposed


def augment_data(spectrum, wavenumbers, background, n_augment=100, noise_level=0.1, bg_level=0.5, max_shift=15, name="Background"):
    """
    Augment a single spectrum by adding noise and random shifts.
    Ensures all values are positive and within reasonable range for Raman spectra.
    """
    augmented_spectra = []
    original_max = 1 #np.max(spectrum)  # Get original spectrum's max value

    # Introduce randomness to noise and baseline
    bg_mult = np.sort(np.maximum(np.random.normal(loc=bg_level, scale=2 * noise_level, size=n_augment), 2 * noise_level))
    noise_mult = np.sort(np.maximum(np.abs(skewnorm.rvs(a=-4, loc=noise_level, scale=0.1 * noise_level, size=n_augment)), 0.5 * noise_level))


    for i in range(n_augment):
        # Add noise
        noise = np.random.normal(0, noise_mult[i] * original_max, len(spectrum))
        noisy_spectrum = spectrum + noise

        # Shift wavenumbers
        shift = np.random.uniform(-max_shift / 2, max_shift / 2)  # Reduced shift range
        shifted_wavenumbers = wavenumbers + shift
        interp_func = interp1d(
            shifted_wavenumbers, noisy_spectrum, kind="cubic", bounds_error=False, fill_value=0
        )
        aug_spectrum = interp_func(wavenumbers)

        # Normalize spectra
        if name != "Background":
            aug_spectrum = aug_spectrum + 1e-6
            aug_spectrum = aug_spectrum / np.max(aug_spectrum)

        # Add baseline curve
        baseline = bg_mult[i] * background
        aug_spectrum = aug_spectrum + baseline

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
        aug_spectra = augment_data(spectrum, wavenumbers, background, n_augment=n_augment,name=name)

        spectra_df = pd.DataFrame(aug_spectra, columns=wavenumbers)
        spectra_df.insert(0, "name", name)
        augmented_data.append(spectra_df)

    # Add Background Class
    spectrum = np.zeros_like(spectra[-1])
    name = "Background"
    aug_spectra = augment_data(spectrum, wavenumbers, background, n_augment=n_augment, name=name)
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
