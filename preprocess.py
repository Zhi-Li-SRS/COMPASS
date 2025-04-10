import os
from typing import List
import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import interp1d
from tqdm import tqdm

def convert_to_csv(xlsx_path, csv_path):
    df = pd.read_excel(xlsx_path)
    df.to_csv(csv_path, index=False)
    print(f"Successfully converted {xlsx_path} to {csv_path}")

def preprocess_csv(input_file: str, output_file: str):
    """
    Preprocess the spectral data file by transposing and reformatting.
    """
    df = pd.read_csv(input_file, index_col=0)

    df_transposed = df.transpose()

    df_transposed.to_csv(output_file)
    return df_transposed


def augment_data(
    spectrum, wavenumbers, background, n_augment=400, noise_level=0.2, bg_level=1.1, bg_scale=1, max_shift=15
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
    background = background_df.to_numpy()[:, 0]
    interp = interp1d(np.linspace(0, 1, len(background)), background)
    background = interp(np.linspace(0, 1, spectra.shape[1]))
    background = np.flip(background)
    background = background / np.max(background)

    augmented_data = []

    for idx, name in enumerate(names):
        spectrum = spectra[idx]
        aug_spectra = augment_data(spectrum, wavenumbers, background, n_augment=n_augment)

        spectra_df = pd.DataFrame(aug_spectra, columns=wavenumbers)
        spectra_df.insert(0, "name", name)
        augmented_data.append(spectra_df)

    df_augmented = pd.concat(augmented_data, ignore_index=True)
    df_augmented.to_csv(output_file, index=False)


def extract_bg_spectra(bg_files, output_csv, wavenumbers, total_samples=800):
    """
    Extract background spectra from real hyperspectral images.

    Args:
        bg_files (list): List of file paths to background tif images.
        output_csv (str): Output CSV file path.
        wavenumbers (np.array): Wavenumbers for interpolation.
        total_samples (int): Total number of spectra to extract.
    """
    sample_per_file = total_samples // len(bg_files)
    samples_distribution = [sample_per_file] * (len(bg_files) - 1)
    samples_distribution.append(total_samples - sum(samples_distribution))

    all_spectra = []
    for bg_file, num_samples in zip(bg_files, samples_distribution):
        print(f"Processing {bg_file} for  {num_samples} samples...")
        bg_image = tifffile.imread(bg_file)
        bg_image = np.flip(bg_image, axis=0)
        N, height, width = bg_image.shape
        img_wavenumbers = np.linspace(wavenumbers[0], wavenumbers[-1], N)
        bg_spectra = bg_image.reshape(N, -1).T  # (height * width, N)
        non_zero_mask = ~np.all(bg_spectra == 0, axis=1)
        valid_spectra = bg_spectra[non_zero_mask]

        num_valid_pixels = valid_spectra.shape[0]
        if num_valid_pixels == 0:
            print(f"No valid pixels found in {bg_file}. Skipping...")
            continue
        if num_valid_pixels < num_samples:
            print(f"Not enough valid pixels in {bg_file}. Using all available pixels.")
            indices = np.arange(num_valid_pixels)
        else:
            indices = np.random.choice(num_valid_pixels, num_samples, replace=False)

        sampled_spectra = valid_spectra[indices]
        processed_spectra = []
        for spectrum in tqdm(sampled_spectra):
            normalized = spectrum / np.max(spectrum)
            f = interp1d(
                img_wavenumbers,
                normalized,
                kind="cubic",
                bounds_error=False,
                fill_value=(spectrum[0], spectrum[-1]),
            )
            interpolated = f(wavenumbers)

            processed_spectra.append(interpolated)
        all_spectra.extend(processed_spectra)

    spectra_df = pd.DataFrame(all_spectra, columns=wavenumbers)
    spectra_df.insert(0, "name", "background")
    spectra_df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_spectra)} background spectra to {output_csv}")

    return spectra_df


def create_combined_dataset(original_csv, bg_csv, output_csv):
    """
    Combine original training data with background spectra

    Args:
        original_csv (str): Path to original training data CSV
        background_csv (str): Path to background spectra CSV
        output_csv (str): Path to save combined dataset
    """

    original_df = pd.read_csv(original_csv)
    background_df = pd.read_csv(bg_csv)

    combined_df = pd.concat([original_df, background_df], ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined dataset saved to {output_csv} with {len(combined_df)} total samples")

    return combined_df


def create_train_reference(lipid_csv: str, protein_csv : str, output_csv: str, target_names: List[str]):
    """
    Create a reference dataset by selecting specific spectra from lipid and protein libraries.

    Args:
        lipid_csv (str): Path to the lipid library CSV.
        protein_csv (str): Path to the protein library CSV.
        output_csv (str): Path to save the combined reference dataset.
        target_names (list): List of names to select from the libraries.
    """
    lipid_df = pd.read_csv(lipid_csv)
    protein_df = pd.read_csv(protein_csv)

    # Filter dataframes
    lipid_filtered = lipid_df[lipid_df['name'].isin(target_names)].copy()
    protein_filtered = protein_df[protein_df['name'].isin(target_names)].copy()

    # Modify names
    lipid_filtered['name'] = lipid_filtered['name'] + '_lipid'
    protein_filtered['name'] = protein_filtered['name'] + '_protein'
    combined_df = pd.concat([lipid_filtered, protein_filtered], ignore_index=True)

    combined_df.to_csv(output_csv, index=False)
    print(f"Reference dataset saved to {output_csv} with {len(combined_df)} total samples")

   

if __name__ == "__main__":
    
    # --- Create Reference Dataset ---
    # target_names = ['D7-glucose', 'D2-fructose', 'D-tyrosine', 'D-methionine', 'D-leucine']
    
    # create_train_reference(
    #     lipid_csv="Raman_dataset/raw/CD_lipid_library.csv",
    #     protein_csv="Raman_dataset/raw/CD_protein_library.csv",
    #     output_csv="Raman_dataset/ten_molecules_train/train_reference.csv",
    #     target_names=target_names
    # )
    # --- End Create Reference Dataset ---
    
    
    # --- Create Augmented Dataset ---
    # create_augmented_dataset(
    #     "Raman_dataset/molecules_9/train_reference.csv",
    #     "Raman_dataset/molecules_9/val_data.csv",
    #     "background/CD_HSI_76.csv",
    #     n_augment=200
    # )
    # --- End Create Augmented Dataset ---

    # -- Extract background spectra ---
    # train_df = pd.read_csv("Raman_dataset/ten_molecules_train/train_data.csv")
    # wavenumbers = train_df.columns[1:].astype(float).values
    # bg_files = [
    #     "Raman_dataset/background/bg1.tif",
    #     "Raman_dataset/background/bg2.tif",
    #     "Raman_dataset/background/bg3.tif",
    # ]
    # extract_bg_spectra(
    #     bg_files=bg_files,
    #     output_csv="Raman_dataset/raw/val_background_spectra.csv",
    #     wavenumbers=wavenumbers,
    #     total_samples=200
    # )
    # --- End Extract background spectra ---
    
    #// --- Create Combined Dataset ---
    create_combined_dataset(
        original_csv="Raman_dataset/molecules_9/val_data.csv",
        bg_csv="Raman_dataset/raw/val_background_spectra.csv",
        output_csv="Raman_dataset/molecules_9/val_data.csv",
    )
    #// --- End Create Combined Dataset ---   
    

   
  