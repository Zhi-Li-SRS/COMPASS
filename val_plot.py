import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def combine_representative_spectra():
    lib_path = os.path.join("Raman_dataset", "library.csv")
    lib_df = pd.read_csv(lib_path)
    allowed = ["d7-glucose", "d2-fructose", "d-tyrosine", "d-methionine", "d-leucine"]
    lib_df = lib_df[lib_df["name"].isin(allowed)]
    wavelengths = np.array([float(col) for col in lib_df.columns[1:]])

    rep_files = glob.glob(os.path.join("represent_spectra", "*.csv"))
    combined = {}
    for file in rep_files:
        subtype = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, sep=r"[\t,]+", engine="python")
        data = df["Mean"].values
        smoothed = savgol_filter(data, window_length=11, polyorder=3) if len(data) >= 11 else data
        norm = smoothed / np.max(smoothed)
        combined[subtype] = norm
    combined_df = pd.DataFrame(combined, index=wavelengths)
    rep_df = combined_df.T.reset_index().rename(columns={"index": "name"})
    rep_df.to_csv("represent_spectra/representative_spectra.csv", index=False)
    return combined_df, lib_df, wavelengths


def plot_spectra(combined_df, lib_df, wavelengths):
    out_dir = os.path.join("represent_spectra", "spectra_plot")
    os.makedirs(out_dir, exist_ok=True)

    colors = {
        "d7-glucose": "#FF0000",  # Red
        "d2-fructose": "#00FF00",  # Green
        "d-tyrosine": "#0000FF",  # Blue
        "d-methionine": "#FF00FF",  # Magenta
        "d-leucine": "#FFD700",  # Gold
    }
    for _, row in lib_df.iterrows():
        subtype = row["name"]
        ref_spec = row.values[1:].astype(float)
        # Plot reference spectrum individually
        plt.figure()
        plt.style.use("ggplot")
        plt.plot(wavelengths, ref_spec, label="Reference", color="black")
        plt.xlabel("Wavelength  (cm$^{-1}$)")
        plt.ylabel("Normalized Intensity (a.u.)")
        plt.title(f"Reference Spectrum: {subtype}", fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{subtype}_reference.svg"))
        plt.close()

        if subtype in combined_df.columns:
            rep_spec = combined_df[subtype].values
            rep_color = colors.get(subtype, "red")
            # Plot predicted representative spectrum individually
            plt.figure()
            plt.style.use("ggplot")
            plt.plot(wavelengths, rep_spec, label="Representative", color=rep_color)
            plt.xlabel("Wavelength (cm$^{-1}$)")
            plt.ylabel("Normalized Intensity (a.u.)")
            plt.title(f"Predicted Pixel Spectrum: {subtype}", fontweight="bold")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{subtype}_predict.svg"))
            plt.close()


if __name__ == "__main__":
    combined_df, lib_df, wavelengths = combine_representative_spectra()
    plot_spectra(combined_df, lib_df, wavelengths)
