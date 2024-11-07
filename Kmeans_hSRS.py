import argparse
import glob
import os

import matplotlib
import numpy as np
import rampy as rp
import tifffile
from matplotlib import pyplot as plt
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def baseline_als(y, lam=10**3, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction
    Args:
        y: input spectrum
        lam: smoothness parameter
        p: asymmetry parameter
        niter: number of iterations
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    y = np.asarray(y)

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return y - z


def normalize_spectrum(spectrum, method="intensity"):
    """
    Normalize spectrum using different methods
    Args:
        spectrum: input spectrum
        method: normalization method ('minmax', 'intensity', 'area')
    """
    if np.all(spectrum == 0):
        return spectrum

    if method == "minmax":
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        if max_val - min_val != 0:
            return (spectrum - min_val) / (max_val - min_val)
        return spectrum

    elif method == "intensity":
        max_val = np.max(np.abs(spectrum))
        if max_val != 0:
            return spectrum / max_val
        return spectrum

    elif method == "area":
        area = np.trapz(spectrum)
        if area != 0:
            return spectrum / area
        return spectrum

    raise ValueError(f"Unknown normalization method: {method}")


def smooth_spectrum(spectrum, lamda=0.2):
    """
    Smooth spectrum using Savitzky-Golay filter
    Args
        spectrum: input spectrum
        lamda: smoothing parameter
    """
    if np.all(spectrum == 0):
        return spectrum

    smoothed = rp.smooth(np.arange(len(spectrum)), spectrum, method="whittaker", Lambda=lamda)
    return smoothed


class hSRSAnalyzer:
    def __init__(
        self, base_folder, output_folder, n_clusters=3, wave_start=2700, wave_end=3105, wave_points=59
    ):
        self.base_folder = base_folder
        self.output_folder = output_folder
        self.n_clusters = n_clusters
        self.wavenumbers = np.linspace(wave_start, wave_end, wave_points)
        self.cmaps = matplotlib.colormaps["tab10"](np.linspace(0, 1, n_clusters))

    def preprocess_spectra(self, spectra):
        """
        Preprocess each spectrum with baseline correction, normalization and smoothing
        Args:
            spectra: input spectra (N x n_pixels)
        """
        processed_spectra = np.zeros_like(spectra, dtype=np.float32)

        for i in range(spectra.shape[1]):
            if np.all(spectra[:, i] == 0):
                continue

            spectrum = spectra[:, i].astype(np.float32)

            spectrum = baseline_als(spectrum)
            spectrum = normalize_spectrum(spectrum, method="intensity")
            spectrum = smooth_spectrum(spectrum)
            processed_spectra[:, i] = spectrum

        return processed_spectra

    def load_and_preprocess(self, image_path):
        """
        Load and preprocess the hyperspectral image stack
        """
        # Load image stack
        image = tifffile.imread(image_path)
        if len(image.shape) != 3:
            raise ValueError("Image should be a hyperspectral image stack")

        N, height, width = image.shape

        pixel_intensities = np.sum(image, axis=0)
        mask = pixel_intensities > 0

        self.background = ~mask
        nonzero_indices = np.where(mask)

        reshaped_spectra = image[:, mask].reshape(N, -1)
        processed_spectra = self.preprocess_spectra(reshaped_spectra)

        return processed_spectra, (N, height, width), nonzero_indices

    def perform_kmeans(self, processed_spectra, original_shape, nonzero_indices):
        N, height, width = original_shape
        spectra_for_clustering = processed_spectra.T  # (n_pixels, N)
        scaler = StandardScaler()
        scaled_spectra = scaler.fit_transform(spectra_for_clustering)

        # Perform K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_spectra)

        # # Calculate silhouette score
        # silhouette_avg = silhouette_score(scaled_spectra, labels)
        # print(f"Silhouette Score: {silhouette_avg:.3f}")

        cluster_map = np.full((height, width), -1, dtype=int)
        y_indices, x_indices = nonzero_indices
        cluster_map[y_indices, x_indices] = labels

        # Calculate mean spectra and standard deviations for each cluster
        cluster_means = np.zeros((N, self.n_clusters))
        cluster_stds = np.zeros((N, self.n_clusters))

        for i in range(self.n_clusters):
            cluster_spectra = processed_spectra[:, labels == i]
            if cluster_spectra.size > 0:
                cluster_means[:, i] = np.mean(cluster_spectra, axis=1)
                cluster_stds[:, i] = np.std(cluster_spectra, axis=1)

        return cluster_map, cluster_means, cluster_stds

    def save_results(self, cluster_map, cluster_means, cluster_stds, image_shape):
        """
        Save clustering map and spectral profiles
        """
        output_path = self.output_folder
        os.makedirs(output_path, exist_ok=True)

        # Save segmentation map
        rgb_image = np.zeros((*image_shape[1:], 3))
        for i in range(self.n_clusters):
            mask = cluster_map == i
            rgb_image[mask] = self.cmaps[i, :3]

        background = (cluster_map == -1) | self.background

        rgb_image[background] = [0, 0, 0]
        tifffile.imwrite(os.path.join(output_path, "cluster_map.tif"), (rgb_image * 255).astype(np.uint8))

        plt.figure(figsize=(12, 8))

        for i in range(self.n_clusters):
            y = cluster_means[:, i] + i * 0.3
            color = self.cmaps[i, :3]
            plt.plot(self.wavenumbers, y, color=color, label=f"Cluster {i+1}")
            plt.fill_between(
                self.wavenumbers, y - cluster_stds[:, i], y + cluster_stds[:, i], color=color, alpha=0.2
            )

        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Normalized Intensity")
        plt.title("Clustered Spectral Profiles")
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(output_path, "spectral_profiles.tif"), dpi=300, bbox_inches="tight")
        plt.close()

    def process_dataset(self):
        """
        Process all hyperspectral image stacks in the base folder
        """

        tiff_format = os.path.join(self.base_folder, "*.tif*")
        image_files = glob.glob(tiff_format)

        for image_path in image_files:
            print(f"Processing {os.path.basename(image_path)}...")

            processed_spectra, original_shape, nonzero_indices = self.load_and_preprocess(image_path)

            cluster_map, cluster_means, cluster_stds = self.perform_kmeans(
                processed_spectra, original_shape, nonzero_indices
            )

            self.save_results(cluster_map, cluster_means, cluster_stds, original_shape)

        print("Processing complete!")


def main():
    parser = argparse.ArgumentParser(description="Hyperspectral SRS Image Analysis")
    parser.add_argument(
        "--base_folder", type=str, default="alveoli", help="Base folder containing the image data"
    )
    parser.add_argument("--output_folder", type=str, default="clustering", help="Output folder for results")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for K-means")
    parser.add_argument("--wave_start", type=float, default=2700, help="Starting wavenumber")
    parser.add_argument("--wave_end", type=float, default=3105, help="Ending wavenumber")
    parser.add_argument("--wave_points", type=int, default=59, help="Number of wavenumber points")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    analyzer = hSRSAnalyzer(
        base_folder=args.base_folder,
        output_folder=args.output_folder,
        n_clusters=args.n_clusters,
        wave_start=args.wave_start,
        wave_end=args.wave_end,
        wave_points=args.wave_points,
    )

    analyzer.process_dataset()


if __name__ == "__main__":
    main()
