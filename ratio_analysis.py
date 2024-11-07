import argparse
import glob
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from skimage import io


class RatioAnalyzer:
    """
    This class handles the processing of ROI (Region of Interest) folders containing
    multimodal images, performs pixel-wise analysis, and generates statistical plots
    """

    def __init__(self):
        """Initialize the label mapping for compari."""
        self.label_info = {1: "Nuceli", 2: "Cytoplasm", 3: "Cell", 4: "Lipid droplets"}
        self.regions_of_interest = ["Nuclei", "Cytoplasm"]

    def process_roi_folder(self, roi_path, folder_name, folder_type):
        """
        Process a single ROI folder containing images.

        Args:
            roi_path (str): Path to the ROI folder
            folder_name (str): Name of the parent folder
            folder_type (str): control or disease

        """
        roi_name = os.path.basename(roi_path).lower()
        try:
            lipid_unsaturation = io.imread(os.path.join(roi_path, "lipid_unsaturation.tif"))
            redox_ratio = io.imread(os.path.join(roi_path, "redox_ratio.tif"))
            segmentation = io.imread(os.path.join(roi_path, f"{roi_name}_labels.tif"))
        except FileNotFoundError as e:
            print(f"Error loading images in {roi_path}: {e}")
            return []

        data = []
        for label_value, label_name in self.label_info.items():
            mask = (segmentation == label_value) & (lipid_unsaturation > 0) & (redox_ratio > 0)
            lipid_ratio = lipid_unsaturation[mask]
            redox_ratio = redox_ratio[mask]

            if len(lipid_ratio) > 0 and len(redox_ratio) > 0:
                for lipid_val, redox_val in zip(lipid_ratio, redox_ratio):
                    if lipid_val < 0.05 or redox_val < 0.05:
                        continue
                    data.append(
                        {
                            "name": folder_name,
                            "type": folder_type,
                            "roi": os.path.basename(roi_path),
                            "label": label_name,
                            "lipid_unsaturation": lipid_val,
                            "redox_ratio": redox_val,
                        }
                    )
        return data

    def process_folder(self, folder_path):
        """
        Process an entire folder containing multiple ROI subfolders (ROI1, ROI2...).

        Args:
            folder_path (str): Path to the main folder

        """
        folder_name = os.path.basename(folder_path)
        folder_type = "control" if "control" in folder_name.lower() else "disease"

        all_roi_data = []
        for roi_folder in glob.glob(os.path.join(folder_path, "ROI*")):
            roi_data = self.process_roi_folder(roi_folder, folder_name, folder_type)
            all_roi_data.extend(roi_data)
        return all_roi_data


class DataProcessor:
    """
    A class for analyzing saved ratio data from CSV files.

    """

    @staticmethod
    def load_and_sample_data(csv_files, sample_size=10000):
        """

        Args:
            csv_files (list): List of CSV file paths
            sample_size (int): Number of samples to take per label

        Returns:
            pd.DataFrame: Combined and sampled data
        """
        all_data = []
        for file in csv_files:
            df = pd.read_csv(file)
            df["type"] = "control" if "control" in file.lower() else "disease"

            # Filter out extreme values
            df = df[(df["lipid_unsaturation"] > 0.1) & (df["lipid_unsaturation"] < 1)]
            df = df[(df["redox_ratio"] > 0.1) & (df["redox_ratio"] < 1)]

            # Sample data if necessary
            if len(df) > sample_size:
                df = (
                    df.groupby("label")
                    .apply(
                        lambda x: x.sample(
                            n=min(len(x), sample_size // len(df["label"].unique())), random_state=42
                        )
                    )
                    .reset_index(drop=True)
                )
            all_data.append(df)
        return pd.concat(all_data, ignore_index=True)


class PlotGenerator:

    def __init__(self):
        plt.style.use("ggplot")
        self.regions = ["Nuclei", "Cytoplasm", "Cell"]

    @staticmethod
    def _get_significance(p_val):
        """
        Convert p-value to significance stars.

        Args:
            p_val (float): P-value from statistical test

        Returns:
            str: Significance stars (* for p<0.05, ** for p<0.01, *** for p<0.001)
        """
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        else:
            return "ns"

    def boxplot_comparison(self, data, output_path, metric):
        """Generate boxplot comparing control vs disease cross different regions."""
        plt.figure(figsize=(8, 5))
        data = data[data["label"].isin(self.regions)]

        sns.boxplot(
            x="label",
            y=metric,
            hue="type",
            data=data,
            order=self.regions,
            hue_order=["control", "disease"],
            palette={"control": "#7FB2D5", "disease": "#F47F72"},
            width=0.2,
            boxprops=dict(alpha=0.7),
            medianprops=dict(color="black"),
        )

        plt.title(
            f"BPD vs Healthy Comparison_{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold"
        )
        plt.xlabel("Region", fontsize=12, fontweight="bold")
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        plt.xticks(fontsize=12, fontweight="bold")
        plt.yticks(fontsize=12, fontweight="bold")

        self._add_significance(data, metric, self.regions)

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"bpd_vs_healthy_comparison_{metric}.svg"), dpi=300)
        plt.close()

    def _add_significance(self, data, metric, regions):
        """Add significance markers to BPD vs healthy comparison plot."""
        y_max = data[metric].max()
        y_range = data[metric].max() - data[metric].min()

        for i, region in enumerate(regions):
            bpd_data = data[(data["type"] == "BPD") & (data["label"] == region)][metric]
            healthy_data = data[(data["type"] == "Healthy") & (data["label"] == region)][metric]

            if len(bpd_data) > 0 and len(healthy_data) > 0:
                _, p_val = stats.ttest_ind(bpd_data, healthy_data)
                y_pos = y_max + 0.05 * y_range
                x_pos = i

                plt.plot(
                    [x_pos - 0.2, x_pos - 0.2, x_pos + 0.2, x_pos + 0.2],
                    [y_pos, y_pos + 0.03 * y_range, y_pos + 0.03 * y_range, y_pos],
                    lw=1,
                    c="black",
                )
                plt.text(
                    x_pos,
                    y_pos + 0.04 * y_range,
                    self._get_significance(p_val),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )


def main():
    """Main function to run the tissue analysis pipeline."""

    parser = argparse.ArgumentParser(description="Analyze tissue images and generate statistical plots.")
    parser.add_argument(
        "--folders", nargs="+", required=True, help="List of folder paths containing tissue images"
    )
    parser.add_argument(
        "--output-dir", default="figures/boxplot", help="Output directory for generated plots"
    )
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of samples to take per label")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analyzer = RatioAnalyzer()
    plot_generator = PlotGenerator()

    # Process folders and generate CSV files
    csv_files = []
    for folder in args.folders:
        data = analyzer.process_folder(folder)
        csv_path = os.path.join(folder, f"{os.path.basename(folder)}.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)
        csv_files.append(csv_path)

    all_data = DataProcessor.load_and_sample_data(csv_files, args.sample_size)

    metrics = ["lipid_unsaturation", "redox_ratio"]
    for metric in metrics:
        plot_generator.boxplot_comparison(all_data, args.output_dir, metric)


if __name__ == "__main__":
    # start_time = time.time()
    main()
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.2f} seconds")
    # sys.exit(0)
