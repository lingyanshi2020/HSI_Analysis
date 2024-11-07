# Multimodal and Hyperspectral Imaging Analysis (JoVE)

This repository contains analysis tools for processing and analyzing multimodal imaging data, and h-SRS image. 

## Features

- **Hyperspectral SRS Analysis (`Kmeans_hSRS.py`)**
  - K-means clustering for chemical compositions in h-SRS data.
  - Spectral profile visualization

- **Ratio Analysis (`ratio_analysis.py`)**
  - Multimodal image processing (lipid unsaturation and redox ratio)
  - Region-specific analysis (nuclei, cytoplasm, whole cell)
  - Statistical comparison between control and disease groups


## Installation

1. Clone this repository:
```bash
git clone https://github.com/Zhi-Li-SRS/HSI_Analysis
cd HSI_Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Hyperspectral SRS Analysis

The `Kmeans_hSRS.py` script performs spectral clustering analysis on h-SRS data.

```bash
python Kmeans_hSRS.py --base_folder /path/to/data \
                      --output_folder /path/to/output \
                      --n_clusters 8 \
                      --wave_start 2700 \
                      --wave_end 3150 \
                      --wave_points 60
```

Parameters:
- `--base_folder`: Directory containing hyperspectral image stacks
- `--output_folder`: Directory for saving results
- `--n_clusters`: Number of clusters for K-means (default: 8)
- `--wave_start`: Starting wavenumber (default: 2700 cm⁻¹)
- `--wave_end`: Ending wavenumber (default: 3150 cm⁻¹)
- `--wave_points`: Number of spectral points (default: 60)

### Ratio Analysis

The `ratio_analysis.py` script processes multimodal imaging data and generates statistical comparisons between control and disease groups.

```bash
python ratio_analysis.py --folders /path/to/control /path/to/disease \
                        --output-dir figures/boxplot \
                        --sample-size 10000
```

Parameters:
- `--folders`: List of folders containing ROI data
- `--output-dir`: Directory for saving generated plots (default: figures/boxplot)
- `--sample-size`: Number of samples per label for analysis (default: 10000)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
