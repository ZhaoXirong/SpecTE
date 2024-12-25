# SpecTE: Parameter Estimation for LAMOST Low-Resolution Stellar Spectra

## Overview

This repository contains the code, pre-trained models, and experimental data associated with the paper "SpecTE: Parameter Estimation for LAMOST Low-Resolution Stellar Spectra Based on Denoising Pre-training". SpecTE (Spectral Transformer Encoder) is a deep learning model designed to estimate stellar atmospheric parameters and chemical element abundances from low-resolution spectra obtained from the Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST).

## Key Features

- **Denoising Pre-training**: SpecTE leverages a denoising pre-training technique to learn a mapping from low-quality spectra to high-quality spectra, enhancing the extraction of high-quality spectral features and improving parameter estimation accuracy.
- **Transformer Architecture**: Utilizes the Transformer model to effectively extract features from sub-bands and analyze correlations between different spectral bands.
- **High Accuracy and Robustness**: Achieves superior performance in estimating stellar parameters and element abundances compared to existing methods like StarNet and StarGRUNet.

## Data and Code Structure

The repository is organized into four main parts, each contained within its respective file:

1. **Data_download_and_preprocessing**: Contains code for downloading and pre-processing the dataset.
2. **SpecTE**: Houses the model architecture and training-related code.
3. **Model_evaluation**: Includes code for evaluating the performance of the SpecTE model.
4. **SpecTE-LAMOST_catalog**: Contains code for generating the SpecTE-LAMOST catalog and evaluating the catalog.


## Prerequisites

- Python 3.8
- PyTorch
- Scikit-learn
- Matplotlib
- NumPy, Pandas, and other standard scientific libraries

