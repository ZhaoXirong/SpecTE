# SpecTE: Parameter Estimation for LAMOST Low-Resolution Stellar Spectra

## Overview

This repository contains the code, pre-trained models, and experimental data associated with the paper "SpecTE: Parameter Estimation for LAMOST Low-Resolution Stellar Spectra Based on Denoising Pre-training". SpecTE (Spectral Transformer Encoder) is a deep learning model designed to estimate stellar atmospheric parameters and chemical element abundances from low-resolution spectra obtained from the Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST).

## Key Features

- **Denoising Pre-training**: SpecTE leverages a denoising pre-training technique to learn a mapping from low-quality spectra to high-quality spectra, enhancing the extraction of high-quality spectral features and improving parameter estimation accuracy.
- **Transformer Architecture**: Utilizes the Transformer model to effectively extract features from sub-bands and analyze correlations between different spectral bands.
- **High Accuracy and Robustness**: Achieves superior performance in estimating stellar parameters and element abundances compared to existing methods like StarNet and StarGRUNet.

## Data

The repository includes:
- Pre-trained SpecTE models for parameter estimation.
- Experimental data used in the paper for reproducibility.
- Stellar catalog with estimated parameters and uncertainties.


## Prerequisites

- Python 3.8
- PyTorch
- Scikit-learn
- Matplotlib
- NumPy, Pandas, and other standard scientific libraries

