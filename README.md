# Audio EQ Infer

**audioeqinfer** is a Python package for discovering audio equalization (EQ) curves of an altered signal through probabilistic inference about the signal source.  
It supports both **importance sampling** and **Metropolis-Hastings (MH)** to infer flexible, data-driven EQ transformations from audio signals.

See examples notebook for usage examples.

Currently in beta.


## Features

- Flexible spline-based parameterization of EQ curves
- Probabilistic inference using:
  - Importance sampling
  - Metropolis-Hastings
- Handles large audio files with online chunked processing
- Modular design with current set up for flow models (`f_X`)
- Clean API for inference and training
- Easily adjustable number of EQ parameters


## Installation
This package is available on pypi

## Requirements

- Python
- PyTorch
- torch
- nflows
- numpy
- scipy
- matplotlib
- statsmodels
- pedalboard


