# Extracting Exchange Coupling via Bayesian Inference

## Overview
This repository contains a high-performance computational pipeline designed to estimate the effective exchange coupling parameters in 2D magnetic materials (like CrI3)[cite: 1]. By coupling Monte Carlo forward-modeling with a Bayesian inference framework, this tool provides statistically rigorous parameter estimates alongside credible uncertainty intervals[cite: 1].

## Key Features
* **High-Performance Simulations:** Implements the Metropolis algorithm for a 2D square-lattice Ising model[cite: 1, 2]. Uses **Numba** for JIT compilation and **Joblib** for parallelization across temperature arrays, drastically reducing computational overhead[cite: 2].
* **Bayesian Inference Framework:** Extracts posterior distributions of the coupling parameter (J) by comparing simulated magnetization curves against experimental data[cite: 1, 2].
* **Automated Data Digitization:** Includes a computer-vision utility script (`PIL` and `Pandas`) to extract, clean, and bin experimental magnetization data directly from literature plots[cite: 3].
* **Synthetic Data Validation:** Features a module to generate simulated, noisy datasets to validate the accuracy and limitations of the inference pipeline[cite: 4].

## Tech Stack
* **Language:** Python 3.x
* **Core Libraries:** NumPy, Pandas, Matplotlib
* **Performance/Optimization:** Numba, Joblib[cite: 2]
* **Image Processing:** Pillow (PIL)[cite: 3]

## Project Structure
* `inference.py` - Core simulation and Bayesian inference logic[cite: 2].
* `data_extractor.py` - Script for extracting ZFC/FC data points from target plot images[cite: 3].
* `synthetic_generator.py` - Generates noisy Ising data for baseline testing[cite: 4].

## Quick Start
1. Clone the repository.
2. Install dependencies: `pip install numpy pandas matplotlib numba joblib pillow`
3. Run `python data_extractor.py` to generate the CSV datasets from the provided image.
4. Run `python inference.py` to execute the Monte Carlo simulation and view the Bayesian fit.
