# CheXpert Experiments

This folder contains the implementation of **Learning to Defer under Expert Drift** on the **CheXpert** dataset.

---

## Dataset

We use the **CheXpert** dataset, a large-scale chest X-ray dataset for multi-label classification of 14 medical conditions.  
For our experiments, we preprocess the data to simulate clinical decision-making scenarios under **expert drift**.

---

## Repository Structure

- **`data/`**  
  Contains scripts and utilities for preparing the CheXpert dataset into a format suitable for defer experiments.  
  We build temporal sequences of X-rays to study how expert performance may vary over time.

- **`evaluate/`**  
  Evaluation scripts for computing system accuracy, expert accuracy, classifier-only performance, coverage, and other metrics.

- **`expert/`**  
  Implements the simulation of experts.  
  We model expert drift by adjusting prediction accuracy across timesteps, simulating radiologists who **start with high accuracy but degrade over time** (or vice versa).

- **`models/`**  
  Stores model definitions:  
  - Baseline models from the original **L2D paper**  
  - Our **temporal L2D models**  
  - General and per-step baselines

- **`notebooks/`**  
  Contains the full source code from the original **L2D (Learn to Defer) paper** for CheXpert.  
  These notebooks serve as references and help validate our re-implementation.

- **`scripts/`**  
  Training scripts for running baselines and our method:  
  - `train_defer_general.py` – General L2D (baseline without temporal modeling)  
  - `train_defer_lstm.py` – Temporal L2D with LSTM (our main method)  
  - `train_defer_perstep.py` – Per-step L2D baseline  

- **`utils/`**  
  Contains utility functions for data loading, metrics calculation, logging, and visualization.

---

## Installation

You can install the required environment via:

```bash
conda env create -f ../environment.yml
conda activate l2d