# HateSpeech Experiments

This folder contains the implementation of **Learning to Defer under Expert Drift** on the **HateSpeech** dataset.

## Dataset

We use a **HateSpeech classification dataset**, which contains text samples labeled for hate/offensive speech detection.  
This dataset is well-suited for studying deferral because **expert annotators may drift over time** (e.g., becoming stricter or more lenient).  
We simulate such drift to evaluate our temporal L2D framework.


## Repository Structure

- **`scripts/`**  
  Training scripts for the three baselines we compare:  
  - `hatespeech_defer_general.py` – General L2D (baseline without temporal modeling)  
  - `hatespeech_defer_lstm.py` – Temporal L2D with LSTM (our main method)  
  - `hatespeech_defer_perstep.py` – Per-step L2D baseline  


## Installation

Install the environment using:

```bash
conda env create -f ../environment.yml
conda activate hs_legacy