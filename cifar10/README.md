# CIFAR-10 Experiments

This folder contains the implementation of **Learning to Defer under Expert Drift** on the **CIFAR-10** dataset.

---

## Dataset

We use the **CIFAR-10** image classification dataset, which consists of **60,000 images** across 10 classes (50,000 for training and 10,000 for testing).  
The task is to classify natural images into categories such as airplanes, cats, and trucks.

---

## Expert Simulation

To simulate expert drift on CIFAR-10, we build an **expert model based on statistics from CIFAR-10H**:  

- **CIFAR-10H** is a crowdsourced dataset that provides **soft labels** from multiple human annotators.  
- We fit parameterized models to these statistics, which allows us to generate **synthetic experts** with drifting accuracy over time.  
- The code for expert simulation is located in the [`expert/`](expert/) folder.

---

## Methods and Baselines

Our framework and baselines are implemented in the [`scripts/`](scripts/) directory:

- **`cifar10_defer_general.py`** – General L2D (baseline without temporal modeling)  
- **`cifar10_defer_gru.py`** – Temporal L2D with GRU (our main method)  
- **`cifar10_defer_perstep.py`** – Per-step L2D baseline  

Each script trains a WideResNet classifier with a defer option and evaluates under expert drift.

---

## Installation

We recommend creating a conda environment from the provided file:

```bash
conda env create -f ../environment.yml
conda activate l2d