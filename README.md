# l2d-with-expert-drift

This repository contains the official implementation of the paper **"Learning to Defer under Expert Drift"**.  
We study the problem of **expert drift**, where human experts (e.g., doctors, annotators, moderators) may change their performance over time. Our framework, **temporal L2D (Learn-to-Defer)**, enables machine learning models to dynamically decide when to predict and when to defer to human experts under such drift.

---

## Implemented Datasets

We provide code and experiments on three real-world datasets:

- [**CheXpert**](chexpert/README.md): medical imaging with radiologist drift  
- [**CIFAR-10**](cifar10/README.md): image classification with synthetic expert drift  
- [**HateSpeech**](hatespeech/README.md): text classification with annotator drift  

We also include **synthetic data experiments** in [synthetic/](synthetic/), for controlled analysis and ablation studies.

---

## Repository Structure
l2d-with-expert-drift/
│
├── chexpert/        # CheXpert dataset experiments
├── cifar10/         # CIFAR-10 dataset experiments
├── hatespeech/      # HateSpeech dataset experiments
├── synthetic/       # Synthetic drift simulations
└── README.md

## Getting Started

### Installation

We recommend using **conda** to manage dependencies.  
Create from our environment file:
conda env create -f environment.yml