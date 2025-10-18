# Adaptive Equilibrium Flow Matching (AEFM)

[![arXiv](https://img.shields.io/badge/arXiv-2507.16521-b31b1b.svg)](https://arxiv.org/abs/2507.16521)

**Adaptive Transition State Refinement with Learned Equilibrium Flows**  
Samir Darouich, Vinh Tong, Tanja Bien, Johannes Kästner, Mathias Niepert

---

## 🧪 Overview

In this work, we introduce a new generative AI approach that improves the quality of initial guesses for TS structures. Our method can be combined with a variety of existing techniques, including both machine learning models and fast, approximate quantum methods, to refine their predictions and bring them closer to chemically accurate results.

AEFM consists of two main innovations:

1) Adaptive Prior
2) Equilibrium Flow Matching

### Adaptive Prior

To simulate the expected error of the low-fidelity method during training, noise is added to the ground-truth TS samples with a magnitude matching the typical low-fidelity error. This enables the model to handle low-fidelity samples from different sources without being explicitly trained on them.

### Equilibrium Flow Matching

In our Equilibrium Flow Matching (EFM) framework, the model learns to predict the endpoint of the integration path and refines this estimate through fixed-point iteration until convergence — effectively acting as a learned approximation of the ODE solution operator. Combined with Anderson acceleration, EFM enables stable and remarkably fast inference, requiring only a fraction of inference steps compared to conventional flow matching. Conceptually, EFM mirrors the inference procedure in Deep Equilibrium Models where a neural network is iterated to convergence at test time to find a fixed point.


https://github.com/user-attachments/assets/e86fbaf5-941e-459b-8e57-e88d07041ac8



## 📦 Installation

We recommend using a conda environment:

```bash
conda create -n aefm python=3.12
conda activate aefm
pip install -e .
```

## ⚙️ Usage

### 1. Training AEFM

```bash
aefm_train experiment=xtb_ci_neb run.data_dir=/your/custom/data/path
```

### 2. Sampling with AEFM
```bash
aefm_sample globals.model=/your/custom/model/path globals.samples_path=/your/custom/samples/path globals.reference_path=/your/custom/reference/path
```

## Reproduction

Pretrained models and the dataset, including the data split, are available at: https://zenodo.org/records/16414436

## 📕 Citation 

@article{darouich2025adaptive,
  title={Adaptive Transition State Refinement with Learned Equilibrium Flows},
  author={Darouich, Samir and Tong, Vinh and Bien, Tanja and K{\"a}stner, Johannes and Niepert, Mathias},
  journal={arXiv preprint arXiv:2507.16521},
  year={2025}
}
