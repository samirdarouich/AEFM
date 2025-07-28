# Adaptive Equilibrium Flow Matching (AEFM)

[![arXiv](https://img.shields.io/badge/arXiv-2507.16521-b31b1b.svg)](https://arxiv.org/abs/2507.16521)

**Adaptive Transition State Refinement with Learned Equilibrium Flows**  
Samir Darouich, Vinh Tong, Tanja Bien, Johannes Kästner, Mathias Niepert

---

## 🧪 Overview

In this work, we introduce a new generative AI approach that improves the quality of initial guesses for TS structures. Our method can be combined with a variety of existing techniques, including both machine learning models and fast, approximate quantum methods, to refine their predictions and bring them closer to chemically accurate results.


## 📦 Installation

We recommend using a conda environment:

```bash
conda create -n aefm python=3.10
conda activate aefm
pip install .
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

## 📕 Citation 

@article{darouich2025adaptive,
  title={Adaptive Transition State Refinement with Learned Equilibrium Flows},
  author={Darouich, Samir and Tong, Vinh and Bien, Tanja and K{\"a}stner, Johannes and Niepert, Mathias},
  journal={arXiv preprint arXiv:2507.16521},
  year={2025}
}