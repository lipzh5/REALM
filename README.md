# REALM: A Coarse-to-Fine Generative Framework for Embodied Reactive Listening

<!-- [![Venue](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc/) -->
[![Status](https://img.shields.io/badge/Status-Under_Review-yellow.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This is the official PyTorch implementation for **REALM** (Reactive Embodied Audio-driven Listening Model). 

**[Project Page](https://anonymous.4open.science/w/REALM-3DCC/) | [Paper](#) | [Demo Video](docs/static/videos/realm_demo_with_audio.mp4) | [Pre-trained Models](#)**

## 📖 Overview
**REALM** (Reactive Embodied Audio-driven Listening Model) is a coarse-to-fine generative framework that synthesizes lifelike, reactive listener motions driven purely by speaker audio. Unlike existing methods that treat listening as an active generation task—which often results in unnatural deviation and expression over-smoothing—REALM explicitly models natural cognitive delays, enforces realistic quiescent states, and disentangles smooth head trajectories from rapid facial micro-expressions. We validate our approach by successfully deploying these synthesized motions directly onto the **Ameca humanoid robot**.

### ✨ Core Contributions
* **Reactive Gated Fusion:** Utilizes a shifted ALiBi mechanism and dynamic gating to explicitly model cognitive reaction delays ($\tau$). By balancing the speaker's acoustic trigger against the listener's motion history, it prevents unnatural deviations from the ground-truth manifold.
* **Coarse-to-Fine Stochastic Refinement:** Decouples smooth, low-frequency head poses from high-frequency facial dynamics. By injecting audio-modulated stochastic noise into the refinement stage, it overcomes deterministic over-smoothing to synthesize lifelike, rapid micro-expressions.
* **Physical Embodiment Pipeline:** Features an inverse kinematic mapping ($\mathbf{q} = \Phi^{-1}(\cdot)$) to translate abstract generative coefficients into hardware-safe control values, bridging the gap between digital avatars and physically embodied agents.

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone [https://github.com/lipzh5/REALM.git](https://github.com/lipzh5/REALM.git)
cd REALM
```

2. Create and activate the conda environment directly from the provided configuration file:
```bash
conda create -n realm python=3.10
conda activate realm
pip install -r requirements.txt
```

## 🗄️ Data Preparation
Please refer to the data prepraration process in [vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline)

## 🚀 Quick Start (Inference)
To generate listener motions using our pre-trained weights:

1. Download the pre-trained REALM checkpoints from [Link to Weights] and place them in the checkpoints/ directory.

2. Run the inference script on the ViCo test/ood set:
```bash
python inference_vico.py \
    --config configs/realm.yaml \
    --checkpoint checkpoints/realm_best_refine.pt \
    --output_dir results/vico_outputs/
```

## 🏋️‍♂️ Training

To train the REALM framework from scratch on your prepared dataset:
```bash
python train.py --config configs/realm.yaml
```

