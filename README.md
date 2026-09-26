# REALM: A Coarse-to-Fine Generative Framework for Embodied Reactive Listening

[![Status](https://img.shields.io/badge/Status-Under_Review-yellow.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

<div align="center">
  <video src="./docs/static/videos/realm_demo_with_audio.mp4" width="60%" controls></video>
</div>
<br>

This is the official PyTorch implementation for **REALM** (Reactive Embodied Audio-driven Listening Model). 

## 📖 Overview
**REALM** (**R**eactive **E**mbodied **A**udio-driven **L**istening **M**odel) is a coarse-to-fine framework for synthesizing responsive listener facial motion conditioned on speaker audio and listener motion history. Addressing the twin challenges of interaction timing and local facial variation, REALM pairs a delay-centered attention prior with adaptive history–audio gating, and augments a stable base motion trajectory with audio-conditioned stochastic expression residuals. We evaluate REALM across conversational benchmarks (ViCo, L2L) and validate its real-world physical viability via deployment on the **Ameca humanoid robot** alongside a double-blind user study.

### ✨ Core Contributions
* **Delay-Aware Reactive Fusion:** Introduces a shifted ALiBi attention prior centered on a nominal response lag ($\tau$) combined with learned adaptive gating ($g_t$). This softly biases cross-attention toward preceding speaker cues while dynamically balancing speaker evidence against listener motion history to maintain behavioral continuity.
* **Expression-Specific Stochastic Refinement:** Employs a coarse-to-fine architecture where a coarse decoder predicts base expression and head-pose trajectories, while a subsequent refinement module injects audio-conditioned stochastic residuals strictly into the non-rigid expression subspace ($\hat{\mathbf{r}}_t = \tilde{\mathbf{r}}_t$). This alleviates deterministic over-smoothing and recovers lifelike micro-dynamics (e.g., blinks and smiles) without destabilizing rigid head pose.
* **Physical Grounding & Embodiment Pipeline:** Implements a deterministic inverse kinematic mapping ($\mathbf{q}_t = \Phi^{-1}(\cdot)$) with relative motion calibration ($\hat{\mathbf{q}}_t = \mathbf{q}_0 + \Delta\mathbf{q}_t$) anchored to the robot's mechanical neutral state, ensuring hardware-safe actuation on physical humanoid hardware.

---

## ⚙️ Installation

1. **Get the code:**

    Since this repository is anonymized for double-blind review, `git clone` is disabled. Please click the **ZIP** button at the top of this page to download the source code as a `.zip` file, and extract it to your local machine.

    ```Bash
    cd REALM-main  # Or the name of the extracted directory
    ```

2. Create and activate the conda environment directly from the provided configuration file:

    ```Bash
    conda create -n realm python=3.10
    conda activate realm
    pip install -r requirements.txt
    ```

## 🗄️ Data Preparation
Please refer to the data preparation process outlined in the [ViCo Challenge Baseline repository](https://github.com/dc3ea9f/vico_challenge_baseline).



## 🚀 Quick Start (Inference)
Note on Double-Blind Compliance: Pre-trained model weights are temporarily withheld to maintain author anonymity during the review process. Full checkpoints and pre-trained models will be released upon paper acceptance.

To generate listener motions using the framework (once weights are available or after training):

1. Place the pre-trained REALM checkpoints in the checkpoints/ directory.

2. Run the inference script on the ViCo test/ood set:


    ```Bash
    python inference_vico.py \
        --config configs/realm.yaml \
        --checkpoint checkpoints/realm_best_refine.pt \
        --output_dir results/vico_outputs/
    ```


## 🏋️‍♂️ Training
To train the REALM framework from scratch on your prepared dataset, run:

```Bash
python train.py --config configs/realm.yaml
```