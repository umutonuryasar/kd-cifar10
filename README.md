# KD-CIFAR10: Knowledge Distillation Ablation Study

A systematic comparison of **Logit-KD** and **Feature-KD** strategies for compressing
ResNet-50 (teacher) into ResNet-18 (student) on CIFAR-10.

> **Paper:** *Knowledge Distillation for Image Classification: A Systematic Ablation of
> Logit-KD and Feature-KD on CIFAR-10* — [coming soon]

---

## Overview

Knowledge Distillation (KD) transfers the "dark knowledge" of a large teacher model
into a smaller student model. Instead of training on hard labels alone, the student
also learns from the teacher's soft output distribution — which encodes inter-class
similarity information invisible in one-hot labels.

This project ablates two distillation paradigms across multiple hyperparameter
configurations to answer:

- Does Logit-KD or Feature-KD produce better students?
- How do distillation weight α and temperature T affect student accuracy?
- How does teacher quality affect the effectiveness of distillation?

### Loss Formulation

$$L_{\text{total}} = \alpha \cdot L_{\text{kd}} + (1 - \alpha) \cdot L_{\text{ce}}$$

Following Hinton et al. (2015), $\alpha$ controls the balance between the distillation
signal and the ground-truth supervision.

### Logit-KD

$$L_{\text{logit}} = T^2 \cdot D_{\text{KL}}\!\left(\text{softmax}\!\left(\frac{z_T}{T}\right) \,\Big\|\, \text{softmax}\!\left(\frac{z_S}{T}\right)\right)$$

Minimizing this KL divergence is equivalent to MLE under the teacher's distribution —
a direct connection to information-theoretic foundations. Temperature $T$ softens the
teacher's output distribution, exposing inter-class similarity ("dark knowledge").

### Feature-KD

$$L_{\text{feat}} = \frac{1}{|L|}\sum_{i \in L} \text{MSE}(\text{proj}_i(S_i),\, T_i) + \beta \cdot \frac{1}{|L|}\sum_{i \in L} \left(1 - \cos(S_i^{\text{GAP}},\, T_i^{\text{GAP}})\right)$$

Aligns intermediate feature maps at all four ResNet stages (layer1–layer4) using MSE
on projected features and cosine similarity on globally pooled representations.

---

## Models

| Model | Params | Role |
|---|---|---|
| ResNet-50 | 23.5M | Teacher |
| ResNet-18 | 11.2M | Student |

**CIFAR-10 architecture modification:** Standard torchvision ResNets use a 7×7 conv
(stride=2) + MaxPool (stride=2) designed for ImageNet (224×224), which reduces a
32×32 CIFAR image to 8×8 before any meaningful learning. We replace this with a
3×3 conv (stride=1) and remove MaxPool entirely, preserving spatial information.

---

## Ablation Grid

### Experiment 1 — Standard Architecture

| Config | Best Acc | Δ Baseline | Δ Teacher |
|---|---|---|---|
| Teacher (ResNet-50) | 89.81% | — | — |
| Baseline (ResNet-18) | 88.98% | — | -0.83% |
| Logit α=0.3 T=4 | 88.83% | -0.15% | -0.98% |
| Logit α=0.5 T=4 | 88.88% | -0.10% | -0.93% |
| Logit α=0.7 T=4 | 88.68% | -0.30% | -1.13% |
| Logit α=0.5 T=2 | **88.94%** | -0.04% | -0.87% |
| Logit α=0.5 T=8 | 88.72% | -0.26% | -1.09% |
| Feature α=0.3 | 88.18% | -0.80% | -1.63% |
| Feature α=0.5 | 88.62% | -0.36% | -1.19% |
| Feature α=0.7 | 88.62% | -0.36% | -1.19% |

**Finding:** With a small teacher-student gap (0.83%), KD does not consistently
outperform the baseline. Teacher quality is the primary bottleneck.

### Experiment 2 — CIFAR-Specific Architecture

*Results coming soon.*

| Config | Best Acc | Δ Baseline | Δ Teacher |
|---|---|---|---|
| Teacher (ResNet-50) | TBD | — | — |
| Baseline (ResNet-18) | TBD | — | — |
| Logit α=0.3 T=2 | TBD | TBD | TBD |
| Logit α=0.5 T=2 | TBD | TBD | TBD |
| Logit α=0.7 T=2 | TBD | TBD | TBD |
| Logit α=0.5 T=3 | TBD | TBD | TBD |
| Logit α=0.5 T=4 | TBD | TBD | TBD |
| Feature α=0.3 | TBD | TBD | TBD |
| Feature α=0.5 | TBD | TBD | TBD |
| Feature α=0.7 | TBD | TBD | TBD |

---

## Installation

```bash
git clone https://github.com/umutonuryasar/kd-cifar10.git
cd kd-cifar10
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

CIFAR-10 in ImageFolder format under `data/`:

```
data/
├── train/   # 50,000 images across 10 class subfolders
└── test/    # 10,000 images across 10 class subfolders
```

## Usage

```bash
# Train teacher
python tools/train.py \
    --model resnet50 --kd-type none \
    --epochs 100 --output-dir runs/teacher_r50

# Train baseline student
python tools/train.py \
    --model resnet18 --kd-type none \
    --epochs 100 --output-dir runs/baseline

# Logit-KD
python tools/train.py \
    --model resnet18 --kd-type logit \
    --alpha 0.5 --temperature 2 \
    --teacher-weights runs/teacher_r50/checkpoint_best.pth \
    --output-dir runs/logit_a0.5_t2

# Feature-KD
python tools/train.py \
    --model resnet18 --kd-type feature \
    --alpha 0.5 --feat-beta 0.5 \
    --teacher-weights runs/teacher_r50/checkpoint_best.pth \
    --output-dir runs/feature_a0.5

# Full ablation
./tools/run_ablation.sh
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | resnet18 | Student architecture (resnet18 / resnet50) |
| `--kd-type` | logit | Distillation type (logit / feature / none) |
| `--alpha` | 0.5 | Distillation weight α ∈ [0, 1] |
| `--temperature` | 4.0 | Softmax temperature T (logit KD only) |
| `--feat-beta` | 0.5 | Cosine weight β inside Feature-KD |
| `--teacher-weights` | None | Path to teacher checkpoint |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.1 | Learning rate |
| `--output-dir` | runs/experiment | Output directory |
| `--device` | cuda | Device (cuda / cpu) |

---

## Project Structure

```
kd-cifar10/
├── src/
│   ├── models/
│   │   ├── resnet.py          # CIFAR-specific ResNet with feature hooks
│   │   └── kd_model.py        # Teacher-Student wrapper
│   ├── distillation/
│   │   ├── logit_kd.py        # Temperature-scaled KL divergence
│   │   └── feature_kd.py      # Multi-layer MSE + cosine alignment
│   ├── losses/
│   │   └── kd_loss.py         # α·L_kd + (1-α)·L_ce
│   └── trainer.py             # Training loop + TensorBoard
├── tools/
│   ├── train.py               # Entry point
│   └── run_ablation.sh        # Full ablation grid
├── notebooks/
│   └── analysis.ipynb         # Result visualization
├── CLAUDE.md                  # Implementation details for Claude Code
└── README.md
```

---

## References

1. Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NeurIPS Workshop.
2. Romero, A., et al. (2015). *FitNets: Hints for Thin Deep Nets.* ICLR.
3. Zagoruyko, S., Komodakis, N. (2017). *Paying More Attention to Attention.* ICLR.
4. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

---
