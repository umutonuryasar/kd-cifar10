# KD-CIFAR10 — Knowledge Distillation Ablation Study

## Project Overview

This project systematically investigates Knowledge Distillation (KD) strategies
for compressing ResNet-50 (teacher) into ResNet-18 (student) on CIFAR-10.

Two distillation paradigms are compared across multiple hyperparameter configurations:
- **Logit-KD**: KL divergence on soft class probabilities (Hinton et al., 2015)
- **Feature-KD**: MSE + cosine similarity on intermediate feature maps (4 ResNet stages)

### Loss Formulation
L_total = α · L_kd + (1 - α) · L_ce

- `L_ce`: standard cross-entropy against ground truth labels
- `L_kd`: distillation loss (LogitKD or FeatureKD)
- `α ∈ [0, 1]`: distillation weight — higher α → more teacher influence

#### Logit-KD (Hinton et al., 2015)

Minimizes KL divergence between temperature-scaled student and teacher distributions:
L_logit = T² · KL( softmax(t_logits/T) || softmax(s_logits/T) )

The T² factor restores gradient magnitude. Minimizing this KL divergence
is equivalent to MLE under the teacher's distribution (CS229 PS3 Q2c connection).

Temperature T controls soft target sharpness:
- T=1: standard softmax (sharp)
- T>1: softer distribution, exposes inter-class similarity ("dark knowledge")

#### Feature-KD

Aligns intermediate representations at all four ResNet stages (layer1–layer4):
L_feat = mean(MSE(proj(s_i), t_i)) + β · mean(1 - cos_sim(s_i_gap, t_i_gap))

- 1×1 Conv projections align student→teacher channel dimensions per layer
- Global average pooling before cosine similarity
- β (feat_beta): cosine similarity weight, default 0.5

---

## Project Structure
kd-cifar10/
├── src/
│   ├── models/
│   │   ├── resnet.py        # ResNet-18/50 with forward hooks for feature extraction
│   │   └── kd_model.py      # Teacher-Student wrapper (teacher frozen)
│   ├── distillation/
│   │   ├── logit_kd.py      # Temperature-scaled KL divergence
│   │   └── feature_kd.py    # Multi-layer MSE + cosine feature alignment
│   ├── losses/
│   │   └── kd_loss.py       # Combined loss: α·L_kd + (1-α)·L_ce
│   └── trainer.py           # Training loop with TensorBoard logging
├── tools/
│   ├── train.py             # Main entry point
│   └── run_ablation.sh      # Full ablation grid
├── notebooks/
│   └── analysis.ipynb       # Result visualization and tables
├── configs/                 # Reserved for future YAML configs
├── data/                    # CIFAR-10 (ImageFolder format)
│   ├── train/               # 50,000 images, 10 class subfolders
│   └── test/                # 10,000 images, 10 class subfolders
└── runs/                    # Training outputs (gitignored)
├── teacher_r50/         # ResNet-50 teacher checkpoint
├── baseline/            # ResNet-18 no-KD baseline
├── logit_a{α}_t{T}/     # Logit-KD runs
└── feature_a{α}/        # Feature-KD runs

---

## Models

| Model | Params | Role | Val Acc |
|---|---|---|---|
| ResNet-50 | 23.5M | Teacher | 89.61% |
| ResNet-18 | 11.2M | Student (baseline) | 88.98% |

Both models use torchvision backbones with a CIFAR-10 head (fc: → 10 classes).
Forward hooks on layer1–layer4 expose intermediate features for Feature-KD.

---

## Ablation Grid

### Logit-KD — α sweep (T=4 fixed)
| Run | α | T | Output dir |
|---|---|---|---|
| 1 | 0.3 | 4 | runs/logit_a0.3_t4 |
| 2 | 0.5 | 4 | runs/logit_a0.5_t4 |
| 3 | 0.7 | 4 | runs/logit_a0.7_t4 |

### Logit-KD — T sweep (α=0.5 fixed)
| Run | α | T | Output dir |
|---|---|---|---|
| 4 | 0.5 | 2 | runs/logit_a0.5_t2 |
| 5 | 0.5 | 8 | runs/logit_a0.5_t8 |

### Feature-KD — α sweep
| Run | α | β | Output dir |
|---|---|---|---|
| 6 | 0.3 | 0.5 | runs/feature_a0.3 |
| 7 | 0.5 | 0.5 | runs/feature_a0.5 |
| 8 | 0.7 | 0.5 | runs/feature_a0.7 |

---

## Usage

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train teacher
```bash
python tools/train.py \
    --model resnet50 \
    --kd-type none \
    --epochs 100 \
    --output-dir runs/teacher_r50
```

### Run full ablation
```bash
./tools/run_ablation.sh
```

### Single KD run
```bash
# Logit-KD
python tools/train.py \
    --model resnet18 \
    --kd-type logit \
    --alpha 0.5 \
    --temperature 4 \
    --teacher-weights runs/teacher_r50/checkpoint_best.pth \
    --output-dir runs/logit_a0.5_t4

# Feature-KD
python tools/train.py \
    --model resnet18 \
    --kd-type feature \
    --alpha 0.5 \
    --feat-beta 0.5 \
    --teacher-weights runs/teacher_r50/checkpoint_best.pth \
    --output-dir runs/feature_a0.5
```

### Analyze results
```bash
pip install jupyter matplotlib seaborn pandas tensorboard
jupyter notebook notebooks/analysis.ipynb
```

---

## Claude Code Instructions

### Conventions
- All source code lives under `src/`, entry points under `tools/`
- Never modify `runs/` — training outputs only
- Loss formulation is `α·L_kd + (1-α)·L_ce` — do not revert to `L_ce + λ·L_kd`
- `alpha` is the distillation weight (0-1), `feat_beta` is the cosine weight inside FeatureKD
- Teacher is always ResNet-50, student is always ResNet-18

### Key design decisions
- Feature hooks registered in `ResNet.__init__` via `register_forward_hook`
- Teacher frozen via `_freeze_teacher()` in `KDModel`, always kept in eval mode
- Projection layers in `FeatureKDLoss` use `F.conv2d` with explicit `.to(device, dtype)` cast — do not use `self.proj(x)` directly (AMP compatibility)
- `run_ablation.sh` runs baseline separately; do not include baseline in KD runs

### When modifying loss formulation
- Always update both `kd_loss.py` AND `CLAUDE.md` ablation table
- Rerun smoke test before full ablation:
```bash
python -c "
from src.losses.kd_loss import KDLoss
import torch
loss_fn = KDLoss(kd_type='logit', alpha=0.5, temperature=4)
sl = torch.randn(4, 10)
tl = torch.randn(4, 10)
lb = torch.randint(0, 10, (4,))
out = loss_fn(sl, tl, lb)
assert 0 < out['loss_total'].item()
print('OK:', {k: round(v.item(), 4) for k, v in out.items()})
"
```

### Git workflow
- Commit after each working change, not after each file
- Branch naming: `feature/`, `fix/`, `exp/` prefixes
- `runs/` is gitignored — checkpoints stay local