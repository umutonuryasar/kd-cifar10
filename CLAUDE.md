# CLAUDE.md — KD-CIFAR10 Knowledge Distillation Ablation Study

## Project Identity

**Goal:** Systematically compare Logit-KD and Feature-KD strategies for compressing
ResNet-50 (teacher) into ResNet-18 (student) on CIFAR-10.

**Research question:** How do distillation weight α and softmax temperature T affect
student accuracy, and does feature-level distillation outperform logit-level distillation?

**Output:** 5–8 page NeurIPS-format paper + clean GitHub repo, extendable to arXiv.

---

## Loss Formulation

```
L_total = α · L_kd + (1 - α) · L_ce
```

- `L_ce`: cross-entropy against ground truth labels
- `L_kd`: LogitKDLoss or FeatureKDLoss
- `α ∈ [0, 1]`: distillation weight (higher → more teacher influence)

**This formulation must never be changed back to `L_ce + λ·L_kd`.**

### Logit-KD (Hinton et al., 2015)

```
L_logit = T² · KL( softmax(t_logits/T) ‖ softmax(s_logits/T) )
```

Minimizing this KL divergence is equivalent to MLE under the teacher's distribution
(connection to CS229 PS3 Q2c). The T² factor restores gradient magnitude.
Temperature T controls soft target sharpness: T=1 is standard softmax, T>1 exposes
inter-class similarity ("dark knowledge").

### Feature-KD

```
L_feat = mean_i[ MSE(proj_i(s_i), t_i) ] + β · mean_i[ 1 - cos_sim(GAP(s_i), GAP(t_i)) ]
```

- `proj_i`: 1×1 Conv2d projecting student channels → teacher channels at layer i
- `GAP`: global average pool before cosine similarity
- `β` (feat_beta): cosine similarity weight, default 0.5
- Layers: layer1, layer2, layer3, layer4 (all four ResNet stages)

---

## Project Structure

```
kd-cifar10/
├── src/
│   ├── models/
│   │   ├── resnet.py           # CIFAR-specific ResNet-18/50 with feature hooks
│   │   └── kd_model.py         # Teacher-Student wrapper (teacher frozen)
│   ├── distillation/
│   │   ├── logit_kd.py         # Temperature-scaled KL divergence
│   │   └── feature_kd.py       # Multi-layer MSE + cosine feature alignment
│   ├── losses/
│   │   └── kd_loss.py          # Combined: α·L_kd + (1-α)·L_ce
│   └── trainer.py              # Training loop + TensorBoard logging
├── tools/
│   ├── train.py                # Main entry point
│   └── run_ablation.sh         # Full ablation grid
├── notebooks/
│   └── analysis.ipynb          # Result visualization and tables
├── data/                       # CIFAR-10 ImageFolder format (gitignored)
│   ├── train/                  # 50,000 images, 10 class subfolders
│   └── test/                   # 10,000 images, 10 class subfolders
└── runs/                       # Training outputs (gitignored)
    ├── teacher_r50/            # ResNet-50 teacher
    ├── baseline/               # ResNet-18 no-KD baseline
    ├── logit_a{α}_t{T}/        # Logit-KD runs
    └── feature_a{α}/           # Feature-KD runs
```

---

## Critical Architecture Decision: CIFAR-Specific ResNet

**This is the most important change in Experiment 2.**

Standard torchvision ResNet was designed for ImageNet (224×224). Its first layers
aggressively downsample: `7×7 conv (stride=2) → MaxPool (stride=2)` reduces
32×32 input to 8×8 after just two operations — a severe information loss.

### Required modification in `src/models/resnet.py`:

```python
# After building the torchvision base model, replace the first two layers:
base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
base.maxpool = nn.Identity()  # remove maxpool entirely
```

Apply this to **both ResNet-18 and ResNet-50**. Expected accuracy improvement:
- Teacher ResNet-50: 89.81% → ~94-95%
- Baseline ResNet-18: 88.98% → ~92-93%
- Teacher-student gap increases significantly → KD becomes meaningful

### Full `src/models/resnet.py` implementation:

```python
"""CIFAR-10 ResNet with forward hooks for Knowledge Distillation.

Critical modification: replaces the ImageNet-oriented 7×7 conv (stride=2)
and MaxPool with a 3×3 conv (stride=1) and Identity. This prevents aggressive
spatial downsampling of 32×32 CIFAR images.

Feature hooks on layer1–layer4 expose intermediate activations for Feature-KD.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet(nn.Module):
    def __init__(
        self,
        variant: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = False,
    ):
        super().__init__()

        if variant == "resnet18":
            base = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif variant == "resnet50":
            base = tv_models.resnet50(
                weights=tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        # CIFAR-10 modification: prevent aggressive downsampling
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        # Replace final FC for CIFAR-10
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)

        self.model = base
        self.features: dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self.model, name)
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet(variant: str, num_classes: int = 10, pretrained: bool = False) -> ResNet:
    return ResNet(variant=variant, num_classes=num_classes, pretrained=pretrained)
```

---

## Experiment History

### Experiment 1 — Baseline Setup (May 3, 2026)

**Configuration:**
- Standard torchvision ResNet (no CIFAR modification)
- Teacher: ResNet-50, 100 epochs, SGD lr=0.1, CosineAnnealingLR
- Student: ResNet-18, 100 epochs, same schedule
- Dataset: CIFAR-10 ImageFolder format (50k train / 10k test)
- Loss: α·L_kd + (1-α)·L_ce

**Results:**

| Config           | Best Acc | Δ Teacher |
|------------------|----------|-----------|
| Teacher (R50)    | 89.81%   | —         |
| Baseline (R18)   | 88.98%   | -0.83%    |
| Logit α=0.3 T=4  | 88.83%   | -0.98%    |
| Logit α=0.5 T=4  | 88.88%   | -0.93%    |
| Logit α=0.7 T=4  | 88.68%   | -1.13%    |
| Logit α=0.5 T=2  | **88.94%** | **-0.87%** |
| Logit α=0.5 T=8  | 88.72%   | -1.09%    |
| Feature α=0.3    | 88.18%   | -1.63%    |
| Feature α=0.5    | 88.62%   | -1.19%    |
| Feature α=0.7    | 88.62%   | -1.19%    |

**Key findings:**
1. **KD did not outperform baseline** — all KD configs below teacher and baseline.
   Root cause: teacher-student gap too small (only 0.83%) for meaningful distillation.
2. **Logit-KD > Feature-KD** consistently across all α values.
3. **T=2 best for Logit-KD** — lower temperature preserves sharper distributions,
   more effective for CIFAR-10's 10 well-separated classes. T=4 (from literature)
   is better suited for ImageNet's 1000 classes.
4. **α=0.5 optimal** — higher α (more KD weight) hurts performance with weak teacher.
5. **Feature-KD underperforms** — likely because 7×7→MaxPool destroys spatial info
   that feature alignment depends on.

**Root cause analysis:**
The standard torchvision ResNet architecture is not suited for 32×32 inputs.
The 7×7 conv (stride=2) + MaxPool (stride=2) reduces feature maps to 8×8 before
any learning begins. This limits both teacher quality (89.81% vs expected 94-95%)
and makes Feature-KD ineffective (no useful spatial features to align).

---

### Experiment 2 — CIFAR-Specific Architecture (TODO)

**Hypothesis:** CIFAR-specific ResNet (3×3 conv, no maxpool) will:
1. Raise teacher accuracy to ~94-95%
2. Raise baseline student to ~92-93%
3. Increase teacher-student gap → KD becomes meaningful
4. Validate whether Logit-KD > Feature-KD holds with proper architecture

**Ablation grid (informed by Experiment 1):**

Logit-KD α sweep (T=2, fixed — best from Exp 1):

| Run | α   | T | Output dir          |
|-----|-----|---|---------------------|
| 1   | 0.3 | 2 | runs/logit_a0.3_t2  |
| 2   | 0.5 | 2 | runs/logit_a0.5_t2  |
| 3   | 0.7 | 2 | runs/logit_a0.7_t2  |

Logit-KD T sweep (α=0.5, fixed — best from Exp 1):

| Run | α   | T | Output dir          |
|-----|-----|---|---------------------|
| 4   | 0.5 | 2 | runs/logit_a0.5_t2  |
| 5   | 0.5 | 3 | runs/logit_a0.5_t3  |
| 6   | 0.5 | 4 | runs/logit_a0.5_t4  |

Feature-KD α sweep:

| Run | α   | β   | Output dir         |
|-----|-----|-----|--------------------|
| 7   | 0.3 | 0.5 | runs/feature_a0.3  |
| 8   | 0.5 | 0.5 | runs/feature_a0.5  |
| 9   | 0.7 | 0.5 | runs/feature_a0.7  |

Note: Run 2 and Run 4 are identical — train once, reuse.

---

## Usage

### Setup

```bash
git clone https://github.com/umutonuryasar/kd-cifar10.git
cd kd-cifar10
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

CIFAR-10 must be in ImageFolder format under `data/`:

```
data/
├── train/
│   ├── airplane/   (5000 images)
│   ├── automobile/
│   └── ...
└── test/
    ├── airplane/   (1000 images)
    └── ...
```

If using the zip from Drive:
```bash
unzip cifar10-imagefolder.zip
mv cifar10-imagefolder data
```

### Smoke Test (run after any architecture change)

```bash
python -c "
import torch
from src.models.resnet import build_resnet
from src.models.kd_model import KDModel
from src.losses.kd_loss import KDLoss

# Verify CIFAR modification
s = build_resnet('resnet18')
t = build_resnet('resnet50')
print(f'Student conv1: {s.model.conv1}')   # must be kernel_size=3, stride=1
print(f'Teacher maxpool: {t.model.maxpool}') # must be Identity

# Verify loss formulation
loss_fn = KDLoss(kd_type='logit', alpha=0.5, temperature=2.0)
sl = torch.randn(4, 10)
tl = torch.randn(4, 10)
lb = torch.randint(0, 10, (4,))
out = loss_fn(sl, tl, lb)
expected = 0.5 * out['loss_kd'].item() + 0.5 * out['loss_ce'].item()
assert abs(out['loss_total'].item() - expected) < 1e-5, 'Loss formulation wrong!'
print(f'Logit-KD OK: {out}')

# Verify feature hooks include layer1
m = KDModel(s, t)
x = torch.randn(2, 3, 32, 32)
_ = m(x)
assert 'layer1' in s.features, 'layer1 hook missing!'
assert 'layer4' in s.features, 'layer4 hook missing!'
print(f'Feature hooks OK: {list(s.features.keys())}')

# Verify feature spatial sizes (with CIFAR modification)
for name, feat in s.features.items():
    print(f'  Student {name}: {tuple(feat.shape)}')

print('All checks passed.')
"
```

### Train Teacher (Experiment 2)

```bash
python tools/train.py \
    --model resnet50 \
    --kd-type none \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --output-dir runs/teacher_r50_v2 \
    --device cuda
```

Expected: ~94-95% val accuracy.

### Train Baseline Student

```bash
python tools/train.py \
    --model resnet18 \
    --kd-type none \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --output-dir runs/baseline_v2 \
    --device cuda
```

Expected: ~92-93% val accuracy.

### Run Full Ablation

```bash
./tools/run_ablation.sh
```

Update `run_ablation.sh` to use `runs/teacher_r50_v2/checkpoint_best.pth` and the
Experiment 2 grid before running.

### Single KD Run

```bash
# Logit-KD (best config from Exp 1: α=0.5, T=2)
python tools/train.py \
    --model resnet18 \
    --kd-type logit \
    --alpha 0.5 \
    --temperature 2 \
    --teacher-weights runs/teacher_r50_v2/checkpoint_best.pth \
    --output-dir runs/logit_a0.5_t2 \
    --device cuda

# Feature-KD
python tools/train.py \
    --model resnet18 \
    --kd-type feature \
    --alpha 0.5 \
    --feat-beta 0.5 \
    --teacher-weights runs/teacher_r50_v2/checkpoint_best.pth \
    --output-dir runs/feature_a0.5 \
    --device cuda
```

---

## Claude Code Instructions

### Conventions

- All source code in `src/`, entry points in `tools/`
- Never modify `runs/` — training outputs only
- Loss formulation is `α·L_kd + (1-α)·L_ce` — never revert to `L_ce + λ·L_kd`
- `alpha`: distillation weight [0,1]
- `feat_beta`: cosine similarity weight inside FeatureKDLoss
- `temperature`: softmax temperature for LogitKDLoss only
- Teacher is always ResNet-50, student is always ResNet-18

### Critical implementation constraints

1. **CIFAR ResNet modification is mandatory:**
   ```python
   base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
   base.maxpool = nn.Identity()
   ```
   This must appear in `ResNet.__init__` for both resnet18 and resnet50.

2. **Feature hooks must include layer1:**
   ```python
   for name in ("layer1", "layer2", "layer3", "layer4"):
   ```
   Missing layer1 causes `KeyError: 'layer1'` in FeatureKDLoss.

3. **Projection layers in FeatureKDLoss use explicit dtype cast:**
   ```python
   proj_weight = self.projections[layer].weight.to(device=device, dtype=dtype)
   s_proj = F.conv2d(s, proj_weight)
   ```
   Never call `self.projections[layer](s)` directly — causes dtype mismatch under AMP.

4. **Teacher stays frozen and in eval mode:**
   ```python
   # In KDModel.train():
   super().train(mode)
   self.teacher.eval()  # always, regardless of mode
   ```

### When modifying the architecture

Always run the smoke test above after any change to `resnet.py` or `kd_model.py`.
Verify:
- `conv1` is 3×3, stride=1
- `maxpool` is Identity
- Feature hooks capture layer1–layer4
- Feature spatial sizes are reasonable (not 1×1 or 2×2)

### When modifying the loss

Verify the alpha formulation:
```python
assert abs(out['loss_total'] - (alpha * out['loss_kd'] + (1-alpha) * out['loss_ce'])) < 1e-5
```

### When updating ablation grid

Update both `tools/run_ablation.sh` AND the ablation table in this CLAUDE.md.
Output directory naming convention:
- Logit-KD: `runs/logit_a{alpha}_t{temperature}` (e.g., `runs/logit_a0.5_t2`)
- Feature-KD: `runs/feature_a{alpha}` (e.g., `runs/feature_a0.5`)
- Teacher: `runs/teacher_r50_v{version}` (e.g., `runs/teacher_r50_v2`)
- Baseline: `runs/baseline_v{version}`

### Git workflow

```bash
# After each working change:
git add src/ tools/ CLAUDE.md
git commit -m "type: short description"
git push origin main
```

Commit types: `feat`, `fix`, `refactor`, `exp` (experiment results)

Never commit: `runs/`, `data/`, `.venv/`, `__pycache__/`

### Key hyperparameters (do not change without reason)

| Parameter      | Value  | Reason                                    |
|----------------|--------|-------------------------------------------|
| Optimizer      | SGD    | Adam generalizes worse on CIFAR           |
| Momentum       | 0.9    | Standard                                  |
| Weight decay   | 5e-4   | Critical for preventing overfitting       |
| Nesterov       | True   | Faster convergence                        |
| LR             | 0.1    | Standard for SGD+CIFAR                    |
| LR schedule    | CosineAnnealingLR (T_max=epochs, eta_min=1e-4) | Smooth decay |
| Batch size     | 128    | Standard                                  |
| Epochs         | 100    | Sufficient for convergence                |
| Clip grad norm | 1.0    | Prevents gradient explosion               |

---

## Paper Outline (CS229 / arXiv)

**Title:** Knowledge Distillation for Image Classification: A Systematic Ablation of
Logit-KD and Feature-KD on CIFAR-10

**Sections:**
1. Introduction — motivation, KL=MLE connection, contributions
2. Related Work — Hinton et al., FitNets, AT, CRD
3. Method — loss formulation, logit-KD derivation, feature-KD design
4. Experiments — setup (two experiments), ablation grid
5. Results — tables, convergence curves, temperature effect plot
6. Discussion — teacher quality effect, T recommendation, Logit vs Feature
7. Conclusion — findings + future work (Bayesian uncertainty, RT-DETR extension)

**Key narrative:** Experiment 1 reveals that teacher quality is the primary bottleneck
for KD effectiveness. Experiment 2 (CIFAR-specific architecture) validates the
distillation methods under proper conditions and reveals the true performance ceiling.

---

## Environment

- Python 3.12
- PyTorch 2.x + CUDA
- torchvision
- tensorboard, matplotlib, seaborn, pandas
- Training hardware: RTX 3050 Laptop (local) or A100 (Colab Pro+)
- OS: Ubuntu 24.04

## File: `.gitignore`

```
.venv/
runs/
data/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
```
