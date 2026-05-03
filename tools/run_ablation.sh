#!/bin/bash
# KD-CIFAR10 Ablation Study
# Loss: L_total = α · L_kd + (1 - α) · L_ce  (Hinton et al., 2015)
#
# Grid:
#   Logit-KD: α ∈ {0.3, 0.5, 0.7} × T ∈ {2, 4, 8}  → 9 runs (T fixed at 4 for α sweep)
#   Feature-KD: α ∈ {0.3, 0.5, 0.7}                 → 3 runs
#   Total: 8 runs (α sweep + T sweep combined)

TEACHER="runs/teacher_r50/checkpoint_best.pth"
EPOCHS=100
BS=128
LR=0.1
DEVICE="cuda"

echo "============================================"
echo "KD-CIFAR10 Ablation Study"
echo "Loss: L_total = α·L_kd + (1-α)·L_ce"
echo "Teacher: $TEACHER"
echo "============================================"

# ── Logit-KD: alpha sweep (T=4 fixed) ────────────────────────────────────
echo "[1/8] Logit-KD α=0.3 T=4"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.3 --temperature 4 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.3_t4 --device $DEVICE

echo "[2/8] Logit-KD α=0.5 T=4"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.5 --temperature 4 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.5_t4 --device $DEVICE

echo "[3/8] Logit-KD α=0.7 T=4"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.7 --temperature 4 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.7_t4 --device $DEVICE

# ── Logit-KD: temperature sweep (α=0.5 fixed) ────────────────────────────
echo "[4/8] Logit-KD α=0.5 T=2"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.5 --temperature 2 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.5_t2 --device $DEVICE

echo "[5/8] Logit-KD α=0.5 T=8"
python tools/train.py --model resnet18 \
    --kd-type logit --alpha 0.5 --temperature 8 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/logit_a0.5_t8 --device $DEVICE

# ── Feature-KD: alpha sweep ───────────────────────────────────────────────
echo "[6/8] Feature-KD α=0.3"
python tools/train.py --model resnet18 \
    --kd-type feature --alpha 0.3 --feat-beta 0.5 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/feature_a0.3 --device $DEVICE

echo "[7/8] Feature-KD α=0.5"
python tools/train.py --model resnet18 \
    --kd-type feature --alpha 0.5 --feat-beta 0.5 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/feature_a0.5 --device $DEVICE

echo "[8/8] Feature-KD α=0.7"
python tools/train.py --model resnet18 \
    --kd-type feature --alpha 0.7 --feat-beta 0.5 \
    --epochs $EPOCHS --batch-size $BS --lr $LR \
    --teacher-weights $TEACHER \
    --output-dir runs/feature_a0.7 --device $DEVICE

echo "============================================"
echo "Ablation complete!"
echo "============================================"
