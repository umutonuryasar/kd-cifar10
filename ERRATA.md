# Errata

This repository contains the original CS229 project implementation of knowledge distillation on CIFAR-10. Two bugs were identified after submission:

## Bug 1: Incomplete Determinism

`tools/train.py` was missing `random.seed(seed)` and `np.random.seed(seed)` calls. Only PyTorch and CUDA seeds were set, leaving Python's built-in RNG and NumPy's RNG unseeded. This may have introduced variance across runs and contaminated seed-based reproducibility estimates.

**Status:** Fixed in this repository (commit: fix determinism: add random and numpy seeds).

## Bug 2: Feature-KD Gradient Clipping (Original CS229 Version)

In the original CS229 submission, Feature-KD projection layer parameters were excluded from `clip_grad_norm_`. This suppressed Feature-KD performance and produced misleading comparisons with Logit-KD.

**Status:** This bug was already corrected in the current codebase prior to this errata. It is documented here for transparency.

## Corrected Results

A fully corrected reimplementation with proper determinism, architecture fixes for 32×32 inputs, and systematic ablation across three teacher-student pairs is available as an arXiv preprint:

> Yaşar, U. O. (2026). *Student Capacity Moderates Knowledge Distillation Effectiveness: A Systematic Study Across ResNet Teacher-Student Pairs on CIFAR-10*. arXiv:2605.31191. https://arxiv.org/abs/2605.31191
