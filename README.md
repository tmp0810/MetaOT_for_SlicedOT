# Amortized Optimal Transport from Sliced Potentials

This repository contains the source code, built on
PyTorch and POT,
to reproduce the experiments for our paper
**Amortized Optimal Transport from Sliced Potentials**.

We propose an amortized method to predict optimal transport (OT) plans using Kantorovich potentials from sliced OT. We introduce two strategies: **RA-OT**, which learns a regression from sliced to full OT potentials via least squares, and **OA-OT**, which directly optimizes the Kantorovich dual objective. In both cases, OT plans are recovered from the estimated potentials. By reusing learned information, these methods efficiently solve repeated OT problems, remain agnostic to measure structure, and achieve strong performance across tasks such as MNIST transport, color transfer, spherical transport, and mini-batch OT flow matching.

---

# Setup

After cloning this repository, install the dependencies with:

```bash
pip install -r requirements.txt
```

# Basic Structure of This Repository

- [`cfg.py`](./cfg.py): Hyperparameter configurations for all solvers and experiments
- [`regression_OT_utils.py`](./regression_OT_utils.py): Core math — vectorized 1-D OT (`emd1D`, `emd1D_dual`) and ridge regression
- [`SWGG.py`](./SWGG.py): SWGG utilities (barycenter, CP, and smooth variants)
- [`Models/`](./Models/): Neural network architectures (`PotentialMLP`, `DenseICNN`, `MetaICNN`)
- [`Data/`](./Data/): Data loaders for MNIST, color transfer, and world pair experiments
- [`Utils/`](./Utils/): Shared utilities (`dotdict`, image saving helpers)
- [`Solvers/`](./Solvers/): All solver implementations
  - [`Solvers/Regression_SlicedOT/`](./Solvers/Regression_SlicedOT/): **RA-OT** — regression-amortized OT (grayscale, world, color)
  - [`Solvers/Objective_SlicedOT/`](./Solvers/Objective_SlicedOT/): **OA-OT** — objective-amortized OT (grayscale, world, color)
  - [`Solvers/Meta_OT/`](./Solvers/Meta_OT/): Meta-OT baseline (grayscale, world, color)
  - [`Solvers/MinSTP/`](./Solvers/MinSTP/): Min-STP baseline (grayscale, world, color)
  - [`Solvers/SWGG/`](./Solvers/SWGG/): min-SWGG baseline (grayscale, world, color)
  - [`Solvers/FlowMatching/`](./Solvers/FlowMatching/): Toy 2-D flow matching experiment
- [`Eval_report/`](./Eval_report/): Unified evaluation and plotting scripts
  - [`eval_grayscale.py`](./Eval_report/eval_grayscale.py): Train & evaluate all methods on MNIST
  - [`eval_worldpair.py`](./Eval_report/eval_worldpair.py): Train & evaluate all methods on spherical OT
  - [`eval_color.py`](./Eval_report/eval_color.py): Train & evaluate all methods on color transfer
  - [`plot_results_grayscale.py`](./Eval_report/plot_results_grayscale.py): Plot MNIST transport interpolations
  - [`plot_results_worldpair.py`](./Eval_report/plot_results_worldpair.py): Plot world pair transport
  - [`plot_results_color.py`](./Eval_report/plot_results_color.py): Plot color transfer results
- [`CIFAR10_OT_CFM/`](./CIFAR10_OT_CFM/): CIFAR-10 flow matching experiment
  - [`RA_OT_CIFAR.py`](./CIFAR10_OT_CFM/RA_OT_CIFAR.py): RA-OT amortized solver for CIFAR-10
  - [`OA_OT_CIFAR.py`](./CIFAR10_OT_CFM/OA_OT_CIFAR.py): OA-OT amortized solver for CIFAR-10
  - [`finetune_cifar10.py`](./CIFAR10_OT_CFM/finetune_cifar10.py): Two-phase fine-tuning script
  - [`compute_fid.py`](./CIFAR10_OT_CFM/compute_fid.py): FID evaluation script

---

# Reproducing Our Experimental Results

All experiments below use **M** training pairs and **N** test pairs.
The example commands use `M=50` and `N=300`.

## 1. Grayscale Experiment (MNIST)

This code will automatically download the MNIST dataset for training and evaluation.

**Train all methods and evaluate:**
```bash
python Eval_report/eval_grayscale.py \
    --M 50 \
    --N 300 \
    --gpu 0 \
    --out ./results/grayscale
```

**Plot transport interpolations:**
```bash
python Eval_report/plot_results_grayscale.py \
    --result_dir ./results/grayscale/M50 \
    --idx all \
    --num_interp 16 --num_iter 20 --batch_size 50000
```

## 2. World Pair Experiment (Spherical OT)

First download the
2020 population density TIFF at 30-second resolution (GPWv4) 
and save the file to `./data/Global_2015_PopulationDensity30sec_GPWv4.tiff`.

**Train all methods and evaluate:**
```bash
python Eval_report/eval_worldpair.py \
    --pop_tiff ./data/Global_2015_PopulationDensity30sec_GPWv4.tiff \
    --M 50 \
    --N 300 \
    --n_supply 100 \
    --n_demand 10000 \
    --gpu 0 \
    --out ./results/worldpair
```

**Plot spherical transport:**
```bash
python Eval_report/plot_results_worldpair.py \
    --result_dir ./results/worldpair/M50 \
    --pop_tiff   ./data/Global_2015_PopulationDensity30sec_GPWv4.tiff \
    --idx        all
```

## 3. Color Transfer Experiment

First download the WikiArt painting images into `./data/paintings` by running:

```bash
python data_color_transfer/download-wikiart.py
```

**Train all methods and evaluate:**
```bash
python Eval_report/eval_color.py \
    --data_dir ./data/paintings \
    --M 50 \
    --N 300 \
    --n_clusters 500 \
    --gpu 0 \
    --out ./results/color
```

**Plot color transfer results:**
```bash
python Eval_report/plot_results_color.py \
    --result_dir ./results/color/M50 \
    --idx        all
```

## 4. Flow Matching — Toy Data (2-D)

**Train all methods and evaluate** (Gaussian → 8Gaussians / Moons / S-Curve):
```bash
python Solvers/FlowMatching/eval_flow.py \
    --device cuda \
    --n_steps 1000 \
    --batch_size 512 \
    --L 100 \
    --eps 0.1 \
    --M_pretrain 50 \
    --T_pretrain 5000
```

**Plot trajectories:**
```bash
python Solvers/FlowMatching/plot_flow.py \
    --result_dir ./results_flow
```

## 5. Flow Matching — CIFAR-10

All commands below run on a single A100 GPU. The experiment fine-tunes a
pretrained I-CFM checkpoint for **10 epochs** using each coupling method.
The amortized methods (RA-OT, OA-OT) follow a **two-phase** protocol:
a short OT pre-training phase followed by U-Net fine-tuning.

### Step 1 — Download the pretrained I-CFM checkpoint

```bash
wget https://github.com/atong01/conditional-flow-matching/releases/download/1.0.4/cfm_cifar10_weights_step_400000.pt
```

### Step 2 — Fine-tune with each coupling method

**I-CFM** (Independent Conditional Flow Matching — baseline, no OT):
```bash
python3 CIFAR10_OT_CFM/finetune_cifar10.py \
    --model "icfm" \
    --checkpoint ./cfm_cifar10_weights_step_400000.pt \
    --batch_size 2048 --cpu_ot True \
    --lr 5e-5 --ema_decay 0.999 --grad_clip 0.5 \
    --accum_steps 8 --finetune_epochs 10 --warmup 100
```

**OT-CFM** (Exact LP OT Conditional Flow Matching — baseline):
```bash
python3 CIFAR10_OT_CFM/finetune_cifar10.py \
    --model "otcfm" \
    --checkpoint ./cfm_cifar10_weights_step_400000.pt \
    --batch_size 2048 --cpu_ot True \
    --lr 5e-5 --ema_decay 0.999 --grad_clip 0.5 \
    --accum_steps 8 --finetune_epochs 10 --warmup 100
```

**RA-OT** (Regression-Amortized OT):
```bash
python3 CIFAR10_OT_CFM/finetune_cifar10.py \
    --model "ra-ot" \
    --checkpoint ./cfm_cifar10_weights_step_400000.pt \
    --batch_size 2048 --cpu_ot True \
    --lr 5e-5 --ema_decay 0.999 --grad_clip 0.5 \
    --finetune_epochs 10 --warmup 100 \
    --accum_steps 8 --pretrain_M 50 --pretrain_L 100 \
    --pretrain_eps 750 --pretrain_ridge 1e-3
```

**OA-OT** (Objective-Amortized OT):
```bash
python3 CIFAR10_OT_CFM/finetune_cifar10.py \
    --model "oa-ot" \
    --checkpoint ./cfm_cifar10_weights_step_400000.pt \
    --batch_size 2048 --cpu_ot True \
    --lr 5e-5 --ema_decay 0.999 --grad_clip 0.5 \
    --finetune_epochs 10 --warmup 100 \
    --accum_steps 8 --pretrain_M 50 --pretrain_L 100 \
    --pretrain_eps 750 --pretrain_T 5000 --pretrain_lr 1e-2
```

### Step 3 — Compute FID scores

Replace `$step` with the final step number printed at the end of fine-tuning.

```bash
python3 CIFAR10_OT_CFM/compute_fid.py --model "icfm"  --step $step --integration_method dopri5
python3 CIFAR10_OT_CFM/compute_fid.py --model "otcfm" --step $step --integration_method dopri5
python3 CIFAR10_OT_CFM/compute_fid.py --model "ra-ot" --step $step --integration_method dopri5
python3 CIFAR10_OT_CFM/compute_fid.py --model "oa-ot" --step $step --integration_method dopri5
```


