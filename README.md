# SC-MambaFew Input Representation Analysis for Bearing Fault Diagnosis

This repository is for the paper-style experiment **"Optimal Input Representation Analysis for Few-shot Learning in Mamba-based Industrial Equipment Fault Diagnosis Model"**.

It is **not** the official SC-MambaFew repository. It builds on the SC-MambaFew backbone and compares how different signal-to-image input representations affect few-shot bearing fault diagnosis performance.

## Main differences from official SC-MambaFew

| Item | Official SC-MambaFew | This repository |
|---|---|---|
| Purpose | Proposes SC-MambaFew architecture | Analyzes input representations using SC-MambaFew |
| Input | Spectrogram mainly | SP, SC, GS, RP, MTF, GAF |
| Dataset focus | HUST + CWRU | CWRU |
| Resolutions | Mostly 64×64 | 64×64 and 128×128 |
| Noise analysis | Not the central focus | Gaussian noise 10 dB evaluation |
| Training setup | Original SC-MambaFew setup | 1-shot/5-shot, 10 epochs, CWRU 10-way |

## Experiment setup

| Item | Value |
|---|---|
| Dataset | CWRU Bearing Dataset |
| Data subset | 12DriveEndFault |
| RPM | 1772, 1750, 1730 |
| Classes | 10 classes: Normal + Ball/Inner/OuterRace6 faults |
| Fault sizes | 0.007, 0.014, 0.021 inch |
| Window size | 2048 time points |
| Sensor channels | DE + FE |
| Flattened signal length | 4096 = 2048 × 2 |
| Shots | 1-shot, 5-shot |
| Image transforms | SP, SC, GS, RP, MTF, GAF |
| Resolutions | 64×64, 128×128 |
| Optimizer | Adam |
| LR | 0.001 |
| Epochs | 10 |

## Installation

```bash
conda create -n mamba_fault python=3.10 -y
conda activate mamba_fault
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version. The Mamba module requires `mamba-ssm` and `causal-conv1d`; Linux/WSL or a CUDA-enabled Linux server is recommended.

## Training

```bash
# 1-shot, Spectrogram, 64×64, 10 epochs
bash scripts/train.sh 1 SP 64 10

# 5-shot, Scalogram, 128×128, 10 epochs
bash scripts/train.sh 5 SC 128 10
```

By default, CWRU data is stored under `./CWRU`. To use another path:

```bash
DATA_PATH="/path/to/CWRU" bash scripts/train.sh 1 SP 64 10
```

## Testing

```bash
# Clean test
bash scripts/test.sh 1 SP 64 checkpoints/cwru_1shot_64_SP/best.pth

# Gaussian noise 10 dB test
bash scripts/test.sh 1 SP 64 checkpoints/cwru_1shot_64_SP/best.pth 10
```

## Run all clean experiments

```bash
bash scripts/run_all_clean.sh
```

## Evaluate top transforms under Gaussian noise 10 dB

```bash
bash scripts/run_all_noise10_eval.sh
```

## Visualize input representations

```bash
python tools/visualize_transforms.py --data_path ./CWRU --save_path assets/transform_comparison.png
```

## Notes

- `window_size` is fixed to 2048 according to the paper experiment. The 4096 value in the code refers to the flattened vector length after concatenating DE and FE channels: `2048 × 2 = 4096`.
- Dataset tensors stay on CPU inside `FewshotDataset`; tensors are moved to CUDA only inside train/test loops.
- `--resolution 64` corresponds to `MainNet(h=16, w=16)`, and `--resolution 128` corresponds to `MainNet(h=32, w=32)`.
