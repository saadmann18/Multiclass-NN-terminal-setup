# Multiclass-NN-terminal-setup
This project mimics SID task from Fearless project. The aim is:

1. model, data, eval(with saved model) setup
2. Tensorboard setup
3. Additive margin soft max analysis
4. ResNet model rebuild

## Overview

This repository contains a simple CNN on MNIST with separate `model.py`, `data.py`, `train.py`, and `eval.py` modules.

Recent updates add:

- Unified device selection: CUDA (NVIDIA), MPS (Apple Silicon), or CPU.
- Mixed-precision training on CUDA only (safe fallbacks elsewhere).
- Jupyter-friendly APIs to run training/evaluation from a single cell.
- Confusion matrix plotting and saving to `artifacts/confusion_matrix.png`.

Key files:

- `train.py` – training entrypoints (`main()` and `run_training()`)
- `eval.py` – evaluation entrypoints (`main()` and `run_evaluation()`)
- `model.py` – CNN definition (outputs logits; no final Softmax)
- `data.py` – MNIST dataloaders
- `artifacts/` – saved model and plots

## Environment Setup

You can use Conda environments for a clean setup. Choose one option:

### Apple Silicon (MPS)

```
conda create -n mnist-cnn python=3.10 -y
conda activate mnist-cnn
conda install -y pytorch torchvision -c pytorch -c conda-forge
```

### NVIDIA CUDA (example: CUDA 12.1)

```
conda create -n mnist-cnn python=3.10 -y
conda activate mnist-cnn
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
```

### CPU-only

```
conda create -n mnist-cnn python=3.10 -y
conda activate mnist-cnn
conda install -y pytorch torchvision cpuonly -c pytorch -c conda-forge
```

Optional (for plotting in notebooks/CLI):

```
conda install -y matplotlib -c conda-forge
```

## Running from the Command Line

All commands assume the project root: `Multiclass-NN-terminal-setup/`

### Train

```
python3 train.py --device auto --epochs 5
```

Choose device explicitly if needed:

- MPS (Apple Silicon): `--device mps`
- CUDA (NVIDIA): `--device cuda`
- CPU: `--device cpu`

Optional (PyTorch 2.x on CUDA/CPU):

```
python3 train.py --device cuda --epochs 5 --compile
```

### Evaluate

```
python3 eval.py --device auto
```

To also display the confusion matrix window (if interactive GUI available):

```
python3 eval.py --device auto --show
```

Output artifacts are saved to `artifacts/`:

- `artifacts/model_mnist_cnn.pth`
- `artifacts/confusion_matrix.png`

## Running from Jupyter (Single Cell)

We provide notebook-friendly functions that avoid argparse and work cleanly in Jupyter.

```
import sys, os
project_root = "/Users/saud06/dev/Multiclass-NN-terminal-setup"  # adjust if needed
if project_root not in sys.path:
    sys.path.append(project_root)

from train import run_training
from eval import run_evaluation

# Select device: 'auto' | 'mps' | 'cuda' | 'cpu'
device = 'mps'
epochs = 5

# 1) Train
model, ckpt_path = run_training(device_pref=device, epochs=epochs, compile=False)

# 2) Evaluate and display confusion matrix inline
acc, avg_loss, conf_mat, per_class_acc, fig_path = run_evaluation(device=device, ckpt_path=ckpt_path, show=True)
print(f"Accuracy: {acc*100:.2f}%  |  Avg loss: {avg_loss:.4f}")
print("Saved plot:", fig_path)
```

### Displaying a Saved Plot in Jupyter

If `matplotlib` is installed:

```
import os
import matplotlib.pyplot as plt

img_path = os.path.join(project_root, "artifacts", "confusion_matrix.png")
img = plt.imread(img_path)
plt.figure(figsize=(6,5))
plt.imshow(img)
plt.axis("off")
plt.title("Confusion Matrix")
plt.show()
```

Without installing `matplotlib`, use IPython display:

```
from IPython.display import Image, display
img_path = os.path.join(project_root, "artifacts", "confusion_matrix.png")
display(Image(filename=img_path))
```

## Notes on Performance & Precision

- Mixed precision (`autocast`) is enabled only on CUDA for stability. CPU/MPS run in full precision.
- The model outputs logits (no final Softmax). Use `CrossEntropyLoss` for training. For probabilities at inference, use `torch.softmax(logits, dim=1)`.
- On CUDA, we enable `channels_last` for better memory access and cuDNN benchmark for speed.
- On CPU, sensible threading defaults are set.

## Troubleshooting

### OpenMP conflict on macOS (libomp)

If you see:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

It means multiple OpenMP runtimes are loaded (e.g., mixing pip and conda packages). Fix by ensuring PyTorch/torchvision come from Conda only and keeping LLVM OpenMP:

```
pip uninstall -y torch torchvision
conda remove intel-openmp
conda install -y -c conda-forge llvm-openmp
conda install -y pytorch torchvision -c pytorch -c conda-forge
```

As a temporary workaround to unblock a single run:

```
KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --device mps --epochs 1
```

### Jupyter argparse errors (ipykernel `-f`)

If running `main()` directly in a notebook causes an error about `-f ...` arguments, use the notebook APIs `run_training()` / `run_evaluation()` as shown above, or ensure the scripts use `parse_known_args()` (already configured).

## Repository Structure

```
.
├── artifacts/
│   ├── model_mnist_cnn.pth
│   └── confusion_matrix.png
├── data.py
├── eval.py
├── model.py
├── train.py
└── README.md
```

