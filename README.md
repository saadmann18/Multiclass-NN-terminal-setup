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

## Quickstart: Docker (Recommended)

The easiest way to run this project is with Docker. No local Python setup is required.

- Build the image:

```
docker build -t mnist-cnn:cpu .
```

- Run end-to-end (train then eval) using Docker Compose:

```
docker compose up --build app
```

- Or run directly with docker (train only by default, per Dockerfile CMD):

```
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$HOME/.torch/datasets:/root/.torch/datasets" \
  -e DEVICE=cpu -e EPOCHS=5 \
  mnist-cnn:cpu python -u train.py --device ${DEVICE:-cpu} --epochs ${EPOCHS:-5}
```

- Evaluate:

```
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$HOME/.torch/datasets:/root/.torch/datasets" \
  -e DEVICE=cpu \
  mnist-cnn:cpu python -u eval.py --device ${DEVICE:-cpu}
```

Artifacts are saved to `./artifacts/` on your host:

- `artifacts/model_mnist_cnn.pth`
- `artifacts/confusion_matrix.png`

## Notes on Performance & Precision

- Mixed precision (`autocast`) is enabled only on CUDA for stability. CPU/MPS run in full precision.
- The model outputs logits (no final Softmax). Use `CrossEntropyLoss` for training. For probabilities at inference, use `torch.softmax(logits, dim=1)`.
- On CUDA, we enable `channels_last` for better memory access and cuDNN benchmark for speed.
- On CPU, sensible threading defaults are set.

## Troubleshooting

- If you don’t see `artifacts/confusion_matrix.png`, ensure volume mounts are set correctly and that you ran evaluation (via Compose `app`/`eval`, or the direct eval command).
- Inside containers, plotting uses the non-interactive `Agg` backend and saves to disk. No GUI pops up.

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
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md
```

## Docker

This project includes a CPU-only Docker setup for easy, reproducible runs without polluting your host environment.

Notes:

- CPU is the default inside Docker. NVIDIA CUDA is optional (see below). Apple Silicon MPS is not available in Docker.
- Artifacts and dataset cache are persisted via bind mounts to your host for faster re-runs.

### 1) Build the image

```
docker build -t mnist-cnn:cpu .
```

### 2) Train (CPU)

The commands below mount two directories so your model checkpoint and downloaded MNIST cache persist between runs:

```
mkdir -p artifacts ~/.torch/datasets

docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$HOME/.torch/datasets:/root/.torch/datasets" \
  -e DEVICE=cpu \
  mnist-cnn:cpu train.py --device cpu --epochs 5
```

You can also use `--device auto` (the default) or set `-e DEVICE=auto`.

### 3) Evaluate (CPU)

After training, evaluate using the saved checkpoint and generate the confusion matrix PNG into `artifacts/`:

```
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$HOME/.torch/datasets:/root/.torch/datasets" \
  -e DEVICE=cpu \
  mnist-cnn:cpu eval.py --device cpu
```

Expected outputs (on host):

- `artifacts/model_mnist_cnn.pth`
- `artifacts/confusion_matrix.png`

### Optional: NVIDIA GPU (CUDA)

If you have an NVIDIA GPU and the NVIDIA Container Toolkit installed, you can run with GPU acceleration. Two options:

1) Use the same image and pass `--gpus all` (you must also install CUDA-compatible torch/torchvision in the image). This requires building a CUDA-enabled image first; the provided Dockerfile is CPU-only.

2) Build a CUDA-enabled image. Example Dockerfile sketch (replace versions with those matching your driver/toolkit):

```
# Example only — not provided by default in this repo
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip libjpeg-turbo-progs libpng16-16 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
# Install CUDA builds of torch/torchvision that match your CUDA version
RUN pip3 install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python3", "-u"]
```

Run with GPU:

```
docker run --rm --gpus all \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$HOME/.torch/datasets:/root/.torch/datasets" \
  -e DEVICE=cuda \
  mnist-cnn:cuda train.py --device cuda --epochs 5
```

Troubleshooting GPU builds is out of scope for this README; ensure your host drivers, CUDA version, and torch/torchvision wheels are compatible.

### Docker Compose

You can also use Docker Compose for shorter commands. The included `docker-compose.yml` defines three services: `app` (train→eval), `train` (train only), and `eval` (eval only).

- End-to-end (train → eval):

```
docker compose up --build app
```

- Override epochs or device at runtime:

```
EPOCHS=5 DEVICE=cpu docker compose up app
```

- Train only:

```
EPOCHS=5 DEVICE=cpu docker compose up train
```

- Eval only:

```
DEVICE=cpu docker compose up eval
```

Notes:

- Volumes mount `./artifacts` to `/app/artifacts` and your host `~/.torch/datasets` to `/root/.torch/datasets` inside the container.
- `DEVICE` defaults to `cpu` and `EPOCHS` defaults to `5` if not set.
- The compose file uses the same `mnist-cnn:cpu` image and sets `MPLBACKEND=Agg` for headless plotting.
