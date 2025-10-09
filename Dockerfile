# syntax=docker/dockerfile:1
# CPU-only image. For GPU, see README section for CUDA.
FROM python:3.10-slim

# System deps required by torchvision image loaders
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    TORCH_HOME=/app/.torch

# Workdir
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create artifacts dir (bind-mount recommended in README)
RUN mkdir -p /app/artifacts /app/.torch

# Default entrypoint runs Python; command can be overridden in Docker Desktop or Compose
ENTRYPOINT ["python", "-u"]
# Default command runs a short CPU training so clicking "Run" does something useful
CMD ["train.py", "--device", "cpu", "--epochs", "1"]
