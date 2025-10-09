# syntax=docker/dockerfile:1
# Multi-stage build for MNIST CNN project
# CPU-only image. For GPU, see docker-compose.gpu.yml

FROM python:3.11-slim as base

# System dependencies required by torchvision and matplotlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        libfreetype6 \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    TORCH_HOME=/app/.torch \
    PYTHONPATH=/app/src

# Create non-root user
RUN useradd --create-home --shell /bin/bash mnist
WORKDIR /app

# Development stage
FROM base as development

# Copy project files
COPY --chown=mnist:mnist . .

# Install package in development mode with dev dependencies
RUN pip install -e ".[dev,docs]"

# Create necessary directories
RUN mkdir -p /app/artifacts /app/logs /app/experiments /app/.torch \
    && chown -R mnist:mnist /app

USER mnist

# Default command for development
CMD ["python", "-m", "mnist_cnn.cli", "train", "--epochs", "2", "--device", "cpu"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY --chown=mnist:mnist src/ ./src/
COPY --chown=mnist:mnist config/ ./config/
COPY --chown=mnist:mnist pyproject.toml ./

# Install package
RUN pip install --no-cache-dir .

# Create necessary directories
RUN mkdir -p /app/artifacts /app/logs /app/.torch \
    && chown -R mnist:mnist /app

USER mnist

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mnist_cnn; print('OK')" || exit 1

# Default command
CMD ["mnist-train", "--epochs", "10", "--device", "cpu"]
