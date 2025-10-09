#!/bin/bash
# Setup script for MNIST CNN development environment

set -e

echo "Setting up MNIST CNN development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing package in development mode..."
pip install -e ".[dev]"

# Install additional development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating project directories..."
mkdir -p artifacts
mkdir -p logs
mkdir -p data

echo "✓ Setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  make test"
echo ""
echo "To train the model:"
echo "  make train"
echo ""
echo "To evaluate the model:"
echo "  make eval"
