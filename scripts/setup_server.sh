#!/bin/bash
# Remote training setup script

set -e

echo "Setting up DLLM training on remote server..."

# Check Python
python --version

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install torch --index-strategy unsafe-best-match
uv pip install -e .

# Install DeepSpeed
echo "Installing DeepSpeed..."
uv pip install deepspeed

# Verify installation
echo "Verifying installation..."
python scripts/verify_server.py

echo "Setup complete!"
