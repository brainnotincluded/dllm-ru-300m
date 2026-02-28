#!/bin/bash
# Quick setup command for Windows server
cd ~/dllm

# Install uv
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Setup environment
uv venv
source .venv/bin/activate
uv pip install torch --index-strategy unsafe-best-match
uv pip install -e .
uv pip install deepspeed

# Verify
python scripts/verify_server.py

echo "Setup complete!"
