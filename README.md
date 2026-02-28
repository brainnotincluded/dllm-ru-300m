# DLLM-RU-300M

Bilingual Russian-English Diffusion Language Model (300M parameters)

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Training

```bash
# Download data
python scripts/download_data.py

# Train tokenizer
python scripts/train_tokenizer.py

# Start training
python train.py --config configs/dllm-300m.yaml
```

## Project Structure

- `src/` - Model architecture and training code
- `scripts/` - Data preparation and utility scripts
- `configs/` - Training configurations
- `data/` - Training data (gitignored)
- `checkpoints/` - Model checkpoints (gitignored)
