# DLLM-RU-300M Design Document

**Date:** 2026-02-28  
**Status:** Approved  
**Model:** Bilingual Russian-English Diffusion LLM (300M params)  
**Target Hardware:** RTX 4090 (24GB VRAM)

---

## Executive Summary

This document describes the architecture and training plan for DLLM-RU-300M, a bilingual Russian-English diffusion language model inspired by state-of-the-art techniques including Inception Labs' Mercury 2 and research from 112 diffusion LLM papers.

**Key Design Decisions:**
- **Size**: 300M parameters (prototype for larger 3B+ model)
- **Languages**: Russian (60%) + English (40%)
- **Architecture**: Modern transformer with continuous flow matching diffusion
- **Training**: ~50B tokens over ~300 hours on RTX 4090
- **Innovation**: Incorporates Mercury 2's speed optimizations (speculative decoding framework, learned noise schedule, advanced KV caching)

---

## 1. Model Architecture

### 1.1 Base Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden Size | 1024 | Standard for 300M-1B models |
| Layers | 24 | Good depth for capacity |
| Attention Heads | 16 | 64 dims per head |
| Context Length | 2048 | Trainable, expandable to 4096 |
| Vocabulary | ~52K | Llama-3 base + Russian tokens |
| Parameters | ~300M | Fits comfortably in 24GB for training |

### 1.2 Modern Transformer Stack

Based on best practices from Llama-3, Mistral, and recent architecture papers:

**Normalizations:**
- RMSNorm pre-normalization (better stability than LayerNorm)
- Applied before attention and FFN blocks

**Activations:**
- SwiGLU activation (GLU variant from PaLM paper)
- FFN dimension: 4 * hidden_size = 4096

**Position Encoding:**
- Rotary Position Embeddings (RoPE)
- Base frequency: 10,000
- Supports context length extrapolation

**Attention:**
- Multi-head self-attention (16 heads)
- FlashAttention-2 implementation
- Causal masking for autoregressive generation

### 1.3 Diffusion-Specific Components

**Continuous Flow Matching (Mercury 2 style):**
- Not just masking - uses probability path interpolation on simplex
- Learnable noise schedule (cosine base with learned adjustments)
- Continuous timesteps t ∈ [0, 1]

**Timestep Embeddings:**
- Sinusoidal base embeddings
- 2-layer MLP projection to hidden dimension
- Added to token embeddings

**Output Head:**
- Linear projection to vocabulary size
- Categorical distribution over tokens
- No bias term (common in modern LLMs)

**Diffusion Training:**
- Random timestep sampling during training
- Flow matching objective (predict velocity)
- Loss: MSE between predicted and target velocity

### 1.4 Speed Optimizations

**Inference-Ready Features:**
- **KV Cache**: Optimized memory layout for fast inference
- **Speculative Decoding Framework**: Ready for Phase 2 (draft model training)
- **Advanced Sampling**: Info-Gain sampler implementation
- **Gradient Checkpointing**: For training only, disabled at inference

---

## 2. Tokenization Strategy

### 2.1 Base Tokenizer

**Starting Point:** Llama-3 tokenizer
- 50,256 base tokens
- Byte-level BPE (Byte Pair Encoding)
- Good multilingual coverage
- Efficient compression

### 2.2 Russian Extension

**Approach:** Extend base tokenizer with Russian-specific tokens

**New Tokens to Add (~2,000-5,000):**
- Russian character combinations
- Common Russian words/morphemes
- Russian punctuation patterns
- Mixed language patterns (e.g., "machine learning" in Cyrillic context)

**Training Process:**
1. Collect 10GB Russian text sample
2. Train BPE on Russian-only to identify gaps
3. Merge top 3,000 most frequent new tokens
4. Final vocab: ~53,000 tokens

**Benefits:**
- Maintains English compatibility
- Better Russian compression (fewer tokens per sentence)
- Can handle code-switching (mixing languages)

---

## 3. Training Data

### 3.1 Dataset Composition

| Source | Language | Size | Percentage | Notes |
|--------|----------|------|------------|-------|
| Russian Wikipedia | Russian | 5GB | 10% | High quality, cleaned |
| Russian Common Crawl | Russian | 30GB | 60% | Diverse web text |
| Russian Books | Russian | 10GB | 20% | Literature, technical |
| SlimPajama | English | 5GB | 10% | General knowledge |

**Total:** ~50B tokens (approximately 100GB compressed)

### 3.2 Data Processing

**Cleaning Pipeline:**
1. Deduplication (exact + fuzzy)
2. Language detection (filter non-target languages)
3. Quality scoring (heuristic filters)
4. PII removal (regex-based)
5. Tokenization verification

**Preprocessing:**
- Chunk into 2048-token sequences
- Add BOS/EOS tokens
- Create train/val split (99%/1%)

### 3.3 Data Loading

**Streaming Architecture:**
- WebDataset format for efficient loading
- Dynamic mixing of data sources
- Resumable training from any checkpoint

---

## 4. Training Configuration

### 4.1 Optimizer Settings

**AdamW Configuration:**
- Learning rate: 3e-4
- Betas: (0.9, 0.95)
- Epsilon: 1e-8
- Weight decay: 0.1
- Gradient clipping: 1.0

**Learning Rate Schedule:**
- Warmup: 2,000 steps (linear warmup)
- Main: Cosine decay to 3e-5
- No restarts

### 4.2 Training Loop

**Batch Configuration:**
- Micro-batch size: 4 (limited by 24GB VRAM)
- Gradient accumulation: 16 steps
- Effective batch size: 64
- Sequence length: 2048

**Memory Optimization:**
- bfloat16 mixed precision (bf16 for weights, fp32 for optimizer)
- Gradient checkpointing (every 2 layers)
- DeepSpeed ZeRO-2 (optimizer state sharding)
- Activation checkpointing enabled

**Training Duration:**
- Max steps: 100,000
- Estimated time: 300 hours (~12 days continuous)
- Checkpoints: Every 1,000 steps
- Validation: Every 500 steps

### 4.3 Diffusion-Specific Training

**Timestep Sampling:**
- Uniform random t ∈ [0, 1]
- Optional: importance sampling (more weight on harder timesteps)

**Noise Schedule:**
- Base: Cosine schedule
- Learned: Small MLP adjusts schedule per-example
- Initial training: Fixed schedule
- After 20K steps: Enable learned schedule

**Objective:**
- Flow matching: Predict velocity field
- Loss: L2 between predicted and target velocity
- Optional: Add auxiliary losses (e.g., consistency loss)

---

## 5. Infrastructure & Deployment

### 5.1 RTX 4090 Server Setup

**Hardware Specifications:**
- GPU: RTX 4090 (24GB VRAM)
- CPU: Modern multi-core (for data loading)
- RAM: 64GB+ recommended
- Storage: 500GB SSD (for datasets + checkpoints)

**Software Stack:**
- OS: Windows 10/11 (as specified)
- Python: 3.10+
- PyTorch: 2.1+ (with CUDA 12.1)
- DeepSpeed: 0.12+
- FlashAttention: 2.3+

### 5.2 SSH Access Setup

**Key-Based Authentication (Passwordless):**

```bash
# On local machine (generate key pair)
ssh-keygen -t ed25519 -C "dllm-training"

# Copy public key to Windows server
ssh-copy-id user@windows-server-ip
# OR manually:
# Copy ~/.ssh/id_ed25519.pub content
# Paste into server's C:\Users\<user>\.ssh\authorized_keys

# Verify (should not ask for password)
ssh user@windows-server-ip
```

**Remote Training Workflow:**
```bash
# 1. SSH to server
ssh user@windows-server-ip

# 2. Create tmux session for persistence
tmux new -s training

# 3. Activate environment & start training
conda activate dllm
cd /path/to/project
python train.py --config configs/dllm-300m.yaml

# 4. Detach: Ctrl+B then D
# Reattach later: tmux attach -t training
```

### 5.3 Monitoring & Logging

**Experiment Tracking:**
- Weights & Biases (wandb) for metrics logging
- TensorBoard for local visualization
- Metrics: loss, learning rate, grad norm, throughput

**Checkpointing:**
- Save every 1,000 steps
- Keep last 10 checkpoints
- Upload best checkpoints to cloud storage (optional)

**Alerting:**
- Notify on training completion/failure
- Monitor GPU temperature (nvidia-smi)

---

## 6. Evaluation Strategy

### 6.1 Automatic Metrics

**During Training:**
- Validation loss (monitored every 500 steps)
- Perplexity on held-out set
- Training loss curve

**Post-Training:**
- BLEU/ROUGE on translation tasks
- Perplexity on Russian/English benchmarks
- Zero-shot classification accuracy

### 6.2 Human Evaluation

**Qualitative Tests:**
- Text completion quality
- Russian grammar correctness
- Code generation (Python)
- Conversation coherence

**Benchmarks:**
- Russian SuperGLUE (if available)
- English SuperGLUE subset
- Custom test sets from Russian web

---

## 7. Future Roadmap

### Phase 2: Scaling Up (After 300M model)

**DLLM-RU-3B:**
- 3 billion parameters
- Requires multi-GPU or longer training
- Same architecture, scaled up

**DLLM-RU-8B:**
- 8 billion parameters
- Full Mercury 2 feature set:
  - Trained draft model for speculative decoding
  - Advanced test-time scaling
  - Optimized serving infrastructure

### Phase 3: Specialization

**Task-Specific Fine-tuning:**
- Instruction following (like ChatGPT)
- Code generation
- Summarization
- Translation

**Deployment:**
- vLLM for fast inference
- API server
- Quantization (4-bit/8-bit) for edge deployment

---

## 8. Key Design Principles

1. **Research-Driven**: Incorporates insights from all 112 papers in the collection
2. **Production-Ready**: Includes monitoring, checkpointing, and error handling
3. **Scalable**: Architecture designed to scale from 300M to 8B+ parameters
4. **Efficient**: Optimized for RTX 4090 constraints
5. **Bilingual**: Genuine bilingual capability, not just English model with Russian tokens

---

## 9. Risk Mitigation

**Training Instability:**
- Gradient clipping
- Careful learning rate tuning
- Mixed precision checks

**Memory Issues:**
- Gradient checkpointing
- ZeRO-2 optimization
- Smaller batch size if needed

**Data Quality:**
- Extensive preprocessing
- Validation monitoring
- Manual spot-checks

---

## 10. References

Key papers informing this design:
- **LLaDA**: Simple and Effective Masked Diffusion Language Models
- **Mercury 2**: Inception Labs (public blog posts)
- **Flow Matching**: Categorical Flow Maps
- **Speed Optimization**: MAGE, ReMix, Info-Gain Sampler papers
- **Architecture**: Llama-3, Mistral design choices

Full paper collection: See `papers_master_catalog.md` for all 112 papers.

---

**Next Step:** Proceed to implementation plan creation via `writing-plans` skill.

**Approved By:** User
**Date:** 2026-02-28
