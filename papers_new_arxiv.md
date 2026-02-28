# New Diffusion LLM Papers (Feb 2025 - Feb 2026)

## 1. Core Models & Architecture (12 papers)
- **NAP: Non-Autoregressive Parallel DLMs** - Why diffusion struggles with truly parallel decoding and how to fix it (Feb 2026)
- **dLLM: Simple Diffusion Language Modeling** - Unified open-source framework for DLMs (Feb 2026)
- **ReMix: Rejection Mixing** - Fast semantic propagation for efficient DLLM inference, 2-8x speedup (Feb 2026)
- **Scaling Beyond Masked Diffusion** - First scaling law study of uniform-state vs masked diffusion (Feb 2026)
- **The Design Space of Tri-Modal Masked Diffusion** - First tri-modal model (text, image-text, audio-text), 3B params (Feb 2026)
- **The Diffusion Duality Chapter II** - Ψ-Samplers and efficient curriculum for uniform-state diffusion (Feb 2026)
- **IDLM: Inverse-distilled Diffusion Language Models** - 4x-64x inference speedup via inverse distillation (Feb 2026)
- **VDLM: Variable Diffusion LMs** - Modular variable diffusion with robust latent-to-text rendering (Feb 2026)
- **XDLM** - Balancing understanding and generation via stationary noise kernel (Feb 2026)
- **LoMDM: Learnable-Order Masked Diffusion** - Jointly learns generation ordering and diffusion (Feb 2026)
- **Info-Gain Sampler** - Information-theoretic sampling for masked diffusion models (Feb 2026)
- **DSL: Discrete Stochastic Localization** - 4x fewer denoiser evaluations for MDLM/ReMDM (Feb 2026)

## 2. Reasoning & Chain-of-Thought (6 papers)
- **Test-Time Scaling via Reward-Guided Stitching** - Stitching noisy diffusion thoughts, 23.8% accuracy improvement (Feb 2026)
- **LaDi-RL: Latent Diffusion Reasoning** - RL in continuous latent space, +9.4% on code generation (Feb 2026)
- **McDiffuSE** - Monte-Carlo Tree Search for slot filling ordering in diffusion LMs (Feb 2026)
- **Prompt Optimization Via Diffusion Language Models** - DLM-based prompt optimization framework (Feb 2026)
- **Why Any-Order Autoregressive Models Need Two-Stream Attention** - Structural-semantic tradeoff analysis (Feb 2026)

## 3. Multimodal & Vision (8 papers)
- **MAGE** - All-[MASK] block knows where to look, 3-4x speedup for long-context (Feb 2026)
- **SkyReels-V4** - Multi-modal video-audio generation with MMDiT architecture (Feb 2026)
- **BitDance** - Binary token autoregressive with diffusion head, 30x speedup for 1024px images (Feb 2026)
- **DiffuSpeech** - Silent Thought, Spoken Answer - unified speech-text diffusion (Jan 2026)
- **DODO: Discrete OCR Diffusion** - 3x faster OCR with block discrete diffusion (Feb 2026)
- **DiMo** - Discrete diffusion for motion generation and understanding (Feb 2026)
- **MVLAD-AD** - Masked Vision-Language-Action Diffusion for autonomous driving (Feb 2026)
- **DriveFine** - Refining-augmented masked diffusion VLA for driving (Feb 2026)

## 4. Training & Optimization (7 papers)
- **TabDLM** - Joint numerical-language diffusion for tabular data generation (Feb 2026)
- **DiSP: Diffusion Self-Purification** - Backdoor defense for multimodal diffusion LMs (Feb 2026)
- **Categorical Flow Maps** - Flow matching for categorical data, SOTA few-step generation (Feb 2026)
- **Sharp Convergence Rates for Masked Diffusion** - Theoretical analysis of Euler and FHS samplers (Feb 2026)
- **Balancing Understanding and Generation** - XDLM unifies MDLM and UDLM paradigms (Feb 2026)
- **Corrected Samplers for Discrete Flow Models** - Time-corrected and location-corrected samplers (Jan 2026)

## 5. Inference & Sampling (6 papers)
- **ILRR: Inference-Time Steering** - Learning-free steering for DLMs with reference sequences (Jan 2026)
- **dLLM-ASR** - 4.44x speedup for speech recognition with diffusion LLMs (Jan 2026)
- **Rejection Mixing** - Continuous mixing state for parallel decoding (Feb 2026)
- **Spectral Generative Flow Models** - Physics-inspired replacement for transformer LLMs (Jan 2026)

## 6. Applications (7 papers)
- **DiffuTruth** - Detecting hallucinations via diffusion model likelihoods (Feb 2026)
- **CogitoRAG** - Cognitive gist-driven RAG with global semantic diffusion (Feb 2026)
- **Towards Latent Diffusion Suitable For Text** - Neural Flow Diffusion Models (Jan 2026)
- **Absorbing Discrete Diffusion for Speech Enhancement** - ADDSE with RQDiT (Feb 2026)
- **Self-Purification Mitigates Backdoors** - Defense for multimodal diffusion LMs (Feb 2026)

## 7. Novel Directions (4 papers)
- **From Words to Amino Acids** - Curse of depth in protein language models with diffusion (Feb 2026)
- **Breaking Semantic-Aware Watermarks** - LLM-guided attacks on diffusion watermarks (Feb 2026)
- **No Caption, No Problem** - Caption-free membership inference for diffusion models (Feb 2026)
- **Breaking AR's Sampling Bottleneck** - Provable acceleration via diffusion LMs (already in collection)

---

## Key Trends Observed (Feb 2025-2026):

1. **Speed is the #1 priority**: Most papers claim 2-64x speedups through various techniques
2. **True parallel decoding**: Moving beyond AR-like behavior to genuine non-autoregressive generation
3. **Multimodal expansion**: Strong push into vision, audio, speech, and motion
4. **Training frameworks**: dLLM framework aims to standardize the field
5. **Reasoning integration**: Combining diffusion with CoT and RL for better reasoning
6. **Long context solutions**: MAGE, UltraLLaDA tackling 128K+ context
7. **Theoretical foundations**: Convergence rates, scaling laws, sampler analysis
8. **Hybrid approaches**: Block diffusion, AR-diffusion interpolation

## Notable Papers to Prioritize:
- **dLLM Framework** - First comprehensive open-source framework
- **Tri-Modal Masked Diffusion** - Largest multimodal study to date
- **Scaling Beyond Masked Diffusion** - Challenges MDLM dominance
- **NAP** - Addresses fundamental parallel decoding limitations
- **Test-Time Scaling** - Novel stitching approach for reasoning
- **Info-Gain Sampler** - Principled information-theoretic sampling

## arXiv Search Results:
- "diffusion language model": 3,187 total papers
- "masked diffusion language": 390 papers  
- "discrete diffusion text": 361 papers
- **~150 new papers in last 12 months**
