# Diffusion Language Model Research Collection
## Master Catalog - 112 Papers

---

## 1. CORE DIFFUSION MODELS (24 papers)

### Foundational Work
- Simplified and Generalized Masked Diffusion for Discrete Data
- Why Masking Diffusion Works: Condition on the Jump Schedule
- Masked Diffusion Models are Secretly Time-Agnostic Masked Models
- Simple and Effective Masked Diffusion Language Models
- Information-Theoretic Discrete Diffusion
- Scaling up Masked Diffusion Models on Text
- Structured Denoising Diffusion Models in Discrete State-Spaces
- Large Language Diffusion Models (LLaDA)
- A Cheaper and Better Diffusion Language Model with Soft-Masked Noise
- DiffusionBERT: Improving Generative Masked Language Models
- Discrete Diffusion Models for Language Generation

### Recent Advances (2025-2026)
- **Why Diffusion Language Models Struggle with Truly Parallel Decoding?** (Feb 2026)
- **dLLM: Simple Diffusion Language Modeling** - Unified framework (Feb 2026)
- **Scaling Beyond Masked Diffusion Language Models** - First scaling law study (Feb 2026)
- **The Design Space of Tri-Modal Masked Diffusion Models** - 3B params, 6.4T tokens (Feb 2026)
- **The Diffusion Duality Chapter II: Ψ-Samplers and Efficient Curriculum** (Feb 2026)
- **IDLM: Inverse-distilled Diffusion Language Models** - 4x-64x speedup (Feb 2026)
- **Unifying Masked Diffusion Models with Various Generation Orders** (Feb 2026)
- **XDLM: Balancing Understanding and Generation** (Feb 2026)
- **Your Absorbing Discrete Diffusion Secretly Models the Bayesian Posterior**

---

## 2. SAMPLING, INFERENCE & SPEED OPTIMIZATION (22 papers)

### Speed Acceleration
- **ReMix: Rejection Mixing** - 2-8x inference speedup (Feb 2026)
- **Info-Gain Sampler** - Information-theoretic sampling (Feb 2026)
- **Discrete Stochastic Localization** - 4x fewer denoiser evaluations (Feb 2026)
- **dLLM-ASR: A Faster Diffusion LLM-based Framework for Speech Recognition** - 4.44x speedup
- **Accelerating Diffusion Language Model Inference via Efficient KV Caching**
- **dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching**
- **dKV-Cache: The Cache for Diffusion Language Models**
- **Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models**
- **Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing**
- **Beyond Autoregression: Fast LLMs via Self-Distillation Through Time**

### Sampling Methods
- **ILRR: Inference-Time Steering for Masked Diffusion Language Models** (Jan 2026)
- Path Planning for Masked Diffusion Model Sampling
- Guided Star-Shaped Masked Diffusion
- Theory-Informed Improvements to Classifier-Free Guidance
- Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models
- **Sharp Convergence Rates for Masked Diffusion Models** (Feb 2026)
- **Corrected Samplers for Discrete Flow Models** (Jan 2026)
- **Categorical Flow Maps** (Feb 2026)

### Long Context
- **MAGE: All-[MASK] Block Already Knows Where to Look** - 3-4x speedup (Feb 2026)
- UltraLLaDA: Scaling the Context Length to 128K
- LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs
- DuoAttention: Efficient Long-Context LLM Inference
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

---

## 3. TRAINING, ALIGNMENT & RL (16 papers)

### Alignment Methods
- Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study
- DPO Meets PPO: Reinforced Token Optimization for RLHF
- ReMax: A Simple, Effective RL Method for Aligning Large Language Models
- Pretrain Value, Not Reward: Decoupled Value Policy Optimization
- Optimizing Safe and Aligned Language Generation: A Multi-Objective GRPO Approach
- LLaDA 1.5: Variance-Reduced Preference Optimization
- VRPO: Rethinking Value Modeling for Robust RL Training
- RRHF: Rank Responses to Align Language Models
- Don't Forget Your Reward Values: Language Model Alignment via Value-based Calibration

### Reinforcement Learning
- **LaDi-RL: Latent Diffusion Reasoning with RL** - +9.4% code generation (Feb 2026)
- Is Reinforcement Learning (Not) for Natural Language Processing?
- **Test-Time Scaling via Reward-Guided Stitching** - +23.8% accuracy (Feb 2026)

### Training Techniques
- **Diffusion Self-Purification** - Backdoor defense for MDLMs (Feb 2026)
- Fast Training of Diffusion Models with Masked Transformers
- **TabDLM: Free-Form Tabular Data Generation** - Joint numerical-language diffusion (Feb 2026)
- **From Words to Amino Acids** - Curse of depth in protein LMs (Feb 2026)

---

## 4. REASONING & CHAIN-OF-THOUGHT (8 papers)

- **Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching** (Feb 2026)
- **LaDi-RL: Latent Diffusion Reasoning with RL** (Feb 2026)
- **McDiffuSE: Monte-Carlo Tree Search for Slot Filling** (Feb 2026)
- **Prompt Optimization Via Diffusion Language Models** (Feb 2026)
- Think While You Generate: Discrete Diffusion with Planned Denoising
- No Compute Left Behind: Rethinking Reasoning and Sampling
- Don't Settle Too Early: Self-Reflective Remasking
- **VDLM: Variable Diffusion LMs** - Separates planning from rendering (Feb 2026)

---

## 5. MULTIMODAL, VISION & SPEECH (12 papers)

### Vision-Language
- **MAGE: All-[MASK] Block Already Knows Where to Look** (Feb 2026)
- LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
- **SkyReels-V4: Multi-modal Video-Audio Generation** - MMDiT architecture (Feb 2026)
- **BitDance: Scaling Autoregressive with Binary Tokens** - 30x speedup (Feb 2026)
- **DODO: Discrete OCR Diffusion Models** - 3x faster OCR (Feb 2026)
- **DiMo: Discrete Diffusion for Motion Generation** (Feb 2026)
- **MVLAD-AD: Masked Vision-Language-Action Diffusion** - Autonomous driving (Feb 2026)
- **DriveFine: Refining-Augmented Masked Diffusion VLA** - Driving (Feb 2026)
- LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs
- Adapting LLaMA Decoder to Vision Transformer

### Speech & Audio
- **DiffuSpeech: Silent Thought, Spoken Answer** - Unified speech-text diffusion (Jan 2026)
- **dLLM-ASR: Faster Diffusion LLM-based ASR** - 4.44x speedup (Jan 2026)
- Absorbing Discrete Diffusion for Speech Enhancement

---

## 6. APPLICATIONS: CODE GENERATION (8 papers)

- Exploring the Power of Diffusion LLMs for Software Engineering
- Dream-Coder 7B: An Open Diffusion Language Model for Code
- DreamOn: Diffusion Language Models For Code Infilling
- Efficient Training of Language Models to Fill in the Middle
- Beyond Autoregression: An Empirical Study of Diffusion LLMs for Code Generation
- DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation
- **LaDi-RL: Latent Diffusion Reasoning with RL** - +9.4% on code (Feb 2026)
- **McDiffuSE: Monte-Carlo Tree Search** - Slot filling for code (Feb 2026)

---

## 7. AR vs DIFFUSION COMPARATIVE STUDIES (10 papers)

- TiDAR: Think in Diffusion, Talk in Autoregression
- Double Descent as a Lens for Sample Efficiency: AR vs Discrete Diffusion
- Beyond Next-Token Prediction: Performance Characterization of Diffusion vs AR
- Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models
- Diffusion Beats Autoregressive in Data-Constrained Settings
- Diffusion Language Models are Super Data Learners
- Diffusion Language Models Know the Answer Before Decoding
- Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective
- **Why Diffusion Language Models Struggle with Truly Parallel Decoding?** (Feb 2026)
- **Why Any-Order Autoregressive Models Need Two-Stream Attention** (Feb 2026)

---

## 8. SCALING LAWS & DATA EFFICIENCY (7 papers)

- Scaling Behavior of Discrete Diffusion Language Models
- Scaling Data-Constrained Language Models
- Scaling Diffusion Language Models via Adaptation from Autoregressive Models
- Scaling Laws for Neural Language Models
- What Language Model Architecture Works Best for Zero-Shot Generalization?
- **Scaling Beyond Masked Diffusion Language Models** - Scaling law study (Feb 2026)
- **The Design Space of Tri-Modal Masked Diffusion Models** - 3B params (Feb 2026)

---

## 9. SAFETY, ROBUSTNESS & INTERPRETABILITY (5 papers)

- **Diffusion Self-Purification** - Backdoor defense (Feb 2026)
- **DiffuTruth: Detecting Hallucinations via Diffusion Likelihoods** (Feb 2026)
- **Breaking Semantic-Aware Watermarks via LLM-Guided Injection** (Feb 2026)
- Self-Purification Mitigates Backdoors in Multimodal Diffusion Language Models
- No Caption, No Problem: Caption-Free Membership Inference for Diffusion Models

---

## 10. ADVANCED APPLICATIONS (8 papers)

- **CogitoRAG: Cognitive Gist-Driven RAG Framework** - With global semantic diffusion (Feb 2026)
- **Towards Latent Diffusion Suitable For Text** - Neural Flow Diffusion Models (Jan 2026)
- **TabDLM: Free-Form Tabular Data Generation** (Feb 2026)
- **DiMo: Discrete Diffusion for Motion Generation** (Feb 2026)
- **Prompt Optimization Via Diffusion Language Models** (Feb 2026)
- Remasking Discrete Diffusion Models with Inference-Time Scaling
- Review, Remask, Refine (R3): Process-Guided Block Diffusion
- Token Weighting for Long-Range Language Modeling

---

## STATISTICS

**Total Papers: 112**

### By Category:
- Core Models: 24 papers
- Sampling & Speed: 22 papers
- Training & Alignment: 16 papers
- Reasoning: 8 papers
- Multimodal: 12 papers
- Code Generation: 8 papers
- AR vs Diffusion: 10 papers
- Scaling Laws: 7 papers
- Safety: 5 papers
- Applications: 8 papers

### By Year:
- 2025-2026 (New): ~37 papers
- 2024 and earlier: ~75 papers

### Key Trends:
1. **Speed is #1 priority** - Most papers claim 2-64x speedups
2. **True parallel decoding** - Moving beyond AR-like behavior
3. **Multimodal explosion** - Vision, audio, speech, motion
4. **Unified frameworks** - dLLM framework standardizing the field
5. **Reasoning integration** - Diffusion + CoT + RL
6. **Long context solved** - 128K+ context achievable
7. **Theoretical foundations** - Convergence rates, scaling laws

---

## MUST-READ PAPERS

### Foundational:
1. Large Language Diffusion Models (LLaDA)
2. DiffusionBERT
3. Simple and Effective Masked Diffusion Language Models

### Game Changers (2025-2026):
1. **dLLM: Simple Diffusion Language Modeling** - First comprehensive framework
2. **The Design Space of Tri-Modal Masked Diffusion** - Largest multimodal study
3. **Scaling Beyond Masked Diffusion** - Challenges MDLM dominance
4. **ReMix** - 2-8x speedup without quality loss
5. **Test-Time Scaling via Reward-Guided Stitching** - Novel reasoning approach
6. **MAGE** - 3-4x long-context speedup
7. **Info-Gain Sampler** - Principled information-theoretic approach

### Implementation:
1. **dLLM Framework** - Code: https://github.com/ZHZisZZ/dllm
2. **Scaling Beyond Masked** - Code: https://github.com/s-sahoo/scaling-dllms
3. **Diffusion Duality** - Code: https://s-sahoo.com/duo-ch2
