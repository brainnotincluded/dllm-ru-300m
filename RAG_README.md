# DLLM Research RAG System

A complete Retrieval-Augmented Generation (RAG) system for the Diffusion LLM research collection, connected to your Ollama server at `62.140.252.238`.

## 🚀 Quick Start

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Index Papers (First Time Only)
```bash
python3 rag_complete.py --index
```
This will:
- Extract text from all 112 PDF papers
- Split into chunks
- Generate embeddings using `all-MiniLM-L6-v2`
- Store in Qdrant vector database

**Time**: ~5-10 minutes for 112 papers

### 3. Start Interactive Mode
```bash
python3 rag_complete.py --interactive
```

## 📖 Usage Examples

### Interactive Mode
```bash
$ python3 rag_complete.py --interactive

DLLM Research RAG - Interactive Mode
====================================
Ollama: http://62.140.252.238:11434
Model: llama3.2

❓ Query: What is the dLLM framework?
💡 Answer: The dLLM framework is a unified open-source framework 
for diffusion language modeling that standardizes training, inference, 
and evaluation components...

❓ Query: Compare diffusion vs autoregressive models
💡 Answer: Diffusion models offer several advantages...
```

### Single Query
```bash
python3 rag_complete.py --query "What are the latest speed optimization techniques?"
```

### With Specific Model
```bash
python3 rag_complete.py --query "Explain masked diffusion" --model qwen3:32b
```

## 🏗️ Architecture

```
User Query
    ↓
[Sentence Transformers] → Embeddings
    ↓
[Qdrant Vector DB] → Retrieve top-k chunks
    ↓
[Prompt Builder] → Context + Query
    ↓
[Ollama Server] → Generate Answer
    ↓
Response with sources
```

## 📁 Files

- `rag_complete.py` - Full RAG implementation
- `rag_system.py` - Simple RAG prototype
- `rag_launcher.py` - Quick launcher script
- `setup_rag.sh` - Setup script

## 🔧 Available Ollama Models

Your server has many powerful models:

| Model | Size | Use Case |
|-------|------|----------|
| `qwen3:32b` | 32B | Best overall |
| `deepseek-r1:32b` | 32B | Reasoning |
| `granite4:latest` | 3.4B | Fast queries |
| `llama3.2:1b-instruct` | 1.2B | Speed |
| `qwen3-coder:30b` | 30B | Code analysis |
| `gemma3:27b` | 27B | General purpose |

**Embedding Models**:
- `nomic-embed-text` (for RAG embeddings)
- `mxbai-embed-large`

## 💡 Example Queries

```bash
# Core concepts
"What is masked diffusion?"
"Explain the dLLM framework"

# Comparisons
"Compare diffusion vs autoregressive models"
"What are the advantages of diffusion LLMs?"

# Technical details
"How does ReMix achieve 2-8x speedup?"
"What is the Tri-Modal Masked Diffusion architecture?"

# Applications
"How are diffusion models used for code generation?"
"What are the latest multimodal diffusion techniques?"

# Trends
"What are the current challenges in diffusion LLMs?"
"Summarize the latest research directions"
```

## 🛠️ Advanced Usage

### Custom Top-k Retrieval
```python
from rag_complete import DLLMRAGQuery

rag = DLLMRAGQuery()
result = rag.query("Your question", top_k=10)
print(result["answer"])
```

### Batch Processing
```python
questions = [
    "What is LLaDA?",
    "How does ReMix work?",
    "Latest speed optimizations?"
]

for q in questions:
    result = rag.query(q)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:200]}...")
```

## 📊 Statistics

- **Total Papers**: 112
- **Categories**: 10
- **Avg Chunks per Paper**: ~20
- **Total Vectors**: ~2,240
- **Embedding Model**: all-MiniLM-L6-v2 (384-dim)

## 🔍 Troubleshooting

### Connection Issues
```bash
# Test Ollama connection
curl http://62.140.252.238:11434/api/tags

# Should return list of models
```

### Missing Dependencies
```bash
source venv/bin/activate
pip install requests qdrant-client PyMuPDF numpy sentence-transformers
```

### Indexing Issues
- Check Qdrant: `python3 rag_complete.py --stats`
- Re-index: `python3 rag_complete.py --index`

## 🎯 Next Steps

1. **Index papers**: `python3 rag_complete.py --index`
2. **Ask questions**: `python3 rag_complete.py --interactive`
3. **Explore**: Try different models with `--model qwen3:32b`

## 📚 Paper Categories

The RAG system covers:
- Core Diffusion Models (24 papers)
- Sampling & Speed Optimization (22 papers)
- Training & Alignment (16 papers)
- Reasoning & CoT (8 papers)
- Multimodal (12 papers)
- Code Generation (8 papers)
- AR vs Diffusion Studies (10 papers)
- Scaling Laws (7 papers)
- Safety & Robustness (5 papers)
- Applications (8 papers)

Total: **112 papers**
