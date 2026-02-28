# AGENTS.md
## Diffusion Language Model (DLLM) Research Assistant

### Project Overview
A comprehensive RAG-enabled research assistant for exploring 112+ diffusion language model papers. The system uses vector embeddings (all-MiniLM-L6-v2) to index papers and connects to an Ollama server for intelligent querying.

**Server Details:**
- **Ollama Host**: 62.140.252.238:11434
- **Available Models**: 41 models including qwen3:32b, deepseek-r1:32b, granite4:latest
- **Paper Collection**: 112 PDFs across 10 categories
- **Vector Database**: Qdrant with 384-dimensional embeddings

---

## Quick Start Commands

### 1. Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
dllm config
```

### 2. First Time Setup
```bash
# Index all papers (5-10 minutes)
dllm index

# Verify indexing
dllm stats
```

### 3. Basic Usage
```bash
# Single query
dllm query "What is the dLLM framework?"

# Interactive chat
dllm chat

# Use specific model
dllm query "speed optimization" --model qwen3:32b

# List all papers
dllm papers

# Search papers by keyword
dllm search "multimodal"
```

---

## CLI Commands Reference

### Core Commands

#### `dllm index`
**Purpose**: Index all PDF papers into the vector database  
**Time**: 5-10 minutes for 112 papers  
**Storage**: Creates ~2,240 vectors in Qdrant

```bash
# Index with default settings
dllm index

# Index from custom directory
dllm index --papers-dir /path/to/papers
```

#### `dllm query`
**Purpose**: Query papers using RAG (Retrieval-Augmented Generation)  
**Parameters**:
- `query_text`: Your question (required)
- `--model`: Ollama model to use (default: llama3.2)
- `--top-k`: Number of chunks to retrieve (default: 5)

```bash
# Simple query
dllm query "What is diffusion language modeling?"

# With specific model
dllm query "Compare AR vs diffusion" --model deepseek-r1:32b

# More context
dllm query "speed techniques" --top-k 10

# Interactive mode
dllm query --interactive
```

#### `dllm chat`
**Purpose**: Start interactive chat session  
**Features**: Context-aware, can switch models mid-session

```bash
# Start chat
dllm chat

# With specific model
dllm chat --model qwen3:32b

# More context chunks
dllm chat --top-k 10
```

**Chat Commands**:
- `/quit` - Exit chat
- `/models` - List available Ollama models
- `/stats` - Show collection statistics
- `/clear` - Clear screen
- `/help` - Show help

#### `dllm stats`
**Purpose**: Display indexing statistics

```bash
dllm stats
```

**Output Example**:
```
📊 COLLECTION STATISTICS
============================================================

📁 Collections:
   • dllm_papers_chunks
     Vectors: 2240
     Status: green
     Distance: Cosine
     Unique papers: 112
```

#### `dllm papers`
**Purpose**: List all papers organized by category

```bash
dllm papers
```

#### `dllm search`
**Purpose**: Search papers by keyword in filename

```bash
# Search for speed-related papers
dllm search "speed"

# Search for specific author/paper
dllm search "LLaDA"
```

#### `dllm config`
**Purpose**: Show system configuration and verify connectivity

```bash
dllm config
```

**Output Example**:
```
⚙️  CONFIGURATION
============================================================

Ollama Host: http://62.140.252.238:11434
Default Model: llama3.2
Papers Directory: ./papers
Qdrant Path: ./rag_data/qdrant

Available Models (41 total):
  • qwen3:32b ⭐
  • deepseek-r1:32b
  • granite4:latest
  ...

Papers: 112 PDFs found
```

---

## Recommended Models by Use Case

### For Deep Reasoning
```bash
dllm query "complex reasoning question" --model deepseek-r1:32b
```
**Why**: Deepseek-r1 is optimized for reasoning tasks

### For Speed and Efficiency
```bash
dllm query "quick question" --model granite4:latest
```
**Why**: granite4 is 3.4B params, fast inference

### For Code-Related Queries
```bash
dllm query "code generation techniques" --model qwen3-coder:30b
```
**Why**: Specialized for code understanding

### For General Queries
```bash
dllm query "what is diffusion?" --model qwen3:32b
```
**Why**: Best overall performance, 32B parameters

---

## Example Queries

### Core Concepts
```bash
dllm query "What are diffusion language models?"
dllm query "Explain masked diffusion"
dllm query "How does discrete diffusion work?"
```

### Comparisons
```bash
dllm query "Compare diffusion vs autoregressive models"
dllm query "What are advantages of diffusion LLMs?"
dllm query "When should I use diffusion over AR?"
```

### Technical Details
```bash
dllm query "How does ReMix achieve 2-8x speedup?"
dllm query "Explain the Tri-Modal Masked Diffusion architecture"
dllm query "What is the dLLM framework?"
dllm query "How does Info-Gain Sampler work?"
```

### Applications
```bash
dllm query "Applications of diffusion models for code generation"
dllm query "How are diffusion models used in vision tasks?"
dllm query "What are the latest multimodal diffusion techniques?"
```

### Research Trends
```bash
dllm query "Current challenges in diffusion LLMs"
dllm query "What are the latest speed optimization techniques?"
dllm query "Summarize recent advances in 2025-2026"
dllm query "Future directions for diffusion language models"
```

---

## Project Structure

```
DLLM/
├── dllm                    # Main CLI executable
├── rag_complete.py         # Core RAG implementation
├── papers/                 # 112 PDF papers
│   ├── 001_Simplified_Masked_Diffusion.pdf
│   ├── ...
│   └── 112_latest_paper.pdf
├── rag_data/
│   └── qdrant/            # Vector database storage
├── papers_master_catalog.md  # Complete paper catalog
├── RAG_README.md          # RAG system documentation
└── venv/                  # Python virtual environment
```

---

## Troubleshooting

### Ollama Connection Issues
```bash
# Test connection
curl http://62.140.252.238:11434/api/tags

# Should return list of models
```

**Solution**: If connection fails, ensure Ollama is running on the server

### Indexing Issues
```bash
# Check if papers exist
ls papers/*.pdf | wc -l

# Re-index if needed
dllm index
```

### No Results from Queries
```bash
# Check if indexing completed
dllm stats

# If no vectors, re-index
dllm index
```

### Model Not Found
```bash
# List available models
dllm config

# Use exact model name from list
dllm query "..." --model "exact-model-name:latest"
```

---

## Paper Categories

The collection is organized into 10 categories:

1. **Core Diffusion Models** (24 papers) - Foundational work including LLaDA, dLLM framework
2. **Sampling & Speed Optimization** (22 papers) - ReMix, MAGE, acceleration techniques
3. **Training & Alignment** (16 papers) - RLHF, DPO, PPO for diffusion models
4. **Reasoning & CoT** (8 papers) - Test-time scaling, Monte Carlo methods
5. **Multimodal** (12 papers) - Vision, speech, audio applications
6. **Code Generation** (8 papers) - DreamCoder, DiffuCoder
7. **AR vs Diffusion Studies** (10 papers) - Comparative analysis
8. **Scaling Laws** (7 papers) - Data efficiency, convergence rates
9. **Safety & Robustness** (5 papers) - Hallucination detection, backdoor defense
10. **Applications** (8 papers) - Tabular data, RAG, novel use cases

---

## Advanced Usage

### Batch Queries
Create a script for batch processing:

```bash
#!/bin/bash
queries=(
  "What is diffusion?"
  "Speed optimization techniques"
  "Code generation applications"
)

for q in "${queries[@]}"; do
  echo "Query: $q"
  dllm query "$q" --model qwen3:32b
  echo "---"
done
```

### Custom Top-K
```bash
# More context for complex questions
dllm query "complex architecture question" --top-k 15

# Less context for simple questions  
dllm query "simple definition" --top-k 3
```

### Export Results
```bash
# Save query results
dllm query "speed techniques" > results.txt

# With model info
dllm query "..." --model qwen3:32b 2>&1 | tee results.txt
```

---

## Environment Variables

```bash
# Override Ollama host
export OLLAMA_HOST="http://different-server:11434"

# Then run commands
dllm query "..."
```

---

## Key Features

✅ **112 Research Papers** - Comprehensive DLLM collection  
✅ **RAG-Enabled** - Retrieves relevant context before generating answers  
✅ **Multiple Models** - Choose from 41 Ollama models  
✅ **Vector Search** - Semantic similarity using all-MiniLM-L6-v2  
✅ **Interactive Mode** - Chat-style interface  
✅ **Fast Indexing** - ~5-10 minutes for full collection  
✅ **Categorized** - Papers organized by topic  
✅ **Source Tracking** - See which papers contributed to answers  

---

## Performance Tips

1. **Use appropriate model**: Use smaller models (granite4) for simple queries, larger models (qwen3:32b) for complex reasoning
2. **Adjust top-k**: Increase for complex questions, decrease for simple ones
3. **Re-index periodically**: If adding new papers, re-index the collection
4. **Interactive mode**: Use for exploration, single queries for specific answers

---

## Citation

When using this research assistant, cite the relevant papers from the collection:

```
Paper: [Title from source list]
Category: [From papers_master_catalog.md]
Available at: ./papers/[filename].pdf
```

---

## Support

For issues or questions:
1. Check `dllm config` for connectivity
2. Verify `dllm stats` shows indexed vectors
3. Review error messages carefully
4. Check Ollama server status: `curl http://62.140.252.238:11434/api/tags`

---

**Last Updated**: 2026-02-27  
**Total Papers**: 112  
**Categories**: 10  
**Models Available**: 41
