# DLLM RAG System Setup Script
# This script sets up the RAG system for the DLLM research project

set -e

echo "🔧 DLLM RAG System Setup"
echo "========================"
echo ""

# Check if we're in the right directory
if [ ! -d "papers" ]; then
    echo "❌ Error: papers/ directory not found"
    echo "   Please run this script from the DLLM project root"
    exit 1
fi

echo "📦 Installing required packages..."

# Core dependencies
pip install -q requests qdrant-client PyMuPDF numpy

# For better embeddings and text processing
pip install -q sentence-transformers langchain langchain-community

# For PDF processing enhancements
pip install -q pytesseract pillow  # Optional: for OCR on scanned PDFs

echo ""
echo "✅ Dependencies installed!"
echo ""

# Create RAG directories
mkdir -p rag_data
mkdir -p rag_data/chunks
mkdir -p rag_data/embeddings

echo "📁 Created RAG data directories"
echo ""

# Check Ollama connection
echo "🔌 Testing Ollama connection..."
python3 << 'EOF'
import requests
try:
    response = requests.get("http://62.140.252.238:11434/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json()
        print("✅ Successfully connected to Ollama server!")
        print(f"   Server: 62.140.252.238:11434")
        print(f"   Models: {len(models.get('models', []))} available")
        for m in models.get('models', []):
            print(f"     - {m['name']}")
    else:
        print(f"⚠️  Ollama returned status {response.status_code}")
except Exception as e:
    print(f"❌ Could not connect to Ollama: {e}")
    print("   Make sure Ollama is running on 62.140.252.238:11434")
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run: python3 rag_system.py"
echo "  2. Use /index to index all papers"
echo "  3. Start asking questions about diffusion LLMs!"
echo ""
echo "Example queries:"
echo "  - 'What are the main advantages of diffusion LLMs over autoregressive models?'"
echo "  - 'Summarize the dLLM framework'"
echo "  - 'What are the latest speed optimization techniques?'"
