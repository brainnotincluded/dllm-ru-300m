# DLLM Research RAG System
# Ollama Server: 62.140.252.238

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Check available packages
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    
try:
    from qdrant_client import QdrantClient
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class DLLMRAGSystem:
    """RAG system for Diffusion LLM research papers with Ollama integration"""
    
    def __init__(self, ollama_host="http://62.140.252.238:11434", qdrant_path="./storage/qdrant"):
        self.ollama_host = ollama_host
        self.qdrant_path = qdrant_path
        self.qdrant_client = None
        self.collection_name = "dllm_papers"
        self.model_name = "llama3.2"  # Default model
        
        print(f"🔧 Initializing DLLM RAG System...")
        print(f"   Ollama Server: {ollama_host}")
        print(f"   Qdrant Path: {qdrant_path}")
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are available"""
        deps = {
            "requests": HAS_REQUESTS,
            "qdrant-client": HAS_QDRANT,
            "PyMuPDF": HAS_PDF,
            "numpy": HAS_NUMPY
        }
        return deps
    
    def check_ollama_connection(self) -> bool:
        """Test connection to Ollama server"""
        if not HAS_REQUESTS:
            print("❌ requests library not installed")
            return False
            
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json()
                print(f"✅ Ollama connection successful")
                print(f"   Available models: {', '.join([m['name'] for m in models.get('models', [])])}")
                return True
            else:
                print(f"❌ Ollama returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List models available on Ollama server"""
        if not HAS_REQUESTS:
            return []
            
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except:
            pass
        return []
    
    def init_qdrant(self) -> bool:
        """Initialize Qdrant connection"""
        if not HAS_QDRANT:
            print("❌ qdrant-client not installed")
            return False
            
        try:
            self.qdrant_client = QdrantClient(path=self.qdrant_path)
            collections = self.qdrant_client.get_collections()
            print(f"✅ Qdrant connected")
            print(f"   Collections: {[c.name for c in collections.collections]}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            return False
    
    def index_papers(self, papers_dir: str = "./papers") -> bool:
        """Index all PDF papers into Qdrant"""
        print(f"\n📚 Indexing papers from {papers_dir}...")
        
        if not HAS_PDF:
            print("❌ PyMuPDF not installed. Run: pip install PyMuPDF")
            return False
            
        if not self.qdrant_client:
            if not self.init_qdrant():
                return False
        
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            print(f"❌ Papers directory not found: {papers_dir}")
            return False
        
        pdf_files = list(papers_path.glob("*.pdf"))
        print(f"   Found {len(pdf_files)} PDF files")
        
        # TODO: Implement actual indexing
        # For now, just check we can read PDFs
        for pdf_file in pdf_files[:3]:  # Test first 3
            try:
                doc = fitz.open(pdf_file)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                print(f"   ✓ {pdf_file.name}: {len(text)} chars")
            except Exception as e:
                print(f"   ✗ {pdf_file.name}: {e}")
        
        print(f"\n💡 To complete indexing, install additional dependencies:")
        print(f"   pip install sentence-transformers langchain")
        return True
    
    def query_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """Send query to Ollama and get response"""
        if not HAS_REQUESTS:
            return "Error: requests library not installed"
        
        model = model or self.model_name
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response')
            else:
                return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    
    def rag_query(self, query: str, model: Optional[str] = None) -> str:
        """Perform RAG query: retrieve relevant papers and generate answer"""
        print(f"\n🔍 RAG Query: {query}")
        
        # Step 1: Retrieve relevant documents (placeholder)
        context = self.retrieve_context(query)
        
        # Step 2: Generate answer with Ollama
        prompt = self.build_rag_prompt(query, context)
        
        print(f"   Querying Ollama ({model or self.model_name})...")
        answer = self.query_ollama(prompt, model)
        
        return answer
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from vector DB"""
        # Placeholder - would use actual vector search
        context = """
Based on the diffusion language model research collection, here are relevant papers:
- LLaDA: Large Language Diffusion Models
- Simple and Effective Masked Diffusion Language Models
- Scaling Beyond Masked Diffusion Language Models
- dLLM: Simple Diffusion Language Modeling framework
"""
        return context
    
    def build_rag_prompt(self, query: str, context: str) -> str:
        """Build RAG prompt with context"""
        prompt = f"""You are a research assistant specializing in Diffusion Language Models (DLLMs). 
Use the following context from research papers to answer the question.

Context:
{context}

Question: {query}

Please provide a detailed, accurate answer based on the research papers. If the context doesn't contain enough information, say so."""
        return prompt
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "="*60)
        print("🤖 DLLM Research RAG System - Interactive Mode")
        print("="*60)
        print(f"Ollama Server: {self.ollama_host}")
        print(f"Model: {self.model_name}")
        print("Commands: /quit, /models, /index, /help")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("\n❓ Query: ").strip()
                
                if not query:
                    continue
                    
                if query == "/quit":
                    print("👋 Goodbye!")
                    break
                elif query == "/models":
                    models = self.list_available_models()
                    print(f"Available models: {', '.join(models) if models else 'None'}")
                elif query == "/index":
                    self.index_papers()
                elif query == "/help":
                    print("Commands:")
                    print("  /quit  - Exit")
                    print("  /models - List Ollama models")
                    print("  /index - Index papers")
                    print("  /help  - Show this help")
                else:
                    answer = self.rag_query(query)
                    print(f"\n💡 Answer:\n{answer}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    rag = DLLMRAGSystem()
    
    # Check dependencies
    print("\n📦 Checking dependencies:")
    deps = rag.check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")
    
    # Check Ollama connection
    print("\n🔌 Checking Ollama connection:")
    if rag.check_ollama_connection():
        print("\n✅ Ollama server is ready!")
    else:
        print("\n⚠️  Could not connect to Ollama server")
        print("   Make sure Ollama is running on 62.140.252.238:11434")
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
