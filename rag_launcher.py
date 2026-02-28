#!/usr/bin/env python3
"""
DLLM RAG System Launcher
Quick start script for the RAG system
"""

import sys
import subprocess
from pathlib import Path

OLLAMA_HOST = "http://62.140.252.238:11434"

def check_venv():
    """Check if we're in the virtual environment"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    return in_venv

def activate_venv():
    """Activate virtual environment"""
    venv_path = Path(__file__).parent / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found. Run: python3 -m venv venv")
        return False
    
    # Set up environment
    import os
    os.environ["PATH"] = str(venv_path / "bin") + ":" + os.environ.get("PATH", "")
    return True

def main():
    print("🚀 DLLM Research RAG System")
    print("=" * 60)
    
    # Check/activate venv
    if not check_venv():
        print("⚠️  Not in virtual environment, attempting to activate...")
        if not activate_venv():
            print("Please activate manually: source venv/bin/activate")
            sys.exit(1)
    
    print(f"✓ Environment ready")
    print(f"✓ Ollama Server: {OLLAMA_HOST}")
    print("")
    
    # Check if papers are indexed
    from qdrant_client import QdrantClient
    
    try:
        client = QdrantClient(path="./storage/qdrant")
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "dllm_papers_chunks" not in collection_names:
            print("📚 Papers not yet indexed!")
            print("")
            print("To start, run one of these commands:")
            print("  1. python3 rag_complete.py --index          # Index all 112 papers")
            print("  2. python3 rag_complete.py --interactive    # Start interactive mode")
            print("")
            print("Or run with a specific query:")
            print('  python3 rag_complete.py --query "What is dLLM framework?"')
        else:
            print("✓ Papers already indexed")
            print("")
            print("Ready to use! Run:")
            print("  python3 rag_complete.py --interactive")
            print("")
            print("Or query directly:")
            print('  python3 rag_complete.py --query "What are the main advantages of diffusion LLMs?"')
    except Exception as e:
        print(f"⚠️  Could not check indexing status: {e}")
        print("")
        print("Try running: python3 rag_complete.py --index")

if __name__ == "__main__":
    main()
