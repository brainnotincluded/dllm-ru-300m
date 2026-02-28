#!/usr/bin/env python3
"""
DLLM Research RAG - Complete Implementation
Indexes all 112 papers and provides RAG queries via Ollama
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import requests

# Try to import optional dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class PaperChunk:
    """Represents a chunk of a research paper"""
    paper_id: str
    paper_title: str
    chunk_id: int
    text: str
    page_number: int
    embedding: Optional[List[float]] = None


class DLLMPaperIndexer:
    """Indexes DLLM papers into Qdrant vector database"""
    
    def __init__(self, qdrant_path: str = "./rag_data/qdrant"):
        self.qdrant_path = qdrant_path
        self.client = None
        self.collection_name = "dllm_papers_chunks"
        self.embedding_model = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
    def init(self) -> bool:
        """Initialize indexer"""
        if not HAS_QDRANT:
            print("❌ qdrant-client not installed")
            return False
        if not HAS_SENTENCE_TRANSFORMERS:
            print("❌ sentence-transformers not installed")
            return False
        if not HAS_PDF:
            print("❌ PyMuPDF not installed")
            return False
            
        # Connect to Qdrant
        try:
            self.client = QdrantClient(path=self.qdrant_path)
            print(f"✅ Connected to Qdrant at {self.qdrant_path}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            return False
        
        # Load embedding model
        try:
            print("🤖 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            return False
        
        # Create collection if needed
        self._ensure_collection()
        return True
    
    def _ensure_collection(self):
        """Ensure the collection exists"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"📁 Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Collection created")
            else:
                print(f"📁 Using existing collection: {self.collection_name}")
        except Exception as e:
            print(f"⚠️  Collection check warning: {e}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, str]:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            doc.close()
            
            # Extract title from filename
            title = pdf_path.stem.replace('_', ' ').replace('-', ' ')
            return title, text
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")
            return pdf_path.stem, ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for sentence boundary within last 100 chars
                for i in range(min(100, len(chunk)), 0, -1):
                    if chunk[-i] in '.!?' and (len(chunk) - i + 1) > chunk_size * 0.5:
                        chunk = chunk[:-i+1]
                        break
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
        
        return chunks
    
    def index_papers(self, papers_dir: str = "./papers") -> Dict[str, int]:
        """Index all papers in directory"""
        papers_path = Path(papers_dir)
        if not papers_path.exists():
            print(f"❌ Papers directory not found: {papers_dir}")
            return {}
        
        pdf_files = sorted(papers_path.glob("*.pdf"))
        print(f"\n📚 Found {len(pdf_files)} PDF files")
        
        stats = {"indexed": 0, "chunks": 0, "errors": 0}
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            try:
                # Extract text
                title, text = self.extract_text_from_pdf(pdf_file)
                
                if not text.strip():
                    print(f"   ⚠️  No text extracted")
                    stats["errors"] += 1
                    continue
                
                print(f"   📝 Extracted {len(text)} characters")
                
                # Create chunks
                chunks = self.chunk_text(text)
                print(f"   ✂️  Created {len(chunks)} chunks")
                
                # Generate embeddings and index
                paper_id = hashlib.md5(pdf_file.name.encode()).hexdigest()[:16]
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk_text).tolist()
                    
                    # Create point
                    point_id = f"{paper_id}_{chunk_idx}"
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "paper_id": paper_id,
                            "paper_title": title,
                            "chunk_id": chunk_idx,
                            "text": chunk_text[:1000],  # Truncate for storage
                            "filename": pdf_file.name,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks)
                        }
                    )
                    
                    # Upsert to Qdrant
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )
                    
                    stats["chunks"] += 1
                
                stats["indexed"] += 1
                print(f"   ✅ Indexed {len(chunks)} chunks")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                stats["errors"] += 1
        
        print(f"\n{'='*60}")
        print(f"Indexing complete!")
        print(f"   Papers indexed: {stats['indexed']}")
        print(f"   Total chunks: {stats['chunks']}")
        print(f"   Errors: {stats['errors']}")
        print(f"{'='*60}")
        
        return stats
    
    def get_stats(self) -> Dict:
        """Get indexing statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}


class DLLMRAGQuery:
    """RAG query system using Ollama"""
    
    def __init__(self, ollama_host: str = "http://62.140.252.238:11434", qdrant_path: str = "./rag_data/qdrant"):
        self.ollama_host = ollama_host
        self.qdrant_client = None
        self.embedding_model = None
        self.collection_name = "dllm_papers_chunks"
        self.default_model = "llama3.2"
        
        if HAS_QDRANT:
            self.qdrant_client = QdrantClient(path=qdrant_path)
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant chunks from Qdrant"""
        if not self.qdrant_client or not self.embedding_model:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                "paper_title": r.payload.get("paper_title", "Unknown"),
                "text": r.payload.get("text", ""),
                "filename": r.payload.get("filename", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "score": r.score
            }
            for r in results
        ]
    
    def query_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """Query Ollama server"""
        model = model or self.default_model
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response')
            else:
                return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
    
    def query(self, question: str, model: Optional[str] = None, top_k: int = 5) -> Dict:
        """Perform RAG query"""
        # Retrieve context
        print(f"🔍 Retrieving relevant papers...")
        contexts = self.retrieve(question, top_k)
        
        if not contexts:
            return {
                "answer": "No relevant documents found. Please index papers first using: python3 rag_index.py",
                "sources": []
            }
        
        print(f"   Found {len(contexts)} relevant chunks")
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}: {ctx['paper_title']}]\n{ctx['text'][:500]}..."
            for i, ctx in enumerate(contexts)
        ])
        
        # Build prompt
        prompt = f"""You are a research assistant specializing in Diffusion Language Models (DLLMs).
Use the following excerpts from research papers to answer the user's question.
Be accurate and cite the source papers when possible.

CONTEXT FROM RESEARCH PAPERS:
{context_str}

USER QUESTION: {question}

Provide a detailed, accurate answer based on the research papers above. If the context doesn't contain enough information, say so."""
        
        # Query Ollama
        print(f"🤖 Querying Ollama ({model or self.default_model})...")
        answer = self.query_ollama(prompt, model)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "title": ctx["paper_title"],
                    "filename": ctx["filename"],
                    "relevance_score": round(ctx["score"], 3)
                }
                for ctx in contexts
            ]
        }


def main():
    """Main CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DLLM Research RAG System")
    parser.add_argument("--index", action="store_true", help="Index all papers")
    parser.add_argument("--query", type=str, help="Query the RAG system")
    parser.add_argument("--stats", action="store_true", help="Show indexing statistics")
    parser.add_argument("--ollama-host", default="http://62.140.252.238:11434", help="Ollama server URL")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.index:
        indexer = DLLMPaperIndexer()
        if indexer.init():
            indexer.index_papers()
    
    elif args.stats:
        indexer = DLLMPaperIndexer()
        if indexer.init():
            stats = indexer.get_stats()
            print(json.dumps(stats, indent=2))
    
    elif args.query:
        rag = DLLMRAGQuery(ollama_host=args.ollama_host)
        result = rag.query(args.query, model=args.model)
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])
        print("\n" + "="*60)
        print("SOURCES:")
        print("="*60)
        for src in result["sources"]:
            print(f"  • {src['title']}")
            print(f"    File: {src['filename']}, Score: {src['relevance_score']}")
    
    elif args.interactive:
        rag = DLLMRAGQuery(ollama_host=args.ollama_host)
        
        print("\n" + "="*60)
        print("🤖 DLLM Research RAG - Interactive Mode")
        print("="*60)
        print(f"Ollama: {args.ollama_host}")
        print(f"Model: {args.model}")
        print("Commands: /quit, /stats, /help")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("\n❓ Query: ").strip()
                
                if not query:
                    continue
                if query == "/quit":
                    break
                if query == "/stats":
                    indexer = DLLMPaperIndexer()
                    if indexer.init():
                        print(json.dumps(indexer.get_stats(), indent=2))
                    continue
                if query == "/help":
                    print("Commands: /quit, /stats, /help")
                    continue
                
                result = rag.query(query, model=args.model)
                print(f"\n💡 Answer:\n{result['answer']}")
                print(f"\n📚 Sources: {', '.join([s['title'][:40] + '...' for s in result['sources'][:3]])}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
