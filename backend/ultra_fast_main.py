"""
Ultra-Fast Islamic AI Backend using Model2Vec + Optimized Data Loading
This implementation reduces startup time from 30 minutes to under 30 seconds
"""

import os
import asyncio
import random
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import json
import sqlite3
from pathlib import Path

# Load environment variables
load_dotenv()

# Lightweight imports - no heavy ML libraries at startup
from pydantic import BaseModel

# Global variables
vector_index = None
llm_client = None
documents_cache = {}

class OptimizedDocument:
    """Lightweight document class"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.embedding = None  # Will be computed lazily

class UltraFastProcessor:
    """Ultra-fast data processor with smart sampling and caching"""
    
    def __init__(self, max_docs: int = 1000):
        self.max_docs = max_docs
        self.cache_file = "fast_cache.json"
        
        # Get project paths
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        self.references_dir = os.path.join(project_root, "references")
        self.sql_chunks_dir = os.path.join(project_root, "sql_chunks")
    
    async def load_documents_ultra_fast(self) -> List[OptimizedDocument]:
        """Load documents with aggressive optimization"""
        print("🚀 Ultra-fast document loading...")
        
        # Try to load from cache first
        if os.path.exists(self.cache_file):
            try:
                return await self._load_from_cache()
            except Exception as e:
                print(f"Cache failed: {e}, rebuilding...")
        
        documents = []
        
        # 1. Load reference texts (small but important)
        ref_docs = await self._load_references()
        documents.extend(ref_docs)
        print(f"📚 Loaded {len(ref_docs)} reference documents")
        
        # 2. Smart SQL sampling with importance weighting
        sql_docs = await self._smart_sql_sample()
        documents.extend(sql_docs)
        print(f"💾 Loaded {len(sql_docs)} SQL documents")
        
        # 3. Cache results for next time
        await self._save_to_cache(documents)
        
        print(f"✅ Total documents: {len(documents)}")
        return documents
    
    async def _load_references(self) -> List[OptimizedDocument]:
        """Load methodology reference texts"""
        documents = []
        
        reference_files = [
            "Kerangka Fiqih Ontologis Nahdlatul Ulama_.txt",
            "Metode Istinbath Al Ahkam dalam NU.txt", 
            "Metode Istinbath Maqashidi.txt",
            "Metode Taqrir Jamai.txt"
        ]
        
        for filename in reference_files:
            filepath = os.path.join(self.references_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split into smaller chunks for better retrieval
                    chunks = self._smart_chunk(content, max_chunk_size=800)
                    
                    for i, chunk in enumerate(chunks):
                        doc = OptimizedDocument(
                            content=chunk,
                            metadata={
                                "source": filename,
                                "type": "methodology",
                                "category": "nahdlatul_ulama_references",
                                "importance": "high",
                                "chunk_id": i
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
        
        return documents
    
    async def _smart_sql_sample(self) -> List[OptimizedDocument]:
        """Intelligent SQL sampling with importance weighting"""
        documents = []
        
        # Get all SQL files
        sql_files = [f for f in os.listdir(self.sql_chunks_dir) if f.endswith('.sql')]
        total_files = len(sql_files)
        print(f"📊 Found {total_files} SQL files")
        
        # Smart sampling strategy
        sample_size = min(500, total_files)  # Cap at 500 for speed
        
        # Prioritize certain file patterns (if any)
        priority_patterns = ['b1_', 'b10_', 'b100_']  # These might be more important
        priority_files = []
        regular_files = []
        
        for file in sql_files:
            if any(file.startswith(pattern) for pattern in priority_patterns):
                priority_files.append(file)
            else:
                regular_files.append(file)
        
        # Take all priority files and random sample of regular files
        sample_files = priority_files.copy()
        remaining_slots = sample_size - len(priority_files)
        
        if remaining_slots > 0:
            sample_files.extend(random.sample(regular_files, min(remaining_slots, len(regular_files))))
        
        print(f"📝 Processing {len(sample_files)} SQL files...")
        
        # Process with speed optimizations
        for i, filename in enumerate(sample_files):
            if i % 50 == 0:
                print(f"  📈 Progress: {i}/{len(sample_files)}...")
                
            filepath = os.path.join(self.sql_chunks_dir, filename)
            try:
                # Fast text extraction
                texts = await self._extract_texts_fast(filepath)
                
                for j, text in enumerate(texts[:2]):  # Limit to 2 texts per file
                    if len(text.strip()) > 50:  # Skip short texts
                        doc = OptimizedDocument(
                            content=text.strip(),
                            metadata={
                                "source": filename,
                                "table": filename.replace('_chunk_0000.sql', ''),
                                "type": "islamic_text",
                                "category": "sql_knowledge",
                                "text_index": j
                            }
                        )
                        documents.append(doc)
                        
            except Exception:
                continue  # Skip problematic files
        
        return documents
    
    async def _extract_texts_fast(self, filepath: str) -> List[str]:
        """Fast text extraction from SQL files"""
        texts = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple regex-like extraction (faster than parsing)
            lines = content.split('\n')
            for line in lines[:10]:  # Only check first 10 lines for speed
                if 'INSERT INTO' in line and 'VALUES' in line:
                    # Extract text between quotes
                    start = line.find("'")
                    if start != -1:
                        end = line.rfind("'")
                        if end > start:
                            text = line[start+1:end]
                            if len(text) > 100:  # Only meaningful text
                                texts.append(text)
                                
        except Exception:
            pass
            
        return texts
    
    def _smart_chunk(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """Smart text chunking optimized for Arabic/Islamic content"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _save_to_cache(self, documents: List[OptimizedDocument]):
        """Save processed documents to cache"""
        try:
            cache_data = []
            for doc in documents:
                cache_data.append({
                    "content": doc.content,
                    "metadata": doc.metadata
                })
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            print("💾 Documents cached for next startup")
            
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    async def _load_from_cache(self) -> List[OptimizedDocument]:
        """Load documents from cache"""
        print("⚡ Loading from cache...")
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        documents = []
        for item in cache_data:
            doc = OptimizedDocument(
                content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        print(f"⚡ Loaded {len(documents)} documents from cache")
        return documents

class LightweightVectorStore:
    """Simple in-memory vector store using basic similarity"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.model = None
    
    async def initialize(self, documents: List[OptimizedDocument]):
        """Initialize with lightweight embeddings"""
        print("🔍 Initializing lightweight vector store...")
        
        try:
            # Try to import model2vec for ultra-fast embeddings
            from model2vec import StaticModel
            
            # Load tiny model (8MB instead of 400MB+)
            print("📥 Loading Model2Vec (8MB)...")
            self.model = StaticModel.from_pretrained("minishlab/potion-base-8M")
            print("✅ Model2Vec loaded")
            
            # Generate embeddings
            texts = [doc.content for doc in documents]
            print(f"🚀 Generating embeddings for {len(texts)} documents...")
            self.embeddings = self.model.encode(texts)
            print("✅ Embeddings generated")
            
        except ImportError:
            print("⚠️ Model2Vec not available, falling back to simple search")
            self.model = None
        
        self.documents = documents
        print("✅ Vector store initialized")
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        if self.model and self.embeddings is not None:
            # Vector similarity search
            query_embedding = self.model.encode([query])
            
            # Simple cosine similarity (numpy-free)
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                # Dot product similarity (good enough for fast search)
                similarity = sum(a * b for a, b in zip(query_embedding[0], doc_embedding))
                similarities.append((similarity, i))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            results = []
            for similarity, idx in similarities[:top_k]:
                results.append({
                    "content": self.documents[idx].content,
                    "metadata": self.documents[idx].metadata,
                    "score": similarity
                })
            
            return results
        else:
            # Fallback to keyword search
            query_lower = query.lower()
            results = []
            
            for doc in self.documents:
                if query_lower in doc.content.lower():
                    results.append({
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": 0.8  # Simple score
                    })
                    
                    if len(results) >= top_k:
                        break
            
            return results

class FastGroqClient:
    """Lightweight Groq client wrapper"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
    
    async def initialize(self):
        """Initialize Groq client"""
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                print("✅ Groq LLM initialized")
            except ImportError:
                print("⚠️ Groq library not available")
        else:
            print("⚠️ No Groq API key found")
    
    async def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Groq"""
        if not self.client:
            return "LLM not available. Please check your API configuration."
        
        try:
            full_prompt = f"""Context: {context}\n\nQuestion: {prompt}\n\nPlease provide a response based on Nahdlatul Ulama Islamic jurisprudence methodology:"""
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Global instances
processor = UltraFastProcessor()
vector_store = LightweightVectorStore()
llm_client = FastGroqClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ultra-fast startup"""
    global vector_store, llm_client
    
    print("🚀 Starting Ultra-Fast Nahdlatul Ulama AI...")
    start_time = asyncio.get_event_loop().time()
    
    # Load documents ultra-fast
    documents = await processor.load_documents_ultra_fast()
    
    # Initialize vector store
    await vector_store.initialize(documents)
    
    # Initialize LLM
    await llm_client.initialize()
    
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"✅ Startup complete in {total_time:.2f} seconds!")
    
    yield
    
    print("🔄 Shutting down...")

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    method: str = "bayani"
    context: str = ""

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    method_used: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

# FastAPI app
app = FastAPI(
    title="Ultra-Fast Nahdlatul Ulama AI",
    description="Lightning-fast Islamic jurisprudence assistant",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "documents_loaded": len(vector_store.documents),
        "mode": "ultra-fast"
    }

@app.get("/methods")
async def get_methods():
    """Get available Islamic reasoning methods"""
    return ["bayani", "qiyasi", "istishlahi", "maqashidi"]

@app.post("/search")
async def search_knowledge(request: SearchRequest):
    """Search Islamic knowledge base"""
    try:
        results = await vector_store.search(request.query, request.limit)
        return {"results": results, "query": request.query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask Islamic jurisprudence question"""
    try:
        # Search for relevant context
        search_results = await vector_store.search(request.question, top_k=3)
        
        # Prepare context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(f"Source: {result['metadata'].get('source', 'Unknown')}\nContent: {result['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response
        answer = await llm_client.generate_response(request.question, context)
        
        return AnswerResponse(
            answer=answer,
            sources=search_results,
            method_used=request.method
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    return {
        "total_documents": len(vector_store.documents),
        "methodology_docs": len([d for d in vector_store.documents if d.metadata.get("type") == "methodology"]),
        "islamic_text_docs": len([d for d in vector_store.documents if d.metadata.get("type") == "islamic_text"]),
        "has_embeddings": vector_store.embeddings is not None,
        "embedding_model": "Model2Vec-8M" if vector_store.model else "None"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ultra_fast_main:app", host="0.0.0.0", port=8000, reload=False)
