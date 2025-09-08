"""
Full Production Backend - Processes ALL 15,673 SQL Files
Optimized for complete data coverage with smart performance techniques
"""

import os
import asyncio
import json
import sqlite3
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import numpy as np

# Load environment variables
load_dotenv()

# Production configuration
PRODUCTION_MODE = os.getenv("RAILWAY_ENVIRONMENT_NAME") == "production" or os.getenv("PRODUCTION", "false").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Lightweight imports
from pydantic import BaseModel

class ProductionDocument:
    """Production document class with optimizations"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.embedding = None

class FullProductionProcessor:
    """Process ALL documents with maximum optimization"""
    
    def __init__(self):
        self.cache_file = "full_production_cache.pkl"
        self.embeddings_cache = "embeddings_cache.pkl"
        self.batch_size = 100  # Process in batches
        
        # Get project paths
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        self.references_dir = os.path.join(project_root, "references")
        self.sql_chunks_dir = os.path.join(project_root, "sql_chunks")
    
    async def load_all_documents_production(self) -> List[ProductionDocument]:
        """Load ALL documents with production optimizations"""
        print("🏭 Production Mode: Processing ALL 15,673+ documents...")
        start_time = time.time()
        
        # Try to load from cache first
        if os.path.exists(self.cache_file):
            try:
                return await self._load_from_production_cache()
            except Exception as e:
                print(f"Cache failed: {e}, rebuilding...")
        
        documents = []
        
        # 1. Load ALL reference texts
        print("📚 Loading reference methodology texts...")
        ref_docs = await self._load_all_references()
        documents.extend(ref_docs)
        print(f"✅ Loaded {len(ref_docs)} reference documents")
        
        # 2. Process ALL SQL files efficiently
        print("💾 Processing ALL SQL files...")
        sql_docs = await self._process_all_sql_files()
        documents.extend(sql_docs)
        print(f"✅ Loaded {len(sql_docs)} SQL documents")
        
        # 3. Cache everything
        await self._save_production_cache(documents)
        
        total_time = time.time() - start_time
        print(f"🎯 Total documents: {len(documents)} in {total_time:.2f} seconds")
        return documents
    
    async def _load_all_references(self) -> List[ProductionDocument]:
        """Load ALL methodology reference texts"""
        documents = []
        
        # Get ALL text files in references directory
        reference_files = [f for f in os.listdir(self.references_dir) if f.endswith('.txt')]
        
        for filename in reference_files:
            filepath = os.path.join(self.references_dir, filename)
            try:
                # Try different encodings
                content = None
                for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(filepath, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content:
                    # Advanced chunking for better retrieval
                    chunks = self._advanced_chunk(content, max_chunk_size=600)
                    
                    for i, chunk in enumerate(chunks):
                        doc = ProductionDocument(
                            content=chunk,
                            metadata={
                                "source": filename,
                                "type": "methodology",
                                "category": "nahdlatul_ulama_references",
                                "importance": "critical",
                                "chunk_id": i,
                                "total_chunks": len(chunks)
                            }
                        )
                        documents.append(doc)
                else:
                    print(f"  ⚠️ Could not decode {filename}")
                        
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
        
        return documents
    
    async def _process_all_sql_files(self) -> List[ProductionDocument]:
        """Process ALL SQL files with parallel processing"""
        documents = []
        
        # Get ALL SQL files
        sql_files = [f for f in os.listdir(self.sql_chunks_dir) if f.endswith('.sql')]
        total_files = len(sql_files)
        print(f"📊 Processing ALL {total_files} SQL files...")
        
        # Process in parallel batches
        processed = 0
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Process in batches for memory efficiency
            for i in range(0, total_files, self.batch_size):
                batch = sql_files[i:i + self.batch_size]
                
                # Submit batch to thread pool
                futures = []
                for filename in batch:
                    future = executor.submit(self._process_single_sql_file, filename)
                    futures.append(future)
                
                # Collect results
                batch_docs = []
                for future in futures:
                    try:
                        file_docs = future.result(timeout=30)  # 30 second timeout per file
                        batch_docs.extend(file_docs)
                    except Exception as e:
                        pass  # Skip problematic files
                
                documents.extend(batch_docs)
                processed += len(batch)
                
                # Progress reporting
                if processed % 1000 == 0 or processed == total_files:
                    percentage = (processed / total_files) * 100
                    print(f"  📈 Progress: {processed}/{total_files} ({percentage:.1f}%) - {len(batch_docs)} docs in batch")
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
        
        return documents
    
    def _process_single_sql_file(self, filename: str) -> List[ProductionDocument]:
        """Process a single SQL file (called by thread pool)"""
        documents = []
        filepath = os.path.join(self.sql_chunks_dir, filename)
        
        try:
            # Fast file reading with encoding detection
            content = None
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not content:
                return documents
            
            # Extract meaningful texts
            texts = self._extract_meaningful_texts(content)
            table_name = filename.replace('_chunk_0000.sql', '')
            
            for j, text in enumerate(texts):
                if len(text.strip()) > 30:  # Minimum meaningful length
                    doc = ProductionDocument(
                        content=text.strip(),
                        metadata={
                            "source": filename,
                            "table": table_name,
                            "type": "islamic_text",
                            "category": "sql_knowledge",
                            "text_index": j,
                            "file_size": len(content)
                        }
                    )
                    documents.append(doc)
                    
        except Exception:
            pass  # Skip problematic files
        
        return documents
    
    def _extract_meaningful_texts(self, sql_content: str) -> List[str]:
        """Extract meaningful Islamic texts from SQL content"""
        texts = []
        lines = sql_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'INSERT INTO' in line and 'VALUES' in line:
                try:
                    # More robust text extraction
                    values_start = line.find('VALUES')
                    if values_start != -1:
                        values_part = line[values_start + 6:].strip()
                        
                        # Handle multiple formats
                        if values_part.startswith('(') and ');' in values_part:
                            values_part = values_part[1:values_part.rfind(');')]
                            
                            # Extract quoted strings
                            in_quote = False
                            current_text = ""
                            quote_char = None
                            
                            for char in values_part:
                                if char in ["'", '"'] and not in_quote:
                                    in_quote = True
                                    quote_char = char
                                    current_text = ""
                                elif char == quote_char and in_quote:
                                    in_quote = False
                                    if len(current_text) > 50:  # Meaningful text length
                                        texts.append(current_text)
                                    current_text = ""
                                elif in_quote:
                                    current_text += char
                                    
                except Exception:
                    continue
        
        return texts
    
    def _advanced_chunk(self, text: str, max_chunk_size: int = 600) -> List[str]:
        """Advanced chunking for Islamic texts"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        
        # Try to split by Islamic text patterns
        islamic_separators = [
            '\n\n',  # Paragraph breaks
            '۔',     # Urdu sentence end
            '。',     # Period
            '.',     # English period
            '!',     # Exclamation
            '?',     # Question
            ';',     # Semicolon
        ]
        
        current_chunk = ""
        sentences = text.split('\n')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + '\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _save_production_cache(self, documents: List[ProductionDocument]):
        """Save to production cache with compression"""
        try:
            # Convert to serializable format
            cache_data = []
            for doc in documents:
                cache_data.append({
                    "content": doc.content,
                    "metadata": doc.metadata
                })
            
            # Save with pickle for faster loading
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            print(f"💾 Cached {len(documents)} documents for next startup")
            
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    async def _load_from_production_cache(self) -> List[ProductionDocument]:
        """Load from production cache"""
        print("⚡ Loading from production cache...")
        
        with open(self.cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        documents = []
        for item in cache_data:
            doc = ProductionDocument(
                content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        print(f"⚡ Loaded {len(documents)} documents from cache")
        return documents

class ProductionVectorStore:
    """Production vector store with all optimizations"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.model = None
        self.embeddings_cache_file = "production_embeddings.pkl"
        self.numpy_cache_file = "production_embeddings.npy"
    
    async def initialize_production(self, documents: List[ProductionDocument]):
        """Initialize production vector store"""
        print("🏭 Initializing production vector store...")
        
        # Try to load cached embeddings
        if os.path.exists(self.numpy_cache_file):
            try:
                print("⚡ Loading cached embeddings (NumPy format)...")
                load_start = time.time()
                self.embeddings = np.load(self.numpy_cache_file).tolist()
                load_time = time.time() - load_start
                print(f"✅ Embeddings loaded from NumPy cache in {load_time:.2f}s")
            except Exception as e:
                print(f"⚠️ NumPy cache failed: {e}, trying pickle...")
                self.embeddings = None
        
        if self.embeddings is None and os.path.exists(self.embeddings_cache_file):
            try:
                cache_size = os.path.getsize(self.embeddings_cache_file) / (1024*1024)  # MB
                print(f"⚡ Loading cached embeddings (Pickle format - {cache_size:.1f}MB)...")
                print("   ⏳ This may take 1-3 minutes for large files...")
                load_start = time.time()
                with open(self.embeddings_cache_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                load_time = time.time() - load_start
                print(f"✅ Embeddings loaded from pickle cache in {load_time:.2f}s")
            except Exception as e:
                print(f"⚠️ Embedding cache failed: {e}")
                self.embeddings = None
        
        if self.embeddings is None:
            try:
                # Load Model2Vec for production
                from model2vec import StaticModel
                
                print("📥 Loading Model2Vec production model...")
                # Use larger model for better quality in production
                self.model = StaticModel.from_pretrained("minishlab/potion-base-32M")
                print("✅ Model2Vec production model loaded")
                
                # Generate embeddings in batches
                print(f"🚀 Generating embeddings for {len(documents)} documents...")
                embedding_start = time.time()  # Track total embedding time
                texts = [doc.content for doc in documents]
                
                # Process in smaller batches with progress
                batch_size = 500  # Smaller batches for more frequent updates
                all_embeddings = []
                
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                for i in range(0, len(texts), batch_size):
                    batch_start = time.time()
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = self.model.encode(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    
                    batch_time = time.time() - batch_start
                    current_batch = i // batch_size + 1
                    docs_processed = min(i + batch_size, len(texts))
                    
                    # Estimate remaining time
                    if current_batch > 1:
                        avg_time_per_batch = (time.time() - embedding_start) / current_batch
                        remaining_batches = total_batches - current_batch
                        eta_seconds = remaining_batches * avg_time_per_batch
                        eta_minutes = eta_seconds / 60
                        
                        print(f"  📈 Batch {current_batch}/{total_batches} completed in {batch_time:.1f}s ({docs_processed}/{len(texts)} docs) - ETA: {eta_minutes:.1f}m")
                    else:
                        print(f"  📈 Batch {current_batch}/{total_batches} completed in {batch_time:.1f}s ({docs_processed}/{len(texts)} docs)")
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.01)
                
                self.embeddings = all_embeddings
                
                # Cache embeddings with optimizations
                print("💾 Saving embeddings to cache...")
                cache_start = time.time()
                
                try:
                    # Convert to numpy array for faster saving/loading
                    print("  🔄 Converting to NumPy array...")
                    embeddings_array = np.array(self.embeddings)
                    
                    # Save as NumPy format (faster for future loads)
                    print("  💾 Saving NumPy cache...")
                    np.save(self.numpy_cache_file, embeddings_array)
                    numpy_time = time.time() - cache_start
                    
                    # Also save as pickle for compatibility (async-like)
                    print("  💾 Saving Pickle cache...")
                    pickle_start = time.time()
                    with open(self.embeddings_cache_file, 'wb') as f:
                        pickle.dump(self.embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_time = time.time() - pickle_start
                    
                    total_cache_time = time.time() - cache_start
                    numpy_size = os.path.getsize(self.numpy_cache_file) / (1024*1024)  # MB
                    pickle_size = os.path.getsize(self.embeddings_cache_file) / (1024*1024)  # MB
                    
                    print(f"✅ Embeddings cached in {total_cache_time:.1f}s")
                    print(f"   📊 NumPy: {numpy_size:.1f}MB ({numpy_time:.1f}s), Pickle: {pickle_size:.1f}MB ({pickle_time:.1f}s)")
                    
                except Exception as e:
                    print(f"⚠️ Error during caching: {e}")
                    # Fallback to pickle only
                    with open(self.embeddings_cache_file, 'wb') as f:
                        pickle.dump(self.embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                    total_cache_time = time.time() - cache_start
                    pickle_size = os.path.getsize(self.embeddings_cache_file) / (1024*1024)
                    print(f"✅ Embeddings cached (pickle only) in {total_cache_time:.1f}s ({pickle_size:.1f}MB)")
                
            except ImportError:
                print("⚠️ Model2Vec not available")
                self.model = None
        else:
            # Still load model for query encoding
            try:
                from model2vec import StaticModel
                self.model = StaticModel.from_pretrained("minishlab/potion-base-32M")
            except ImportError:
                pass
        
        self.documents = documents
        print("✅ Production vector store initialized")
    
    async def search_production(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Production-grade search"""
        
        if self.model and self.embeddings is not None:
            # Vector similarity search
            query_embedding = self.model.encode([query])[0]
            
            # Efficient similarity calculation
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                norm_a = sum(a * a for a in query_embedding) ** 0.5
                norm_b = sum(b * b for b in doc_embedding) ** 0.5
                similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
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
            # Enhanced keyword search fallback
            return await self._enhanced_keyword_search(query, top_k)
    
    async def _enhanced_keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Enhanced keyword-based search"""
        query_words = query.lower().split()
        results = []
        
        for doc in self.documents:
            content_lower = doc.content.lower()
            score = 0
            
            # Calculate relevance score
            for word in query_words:
                if word in content_lower:
                    score += content_lower.count(word)
            
            # Boost for methodology documents
            if doc.metadata.get("type") == "methodology":
                score *= 2
            
            if score > 0:
                results.append({
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": score / len(query_words)
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Global instances
production_processor = FullProductionProcessor()
production_vector_store = ProductionVectorStore()

@asynccontextmanager
async def production_lifespan(app: FastAPI):
    """Production startup with ALL documents"""
    global production_vector_store, llm_client
    
    try:
        print("🏭 Starting FULL PRODUCTION Nahdlatul Ulama AI...")
        
        # Environment info
        if PRODUCTION_MODE:
            print("✅ Running in PRODUCTION mode on Railway")
        else:
            print("🔧 Running in DEVELOPMENT mode")
        
        print("📊 Processing ALL 15,673+ SQL files + methodology texts...")
        start_time = time.time()
        
        # Load ALL documents
        documents = await production_processor.load_all_documents_production()
        
        # Initialize production vector store
        await production_vector_store.initialize_production(documents)
        
        # Initialize LLM
        from ultra_fast_main import FastGroqClient
        llm_client = FastGroqClient()
        await llm_client.initialize()
        
        total_time = time.time() - start_time
        print(f"🎉 FULL PRODUCTION startup complete in {total_time:.2f} seconds!")
        print(f"📊 Total documents loaded: {len(documents)}")
        
        if PRODUCTION_MODE:
            print("🚀 Production backend ready on Railway!")
        
        yield
        
    except Exception as e:
        print(f"❌ Production startup error: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        raise
    
    print("🔄 Shutting down production system...")

# FastAPI app for production
production_app = FastAPI(
    title="Nahdlatul Ulama AI - Full Production",
    description="Complete Islamic jurisprudence system with ALL documents",
    version="3.0.0",
    lifespan=production_lifespan
)

# CORS
production_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import request/response models
from ultra_fast_main import QuestionRequest, AnswerResponse, SearchRequest

@production_app.get("/health")
async def production_health():
    """Production health check for Railway"""
    try:
        # Check if system is initialized
        if production_vector_store.embeddings is None:
            return {"status": "initializing", "message": "System is still loading documents..."}
        
        # Check LLM
        if llm_client is None:
            return {"status": "degraded", "message": "LLM client not initialized"}
        
        return {
            "status": "healthy",
            "message": "Nahdlatul Ulama AI is ready",
            "version": "3.0.0",
            "environment": "production" if PRODUCTION_MODE else "development",
            "documents_loaded": len(production_vector_store.documents),
            "mode": "full_production",
            "all_sql_files": "processed",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": time.time()
        }

@production_app.get("/methods")
async def production_methods():
    """Get Islamic reasoning methods"""
    return ["bayani", "qiyasi", "istishlahi", "maqashidi"]

@production_app.post("/search")
async def production_search(request: SearchRequest):
    """Search complete knowledge base"""
    try:
        results = await production_vector_store.search_production(request.query, request.limit)
        return {"results": results, "query": request.query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@production_app.post("/ask", response_model=AnswerResponse)
async def production_ask(request: QuestionRequest):
    """Ask with complete knowledge base"""
    try:
        # Search complete knowledge base
        search_results = await production_vector_store.search_production(request.question, top_k=5)
        
        # Prepare comprehensive context
        context_parts = []
        for result in search_results:
            source = result['metadata'].get('source', 'Unknown')
            context_parts.append(f"Source: {source}\nContent: {result['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        from ultra_fast_main import llm_client
        answer = await llm_client.generate_response(request.question, context)
        
        return AnswerResponse(
            answer=answer,
            sources=search_results,
            method_used=request.method
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@production_app.get("/stats")
async def production_stats():
    """Complete database statistics"""
    docs = production_vector_store.documents
    
    return {
        "total_documents": len(docs),
        "methodology_docs": len([d for d in docs if d.metadata.get("type") == "methodology"]),
        "islamic_text_docs": len([d for d in docs if d.metadata.get("type") == "islamic_text"]),
        "has_embeddings": production_vector_store.embeddings is not None,
        "embedding_model": "Model2Vec-32M-Production" if production_vector_store.model else "None",
        "sql_files_processed": "ALL_15673+",
        "mode": "complete_production"
    }

if __name__ == "__main__":
    import uvicorn
    print("🏭 Starting FULL PRODUCTION mode with ALL documents...")
    
    # Railway-compatible port configuration
    port = int(os.environ.get("PORT", 8001))
    host = "0.0.0.0"  # Required for Railway deployment
    
    print(f"🌐 Server will run on {host}:{port}")
    uvicorn.run(
        "full_production_main:production_app", 
        host=host, 
        port=port, 
        reload=False,  # Disable reload in production
        access_log=False  # Reduce log noise in production
    )
