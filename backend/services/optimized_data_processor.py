import os
import sys
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import random

class OptimizedDataProcessor:
    """Optimized processor for large datasets with progress tracking"""
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for faster processing
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Get directories
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(backend_dir)
        self.references_dir = os.path.join(project_root, "references")
        self.sql_chunks_dir = os.path.join(project_root, "sql_chunks")
    
    async def quick_load_sample(self) -> List[Document]:
        """Load a sample for quick testing and development"""
        print(f"🚀 Quick loading sample of {self.sample_size} documents...")
        documents = []
        
        # Always load reference texts (small but important)
        ref_docs = await self.load_reference_texts()
        documents.extend(ref_docs)
        print(f"📚 Loaded {len(ref_docs)} reference documents")
        
        # Load a random sample of SQL files
        sql_docs = await self.load_sql_sample()
        documents.extend(sql_docs)
        print(f"💾 Loaded {len(sql_docs)} SQL documents")
        
        print(f"✅ Total sample documents: {len(documents)}")
        return documents
    
    async def load_reference_texts(self) -> List[Document]:
        """Load methodology reference texts (always include these)"""
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
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "type": "methodology",
                            "category": "nahdlatul_ulama_references",
                            "importance": "high"
                        }
                    )
                    
                    chunks = self.text_splitter.split_documents([doc])
                    documents.extend(chunks)
                    print(f"  ✅ {filename}: {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {filename}: {e}")
        
        return documents
    
    async def load_sql_sample(self) -> List[Document]:
        """Load a random sample of SQL files for development"""
        documents = []
        
        # Get all SQL files
        sql_files = [f for f in os.listdir(self.sql_chunks_dir) if f.endswith('.sql')]
        total_files = len(sql_files)
        print(f"📊 Found {total_files} SQL files total")
        
        # Take a random sample
        sample_files = random.sample(sql_files, min(self.sample_size, total_files))
        print(f"📝 Processing {len(sample_files)} random SQL files...")
        
        # Process with progress tracking
        for i, filename in enumerate(sample_files):
            if i % 100 == 0:
                print(f"  📈 Progress: {i}/{len(sample_files)} files...")
                
            filepath = os.path.join(self.sql_chunks_dir, filename)
            try:
                table_name = filename.replace('_chunk_0000.sql', '')
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                texts = self._extract_texts_from_sql(sql_content)
                
                for j, text in enumerate(texts[:3]):  # Limit to 3 texts per file
                    if text.strip() and len(text) > 50:  # Skip short/empty texts
                        doc = Document(
                            page_content=text.strip(),
                            metadata={
                                "source": filename,
                                "table": table_name,
                                "type": "islamic_text",
                                "category": "sql_knowledge",
                                "text_index": j
                            }
                        )
                        
                        chunks = self.text_splitter.split_documents([doc])
                        documents.extend(chunks)
                        
            except Exception as e:
                print(f"  ⚠️ Error processing {filename}: {e}")
        
        return documents
    
    def _extract_texts_from_sql(self, sql_content: str) -> List[str]:
        """Extract text values from SQL INSERT statements"""
        texts = []
        lines = sql_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('INSERT INTO') and 'VALUES' in line:
                try:
                    # Extract VALUES part
                    values_part = line.split('VALUES')[1].strip()
                    if values_part.startswith('(') and values_part.endswith(');'):
                        values_part = values_part[1:-2]  # Remove parentheses and semicolon
                        
                        # Split by comma and clean up
                        values = [v.strip().strip("'\"") for v in values_part.split(',')]
                        
                        # Look for text content (usually longer values)
                        for value in values:
                            if len(value) > 100 and not value.isdigit():
                                texts.append(value)
                                
                except Exception:
                    continue
        
        return texts
    
    async def full_load_with_progress(self) -> List[Document]:
        """Load all data with detailed progress tracking (for production)"""
        print("🏗️ Starting full data load...")
        documents = []
        
        # Load references first
        ref_docs = await self.load_reference_texts()
        documents.extend(ref_docs)
        
        # Load all SQL files with progress
        sql_files = [f for f in os.listdir(self.sql_chunks_dir) if f.endswith('.sql')]
        total_files = len(sql_files)
        
        print(f"📊 Processing {total_files} SQL files in batches...")
        
        batch_size = 50  # Smaller batches for better progress tracking
        processed = 0
        
        for i in range(0, total_files, batch_size):
            batch = sql_files[i:i + batch_size]
            batch_docs = []
            
            for filename in batch:
                filepath = os.path.join(self.sql_chunks_dir, filename)
                try:
                    table_name = filename.replace('_chunk_0000.sql', '')
                    
                    with open(filepath, 'r', encoding='utf-8') as f:
                        sql_content = f.read()
                    
                    texts = self._extract_texts_from_sql(sql_content)
                    
                    for j, text in enumerate(texts):
                        if text.strip() and len(text) > 30:
                            doc = Document(
                                page_content=text.strip(),
                                metadata={
                                    "source": filename,
                                    "table": table_name,
                                    "type": "islamic_text",
                                    "category": "sql_knowledge",
                                    "text_index": j
                                }
                            )
                            
                            chunks = self.text_splitter.split_documents([doc])
                            batch_docs.extend(chunks)
                            
                except Exception as e:
                    pass  # Skip problematic files
            
            documents.extend(batch_docs)
            processed += len(batch)
            
            print(f"📈 Progress: {processed}/{total_files} files ({processed/total_files*100:.1f}%) - {len(batch_docs)} docs in batch")
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        print(f"✅ Full load complete: {len(documents)} total documents")
        return documents


# Quick test function
async def quick_test():
    """Quick test with sample data"""
    processor = OptimizedDataProcessor(sample_size=200)
    documents = await processor.quick_load_sample()
    print(f"🎯 Quick test complete: {len(documents)} documents ready for embedding")
    return documents

if __name__ == "__main__":
    import asyncio
    asyncio.run(quick_test())
