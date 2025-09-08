#!/usr/bin/env python3
"""
Data Processing Pipeline for Nahdlatul Ulama AI
Processes Islamic texts and creates vector embeddings for RAG system
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from services.data_processor import DataProcessor
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

class DataPipeline:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.output_dir = Path("./processed_data")
        self.vector_db_dir = Path("./vector_db")
        
    async def run_full_pipeline(self):
        """Run the complete data processing pipeline"""
        print("🚀 Starting Nahdlatul Ulama AI Data Pipeline...")
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        # Step 1: Load all documents
        print("\n📚 Step 1: Loading Islamic texts and methodology...")
        documents = await self.processor.load_all_data()
        
        if not documents:
            print("❌ No documents loaded! Check your data directories.")
            return
        
        # Step 2: Create vector store
        print(f"\n🔍 Step 2: Creating vector embeddings for {len(documents)} documents...")
        vector_store = await self._create_vector_store(documents)
        
        # Step 3: Test the system
        print("\n🧪 Step 3: Testing the system...")
        await self._test_system(vector_store)
        
        print("\n✅ Data pipeline completed successfully!")
        print(f"📊 Vector database saved to: {self.vector_db_dir}")
        
    async def _create_vector_store(self, documents):
        """Create and persist vector store"""
        try:
            # Create ChromaDB vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.vector_db_dir)
            )
            
            # Persist the database
            vector_store.persist()
            
            print(f"✅ Vector store created with {len(documents)} documents")
            return vector_store
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    async def _test_system(self, vector_store):
        """Test the vector store with sample queries"""
        test_queries = [
            "Apa itu fiqh menurut NU?",
            "Bagaimana metode istinbath dalam NU?",
            "Jelaskan tentang Ahlussunnah wal Jama'ah",
            "Apa prinsip tawassuth dalam NU?"
        ]
        
        print("Testing with sample queries:")
        for i, query in enumerate(test_queries, 1):
            try:
                results = vector_store.similarity_search(query, k=2)
                print(f"\n{i}. Query: '{query}'")
                print(f"   Found {len(results)} relevant documents")
                
                if results:
                    top_result = results[0]
                    source = top_result.metadata.get("source", "Unknown")
                    content_preview = top_result.page_content[:150] + "..."
                    print(f"   Top result from: {source}")
                    print(f"   Preview: {content_preview}")
                
            except Exception as e:
                print(f"❌ Error testing query '{query}': {e}")

    async def create_sample_dataset(self):
        """Create a smaller sample dataset for testing"""
        print("📦 Creating sample dataset...")
        
        # Load a subset of data
        reference_docs = await self.processor.load_reference_texts()
        
        # Take first 100 SQL documents for testing
        sql_docs = await self.processor.load_sql_chunks()
        sample_sql_docs = sql_docs[:100] if len(sql_docs) > 100 else sql_docs
        
        sample_documents = reference_docs + sample_sql_docs
        
        if sample_documents:
            sample_vector_store = Chroma.from_documents(
                documents=sample_documents,
                embedding=self.embeddings,
                persist_directory="./sample_vector_db"
            )
            sample_vector_store.persist()
            
            print(f"✅ Sample dataset created with {len(sample_documents)} documents")
            print("💡 Use this for development and testing")
        else:
            print("❌ No sample documents available")

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Nahdlatul Ulama AI Data Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["full", "sample", "test"],
        default="full",
        help="Processing mode: full pipeline, sample dataset, or test only"
    )
    
    args = parser.parse_args()
    
    pipeline = DataPipeline()
    
    try:
        if args.mode == "full":
            asyncio.run(pipeline.run_full_pipeline())
        elif args.mode == "sample":
            asyncio.run(pipeline.create_sample_dataset())
        elif args.mode == "test":
            # Test existing vector store
            if Path("./vector_db").exists():
                print("🧪 Testing existing vector store...")
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = Chroma(
                    persist_directory="./vector_db",
                    embedding_function=embeddings
                )
                asyncio.run(pipeline._test_system(vector_store))
            else:
                print("❌ No vector store found. Run with --mode full first.")
                
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline stopped by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
