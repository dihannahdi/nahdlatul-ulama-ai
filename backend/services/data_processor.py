import os
import sqlite3
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    """Process Islamic texts and SQL chunks into vector documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        # Get the project root directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(backend_dir)
        self.references_dir = os.path.join(project_root, "references")
        self.sql_chunks_dir = os.path.join(project_root, "sql_chunks")
    
    async def load_reference_texts(self) -> List[Document]:
        """Load and process reference methodology texts"""
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
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "type": "methodology",
                            "category": "nahdlatul_ulama_references"
                        }
                    )
                    
                    # Split into chunks
                    chunks = self.text_splitter.split_documents([doc])
                    documents.extend(chunks)
                    
                    print(f"✅ Loaded {filename}: {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
        
        return documents
    
    async def load_sql_chunks(self) -> List[Document]:
        """Load and process SQL chunks containing Islamic texts"""
        documents = []
        sql_files = []
        
        # Get all SQL files
        for filename in os.listdir(self.sql_chunks_dir):
            if filename.endswith('.sql'):
                sql_files.append(filename)
        
        # Process in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(sql_files), batch_size):
            batch = sql_files[i:i + batch_size]
            batch_docs = await self._process_sql_batch(batch)
            documents.extend(batch_docs)
            print(f"📊 Processed SQL batch {i//batch_size + 1}: {len(batch_docs)} documents")
        
        return documents
    
    async def _process_sql_batch(self, sql_files: List[str]) -> List[Document]:
        """Process a batch of SQL files"""
        documents = []
        
        for filename in sql_files:
            filepath = os.path.join(self.sql_chunks_dir, filename)
            try:
                # Extract table name from filename
                table_name = filename.replace('_chunk_0000.sql', '')
                
                # Read SQL file
                with open(filepath, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                # Extract INSERT statements
                texts = self._extract_texts_from_sql(sql_content)
                
                for i, text in enumerate(texts):
                    if text.strip():  # Skip empty texts
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "table": table_name,
                                "type": "islamic_text",
                                "category": "sql_chunks",
                                "chunk_id": i
                            }
                        )
                        
                        # Split long texts
                        chunks = self.text_splitter.split_documents([doc])
                        documents.extend(chunks)
                        
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
        
        return documents
    
    def _extract_texts_from_sql(self, sql_content: str) -> List[str]:
        """Extract text content from SQL INSERT statements"""
        texts = []
        lines = sql_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('INSERT INTO') and 'VALUES' in line:
                try:
                    # Extract values from INSERT statement
                    values_part = line.split('VALUES')[1].strip()
                    
                    # Parse the values (simplified parser)
                    if values_part.startswith('(') and values_part.endswith(');'):
                        values_content = values_part[1:-2]  # Remove ( and );
                        
                        # Split by comma but be careful with quoted strings
                        parts = self._smart_split_values(values_content)
                        
                        if len(parts) >= 2:  # Assuming content is the second field
                            content = parts[1].strip()
                            
                            # Remove quotes
                            if content.startswith("'") and content.endswith("'"):
                                content = content[1:-1]
                            
                            # Unescape quotes
                            content = content.replace("''", "'")
                            
                            # Only include substantial content
                            if len(content) > 50 and not content.startswith('http'):
                                texts.append(content)
                                
                except Exception as e:
                    # Skip malformed lines
                    continue
        
        return texts
    
    def _smart_split_values(self, values_str: str) -> List[str]:
        """Smart split of SQL VALUES considering quoted strings"""
        parts = []
        current_part = ""
        in_quotes = False
        i = 0
        
        while i < len(values_str):
            char = values_str[i]
            
            if char == "'" and (i == 0 or values_str[i-1] != "'"):
                in_quotes = not in_quotes
                current_part += char
            elif char == "," and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
            
            i += 1
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts
    
    async def load_all_data(self) -> List[Document]:
        """Load all Islamic texts and methodology references"""
        print("📚 Loading Islamic methodology references...")
        reference_docs = await self.load_reference_texts()
        
        print("📊 Loading Islamic text database...")  
        sql_docs = await self.load_sql_chunks()
        
        all_documents = reference_docs + sql_docs
        
        print(f"✅ Total documents loaded: {len(all_documents)}")
        print(f"   - Methodology references: {len(reference_docs)}")
        print(f"   - Islamic texts: {len(sql_docs)}")
        
        return all_documents
