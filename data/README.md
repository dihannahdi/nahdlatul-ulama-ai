# Data Processing for Nahdlatul Ulama AI

This directory contains scripts to process Islamic texts and create vector embeddings.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   cd ../backend
   pip install -r requirements.txt
   ```

2. **Run data processing:**
   ```bash
   # Full pipeline (processes all data)
   python process_data.py --mode full
   
   # Sample dataset (for testing)
   python process_data.py --mode sample
   
   # Test existing database
   python process_data.py --mode test
   ```

## 📊 Processing Steps

1. **Load References**: Islamic methodology texts from `/references`
2. **Load SQL Chunks**: Islamic knowledge from `/sql_chunks`  
3. **Create Embeddings**: Using SentenceTransformers
4. **Build Vector Store**: ChromaDB for semantic search
5. **Test System**: Sample queries for validation

## 📁 Output

- `./processed_data/` - Processed text files
- `./vector_db/` - ChromaDB vector database
- `./sample_vector_db/` - Sample database for testing

## 🔧 Configuration

The pipeline uses:
- **Embedding Model**: `all-MiniLM-L6-v2` (free, 384 dimensions)
- **Vector DB**: ChromaDB (free, self-hosted)
- **Text Splitter**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)

## 📈 Performance

- **Processing time**: ~5-10 minutes (depends on data size)
- **Memory usage**: ~2-4GB (during processing)
- **Storage**: ~500MB-1GB (for vector database)

## 🧪 Testing

The pipeline includes automatic testing with sample queries:
- Islamic jurisprudence questions
- NU methodology inquiries
- Text retrieval validation

## 🔍 Sample Queries

- "Apa itu fiqh menurut NU?"
- "Bagaimana metode istinbath dalam NU?"
- "Jelaskan tentang Ahlussunnah wal Jama'ah"
- "Apa prinsip tawassuth dalam NU?"

## 🛠️ Troubleshooting

### Memory Issues
```bash
# Use sample mode for testing
python process_data.py --mode sample
```

### Import Errors
```bash
# Make sure backend dependencies are installed
cd ../backend && pip install -r requirements.txt
```

### Data Not Found
- Check `/references` and `/sql_chunks` directories
- Ensure files have proper encoding (UTF-8)

## 📊 Data Statistics

After processing, you'll see:
- Total documents processed
- Vector embeddings created
- Test query results
- Storage size information
