from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Free LLM imports
try:
    from langchain_groq import ChatGroq
except ImportError:
    from langchain_google_genai import ChatGoogleGenerativeAI

from services.data_processor import DataProcessor
from services.islamic_reasoning import IslamicReasoningEngine
from models.schemas import QuestionRequest, AnswerResponse, ChatHistory, IslamicMethod

# Global variables for models
vector_store = None
llm = None
islamic_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and vector store on startup"""
    global vector_store, llm, islamic_engine
    
    print("🚀 Initializing Nahdlatul Ulama AI Backend...")
    
    # Initialize embeddings (free sentence transformers)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Initialize LLM (try Groq first, fallback to Gemini)
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if groq_api_key:
        # Set the API key as environment variable for ChatGroq
        os.environ["GROQ_API_KEY"] = groq_api_key
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Current working model (Sept 2025)
            temperature=0.1
        )
        print("✅ Using Groq LLM (Llama 3.3 70B)")
    elif google_api_key:
        llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-pro",  # Free tier
            temperature=0.1
        )
        print("✅ Using Google Gemini")
    else:
        raise RuntimeError("No LLM API key found. Set GROQ_API_KEY or GOOGLE_API_KEY")
    
    # Initialize data processor (optimized for development)
    from services.optimized_data_processor import OptimizedDataProcessor
    data_processor = OptimizedDataProcessor(sample_size=500)  # Start with 500 samples
    
    # Load and process Islamic texts
    print("📚 Loading Islamic knowledge base (sample mode)...")
    documents = await data_processor.quick_load_sample()
    
    # Create vector store
    print("🔍 Creating vector embeddings...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialize Islamic reasoning engine
    islamic_engine = IslamicReasoningEngine(vector_store, llm)
    
    print("✅ Backend initialization complete!")
    yield
    
    # Cleanup
    if vector_store:
        vector_store.persist()

app = FastAPI(
    title="Nahdlatul Ulama AI",
    description="Islamic Jurisprudence Assistant using NU Methodology",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Nahdlatul Ulama AI Backend",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "vector_store": vector_store is not None,
        "llm": llm is not None,
        "islamic_engine": islamic_engine is not None
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint for Islamic jurisprudence questions"""
    try:
        if not islamic_engine:
            raise HTTPException(status_code=503, detail="Islamic reasoning engine not initialized")
        
        # Process the question using Islamic methodology
        method = request.method or IslamicMethod.BAYANI  # Default to bayani if no method specified
        result = await islamic_engine.process_question(
            question=request.question,
            method=method,
            chat_history=request.chat_history
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            method_used=result["method_used"],
            confidence=result["confidence"],
            islamic_principles=result["islamic_principles"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/methods")
async def get_available_methods():
    """Get available Islamic reasoning methods"""
    return {
        "methods": [
            {
                "name": "bayani",
                "description": "Textual analysis method - استنباط من النصوص",
                "arabic": "البياني"
            },
            {
                "name": "qiyasi", 
                "description": "Analogical reasoning method - القياس",
                "arabic": "القياسي"
            },
            {
                "name": "istishlahi",
                "description": "Public interest method - المصلحة",
                "arabic": "الاستصلاحي"
            },
            {
                "name": "maqashidi",
                "description": "Objectives of Sharia method - مقاصد الشريعة", 
                "arabic": "المقاصدي"
            }
        ]
    }

@app.get("/search")
async def search_texts(query: str, limit: int = 5):
    """Search Islamic texts by similarity"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        # Perform similarity search
        docs = vector_store.similarity_search(query, k=limit)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return {"query": query, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = {
            "total_documents": 0,
            "available_methods": 4,
            "supported_languages": ["Arabic", "Indonesian", "English"],
            "islamic_principles": ["Tawassuth", "Tasamuh", "Tawazun", "I'tidal"]
        }
        
        if vector_store:
            # Get collection info if available
            try:
                stats["total_documents"] = vector_store._collection.count()
            except:
                stats["total_documents"] = "Unknown"
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
