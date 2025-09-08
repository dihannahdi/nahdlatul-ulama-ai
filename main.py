#!/usr/bin/env python3
"""
Railway deployment entry point for Nahdlatul Ulama AI Backend
This file provides the main entry point that Railway can automatically detect.
"""

import sys
import os

print("🚀 Railway Deployment Starting... [2024-12-30]")
print(f"📁 Working directory: {os.getcwd()}")
print(f"🐍 Python executable: {sys.executable}")
print(f"📊 Environment PORT: {os.environ.get('PORT', 'Not set')}")

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

print(f"📦 Python path updated: {sys.path[-1]}")

# Verify required files exist
print("\n🔍 Checking required files...")
if os.path.exists("/app/sql_chunks"):
    chunk_count = len([f for f in os.listdir("/app/sql_chunks") if f.endswith('.sql')])
    print(f"✅ sql_chunks directory found with {chunk_count} SQL files")
else:
    print("❌ sql_chunks directory not found")

if os.path.exists("/app/backend/full_production_main.py"):
    print("✅ Backend module found at /app/backend/full_production_main.py")
else:
    print("❌ Backend module not found at /app/backend/full_production_main.py")

print("\n🔄 Importing production app...")

# Import the FastAPI app from the backend (correct app name)
from full_production_main import production_app as app

print("✅ Production app imported successfully")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8001))
    
    print(f"\n🌐 Starting server on 0.0.0.0:{port}")
    print("🔄 Server starting...")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
