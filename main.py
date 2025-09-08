#!/usr/bin/env python3
"""
Railway deployment entry point for Nahdlatul Ulama AI Backend
This file provides the main entry point that Railway can automatically detect.
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import the FastAPI app from the backend
from full_production_main import app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)
