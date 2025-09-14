#!/usr/bin/env python3
"""
Standalone script to run the Python FastAPI backend
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    import uvicorn
    from main import app
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    reload = os.getenv("NODE_ENV", "development") == "development"
    
    print(f"🚀 Starting FastAPI server on {host}:{port}")
    print(f"📊 Environment: {os.getenv('NODE_ENV', 'development')}")
    print(f"🔄 Auto-reload: {reload}")
    print(f"📖 API Documentation: http://{host}:{port}/api/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info" if not reload else "debug"
    )