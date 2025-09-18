#!/usr/bin/env python3
"""
Railway Entry Point for Copenhagen Event Recommender
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

if __name__ == "__main__":
    import uvicorn
    
    # Railway sets PORT, fallback to API_PORT, then default
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    print(f"Starting Copenhagen Event Recommender on {host}:{port}")
    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=False,
        access_log=True
    )
