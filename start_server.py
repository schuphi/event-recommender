#!/usr/bin/env python3
"""
Startup script for Copenhagen Event Recommender.
Starts the backend server with correct configuration.
"""

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Start the server with proper configuration."""

    # Load environment variables
    load_dotenv()

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("🚀 Starting Copenhagen Event Recommender Server...")
    print(f"📂 Project root: {project_root}")
    print(f"🗄️  Database: {os.getenv('DATABASE_URL', 'data/events.duckdb')}")
    print(f"🌐 Server will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        # Start the server
        subprocess.run([
            sys.executable,
            "-m", "uvicorn",
            "backend.app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)

    except KeyboardInterrupt:
        print("\n✨ Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())