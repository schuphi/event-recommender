#!/usr/bin/env python3
"""
Debug .env file loading
"""

import os
from dotenv import load_dotenv

print("Debugging .env file...")

# Check if .env file exists
if os.path.exists('.env'):
    print("OK .env file found")
    
    # Try to read it manually
    with open('.env', 'r') as f:
        content = f.read()
        print(f"File .env content length: {len(content)} characters")
        print(f"First 100 characters: {content[:100]}...")
else:
    print("ERROR .env file not found in current directory")
    print(f"Current directory: {os.getcwd()}")

# Load environment variables
print("\nLoading .env...")
load_dotenv()

# Check if DATABASE_URL is loaded
database_url = os.getenv("DATABASE_URL")
if database_url:
    print("OK DATABASE_URL loaded successfully")
    # Hide password for security
    safe_url = database_url.split('@')[1] if '@' in database_url else 'unknown'
    print(f"Connecting to: {safe_url}")
else:
    print("ERROR DATABASE_URL not found in environment")
    print("Available env vars starting with 'D':")
    for key in os.environ:
        if key.startswith('D'):
            print(f"  - {key}")

print(f"\nWorking directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
