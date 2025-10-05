#!/usr/bin/env python3
"""
Simple test to verify Supabase connection works.
Run this before testing the full auth system.
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def test_connection():
    """Test basic database connection."""
    
    # Load environment variables
    load_dotenv()
    
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("❌ DATABASE_URL not found in environment")
        print("Make sure you have a .env file with DATABASE_URL set")
        return False
    
    print(f"🔗 Testing connection to: {database_url.split('@')[1] if '@' in database_url else 'database'}")
    
    try:
        # Test connection
        conn = await asyncpg.connect(database_url)
        print("✅ Database connection successful!")
        
        # Test basic query
        result = await conn.fetchval("SELECT version()")
        print(f"📊 PostgreSQL version: {result}")
        
        # Test if our tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('users', 'events', 'venues')
        """)
        
        table_names = [row['table_name'] for row in tables]
        print(f"📋 Found tables: {table_names}")
        
        if 'users' in table_names:
            print("✅ Users table exists - schema is set up!")
        else:
            print("⚠️  Users table not found - need to run schema")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if success:
        print("\n🚀 Ready to test authentication!")
    else:
        print("\n🔧 Fix connection issues before proceeding")
