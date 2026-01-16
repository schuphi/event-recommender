#!/usr/bin/env python3
"""
Direct connection test without .env file.
Replace YOUR_PASSWORD with actual password.
"""

import asyncio
import asyncpg

async def test_direct_connection():
    """Test connection with hardcoded URL (for debugging only)."""
    
    # REPLACE YOUR_PASSWORD with your actual Supabase password
    database_url = "postgresql://postgres:YOUR_PASSWORD@db.bgmylhcqhsrvmmlqjhxo.supabase.co:5432/postgres"
    
    print(f"🔗 Testing direct connection...")
    
    try:
        conn = await asyncpg.connect(database_url)
        print("✅ Direct connection successful!")
        
        result = await conn.fetchval("SELECT version()")
        print(f"📊 PostgreSQL version: {result}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False

if __name__ == "__main__":
    print("⚠️  Remember to replace YOUR_PASSWORD with actual password!")
    success = asyncio.run(test_direct_connection())
