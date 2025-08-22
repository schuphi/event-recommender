#!/usr/bin/env python3
"""
Test script to verify backend functionality with LangChain and DuckDB.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from fastapi.testclient import TestClient
from backend.app.main_simple import app

def test_backend_functionality():
    """Test basic backend functionality."""
    client = TestClient(app)
    
    print("Testing Copenhagen Event Recommender Backend...")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"   Service: {health_data.get('service')}")
        print("   PASS - Health check passed")
    else:
        print("   FAIL - Health check failed")
        return False
    
    # Test 2: Events endpoint
    print("\n2. Testing events endpoint...")
    response = client.get("/events")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        events_data = response.json()
        events_count = len(events_data.get('events', []))
        print(f"   Events returned: {events_count}")
        print("   PASS - Events endpoint working")
    else:
        print("   FAIL - Events endpoint failed")
        return False
    
    # Test 3: Recommendations endpoint
    print("\n3. Testing recommendations endpoint...")
    test_request = {
        "user_preferences": {
            "preferred_genres": ["techno", "electronic"],
            "max_price": 300
        },
        "num_recommendations": 5
    }
    response = client.post("/recommend", json=test_request)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        rec_data = response.json()
        rec_count = len(rec_data.get('recommendations', []))
        has_fallback = rec_data.get('fallback', False)
        print(f"   Recommendations returned: {rec_count}")
        print(f"   Using fallback: {has_fallback}")
        if has_fallback:
            print(f"   Fallback reason: {rec_data.get('error', 'Unknown')}")
        print("   PASS - Recommendations endpoint working")
    else:
        print("   FAIL - Recommendations endpoint failed")
        return False
    
    # Test 4: Search endpoint
    print("\n4. Testing search endpoint...")
    search_request = {"query": "techno music"}
    response = client.post("/search", json=search_request)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        search_data = response.json()
        results_count = len(search_data.get('results', []))
        has_fallback = search_data.get('fallback', False)
        print(f"   Search results: {results_count}")
        print(f"   Using fallback: {has_fallback}")
        if has_fallback:
            print(f"   Fallback reason: {search_data.get('error', 'Unknown')}")
        print("   PASS - Search endpoint working")
    else:
        print("   FAIL - Search endpoint failed")
        return False
    
    return True

def test_langchain_integration():
    """Test LangChain integration specifically."""
    print("\n5. Testing LangChain Integration...")
    print("-" * 30)
    
    try:
        # Test LangChain recommender directly
        sys.path.append('backend')
        from backend.app.services.langchain_recommender import LangChainRecommender
        
        # Initialize with correct database path
        recommender = LangChainRecommender(db_path='data/events.duckdb')
        
        print("   PASS - LangChain recommender imported successfully")
        
        # Test search functionality
        test_prefs = {
            "preferred_genres": ["electronic", "techno"],
            "max_price": 250
        }
        
        results = recommender.get_recommendations(
            user_preferences=test_prefs,
            num_recommendations=3
        )
        
        print(f"   PASS - LangChain recommendations: {len(results)} results")
        for i, rec in enumerate(results[:2], 1):
            print(f"     {i}. Event ID: {rec.event_id}, Score: {rec.score:.2f}")
        
        # Test search
        search_results = recommender.search_events("electronic music", limit=3)
        print(f"   PASS - LangChain search: {len(search_results)} results")
        
        return True
        
    except Exception as e:
        print(f"   FAIL - LangChain integration failed: {str(e)}")
        return False

def test_duckdb_connection():
    """Test DuckDB database connection."""
    print("\n6. Testing DuckDB Connection...")
    print("-" * 30)
    
    try:
        import duckdb
        
        # Test connection to database
        conn = duckdb.connect('data/events.duckdb')
        
        # Check tables
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        print(f"   PASS - Database connected, tables: {table_names}")
        
        # Check events count
        event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        venue_count = conn.execute("SELECT COUNT(*) FROM venues").fetchone()[0]
        
        print(f"   PASS - Events: {event_count}, Venues: {venue_count}")
        
        # Sample event
        sample_event = conn.execute("SELECT title, venue_id FROM events LIMIT 1").fetchone()
        if sample_event:
            print(f"   PASS - Sample event: {sample_event[0]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   FAIL - DuckDB connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Copenhagen Event Recommender Backend Verification")
    print("=" * 60)
    
    # Run all tests
    basic_test_passed = test_backend_functionality()
    duckdb_test_passed = test_duckdb_connection()
    langchain_test_passed = test_langchain_integration()
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Basic FastAPI functionality: {'PASS' if basic_test_passed else 'FAIL'}")
    print(f"DuckDB integration:          {'PASS' if duckdb_test_passed else 'FAIL'}")
    print(f"LangChain integration:       {'PASS' if langchain_test_passed else 'FAIL'}")
    
    overall_status = basic_test_passed and duckdb_test_passed and langchain_test_passed
    print(f"\nOVERALL STATUS: {'BACKEND WORKING CORRECTLY' if overall_status else 'ISSUES DETECTED'}")
    
    if overall_status:
        print("\nSUCCESS: The backend is properly configured with LangChain and DuckDB!")
        print("   - FastAPI endpoints are responding")
        print("   - DuckDB database is accessible with sample data")
        print("   - LangChain semantic search is functional")
        print("   - All required dependencies are installed")
    else:
        print("\nWARNING: Some issues were detected. Check the test output above for details.")