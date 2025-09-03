#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Copenhagen Event Recommender tests.
"""

import pytest
import asyncio
import duckdb
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock
import json

# Add backend to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data-collection"))

# Handle hyphenated directory name for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util

# Create alias for data-collection -> data_collection
spec = importlib.util.spec_from_file_location("data_collection", Path(__file__).parent.parent / "data-collection" / "__init__.py")
if spec and spec.loader:
    data_collection_module = importlib.util.module_from_spec(spec)
    sys.modules["data_collection"] = data_collection_module

# Disable PyTorch-dependent imports for testing
os.environ['DISABLE_TORCH'] = 'true'

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import Settings
from backend.app.services.database_service import DatabaseService

# Test configuration
TEST_DATABASE_URL = "test_events.duckdb"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Test configuration settings."""
    return Settings(
        DATABASE_URL=TEST_DATABASE_URL,
        API_HOST="localhost",
        API_PORT=8001,
        CORS_ORIGINS=["http://localhost:3000", "http://testserver"],
        JWT_SECRET_KEY="test-secret-key-for-testing-only",
        ENABLE_METRICS=True,
        LOG_LEVEL="INFO"
    )

@pytest.fixture
def test_db():
    """Create test database with sample data."""
    # Create temporary database
    if os.path.exists(TEST_DATABASE_URL):
        os.remove(TEST_DATABASE_URL)
    
    conn = duckdb.connect(TEST_DATABASE_URL)
    
    # Create schema
    conn.execute("""
        CREATE TABLE venues (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            address VARCHAR,
            lat DOUBLE,
            lon DOUBLE,
            neighborhood VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE artists (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            genres JSON,
            popularity_score FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE events (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            description TEXT,
            date_time TIMESTAMP,
            end_date_time TIMESTAMP,
            price_min DECIMAL,
            price_max DECIMAL,
            venue_id VARCHAR,
            artist_ids JSON,
            popularity_score FLOAT DEFAULT 0.0,
            h3_index VARCHAR,
            source VARCHAR,
            source_url VARCHAR,
            image_url VARCHAR,
            status VARCHAR DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE users (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            email VARCHAR,
            preferences JSON,
            location_lat DOUBLE,
            location_lon DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE interactions (
            id VARCHAR PRIMARY KEY,
            user_id VARCHAR,
            event_id VARCHAR,
            interaction_type VARCHAR,
            rating FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert test data
    _insert_test_data(conn)
    
    conn.close()
    
    yield TEST_DATABASE_URL
    
    # Cleanup
    if os.path.exists(TEST_DATABASE_URL):
        os.remove(TEST_DATABASE_URL)

def _insert_test_data(conn):
    """Insert test data into database."""
    
    # Test venues
    venues = [
        ("venue_1", "Culture Box", "Kronprinsessegade 54A, København K", 55.6826, 12.5941, "Indre By"),
        ("venue_2", "Vega", "Enghavevej 40, Vesterbro", 55.6667, 12.5419, "Vesterbro"),
        ("venue_3", "Rust", "Guldbergsgade 8, Nørrebro", 55.6889, 12.5531, "Nørrebro")
    ]
    
    for venue in venues:
        conn.execute(
            "INSERT INTO venues (id, name, address, lat, lon, neighborhood) VALUES (?, ?, ?, ?, ?, ?)",
            venue
        )
    
    # Test artists
    artists = [
        ("artist_1", "Kollektiv Turmstrasse", '["techno", "electronic"]', 0.8),
        ("artist_2", "Agnes Obel", '["indie", "alternative"]', 0.9),
        ("artist_3", "WhoMadeWho", '["electronic", "indie"]', 0.7)
    ]
    
    for artist in artists:
        conn.execute(
            "INSERT INTO artists (id, name, genres, popularity_score) VALUES (?, ?, ?, ?)",
            artist
        )
    
    # Test events
    now = datetime.now()
    events = [
        (
            "event_1", 
            "Kollektiv Turmstrasse Live", 
            "Electronic techno night with the German duo",
            now + timedelta(days=7),
            now + timedelta(days=7, hours=6),
            200.0, 300.0,
            "venue_1",
            '["artist_1"]',
            0.8,
            "881f1d4927fffff",
            "eventbrite",
            "https://eventbrite.com/e/test-event",
            "https://example.com/image1.jpg"
        ),
        (
            "event_2",
            "Agnes Obel Concert",
            "Intimate indie performance",
            now + timedelta(days=14),
            now + timedelta(days=14, hours=3),
            400.0, 600.0,
            "venue_2",
            '["artist_2"]',
            0.9,
            "881f1d4937fffff",
            "meetup",
            "https://meetup.com/test-event",
            "https://example.com/image2.jpg"
        )
    ]
    
    for event in events:
        conn.execute("""
            INSERT INTO events (id, title, description, date_time, end_date_time, price_min, price_max, 
                              venue_id, artist_ids, popularity_score, h3_index, source, source_url, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, event)
    
    # Test users
    users = [
        ("user_1", "Test User 1", "test1@example.com", '{"preferred_genres": ["techno", "electronic"]}', 55.6761, 12.5683),
        ("user_2", "Test User 2", "test2@example.com", '{"preferred_genres": ["indie", "alternative"]}', 55.6761, 12.5683)
    ]
    
    for user in users:
        conn.execute(
            "INSERT INTO users (id, name, email, preferences, location_lat, location_lon) VALUES (?, ?, ?, ?, ?, ?)",
            user
        )
    
    # Test interactions
    interactions = [
        ("int_1", "user_1", "event_1", "like", 5.0),
        ("int_2", "user_1", "event_2", "going", 4.0),
        ("int_3", "user_2", "event_2", "like", 5.0)
    ]
    
    for interaction in interactions:
        conn.execute(
            "INSERT INTO interactions (id, user_id, event_id, interaction_type, rating) VALUES (?, ?, ?, ?, ?)",
            interaction
        )

@pytest.fixture
def client(test_settings, test_db):
    """FastAPI test client."""
    # Override settings for testing
    app.dependency_overrides = {}
    
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def db_service(test_db):
    """Database service for testing."""
    return DatabaseService()

@pytest.fixture
def sample_events():
    """Sample event data for testing."""
    now = datetime.now()
    return [
        {
            'id': 'test_event_1',
            'title': 'Test Techno Night',
            'description': 'Underground electronic music event',
            'start_time': now + timedelta(days=1),
            'end_time': now + timedelta(days=1, hours=6),
            'venue_name': 'Culture Box',
            'venue_address': 'Kronprinsessegade 54A, København K',
            'venue_lat': 55.6826,
            'venue_lon': 12.5941,
            'price_min': 150.0,
            'price_max': 250.0,
            'artists': ['DJ Test', 'Producer Example'],
            'genres': ['techno', 'electronic'],
            'source': 'test',
            'popularity_score': 0.7
        },
        {
            'id': 'test_event_2',
            'title': 'Indie Concert Test',
            'description': 'Alternative music performance',
            'start_time': now + timedelta(days=2),
            'venue_name': 'Vega',
            'venue_address': 'Enghavevej 40, Vesterbro',
            'venue_lat': 55.6667,
            'venue_lon': 12.5419,
            'price_min': 300.0,
            'price_max': 500.0,
            'artists': ['Indie Artist'],
            'genres': ['indie', 'alternative'],
            'source': 'test',
            'popularity_score': 0.8
        }
    ]

@pytest.fixture
def sample_user_preferences():
    """Sample user preferences for testing."""
    return {
        'preferred_genres': ['techno', 'electronic'],
        'preferred_artists': ['Kollektiv Turmstrasse'],
        'preferred_venues': ['Culture Box'],
        'price_range': [100, 400],
        'location_lat': 55.6761,
        'location_lon': 12.5683,
        'preferred_times': [20, 21, 22],
        'preferred_days': [4, 5, 6]
    }

@pytest.fixture
def mock_external_apis():
    """Mock external API responses."""
    return {
        'spotify_auth': {
            'access_token': 'mock_spotify_token',
            'token_type': 'Bearer',
            'expires_in': 3600
        },
        'spotify_search': {
            'artists': {
                'items': [{
                    'id': 'mock_artist_id',
                    'name': 'Mock Artist',
                    'genres': ['electronic', 'techno'],
                    'popularity': 80,
                    'images': [{'url': 'https://example.com/artist.jpg'}]
                }]
            }
        },
        'lastfm_artist': {
            'artist': {
                'name': 'Mock Artist',
                'bio': {'summary': 'Mock artist bio'},
                'tags': {'tag': [{'name': 'electronic'}, {'name': 'techno'}]},
                'stats': {'listeners': '50000', 'playcount': '500000'}
            }
        },
        'eventbrite_events': {
            'events': [{
                'id': 'mock_eventbrite_id',
                'name': {'text': 'Mock Eventbrite Event'},
                'description': {'text': 'Mock event description'},
                'start': {'local': '2024-12-01T20:00:00'},
                'venue': {
                    'name': 'Mock Venue',
                    'address': {'localized_area_display': 'Copenhagen'},
                    'latitude': '55.6761',
                    'longitude': '12.5683'
                },
                'url': 'https://eventbrite.com/mock-event'
            }]
        }
    }

@pytest.fixture
def mock_instagram_posts():
    """Mock Instagram post data."""
    return [
        {
            'shortcode': 'mock_post_1',
            'caption': 'Tonight at Culture Box! Techno night with amazing lineup #techno #copenhagenevents',
            'caption_hashtags': ['techno', 'copenhagenevents', 'cultureboxcph'],
            'likes': 150,
            'comments': 25,
            'url': 'https://instagram.com/p/mock_post_1/',
            'date': datetime.now() - timedelta(hours=2)
        }
    ]

@pytest.fixture
def mock_tiktok_videos():
    """Mock TikTok video data."""
    return [
        {
            'id': 'mock_video_1',
            'desc': 'Insane techno party at Culture Box tonight! #techno #copenhagenevents',
            'createTime': int((datetime.now() - timedelta(hours=1)).timestamp()),
            'stats': {'playCount': 5000, 'diggCount': 200, 'commentCount': 30},
            'textExtra': [{'hashtagName': 'techno'}, {'hashtagName': 'copenhagenevents'}],
            'author': {'uniqueId': 'mock_user'},
            'video': {'cover': 'https://tiktok.com/mock_cover.jpg'}
        }
    ]

# Async test helpers
@pytest.fixture
def async_client(client):
    """Async test client wrapper."""
    return client

# Data validation helpers
@pytest.fixture
def validation_test_data():
    """Test data for validation testing."""
    return {
        'valid_event': {
            'id': 'valid_test_event',
            'title': 'Valid Test Event',
            'description': 'This is a valid test event for validation testing',
            'start_time': datetime.now() + timedelta(days=1),
            'venue_name': 'Test Venue',
            'venue_lat': 55.6761,
            'venue_lon': 12.5683,
            'source': 'test'
        },
        'invalid_event_missing_title': {
            'id': 'invalid_test_event',
            'description': 'Missing title',
            'start_time': datetime.now() + timedelta(days=1),
            'venue_name': 'Test Venue',
            'source': 'test'
        },
        'invalid_event_bad_coordinates': {
            'id': 'invalid_coords_event',
            'title': 'Invalid Coordinates Event',
            'description': 'Has invalid coordinates',
            'start_time': datetime.now() + timedelta(days=1),
            'venue_name': 'Test Venue',
            'venue_lat': 999.0,  # Invalid latitude
            'venue_lon': 999.0,  # Invalid longitude
            'source': 'test'
        }
    }

# Performance testing helpers
@pytest.fixture
def performance_test_events():
    """Generate large dataset for performance testing."""
    events = []
    base_time = datetime.now()
    
    for i in range(1000):
        event = {
            'id': f'perf_event_{i}',
            'title': f'Performance Test Event {i}',
            'description': f'Test event {i} for performance testing',
            'start_time': base_time + timedelta(days=i % 30),
            'venue_name': f'Test Venue {i % 10}',
            'venue_lat': 55.6761 + (i % 100) * 0.001,
            'venue_lon': 12.5683 + (i % 100) * 0.001,
            'artists': [f'Artist {i}', f'Artist {i+1}'],
            'genres': ['test_genre'] * (i % 3 + 1),
            'source': 'performance_test',
            'popularity_score': (i % 100) / 100.0
        }
        events.append(event)
    
    return events

# ML model testing helpers
@pytest.fixture
def ml_test_data():
    """Test data for ML model testing."""
    return {
        'training_interactions': [
            {'user_id': 'user_1', 'event_id': 'event_1', 'interaction_type': 'like', 'rating': 5.0},
            {'user_id': 'user_1', 'event_id': 'event_2', 'interaction_type': 'going', 'rating': 4.0},
            {'user_id': 'user_2', 'event_id': 'event_1', 'interaction_type': 'like', 'rating': 4.0},
            {'user_id': 'user_2', 'event_id': 'event_3', 'interaction_type': 'save', 'rating': 3.0}
        ],
        'test_user_preferences': {
            'user_1': ['techno', 'electronic'],
            'user_2': ['indie', 'alternative']
        }
    }

# Security testing helpers
@pytest.fixture
def security_test_payloads():
    """Test payloads for security testing."""
    return {
        'sql_injection': [
            "'; DROP TABLE events; --",
            "1' OR '1'='1",
            "admin'/*",
            "' UNION SELECT * FROM users --"
        ],
        'xss_payloads': [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>"
        ],
        'oversized_requests': {
            'title': 'A' * 10000,  # Very long title
            'description': 'B' * 100000,  # Very long description
            'artists': ['Artist'] * 1000  # Too many artists
        }
    }

# Cleanup helpers
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    
    # Clean up any temporary files created during testing
    temp_files = [
        'test_events.duckdb',
        'test_cache.pkl',
        'test_embeddings.json'
    ]
    
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass  # Ignore cleanup errors