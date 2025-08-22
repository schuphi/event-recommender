#!/usr/bin/env python3
"""
Database initialization script for Copenhagen Event Recommender.
Creates tables, indexes, and inserts sample data.
"""

import duckdb
import json
import h3
from pathlib import Path
from datetime import datetime, timedelta
import uuid

# Copenhagen coordinates for H3 indexing
COPENHAGEN_CENTER = (55.6761, 12.5683)  # lat, lon
COPENHAGEN_BOUNDS = {
    'lat_min': 55.6150, 'lat_max': 55.7350,
    'lon_min': 12.4500, 'lon_max': 12.6500
}

def init_database(db_path: str = "data/events.duckdb"):
    """Initialize database with schema and sample data."""
    
    # Ensure data directory exists
    Path(db_path).parent.mkdir(exist_ok=True)
    
    # Connect to DuckDB
    conn = duckdb.connect(db_path)
    
    # Read and execute schema
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        schema_sql = f.read()
    
    # Execute schema (split by statements)
    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
    for stmt in statements:
        try:
            conn.execute(stmt)
        except Exception as e:
            print(f"Error executing statement: {e}")
            print(f"Statement: {stmt[:100]}...")
    
    print("✓ Database schema created")
    
    # Insert sample venues
    venues = [
        {
            'id': str(uuid.uuid4()),
            'name': 'Vega',
            'address': 'Enghavevej 40, 1674 København V',
            'lat': 55.6667, 'lon': 12.5419,
            'neighborhood': 'Vesterbro',
            'venue_type': 'concert_hall',
            'capacity': 1550
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Culture Box',
            'address': 'Kronprinsessegade 54A, 1306 København K',
            'lat': 55.6826, 'lon': 12.5941,
            'neighborhood': 'Indre By',
            'venue_type': 'club',
            'capacity': 600
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Rust',
            'address': 'Guldbergsgade 8, 2200 København N',
            'lat': 55.6889, 'lon': 12.5531,
            'neighborhood': 'Nørrebro',
            'venue_type': 'club',
            'capacity': 700
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Loppen',
            'address': 'Sydområdet 4B, 1440 København K',
            'lat': 55.6771, 'lon': 12.5989,
            'neighborhood': 'Christiania',
            'venue_type': 'club',
            'capacity': 300
        }
    ]
    
    # Add H3 indexes to venues
    for venue in venues:
        venue['h3_index'] = h3.latlng_to_cell(venue['lat'], venue['lon'], 8)
    
    # Insert venues
    for venue in venues:
        conn.execute("""
            INSERT INTO venues (id, name, address, lat, lon, h3_index, neighborhood, venue_type, capacity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            venue['id'], venue['name'], venue['address'], 
            venue['lat'], venue['lon'], venue['h3_index'],
            venue['neighborhood'], venue['venue_type'], venue['capacity']
        ])
    
    print(f"✓ Inserted {len(venues)} sample venues")
    
    # Insert sample artists
    artists = [
        {
            'id': str(uuid.uuid4()),
            'name': 'Kollektiv Turmstrasse',
            'genres': ['techno', 'electronic', 'minimal'],
            'popularity_score': 0.8
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Agnes Obel',
            'genres': ['indie', 'alternative', 'classical'],
            'popularity_score': 0.9
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Trentemøller',
            'genres': ['electronic', 'ambient', 'indie'],
            'popularity_score': 0.85
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Iceage',
            'genres': ['punk', 'post-punk', 'rock'],
            'popularity_score': 0.7
        }
    ]
    
    for artist in artists:
        conn.execute("""
            INSERT INTO artists (id, name, genres, popularity_score)
            VALUES (?, ?, ?, ?)
        """, [
            artist['id'], artist['name'], 
            json.dumps(artist['genres']), artist['popularity_score']
        ])
    
    print(f"✓ Inserted {len(artists)} sample artists")
    
    # Insert sample events
    events = []
    now = datetime.now()
    
    for i, venue in enumerate(venues):
        for j, artist in enumerate(artists):
            if (i + j) % 2 == 0:  # Create events for some artist-venue combinations
                event_date = now + timedelta(days=i*7 + j*2)
                event = {
                    'id': str(uuid.uuid4()),
                    'title': f'{artist["name"]} Live at {venue["name"]}',
                    'description': f'Experience {artist["name"]} live in an intimate setting at {venue["name"]}. A night of incredible {", ".join(artist["genres"][:2])} music.',
                    'date_time': event_date,
                    'price_min': 200.0 + i * 50,
                    'price_max': 400.0 + i * 100,
                    'venue_id': venue['id'],
                    'artist_ids': [artist['id']],
                    'source': 'sample_data',
                    'h3_index': venue['h3_index'],
                    'popularity_score': artist['popularity_score'] * 0.8
                }
                events.append(event)
    
    for event in events:
        conn.execute("""
            INSERT INTO events (
                id, title, description, date_time, price_min, price_max,
                venue_id, artist_ids, source, h3_index, popularity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            event['id'], event['title'], event['description'],
            event['date_time'], event['price_min'], event['price_max'],
            event['venue_id'], json.dumps(event['artist_ids']),
            event['source'], event['h3_index'], event['popularity_score']
        ])
        
        # Insert event-artist relationships
        for artist_id in event['artist_ids']:
            conn.execute("""
                INSERT INTO event_artists (event_id, artist_id, is_headliner)
                VALUES (?, ?, ?)
            """, [event['id'], artist_id, True])
    
    print(f"✓ Inserted {len(events)} sample events")
    
    # Create sample users with preferences
    users = [
        {
            'id': str(uuid.uuid4()),
            'name': 'Demo User 1',
            'preferences': {
                'genres': ['techno', 'electronic'],
                'price_cap': 500,
                'radius_km': 5
            },
            'location_lat': 55.6761,
            'location_lon': 12.5683
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Demo User 2', 
            'preferences': {
                'genres': ['indie', 'alternative'],
                'price_cap': 300,
                'radius_km': 10
            },
            'location_lat': 55.6890,
            'location_lon': 12.5530
        }
    ]
    
    for user in users:
        user['h3_index'] = h3.latlng_to_cell(user['location_lat'], user['location_lon'], 8)
        
        conn.execute("""
            INSERT INTO users (id, name, preferences, location_lat, location_lon, h3_index)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            user['id'], user['name'], json.dumps(user['preferences']),
            user['location_lat'], user['location_lon'], user['h3_index']
        ])
    
    print(f"✓ Inserted {len(users)} sample users")
    
    # Insert sample interactions for collaborative filtering
    interactions = [
        {'id': 1, 'user_id': users[0]['id'], 'event_id': events[0]['id'], 'interaction_type': 'like'},
        {'id': 2, 'user_id': users[0]['id'], 'event_id': events[1]['id'], 'interaction_type': 'going'},
        {'id': 3, 'user_id': users[1]['id'], 'event_id': events[1]['id'], 'interaction_type': 'like'},
        {'id': 4, 'user_id': users[1]['id'], 'event_id': events[2]['id'], 'interaction_type': 'dislike'},
    ]
    
    for interaction in interactions:
        conn.execute("""
            INSERT INTO interactions (id, user_id, event_id, interaction_type)
            VALUES (?, ?, ?, ?)
        """, [interaction['id'], interaction['user_id'], interaction['event_id'], interaction['interaction_type']])
    
    print(f"✓ Inserted {len(interactions)} sample interactions")
    
    # Verify database
    result = conn.execute("SELECT COUNT(*) FROM events").fetchone()
    print(f"✓ Database ready with {result[0]} events")
    
    conn.close()

if __name__ == "__main__":
    init_database()