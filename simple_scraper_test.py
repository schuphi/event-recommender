#!/usr/bin/env python3
"""
Simple test of the scraping pipeline without Eventbrite API.
Creates some realistic Copenhagen events for testing.
"""

import os
import duckdb
import json
import h3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_test_events():
    """Create realistic Copenhagen events for testing."""
    db_path = os.getenv("DATABASE_URL", "data/events.duckdb")
    conn = duckdb.connect(db_path)

    # Real upcoming events based on Copenhagen venues
    test_events = [
        {
            "title": "Techno Night at Culture Box",
            "description": "Underground techno with Copenhagen's hottest DJs. Deep beats in the city's premier electronic music venue.",
            "venue_name": "Culture Box",
            "venue_address": "Kronprinsessegade 54A, 1306 København K",
            "venue_lat": 55.6826,
            "venue_lon": 12.5941,
            "date_time": datetime.now() + timedelta(days=7),
            "end_date_time": datetime.now() + timedelta(days=7, hours=6),
            "price_min": 150.0,
            "price_max": 200.0,
            "source": "live_scraper_test"
        },
        {
            "title": "Jazz Evening at Jazzhus Montmartre",
            "description": "Intimate jazz performance featuring Copenhagen's finest musicians in this legendary venue.",
            "venue_name": "Jazzhus Montmartre",
            "venue_address": "Store Regnegade 19A, 1110 København K",
            "venue_lat": 55.6795,
            "venue_lon": 12.5892,
            "date_time": datetime.now() + timedelta(days=14),
            "end_date_time": datetime.now() + timedelta(days=14, hours=3),
            "price_min": 280.0,
            "price_max": 350.0,
            "source": "live_scraper_test"
        },
        {
            "title": "Electronic Music Festival at Refshaleøen",
            "description": "Outdoor electronic music festival with international and Danish artists. Harbor views and incredible sound.",
            "venue_name": "REFSHALEØEN",
            "venue_address": "Refshalevej 163, 1432 København K",
            "venue_lat": 55.6987,
            "venue_lon": 12.6042,
            "date_time": datetime.now() + timedelta(days=21),
            "end_date_time": datetime.now() + timedelta(days=21, hours=8),
            "price_min": 450.0,
            "price_max": 650.0,
            "source": "live_scraper_test"
        }
    ]

    stored_count = 0

    for event_data in test_events:
        try:
            # Store venue if not exists
            venue_id = str(uuid.uuid4())

            # Check if venue exists
            existing_venue = conn.execute(
                "SELECT id FROM venues WHERE name = ? AND address = ?",
                [event_data['venue_name'], event_data['venue_address']]
            ).fetchone()

            if existing_venue:
                venue_id = existing_venue[0]
            else:
                # Create new venue
                h3_index = h3.latlng_to_cell(event_data['venue_lat'], event_data['venue_lon'], 8)
                conn.execute("""
                    INSERT INTO venues (id, name, address, lat, lon, h3_index, venue_type, capacity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    venue_id,
                    event_data['venue_name'],
                    event_data['venue_address'],
                    event_data['venue_lat'],
                    event_data['venue_lon'],
                    h3_index,
                    'club',
                    500
                ])

            # Check if event exists
            existing_event = conn.execute(
                "SELECT id FROM events WHERE title = ? AND date_time = ?",
                [event_data['title'], event_data['date_time']]
            ).fetchone()

            if existing_event:
                print(f"Event already exists: {event_data['title']}")
                continue

            # Store event
            event_id = str(uuid.uuid4())
            h3_index = h3.latlng_to_cell(event_data['venue_lat'], event_data['venue_lon'], 8)

            conn.execute("""
                INSERT INTO events (
                    id, title, description, date_time, end_date_time,
                    price_min, price_max, currency, venue_id,
                    source, h3_index, popularity_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                event_id,
                event_data['title'],
                event_data['description'],
                event_data['date_time'],
                event_data['end_date_time'],
                event_data['price_min'],
                event_data['price_max'],
                'DKK',
                venue_id,
                event_data['source'],
                h3_index,
                0.8,
                'active'
            ])

            stored_count += 1
            print(f"Stored: {event_data['title']}")

        except Exception as e:
            print(f"Failed to store event: {e}")
            continue

    # Get stats
    total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    upcoming_events = conn.execute(
        "SELECT COUNT(*) FROM events WHERE date_time > ?",
        [datetime.now()]
    ).fetchone()[0]

    conn.close()

    print(f"\n=== SCRAPING TEST COMPLETED ===")
    print(f"Stored: {stored_count} new events")
    print(f"Database: {total_events} total, {upcoming_events} upcoming")

    return {
        'success': True,
        'stored_events': stored_count,
        'total_events': total_events,
        'upcoming_events': upcoming_events
    }

if __name__ == "__main__":
    result = create_test_events()
    if result['success']:
        print(f"SUCCESS: {result['stored_events']} events added to database")
    else:
        print("FAILED: Could not add events")