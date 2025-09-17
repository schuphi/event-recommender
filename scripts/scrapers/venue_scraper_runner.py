#!/usr/bin/env python3
"""
Runner for Copenhagen venue scraper with database integration.
Replaces the non-functional social media scrapers with real venue website scraping.
"""

import os
import sys
import duckdb
import uuid
import h3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Add the venue scrapers directory to Python path
sys.path.append(str(Path(__file__).parent / "data-collection" / "scrapers" / "venue_scrapers"))

from copenhagen_venues import CopenhagenVenueScraper

# Load environment variables
load_dotenv()


class VenueScraperDatabase:
    """Database integration for venue scraper."""

    def __init__(self):
        self.db_path = os.getenv("DATABASE_URL", "data/events.duckdb")

    def connect_db(self):
        """Connect to DuckDB database."""
        return duckdb.connect(self.db_path)

    def store_venue_if_not_exists(self, conn, venue_data):
        """Store venue if it doesn't exist, return venue_id."""
        existing = conn.execute(
            "SELECT id FROM venues WHERE name = ? AND address = ?",
            [venue_data['name'], venue_data['address']]
        ).fetchone()

        if existing:
            return existing[0]

        venue_id = str(uuid.uuid4())
        h3_index = h3.latlng_to_cell(venue_data['lat'], venue_data['lon'], 8)

        conn.execute("""
            INSERT INTO venues (id, name, address, lat, lon, h3_index, venue_type, capacity, neighborhood)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            venue_id, venue_data['name'], venue_data['address'],
            venue_data['lat'], venue_data['lon'], h3_index,
            'concert_venue', 600, self._get_neighborhood(venue_data['name'])
        ])

        return venue_id

    def store_scraped_events(self, events):
        """Store scraped events in database."""
        conn = self.connect_db()
        stored_count = 0
        updated_count = 0

        for event in events:
            try:
                # Store venue
                venue_data = {
                    'name': event.venue_name,
                    'address': event.venue_address,
                    'lat': event.venue_lat,
                    'lon': event.venue_lon
                }
                venue_id = self.store_venue_if_not_exists(conn, venue_data)

                # Check if event already exists
                existing = conn.execute("""
                    SELECT id FROM events
                    WHERE title = ? AND venue_id = ? AND date_time = ?
                """, [event.title, venue_id, event.date_time]).fetchone()

                if existing:
                    # Update existing event
                    conn.execute("""
                        UPDATE events SET
                            description = ?, end_date_time = ?, price_min = ?, price_max = ?,
                            source = ?, popularity_score = ?, status = ?, updated_at = ?
                        WHERE id = ?
                    """, [
                        event.description, event.end_date_time, event.price_min, event.price_max,
                        event.source, 0.7, 'active', datetime.now(), existing[0]
                    ])
                    updated_count += 1
                    continue

                # Store new event
                event_id = str(uuid.uuid4())
                h3_index = h3.latlng_to_cell(event.venue_lat, event.venue_lon, 8)

                conn.execute("""
                    INSERT INTO events (
                        id, title, description, date_time, end_date_time,
                        price_min, price_max, currency, venue_id,
                        source, h3_index, popularity_score, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    event_id, event.title, event.description,
                    event.date_time, event.end_date_time,
                    event.price_min, event.price_max, 'DKK',
                    venue_id, event.source, h3_index,
                    0.7, 'active', datetime.now(), datetime.now()
                ])

                stored_count += 1

            except Exception as e:
                print(f"Failed to store event '{event.title}': {e}")
                continue

        # Get final stats
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        upcoming_events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE date_time > ?",
            [datetime.now()]
        ).fetchone()[0]

        # Show source breakdown
        source_breakdown = conn.execute("""
            SELECT source, COUNT(*) as count
            FROM events
            WHERE date_time > ?
            GROUP BY source
            ORDER BY count DESC
        """, [datetime.now()]).fetchall()

        conn.close()

        print(f"\n=== VENUE SCRAPING COMPLETED ===")
        print(f"New events stored: {stored_count}")
        print(f"Events updated: {updated_count}")
        print(f"Total events in database: {total_events}")
        print(f"Upcoming events: {upcoming_events}")
        print(f"\nEvents by source:")
        for source, count in source_breakdown:
            print(f"  - {source}: {count} events")

        return {
            'success': True,
            'stored_events': stored_count,
            'updated_events': updated_count,
            'total_events': total_events,
            'upcoming_events': upcoming_events,
            'source_breakdown': dict(source_breakdown)
        }

    def _get_neighborhood(self, venue_name):
        """Get neighborhood for venue."""
        neighborhood_map = {
            'Rust': 'Nørrebro',
            'Pumpehuset': 'Indre By',
            'Loppen': 'Christiania',
            'Amager Bio': 'Amager',
            'Beta': 'Amager'
        }
        return neighborhood_map.get(venue_name, 'Copenhagen')


def run_venue_scraping():
    """Main function to run venue scraping."""
    print("=== Starting Copenhagen Venue Scraping ===")

    # Initialize scraper
    scraper = CopenhagenVenueScraper()

    # Scrape events from all venues
    print("Scraping events from Copenhagen venues...")
    events = scraper.scrape_all_venues(days_ahead=90)

    if not events:
        print("No events found from venue scraping")
        return {'success': False, 'error': 'No events found'}

    print(f"Successfully scraped {len(events)} events")

    # Store in database
    print("Storing events in database...")
    db = VenueScraperDatabase()
    result = db.store_scraped_events(events)

    return result


def main():
    """Main entry point."""
    try:
        result = run_venue_scraping()

        if result['success']:
            print(f"\nSUCCESS: Venue scraping completed")
            print(f"- Stored: {result['stored_events']} new events")
            print(f"- Updated: {result['updated_events']} existing events")
            print(f"- Total database events: {result['total_events']}")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()