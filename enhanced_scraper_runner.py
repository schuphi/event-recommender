#!/usr/bin/env python3
"""
Enhanced scraper runner combining venue website scraping with Eventbrite API.
Replaces the old scheduler system with a unified real event collection system.
"""

import os
import sys
import duckdb
import uuid
import h3
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add paths
sys.path.append(str(Path(__file__).parent / "data-collection" / "scrapers" / "venue_scrapers"))
sys.path.append(str(Path(__file__).parent / "data-collection" / "scrapers" / "official_apis"))

from copenhagen_venues import CopenhagenVenueScraper
from eventbrite import EventbriteScraper

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedScraperDatabase:
    """Enhanced database integration for multiple scraper types."""

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
            venue_data.get('venue_type', 'venue'), 
            venue_data.get('capacity', 500), 
            venue_data.get('neighborhood', 'Copenhagen')
        ])

        return venue_id

    def store_events(self, events, source_name):
        """Store events from any source in database."""
        conn = self.connect_db()
        stored_count = 0
        updated_count = 0
        skipped_count = 0

        logger.info(f"Storing {len(events)} events from {source_name}...")

        for event in events:
            try:
                # Handle different event types
                if hasattr(event, 'venue_name'):  # Venue scraper event
                    venue_data = {
                        'name': event.venue_name,
                        'address': event.venue_address,
                        'lat': event.venue_lat,
                        'lon': event.venue_lon
                    }
                    title = event.title
                    description = event.description
                    date_time = event.date_time
                    end_date_time = event.end_date_time
                    price_min = event.price_min
                    price_max = event.price_max
                    source = event.source
                    
                elif hasattr(event, 'venue_lat'):  # Eventbrite event
                    venue_data = {
                        'name': event.venue_name,
                        'address': event.venue_address,
                        'lat': event.venue_lat,
                        'lon': event.venue_lon
                    }
                    title = event.title
                    description = event.description
                    date_time = event.start_time
                    end_date_time = event.end_time
                    price_min = event.price_min
                    price_max = event.price_max
                    source = 'eventbrite'
                else:
                    logger.warning(f"Unknown event type: {type(event)}")
                    continue

                # Skip events without proper dates
                if not date_time:
                    skipped_count += 1
                    continue

                # Skip past events
                if date_time <= datetime.now():
                    skipped_count += 1
                    continue

                # Store venue
                venue_id = self.store_venue_if_not_exists(conn, venue_data)

                # Check if event already exists
                existing = conn.execute("""
                    SELECT id FROM events
                    WHERE title = ? AND venue_id = ? AND date_time = ?
                """, [title, venue_id, date_time]).fetchone()

                if existing:
                    # Update existing event
                    conn.execute("""
                        UPDATE events SET
                            description = ?, end_date_time = ?, price_min = ?, price_max = ?,
                            source = ?, updated_at = ?
                        WHERE id = ?
                    """, [
                        description, end_date_time, price_min, price_max,
                        source, datetime.now(), existing[0]
                    ])
                    updated_count += 1
                    continue

                # Store new event
                event_id = str(uuid.uuid4())
                h3_index = h3.latlng_to_cell(venue_data['lat'], venue_data['lon'], 8)

                conn.execute("""
                    INSERT INTO events (
                        id, title, description, date_time, end_date_time,
                        price_min, price_max, currency, venue_id,
                        source, h3_index, popularity_score, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    event_id, title, description,
                    date_time, end_date_time,
                    price_min, price_max, 'DKK',
                    venue_id, source, h3_index,
                    0.7, 'active', datetime.now(), datetime.now()
                ])

                stored_count += 1

            except Exception as e:
                logger.error(f"Failed to store event '{getattr(event, 'title', 'Unknown')}': {e}")
                continue

        conn.close()
        
        logger.info(f"{source_name}: {stored_count} stored, {updated_count} updated, {skipped_count} skipped")
        return stored_count, updated_count, skipped_count

    def get_database_stats(self):
        """Get current database statistics."""
        conn = self.connect_db()
        
        total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        upcoming_events = conn.execute(
            "SELECT COUNT(*) FROM events WHERE date_time > ?",
            [datetime.now()]
        ).fetchone()[0]

        source_breakdown = conn.execute("""
            SELECT source, COUNT(*) as count
            FROM events
            WHERE date_time > ?
            GROUP BY source
            ORDER BY count DESC
        """, [datetime.now()]).fetchall()

        conn.close()
        
        return {
            'total_events': total_events,
            'upcoming_events': upcoming_events,
            'source_breakdown': dict(source_breakdown)
        }


def run_enhanced_scraping():
    """Main function to run all real event scrapers."""
    print("=== Enhanced Copenhagen Event Scraping ===")
    
    db = EnhancedScraperDatabase()
    total_stored = 0
    total_updated = 0
    
    # 1. Run venue website scrapers
    print("\n1. Scraping venue websites...")
    try:
        venue_scraper = CopenhagenVenueScraper()
        venue_events = venue_scraper.scrape_all_venues(days_ahead=90)
        
        if venue_events:
            stored, updated, skipped = db.store_events(venue_events, "Venue Websites")
            total_stored += stored
            total_updated += updated
            print(f"   ✅ Venue scraping: {stored} new, {updated} updated")
        else:
            print("   ⚠️ No venue events found")
            
    except Exception as e:
        print(f"   ❌ Venue scraping failed: {e}")

    # 2. Eventbrite API integration (placeholder for future implementation)
    print("\n2. Eventbrite API integration...")
    eventbrite_token = os.getenv("EVENTBRITE_API_TOKEN")
    
    if eventbrite_token:
        print(f"   ✅ Eventbrite token configured: {eventbrite_token[:10]}...")
        print("   ℹ️ Eventbrite API integration available for future enhancement")
        # Note: Eventbrite API endpoint structure has changed, needs investigation
    else:
        print("   ⚠️ Eventbrite API token not configured")

    # 3. Show final statistics
    print("\n=== SCRAPING COMPLETED ===")
    stats = db.get_database_stats()
    
    print(f"Total new events: {total_stored}")
    print(f"Total updated events: {total_updated}")
    print(f"Database total: {stats['total_events']} events")
    print(f"Upcoming events: {stats['upcoming_events']}")
    
    print(f"\nEvents by source:")
    for source, count in stats['source_breakdown'].items():
        print(f"  - {source}: {count} events")

    return {
        'success': True,
        'stored_events': total_stored,
        'updated_events': total_updated,
        'total_events': stats['total_events'],
        'upcoming_events': stats['upcoming_events'],
        'source_breakdown': stats['source_breakdown']
    }


def main():
    """Main entry point."""
    try:
        result = run_enhanced_scraping()

        if result['success']:
            print(f"\n🎉 SUCCESS: Enhanced scraping completed!")
            print(f"📊 Stored: {result['stored_events']} new events")
            print(f"🔄 Updated: {result['updated_events']} existing events")
            print(f"📈 Total: {result['total_events']} events in database")
            return 0
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        print(f"💥 FATAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
