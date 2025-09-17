#!/usr/bin/env python3
"""
Daily scraper runner for Copenhagen Event Recommender.
Collects live data from Eventbrite and stores in database.
"""

import os
import sys
import logging
import duckdb
import json
import h3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "data-collection"))

# Import scraper directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eventbrite",
    project_root / "data-collection" / "scrapers" / "official_apis" / "eventbrite.py"
)
eventbrite_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eventbrite_module)
EventbriteScraper = eventbrite_module.EventbriteScraper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../data/cache/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EventDataPipeline:
    """Pipeline for collecting and storing event data."""

    def __init__(self):
        self.db_path = os.getenv("DATABASE_URL", "../../data/events/events.duckdb")
        self.eventbrite_token = os.getenv("EVENTBRITE_API_TOKEN")

        if not self.eventbrite_token:
            raise ValueError("EVENTBRITE_API_TOKEN not found in environment")

        # Ensure cache directory exists
        Path("../../data/cache").mkdir(parents=True, exist_ok=True)

    def connect_db(self):
        """Connect to DuckDB database."""
        return duckdb.connect(self.db_path)

    def scrape_eventbrite_events(self, max_events=200):
        """Scrape events from Eventbrite API."""
        logger.info("Starting Eventbrite scraping...")

        scraper = EventbriteScraper(self.eventbrite_token)

        # Search for events in next 60 days
        end_date = datetime.now() + timedelta(days=60)
        events = scraper.search_events(
            categories=["103", "110", "105"],  # Music, Nightlife, Arts
            end_date=end_date,
            max_results=max_events
        )

        logger.info(f"Scraped {len(events)} events from Eventbrite")
        return events

    def store_venue(self, conn, venue_data):
        """Store venue data, return venue_id."""
        venue_id = str(uuid.uuid4())

        # Check if venue already exists by name and address
        existing = conn.execute(
            "SELECT id FROM venues WHERE name = ? AND address = ?",
            [venue_data['name'], venue_data['address']]
        ).fetchone()

        if existing:
            return existing[0]

        # Insert new venue
        h3_index = h3.latlng_to_cell(venue_data['lat'], venue_data['lon'], 8)

        conn.execute("""
            INSERT INTO venues (id, name, address, lat, lon, h3_index, venue_type, capacity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            venue_id,
            venue_data['name'],
            venue_data['address'],
            venue_data['lat'],
            venue_data['lon'],
            h3_index,
            'venue',  # Default type
            None  # Unknown capacity
        ])

        return venue_id

    def store_events(self, events):
        """Store scraped events in database."""
        if not events:
            logger.warning("No events to store")
            return 0

        conn = self.connect_db()
        stored_count = 0

        for event in events:
            try:
                # Check if event already exists
                existing = conn.execute(
                    "SELECT id FROM events WHERE source_url = ? OR title = ? AND date_time = ?",
                    [event.url, event.title, event.start_time]
                ).fetchone()

                if existing:
                    logger.debug(f"Event already exists: {event.title}")
                    continue

                # Store venue
                venue_data = {
                    'name': event.venue_name,
                    'address': event.venue_address,
                    'lat': event.venue_lat,
                    'lon': event.venue_lon
                }
                venue_id = self.store_venue(conn, venue_data)

                # Store event
                event_id = str(uuid.uuid4())
                h3_index = h3.latlng_to_cell(event.venue_lat, event.venue_lon, 8)

                # Convert prices from USD to DKK (rough conversion)
                price_min_dkk = event.price_min * 6.5 if event.price_min else None
                price_max_dkk = event.price_max * 6.5 if event.price_max else None

                conn.execute("""
                    INSERT INTO events (
                        id, title, description, date_time, end_date_time,
                        price_min, price_max, currency, venue_id,
                        source, source_url, image_url, h3_index,
                        popularity_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    event_id,
                    event.title,
                    event.description,
                    event.start_time,
                    event.end_time,
                    price_min_dkk,
                    price_max_dkk,
                    'DKK',
                    venue_id,
                    'eventbrite',
                    event.url,
                    event.image_url,
                    h3_index,
                    0.7,  # Default popularity
                    'active'
                ])

                stored_count += 1
                logger.debug(f"Stored event: {event.title}")

            except Exception as e:
                logger.error(f"Failed to store event {event.title}: {e}")
                continue

        conn.close()
        logger.info(f"Stored {stored_count} new events in database")
        return stored_count

    def cleanup_old_events(self, days_old=7):
        """Remove events that have ended more than specified days ago."""
        conn = self.connect_db()

        cutoff_date = datetime.now() - timedelta(days=days_old)

        result = conn.execute("""
            DELETE FROM events
            WHERE date_time < ? AND status = 'active'
        """, [cutoff_date])

        deleted_count = result.fetchone()
        conn.close()

        logger.info(f"Cleaned up old events (older than {days_old} days)")
        return deleted_count

    def run_daily_scrape(self):
        """Run the complete daily scraping pipeline."""
        logger.info("=== Starting daily scraping pipeline ===")

        try:
            # Scrape Eventbrite events
            events = self.scrape_eventbrite_events(max_events=200)

            # Store events in database
            stored_count = self.store_events(events)

            # Cleanup old events
            self.cleanup_old_events(days_old=7)

            # Get updated stats
            conn = self.connect_db()
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            upcoming_events = conn.execute(
                "SELECT COUNT(*) FROM events WHERE date_time > ?",
                [datetime.now()]
            ).fetchone()[0]
            conn.close()

            logger.info(f"=== Scraping completed successfully ===")
            logger.info(f"Stored: {stored_count} new events")
            logger.info(f"Database: {total_events} total, {upcoming_events} upcoming")

            return {
                'success': True,
                'scraped_events': len(events),
                'stored_events': stored_count,
                'total_events': total_events,
                'upcoming_events': upcoming_events
            }

        except Exception as e:
            logger.error(f"Scraping pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main entry point for scraper runner."""
    try:
        pipeline = EventDataPipeline()
        result = pipeline.run_daily_scrape()

        if result['success']:
            print(f"SUCCESS: Scraping successful: {result['stored_events']} new events stored")
            sys.exit(0)
        else:
            print(f"FAILED: Scraping failed: {result['error']}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Runner failed: {e}")
        print(f"ERROR: Runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()