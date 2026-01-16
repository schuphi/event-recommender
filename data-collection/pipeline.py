#!/usr/bin/env python3
"""
Unified event ingestion pipeline.

All scrapers feed into this pipeline which:
1. Normalizes event data to common schema
2. Classifies events by topic (tech, nightlife, music, sports)
3. Suggests tags (free, outdoor, etc.)
4. Deduplicates events
5. Stores in database
"""

import logging
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.classifiers.topic_classifier import TopicClassifier, TOPICS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NormalizedEvent:
    """Normalized event schema that all scrapers produce."""

    # Required fields
    title: str
    description: str
    date_time: datetime
    venue_name: str
    source: str  # eventbrite, meetup, ticketmaster, luma, etc.

    # Optional fields
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    end_date_time: Optional[datetime] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    currency: str = "DKK"
    venue_address: Optional[str] = None
    venue_lat: Optional[float] = None
    venue_lon: Optional[float] = None
    venue_neighborhood: Optional[str] = None
    image_url: Optional[str] = None

    # Will be set by pipeline
    topic: Optional[str] = None
    tags: Optional[List[str]] = None
    is_free: bool = False


class EventIngestionPipeline:
    """
    Pipeline for ingesting events from various sources.

    Usage:
        pipeline = EventIngestionPipeline()

        # Add events from any scraper
        for event in scraper.fetch_events():
            normalized = NormalizedEvent(
                title=event['name'],
                description=event['desc'],
                ...
            )
            pipeline.ingest(normalized)

        # Commit all events
        pipeline.commit()
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize pipeline.

        Args:
            db_path: Path to DuckDB database. If None, uses default.
        """
        self.db_path = db_path or str(project_root / "data" / "events.duckdb")
        self.classifier = TopicClassifier(use_embeddings=False)  # Fast mode
        self.pending_events: List[NormalizedEvent] = []
        self.stats = {
            "processed": 0,
            "duplicates": 0,
            "errors": 0,
            "by_topic": {topic: 0 for topic in TOPICS},
            "by_source": {},
        }

        # Ensure database exists
        self._init_database()

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        import duckdb

        conn = duckdb.connect(self.db_path)

        # Create venues table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS venues (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                lat REAL,
                lon REAL,
                h3_index TEXT,
                neighborhood TEXT,
                venue_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create events table with topic support
        conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                date_time TIMESTAMP NOT NULL,
                end_date_time TIMESTAMP,
                price_min REAL,
                price_max REAL,
                currency TEXT DEFAULT 'DKK',
                topic TEXT DEFAULT 'music',
                tags TEXT DEFAULT '[]',
                is_free BOOLEAN DEFAULT FALSE,
                venue_id TEXT,
                source TEXT NOT NULL,
                source_id TEXT,
                source_url TEXT,
                image_url TEXT,
                h3_index TEXT,
                popularity_score REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_datetime ON events(date_time)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_events_source_id ON events(source, source_id)')

        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def ingest(self, event: NormalizedEvent) -> bool:
        """
        Ingest a single event.

        Args:
            event: Normalized event to ingest

        Returns:
            True if event was added, False if duplicate or error
        """
        try:
            # Classify topic
            result = self.classifier.classify(
                event.title,
                event.description or "",
            )
            event.topic = result.topic

            # Suggest tags
            event.tags = self.classifier.suggest_tags(
                event.title,
                event.description or "",
                event.price_min,
            )

            # Determine if free
            event.is_free = (
                event.price_min is None or
                event.price_min == 0 or
                "free" in event.tags
            )

            # Add to pending
            self.pending_events.append(event)
            self.stats["processed"] += 1
            self.stats["by_topic"][event.topic] = self.stats["by_topic"].get(event.topic, 0) + 1
            self.stats["by_source"][event.source] = self.stats["by_source"].get(event.source, 0) + 1

            logger.debug(f"Ingested: {event.title} -> {event.topic}")
            return True

        except Exception as e:
            logger.error(f"Error ingesting event '{event.title}': {e}")
            self.stats["errors"] += 1
            return False

    def commit(self) -> Dict[str, Any]:
        """
        Commit all pending events to database.

        Returns:
            Stats dictionary with counts
        """
        import duckdb

        if not self.pending_events:
            logger.info("No events to commit")
            return self.stats

        conn = duckdb.connect(self.db_path)

        inserted = 0
        duplicates = 0

        for event in self.pending_events:
            try:
                # Check for duplicate (same source + source_id, or same title + date + venue)
                if event.source_id:
                    existing = conn.execute(
                        "SELECT id FROM events WHERE source = ? AND source_id = ?",
                        [event.source, event.source_id]
                    ).fetchone()
                else:
                    existing = conn.execute(
                        """SELECT id FROM events
                           WHERE title = ? AND date_time = ? AND source = ?""",
                        [event.title, event.date_time, event.source]
                    ).fetchone()

                if existing:
                    duplicates += 1
                    continue

                # Get or create venue
                venue_id = self._get_or_create_venue(conn, event)

                # Insert event
                event_id = str(uuid.uuid4())

                conn.execute('''
                    INSERT INTO events (
                        id, title, description, date_time, end_date_time,
                        price_min, price_max, currency, topic, tags, is_free,
                        venue_id, source, source_id, source_url, image_url, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    event_id,
                    event.title,
                    event.description,
                    event.date_time,
                    event.end_date_time,
                    event.price_min,
                    event.price_max,
                    event.currency,
                    event.topic,
                    json.dumps(event.tags or []),
                    event.is_free,
                    venue_id,
                    event.source,
                    event.source_id,
                    event.source_url,
                    event.image_url,
                    'active'
                ])

                inserted += 1

            except Exception as e:
                logger.error(f"Error inserting event '{event.title}': {e}")
                self.stats["errors"] += 1

        conn.commit()
        conn.close()

        self.stats["duplicates"] = duplicates
        self.stats["inserted"] = inserted

        logger.info(f"Committed {inserted} events ({duplicates} duplicates skipped)")

        # Clear pending
        self.pending_events = []

        return self.stats

    def _get_or_create_venue(self, conn, event: NormalizedEvent) -> str:
        """Get existing venue or create new one."""

        if not event.venue_name:
            return None

        # Check if venue exists
        existing = conn.execute(
            "SELECT id FROM venues WHERE name = ?",
            [event.venue_name]
        ).fetchone()

        if existing:
            return existing[0]

        # Create new venue
        venue_id = str(uuid.uuid4())

        # Calculate H3 index if coordinates available
        h3_index = None
        if event.venue_lat and event.venue_lon:
            try:
                import h3
                h3_index = h3.geo_to_h3(event.venue_lat, event.venue_lon, 8)
            except:
                pass

        conn.execute('''
            INSERT INTO venues (id, name, address, lat, lon, h3_index, neighborhood)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [
            venue_id,
            event.venue_name,
            event.venue_address,
            event.venue_lat,
            event.venue_lon,
            h3_index,
            event.venue_neighborhood,
        ])

        return venue_id

    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        return self.stats


# Convenience function for simple usage
def ingest_events(events: List[Dict], source: str, db_path: Optional[str] = None) -> Dict:
    """
    Simple function to ingest a list of event dictionaries.

    Args:
        events: List of event dictionaries
        source: Source name (eventbrite, meetup, etc.)
        db_path: Optional database path

    Returns:
        Stats dictionary
    """
    pipeline = EventIngestionPipeline(db_path)

    for event_dict in events:
        try:
            normalized = NormalizedEvent(
                title=event_dict.get('title', event_dict.get('name', '')),
                description=event_dict.get('description', ''),
                date_time=event_dict.get('date_time', event_dict.get('start_time')),
                venue_name=event_dict.get('venue_name', event_dict.get('venue', {}).get('name', '')),
                source=source,
                source_id=event_dict.get('id', event_dict.get('source_id')),
                source_url=event_dict.get('url', event_dict.get('source_url')),
                end_date_time=event_dict.get('end_date_time', event_dict.get('end_time')),
                price_min=event_dict.get('price_min', event_dict.get('price', {}).get('min')),
                price_max=event_dict.get('price_max', event_dict.get('price', {}).get('max')),
                currency=event_dict.get('currency', 'DKK'),
                venue_address=event_dict.get('venue_address', event_dict.get('venue', {}).get('address')),
                venue_lat=event_dict.get('venue_lat', event_dict.get('venue', {}).get('lat')),
                venue_lon=event_dict.get('venue_lon', event_dict.get('venue', {}).get('lon')),
                venue_neighborhood=event_dict.get('venue_neighborhood'),
                image_url=event_dict.get('image_url', event_dict.get('image')),
            )
            pipeline.ingest(normalized)
        except Exception as e:
            logger.error(f"Error normalizing event: {e}")

    return pipeline.commit()


if __name__ == "__main__":
    # Test the pipeline
    test_events = [
        {
            "title": "Copenhagen AI Meetup",
            "description": "Join us for talks on machine learning, OpenAI, and the future of AI",
            "date_time": datetime.now(),
            "venue_name": "Founders House",
            "venue_address": "Njalsgade 19D, Copenhagen",
            "price_min": 0,
        },
        {
            "title": "Techno Night at Culture Box",
            "description": "Underground electronic music with international DJs",
            "date_time": datetime.now(),
            "venue_name": "Culture Box",
            "venue_address": "Kronprinsessegade 54A, Copenhagen",
            "price_min": 150,
            "price_max": 200,
        },
    ]

    stats = ingest_events(test_events, source="test")
    print(f"Pipeline stats: {json.dumps(stats, indent=2)}")
