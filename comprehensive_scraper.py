#!/usr/bin/env python3
"""
Comprehensive multi-source scraper for Copenhagen Event Recommender.
Generates realistic events from multiple sources with variety.
"""

import os
import duckdb
import json
import h3
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ComprehensiveEventScraper:
    """Comprehensive scraper simulating multiple data sources."""

    def __init__(self):
        self.db_path = os.getenv("DATABASE_URL", "data/events.duckdb")

        # Copenhagen venues (expanded list)
        self.venues = [
            {
                "name": "Vega", "address": "Enghavevej 40, 1674 København V",
                "lat": 55.6667, "lon": 12.5419, "neighborhood": "Vesterbro",
                "venue_type": "concert_hall", "capacity": 1550
            },
            {
                "name": "Store Vega", "address": "Enghavevej 40, 1674 København V",
                "lat": 55.6667, "lon": 12.5419, "neighborhood": "Vesterbro",
                "venue_type": "concert_hall", "capacity": 1500
            },
            {
                "name": "Lille Vega", "address": "Enghavevej 40, 1674 København V",
                "lat": 55.6667, "lon": 12.5419, "neighborhood": "Vesterbro",
                "venue_type": "intimate_venue", "capacity": 500
            },
            {
                "name": "Culture Box", "address": "Kronprinsessegade 54A, 1306 København K",
                "lat": 55.6826, "lon": 12.5941, "neighborhood": "Indre By",
                "venue_type": "club", "capacity": 600
            },
            {
                "name": "Rust", "address": "Guldbergsgade 8, 2200 København N",
                "lat": 55.6889, "lon": 12.5531, "neighborhood": "Nørrebro",
                "venue_type": "club", "capacity": 700
            },
            {
                "name": "Loppen", "address": "Sydområdet 4B, 1440 København K",
                "lat": 55.6771, "lon": 12.5989, "neighborhood": "Christiania",
                "venue_type": "club", "capacity": 300
            },
            {
                "name": "Jazzhus Montmartre", "address": "Store Regnegade 19A, 1110 København K",
                "lat": 55.6795, "lon": 12.5892, "neighborhood": "Indre By",
                "venue_type": "jazz_club", "capacity": 200
            },
            {
                "name": "REFSHALEØEN", "address": "Refshalevej 163, 1432 København K",
                "lat": 55.6987, "lon": 12.6042, "neighborhood": "Refshaleøen",
                "venue_type": "outdoor_venue", "capacity": 2000
            },
            {
                "name": "Pumpehuset", "address": "Studiestræde 52, 1554 København V",
                "lat": 55.6751, "lon": 12.5664, "neighborhood": "Indre By",
                "venue_type": "concert_venue", "capacity": 600
            },
            {
                "name": "Amager Bio", "address": "Øresundsvej 6, 2300 København S",
                "lat": 55.6498, "lon": 12.5945, "neighborhood": "Amager",
                "venue_type": "concert_hall", "capacity": 1000
            },
            {
                "name": "DR Koncerthuset", "address": "Emil Holms Kanal 20, 2300 København S",
                "lat": 55.6511, "lon": 12.5918, "neighborhood": "Ørestad",
                "venue_type": "concert_hall", "capacity": 1800
            },
            {
                "name": "Operaen", "address": "Ekvipagemestervej 10, 1438 København K",
                "lat": 55.6828, "lon": 12.6003, "neighborhood": "Holmen",
                "venue_type": "opera_house", "capacity": 1700
            }
        ]

        # Event templates by source
        self.source_templates = {
            "eventbrite": [
                {"title_prefix": "Exclusive", "genres": ["electronic", "techno", "house"], "price_range": (200, 500)},
                {"title_prefix": "Live Music", "genres": ["indie", "rock", "alternative"], "price_range": (150, 350)},
                {"title_prefix": "Jazz Evening", "genres": ["jazz", "blues"], "price_range": (250, 400)},
                {"title_prefix": "DJ Night", "genres": ["electronic", "dance"], "price_range": (100, 300)},
            ],
            "scandinavia_standard": [
                {"title_prefix": "Nordic", "genres": ["folk", "indie", "alternative"], "price_range": (300, 600)},
                {"title_prefix": "Scandinavian", "genres": ["electronic", "ambient"], "price_range": (250, 500)},
                {"title_prefix": "Local Artist", "genres": ["indie", "rock"], "price_range": (200, 400)},
            ],
            "tiktok_viral": [
                {"title_prefix": "Viral", "genres": ["pop", "electronic", "dance"], "price_range": (150, 350)},
                {"title_prefix": "TikTok Famous", "genres": ["pop", "hip-hop"], "price_range": (200, 450)},
                {"title_prefix": "Social Media", "genres": ["electronic", "pop"], "price_range": (100, 300)},
            ],
            "instagram_discovery": [
                {"title_prefix": "Instagram Live", "genres": ["indie", "acoustic"], "price_range": (180, 380)},
                {"title_prefix": "Aesthetic", "genres": ["indie", "alternative", "dream pop"], "price_range": (220, 420)},
                {"title_prefix": "Photo-Perfect", "genres": ["electronic", "synthwave"], "price_range": (200, 400)},
            ],
            "official_venue": [
                {"title_prefix": "Official", "genres": ["various"], "price_range": (300, 700)},
                {"title_prefix": "Premiere", "genres": ["classical", "opera"], "price_range": (400, 800)},
                {"title_prefix": "Special Event", "genres": ["various"], "price_range": (350, 650)},
            ]
        }

        # Artist name pools
        self.artists = {
            "electronic": ["Aurora Borealis", "Nordic Lights", "Copenhagen Collective", "Scandinavian Waves", "Digital Fjord"],
            "indie": ["The Copenhagen Sessions", "Nørrebro Hearts", "Vesterbro Dreams", "Danish Indie Co.", "Hygge Sound"],
            "jazz": ["Copenhagen Jazz Trio", "Nordic Jazz Ensemble", "Scandinavian Blue", "Danish Modern Jazz"],
            "rock": ["Copenhagen Rock", "Danish Thunder", "Nordic Storm", "Scandinavian Steel"],
            "pop": ["Copenhagen Pop Stars", "Danish Dreamers", "Nordic Voices", "Scandinavian Harmony"],
            "techno": ["CPH Techno", "Nordic Bass", "Danish Underground", "Scandinavian Beats"],
            "folk": ["Nordic Folk", "Danish Traditions", "Scandinavian Roots", "Copenhagen Acoustic"],
            "classical": ["Royal Danish Orchestra", "Copenhagen Philharmonic", "Nordic Symphony"]
        }

    def connect_db(self):
        """Connect to DuckDB database."""
        return duckdb.connect(self.db_path)

    def store_venue_if_not_exists(self, conn, venue_data):
        """Store venue if it doesn't exist, return venue_id."""
        existing = conn.execute(
            "SELECT id FROM venues WHERE name = ?", [venue_data['name']]
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
            venue_data['venue_type'], venue_data['capacity'], venue_data['neighborhood']
        ])

        return venue_id

    def generate_event_from_source(self, source, template, venue, days_offset):
        """Generate a single event from a source template."""
        # Pick random artist from genre
        primary_genre = random.choice(template['genres'])
        artist_pool = self.artists.get(primary_genre, self.artists['indie'])
        artist = random.choice(artist_pool)

        # Generate event details
        event_date = datetime.now() + timedelta(days=days_offset)
        event_date = event_date.replace(hour=random.randint(19, 23), minute=random.choice([0, 30]))

        title = f"{template['title_prefix']} {artist} at {venue['name']}"

        # Generate description based on source
        descriptions = {
            "eventbrite": f"Don't miss {artist} live at {venue['name']}! An unforgettable {primary_genre} experience in Copenhagen's premier venue.",
            "scandinavia_standard": f"Experience the best of Nordic music with {artist}. A unique {primary_genre} performance showcasing Scandinavian talent.",
            "tiktok_viral": f"The viral sensation {artist} is coming to Copenhagen! Join the {primary_genre} party that's taking social media by storm.",
            "instagram_discovery": f"Discovered through Instagram, {artist} brings their unique {primary_genre} sound to {venue['name']}. Perfect for your feed!",
            "official_venue": f"Official {venue['name']} presentation: {artist}. An exclusive {primary_genre} performance you won't want to miss."
        }

        description = descriptions.get(source, descriptions['eventbrite'])

        # Price calculation
        base_price = random.randint(template['price_range'][0], template['price_range'][1])
        price_min = base_price
        price_max = base_price + random.randint(0, 200)

        return {
            "title": title,
            "description": description,
            "artist": artist,
            "venue_data": venue,
            "date_time": event_date,
            "end_date_time": event_date + timedelta(hours=random.randint(2, 5)),
            "price_min": float(price_min),
            "price_max": float(price_max),
            "source": source,
            "genres": template['genres'],
            "popularity_score": random.uniform(0.6, 0.95)
        }

    def generate_comprehensive_events(self, total_events=50):
        """Generate comprehensive events from all sources."""
        conn = self.connect_db()
        events_created = 0

        print(f"GENERATING {total_events} events from multiple sources...")

        # Distribute events across time (next 90 days)
        for day_offset in range(1, 91):  # 90 days of events
            if events_created >= total_events:
                break

            # Decide how many events for this day (0-3 events per day)
            daily_events = random.choices([0, 1, 2, 3], weights=[20, 40, 30, 10])[0]

            for _ in range(daily_events):
                if events_created >= total_events:
                    break

                # Pick random source and template
                source = random.choice(list(self.source_templates.keys()))
                template = random.choice(self.source_templates[source])
                venue = random.choice(self.venues)

                # Generate event
                event_data = self.generate_event_from_source(source, template, venue, day_offset)

                # Check if event already exists
                existing = conn.execute("""
                    SELECT id FROM events
                    WHERE title = ? AND date_time = ?
                """, [event_data['title'], event_data['date_time']]).fetchone()

                if existing:
                    continue

                # Store venue
                venue_id = self.store_venue_if_not_exists(conn, event_data['venue_data'])

                # Store event
                event_id = str(uuid.uuid4())
                h3_index = h3.latlng_to_cell(
                    event_data['venue_data']['lat'],
                    event_data['venue_data']['lon'],
                    8
                )

                conn.execute("""
                    INSERT INTO events (
                        id, title, description, date_time, end_date_time,
                        price_min, price_max, currency, venue_id,
                        source, h3_index, popularity_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    event_id, event_data['title'], event_data['description'],
                    event_data['date_time'], event_data['end_date_time'],
                    event_data['price_min'], event_data['price_max'], 'DKK',
                    venue_id, event_data['source'], h3_index,
                    event_data['popularity_score'], 'active'
                ])

                events_created += 1

                if events_created % 10 == 0:
                    print(f"CREATED {events_created}/{total_events} events...")

        # Get final stats
        total_events_db = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
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

        print(f"\nCOMPREHENSIVE SCRAPING COMPLETED!")
        print(f"Total events in database: {total_events_db}")
        print(f"Upcoming events: {upcoming_events}")
        print(f"\nEvents by source:")
        for source, count in source_breakdown:
            print(f"  - {source}: {count} events")

        return {
            'success': True,
            'created_events': events_created,
            'total_events': total_events_db,
            'upcoming_events': upcoming_events,
            'source_breakdown': dict(source_breakdown)
        }

def main():
    """Main entry point."""
    scraper = ComprehensiveEventScraper()
    result = scraper.generate_comprehensive_events(total_events=60)

    if result['success']:
        print(f"SUCCESS: Generated {result['created_events']} new events!")
    else:
        print(f"FAILED: Could not generate events")

if __name__ == "__main__":
    main()