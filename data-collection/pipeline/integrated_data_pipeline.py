#!/usr/bin/env python3
"""
Integrated data collection pipeline for Copenhagen Event Recommender.
Orchestrates data collection, validation, enrichment, and deduplication.
"""

import logging
import json
import duckdb
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Import all our components
from ..scrapers.official_apis.eventbrite import EventbriteScraper, EventbriteEvent
from ..scrapers.official_apis.meetup import MeetupScraper, MeetupEvent
from ..scrapers.social_scrapers.viral_discovery import ViralEventDiscoveryEngine
from ..scrapers.social_scrapers.instagram import InstagramEventScraper, InstagramEvent
from ..scrapers.social_scrapers.tiktok import TikTokEventScraper, TikTokEvent
from ..validation.data_validator import EventDataValidator, ValidationStatus
from ..enrichment.artist_genre_enricher import ArtistGenreEnricher
from ..deduplication.duplicate_detector import EventDuplicateDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for data collection pipeline."""
    
    # Database
    db_path: str = "data/events.duckdb"
    
    # API credentials
    eventbrite_token: Optional[str] = None
    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    lastfm_api_key: Optional[str] = None
    
    # Social media credentials
    instagram_username: Optional[str] = None
    instagram_password: Optional[str] = None
    
    # Collection parameters
    days_ahead: int = 60
    max_events_per_source: int = 1000
    enable_viral_discovery: bool = True
    enable_instagram_scraping: bool = True
    enable_tiktok_scraping: bool = True
    enable_enrichment: bool = True
    enable_deduplication: bool = True
    
    # Social scraping parameters
    instagram_venues: Optional[List[str]] = None
    tiktok_venues: Optional[List[str]] = None
    social_days_back: int = 14
    
    # Quality thresholds
    min_confidence_score: float = 0.3
    duplicate_similarity_threshold: float = 0.7
    
    # Output
    output_dir: str = "data-collection/output"
    save_raw_data: bool = True
    save_processed_data: bool = True

@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""
    
    # Source statistics
    eventbrite_events: int = 0
    meetup_events: int = 0
    instagram_events: int = 0
    tiktok_events: int = 0
    viral_events: int = 0
    total_raw_events: int = 0
    
    # Processing statistics
    validation_passed: int = 0
    validation_failed: int = 0
    enrichment_successful: int = 0
    duplicates_removed: int = 0
    final_unique_events: int = 0
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Quality metrics
    avg_confidence_score: float = 0.0
    enrichment_coverage: float = 0.0
    duplicate_detection_accuracy: float = 0.0

class IntegratedDataPipeline:
    """Integrated data collection and processing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.eventbrite_scraper = None
        if config.eventbrite_token:
            self.eventbrite_scraper = EventbriteScraper(config.eventbrite_token)
        
        self.meetup_scraper = MeetupScraper()
        
        self.instagram_scraper = None
        if config.enable_instagram_scraping:
            self.instagram_scraper = InstagramEventScraper(
                username=config.instagram_username,
                password=config.instagram_password
            )
        
        self.tiktok_scraper = None
        if config.enable_tiktok_scraping:
            self.tiktok_scraper = TikTokEventScraper()
        
        self.viral_engine = None
        if config.enable_viral_discovery:
            self.viral_engine = ViralEventDiscoveryEngine()
        
        self.validator = EventDataValidator()
        
        self.enricher = None
        if config.enable_enrichment:
            self.enricher = ArtistGenreEnricher(
                spotify_client_id=config.spotify_client_id,
                spotify_client_secret=config.spotify_client_secret,
                lastfm_api_key=config.lastfm_api_key
            )
        
        self.duplicate_detector = None
        if config.enable_deduplication:
            self.duplicate_detector = EventDuplicateDetector()
        
        # Statistics
        self.stats = PipelineStats()
    
    def run_pipeline(self) -> PipelineStats:
        """
        Run the complete data collection pipeline.
        
        Returns:
            PipelineStats with execution statistics
        """
        
        logger.info("Starting integrated data collection pipeline")
        self.stats.start_time = datetime.now()
        
        try:
            # Step 1: Data Collection
            logger.info("Step 1: Collecting raw event data...")
            raw_events = self._collect_all_events()
            self.stats.total_raw_events = len(raw_events)
            
            if self.config.save_raw_data:
                self._save_raw_data(raw_events)
            
            # Step 2: Data Validation
            logger.info("Step 2: Validating and cleaning event data...")
            validated_events = self._validate_events(raw_events)
            
            # Step 3: Artist/Genre Enrichment
            if self.enricher:
                logger.info("Step 3: Enriching artist and genre information...")
                enriched_events = self._enrich_events(validated_events)
            else:
                enriched_events = validated_events
                logger.info("Step 3: Skipping enrichment (disabled)")
            
            # Step 4: Duplicate Detection and Removal
            if self.duplicate_detector:
                logger.info("Step 4: Detecting and removing duplicates...")
                unique_events = self._deduplicate_events(enriched_events)
            else:
                unique_events = enriched_events
                logger.info("Step 4: Skipping deduplication (disabled)")
            
            # Step 5: Database Storage
            logger.info("Step 5: Storing events in database...")
            stored_count = self._store_events(unique_events)
            self.stats.final_unique_events = stored_count
            
            # Step 6: Generate Report
            logger.info("Step 6: Generating pipeline report...")
            self._generate_report()
            
            if self.config.save_processed_data:
                self._save_processed_data(unique_events)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                duration = self.stats.end_time - self.stats.start_time
                self.stats.duration_seconds = duration.total_seconds()
        
        logger.info(f"Pipeline completed in {self.stats.duration_seconds:.1f} seconds")
        logger.info(f"Processed {self.stats.total_raw_events} → {self.stats.final_unique_events} events")
        
        return self.stats
    
    def _collect_all_events(self) -> List[Dict]:
        """Collect events from all enabled sources."""
        
        all_events = []
        
        # Date range for collection
        start_date = datetime.now()
        end_date = start_date + timedelta(days=self.config.days_ahead)
        
        # Collect from sources in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            
            # Eventbrite
            if self.eventbrite_scraper:
                future = executor.submit(
                    self._collect_eventbrite_events, 
                    start_date, 
                    end_date
                )
                futures.append(('eventbrite', future))
            
            # Meetup
            future = executor.submit(
                self._collect_meetup_events, 
                start_date, 
                end_date
            )
            futures.append(('meetup', future))
            
            # Instagram
            if self.instagram_scraper:
                future = executor.submit(self._collect_instagram_events)
                futures.append(('instagram', future))
            
            # TikTok
            if self.tiktok_scraper:
                future = executor.submit(self._collect_tiktok_events)
                futures.append(('tiktok', future))
            
            # Viral Discovery
            if self.viral_engine:
                future = executor.submit(self._collect_viral_events)
                futures.append(('viral', future))
            
            # Collect results
            for source, future in futures:
                try:
                    events = future.result(timeout=300)  # 5 minute timeout
                    
                    if source == 'eventbrite':
                        self.stats.eventbrite_events = len(events)
                    elif source == 'meetup':
                        self.stats.meetup_events = len(events)
                    elif source == 'instagram':
                        self.stats.instagram_events = len(events)
                    elif source == 'tiktok':
                        self.stats.tiktok_events = len(events)
                    elif source == 'viral':
                        self.stats.viral_events = len(events)
                    
                    all_events.extend(events)
                    logger.info(f"Collected {len(events)} events from {source}")
                    
                except Exception as e:
                    logger.error(f"Failed to collect from {source}: {e}")
        
        logger.info(f"Total raw events collected: {len(all_events)}")
        return all_events
    
    def _collect_eventbrite_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect events from Eventbrite."""
        
        events = self.eventbrite_scraper.search_events(
            start_date=start_date,
            end_date=end_date,
            max_results=self.config.max_events_per_source
        )
        
        # Convert to standard format
        standardized_events = []
        for event in events:
            standardized = {
                'id': f"eventbrite_{event.id}",
                'title': event.title,
                'description': event.description,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'venue_name': event.venue_name,
                'venue_address': event.venue_address,
                'venue_lat': event.venue_lat,
                'venue_lon': event.venue_lon,
                'price_min': event.price_min,
                'price_max': event.price_max,
                'source_url': event.url,
                'image_url': event.image_url,
                'source': 'eventbrite',
                'artists': [],  # Will be enriched later
                'genres': []    # Will be enriched later
            }
            standardized_events.append(standardized)
        
        return standardized_events
    
    def _collect_meetup_events(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Collect events from Meetup."""
        
        events = self.meetup_scraper.search_events(
            start_date=start_date,
            end_date=end_date,
            max_results=self.config.max_events_per_source
        )
        
        # Convert to standard format
        standardized_events = []
        for event in events:
            standardized = {
                'id': f"meetup_{event.id}",
                'title': event.title,
                'description': event.description,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'venue_name': event.venue_name,
                'venue_address': event.venue_address,
                'venue_lat': event.venue_lat,
                'venue_lon': event.venue_lon,
                'price_min': None,  # Meetup events typically free
                'price_max': None,
                'source_url': event.url,
                'image_url': event.image_url,
                'source': 'meetup',
                'popularity_score': min(1.0, event.attendee_count / 100.0),  # Normalize attendance
                'artists': [],  # Will be enriched later
                'genres': []    # Will be enriched later
            }
            standardized_events.append(standardized)
        
        return standardized_events
    
    def _collect_instagram_events(self) -> List[Dict]:
        """Collect events from Instagram venue accounts."""
        
        venue_usernames = self.config.instagram_venues or [
            'vega_copenhagen', 'rust_cph', 'culturebox_cph', 'loppen_official',
            'jolene_cph', 'kb18_cph', 'pumpehuset', 'amager_bio', 'alice_cph'
        ]
        
        events = self.instagram_scraper.scrape_venues(
            venue_usernames=venue_usernames,
            days_back=self.config.social_days_back,
            max_posts_per_venue=20
        )
        
        # Convert to standard format
        standardized_events = []
        for event in events:
            standardized = {
                'id': f"instagram_{event.id}",
                'title': event.title,
                'description': event.description,
                'start_time': event.date_time or (datetime.now() + timedelta(days=1)),
                'end_time': None,
                'venue_name': event.venue_name,
                'venue_address': f"Copenhagen, Denmark",  # Default for Instagram events
                'venue_lat': None,  # Would need geocoding
                'venue_lon': None,
                'price_min': None,  # Unknown for Instagram events
                'price_max': None,
                'source_url': event.post_url,
                'image_url': event.image_url,
                'source': 'instagram',
                'popularity_score': min(1.0, event.likes / 1000.0),  # Normalize likes
                'artists': event.detected_artists,
                'genres': event.detected_genres
            }
            standardized_events.append(standardized)
        
        return standardized_events
    
    def _collect_tiktok_events(self) -> List[Dict]:
        """Collect events from TikTok venue accounts and viral hashtags."""
        
        venue_usernames = self.config.tiktok_venues or [
            'vegacph', 'rustcph', 'cultureboxcph', 'loppencph',
            'jolenecph', 'kb18cph', 'pumpehusetcph', 'amagerbio'
        ]
        
        events = self.tiktok_scraper.scrape_venues(
            venue_usernames=venue_usernames,
            days_back=self.config.social_days_back,
            max_videos_per_venue=15
        )
        
        # Convert to standard format
        standardized_events = []
        for event in events:
            standardized = {
                'id': f"tiktok_{event.id}",
                'title': event.title,
                'description': event.description,
                'start_time': event.date_time or (datetime.now() + timedelta(days=1)),
                'end_time': None,
                'venue_name': event.venue_name,
                'venue_address': f"Copenhagen, Denmark",  # Default for TikTok events
                'venue_lat': None,  # Would need geocoding
                'venue_lon': None,
                'price_min': None,  # Unknown for TikTok events
                'price_max': None,
                'source_url': event.video_url,
                'image_url': event.thumbnail_url,
                'source': 'tiktok',
                'popularity_score': min(1.0, (event.views + event.likes * 10) / 100000.0),  # Weighted popularity
                'artists': event.detected_artists,
                'genres': event.detected_genres
            }
            standardized_events.append(standardized)
        
        return standardized_events
    
    def _collect_viral_events(self) -> List[Dict]:
        """Collect viral events from social media."""
        
        trending_events = self.viral_engine.discover_viral_events(
            days_back=3,
            min_viral_score=0.5,
            max_events=self.config.max_events_per_source
        )
        
        # Convert to standard format
        standardized_events = []
        for event in trending_events:
            # Use primary signal for basic info
            primary_signal = max(event.platform_signals, key=lambda s: s.engagement_score)
            
            standardized = {
                'id': f"viral_{event.event_key}",
                'title': event.title,
                'description': primary_signal.description,
                'start_time': datetime.now() + timedelta(hours=24),  # Assume next day
                'end_time': None,
                'venue_name': event.venue_name,
                'venue_address': event.location,
                'venue_lat': None,  # Would need geocoding
                'venue_lon': None,
                'price_min': None,  # Unknown for viral events
                'price_max': None,
                'source_url': '',
                'image_url': None,
                'source': 'viral_discovery',
                'popularity_score': event.viral_score,
                'artists': [],  # Will be enriched later
                'genres': []    # Will be enriched later
            }
            standardized_events.append(standardized)
        
        return standardized_events
    
    def _validate_events(self, events: List[Dict]) -> List[Dict]:
        """Validate and clean event data."""
        
        validated_events = []
        
        for event in events:
            try:
                result = self.validator.validate_event(event)
                
                if result.status == ValidationStatus.VALID:
                    validated_events.append(result.cleaned_data)
                    self.stats.validation_passed += 1
                
                elif result.status == ValidationStatus.NEEDS_ENRICHMENT:
                    # Keep events that need enrichment
                    validated_events.append(result.cleaned_data or event)
                    self.stats.validation_passed += 1
                
                elif result.status == ValidationStatus.SUSPICIOUS:
                    # Keep suspicious events if confidence is above threshold
                    if result.confidence_score >= self.config.min_confidence_score:
                        validated_events.append(result.cleaned_data or event)
                        self.stats.validation_passed += 1
                    else:
                        self.stats.validation_failed += 1
                        logger.warning(f"Rejected suspicious event: {event.get('title', 'Unknown')}")
                
                else:
                    # Reject invalid or duplicate events
                    self.stats.validation_failed += 1
                    logger.warning(f"Rejected invalid event: {event.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Validation failed for event {event.get('id', 'unknown')}: {e}")
                self.stats.validation_failed += 1
        
        logger.info(f"Validation: {self.stats.validation_passed} passed, {self.stats.validation_failed} failed")
        return validated_events
    
    def _enrich_events(self, events: List[Dict]) -> List[Dict]:
        """Enrich events with artist and genre information."""
        
        enriched_events = []
        successful_enrichments = 0
        
        for event in events:
            try:
                result = self.enricher.enrich_event_artists(event, deep_enrichment=True)
                
                if result.enriched_artists:
                    # Update event with enriched data
                    event['artists'] = [artist.canonical_name for artist in result.enriched_artists]
                    event['genres'] = result.enhanced_genres
                    
                    # Add artist popularity scores
                    if result.enriched_artists:
                        avg_popularity = sum(a.popularity_score for a in result.enriched_artists) / len(result.enriched_artists)
                        event['popularity_score'] = max(event.get('popularity_score', 0), avg_popularity)
                    
                    successful_enrichments += 1
                
                enriched_events.append(event)
                
            except Exception as e:
                logger.warning(f"Enrichment failed for event {event.get('id', 'unknown')}: {e}")
                enriched_events.append(event)  # Keep original event
        
        self.stats.enrichment_successful = successful_enrichments
        self.stats.enrichment_coverage = successful_enrichments / len(events) if events else 0
        
        logger.info(f"Enrichment: {successful_enrichments}/{len(events)} events enriched")
        return enriched_events
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """Remove duplicate events."""
        
        if len(events) < 2:
            return events
        
        try:
            result = self.duplicate_detector.detect_duplicates(events)
            
            # Create set of IDs to remove (keep first in each group)
            ids_to_remove = set()
            for group in result.duplicate_groups:
                if len(group) > 1:
                    # Remove all but the first event in each group
                    ids_to_remove.update(group[1:])
            
            # Filter out duplicate events
            unique_events = []
            for event in events:
                event_id = event.get('id', '')
                if event_id not in ids_to_remove:
                    unique_events.append(event)
            
            self.stats.duplicates_removed = result.duplicates_found
            self.stats.duplicate_detection_accuracy = result.confidence_scores.get('overall_confidence', 0.0)
            
            logger.info(f"Deduplication: Removed {result.duplicates_found} duplicates in {len(result.duplicate_groups)} groups")
            
            return unique_events
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            return events  # Return original events if deduplication fails
    
    def _store_events(self, events: List[Dict]) -> int:
        """Store events in DuckDB database."""
        
        if not events:
            logger.warning("No events to store")
            return 0
        
        try:
            # Ensure database directory exists
            db_path = Path(self.config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            conn = duckdb.connect(str(db_path))
            
            # Create tables if they don't exist
            self._create_database_schema(conn)
            
            stored_count = 0
            
            for event in events:
                try:
                    # Insert event
                    event_values = self._prepare_event_for_db(event)
                    
                    conn.execute("""
                        INSERT OR REPLACE INTO events 
                        (id, title, description, date_time, end_date_time, price_min, price_max, 
                         venue_id, artist_ids, popularity_score, h3_index, source, source_url, 
                         image_url, status, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, event_values)
                    
                    stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to store event {event.get('id', 'unknown')}: {e}")
            
            conn.close()
            logger.info(f"Stored {stored_count} events in database")
            return stored_count
            
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            return 0
    
    def _create_database_schema(self, conn):
        """Create database schema if it doesn't exist."""
        
        # Events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
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
        
        # Venues table  
        conn.execute("""
            CREATE TABLE IF NOT EXISTS venues (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                address VARCHAR,
                lat DOUBLE,
                lon DOUBLE,
                neighborhood VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Artists table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artists (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                genres JSON,
                popularity_score FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _prepare_event_for_db(self, event: Dict) -> Tuple:
        """Prepare event data for database insertion."""
        
        # Generate venue ID and store venue
        venue_id = self._store_venue(event)
        
        # Store artists and get IDs
        artist_ids = self._store_artists(event)
        
        # Calculate H3 index
        h3_index = event.get('venue_h3_index', '')
        
        return (
            event.get('id'),
            event.get('title'),
            event.get('description'),
            event.get('start_time'),
            event.get('end_time'),
            event.get('price_min'),
            event.get('price_max'),
            venue_id,
            json.dumps(artist_ids) if artist_ids else '[]',
            event.get('popularity_score', 0.0),
            h3_index,
            event.get('source'),
            event.get('source_url'),
            event.get('image_url'),
            'active',
            datetime.now(),
            datetime.now()
        )
    
    def _store_venue(self, event: Dict) -> str:
        """Store venue and return venue ID."""
        
        venue_name = event.get('venue_name', '')
        if not venue_name:
            return ''
        
        venue_id = f"venue_{hash(venue_name.lower()) % 100000}"
        
        # This would normally check if venue exists first
        # For now, just return the generated ID
        return venue_id
    
    def _store_artists(self, event: Dict) -> List[str]:
        """Store artists and return artist IDs."""
        
        artists = event.get('artists', [])
        artist_ids = []
        
        for artist_name in artists:
            if artist_name:
                artist_id = f"artist_{hash(artist_name.lower()) % 100000}"
                artist_ids.append(artist_id)
        
        return artist_ids
    
    def _generate_report(self):
        """Generate pipeline execution report."""
        
        # Calculate additional statistics
        if self.stats.validation_passed > 0:
            self.stats.avg_confidence_score = 0.8  # Placeholder
        
        report = {
            'pipeline_execution': {
                'start_time': self.stats.start_time.isoformat() if self.stats.start_time else None,
                'end_time': self.stats.end_time.isoformat() if self.stats.end_time else None,
                'duration_seconds': self.stats.duration_seconds,
                'configuration': asdict(self.config)
            },
            'data_collection': {
                'eventbrite_events': self.stats.eventbrite_events,
                'meetup_events': self.stats.meetup_events,
                'instagram_events': self.stats.instagram_events,
                'tiktok_events': self.stats.tiktok_events,
                'viral_events': self.stats.viral_events,
                'total_raw_events': self.stats.total_raw_events
            },
            'data_processing': {
                'validation_passed': self.stats.validation_passed,
                'validation_failed': self.stats.validation_failed,
                'enrichment_successful': self.stats.enrichment_successful,
                'enrichment_coverage': self.stats.enrichment_coverage,
                'duplicates_removed': self.stats.duplicates_removed,
                'final_unique_events': self.stats.final_unique_events
            },
            'quality_metrics': {
                'avg_confidence_score': self.stats.avg_confidence_score,
                'duplicate_detection_accuracy': self.stats.duplicate_detection_accuracy,
                'data_retention_rate': self.stats.final_unique_events / self.stats.total_raw_events if self.stats.total_raw_events > 0 else 0
            }
        }
        
        # Save report
        report_path = Path(self.config.output_dir) / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Pipeline report saved to {report_path}")
    
    def _save_raw_data(self, events: List[Dict]):
        """Save raw collected data."""
        
        output_path = Path(self.config.output_dir) / f"raw_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2, default=str)
        
        logger.info(f"Raw data saved to {output_path}")
    
    def _save_processed_data(self, events: List[Dict]):
        """Save processed event data."""
        
        output_path = Path(self.config.output_dir) / f"processed_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2, default=str)
        
        logger.info(f"Processed data saved to {output_path}")

def main():
    """Main function to run the integrated data pipeline."""
    
    # Load configuration from environment variables
    config = PipelineConfig(
        # Official API credentials
        eventbrite_token=os.getenv('EVENTBRITE_API_TOKEN'),
        spotify_client_id=os.getenv('SPOTIFY_CLIENT_ID'),
        spotify_client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
        lastfm_api_key=os.getenv('LASTFM_API_KEY'),
        
        # Social media credentials (optional - works without login)
        instagram_username=os.getenv('INSTAGRAM_USERNAME'),
        instagram_password=os.getenv('INSTAGRAM_PASSWORD'),
        
        # Pipeline settings
        days_ahead=60,
        max_events_per_source=500,
        enable_viral_discovery=True,
        enable_instagram_scraping=True,
        enable_tiktok_scraping=True,
        enable_enrichment=True,
        enable_deduplication=True,
        social_days_back=14,
        min_confidence_score=0.3
    )
    
    # Create and run pipeline
    pipeline = IntegratedDataPipeline(config)
    
    try:
        stats = pipeline.run_pipeline()
        
        print("\n" + "="*60)
        print("DATA COLLECTION PIPELINE COMPLETED")
        print("="*60)
        print(f"Duration: {stats.duration_seconds:.1f} seconds")
        print(f"Raw events collected: {stats.total_raw_events}")
        print(f"  - Eventbrite: {stats.eventbrite_events}")
        print(f"  - Meetup: {stats.meetup_events}")
        print(f"  - Instagram: {stats.instagram_events}")
        print(f"  - TikTok: {stats.tiktok_events}")
        print(f"  - Viral Discovery: {stats.viral_events}")
        print(f"Final unique events: {stats.final_unique_events}")
        print(f"Data retention rate: {(stats.final_unique_events/stats.total_raw_events*100) if stats.total_raw_events > 0 else 0:.1f}%")
        print(f"Enrichment coverage: {stats.enrichment_coverage*100:.1f}%")
        print(f"Duplicates removed: {stats.duplicates_removed}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())