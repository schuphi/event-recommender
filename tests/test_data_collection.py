#!/usr/bin/env python3
"""
Data collection pipeline tests for Copenhagen Event Recommender.
Tests scrapers, validation, enrichment, and deduplication components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import os
from typing import List, Dict


class TestEventbriteScraperBasic:
    """Test basic Eventbrite scraper functionality."""
    
    def test_scraper_initialization(self):
        """Test Eventbrite scraper can be initialized."""
        from data-collection.scrapers.eventbrite_scraper import EventbriteEventScraper
        
        scraper = EventbriteEventScraper()
        assert scraper is not None
        assert hasattr(scraper, 'scrape_events')
    
    @patch('requests.get')
    def test_eventbrite_api_call(self, mock_get):
        """Test Eventbrite API call handling."""
        from data-collection.scrapers.eventbrite_scraper import EventbriteEventScraper
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'events': [
                {
                    'id': 'test_event_1',
                    'name': {'text': 'Test Event'},
                    'description': {'text': 'Test Description'},
                    'start': {'local': '2024-12-01T20:00:00'},
                    'venue': {
                        'name': 'Test Venue',
                        'address': {'localized_area_display': 'Copenhagen'},
                        'latitude': '55.6761',
                        'longitude': '12.5683'
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        scraper = EventbriteEventScraper(api_token='test_token')
        events = scraper.scrape_events(location='Copenhagen', max_events=10)
        
        assert isinstance(events, list)
        assert len(events) > 0
        assert 'id' in events[0]
        assert 'title' in events[0]
    
    @patch('requests.get')
    def test_eventbrite_error_handling(self, mock_get):
        """Test Eventbrite API error handling."""
        from data-collection.scrapers.eventbrite_scraper import EventbriteEventScraper
        
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limit
        mock_response.raise_for_status.side_effect = Exception("Rate limited")
        mock_get.return_value = mock_response
        
        scraper = EventbriteEventScraper(api_token='test_token')
        events = scraper.scrape_events(location='Copenhagen')
        
        # Should handle error gracefully
        assert isinstance(events, list)
        assert len(events) == 0


class TestMeetupScraper:
    """Test Meetup scraper functionality."""
    
    def test_meetup_scraper_initialization(self):
        """Test Meetup scraper initialization."""
        from data-collection.scrapers.meetup_scraper import MeetupEventScraper
        
        scraper = MeetupEventScraper()
        assert scraper is not None
        assert hasattr(scraper, 'scrape_events')
    
    @patch('requests.get')
    def test_meetup_events_parsing(self, mock_get):
        """Test Meetup events parsing."""
        from data-collection.scrapers.meetup_scraper import MeetupEventScraper
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'id': 'meetup_event_1',
                    'name': 'Copenhagen Tech Meetup',
                    'description': 'Technology meetup in Copenhagen',
                    'time': int((datetime.now() + timedelta(days=7)).timestamp() * 1000),
                    'venue': {
                        'name': 'Tech Hub',
                        'address_1': 'Test Street 1',
                        'city': 'Copenhagen',
                        'lat': 55.6761,
                        'lon': 12.5683
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        scraper = MeetupEventScraper()
        events = scraper.scrape_events(location='Copenhagen')
        
        assert isinstance(events, list)
        assert len(events) > 0
        
        event = events[0]
        assert 'id' in event
        assert 'title' in event
        assert 'venue_lat' in event
        assert 'venue_lon' in event


class TestInstagramScraper:
    """Test Instagram scraper functionality."""
    
    def test_instagram_scraper_initialization(self):
        """Test Instagram scraper can be initialized."""
        from data-collection.scrapers.social_scrapers.instagram import InstagramEventScraper
        
        scraper = InstagramEventScraper()
        assert scraper is not None
        assert hasattr(scraper, 'scrape_venues')
    
    def test_event_keyword_detection(self):
        """Test Instagram event keyword detection."""
        from data-collection.scrapers.social_scrapers.instagram import InstagramEventScraper
        
        scraper = InstagramEventScraper()
        
        # Mock post with event keywords
        mock_post = Mock()
        mock_post.caption = "Tonight at Culture Box! Techno night with amazing lineup"
        mock_post.caption_hashtags = ['techno', 'cultureboxcph']
        
        is_event = scraper._is_event_post(mock_post)
        assert is_event is True
        
        # Mock post without event keywords
        mock_post.caption = "Beautiful sunset in Copenhagen"
        mock_post.caption_hashtags = ['copenhagen', 'sunset']
        
        is_event = scraper._is_event_post(mock_post)
        assert is_event is False
    
    def test_artist_extraction(self):
        """Test artist name extraction from Instagram posts."""
        from data-collection.scrapers.social_scrapers.instagram import InstagramEventScraper
        
        scraper = InstagramEventScraper()
        
        text = "Tonight featuring Kollektiv Turmstrasse and Agnes Obel at Culture Box"
        artists = scraper._extract_artists_from_text(text)
        
        assert isinstance(artists, list)
        assert 'Kollektiv Turmstrasse' in artists
        assert 'Agnes Obel' in artists
        assert 'Culture Box' not in artists  # Should filter venue names
    
    def test_genre_extraction(self):
        """Test genre extraction from Instagram content."""
        from data-collection.scrapers.social_scrapers.instagram import InstagramEventScraper
        
        scraper = InstagramEventScraper()
        
        text = "Amazing techno and house music tonight #electronic #technovibes"
        genres = scraper._extract_genres_from_text(text)
        
        assert isinstance(genres, list)
        assert 'techno' in genres
        assert 'house' in genres
        assert 'electronic' in genres


class TestTikTokScraper:
    """Test TikTok scraper functionality."""
    
    def test_tiktok_scraper_initialization(self):
        """Test TikTok scraper initialization."""
        from data-collection.scrapers.social_scrapers.tiktok import TikTokEventScraper
        
        scraper = TikTokEventScraper()
        assert scraper is not None
        assert hasattr(scraper, 'scrape_venues')
    
    def test_viral_content_detection(self):
        """Test viral event content detection."""
        from data-collection.scrapers.social_scrapers.tiktok import TikTokEventScraper
        
        scraper = TikTokEventScraper()
        
        # Mock viral video
        viral_video = {
            'desc': 'Insane underground party in Copenhagen tonight! You need to go #viral #copenhagenevents',
            'stats': {
                'playCount': 50000,
                'diggCount': 2000,
                'commentCount': 150
            },
            'author': {'uniqueId': 'user123'},  # Not venue account
            'textExtra': [
                {'hashtagName': 'viral'},
                {'hashtagName': 'copenhagenevents'}
            ]
        }
        
        is_viral = scraper._is_viral_event_content(viral_video)
        assert is_viral is True
        
        # Mock promotional video (not viral)
        promo_video = {
            'desc': 'Join us tonight at Vega for a concert',
            'stats': {
                'playCount': 1000,
                'diggCount': 50,
                'commentCount': 10
            },
            'author': {'uniqueId': 'vegacph'},  # Venue account
            'textExtra': []
        }
        
        is_viral = scraper._is_viral_event_content(promo_video)
        assert is_viral is False


class TestDataValidation:
    """Test data validation pipeline."""
    
    def test_validator_initialization(self):
        """Test data validator initialization."""
        from data-collection.validation.data_validator import EventDataValidator
        
        validator = EventDataValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_event')
    
    def test_valid_event_validation(self, validation_test_data):
        """Test validation of valid event data."""
        from data-collection.validation.data_validator import EventDataValidator
        
        validator = EventDataValidator()
        valid_event = validation_test_data['valid_event']
        
        result = validator.validate_event(valid_event)
        
        assert result.status == 'valid'
        assert result.confidence_score > 0.8
        assert len(result.issues) == 0
        assert result.cleaned_data is not None
    
    def test_invalid_event_validation(self, validation_test_data):
        """Test validation of invalid event data."""
        from data-collection.validation.data_validator import EventDataValidator
        
        validator = EventDataValidator()
        invalid_event = validation_test_data['invalid_event_missing_title']
        
        result = validator.validate_event(invalid_event)
        
        assert result.status in ['invalid', 'needs_review']
        assert len(result.issues) > 0
        assert any('title' in issue.lower() for issue in result.issues)
    
    def test_geographic_validation(self, validation_test_data):
        """Test Copenhagen geographic validation."""
        from data-collection.validation.data_validator import EventDataValidator
        
        validator = EventDataValidator()
        
        # Valid Copenhagen coordinates
        valid_event = validation_test_data['valid_event'].copy()
        result = validator.validate_event(valid_event)
        assert result.status == 'valid'
        
        # Invalid coordinates
        invalid_event = validation_test_data['invalid_event_bad_coordinates']
        result = validator.validate_event(invalid_event)
        assert result.status in ['invalid', 'needs_review']
        assert any('coordinate' in issue.lower() or 'location' in issue.lower() 
                  for issue in result.issues)
    
    def test_temporal_validation(self, validation_test_data):
        """Test temporal validation of events."""
        from data-collection.validation.data_validator import EventDataValidator
        
        validator = EventDataValidator()
        
        # Past event (should be flagged)
        past_event = validation_test_data['valid_event'].copy()
        past_event['start_time'] = datetime.now() - timedelta(days=1)
        
        result = validator.validate_event(past_event)
        # May be valid but with lower confidence
        assert result.confidence_score < 0.9


class TestArtistGenreEnrichment:
    """Test artist and genre enrichment."""
    
    def test_enricher_initialization(self):
        """Test artist genre enricher initialization."""
        from data-collection.enrichment.artist_genre_enricher import ArtistGenreEnricher
        
        enricher = ArtistGenreEnricher()
        assert enricher is not None
        assert hasattr(enricher, 'enrich_event_artists')
    
    @patch('requests.post')
    @patch('requests.get')
    def test_spotify_enrichment(self, mock_get, mock_post):
        """Test Spotify API enrichment."""
        from data-collection.enrichment.artist_genre_enricher import ArtistGenreEnricher
        
        # Mock Spotify auth
        mock_post.return_value.json.return_value = {
            'access_token': 'test_token',
            'token_type': 'Bearer',
            'expires_in': 3600
        }
        
        # Mock Spotify search
        mock_get.return_value.json.return_value = {
            'artists': {
                'items': [
                    {
                        'id': 'spotify_artist_id',
                        'name': 'Test Artist',
                        'genres': ['electronic', 'techno'],
                        'popularity': 80,
                        'images': [{'url': 'https://example.com/image.jpg'}]
                    }
                ]
            }
        }
        mock_get.return_value.status_code = 200
        
        enricher = ArtistGenreEnricher(
            spotify_client_id='test_id',
            spotify_client_secret='test_secret'
        )
        
        event_data = {'artists': ['Test Artist']}
        enriched = enricher.enrich_event_artists(event_data)
        
        assert 'enriched_artists' in enriched
        assert len(enriched['enriched_artists']) > 0
        assert 'genres' in enriched['enriched_artists'][0]
    
    def test_genre_normalization(self):
        """Test genre normalization and mapping."""
        from data-collection.enrichment.artist_genre_enricher import ArtistGenreEnricher
        
        enricher = ArtistGenreEnricher()
        
        # Test various genre variations
        raw_genres = ['EDM', 'Progressive House', 'Minimal Techno', 'Deep House']
        normalized = enricher._normalize_genres(raw_genres)
        
        assert isinstance(normalized, list)
        assert 'electronic' in normalized  # EDM should map to electronic
        assert 'house' in normalized  # Both house variants should map to house
        assert 'techno' in normalized  # Minimal Techno should map to techno
    
    def test_enrichment_caching(self):
        """Test artist enrichment caching."""
        from data-collection.enrichment.artist_genre_enricher import ArtistGenreEnricher
        
        enricher = ArtistGenreEnricher()
        
        # Mock enrichment data
        artist_data = {
            'name': 'Test Artist',
            'genres': ['electronic', 'techno'],
            'popularity_score': 0.8
        }
        
        # Cache the data
        enricher._cache_artist_data('Test Artist', artist_data)
        
        # Retrieve from cache
        cached_data = enricher._get_cached_artist_data('Test Artist')
        
        assert cached_data is not None
        assert cached_data['name'] == 'Test Artist'
        assert 'genres' in cached_data


class TestDuplicateDetection:
    """Test duplicate detection system."""
    
    def test_detector_initialization(self):
        """Test duplicate detector initialization."""
        from data-collection.deduplication.duplicate_detector import DuplicateDetector
        
        detector = DuplicateDetector()
        assert detector is not None
        assert hasattr(detector, 'detect_duplicates')
    
    def test_exact_duplicate_detection(self):
        """Test detection of exact duplicates."""
        from data-collection.deduplication.duplicate_detector import DuplicateDetector
        
        detector = DuplicateDetector()
        
        events = [
            {
                'id': 'event_1',
                'title': 'Techno Night at Culture Box',
                'venue_name': 'Culture Box',
                'start_time': datetime.now() + timedelta(days=1)
            },
            {
                'id': 'event_2',
                'title': 'Techno Night at Culture Box',  # Exact match
                'venue_name': 'Culture Box',
                'start_time': datetime.now() + timedelta(days=1)
            }
        ]
        
        duplicates = detector.detect_duplicates(events)
        
        assert len(duplicates) > 0
        duplicate_group = duplicates[0]
        assert len(duplicate_group) == 2
        assert duplicate_group[0]['similarity_score'] > 0.9
    
    def test_fuzzy_duplicate_detection(self):
        """Test detection of fuzzy duplicates."""
        from data-collection.deduplication.duplicate_detector import DuplicateDetector
        
        detector = DuplicateDetector(similarity_threshold=0.7)
        
        events = [
            {
                'id': 'event_1',
                'title': 'Electronic Music Night',
                'venue_name': 'Culture Box',
                'start_time': datetime.now() + timedelta(days=1)
            },
            {
                'id': 'event_2', 
                'title': 'Electronic Night',  # Similar title
                'venue_name': 'Culturebox',  # Similar venue
                'start_time': datetime.now() + timedelta(days=1, hours=1)  # Similar time
            }
        ]
        
        duplicates = detector.detect_duplicates(events)
        
        assert len(duplicates) > 0
        duplicate_group = duplicates[0]
        assert len(duplicate_group) == 2
        assert 0.7 <= duplicate_group[0]['similarity_score'] < 0.9
    
    def test_venue_name_normalization(self):
        """Test venue name normalization for duplicate detection."""
        from data-collection.deduplication.duplicate_detector import DuplicateDetector
        
        detector = DuplicateDetector()
        
        # Test various venue name variations
        variations = [
            'Culture Box',
            'Culturebox',
            'culture box',
            'Culture Box Copenhagen'
        ]
        
        normalized_names = [detector._normalize_venue_name(name) for name in variations]
        
        # All should normalize to similar form
        assert len(set(normalized_names)) <= 2  # Should be very similar


class TestIntegratedPipeline:
    """Test integrated data collection pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from data-collection.pipeline.integrated_data_pipeline import IntegratedDataPipeline
        
        pipeline = IntegratedDataPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'run_collection')
    
    @patch('data-collection.scrapers.eventbrite_scraper.EventbriteEventScraper.scrape_events')
    @patch('data-collection.scrapers.meetup_scraper.MeetupEventScraper.scrape_events')  
    def test_multi_source_collection(self, mock_meetup, mock_eventbrite):
        """Test collection from multiple sources."""
        from data-collection.pipeline.integrated_data_pipeline import IntegratedDataPipeline
        
        # Mock scraper responses
        mock_eventbrite.return_value = [
            {'id': 'eb_1', 'title': 'Eventbrite Event', 'source': 'eventbrite'}
        ]
        mock_meetup.return_value = [
            {'id': 'mu_1', 'title': 'Meetup Event', 'source': 'meetup'}
        ]
        
        pipeline = IntegratedDataPipeline()
        
        # Mock configuration
        config = Mock()
        config.enable_eventbrite = True
        config.enable_meetup = True
        config.enable_instagram_scraping = False
        config.enable_tiktok_scraping = False
        
        events = pipeline.run_collection(config)
        
        assert isinstance(events, list)
        assert len(events) >= 2
        
        # Should have events from both sources
        sources = [event['source'] for event in events]
        assert 'eventbrite' in sources
        assert 'meetup' in sources
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling when scrapers fail."""
        from data-collection.pipeline.integrated_data_pipeline import IntegratedDataPipeline
        
        pipeline = IntegratedDataPipeline()
        
        # Mock failing scraper
        with patch.object(pipeline, '_collect_eventbrite_events') as mock_eventbrite:
            mock_eventbrite.side_effect = Exception("API Error")
            
            config = Mock()
            config.enable_eventbrite = True
            config.enable_meetup = False
            config.enable_instagram_scraping = False
            config.enable_tiktok_scraping = False
            
            # Should handle error gracefully
            events = pipeline.run_collection(config)
            
            # Should return empty list, not crash
            assert isinstance(events, list)
    
    def test_data_quality_metrics(self):
        """Test data quality metrics calculation."""
        from data-collection.pipeline.integrated_data_pipeline import IntegratedDataPipeline
        
        pipeline = IntegratedDataPipeline()
        
        events = [
            {'id': '1', 'title': 'Event 1', 'validation_status': 'valid'},
            {'id': '2', 'title': 'Event 2', 'validation_status': 'valid'}, 
            {'id': '3', 'title': 'Event 3', 'validation_status': 'invalid'},
            {'id': '4', 'title': 'Event 4', 'validation_status': 'needs_review'}
        ]
        
        metrics = pipeline._calculate_quality_metrics(events)
        
        assert 'total_events' in metrics
        assert 'valid_events' in metrics
        assert 'validation_rate' in metrics
        assert 'quality_score' in metrics
        
        assert metrics['total_events'] == 4
        assert metrics['valid_events'] == 2
        assert metrics['validation_rate'] == 0.5


class TestDataPersistence:
    """Test data persistence and storage."""
    
    def test_event_storage(self, sample_events, db_service):
        """Test storing events in database."""
        
        # Store events
        for event in sample_events:
            result = db_service.store_event(event)
            assert result is not None
        
        # Retrieve events
        stored_events = db_service.get_all_events()
        assert len(stored_events) >= len(sample_events)
    
    def test_data_export_import(self, sample_events):
        """Test data export and import functionality."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
            json.dump(sample_events, f)
        
        try:
            # Import data
            with open(export_path, 'r') as f:
                imported_events = json.load(f)
            
            assert len(imported_events) == len(sample_events)
            assert imported_events[0]['id'] == sample_events[0]['id']
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestPerformanceAndScaling:
    """Test data collection performance and scaling."""
    
    def test_large_dataset_processing(self, performance_test_events):
        """Test processing of large event datasets."""
        from data-collection.validation.data_validator import EventDataValidator
        
        validator = EventDataValidator()
        
        import time
        start_time = time.time()
        
        valid_count = 0
        for event in performance_test_events[:100]:  # Test with 100 events
            result = validator.validate_event(event)
            if result.status == 'valid':
                valid_count += 1
        
        end_time = time.time()
        
        # Should process reasonably fast
        assert end_time - start_time < 10.0  # 10 seconds max for 100 events
        assert valid_count > 0
    
    def test_concurrent_scraping_simulation(self):
        """Test concurrent scraping performance."""
        from concurrent.futures import ThreadPoolExecutor
        from data-collection.scrapers.eventbrite_scraper import EventbriteEventScraper
        
        def mock_scrape():
            scraper = EventbriteEventScraper()
            # Simulate API call delay
            import time
            time.sleep(0.1)
            return [{'id': 'test', 'title': 'Test Event'}]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(mock_scrape) for _ in range(6)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        
        # Concurrent execution should be faster than sequential
        assert end_time - start_time < 1.0  # Should complete in under 1 second
        assert len(results) == 6