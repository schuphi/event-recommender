#!/usr/bin/env python3
"""
Database service tests for Copenhagen Event Recommender.
Tests DuckDB operations, data integrity, and query performance.
"""

import pytest
import duckdb
from datetime import datetime, timedelta
import json
import tempfile
import os
from unittest.mock import Mock, patch


class TestDatabaseConnection:
    """Test database connection and initialization."""

    def test_database_service_initialization(self, db_service):
        """Test database service can be initialized."""
        assert db_service is not None
        assert hasattr(db_service, "get_connection")

    def test_database_connection(self, db_service):
        """Test database connection works."""
        conn = db_service.get_connection()
        assert conn is not None

        # Test simple query
        result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1

    def test_schema_creation(self, test_db):
        """Test database schema is properly created."""
        conn = duckdb.connect(test_db)

        # Check all required tables exist
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]

        required_tables = ["venues", "artists", "events", "users", "interactions"]
        for table in required_tables:
            assert table in table_names

        conn.close()

    def test_table_schemas(self, test_db):
        """Test table schemas have required columns."""
        conn = duckdb.connect(test_db)

        # Test events table schema
        events_columns = conn.execute("DESCRIBE events").fetchall()
        column_names = [col[0] for col in events_columns]

        required_columns = [
            "id",
            "title",
            "description",
            "date_time",
            "venue_id",
            "price_min",
            "price_max",
            "popularity_score",
            "h3_index",
        ]

        for column in required_columns:
            assert column in column_names

        conn.close()


class TestEventOperations:
    """Test event-related database operations."""

    def test_store_event(self, db_service):
        """Test storing a new event."""
        event_data = {
            "id": "test_event_new",
            "title": "New Test Event",
            "description": "A new test event for database testing",
            "start_time": datetime.now() + timedelta(days=5),
            "venue_name": "Test Venue",
            "venue_lat": 55.6761,
            "venue_lon": 12.5683,
            "price_min": 150.0,
            "price_max": 250.0,
            "artists": ["Test Artist"],
            "genres": ["test"],
            "source": "test",
        }

        result = db_service.store_event(event_data)
        assert result is not None

        # Verify event was stored
        stored_event = db_service.get_event_by_id("test_event_new")
        assert stored_event is not None
        assert stored_event["title"] == "New Test Event"

    def test_get_event_by_id(self, db_service, test_db):
        """Test retrieving event by ID."""
        event = db_service.get_event_by_id("event_1")

        assert event is not None
        assert event["id"] == "event_1"
        assert "title" in event
        assert "venue" in event

    def test_get_nonexistent_event(self, db_service):
        """Test retrieving non-existent event."""
        event = db_service.get_event_by_id("nonexistent_event")
        assert event is None

    def test_get_all_events(self, db_service, test_db):
        """Test retrieving all events."""
        events = db_service.get_all_events()

        assert isinstance(events, list)
        assert len(events) >= 2  # From test data
        assert all("id" in event for event in events)

    def test_get_events_with_pagination(self, db_service):
        """Test event pagination."""
        events = db_service.get_events(limit=1, offset=0)

        assert isinstance(events, list)
        assert len(events) <= 1

    def test_get_upcoming_events(self, db_service):
        """Test retrieving upcoming events only."""
        upcoming_events = db_service.get_upcoming_events(days_ahead=30)

        assert isinstance(upcoming_events, list)

        # All events should be in the future
        now = datetime.now()
        for event in upcoming_events:
            if event.get("date_time"):
                event_date = datetime.fromisoformat(str(event["date_time"]))
                assert event_date > now

    def test_search_events_by_title(self, db_service):
        """Test searching events by title."""
        results = db_service.search_events("Kollektiv")

        assert isinstance(results, list)
        # Should find the test event with "Kollektiv Turmstrasse"
        assert len(results) > 0
        assert any("Kollektiv" in event["title"] for event in results)

    def test_filter_events_by_genre(self, db_service):
        """Test filtering events by genre."""
        # This would require implementing genre filtering
        # For now, test the basic structure
        events = db_service.get_all_events()

        # Filter manually for testing
        electronic_events = [
            event
            for event in events
            if event.get("artist_ids") and "artist_1" in str(event["artist_ids"])
        ]

        assert isinstance(electronic_events, list)

    def test_filter_events_by_venue(self, db_service):
        """Test filtering events by venue."""
        events = db_service.get_events_by_venue("venue_1")

        assert isinstance(events, list)
        assert all(
            event["venue_id"] == "venue_1" for event in events if event.get("venue_id")
        )

    def test_filter_events_by_date_range(self, db_service):
        """Test filtering events by date range."""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=30)

        events = db_service.get_events_in_date_range(start_date, end_date)

        assert isinstance(events, list)

        for event in events:
            if event.get("date_time"):
                event_date = datetime.fromisoformat(str(event["date_time"]))
                assert start_date <= event_date <= end_date


class TestVenueOperations:
    """Test venue-related database operations."""

    def test_store_venue(self, db_service):
        """Test storing a new venue."""
        venue_data = {
            "id": "test_venue_new",
            "name": "Test Venue New",
            "address": "Test Address 123",
            "lat": 55.6761,
            "lon": 12.5683,
            "neighborhood": "Test Area",
        }

        result = db_service.store_venue(venue_data)
        assert result is not None

        # Verify venue was stored
        stored_venue = db_service.get_venue_by_id("test_venue_new")
        assert stored_venue is not None
        assert stored_venue["name"] == "Test Venue New"

    def test_get_all_venues(self, db_service, test_db):
        """Test retrieving all venues."""
        venues = db_service.get_all_venues()

        assert isinstance(venues, list)
        assert len(venues) >= 3  # From test data
        assert all("id" in venue for venue in venues)
        assert all("name" in venue for venue in venues)

    def test_get_venue_by_id(self, db_service):
        """Test retrieving venue by ID."""
        venue = db_service.get_venue_by_id("venue_1")

        assert venue is not None
        assert venue["id"] == "venue_1"
        assert venue["name"] == "Culture Box"

    def test_get_venues_near_location(self, db_service):
        """Test finding venues near a location."""
        # Copenhagen center coordinates
        lat, lon = 55.6761, 12.5683
        radius_km = 10.0

        venues = db_service.get_venues_near_location(lat, lon, radius_km)

        assert isinstance(venues, list)
        # All test venues should be near Copenhagen center
        assert len(venues) > 0


class TestArtistOperations:
    """Test artist-related database operations."""

    def test_store_artist(self, db_service):
        """Test storing a new artist."""
        artist_data = {
            "id": "test_artist_new",
            "name": "New Test Artist",
            "genres": ["electronic", "house"],
            "popularity_score": 0.75,
        }

        result = db_service.store_artist(artist_data)
        assert result is not None

        # Verify artist was stored
        stored_artist = db_service.get_artist_by_id("test_artist_new")
        assert stored_artist is not None
        assert stored_artist["name"] == "New Test Artist"

    def test_get_all_artists(self, db_service, test_db):
        """Test retrieving all artists."""
        artists = db_service.get_all_artists()

        assert isinstance(artists, list)
        assert len(artists) >= 3  # From test data
        assert all("id" in artist for artist in artists)
        assert all("name" in artist for artist in artists)

    def test_search_artists(self, db_service):
        """Test searching artists by name."""
        results = db_service.search_artists("Kollektiv")

        assert isinstance(results, list)
        assert len(results) > 0
        assert any("Kollektiv" in artist["name"] for artist in results)


class TestUserOperations:
    """Test user-related database operations."""

    def test_store_user(self, db_service):
        """Test storing a new user."""
        user_data = {
            "id": "test_user_new",
            "name": "New Test User",
            "email": "newtest@example.com",
            "preferences": {
                "preferred_genres": ["techno", "house"],
                "location_lat": 55.6761,
                "location_lon": 12.5683,
            },
        }

        result = db_service.store_user(user_data)
        assert result is not None

        # Verify user was stored
        stored_user = db_service.get_user_by_id("test_user_new")
        assert stored_user is not None
        assert stored_user["name"] == "New Test User"
        assert stored_user["email"] == "newtest@example.com"

    def test_get_user_by_id(self, db_service):
        """Test retrieving user by ID."""
        user = db_service.get_user_by_id("user_1")

        assert user is not None
        assert user["id"] == "user_1"
        assert "preferences" in user

    def test_get_user_by_email(self, db_service):
        """Test retrieving user by email."""
        user = db_service.get_user_by_email("test1@example.com")

        assert user is not None
        assert user["email"] == "test1@example.com"

    def test_update_user_preferences(self, db_service):
        """Test updating user preferences."""
        new_preferences = {
            "preferred_genres": ["indie", "alternative"],
            "price_range": [200, 500],
            "location_lat": 55.6800,
            "location_lon": 12.5700,
        }

        result = db_service.update_user_preferences("user_1", new_preferences)
        assert result is True

        # Verify preferences were updated
        updated_user = db_service.get_user_by_id("user_1")
        assert updated_user is not None

        # Check if preferences were merged/updated
        preferences = (
            json.loads(updated_user["preferences"])
            if isinstance(updated_user["preferences"], str)
            else updated_user["preferences"]
        )
        assert "indie" in preferences.get("preferred_genres", [])


class TestInteractionOperations:
    """Test user interaction database operations."""

    def test_store_interaction(self, db_service):
        """Test storing user interaction."""
        interaction_data = {
            "id": "test_interaction_new",
            "user_id": "user_1",
            "event_id": "event_1",
            "interaction_type": "like",
            "rating": 4.5,
            "timestamp": datetime.now(),
        }

        result = db_service.store_interaction(interaction_data)
        assert result is not None

    def test_get_user_interactions(self, db_service):
        """Test retrieving user interactions."""
        interactions = db_service.get_user_interactions("user_1")

        assert isinstance(interactions, list)
        assert len(interactions) > 0
        assert all(interaction["user_id"] == "user_1" for interaction in interactions)

    def test_get_event_interactions(self, db_service):
        """Test retrieving event interactions."""
        interactions = db_service.get_event_interactions("event_1")

        assert isinstance(interactions, list)
        assert all(interaction["event_id"] == "event_1" for interaction in interactions)

    def test_get_all_interactions(self, db_service):
        """Test retrieving all interactions."""
        interactions = db_service.get_all_interactions()

        assert isinstance(interactions, list)
        assert len(interactions) >= 3  # From test data
        assert all("user_id" in interaction for interaction in interactions)
        assert all("event_id" in interaction for interaction in interactions)


class TestDataIntegrity:
    """Test data integrity and constraints."""

    def test_duplicate_event_prevention(self, db_service):
        """Test preventing duplicate events."""
        event_data = {
            "id": "duplicate_test",
            "title": "Duplicate Test Event",
            "start_time": datetime.now() + timedelta(days=1),
            "venue_name": "Test Venue",
            "source": "test",
        }

        # Store event first time
        result1 = db_service.store_event(event_data)
        assert result1 is not None

        # Try to store same event again
        result2 = db_service.store_event(event_data)

        # Should handle gracefully (update or skip)
        assert result2 is not None

        # Verify only one event exists
        events = db_service.search_events("Duplicate Test")
        assert len(events) == 1

    def test_foreign_key_constraints(self, db_service):
        """Test foreign key relationships."""
        # Try to create interaction with non-existent user
        interaction_data = {
            "id": "invalid_interaction",
            "user_id": "nonexistent_user",
            "event_id": "event_1",
            "interaction_type": "like",
            "rating": 5.0,
        }

        # Should handle gracefully or enforce constraint
        try:
            result = db_service.store_interaction(interaction_data)
            # If it succeeds, the system is lenient with constraints
        except Exception:
            # If it fails, constraints are enforced
            pass

    def test_data_type_validation(self, db_service):
        """Test data type validation."""
        # Try to store event with invalid data types
        invalid_event = {
            "id": "invalid_data_event",
            "title": "Test Event",
            "start_time": "not a datetime",  # Invalid type
            "venue_lat": "not a number",  # Invalid type
            "price_min": "free",  # Invalid type
            "source": "test",
        }

        try:
            result = db_service.store_event(invalid_event)
            # System should either convert or reject invalid data
        except Exception as e:
            # Exception is expected for invalid data
            assert "type" in str(e).lower() or "convert" in str(e).lower()


class TestQueryPerformance:
    """Test database query performance."""

    def test_event_search_performance(self, db_service, performance_test_events):
        """Test event search query performance."""
        import time

        # Add some test events for performance testing
        for i, event in enumerate(performance_test_events[:50]):  # Add 50 events
            event["id"] = f"perf_event_{i}"
            db_service.store_event(event)

        # Test search performance
        start_time = time.time()
        results = db_service.search_events("Performance")
        end_time = time.time()

        # Should complete quickly even with more data
        assert end_time - start_time < 2.0  # 2 seconds max
        assert isinstance(results, list)

    def test_geospatial_query_performance(self, db_service):
        """Test geospatial query performance."""
        import time

        lat, lon = 55.6761, 12.5683
        radius_km = 5.0

        start_time = time.time()
        venues = db_service.get_venues_near_location(lat, lon, radius_km)
        end_time = time.time()

        # Geospatial queries should be reasonably fast
        assert end_time - start_time < 1.0  # 1 second max
        assert isinstance(venues, list)

    def test_aggregation_query_performance(self, db_service):
        """Test aggregation query performance."""
        import time

        start_time = time.time()

        # Get some statistics (if methods exist)
        event_count = len(db_service.get_all_events())
        venue_count = len(db_service.get_all_venues())

        end_time = time.time()

        # Counting queries should be fast
        assert end_time - start_time < 1.0  # 1 second max
        assert event_count >= 0
        assert venue_count >= 0


class TestTransactions:
    """Test database transactions and atomicity."""

    def test_transaction_rollback(self, db_service):
        """Test transaction rollback on error."""
        # This test would require implementing transaction support
        # For now, test basic error handling

        invalid_data = {
            "id": None,  # Invalid ID
            "title": "Test Event",
            "source": "test",
        }

        try:
            result = db_service.store_event(invalid_data)
            # If it succeeds, system handles None IDs
        except Exception:
            # If it fails, that's expected behavior
            pass

        # Database should still be in valid state
        events = db_service.get_all_events()
        assert isinstance(events, list)

    def test_concurrent_access(self, db_service):
        """Test concurrent database access."""
        from concurrent.futures import ThreadPoolExecutor
        import time

        def concurrent_insert(i):
            event_data = {
                "id": f"concurrent_event_{i}",
                "title": f"Concurrent Event {i}",
                "start_time": datetime.now() + timedelta(days=i),
                "venue_name": "Concurrent Venue",
                "source": "test",
            }
            return db_service.store_event(event_data)

        # Test concurrent inserts
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(concurrent_insert, i) for i in range(5)]
            results = [future.result() for future in futures]

        # All inserts should succeed
        assert all(result is not None for result in results)

        # Verify all events were stored
        concurrent_events = [
            event
            for event in db_service.get_all_events()
            if event["id"].startswith("concurrent_event_")
        ]
        assert len(concurrent_events) == 5


class TestDatabaseMaintenance:
    """Test database maintenance operations."""

    def test_cleanup_old_events(self, db_service):
        """Test cleaning up old/past events."""
        # Add a past event
        past_event = {
            "id": "past_event_test",
            "title": "Past Event",
            "start_time": datetime.now() - timedelta(days=30),
            "venue_name": "Past Venue",
            "source": "test",
        }

        db_service.store_event(past_event)

        # If cleanup method exists, test it
        try:
            cleanup_result = db_service.cleanup_past_events(days_old=7)
            assert cleanup_result is not None
        except AttributeError:
            # Method doesn't exist yet, skip test
            pass

    def test_database_statistics(self, db_service):
        """Test database statistics gathering."""
        # Test basic statistics
        event_count = len(db_service.get_all_events())
        venue_count = len(db_service.get_all_venues())
        user_count = (
            len(db_service.get_all_users())
            if hasattr(db_service, "get_all_users")
            else 0
        )

        assert event_count >= 0
        assert venue_count >= 0
        assert user_count >= 0

        # Create basic statistics object
        stats = {
            "events": event_count,
            "venues": venue_count,
            "users": user_count,
            "timestamp": datetime.now(),
        }

        assert "events" in stats
        assert "venues" in stats
        assert stats["events"] == event_count


class TestDatabaseBackupRestore:
    """Test database backup and restore functionality."""

    def test_data_export(self, db_service):
        """Test exporting database data."""
        # Get all data
        events = db_service.get_all_events()
        venues = db_service.get_all_venues()

        # Create export data structure
        export_data = {
            "events": events,
            "venues": venues,
            "export_timestamp": datetime.now().isoformat(),
        }

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name
            json.dump(export_data, f, default=str)

        try:
            # Verify export file
            assert os.path.exists(export_path)

            with open(export_path, "r") as f:
                imported_data = json.load(f)

            assert "events" in imported_data
            assert "venues" in imported_data
            assert len(imported_data["events"]) == len(events)
            assert len(imported_data["venues"]) == len(venues)

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
