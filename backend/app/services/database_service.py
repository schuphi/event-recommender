#!/usr/bin/env python3
"""
Database service for Copenhagen Event Recommender.
Async wrapper around DuckDB with proper error handling and data validation.
"""

import duckdb
import asyncio
import json
import h3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Import with absolute paths to avoid relative import issues
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.models.requests import (
    UserPreferencesRequest,
    RecommendationFilters,
    DateFilter,
    InteractionType,
)
from backend.app.models.responses import (
    EventResponse,
    UserResponse,
    InteractionResponse,
    VenueResponse,
    ArtistResponse,
)
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    db_path: str = "data/events.duckdb"
    connection_timeout: int = 30
    max_retries: int = 3


class DatabaseService:
    """
    Async database service for DuckDB operations.

    Provides:
    - User management with preferences
    - Event querying with filters and geo-indexing
    - Interaction tracking for collaborative filtering
    - Venue and artist data access
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database service."""
        self.config = config or DatabaseConfig()
        if settings.DATABASE_URL and not settings.DATABASE_URL.startswith("../../"):
            self.config.db_path = settings.DATABASE_URL
        elif config and config.db_path:
            # Use provided config path
            pass
        else:
            # Default to absolute path
            from pathlib import Path

            self.config.db_path = str(
                Path(__file__).parent.parent.parent.parent / "data" / "events.duckdb"
            )
        self._connection = None

    async def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection (singleton pattern)."""
        if self._connection is None:
            try:
                # Ensure database file exists
                db_path = Path(self.config.db_path)
                if not db_path.exists():
                    logger.error(f"Database file not found: {db_path}")
                    raise FileNotFoundError(f"Database file not found: {db_path}")

                # Connect to DuckDB
                self._connection = duckdb.connect(str(db_path))
                logger.info(f"Connected to database: {db_path}")

            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise

        return self._connection

    async def _execute_query(
        self, query: str, params: Optional[List] = None
    ) -> List[Tuple]:
        """Execute query with error handling and retries."""
        for attempt in range(self.config.max_retries):
            try:
                conn = await self._get_connection()

                # Run in thread pool for async behavior
                loop = asyncio.get_event_loop()

                if params:
                    result = await loop.run_in_executor(
                        None, lambda: conn.execute(query, params).fetchall()
                    )
                else:
                    result = await loop.run_in_executor(
                        None, lambda: conn.execute(query).fetchall()
                    )

                return result

            except Exception as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Query failed after {self.config.max_retries} attempts: {query[:100]}"
                    )
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

    async def _execute_command(self, query: str, params: Optional[List] = None) -> None:
        """Execute command (INSERT/UPDATE/DELETE) with error handling."""
        for attempt in range(self.config.max_retries):
            try:
                conn = await self._get_connection()
                loop = asyncio.get_event_loop()

                if params:
                    await loop.run_in_executor(
                        None, lambda: conn.execute(query, params)
                    )
                else:
                    await loop.run_in_executor(None, lambda: conn.execute(query))
                return

            except Exception as e:
                logger.warning(f"Command attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Command failed after {self.config.max_retries} attempts: {query[:100]}"
                    )
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            result = await self._execute_query("SELECT 1")
            return len(result) == 1 and result[0][0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    # User Management

    async def create_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        location_lat: Optional[float] = None,
        location_lon: Optional[float] = None,
    ) -> UserResponse:
        """Create a new user with preferences."""

        # Generate H3 index for location
        h3_index = None
        if location_lat is not None and location_lon is not None:
            h3_index = h3.latlng_to_cell(location_lat, location_lon, 8)

        # Insert user
        await self._execute_command(
            """
            INSERT INTO users (id, name, preferences, location_lat, location_lon, h3_index)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                user_id,
                name,
                json.dumps(preferences) if preferences else None,
                location_lat,
                location_lon,
                h3_index,
            ],
        )

        # Return created user
        return await self.get_user(user_id)

    async def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        result = await self._execute_query(
            """
            SELECT u.id, u.name, u.preferences, u.location_lat, u.location_lon, u.created_at,
                   COUNT(i.id) as interaction_count
            FROM users u
            LEFT JOIN interactions i ON u.id = i.user_id
            WHERE u.id = ?
            GROUP BY u.id, u.name, u.preferences, u.location_lat, u.location_lon, u.created_at
        """,
            [user_id],
        )

        if not result:
            return None

        row = result[0]
        return UserResponse(
            user_id=row[0],
            name=row[1],
            preferences=json.loads(row[2]) if row[2] else None,
            location_lat=row[3],
            location_lon=row[4],
            created_at=row[5],
            interaction_count=row[6],
        )

    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        location_lat: Optional[float] = None,
        location_lon: Optional[float] = None,
    ) -> None:
        """Update user preferences."""

        # Generate H3 index for new location
        h3_index = None
        if location_lat is not None and location_lon is not None:
            h3_index = h3.latlng_to_cell(location_lat, location_lon, 8)

        await self._execute_command(
            """
            UPDATE users 
            SET preferences = ?, location_lat = ?, location_lon = ?, h3_index = ?
            WHERE id = ?
        """,
            [json.dumps(preferences), location_lat, location_lon, h3_index, user_id],
        )

    # Event Management

    def _build_event_filters(self, filters: Dict[str, Any]) -> Tuple[str, List]:
        """Build WHERE clause and parameters for event filtering."""

        conditions = ["e.status = 'active'"]
        params = []

        # Date filters
        if filters.get("date_from"):
            conditions.append("e.date_time >= ?")
            params.append(filters["date_from"])

        if filters.get("date_to"):
            conditions.append("e.date_time <= ?")
            params.append(filters["date_to"])

        # Price filters
        if filters.get("min_price") is not None:
            conditions.append("e.price_min >= ?")
            params.append(filters["min_price"])

        if filters.get("max_price") is not None:
            conditions.append("e.price_max <= ?")
            params.append(filters["max_price"])

        # Neighborhood filter
        if filters.get("neighborhood"):
            conditions.append("v.neighborhood = ?")
            params.append(filters["neighborhood"])

        # Genre filter (JSON array contains)
        if filters.get("genres"):
            # For each genre, check if it exists in the artist genres
            genre_conditions = []
            for genre in filters["genres"]:
                genre_conditions.append(
                    "EXISTS (SELECT 1 FROM event_artists ea JOIN artists a ON ea.artist_id = a.id WHERE ea.event_id = e.id AND JSON_EXTRACT(a.genres, '$') LIKE ?)"
                )
                params.append(f'%"{genre}"%')
            if genre_conditions:
                conditions.append(f"({' OR '.join(genre_conditions)})")

        where_clause = " AND ".join(conditions)
        return where_clause, params

    async def get_events(
        self,
        limit: int = 50,
        offset: int = 0,
        neighborhood: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        genres: Optional[List[str]] = None,
    ) -> List[EventResponse]:
        """Get events with optional filtering."""

        # Build filters
        filters = {
            "neighborhood": neighborhood,
            "date_from": date_from,
            "date_to": date_to,
            "min_price": min_price,
            "max_price": max_price,
            "genres": genres,
        }

        where_clause, params = self._build_event_filters(filters)

        # Add pagination params
        params.extend([limit, offset])

        query = f"""
            SELECT 
                e.id, e.title, e.description, e.date_time, e.end_date_time,
                e.price_min, e.price_max, e.currency, e.image_url, e.source, e.source_url,
                e.popularity_score,
                v.name as venue_name, v.address as venue_address, v.lat as venue_lat, 
                v.lon as venue_lon, v.neighborhood,
                GROUP_CONCAT(a.name) as artist_names,
                e.artist_ids
            FROM events e
            JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            WHERE {where_clause}
            GROUP BY e.id, e.title, e.description, e.date_time, e.end_date_time,
                     e.price_min, e.price_max, e.currency, e.image_url, e.source, e.source_url,
                     e.popularity_score, v.name, v.address, v.lat, v.lon, v.neighborhood, e.artist_ids
            ORDER BY e.date_time ASC
            LIMIT ? OFFSET ?
        """

        result = await self._execute_query(query, params)

        events = []
        for row in result:
            # Parse artist names and genres
            artist_names = row[17].split(",") if row[17] else []
            artist_ids = json.loads(row[18]) if row[18] else []

            # Get genres for these artists
            genres = []
            if artist_ids:
                genre_query = """
                    SELECT DISTINCT JSON_EXTRACT(genres, '$[*]') as genre_list
                    FROM artists 
                    WHERE id IN ({})
                """.format(
                    ",".join("?" * len(artist_ids))
                )

                genre_result = await self._execute_query(genre_query, artist_ids)
                for genre_row in genre_result:
                    if genre_row[0]:
                        try:
                            artist_genres = json.loads(genre_row[0])
                            if isinstance(artist_genres, list):
                                genres.extend(artist_genres)
                        except:
                            pass

            event = EventResponse(
                event_id=row[0],
                title=row[1],
                description=row[2],
                date_time=row[3],
                end_time=row[4],
                price_min=row[5],
                price_max=row[6],
                currency=row[7] or "DKK",
                image_url=row[8],
                source=row[9],
                source_url=row[10],
                popularity_score=row[11] or 0.0,
                venue_name=row[12],
                venue_address=row[13],
                venue_lat=row[14],
                venue_lon=row[15],
                neighborhood=row[16],
                artists=artist_names,
                genres=list(set(genres)),  # Remove duplicates
            )
            events.append(event)

        return events

    async def get_event(self, event_id: str) -> Optional[EventResponse]:
        """Get single event by ID."""
        events = await self.get_events(
            limit=1, offset=0
        )  # This will need a filter by ID

        # For now, use a direct query
        result = await self._execute_query(
            """
            SELECT 
                e.id, e.title, e.description, e.date_time, e.end_date_time,
                e.price_min, e.price_max, e.currency, e.image_url, e.source, e.source_url,
                e.popularity_score,
                v.name as venue_name, v.address as venue_address, v.lat as venue_lat, 
                v.lon as venue_lon, v.neighborhood,
                e.artist_ids
            FROM events e
            JOIN venues v ON e.venue_id = v.id
            WHERE e.id = ? AND e.status = 'active'
        """,
            [event_id],
        )

        if not result:
            return None

        row = result[0]

        # Get artist details
        artist_ids = json.loads(row[17]) if row[17] else []
        artists = []
        genres = []

        if artist_ids:
            artist_result = await self._execute_query(
                """
                SELECT name, genres FROM artists WHERE id IN ({})
            """.format(
                    ",".join("?" * len(artist_ids))
                ),
                artist_ids,
            )

            for artist_row in artist_result:
                artists.append(artist_row[0])
                if artist_row[1]:
                    try:
                        artist_genres = json.loads(artist_row[1])
                        if isinstance(artist_genres, list):
                            genres.extend(artist_genres)
                    except:
                        pass

        return EventResponse(
            event_id=row[0],
            title=row[1],
            description=row[2],
            date_time=row[3],
            end_time=row[4],
            price_min=row[5],
            price_max=row[6],
            currency=row[7] or "DKK",
            image_url=row[8],
            source=row[9],
            source_url=row[10],
            popularity_score=row[11] or 0.0,
            venue_name=row[12],
            venue_address=row[13],
            venue_lat=row[14],
            venue_lon=row[15],
            neighborhood=row[16],
            artists=artists,
            genres=list(set(genres)),
        )

    # Interaction Management

    async def record_interaction(
        self,
        user_id: str,
        event_id: str,
        interaction_type: str,
        rating: Optional[float] = None,
        source: Optional[str] = None,
        position: Optional[int] = None,
    ) -> None:
        """Record user interaction with event."""

        # Validate interaction type
        valid_types = [t.value for t in InteractionType]
        if interaction_type not in valid_types:
            raise ValueError(f"Invalid interaction type: {interaction_type}")

        # For DuckDB, we need to handle upsert manually
        # First try to update existing interaction
        update_result = await self._execute_query(
            """
            SELECT COUNT(*) FROM interactions 
            WHERE user_id = ? AND event_id = ? AND interaction_type = ?
        """,
            [user_id, event_id, interaction_type],
        )

        if update_result[0][0] > 0:
            # Update existing interaction
            await self._execute_command(
                """
                UPDATE interactions 
                SET rating = ?, source = ?, position = ?, timestamp = CURRENT_TIMESTAMP
                WHERE user_id = ? AND event_id = ? AND interaction_type = ?
            """,
                [rating, source, position, user_id, event_id, interaction_type],
            )
        else:
            # Get next available ID (manual ID generation for DuckDB)
            next_id_result = await self._execute_query(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM interactions"
            )
            next_id = next_id_result[0][0]

            # Insert new interaction with manual ID
            await self._execute_command(
                """
                INSERT INTO interactions (id, user_id, event_id, interaction_type, rating, source, position)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    next_id,
                    user_id,
                    event_id,
                    interaction_type,
                    rating,
                    source,
                    position,
                ],
            )

    async def get_user_interactions(
        self, user_id: str, limit: int = 50, interaction_type: Optional[str] = None
    ) -> List[InteractionResponse]:
        """Get user interaction history."""

        conditions = ["user_id = ?"]
        params = [user_id]

        if interaction_type:
            conditions.append("interaction_type = ?")
            params.append(interaction_type)

        params.extend([limit])

        where_clause = " AND ".join(conditions)

        result = await self._execute_query(
            f"""
            SELECT id, user_id, event_id, interaction_type, rating, timestamp, source, position
            FROM interactions
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            params,
        )

        interactions = []
        for row in result:
            interaction = InteractionResponse(
                id=row[0],
                user_id=row[1],
                event_id=row[2],
                interaction_type=row[3],
                rating=row[4],
                timestamp=row[5],
                source=row[6],
                position=row[7],
            )
            interactions.append(interaction)

        return interactions

    # Sync wrapper methods for tests
    
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection (sync method for tests)."""
        if not self._connection:
            self._connection = duckdb.connect(self.config.db_path)
            # Ensure tables exist
            self._create_tables_sync()
        return self._connection
    
    def _create_tables_sync(self):
        """Create database tables synchronously."""
        try:
            # Events table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    start_time TIMESTAMP NOT NULL,
                    venue_id TEXT NOT NULL,
                    social_link TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Venues table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS venues (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    address TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Artists table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS artists (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Event-Artist junction table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS event_artists (
                    id INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    artist_id TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events(id),
                    FOREIGN KEY (artist_id) REFERENCES artists(id)
                )
            """)
            
            # Interactions table (upvote/downvote)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (event_id) REFERENCES events(id)
                )
            """)
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")

    # Event Operations
    
    async def store_event(self, event_data: Dict[str, Any]) -> str:
        """Store event. Returns event_id. Check for duplicates using venue_id + date."""
        import uuid
        
        event_id = str(uuid.uuid4())
        
        # Check for duplicates (same venue + date)
        existing = await self._execute_query("""
            SELECT id FROM events 
            WHERE venue_id = ? AND DATE(start_time) = DATE(?)
            AND status != 'cancelled'
        """, [event_data.get('venue_id'), event_data.get('start_time')])
        
        if existing:
            logger.info(f"Duplicate event found for venue {event_data.get('venue_id')} on {event_data.get('start_time')}")
            return existing[0]['id']
        
        await self._execute_command("""
            INSERT INTO events (id, title, description, start_time, venue_id, social_link, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            event_id,
            event_data.get('title'),
            event_data.get('description'),
            event_data.get('start_time'),
            event_data.get('venue_id'), 
            event_data.get('social_link'),
            event_data.get('status', 'active')
        ])
        
        # Handle artists if provided
        if 'artists' in event_data and event_data['artists']:
            for artist_name in event_data['artists']:
                artist_id = await self.store_artist({'name': artist_name})
                await self._execute_command("""
                    INSERT INTO event_artists (event_id, artist_id)
                    VALUES (?, ?)
                """, [event_id, artist_id])
        
        return event_id
    
    async def get_event_by_id(self, event_id: str) -> Optional[EventResponse]:
        """Get single event by ID. Return None if not found."""
        results = await self._execute_query("""
            SELECT e.*, v.name as venue_name, v.address as venue_address,
                   v.latitude, v.longitude,
                   GROUP_CONCAT(a.name, ', ') as artists
            FROM events e
            LEFT JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            WHERE e.id = ?
            GROUP BY e.id
        """, [event_id])
        
        if not results:
            return None
            
        event_data = results[0]
        return EventResponse(
            id=event_data['id'],
            title=event_data['title'],
            description=event_data.get('description'),
            start_time=event_data['start_time'],
            venue=VenueResponse(
                id=event_data['venue_id'],
                name=event_data.get('venue_name', ''),
                address=event_data.get('venue_address', ''),
                latitude=event_data.get('latitude'),
                longitude=event_data.get('longitude')
            ),
            artists=event_data.get('artists', '').split(', ') if event_data.get('artists') else [],
            social_link=event_data.get('social_link')
        )
    
    async def get_all_events(self) -> List[EventResponse]:
        """Get all events (past + future) sorted by popularity (interaction score)."""
        results = await self._execute_query("""
            SELECT e.*, v.name as venue_name, v.address as venue_address,
                   v.latitude, v.longitude,
                   GROUP_CONCAT(a.name, ', ') as artists,
                   COUNT(CASE WHEN i.interaction_type = 'upvote' THEN 1 END) as upvotes,
                   COUNT(CASE WHEN i.interaction_type = 'downvote' THEN 1 END) as downvotes
            FROM events e
            LEFT JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            LEFT JOIN interactions i ON e.id = i.event_id
            WHERE e.status != 'cancelled'
            GROUP BY e.id, e.title, e.description, e.start_time, e.venue_id, e.social_link, e.status, e.created_at
            ORDER BY (upvotes - downvotes) DESC, e.start_time DESC
        """)
        
        events = []
        for event_data in results:
            events.append(EventResponse(
                id=event_data['id'],
                title=event_data['title'], 
                description=event_data.get('description'),
                start_time=event_data['start_time'],
                venue=VenueResponse(
                    id=event_data['venue_id'],
                    name=event_data.get('venue_name', ''),
                    address=event_data.get('venue_address', ''),
                    latitude=event_data.get('latitude'),
                    longitude=event_data.get('longitude')
                ),
                artists=event_data.get('artists', '').split(', ') if event_data.get('artists') else [],
                social_link=event_data.get('social_link')
            ))
        
        return events
    
    async def get_upcoming_events(self, limit: int = 50) -> List[EventResponse]:
        """Get future events only, sorted by date."""
        results = await self._execute_query("""
            SELECT e.*, v.name as venue_name, v.address as venue_address,
                   v.latitude, v.longitude,
                   GROUP_CONCAT(a.name, ', ') as artists
            FROM events e
            LEFT JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            WHERE e.start_time > CURRENT_TIMESTAMP AND e.status != 'cancelled'
            GROUP BY e.id
            ORDER BY e.start_time ASC
            LIMIT ?
        """, [limit])
        
        events = []
        for event_data in results:
            events.append(EventResponse(
                id=event_data['id'],
                title=event_data['title'],
                description=event_data.get('description'),
                start_time=event_data['start_time'],
                venue=VenueResponse(
                    id=event_data['venue_id'],
                    name=event_data.get('venue_name', ''),
                    address=event_data.get('venue_address', ''),
                    latitude=event_data.get('latitude'),
                    longitude=event_data.get('longitude')
                ),
                artists=event_data.get('artists', '').split(', ') if event_data.get('artists') else [],
                social_link=event_data.get('social_link')
            ))
        
        return events
    
    async def search_events(self, query: str) -> List[EventResponse]:
        """Keyword search on event title, description, artist names."""
        search_term = f"%{query}%"
        results = await self._execute_query("""
            SELECT DISTINCT e.*, v.name as venue_name, v.address as venue_address,
                   v.latitude, v.longitude,
                   GROUP_CONCAT(a.name, ', ') as artists
            FROM events e
            LEFT JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            WHERE (e.title LIKE ? OR e.description LIKE ? OR a.name LIKE ?)
            AND e.status != 'cancelled'
            GROUP BY e.id
            ORDER BY e.start_time DESC
        """, [search_term, search_term, search_term])
        
        events = []
        for event_data in results:
            events.append(EventResponse(
                id=event_data['id'],
                title=event_data['title'],
                description=event_data.get('description'),
                start_time=event_data['start_time'],
                venue=VenueResponse(
                    id=event_data['venue_id'],
                    name=event_data.get('venue_name', ''),
                    address=event_data.get('venue_address', ''),
                    latitude=event_data.get('latitude'),
                    longitude=event_data.get('longitude')
                ),
                artists=event_data.get('artists', '').split(', ') if event_data.get('artists') else [],
                social_link=event_data.get('social_link')
            ))
        
        return events
    
    async def get_events_by_venue(self, venue_id: str) -> List[EventResponse]:
        """Get all events for a venue (past + future) for preference calibration."""
        results = await self._execute_query("""
            SELECT e.*, v.name as venue_name, v.address as venue_address,
                   v.latitude, v.longitude,
                   GROUP_CONCAT(a.name, ', ') as artists
            FROM events e
            LEFT JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            WHERE e.venue_id = ? AND e.status != 'cancelled'
            GROUP BY e.id
            ORDER BY e.start_time DESC
        """, [venue_id])
        
        events = []
        for event_data in results:
            events.append(EventResponse(
                id=event_data['id'],
                title=event_data['title'],
                description=event_data.get('description'),
                start_time=event_data['start_time'],
                venue=VenueResponse(
                    id=event_data['venue_id'],
                    name=event_data.get('venue_name', ''),
                    address=event_data.get('venue_address', ''),
                    latitude=event_data.get('latitude'),
                    longitude=event_data.get('longitude')
                ),
                artists=event_data.get('artists', '').split(', ') if event_data.get('artists') else [],
                social_link=event_data.get('social_link')
            ))
        
        return events
    
    async def get_events_in_date_range(self, start_date: datetime, end_date: datetime) -> List[EventResponse]:
        """Filter events by date range."""
        results = await self._execute_query("""
            SELECT e.*, v.name as venue_name, v.address as venue_address,
                   v.latitude, v.longitude,
                   GROUP_CONCAT(a.name, ', ') as artists
            FROM events e
            LEFT JOIN venues v ON e.venue_id = v.id
            LEFT JOIN event_artists ea ON e.id = ea.event_id
            LEFT JOIN artists a ON ea.artist_id = a.id
            WHERE e.start_time >= ? AND e.start_time <= ? AND e.status != 'cancelled'
            GROUP BY e.id
            ORDER BY e.start_time ASC
        """, [start_date, end_date])
        
        events = []
        for event_data in results:
            events.append(EventResponse(
                id=event_data['id'],
                title=event_data['title'],
                description=event_data.get('description'),
                start_time=event_data['start_time'],
                venue=VenueResponse(
                    id=event_data['venue_id'],
                    name=event_data.get('venue_name', ''),
                    address=event_data.get('venue_address', ''),
                    latitude=event_data.get('latitude'),
                    longitude=event_data.get('longitude')
                ),
                artists=event_data.get('artists', '').split(', ') if event_data.get('artists') else [],
                social_link=event_data.get('social_link')
            ))
        
        return events

    # Venue Operations
    
    async def store_venue(self, venue_data: Dict[str, Any]) -> str:
        """Store venue with address, location coords. Returns venue_id."""
        import uuid
        
        venue_id = str(uuid.uuid4())
        
        await self._execute_command("""
            INSERT INTO venues (id, name, address, latitude, longitude)
            VALUES (?, ?, ?, ?, ?)
        """, [
            venue_id,
            venue_data.get('name'),
            venue_data.get('address'),
            venue_data.get('latitude'),
            venue_data.get('longitude')
        ])
        
        return venue_id
    
    async def get_venue_by_id(self, venue_id: str) -> Optional[VenueResponse]:
        """Get venue by ID."""
        results = await self._execute_query("""
            SELECT * FROM venues WHERE id = ?
        """, [venue_id])
        
        if not results:
            return None
            
        venue_data = results[0]
        return VenueResponse(
            id=venue_data['id'],
            name=venue_data['name'],
            address=venue_data['address'],
            latitude=venue_data.get('latitude'),
            longitude=venue_data.get('longitude')
        )
    
    async def get_all_venues(self) -> List[VenueResponse]:
        """Get all venues."""
        results = await self._execute_query("SELECT * FROM venues ORDER BY name")
        
        venues = []
        for venue_data in results:
            venues.append(VenueResponse(
                id=venue_data['id'],
                name=venue_data['name'],
                address=venue_data['address'],
                latitude=venue_data.get('latitude'),
                longitude=venue_data.get('longitude')
            ))
        
        return venues
    
    async def get_venues_near_location(self, lat: float, lng: float, radius_km: float) -> List[VenueResponse]:
        """Find venues within radius."""
        # Simple distance calculation - for production use proper geospatial functions
        results = await self._execute_query("""
            SELECT *, 
                   (6371 * acos(cos(radians(?)) * cos(radians(latitude)) * 
                   cos(radians(longitude) - radians(?)) + sin(radians(?)) * 
                   sin(radians(latitude)))) AS distance
            FROM venues 
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            HAVING distance < ?
            ORDER BY distance
        """, [lat, lng, lat, radius_km])
        
        venues = []
        for venue_data in results:
            venues.append(VenueResponse(
                id=venue_data['id'],
                name=venue_data['name'],
                address=venue_data['address'],
                latitude=venue_data.get('latitude'),
                longitude=venue_data.get('longitude')
            ))
        
        return venues

    # Artist Operations
    
    async def store_artist(self, artist_data: Dict[str, Any]) -> str:
        """Store artist. Returns artist_id."""
        import uuid
        
        # Check if artist already exists
        existing = await self._execute_query("""
            SELECT id FROM artists WHERE name = ?
        """, [artist_data.get('name')])
        
        if existing:
            return existing[0]['id']
        
        artist_id = str(uuid.uuid4())
        
        await self._execute_command("""
            INSERT INTO artists (id, name)
            VALUES (?, ?)
        """, [artist_id, artist_data.get('name')])
        
        return artist_id
    
    async def get_all_artists(self) -> List[ArtistResponse]:
        """Get all artists."""
        results = await self._execute_query("SELECT * FROM artists ORDER BY name")
        
        artists = []
        for artist_data in results:
            artists.append(ArtistResponse(
                id=artist_data['id'],
                name=artist_data['name']
            ))
        
        return artists
    
    async def search_artists(self, query: str) -> List[ArtistResponse]:
        """Keyword search on artist names."""
        search_term = f"%{query}%"
        results = await self._execute_query("""
            SELECT * FROM artists WHERE name LIKE ? ORDER BY name
        """, [search_term])
        
        artists = []
        for artist_data in results:
            artists.append(ArtistResponse(
                id=artist_data['id'],
                name=artist_data['name']
            ))
        
        return artists

    # User Operations
    
    async def store_user(self, user_data: Dict[str, Any]) -> str:
        """Store user. Returns user_id."""
        import uuid
        
        user_id = str(uuid.uuid4())
        
        await self._execute_command("""
            INSERT INTO users (id, email, name, preferences)
            VALUES (?, ?, ?, ?)
        """, [
            user_id,
            user_data.get('email'),
            user_data.get('name'),
            json.dumps(user_data.get('preferences', {}))
        ])
        
        return user_id
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        results = await self._execute_query("""
            SELECT * FROM users WHERE id = ?
        """, [user_id])
        
        if not results:
            return None
            
        user_data = results[0]
        preferences = json.loads(user_data.get('preferences', '{}'))
        
        return UserResponse(
            id=user_data['id'],
            email=user_data['email'],
            name=user_data.get('name'),
            preferences=preferences
        )
    
    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email."""
        results = await self._execute_query("""
            SELECT * FROM users WHERE email = ?
        """, [email])
        
        if not results:
            return None
            
        user_data = results[0]
        preferences = json.loads(user_data.get('preferences', '{}'))
        
        return UserResponse(
            id=user_data['id'],
            email=user_data['email'],
            name=user_data.get('name'),
            preferences=preferences
        )

    # Interaction Operations (Up/Downvote)
    
    async def store_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """Store upvote/downvote. Returns interaction_id."""
        import uuid
        
        interaction_id = str(uuid.uuid4())
        
        await self._execute_command("""
            INSERT INTO interactions (id, user_id, event_id, interaction_type)
            VALUES (?, ?, ?, ?)
        """, [
            interaction_id,
            interaction_data.get('user_id'),
            interaction_data.get('event_id'),
            interaction_data.get('interaction_type')  # 'upvote' or 'downvote'
        ])
        
        return interaction_id
    
    async def get_event_interactions(self, event_id: str) -> List[InteractionResponse]:
        """Get all interactions for an event."""
        results = await self._execute_query("""
            SELECT * FROM interactions WHERE event_id = ?
            ORDER BY created_at DESC
        """, [event_id])
        
        interactions = []
        for interaction_data in results:
            interactions.append(InteractionResponse(
                id=interaction_data['id'],
                user_id=interaction_data['user_id'],
                event_id=interaction_data['event_id'],
                interaction_type=InteractionType(interaction_data['interaction_type']),
                timestamp=interaction_data['created_at']
            ))
        
        return interactions
    
    async def get_all_interactions(self) -> List[InteractionResponse]:
        """Get all interactions."""
        results = await self._execute_query("""
            SELECT * FROM interactions ORDER BY created_at DESC
        """)
        
        interactions = []
        for interaction_data in results:
            interactions.append(InteractionResponse(
                id=interaction_data['id'],
                user_id=interaction_data['user_id'],
                event_id=interaction_data['event_id'],
                interaction_type=InteractionType(interaction_data['interaction_type']),
                timestamp=interaction_data['created_at']
            ))
        
        return interactions

    # Cleanup

    async def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
