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

    # Cleanup

    async def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
