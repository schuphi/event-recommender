#!/usr/bin/env python3
"""
Analytics service for Copenhagen Event Recommender.
Tracks usage metrics and provides analytics dashboard data.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json

# Import with absolute paths to avoid relative import issues
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.models.responses import AnalyticsResponse
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Single analytics event."""

    event_type: str
    user_id: Optional[str]
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsConfig:
    """Analytics service configuration."""

    buffer_size: int = 1000
    flush_interval_seconds: int = 60
    retention_days: int = 30
    enable_detailed_tracking: bool = True


class AnalyticsService:
    """
    Analytics service for usage tracking and metrics.

    Features:
    - In-memory event buffering for performance
    - Periodic database flushing
    - Real-time metrics calculation
    - Dashboard data aggregation
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize analytics service."""
        self.config = config or AnalyticsConfig()

        # In-memory event buffer
        self._event_buffer: deque = deque(maxlen=self.config.buffer_size)

        # Real-time metrics
        self._metrics = {
            "total_users": set(),
            "active_users_today": set(),
            "total_interactions": 0,
            "total_recommendations": 0,
            "recommendation_requests_today": 0,
            "search_requests_today": 0,
            "user_registrations_today": 0,
        }

        # Daily counters (reset at midnight)
        self._daily_counters = defaultdict(int)
        self._last_reset_date = datetime.now().date()

        # Background task for flushing
        self._flush_task = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize analytics service."""
        logger.info("Initializing AnalyticsService...")

        self._running = True

        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        # Load existing metrics from database
        await self._load_existing_metrics()

        logger.info("AnalyticsService initialized")

    async def cleanup(self) -> None:
        """Cleanup analytics service."""
        logger.info("Cleaning up AnalyticsService...")

        self._running = False

        # Cancel background task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_events()

        logger.info("AnalyticsService cleanup completed")

    async def _flush_loop(self) -> None:
        """Background task to flush events to database."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_events()
                await self._reset_daily_counters_if_needed()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analytics flush loop error: {e}")

    async def _flush_events(self) -> None:
        """Flush buffered events to database."""
        if not self._event_buffer:
            return

        try:
            # Get events to flush
            events_to_flush = list(self._event_buffer)
            self._event_buffer.clear()

            logger.debug(f"Flushing {len(events_to_flush)} analytics events")

            # Write to database
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            for event in events_to_flush:
                # For now, just log to recommendation_logs table
                # In a full implementation, we'd have a dedicated analytics table
                await self._write_event_to_db(db, event)

            await db.close()
            logger.debug("Analytics events flushed successfully")

        except Exception as e:
            logger.error(f"Failed to flush analytics events: {e}")

    async def _write_event_to_db(
        self, db: "DatabaseService", event: AnalyticsEvent
    ) -> None:
        """Write single analytics event to database."""
        # This is a simplified implementation
        # In production, you'd want a dedicated analytics table

        try:
            if event.event_type == "recommendation_request":
                # Write to recommendation_logs table
                await db._execute_command(
                    """
                    INSERT INTO recommendation_logs (
                        user_id, request_timestamp, filters, 
                        model_version, response_time_ms
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    [
                        event.user_id,
                        event.timestamp,
                        json.dumps(event.data.get("filters", {})),
                        event.data.get("model_version", "1.0.0"),
                        event.data.get("response_time_ms", 0),
                    ],
                )
        except Exception as e:
            logger.warning(f"Failed to write analytics event to DB: {e}")

    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if date has changed."""
        current_date = datetime.now().date()
        if current_date != self._last_reset_date:
            logger.info("Resetting daily analytics counters")

            self._daily_counters.clear()
            self._metrics["active_users_today"].clear()
            self._metrics["recommendation_requests_today"] = 0
            self._metrics["search_requests_today"] = 0
            self._metrics["user_registrations_today"] = 0

            self._last_reset_date = current_date

    async def _load_existing_metrics(self) -> None:
        """Load existing metrics from database for dashboard."""
        try:
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            # Load total users
            users_result = await db._execute_query("SELECT COUNT(*) FROM users")
            if users_result:
                self._metrics["total_users"] = set(range(users_result[0][0]))

            # Load total interactions
            interactions_result = await db._execute_query(
                "SELECT COUNT(*) FROM interactions"
            )
            if interactions_result:
                self._metrics["total_interactions"] = interactions_result[0][0]

            # Load recommendation requests (approximate from logs)
            recommendations_result = await db._execute_query(
                "SELECT COUNT(*) FROM recommendation_logs"
            )
            if recommendations_result:
                self._metrics["total_recommendations"] = recommendations_result[0][0]

            await db.close()
            logger.info("Loaded existing analytics metrics")

        except Exception as e:
            logger.warning(f"Failed to load existing metrics: {e}")

    def _track_event(
        self, event_type: str, user_id: Optional[str] = None, **data
    ) -> None:
        """Track analytics event (synchronous)."""
        event = AnalyticsEvent(
            event_type=event_type, user_id=user_id, timestamp=datetime.now(), data=data
        )

        self._event_buffer.append(event)
        self._update_real_time_metrics(event)

    def _update_real_time_metrics(self, event: AnalyticsEvent) -> None:
        """Update real-time metrics."""
        if event.user_id:
            self._metrics["total_users"].add(event.user_id)
            self._metrics["active_users_today"].add(event.user_id)

        if event.event_type == "recommendation_request":
            self._metrics["total_recommendations"] += 1
            self._metrics["recommendation_requests_today"] += 1

        elif event.event_type == "search_request":
            self._metrics["search_requests_today"] += 1

        elif event.event_type == "user_registration":
            self._metrics["user_registrations_today"] += 1

        elif event.event_type == "interaction":
            self._metrics["total_interactions"] += 1

    # Public tracking methods (called by API endpoints)

    async def track_user_registration(self, user_id: str) -> None:
        """Track user registration."""
        self._track_event("user_registration", user_id=user_id)

    async def track_preference_update(self, user_id: str) -> None:
        """Track user preference update."""
        self._track_event("preference_update", user_id=user_id)

    async def track_recommendation_request(
        self,
        user_id: Optional[str],
        num_returned: int,
        filters: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[int] = None,
        model_version: str = "1.0.0",
    ) -> None:
        """Track recommendation request."""
        self._track_event(
            "recommendation_request",
            user_id=user_id,
            num_returned=num_returned,
            filters=filters,
            response_time_ms=response_time_ms,
            model_version=model_version,
        )

    async def track_similar_events_request(
        self, event_id: str, num_returned: int
    ) -> None:
        """Track similar events request."""
        self._track_event(
            "similar_events_request", event_id=event_id, num_returned=num_returned
        )

    async def track_interaction(
        self, user_id: str, event_id: str, interaction_type: str
    ) -> None:
        """Track user interaction."""
        self._track_event(
            "interaction",
            user_id=user_id,
            event_id=event_id,
            interaction_type=interaction_type,
        )

    async def track_search(
        self, query: str, num_results: int, user_id: Optional[str] = None
    ) -> None:
        """Track search request."""
        self._track_event(
            "search_request", user_id=user_id, query=query, num_results=num_results
        )

    # Analytics dashboard methods

    async def get_dashboard_data(self, days: int = 7) -> AnalyticsResponse:
        """Get analytics dashboard data."""
        logger.info(f"Generating analytics dashboard for {days} days")

        try:
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get event analytics
            events_result = await db._execute_query(
                """
                SELECT COUNT(*) FROM events WHERE created_at >= ?
            """,
                [start_date],
            )
            total_events = events_result[0][0] if events_result else 0

            # Get user analytics
            users_result = await db._execute_query("SELECT COUNT(*) FROM users")
            total_users = users_result[0][0] if users_result else 0

            active_users_result = await db._execute_query(
                """
                SELECT COUNT(DISTINCT user_id) FROM interactions 
                WHERE timestamp >= ?
            """,
                [start_date],
            )
            active_users = active_users_result[0][0] if active_users_result else 0

            # Get interaction analytics
            interactions_result = await db._execute_query(
                """
                SELECT COUNT(*) FROM interactions WHERE timestamp >= ?
            """,
                [start_date],
            )
            total_interactions = interactions_result[0][0] if interactions_result else 0

            # Get genre breakdown
            genres_result = await db._execute_query(
                """
                SELECT a.genres, COUNT(*) as count
                FROM events e
                JOIN event_artists ea ON e.id = ea.event_id  
                JOIN artists a ON ea.artist_id = a.id
                WHERE e.created_at >= ?
                GROUP BY a.genres
                LIMIT 10
            """,
                [start_date],
            )

            events_by_genre = {}
            for row in genres_result:
                if row[0]:
                    try:
                        genre_list = json.loads(row[0])
                        for genre in genre_list:
                            events_by_genre[genre] = (
                                events_by_genre.get(genre, 0) + row[1]
                            )
                    except:
                        pass

            # Get neighborhood breakdown
            neighborhoods_result = await db._execute_query(
                """
                SELECT v.neighborhood, COUNT(*) as count
                FROM events e
                JOIN venues v ON e.venue_id = v.id
                WHERE e.created_at >= ?
                GROUP BY v.neighborhood
            """,
                [start_date],
            )

            events_by_neighborhood = {
                row[0]: row[1] for row in neighborhoods_result if row[0]
            }

            # Get popular venues
            venues_result = await db._execute_query(
                """
                SELECT v.name, v.neighborhood, COUNT(*) as event_count
                FROM events e  
                JOIN venues v ON e.venue_id = v.id
                WHERE e.created_at >= ?
                GROUP BY v.id, v.name, v.neighborhood
                ORDER BY event_count DESC
                LIMIT 5
            """,
                [start_date],
            )

            popular_venues = [
                {"name": row[0], "neighborhood": row[1], "event_count": row[2]}
                for row in venues_result
            ]

            # Get interaction breakdown
            interaction_breakdown_result = await db._execute_query(
                """
                SELECT interaction_type, COUNT(*) as count
                FROM interactions
                WHERE timestamp >= ?
                GROUP BY interaction_type
            """,
                [start_date],
            )

            interaction_breakdown = {
                row[0]: row[1] for row in interaction_breakdown_result
            }

            await db.close()

            # Calculate derived metrics
            recommendation_ctr = 0.1  # Placeholder
            avg_session_length = 5.2  # Placeholder in minutes
            avg_recommendation_score = 0.75  # Placeholder
            cold_start_percentage = 0.3  # Placeholder

            # Build response
            response = AnalyticsResponse(
                period_days=days,
                total_users=total_users,
                active_users=active_users,
                total_interactions=total_interactions,
                total_recommendations=self._metrics["total_recommendations"],
                total_events=total_events,
                events_by_genre=events_by_genre,
                events_by_neighborhood=events_by_neighborhood,
                popular_venues=popular_venues,
                interaction_breakdown=interaction_breakdown,
                avg_session_length=avg_session_length,
                recommendation_ctr=recommendation_ctr,
                avg_recommendation_score=avg_recommendation_score,
                cold_start_percentage=cold_start_percentage,
                timestamp=datetime.now(),
            )

            logger.info("Analytics dashboard generated successfully")
            return response

        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")

            # Return empty response on error
            return AnalyticsResponse(
                period_days=days,
                total_users=0,
                active_users=0,
                total_interactions=0,
                total_recommendations=0,
                total_events=0,
                events_by_genre={},
                events_by_neighborhood={},
                popular_venues=[],
                interaction_breakdown={},
                avg_session_length=0.0,
                recommendation_ctr=0.0,
                avg_recommendation_score=0.0,
                cold_start_percentage=0.0,
                timestamp=datetime.now(),
            )

    async def get_event_analytics(
        self, event_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Get analytics for specific event."""
        logger.info(f"Getting analytics for event {event_id}")

        try:
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get event details
            event = await db.get_event(event_id)
            if not event:
                return {"error": "Event not found"}

            # Get interaction stats for this event
            interactions_result = await db._execute_query(
                """
                SELECT interaction_type, COUNT(*) as count
                FROM interactions
                WHERE event_id = ? AND timestamp >= ?
                GROUP BY interaction_type
            """,
                [event_id, start_date],
            )

            interactions = {row[0]: row[1] for row in interactions_result}

            # Get total views/interactions
            total_interactions = sum(interactions.values())

            # Get recommendation stats (how often this event was recommended)
            recommendation_stats = {
                "times_recommended": 0,  # Would need to parse recommendation_logs
                "avg_recommendation_score": 0.0,
                "conversion_rate": 0.0,
            }

            await db.close()

            return {
                "event_id": event_id,
                "event_title": event.title,
                "venue": event.venue_name,
                "period_days": days,
                "total_interactions": total_interactions,
                "interaction_breakdown": interactions,
                "recommendation_stats": recommendation_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get event analytics: {e}")
            return {"error": str(e)}
