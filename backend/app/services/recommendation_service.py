#!/usr/bin/env python3
"""
Recommendation service for Copenhagen Event Recommender.
Orchestrates content-based, collaborative filtering, and hybrid ML models.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path
from dataclasses import dataclass

# Import with absolute paths to avoid relative import issues
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # For ML modules

from backend.app.models.requests import (
    UserPreferencesRequest,
    RecommendationFilters,
    DateFilter,
)
from backend.app.models.responses import (
    RecommendationResponse,
    RecommendedEvent,
    EventResponse,
    RecommendationExplanation,
    ModelStatusResponse,
)
from backend.app.core.config import settings
from backend.app.services.custom_collaborative_filter import (
    SingleUserCollaborativeFilter,
)

logger = logging.getLogger(__name__)


@dataclass
class RecommendationConfig:
    """Configuration for recommendation service."""

    models_dir: str = "ml/models"
    enable_content_based: bool = True
    enable_collaborative: bool = True
    enable_hybrid: bool = True
    fallback_to_popularity: bool = True
    model_load_timeout: int = 30


class RecommendationService:
    """
    Recommendation service that orchestrates multiple ML models.

    Architecture:
    1. Content-based filtering (semantic similarity + structured features)
    2. Collaborative filtering (user-item interactions)
    3. Hybrid ranking (neural network combining both)
    4. Fallback to popularity-based recommendations
    """

    def __init__(self, config: Optional[RecommendationConfig] = None):
        """Initialize recommendation service."""
        self.config = config or RecommendationConfig()

        # ML Models (loaded lazily)
        self._content_model = None
        self._collaborative_model = None
        self._hybrid_model = None
        self._langchain_recommender = None
        self._custom_collaborative_filter = None

        # Model status
        self._models_loaded = False
        self._model_load_error = None
        self._last_training = None

        # Performance tracking
        self._recommendation_count = 0
        self._avg_response_time = 0.0

    async def initialize(self) -> None:
        """Initialize and load all ML models."""
        logger.info("Initializing RecommendationService...")

        try:
            await self._load_models()
            self._models_loaded = True
            logger.info("RecommendationService initialized successfully")

        except Exception as e:
            self._model_load_error = str(e)
            logger.error(f"Failed to initialize RecommendationService: {e}")
            if not self.config.fallback_to_popularity:
                raise
            logger.warning("Continuing with fallback recommendations only")

    async def _load_models(self) -> None:
        """Load ML models in parallel."""
        logger.info("Loading ML models...")

        # Load models in thread pool for async behavior
        loop = asyncio.get_event_loop()

        tasks = []

        # Load content-based model
        if self.config.enable_content_based:
            tasks.append(loop.run_in_executor(None, self._load_content_model))

        # Load collaborative filtering model
        if self.config.enable_collaborative:
            tasks.append(loop.run_in_executor(None, self._load_collaborative_model))

        # Load hybrid model
        if self.config.enable_hybrid:
            tasks.append(loop.run_in_executor(None, self._load_hybrid_model))

        # Load LangChain recommender for semantic search
        tasks.append(loop.run_in_executor(None, self._load_langchain_recommender))

        # Load custom collaborative filter
        tasks.append(loop.run_in_executor(None, self._load_custom_collaborative_filter))

        # Execute all loading tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("ML models loaded")

    def _load_content_model(self):
        """Load content-based recommendation model."""
        try:
            from ml.models.content_based import ContentBasedRecommender
            from ml.embeddings.content_embedder import ContentEmbedder

            embedder = ContentEmbedder()
            self._content_model = ContentBasedRecommender(
                embedder=embedder,
                model_cache_dir=self.config.models_dir + "/content_based",
            )
            logger.info("Content-based model loaded")

        except Exception as e:
            logger.warning(f"Failed to load content-based model: {e}")
            self._content_model = None

    def _load_collaborative_model(self):
        """Load collaborative filtering model."""
        try:
            from ml.models.collaborative_filtering import (
                CollaborativeFilteringRecommender,
            )

            self._collaborative_model = CollaborativeFilteringRecommender(
                model_cache_dir=self.config.models_dir + "/collaborative"
            )
            logger.info("Collaborative filtering model loaded")

        except Exception as e:
            logger.warning(f"Failed to load collaborative model: {e}")
            self._collaborative_model = None

    def _load_hybrid_model(self):
        """Load hybrid ranking model."""
        try:
            from ml.models.hybrid_ranker import HybridRecommender

            self._hybrid_model = HybridRecommender(
                content_recommender=self._content_model,
                cf_recommender=self._collaborative_model,
                model_cache_dir=self.config.models_dir + "/hybrid",
            )
            logger.info("Hybrid ranking model loaded")

        except Exception as e:
            logger.warning(f"Failed to load hybrid model: {e}")
            self._hybrid_model = None

    def _load_langchain_recommender(self):
        """Load LangChain recommender for semantic search."""
        try:
            from backend.app.services.langchain_recommender import recommender

            self._langchain_recommender = recommender
            logger.info("LangChain recommender loaded")

        except Exception as e:
            logger.warning(f"Failed to load LangChain recommender: {e}")
            self._langchain_recommender = None

    def _load_custom_collaborative_filter(self):
        """Load custom collaborative filter."""
        try:
            self._custom_collaborative_filter = SingleUserCollaborativeFilter()
            logger.info("Custom collaborative filter loaded")

        except Exception as e:
            logger.warning(f"Failed to load custom collaborative filter: {e}")
            self._custom_collaborative_filter = None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up RecommendationService...")

        # Cleanup models if they have cleanup methods
        if self._content_model and hasattr(self._content_model, "cleanup"):
            await asyncio.get_event_loop().run_in_executor(
                None, self._content_model.cleanup
            )

        if self._collaborative_model and hasattr(self._collaborative_model, "cleanup"):
            await asyncio.get_event_loop().run_in_executor(
                None, self._collaborative_model.cleanup
            )

        logger.info("RecommendationService cleanup completed")

    async def health_check(self) -> bool:
        """Check if recommendation service is healthy."""
        if not self._models_loaded:
            return False

        # Check if at least one model is available
        available_models = sum(
            [
                self._content_model is not None,
                self._collaborative_model is not None,
                self._hybrid_model is not None,
                self._langchain_recommender is not None,
                self._custom_collaborative_filter is not None,
            ]
        )

        return available_models > 0 or self.config.fallback_to_popularity

    async def get_recommendations(
        self,
        user_id: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        location_lat: Optional[float] = None,
        location_lon: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        num_recommendations: int = 10,
        use_collaborative: bool = True,
        diversity_factor: float = 0.1,
    ) -> RecommendationResponse:
        """
        Get personalized event recommendations.

        Args:
            user_id: User identifier
            user_preferences: User preference dictionary
            location_lat: User latitude
            location_lon: User longitude
            filters: Filtering options
            num_recommendations: Number of recommendations to return
            use_collaborative: Whether to use collaborative filtering
            diversity_factor: Diversity promotion factor (0-1)

        Returns:
            RecommendationResponse with ranked recommendations
        """
        start_time = datetime.now()
        session_id = str(uuid.uuid4())

        try:
            logger.info(
                f"Generating recommendations for user {user_id}, session {session_id}"
            )

            # Get candidate events from database
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            # Apply basic filters to get candidate events
            candidate_events = await self._get_candidate_events(
                db, filters, location_lat, location_lon
            )

            if not candidate_events:
                logger.warning("No candidate events found")
                return self._empty_recommendation_response(session_id, start_time)

            logger.info(f"Found {len(candidate_events)} candidate events")

            # Generate recommendations using available models
            recommendations = await self._generate_recommendations(
                candidate_events=candidate_events,
                user_id=user_id,
                user_preferences=user_preferences,
                location_lat=location_lat,
                location_lon=location_lon,
                use_collaborative=use_collaborative,
                diversity_factor=diversity_factor,
                num_recommendations=num_recommendations,
            )

            # Build response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)

            response = RecommendationResponse(
                user_id=user_id,
                session_id=session_id,
                events=recommendations,
                total_candidates=len(candidate_events),
                model_version="1.0.0",
                timestamp=datetime.now(),
                processing_time_ms=int(processing_time),
                filters_applied=filters,
                cold_start=(user_id is None),
            )

            logger.info(
                f"Generated {len(recommendations)} recommendations in {processing_time:.1f}ms"
            )
            return response

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Return fallback recommendations
            return await self._fallback_recommendations(
                session_id, start_time, num_recommendations
            )

    async def _get_candidate_events(
        self,
        db: "DatabaseService",
        filters: Optional[Dict[str, Any]],
        location_lat: Optional[float],
        location_lon: Optional[float],
    ) -> List[EventResponse]:
        """Get candidate events from database with basic filtering."""

        # Convert filters to database query parameters
        filter_params = {}

        if filters:
            if filters.get("min_price"):
                filter_params["min_price"] = filters["min_price"]
            if filters.get("max_price"):
                filter_params["max_price"] = filters["max_price"]
            if filters.get("neighborhoods"):
                # Use first neighborhood for now
                filter_params["neighborhood"] = filters["neighborhoods"][0]
            if filters.get("genres"):
                filter_params["genres"] = filters["genres"]

        # Date filtering
        now = datetime.now()
        filter_params["date_from"] = now

        if filters and filters.get("date_filter"):
            date_filter = filters["date_filter"]
            if date_filter == "today":
                filter_params["date_to"] = now.replace(hour=23, minute=59)
            elif date_filter == "this_week":
                filter_params["date_to"] = now + timedelta(days=7)
            elif date_filter == "this_month":
                filter_params["date_to"] = now + timedelta(days=30)
        else:
            # Default: next 30 days
            filter_params["date_to"] = now + timedelta(days=30)

        # Get events from database
        events = await db.get_events(
            limit=100, **filter_params  # Get more candidates for better recommendations
        )

        await db.close()
        return events

    async def _generate_recommendations(
        self,
        candidate_events: List[EventResponse],
        user_id: Optional[str],
        user_preferences: Optional[Dict[str, Any]],
        location_lat: Optional[float],
        location_lon: Optional[float],
        use_collaborative: bool,
        diversity_factor: float,
        num_recommendations: int,
    ) -> List[RecommendedEvent]:
        """Generate recommendations using available ML models."""

        # Strategy: Try models in order of preference
        # 1. Hybrid model (if available and user has history)
        # 2. Content-based model
        # 3. LangChain semantic search
        # 4. Popularity-based fallback

        if (
            self._hybrid_model
            and self._content_model
            and self._collaborative_model
            and user_id
            and use_collaborative
        ):

            return await self._hybrid_recommendations(
                candidate_events,
                user_id,
                user_preferences,
                location_lat,
                location_lon,
                diversity_factor,
                num_recommendations,
            )

        elif self._custom_collaborative_filter and user_id:
            return await self._custom_collaborative_recommendations(
                candidate_events,
                user_id,
                user_preferences,
                location_lat,
                location_lon,
                num_recommendations,
            )

        elif self._content_model:
            return await self._content_based_recommendations(
                candidate_events,
                user_preferences,
                location_lat,
                location_lon,
                num_recommendations,
            )

        elif self._langchain_recommender:
            return await self._langchain_recommendations(
                candidate_events, user_preferences, num_recommendations
            )

        else:
            return await self._popularity_based_recommendations(
                candidate_events, num_recommendations
            )

    async def _popularity_based_recommendations(
        self, candidate_events: List[EventResponse], num_recommendations: int
    ) -> List[RecommendedEvent]:
        """Fallback to popularity-based recommendations."""

        logger.info("Using popularity-based recommendations")

        # Sort by popularity score (with some randomization)
        import random

        scored_events = []
        for i, event in enumerate(candidate_events):
            # Combine popularity score with slight randomization
            score = event.popularity_score + random.uniform(-0.1, 0.1)
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]

            recommended_event = RecommendedEvent(
                event=event,
                recommendation_score=score,
                rank=i + 1,
                explanation=RecommendationExplanation(
                    overall_score=score,
                    components={"popularity": event.popularity_score},
                    reasons=["Popular event", "Based on general popularity"],
                    model_confidence=0.5,
                ),
            )
            scored_events.append((score, recommended_event))

        # Sort by score and return top N
        scored_events.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        for rank, (score, rec_event) in enumerate(scored_events[:num_recommendations]):
            rec_event.rank = rank + 1
            recommendations.append(rec_event)

        return recommendations

    async def _langchain_recommendations(
        self,
        candidate_events: List[EventResponse],
        user_preferences: Optional[Dict[str, Any]],
        num_recommendations: int,
    ) -> List[RecommendedEvent]:
        """Use LangChain for semantic recommendations."""

        logger.info("Using LangChain semantic recommendations")

        try:
            # Build search query from user preferences
            search_query = self._build_search_query_from_preferences(user_preferences)

            # Get LangChain recommendations
            loop = asyncio.get_event_loop()
            langchain_results = await loop.run_in_executor(
                None,
                lambda: self._langchain_recommender.get_recommendations(
                    user_preferences=user_preferences or {},
                    num_recommendations=num_recommendations * 2,  # Get more candidates
                ),
            )

            # Convert to RecommendedEvent format
            recommendations = []
            event_id_map = {event.event_id: event for event in candidate_events}

            for i, rec in enumerate(langchain_results[:num_recommendations]):
                if rec.event_id in event_id_map:
                    event = event_id_map[rec.event_id]

                    recommended_event = RecommendedEvent(
                        event=event,
                        recommendation_score=rec.score,
                        rank=i + 1,
                        explanation=RecommendationExplanation(
                            overall_score=rec.score,
                            components={"semantic_similarity": rec.score},
                            reasons=rec.reasons,
                            model_confidence=0.8,
                        ),
                    )
                    recommendations.append(recommended_event)

            return recommendations

        except Exception as e:
            logger.warning(f"LangChain recommendations failed: {e}")
            return await self._popularity_based_recommendations(
                candidate_events, num_recommendations
            )

    def _build_search_query_from_preferences(
        self, preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Build search query from user preferences."""
        if not preferences:
            return "popular events in Copenhagen"

        query_parts = []

        if preferences.get("preferred_genres"):
            genres = preferences["preferred_genres"][:3]  # Limit to top 3
            query_parts.append(f"Music genres: {', '.join(genres)}")

        if preferences.get("preferred_artists"):
            artists = preferences["preferred_artists"][:2]  # Limit to top 2
            query_parts.append(f"Artists like: {', '.join(artists)}")

        if preferences.get("preferred_venues"):
            venues = preferences["preferred_venues"][:2]
            query_parts.append(f"Venues: {', '.join(venues)}")

        return " ".join(query_parts) if query_parts else "popular events in Copenhagen"

    async def _custom_collaborative_recommendations(
        self,
        candidate_events: List[EventResponse],
        user_id: str,
        user_preferences: Optional[Dict[str, Any]],
        location_lat: Optional[float],
        location_lon: Optional[float],
        num_recommendations: int,
    ) -> List[RecommendedEvent]:
        """Generate recommendations using custom collaborative filter."""

        logger.info("Using custom collaborative filtering recommendations")

        try:
            # Get user's interaction history
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            # Get user interactions
            user_interactions_data = await db.get_user_interactions(user_id, limit=100)

            # Convert to the format expected by collaborative filter
            user_interactions = []
            for interaction in user_interactions_data:
                user_interactions.append(
                    {
                        "event_id": interaction.event_id,
                        "interaction_type": interaction.interaction_type,
                        "rating": interaction.rating,
                        "timestamp": interaction.timestamp,
                    }
                )

            await db.close()

            if not user_interactions:
                logger.info("No user interactions found, falling back to popularity")
                return await self._popularity_based_recommendations(
                    candidate_events, num_recommendations
                )

            # Train collaborative filter on user's interaction history
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._custom_collaborative_filter.learn_from_interactions(
                    user_interactions, candidate_events
                ),
            )

            # Get liked events as seeds
            seed_events = [
                interaction.event_id
                for interaction in user_interactions_data
                if interaction.interaction_type in ["like", "going", "went"]
                or (interaction.rating and interaction.rating >= 4)
            ]

            if not seed_events:
                logger.info("No liked events found, using all interactions as seeds")
                seed_events = [
                    interaction.event_id for interaction in user_interactions_data
                ]

            # Get collaborative recommendations
            user_context = {
                "current_time": datetime.now(),
                "preferences": user_preferences or {},
                "location": {"lat": location_lat, "lon": location_lon},
            }

            collab_recommendations = await loop.run_in_executor(
                None,
                lambda: self._custom_collaborative_filter.get_collaborative_recommendations(
                    seed_events=seed_events,
                    candidate_events=candidate_events,
                    user_context=user_context,
                    num_recommendations=num_recommendations,
                ),
            )

            # Convert to RecommendedEvent format
            recommendations = []
            for i, (event, score, reasons) in enumerate(collab_recommendations):
                recommended_event = RecommendedEvent(
                    event=event,
                    recommendation_score=score,
                    rank=i + 1,
                    explanation=RecommendationExplanation(
                        overall_score=score,
                        components={"collaborative_score": score},
                        reasons=reasons,
                        model_confidence=0.8,
                    ),
                )
                recommendations.append(recommended_event)

            # Log learning stats
            stats = self._custom_collaborative_filter.get_learning_stats()
            logger.info(f"Custom collaborative filter stats: {stats}")

            return recommendations

        except Exception as e:
            logger.warning(f"Custom collaborative recommendations failed: {e}")
            return await self._popularity_based_recommendations(
                candidate_events, num_recommendations
            )

    async def _content_based_recommendations(
        self,
        candidate_events: List[EventResponse],
        user_preferences: Optional[Dict[str, Any]],
        location_lat: Optional[float],
        location_lon: Optional[float],
        num_recommendations: int,
    ) -> List[RecommendedEvent]:
        """
        Generate content-based recommendations using semantic similarity.

        Uses the ContentBasedRecommender to find events similar to user preferences.
        """
        logger.info("Using content-based recommendations")

        try:
            from ml.models.content_based import ContentBasedRecommender, UserPreferences

            loop = asyncio.get_event_loop()

            # Convert EventResponse objects to dicts for the model
            events_as_dicts = []
            event_id_map = {}

            for event in candidate_events:
                event_dict = {
                    "id": event.event_id,
                    "title": event.title,
                    "description": event.description or "",
                    "genres": event.genres or [],
                    "artists": event.artists or [],
                    "venue_name": event.venue_name or "",
                    "venue_lat": event.venue_lat,
                    "venue_lon": event.venue_lon,
                    "price_min": event.price_min,
                    "price_max": event.price_max,
                    "popularity_score": event.popularity_score or 0.0,
                    "date_time": event.date_time,
                    "topic": getattr(event, "topic", "music"),
                }
                events_as_dicts.append(event_dict)
                event_id_map[event.event_id] = event

            # Initialize and fit the content model if not already done
            if self._content_model is None:
                self._load_content_model()

            if self._content_model is None:
                logger.warning("Content model not available, falling back to popularity")
                return await self._popularity_based_recommendations(
                    candidate_events, num_recommendations
                )

            # Fit the model on candidate events
            await loop.run_in_executor(
                None, lambda: self._content_model.fit(events_as_dicts)
            )

            # Build UserPreferences from user_preferences dict
            prefs = user_preferences or {}
            user_prefs = UserPreferences(
                preferred_genres=prefs.get("preferred_genres", []),
                preferred_artists=prefs.get("preferred_artists", []),
                preferred_venues=prefs.get("preferred_venues", []),
                price_range=(
                    prefs.get("price_min", 0),
                    prefs.get("price_max", 1000),
                ),
                location_lat=location_lat,
                location_lon=location_lon,
                preferred_times=prefs.get("preferred_times", []),
                preferred_days=prefs.get("preferred_days", []),
            )

            # Get recommendations from content model
            scored_events = await loop.run_in_executor(
                None,
                lambda: self._content_model.recommend(
                    user_prefs,
                    candidate_event_ids=[e["id"] for e in events_as_dicts],
                    num_recommendations=num_recommendations,
                    diversity_factor=0.1,
                ),
            )

            # Convert to RecommendedEvent format
            recommendations = []
            for rank, (event_id, score) in enumerate(scored_events):
                if event_id in event_id_map:
                    event = event_id_map[event_id]

                    # Get explanation if available
                    explanation_data = {}
                    try:
                        explanation_data = await loop.run_in_executor(
                            None,
                            lambda eid=event_id: self._content_model.explain_recommendation(
                                eid, user_prefs
                            ),
                        )
                    except Exception:
                        pass

                    # Build explanation reasons
                    reasons = []
                    if explanation_data.get("genre_match", {}).get("matched_genres"):
                        matched = explanation_data["genre_match"]["matched_genres"]
                        reasons.append(f"Matches genres: {', '.join(matched[:3])}")
                    if explanation_data.get("location", {}).get("distance_km"):
                        dist = explanation_data["location"]["distance_km"]
                        reasons.append(f"{dist} km away")
                    if not reasons:
                        reasons.append("Similar to your preferences")

                    recommended_event = RecommendedEvent(
                        event=event,
                        recommendation_score=float(score),
                        rank=rank + 1,
                        explanation=RecommendationExplanation(
                            overall_score=float(score),
                            components={
                                "content_similarity": float(score),
                                "popularity": event.popularity_score or 0.0,
                            },
                            reasons=reasons,
                            model_confidence=0.75,
                        ),
                    )
                    recommendations.append(recommended_event)

            logger.info(f"Content-based recommender returned {len(recommendations)} events")
            return recommendations

        except Exception as e:
            logger.warning(f"Content-based recommendations failed: {e}")
            return await self._popularity_based_recommendations(
                candidate_events, num_recommendations
            )

    async def _hybrid_recommendations(
        self,
        candidate_events: List[EventResponse],
        user_id: str,
        user_preferences: Optional[Dict[str, Any]],
        location_lat: Optional[float],
        location_lon: Optional[float],
        diversity_factor: float,
        num_recommendations: int,
    ) -> List[RecommendedEvent]:
        """
        Generate hybrid recommendations combining content and collaborative signals.

        For now, uses content-based as primary with popularity boost.
        """
        logger.info("Using hybrid recommendations")

        try:
            # Get content-based recommendations
            content_recs = await self._content_based_recommendations(
                candidate_events,
                user_preferences,
                location_lat,
                location_lon,
                num_recommendations * 2,  # Get more for re-ranking
            )

            if not content_recs:
                return await self._popularity_based_recommendations(
                    candidate_events, num_recommendations
                )

            # Re-rank with hybrid scoring
            for rec in content_recs:
                content_score = rec.recommendation_score
                popularity_score = rec.event.popularity_score or 0.0

                # Hybrid formula: 70% content, 30% popularity
                hybrid_score = 0.7 * content_score + 0.3 * popularity_score

                # Apply diversity penalty for same topics (if multiple from same topic)
                rec.recommendation_score = hybrid_score
                rec.explanation.components["hybrid_score"] = hybrid_score

            # Sort by hybrid score and take top N
            content_recs.sort(key=lambda x: x.recommendation_score, reverse=True)
            recommendations = content_recs[:num_recommendations]

            # Update ranks
            for i, rec in enumerate(recommendations):
                rec.rank = i + 1

            return recommendations

        except Exception as e:
            logger.warning(f"Hybrid recommendations failed: {e}")
            return await self._popularity_based_recommendations(
                candidate_events, num_recommendations
            )

    async def get_similar_events(
        self, event_id: str, num_recommendations: int = 10
    ) -> List[RecommendedEvent]:
        """Get events similar to a specific event."""

        logger.info(f"Finding similar events to {event_id}")

        try:
            # Get the source event
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()
            source_event = await db.get_event(event_id)

            if not source_event:
                logger.warning(f"Source event {event_id} not found")
                return []

            # Get candidate events
            candidate_events = await self._get_candidate_events(
                db, filters=None, location_lat=None, location_lon=None
            )

            # Remove source event from candidates
            candidate_events = [e for e in candidate_events if e.event_id != event_id]

            await db.close()

            if self._langchain_recommender:
                # Use LangChain for semantic similarity
                loop = asyncio.get_event_loop()
                search_query = (
                    f"{source_event.title} {' '.join(source_event.genres[:2])}"
                )

                similar_results = await loop.run_in_executor(
                    None,
                    lambda: self._langchain_recommender.search_events(
                        search_query, limit=num_recommendations
                    ),
                )

                # Convert to RecommendedEvent format
                recommendations = []
                event_id_map = {event.event_id: event for event in candidate_events}

                for i, result in enumerate(similar_results):
                    if result["id"] in event_id_map:
                        event = event_id_map[result["id"]]

                        recommended_event = RecommendedEvent(
                            event=event,
                            recommendation_score=result["relevance"],
                            rank=i + 1,
                            explanation=RecommendationExplanation(
                                overall_score=result["relevance"],
                                components={"similarity": result["relevance"]},
                                reasons=[
                                    f"Similar to {source_event.title}",
                                    "Semantic similarity",
                                ],
                                model_confidence=0.8,
                            ),
                        )
                        recommendations.append(recommended_event)

                return recommendations
            else:
                # Fallback to genre/venue similarity
                return await self._simple_similarity_recommendations(
                    source_event, candidate_events, num_recommendations
                )

        except Exception as e:
            logger.error(f"Similar events failed: {e}")
            return []

    async def _simple_similarity_recommendations(
        self,
        source_event: EventResponse,
        candidate_events: List[EventResponse],
        num_recommendations: int,
    ) -> List[RecommendedEvent]:
        """Simple similarity based on genres and venue."""

        scored_events = []

        for event in candidate_events:
            score = 0.0

            # Genre similarity
            common_genres = set(source_event.genres) & set(event.genres)
            if common_genres:
                score += 0.5 * (
                    len(common_genres)
                    / max(len(source_event.genres), len(event.genres))
                )

            # Same venue
            if event.venue_name == source_event.venue_name:
                score += 0.3

            # Same neighborhood
            if event.neighborhood == source_event.neighborhood:
                score += 0.2

            if score > 0:
                recommended_event = RecommendedEvent(
                    event=event,
                    recommendation_score=score,
                    rank=0,
                    explanation=RecommendationExplanation(
                        overall_score=score,
                        components={"genre_similarity": score},
                        reasons=[
                            (
                                f"Similar genres: {', '.join(common_genres)}"
                                if common_genres
                                else "Similar venue/location"
                            )
                        ],
                        model_confidence=0.6,
                    ),
                )
                scored_events.append((score, recommended_event))

        # Sort and return top N
        scored_events.sort(key=lambda x: x[0], reverse=True)
        recommendations = []

        for rank, (score, rec_event) in enumerate(scored_events[:num_recommendations]):
            rec_event.rank = rank + 1
            recommendations.append(rec_event)

        return recommendations

    async def search_events(
        self,
        query: str,
        location_lat: Optional[float] = None,
        location_lon: Optional[float] = None,
        max_distance_km: float = 50.0,
        date_filter: Optional[str] = None,
        limit: int = 20,
    ) -> List[EventResponse]:
        """Search events using semantic search."""

        logger.info(f"Searching events for query: '{query}'")

        if self._langchain_recommender:
            try:
                loop = asyncio.get_event_loop()
                search_results = await loop.run_in_executor(
                    None,
                    lambda: self._langchain_recommender.search_events(
                        query, limit=limit
                    ),
                )

                # Convert search results to EventResponse
                from backend.app.services.database_service import DatabaseService

                db = DatabaseService()

                events = []
                for result in search_results:
                    event = await db.get_event(result["id"])
                    if event:
                        events.append(event)

                await db.close()
                return events

            except Exception as e:
                logger.warning(f"LangChain search failed: {e}")

        # Fallback to simple text matching
        return await self._simple_text_search(query, limit)

    async def _simple_text_search(self, query: str, limit: int) -> List[EventResponse]:
        """Simple text search fallback."""
        from backend.app.services.database_service import DatabaseService

        db = DatabaseService()
        all_events = await db.get_events(limit=100)
        await db.close()

        # Simple text matching
        query_lower = query.lower()
        matching_events = []

        for event in all_events:
            if (
                query_lower in event.title.lower()
                or (event.description and query_lower in event.description.lower())
                or any(query_lower in genre.lower() for genre in event.genres)
                or any(query_lower in artist.lower() for artist in event.artists)
            ):
                matching_events.append(event)

        return matching_events[:limit]

    async def update_with_interaction(
        self,
        user_id: str,
        event_id: str,
        interaction_type: str,
        rating: Optional[float] = None,
    ) -> None:
        """Update models with user interaction."""
        logger.info(
            f"Updating models with interaction: {user_id} -> {event_id} ({interaction_type})"
        )

        # For now, just log the interaction
        # In a full implementation, this would update the collaborative filtering model

        if self._collaborative_model and hasattr(
            self._collaborative_model, "update_with_interaction"
        ):
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._collaborative_model.update_with_interaction(
                        user_id, event_id, interaction_type, rating
                    ),
                )
            except Exception as e:
                logger.warning(f"Model update failed: {e}")

    async def retrain_models(self) -> None:
        """Retrain ML models with latest data."""
        logger.info("Starting model retraining...")

        try:
            # This would trigger a full model retraining process
            # For now, just update the timestamp
            self._last_training = datetime.now()
            logger.info("Model retraining completed (placeholder)")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            raise

    async def get_model_status(self) -> ModelStatusResponse:
        """Get model training status and metrics."""

        content_status = {
            "loaded": self._content_model is not None,
            "model_type": "content_based",
            "last_updated": (
                self._last_training.isoformat() if self._last_training else None
            ),
        }

        collaborative_status = {
            "loaded": self._collaborative_model is not None,
            "model_type": "collaborative_filtering",
            "last_updated": (
                self._last_training.isoformat() if self._last_training else None
            ),
        }

        hybrid_status = {
            "loaded": self._hybrid_model is not None,
            "model_type": "hybrid_ranker",
            "last_updated": (
                self._last_training.isoformat() if self._last_training else None
            ),
        }

        return ModelStatusResponse(
            content_model=content_status,
            collaborative_model=collaborative_status,
            hybrid_model=hybrid_status,
            last_training=self._last_training,
            is_training=False,
            training_progress=None,
        )

    def _empty_recommendation_response(
        self, session_id: str, start_time: datetime
    ) -> RecommendationResponse:
        """Create empty recommendation response."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return RecommendationResponse(
            user_id=None,
            session_id=session_id,
            events=[],
            total_candidates=0,
            model_version="1.0.0",
            timestamp=datetime.now(),
            processing_time_ms=int(processing_time),
            cold_start=True,
        )

    async def _fallback_recommendations(
        self, session_id: str, start_time: datetime, num_recommendations: int
    ) -> RecommendationResponse:
        """Generate fallback recommendations when main pipeline fails."""

        try:
            from backend.app.services.database_service import DatabaseService

            db = DatabaseService()

            # Get popular events as fallback
            events = await db.get_events(limit=num_recommendations)
            await db.close()

            recommendations = await self._popularity_based_recommendations(
                events, num_recommendations
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return RecommendationResponse(
                user_id=None,
                session_id=session_id,
                events=recommendations,
                total_candidates=len(events),
                model_version="1.0.0-fallback",
                timestamp=datetime.now(),
                processing_time_ms=int(processing_time),
                cold_start=True,
            )

        except Exception as e:
            logger.error(f"Fallback recommendations failed: {e}")
            return self._empty_recommendation_response(session_id, start_time)

    def _update_performance_metrics(self, processing_time_ms: float) -> None:
        """Update performance tracking metrics."""
        self._recommendation_count += 1

        # Update rolling average response time
        if self._recommendation_count == 1:
            self._avg_response_time = processing_time_ms
        else:
            alpha = 0.1  # Smoothing factor
            self._avg_response_time = (
                alpha * processing_time_ms + (1 - alpha) * self._avg_response_time
            )
