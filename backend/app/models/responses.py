#!/usr/bin/env python3
"""
Pydantic response models for the Copenhagen Event Recommender API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class EventResponse(BaseModel):
    """Response model for event data."""

    event_id: str
    title: str
    description: Optional[str] = None
    date_time: datetime
    end_time: Optional[datetime] = None
    venue_name: str
    venue_lat: float
    venue_lon: float
    venue_address: Optional[str] = None
    neighborhood: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    currency: str = Field(default="DKK")
    genres: List[str] = Field(default=[])
    artists: List[str] = Field(default=[])
    popularity_score: float = Field(default=0.0, ge=0, le=1)
    image_url: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None


class RecommendationExplanation(BaseModel):
    """Explanation for why an event was recommended."""

    overall_score: float = Field(..., ge=0, le=1)
    components: Dict[str, float] = Field(default={})
    reasons: List[str] = Field(default=[])
    model_confidence: float = Field(default=0.0, ge=0, le=1)


class RecommendedEvent(BaseModel):
    """Event with recommendation metadata."""

    event: EventResponse
    recommendation_score: float = Field(..., ge=0, le=1)
    rank: int = Field(..., ge=1)
    explanation: Optional[RecommendationExplanation] = None
    distance_km: Optional[float] = None
    predicted_attendance: Optional[int] = None


class RecommendationResponse(BaseModel):
    """Response for recommendation requests."""

    user_id: Optional[str] = None
    session_id: str
    events: List[RecommendedEvent]
    total_candidates: int
    model_version: str
    timestamp: datetime
    processing_time_ms: int
    filters_applied: Optional[Dict[str, Any]] = None
    cold_start: bool = Field(
        default=False, description="Whether this was a cold start recommendation"
    )


class UserResponse(BaseModel):
    """Response model for user data."""

    user_id: str
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    created_at: datetime
    last_active: Optional[datetime] = None
    interaction_count: int = Field(default=0)


class InteractionResponse(BaseModel):
    """Response model for interaction data."""

    id: int
    user_id: str
    event_id: str
    interaction_type: str
    rating: Optional[float] = None
    timestamp: datetime
    source: Optional[str] = None
    position: Optional[int] = None


class SearchResponse(BaseModel):
    """Response for search requests."""

    query: str
    events: List[EventResponse]
    total_results: int
    search_time_ms: int
    suggestions: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="healthy, degraded, or unhealthy")
    timestamp: datetime
    version: str
    uptime_seconds: Optional[int] = None
    services: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class AnalyticsResponse(BaseModel):
    """Analytics dashboard response."""

    period_days: int
    total_users: int
    active_users: int
    total_interactions: int
    total_recommendations: int

    # Event analytics
    total_events: int
    events_by_genre: Dict[str, int]
    events_by_neighborhood: Dict[str, int]
    popular_venues: List[Dict[str, Any]]

    # User behavior
    interaction_breakdown: Dict[str, int]
    avg_session_length: float
    recommendation_ctr: float

    # Model performance
    model_accuracy: Optional[float] = None
    avg_recommendation_score: float
    cold_start_percentage: float

    timestamp: datetime


class ModelStatusResponse(BaseModel):
    """Model training status response."""

    content_model: Dict[str, Any]
    collaborative_model: Dict[str, Any]
    hybrid_model: Dict[str, Any]
    last_training: Optional[datetime] = None
    next_training: Optional[datetime] = None
    is_training: bool = Field(default=False)
    training_progress: Optional[float] = None


class VenueResponse(BaseModel):
    """Response model for venue data."""

    venue_id: str
    name: str
    address: Optional[str] = None
    lat: float
    lon: float
    neighborhood: Optional[str] = None
    venue_type: Optional[str] = None
    capacity: Optional[int] = None
    website: Optional[str] = None
    upcoming_events_count: int = Field(default=0)


class ArtistResponse(BaseModel):
    """Response model for artist data."""

    artist_id: str
    name: str
    genres: List[str] = Field(default=[])
    popularity_score: float = Field(default=0.0, ge=0, le=1)
    spotify_id: Optional[str] = None
    upcoming_events_count: int = Field(default=0)


class TrendingResponse(BaseModel):
    """Response for trending content."""

    trending_events: List[EventResponse]
    trending_venues: List[VenueResponse]
    trending_artists: List[ArtistResponse]
    trending_genres: List[Dict[str, Any]]
    viral_hashtags: List[Dict[str, Any]]
    timestamp: datetime


class SimilarEventsResponse(BaseModel):
    """Response for similar events."""

    source_event_id: str
    similar_events: List[RecommendedEvent]
    similarity_method: str
    timestamp: datetime


class ColdStartResponse(BaseModel):
    """Response for cold start onboarding."""

    user_id: str
    step: int
    total_steps: int
    questions: List[Dict[str, Any]]
    progress_percentage: float
    recommendations_available: bool = Field(default=False)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
