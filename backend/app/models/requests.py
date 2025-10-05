#!/usr/bin/env python3
"""
Pydantic request models for the Copenhagen Event Recommender API.
"""

from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class InteractionType(str, Enum):
    """Valid interaction types."""

    LIKE = "like"
    DISLIKE = "dislike"
    GOING = "going"
    WENT = "went"
    SAVE = "save"


class DateFilter(str, Enum):
    """Date filter options."""

    TODAY = "today"
    TOMORROW = "tomorrow"
    THIS_WEEK = "this_week"
    THIS_WEEKEND = "this_weekend"
    NEXT_WEEK = "next_week"
    THIS_MONTH = "this_month"


class UserPreferencesRequest(BaseModel):
    """User preferences for recommendations."""

    preferred_genres: List[str] = Field(
        default=[], description="Preferred music genres"
    )
    preferred_artists: List[str] = Field(default=[], description="Preferred artists")
    preferred_venues: List[str] = Field(default=[], description="Preferred venues")
    price_range: tuple[float, float] = Field(
        default=(0, 1000), description="Price range (min, max) in DKK"
    )
    preferred_times: List[int] = Field(default=[], description="Preferred hours (0-23)")
    preferred_days: List[int] = Field(
        default=[], description="Preferred days of week (0-6)"
    )
    max_distance_km: float = Field(
        default=20.0, description="Maximum distance in kilometers"
    )
    location_lat: Optional[float] = Field(None, description="User latitude")
    location_lon: Optional[float] = Field(None, description="User longitude")

    @validator("preferred_times")
    def validate_times(cls, v):
        if v and any(hour < 0 or hour > 23 for hour in v):
            raise ValueError("Hours must be between 0 and 23")
        return v

    @validator("preferred_days")
    def validate_days(cls, v):
        if v and any(day < 0 or day > 6 for day in v):
            raise ValueError("Days must be between 0 (Monday) and 6 (Sunday)")
        return v

    @validator("price_range")
    def validate_price_range(cls, v):
        if v[0] < 0 or v[1] < v[0]:
            raise ValueError("Invalid price range")
        return v


class RecommendationFilters(BaseModel):
    """Filters for recommendations."""

    date_filter: Optional[DateFilter] = None
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)
    genres: Optional[List[str]] = None
    venues: Optional[List[str]] = None
    neighborhoods: Optional[List[str]] = None
    min_popularity: Optional[float] = Field(None, ge=0, le=1)
    include_past_events: bool = Field(default=False)

    @validator("max_price")
    def validate_max_price(cls, v, values):
        if v is not None and "min_price" in values and values["min_price"] is not None:
            if v < values["min_price"]:
                raise ValueError("max_price must be >= min_price")
        return v


class RecommendationRequest(BaseModel):
    """Request for event recommendations."""

    user_id: Optional[str] = Field(None, description="User ID (if authenticated)")
    user_preferences: UserPreferencesRequest
    location_lat: Optional[float] = Field(None, description="Current latitude")
    location_lon: Optional[float] = Field(None, description="Current longitude")
    num_recommendations: int = Field(
        default=10, ge=1, le=50, description="Number of recommendations"
    )
    filters: Optional[RecommendationFilters] = None
    use_collaborative: bool = Field(
        default=True, description="Use collaborative filtering"
    )
    diversity_factor: float = Field(
        default=0.1, ge=0, le=1, description="Diversity promotion factor"
    )
    explain: bool = Field(
        default=False, description="Include explanations for recommendations"
    )


class UserRegistrationRequest(BaseModel):
    """Request to register a new user."""

    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    name: Optional[str] = Field(None, max_length=100, description="User display name")
    preferences: Optional[UserPreferencesRequest] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None


class InteractionRequest(BaseModel):
    """Request to record user interaction."""

    user_id: Optional[str] = Field(None, description="User ID (if not authenticated)")
    event_id: str = Field(..., description="Event ID")
    interaction_type: InteractionType
    rating: Optional[float] = Field(
        None, ge=1, le=5, description="Rating for 'went' interactions"
    )
    source: Optional[str] = Field(
        None, description="Source of interaction (feed, search, etc.)"
    )
    position: Optional[int] = Field(None, description="Position in recommendation list")

    @validator("rating")
    def validate_rating(cls, v, values):
        if (
            "interaction_type" in values
            and values["interaction_type"] == InteractionType.WENT
        ):
            if v is None:
                raise ValueError('Rating required for "went" interactions')
        return v


class EventSearchRequest(BaseModel):
    """Request to search events."""

    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    max_distance_km: float = Field(
        default=50.0, ge=0, description="Maximum distance from location"
    )
    date_filter: Optional[DateFilter] = None
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")


class BulkInteractionRequest(BaseModel):
    """Request to record multiple interactions."""

    interactions: List[InteractionRequest] = Field(..., max_items=100)


class FeedbackRequest(BaseModel):
    """Request to provide feedback on recommendations."""

    recommendation_id: str = Field(..., description="Recommendation session ID")
    user_id: Optional[str] = None
    feedback_type: str = Field(..., description="Type of feedback")
    feedback_data: Dict[str, Any] = Field(
        default={}, description="Additional feedback data"
    )
    comments: Optional[str] = Field(None, max_length=500, description="User comments")


class ColdStartRequest(BaseModel):
    """Request for cold start preference collection."""

    user_id: str = Field(..., description="New user ID")
    step: int = Field(..., ge=1, le=5, description="Onboarding step (1-5)")
    responses: Dict[str, Any] = Field(..., description="User responses for this step")
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None


class ModelFeedbackRequest(BaseModel):
    """Request to provide feedback on model performance."""

    user_id: str
    session_id: str
    model_version: str
    feedback_score: float = Field(
        ..., ge=0, le=1, description="Overall satisfaction (0-1)"
    )
    recommendation_quality: float = Field(..., ge=0, le=1)
    diversity_satisfaction: float = Field(..., ge=0, le=1)
    relevance_score: float = Field(..., ge=0, le=1)
    comments: Optional[str] = Field(None, max_length=1000)


# Authentication Models
class UserRegisterRequest(BaseModel):
    """Request to register a new user."""
    
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    name: str = Field(..., min_length=1, max_length=100, description="User's display name")


class UserLoginRequest(BaseModel):
    """Request to login a user."""
    
    email: EmailStr
    password: str
