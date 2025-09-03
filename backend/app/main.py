#!/usr/bin/env python3
"""
FastAPI backend for Copenhagen Event Recommender.
Provides recommendation endpoints and user interaction tracking.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import logging
from pathlib import Path
from typing import List, Optional, Dict
import os
from datetime import datetime

# Fix relative imports for direct execution
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.requests import (
    RecommendationRequest, UserPreferencesRequest, InteractionRequest,
    UserRegistrationRequest, EventSearchRequest
)
from models.responses import (
    RecommendationResponse, EventResponse, UserResponse, 
    HealthResponse, AnalyticsResponse
)
from services.recommendation_service import RecommendationService
from services.database_service import DatabaseService
from services.analytics_service import AnalyticsService
from core.config import Settings
from core.dependencies import get_current_user, get_db_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Copenhagen Event Recommender API",
    description="Hybrid ML-powered event recommendation system for Copenhagen nightlife",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize services
recommendation_service = RecommendationService()
analytics_service = AnalyticsService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Copenhagen Event Recommender API...")
    
    try:
        # Initialize recommendation models
        await recommendation_service.initialize()
        logger.info("Recommendation service initialized")
        
        # Initialize analytics
        await analytics_service.initialize()
        logger.info("Analytics service initialized")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Copenhagen Event Recommender API...")
    
    try:
        await recommendation_service.cleanup()
        await analytics_service.cleanup()
        logger.info("API shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    try:
        # Check database connection
        db_service = DatabaseService()
        db_healthy = await db_service.health_check()
        
        # Check recommendation service
        rec_healthy = await recommendation_service.health_check()
        
        status = "healthy" if (db_healthy and rec_healthy) else "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            services={
                "database": "healthy" if db_healthy else "unhealthy",
                "recommendations": "healthy" if rec_healthy else "unhealthy"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            error=str(e)
        )

# User endpoints
@app.post("/users/register", response_model=UserResponse)
async def register_user(
    request: UserRegistrationRequest,
    db: DatabaseService = Depends(get_db_service),
    background_tasks: BackgroundTasks = None
):
    """Register a new user with preferences."""
    
    try:
        user = await db.create_user(
            user_id=request.user_id,
            name=request.name,
            preferences=request.preferences.dict() if request.preferences else None,
            location_lat=request.location_lat,
            location_lon=request.location_lon
        )
        
        # Track registration in background
        if background_tasks:
            background_tasks.add_task(
                analytics_service.track_user_registration,
                user_id=request.user_id
            )
        
        return UserResponse(
            user_id=user.user_id,
            name=user.name,
            preferences=user.preferences,
            created_at=user.created_at
        )
        
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: DatabaseService = Depends(get_db_service)
):
    """Get user information."""
    
    try:
        user = await db.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            user_id=user.user_id,
            name=user.name,
            preferences=user.preferences,
            created_at=user.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/{user_id}/preferences")
async def update_user_preferences(
    user_id: str,
    request: UserPreferencesRequest,
    db: DatabaseService = Depends(get_db_service),
    background_tasks: BackgroundTasks = None
):
    """Update user preferences."""
    
    try:
        await db.update_user_preferences(
            user_id=user_id,
            preferences=request.dict(),
            location_lat=request.location_lat,
            location_lon=request.location_lon
        )
        
        # Track preference update
        if background_tasks:
            background_tasks.add_task(
                analytics_service.track_preference_update,
                user_id=user_id
            )
        
        return {"status": "success", "message": "Preferences updated"}
        
    except Exception as e:
        logger.error(f"Update preferences failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Event endpoints
@app.get("/events", response_model=List[EventResponse])
async def get_events(
    limit: int = 50,
    offset: int = 0,
    neighborhood: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    genres: Optional[str] = None,  # Comma-separated
    db: DatabaseService = Depends(get_db_service)
):
    """Get events with optional filtering."""
    
    try:
        # Parse genres
        genre_list = genres.split(',') if genres else None
        
        events = await db.get_events(
            limit=limit,
            offset=offset,
            neighborhood=neighborhood,
            date_from=date_from,
            date_to=date_to,
            min_price=min_price,
            max_price=max_price,
            genres=genre_list
        )
        
        # Events are already EventResponse objects from DatabaseService
        return events
        
    except Exception as e:
        logger.error(f"Get events failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: str,
    db: DatabaseService = Depends(get_db_service)
):
    """Get specific event details."""
    
    try:
        event = await db.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Event is already an EventResponse object
        return event
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get event failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get personalized event recommendations."""
    
    try:
        # Get recommendations from service
        recommendations = await recommendation_service.get_recommendations(
            user_id=request.user_id or current_user,
            user_preferences=request.user_preferences,
            location_lat=request.location_lat,
            location_lon=request.location_lon,
            filters=request.filters,
            num_recommendations=request.num_recommendations,
            use_collaborative=request.use_collaborative,
            diversity_factor=request.diversity_factor
        )
        
        # Track recommendation request in background
        background_tasks.add_task(
            analytics_service.track_recommendation_request,
            user_id=request.user_id or current_user,
            num_returned=len(recommendations.events),
            filters=request.filters.dict() if request.filters else None
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/similar")
async def get_similar_events(
    event_id: str,
    num_recommendations: int = 10,
    background_tasks: BackgroundTasks = None
):
    """Get events similar to a specific event."""
    
    try:
        similar_events = await recommendation_service.get_similar_events(
            event_id=event_id,
            num_recommendations=num_recommendations
        )
        
        if background_tasks:
            background_tasks.add_task(
                analytics_service.track_similar_events_request,
                event_id=event_id,
                num_returned=len(similar_events)
            )
        
        return {
            "event_id": event_id,
            "similar_events": similar_events,
            "count": len(similar_events)
        }
        
    except Exception as e:
        logger.error(f"Similar events failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User interaction endpoints
@app.post("/interactions")
async def record_interaction(
    request: InteractionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = Depends(get_current_user),
    db: DatabaseService = Depends(get_db_service)
):
    """Record user interaction with an event."""
    
    try:
        # Record interaction
        await db.record_interaction(
            user_id=request.user_id or current_user,
            event_id=request.event_id,
            interaction_type=request.interaction_type,
            rating=request.rating,
            source=request.source,
            position=request.position
        )
        
        # Update recommendation models in background
        background_tasks.add_task(
            recommendation_service.update_with_interaction,
            user_id=request.user_id or current_user,
            event_id=request.event_id,
            interaction_type=request.interaction_type,
            rating=request.rating
        )
        
        # Track analytics
        background_tasks.add_task(
            analytics_service.track_interaction,
            user_id=request.user_id or current_user,
            event_id=request.event_id,
            interaction_type=request.interaction_type
        )
        
        return {"status": "success", "message": "Interaction recorded"}
        
    except Exception as e:
        logger.error(f"Record interaction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/interactions")
async def get_user_interactions(
    user_id: str,
    limit: int = 50,
    interaction_type: Optional[str] = None,
    db: DatabaseService = Depends(get_db_service)
):
    """Get user's interaction history."""
    
    try:
        interactions = await db.get_user_interactions(
            user_id=user_id,
            limit=limit,
            interaction_type=interaction_type
        )
        
        return {
            "user_id": user_id,
            "interactions": interactions,
            "count": len(interactions)
        }
        
    except Exception as e:
        logger.error(f"Get interactions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search endpoint
@app.post("/search")
async def search_events(
    request: EventSearchRequest,
    background_tasks: BackgroundTasks = None,
    db: DatabaseService = Depends(get_db_service)
):
    """Search events by text query."""
    
    try:
        events = await recommendation_service.search_events(
            query=request.query,
            location_lat=request.location_lat,
            location_lon=request.location_lon,
            max_distance_km=request.max_distance_km,
            date_filter=request.date_filter,
            limit=request.limit
        )
        
        # Track search in background
        if background_tasks:
            background_tasks.add_task(
                analytics_service.track_search,
                query=request.query,
                num_results=len(events)
            )
        
        return {
            "query": request.query,
            "events": events,
            "count": len(events)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/dashboard", response_model=AnalyticsResponse)
async def get_analytics_dashboard(
    days: int = 7,
    admin_user: str = Depends(get_current_user)  # Require authentication
):
    """Get analytics dashboard data."""
    
    try:
        analytics = await analytics_service.get_dashboard_data(days=days)
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/events/{event_id}")
async def get_event_analytics(
    event_id: str,
    days: int = 30
):
    """Get analytics for a specific event."""
    
    try:
        analytics = await analytics_service.get_event_analytics(
            event_id=event_id,
            days=days
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Event analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints
@app.post("/admin/retrain-models")
async def retrain_models(
    background_tasks: BackgroundTasks,
    admin_user: str = Depends(get_current_user)
):
    """Trigger model retraining (admin only)."""
    
    try:
        # Trigger retraining in background
        background_tasks.add_task(recommendation_service.retrain_models)
        
        return {"status": "success", "message": "Model retraining started"}
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/model-status")
async def get_model_status(
    admin_user: str = Depends(get_current_user)
):
    """Get model training status and metrics."""
    
    try:
        status = await recommendation_service.get_model_status()
        return status
        
    except Exception as e:
        logger.error(f"Get model status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )