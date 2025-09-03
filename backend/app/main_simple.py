#!/usr/bin/env python3
"""
Minimal FastAPI backend for Copenhagen Event Recommender.
For Lovable integration testing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="Copenhagen Event Recommender API",
    description="AI-powered event recommendations for Copenhagen nightlife",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "copenhagen-event-recommender",
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Copenhagen Event Recommender API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/events")
async def get_events():
    """Get sample events for frontend testing"""
    return {
        "events": [
            {
                "id": "event_1",
                "title": "Techno Night at Culture Box",
                "description": "Underground techno experience in Copenhagen's premier electronic venue",
                "date_time": "2024-01-20T22:00:00",
                "venue": {
                    "name": "Culture Box",
                    "address": "Kronprinsensgade 54A, 1306 København",
                    "neighborhood": "Indre By",
                },
                "price_min": 150,
                "price_max": 200,
                "genres": ["techno", "electronic"],
                "source": "manual",
            },
            {
                "id": "event_2",
                "title": "Jazz Evening at Jazzhus Montmartre",
                "description": "Intimate jazz performance in historic venue",
                "date_time": "2024-01-21T20:00:00",
                "venue": {
                    "name": "Jazzhus Montmartre",
                    "address": "Store Regnegade 19A, 1110 København",
                    "neighborhood": "Indre By",
                },
                "price_min": 250,
                "price_max": 350,
                "genres": ["jazz"],
                "source": "manual",
            },
        ],
        "total": 2,
    }


@app.post("/recommend")
async def get_recommendations(request: Dict = None):
    """Get personalized event recommendations using LangChain"""
    try:
        # Import here to avoid startup issues if LangChain isn't configured
        from .services.langchain_recommender import recommender

        # Extract preferences from request
        user_preferences = request.get("user_preferences", {}) if request else {}
        location_lat = request.get("location_lat") if request else None
        location_lon = request.get("location_lon") if request else None
        num_recommendations = request.get("num_recommendations", 10) if request else 10

        # Get LangChain recommendations
        langchain_recs = recommender.get_recommendations(
            user_preferences=user_preferences,
            location_lat=location_lat,
            location_lon=location_lon,
            num_recommendations=num_recommendations,
        )

        # Convert to API format
        recommendations = []
        for rec in langchain_recs:
            recommendations.append(
                {
                    "event_id": rec.event_id,
                    "score": rec.score,
                    "reasons": rec.reasons,
                    "explanation": rec.explanation,
                }
            )

        return {
            "recommendations": recommendations,
            "total": len(recommendations),
            "request_id": f"req_{datetime.now().timestamp()}",
            "powered_by": "LangChain + Semantic Search",
        }

    except Exception as e:
        # Fallback to simple recommendations if LangChain fails
        return {
            "recommendations": [
                {
                    "event_id": "event_1",
                    "score": 0.92,
                    "reasons": [
                        "Matches your techno preferences",
                        "Popular venue",
                        "Good time slot",
                    ],
                    "explanation": "Recommended because you like electronic music and Culture Box is highly rated",
                },
                {
                    "event_id": "event_2",
                    "score": 0.78,
                    "reasons": ["Diverse music taste", "Quality venue"],
                    "explanation": "Expanding your musical horizons with quality jazz",
                },
            ],
            "total": 2,
            "request_id": f"req_{datetime.now().timestamp()}",
            "fallback": True,
            "error": str(e),
        }


@app.post("/interactions")
async def record_interaction(interaction: Dict):
    """Record user interaction with events"""
    return {
        "success": True,
        "interaction_id": f"int_{datetime.now().timestamp()}",
        "recorded_at": datetime.now().isoformat(),
    }


@app.post("/search")
async def search_events(query: Dict):
    """Search events by query using LangChain semantic search"""
    try:
        from .services.langchain_recommender import recommender

        search_query = query.get("query", "") if query else ""
        if not search_query:
            return {"results": [], "query": "", "total": 0}

        # Use LangChain semantic search
        results = recommender.search_events(search_query, limit=10)

        return {
            "results": results,
            "query": search_query,
            "total": len(results),
            "powered_by": "LangChain Semantic Search",
        }

    except Exception as e:
        # Fallback to simple search
        return {
            "results": [
                {
                    "id": "event_1",
                    "title": "Techno Night at Culture Box",
                    "relevance": 0.95,
                    "match_type": "title",
                }
            ],
            "query": query.get("query", ""),
            "total": 1,
            "fallback": True,
            "error": str(e),
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
