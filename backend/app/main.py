#!/usr/bin/env python3
"""
Copenhagen Event Recommender API - MVP Version
Works with existing database schema and real event data.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import duckdb
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import logging
import os
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Copenhagen Event Recommender API - MVP",
    description="AI-powered event recommendations for Copenhagen nightlife",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
# In production, be more restrictive
if os.getenv("ENVIRONMENT") == "production":
    cors_origins = [origin.strip() for origin in cors_origins if origin.strip() != "*"]
    allow_all = False
else:
    allow_all = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins + (["*"] if allow_all else []),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection - use absolute path for MVP
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = os.getenv("DATABASE_URL", os.path.join(project_root, "data", "events.duckdb"))

class Event(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    date_time: datetime
    end_date_time: Optional[datetime] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    currency: str = "DKK"
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_neighborhood: Optional[str] = None
    genres: Optional[List[str]] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    image_url: Optional[str] = None
    popularity_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_status: str
    events_count: int
    upcoming_events: int

def get_db_connection():
    """Get database connection."""
    try:
        if not Path(DATABASE_URL).exists():
            logger.warning(f"Database file not found at {DATABASE_URL}")
            return None
        return duckdb.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Copenhagen Event Recommender API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "events": "/events",
            "event_detail": "/events/{event_id}",
            "search": "/search",
            "recommendations": "/recommend/{user_id}",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    conn = get_db_connection()
    if conn is None:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database_status="disconnected",
            events_count=0,
            upcoming_events=0
        )
    
    try:
        # Count total events
        total_result = conn.execute("SELECT COUNT(*) FROM events WHERE status = 'active'").fetchone()
        events_count = total_result[0] if total_result else 0
        
        # Count upcoming events
        upcoming_result = conn.execute(
            "SELECT COUNT(*) FROM events WHERE status = 'active' AND date_time > ?", 
            [datetime.now()]
        ).fetchone()
        upcoming_events = upcoming_result[0] if upcoming_result else 0
        
        conn.close()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0", 
            database_status="connected",
            events_count=events_count,
            upcoming_events=upcoming_events
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database_status="error",
            events_count=0,
            upcoming_events=0
        )

@app.get("/events", response_model=List[Event])
async def get_events(
    limit: int = Query(default=20, le=100, description="Maximum number of events to return"),
    offset: int = Query(default=0, ge=0, description="Number of events to skip"),
    upcoming_only: bool = Query(default=True, description="Only return upcoming events"),
    min_price: Optional[float] = Query(default=None, description="Minimum price filter"),
    max_price: Optional[float] = Query(default=None, description="Maximum price filter")
):
    """Get events from the database."""
    
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        # Build query conditions
        where_conditions = ["status = 'active'"]
        params = []
        
        if upcoming_only:
            where_conditions.append("date_time > ?")
            params.append(datetime.now())
        
        if min_price is not None:
            where_conditions.append("price_min >= ?")
            params.append(min_price)
            
        if max_price is not None:
            where_conditions.append("price_max <= ?")
            params.append(max_price)
        
        where_clause = " AND ".join(where_conditions)
        
        # Main query with venue join
        query = f"""
        SELECT 
            e.id,
            e.title,
            e.description,
            e.date_time,
            e.end_date_time,
            e.price_min,
            e.price_max,
            e.currency,
            e.source,
            e.source_url,
            e.image_url,
            e.popularity_score,
            v.name as venue_name,
            v.address as venue_address,
            v.neighborhood as venue_neighborhood
        FROM events e
        LEFT JOIN venues v ON e.venue_id = v.id
        WHERE {where_clause}
        ORDER BY e.date_time ASC
        LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        
        df = conn.execute(query, params).df()
        conn.close()
        
        # Convert to Event models
        events = []
        for _, row in df.iterrows():
            try:
                event = Event(
                    id=str(row['id']),
                    title=str(row['title']),
                    description=str(row['description']) if pd.notna(row['description']) else None,
                    date_time=pd.to_datetime(row['date_time']),
                    end_date_time=pd.to_datetime(row['end_date_time']) if pd.notna(row['end_date_time']) else None,
                    price_min=float(row['price_min']) if pd.notna(row['price_min']) else None,
                    price_max=float(row['price_max']) if pd.notna(row['price_max']) else None,
                    currency=str(row['currency']) if pd.notna(row['currency']) else "DKK",
                    venue_name=str(row['venue_name']) if pd.notna(row['venue_name']) else None,
                    venue_address=str(row['venue_address']) if pd.notna(row['venue_address']) else None,
                    venue_neighborhood=str(row['venue_neighborhood']) if pd.notna(row['venue_neighborhood']) else None,
                    source=str(row['source']) if pd.notna(row['source']) else None,
                    source_url=str(row['source_url']) if pd.notna(row['source_url']) else None,
                    image_url=str(row['image_url']) if pd.notna(row['image_url']) else None,
                    popularity_score=float(row['popularity_score']) if pd.notna(row['popularity_score']) else None,
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse event row: {e}")
                continue
        
        return events
        
    except Exception as e:
        logger.error(f"Get events failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events/{event_id}", response_model=Event)
async def get_event(event_id: str):
    """Get a specific event by ID."""
    
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        query = """
        SELECT 
            e.id,
            e.title,
            e.description,
            e.date_time,
            e.end_date_time,
            e.price_min,
            e.price_max,
            e.currency,
            e.source,
            e.source_url,
            e.image_url,
            e.popularity_score,
            v.name as venue_name,
            v.address as venue_address,
            v.neighborhood as venue_neighborhood
        FROM events e
        LEFT JOIN venues v ON e.venue_id = v.id
        WHERE e.id = ? AND e.status = 'active'
        """
        
        result = conn.execute(query, [event_id]).fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Convert to Event model
        return Event(
            id=str(result[0]),
            title=str(result[1]),
            description=str(result[2]) if result[2] else None,
            date_time=pd.to_datetime(result[3]),
            end_date_time=pd.to_datetime(result[4]) if result[4] else None,
            price_min=float(result[5]) if result[5] else None,
            price_max=float(result[6]) if result[6] else None,
            currency=str(result[7]) if result[7] else "DKK",
            source=str(result[8]) if result[8] else None,
            source_url=str(result[9]) if result[9] else None,
            image_url=str(result[10]) if result[10] else None,
            popularity_score=float(result[11]) if result[11] else None,
            venue_name=str(result[12]) if result[12] else None,
            venue_address=str(result[13]) if result[13] else None,
            venue_neighborhood=str(result[14]) if result[14] else None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get event failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{user_id}")
async def get_recommendations(user_id: str, limit: int = Query(default=10, le=20)):
    """Get personalized recommendations (MVP: popularity-based)."""
    
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        # For MVP, recommend based on popularity and upcoming events
        query = """
        SELECT 
            e.id,
            e.title,
            e.description,
            e.date_time,
            e.end_date_time,
            e.price_min,
            e.price_max,
            e.currency,
            e.source,
            e.source_url,
            e.image_url,
            e.popularity_score,
            v.name as venue_name,
            v.address as venue_address,
            v.neighborhood as venue_neighborhood
        FROM events e
        LEFT JOIN venues v ON e.venue_id = v.id
        WHERE e.status = 'active' AND e.date_time > ?
        ORDER BY e.popularity_score DESC, e.date_time ASC
        LIMIT ?
        """
        
        df = conn.execute(query, [datetime.now(), limit]).df()
        conn.close()
        
        recommendations = []
        for _, row in df.iterrows():
            try:
                event = Event(
                    id=str(row['id']),
                    title=str(row['title']),
                    description=str(row['description']) if pd.notna(row['description']) else None,
                    date_time=pd.to_datetime(row['date_time']),
                    end_date_time=pd.to_datetime(row['end_date_time']) if pd.notna(row['end_date_time']) else None,
                    price_min=float(row['price_min']) if pd.notna(row['price_min']) else None,
                    price_max=float(row['price_max']) if pd.notna(row['price_max']) else None,
                    currency=str(row['currency']) if pd.notna(row['currency']) else "DKK",
                    venue_name=str(row['venue_name']) if pd.notna(row['venue_name']) else None,
                    venue_address=str(row['venue_address']) if pd.notna(row['venue_address']) else None,
                    venue_neighborhood=str(row['venue_neighborhood']) if pd.notna(row['venue_neighborhood']) else None,
                    source=str(row['source']) if pd.notna(row['source']) else None,
                    source_url=str(row['source_url']) if pd.notna(row['source_url']) else None,
                    image_url=str(row['image_url']) if pd.notna(row['image_url']) else None,
                    popularity_score=float(row['popularity_score']) if pd.notna(row['popularity_score']) else None,
                )
                recommendations.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse recommendation row: {e}")
                continue
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "total": len(recommendations),
            "method": "popularity_based",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_events(
    query: str = Query(..., description="Search query"),
    limit: int = Query(default=20, le=50)
):
    """Search events by text query."""
    
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        search_query = f"%{query.lower()}%"
        
        sql = """
        SELECT 
            e.id,
            e.title,
            e.description,
            e.date_time,
            e.end_date_time,
            e.price_min,
            e.price_max,
            e.currency,
            e.source,
            e.source_url,
            e.image_url,
            e.popularity_score,
            v.name as venue_name,
            v.address as venue_address,
            v.neighborhood as venue_neighborhood
        FROM events e
        LEFT JOIN venues v ON e.venue_id = v.id
        WHERE e.status = 'active' 
        AND e.date_time > ?
        AND (LOWER(e.title) LIKE ? OR LOWER(e.description) LIKE ? OR LOWER(v.name) LIKE ?)
        ORDER BY e.popularity_score DESC, e.date_time ASC
        LIMIT ?
        """
        
        df = conn.execute(sql, [datetime.now(), search_query, search_query, search_query, limit]).df()
        conn.close()
        
        results = []
        for _, row in df.iterrows():
            try:
                event = Event(
                    id=str(row['id']),
                    title=str(row['title']),
                    description=str(row['description']) if pd.notna(row['description']) else None,
                    date_time=pd.to_datetime(row['date_time']),
                    end_date_time=pd.to_datetime(row['end_date_time']) if pd.notna(row['end_date_time']) else None,
                    price_min=float(row['price_min']) if pd.notna(row['price_min']) else None,
                    price_max=float(row['price_max']) if pd.notna(row['price_max']) else None,
                    currency=str(row['currency']) if pd.notna(row['currency']) else "DKK",
                    venue_name=str(row['venue_name']) if pd.notna(row['venue_name']) else None,
                    venue_address=str(row['venue_address']) if pd.notna(row['venue_address']) else None,
                    venue_neighborhood=str(row['venue_neighborhood']) if pd.notna(row['venue_neighborhood']) else None,
                    source=str(row['source']) if pd.notna(row['source']) else None,
                    source_url=str(row['source_url']) if pd.notna(row['source_url']) else None,
                    image_url=str(row['image_url']) if pd.notna(row['image_url']) else None,
                    popularity_score=float(row['popularity_score']) if pd.notna(row['popularity_score']) else None,
                )
                results.append(event)
            except Exception as e:
                logger.warning(f"Failed to parse search result row: {e}")
                continue
        
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        stats = {}
        
        # Total active events
        result = conn.execute("SELECT COUNT(*) FROM events WHERE status = 'active'").fetchone()
        stats["total_events"] = result[0] if result else 0
        
        # Upcoming events  
        result = conn.execute(
            "SELECT COUNT(*) FROM events WHERE status = 'active' AND date_time > ?", 
            [datetime.now()]
        ).fetchone()
        stats["upcoming_events"] = result[0] if result else 0
        
        # Events by venue
        try:
            df = conn.execute("""
                SELECT v.name, COUNT(*) as count 
                FROM events e 
                JOIN venues v ON e.venue_id = v.id 
                WHERE e.status = 'active' AND e.date_time > ?
                GROUP BY v.name 
                ORDER BY count DESC 
                LIMIT 10
            """, [datetime.now()]).df()
            stats["by_venue"] = df.to_dict("records")
        except:
            stats["by_venue"] = []
        
        # Price range distribution
        try:
            df = conn.execute("""
                SELECT 
                    CASE 
                        WHEN price_min < 100 THEN 'Under 100 DKK'
                        WHEN price_min < 200 THEN '100-200 DKK'
                        WHEN price_min < 300 THEN '200-300 DKK'
                        ELSE 'Over 300 DKK'
                    END as price_range,
                    COUNT(*) as count
                FROM events 
                WHERE status = 'active' AND date_time > ? AND price_min IS NOT NULL
                GROUP BY price_range
            """, [datetime.now()]).df()
            stats["by_price_range"] = df.to_dict("records")
        except:
            stats["by_price_range"] = []
        
        conn.close()
        return stats
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Railway sets PORT, fallback to API_PORT, then default
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting Copenhagen Event Recommender MVP API")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Database: {DATABASE_URL}")
    logger.info(f"CORS Origins: {cors_origins}")
    
    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )