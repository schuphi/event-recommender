#!/usr/bin/env python3
"""
Sync local events to Railway database via API.
"""

import requests
import duckdb
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAILWAY_API_BASE = "https://event-recommender-production.up.railway.app"

def sync_events_to_railway():
    """Sync all local events to Railway"""
    
    # Read from local database
    conn = duckdb.connect("data/events.duckdb")
    
    # Get all events with venue information
    events = conn.execute("""
        SELECT 
            e.id, e.title, e.description, e.date_time, e.end_date_time,
            e.price_min, e.price_max, e.currency, e.source, e.source_url,
            v.name as venue_name, v.address as venue_address, 
            v.neighborhood as venue_neighborhood, v.lat, v.lon
        FROM events e 
        JOIN venues v ON e.venue_id = v.id
        WHERE e.date_time > CURRENT_TIMESTAMP
        ORDER BY e.date_time
    """).fetchall()
    
    logger.info(f"Found {len(events)} upcoming events to sync")
    
    synced = 0
    
    for event in events:
        event_data = {
            "title": event[1],
            "description": event[2] or "",
            "date_time": str(event[3]),
            "venue_name": event[10],
            "venue_address": event[11],
            "venue_neighborhood": event[12],
            "price_min": float(event[5]) if event[5] else 0,
            "price_max": float(event[6]) if event[6] else 0,
            "source": event[8] or "local_sync"
        }
        
        try:
            # Try to create event via Railway admin API
            response = requests.post(
                f"{RAILWAY_API_BASE}/admin/events",
                json=event_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Synced: {event[1]}")
                synced += 1
            else:
                logger.warning(f"⚠️  Failed to sync {event[1]}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error syncing {event[1]}: {e}")
            continue
    
    conn.close()
    logger.info(f"🎉 Successfully synced {synced}/{len(events)} events to Railway")

if __name__ == "__main__":
    sync_events_to_railway()