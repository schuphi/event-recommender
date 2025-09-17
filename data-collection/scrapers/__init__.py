"""
Data collection package for Copenhagen Event Recommender.
"""

from .official_apis.eventbrite import EventbriteScraper, EventbriteEvent
from .venue_scrapers.copenhagen_venues import CopenhagenVenueScraper, VenueEvent

__all__ = [
    "EventbriteScraper",
    "EventbriteEvent",
    "CopenhagenVenueScraper",
    "VenueEvent",
]
