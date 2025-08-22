"""
Data collection package for Copenhagen Event Recommender.
"""

from .official_apis.eventbrite import EventbriteScraper, EventbriteEvent
from .official_apis.meetup import MeetupScraper, MeetupEvent
from .social_scrapers.instagram import InstagramEventScraper, InstagramEvent  
from .social_scrapers.tiktok import TikTokEventScraper, TikTokEvent
from .social_scrapers.instagram_viral import InstagramViralEventScraper
from .social_scrapers.viral_discovery import ViralEventDiscoveryEngine, TrendingEvent

__all__ = [
    'EventbriteScraper', 'EventbriteEvent',
    'MeetupScraper', 'MeetupEvent', 
    'InstagramEventScraper', 'InstagramEvent',
    'TikTokEventScraper', 'TikTokEvent',
    'InstagramViralEventScraper', 'ViralEventDiscoveryEngine', 'TrendingEvent'
]