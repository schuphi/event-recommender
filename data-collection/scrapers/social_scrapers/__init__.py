#!/usr/bin/env python3
"""
Social Scrapers Package for Copenhagen Event Discovery.

This package contains scrapers for various social media platforms:
- Instagram venue and viral event scraping
- TikTok event content discovery with 2025 methods
- Integrated social scraper manager

Usage:
    from social_scrapers import SocialScraperManager
    
    # Initialize and run comprehensive scraping
    manager = SocialScraperManager()
    events = await manager.scrape_all_platforms()
"""

from .instagram import InstagramEventScraper, InstagramEvent
from .instagram_viral import InstagramViralEventScraper
from .tiktok_modern import ModernTikTokScraper, TikTokEvent
from .social_scraper_manager import SocialScraperManager

__all__ = [
    'InstagramEventScraper',
    'InstagramEvent', 
    'InstagramViralEventScraper',
    'ModernTikTokScraper',
    'TikTokEvent',
    'SocialScraperManager'
]

__version__ = "1.0.0"
__author__ = "Event Recommender Team"
__description__ = "Social media scrapers for Copenhagen nightlife event discovery"