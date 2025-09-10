#!/usr/bin/env python3
"""
Social Scraper Manager for Copenhagen Event Discovery (2025).
Coordinates Instagram and TikTok scrapers for comprehensive event data collection.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path
from dataclasses import asdict

from instagram import InstagramEventScraper, InstagramEvent
from instagram_viral import InstagramViralEventScraper
from tiktok_modern import ModernTikTokScraper, TikTokEvent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialScraperManager:
    """Manages and coordinates social media scrapers for event discovery."""
    
    def __init__(self, 
                 instagram_username: Optional[str] = None, 
                 instagram_password: Optional[str] = None,
                 tiktok_ms_token: Optional[str] = None,
                 output_dir: str = "scraped_events"):
        """
        Initialize the social scraper manager.
        
        Args:
            instagram_username: Optional Instagram login username
            instagram_password: Optional Instagram login password
            tiktok_ms_token: Optional TikTok MS token for authenticated requests
            output_dir: Directory to save scraped data
        """
        self.instagram_username = instagram_username
        self.instagram_password = instagram_password
        self.tiktok_ms_token = tiktok_ms_token
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scrapers
        self.instagram_scraper = None
        self.instagram_viral_scraper = None
        self.tiktok_scraper = None
    
    def setup_instagram_scrapers(self):
        """Initialize Instagram scrapers."""
        logger.info("Setting up Instagram scrapers...")
        
        # Standard venue scraper
        self.instagram_scraper = InstagramEventScraper(
            username=self.instagram_username,
            password=self.instagram_password
        )
        
        # Viral event discovery scraper
        self.instagram_viral_scraper = InstagramViralEventScraper(
            username=self.instagram_username,
            password=self.instagram_password
        )
        
        logger.info("Instagram scrapers initialized successfully")
    
    async def setup_tiktok_scraper(self):
        """Initialize TikTok scraper."""
        logger.info("Setting up TikTok scraper...")
        
        self.tiktok_scraper = ModernTikTokScraper(ms_token=self.tiktok_ms_token)
        await self.tiktok_scraper.initialize_api()
        
        logger.info("TikTok scraper initialized successfully")
    
    async def scrape_all_platforms(self, 
                                   days_back: int = 7, 
                                   max_events_per_platform: int = 50,
                                   include_viral: bool = True) -> Dict[str, List]:
        """
        Scrape events from all social media platforms.
        
        Args:
            days_back: How many days back to search
            max_events_per_platform: Maximum events per platform
            include_viral: Whether to include viral content discovery
            
        Returns:
            Dictionary with events from each platform
        """
        
        all_events = {
            "instagram_venues": [],
            "instagram_viral": [],
            "tiktok": [],
            "combined": []
        }
        
        # Setup scrapers
        if not self.instagram_scraper:
            self.setup_instagram_scrapers()
        
        if not self.tiktok_scraper:
            await self.setup_tiktok_scraper()
        
        # Scrape Instagram venues
        logger.info("Scraping Instagram venue accounts...")
        try:
            instagram_events = self.instagram_scraper.scrape_venues(
                days_back=days_back,
                max_posts_per_venue=max_events_per_platform // 10
            )
            all_events["instagram_venues"] = instagram_events
            logger.info(f"Found {len(instagram_events)} events from Instagram venues")
        except Exception as e:
            logger.error(f"Error scraping Instagram venues: {e}")
        
        # Scrape Instagram viral content
        if include_viral:
            logger.info("Scraping Instagram viral content...")
            try:
                viral_events = self.instagram_viral_scraper.scrape_viral_events(
                    days_back=days_back,
                    max_posts_per_hashtag=max_events_per_platform // 20,
                    min_engagement=50
                )
                all_events["instagram_viral"] = viral_events
                logger.info(f"Found {len(viral_events)} viral events from Instagram")
            except Exception as e:
                logger.error(f"Error scraping Instagram viral content: {e}")
        
        # Scrape TikTok
        logger.info("Scraping TikTok content...")
        try:
            async with self.tiktok_scraper:
                tiktok_events = await self.tiktok_scraper.scrape_events(
                    max_events=max_events_per_platform,
                    days_back=days_back
                )
                all_events["tiktok"] = tiktok_events
                logger.info(f"Found {len(tiktok_events)} events from TikTok")
        except Exception as e:
            logger.error(f"Error scraping TikTok: {e}")
        
        # Combine and deduplicate events
        all_events["combined"] = self._combine_and_deduplicate_events(
            all_events["instagram_venues"] + 
            all_events["instagram_viral"], 
            all_events["tiktok"]
        )
        
        logger.info(f"Total unique events found: {len(all_events['combined'])}")
        return all_events
    
    def _combine_and_deduplicate_events(self, 
                                      instagram_events: List[InstagramEvent], 
                                      tiktok_events: List[TikTokEvent]) -> List[Dict]:
        """
        Combine and deduplicate events from different platforms.
        
        Args:
            instagram_events: List of Instagram events
            tiktok_events: List of TikTok events
            
        Returns:
            Combined and deduplicated list of events
        """
        
        combined_events = []
        seen_venues_and_dates = set()
        
        # Process Instagram events
        for event in instagram_events:
            event_dict = self._instagram_event_to_dict(event)
            event_key = self._create_event_key(event_dict)
            
            if event_key not in seen_venues_and_dates:
                seen_venues_and_dates.add(event_key)
                combined_events.append(event_dict)
        
        # Process TikTok events
        for event in tiktok_events:
            event_dict = self._tiktok_event_to_dict(event)
            event_key = self._create_event_key(event_dict)
            
            if event_key not in seen_venues_and_dates:
                seen_venues_and_dates.add(event_key)
                combined_events.append(event_dict)
        
        # Sort by engagement/viral score
        combined_events.sort(key=lambda x: x.get('engagement_score', 0), reverse=True)
        
        return combined_events
    
    def _instagram_event_to_dict(self, event: InstagramEvent) -> Dict:
        """Convert Instagram event to standardized dictionary."""
        return {
            "platform": "instagram",
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "venue_name": event.venue_name,
            "venue_username": event.venue_username,
            "url": event.post_url,
            "image_url": event.image_url,
            "date_time": event.date_time.isoformat() if event.date_time else None,
            "hashtags": event.hashtags,
            "detected_artists": event.detected_artists,
            "detected_genres": event.detected_genres,
            "engagement_score": event.likes + event.comments * 2,
            "likes": event.likes,
            "comments": event.comments,
            "views": None  # Instagram doesn't provide view count
        }
    
    def _tiktok_event_to_dict(self, event: TikTokEvent) -> Dict:
        """Convert TikTok event to standardized dictionary."""
        return {
            "platform": "tiktok",
            "id": event.id,
            "title": event.title,
            "description": event.description,
            "venue_name": event.venue_name,
            "venue_username": event.username,
            "url": event.video_url,
            "image_url": event.thumbnail_url,
            "date_time": event.date_time.isoformat() if event.date_time else None,
            "hashtags": event.hashtags,
            "detected_artists": event.detected_artists,
            "detected_genres": event.detected_genres,
            "engagement_score": event.viral_score,
            "likes": event.likes,
            "comments": event.comments,
            "views": event.views,
            "shares": event.shares,
            "is_trending": event.is_trending,
            "music_title": event.music_title,
            "music_author": event.music_author,
            "duration": event.duration
        }
    
    def _create_event_key(self, event_dict: Dict) -> str:
        """Create a unique key for event deduplication."""
        venue = event_dict.get('venue_name', '').lower()
        date = event_dict.get('date_time', '')
        title_words = event_dict.get('title', '').lower().split()[:3]  # First 3 words
        title_key = '_'.join(title_words)
        
        return f"{venue}_{date}_{title_key}"
    
    def save_events_to_json(self, events_data: Dict[str, List], filename: str = None):
        """Save scraped events to JSON file."""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"copenhagen_events_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert datetime objects to ISO strings for JSON serialization
        json_data = {
            "scraped_at": datetime.now().isoformat(),
            "total_events": {
                "instagram_venues": len(events_data.get("instagram_venues", [])),
                "instagram_viral": len(events_data.get("instagram_viral", [])),
                "tiktok": len(events_data.get("tiktok", [])),
                "combined_unique": len(events_data.get("combined", []))
            },
            "events": events_data["combined"]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved events to {filepath}")
        return filepath
    
    def get_top_events(self, events: List[Dict], limit: int = 10) -> List[Dict]:
        """Get top events by engagement score."""
        return sorted(events, key=lambda x: x.get('engagement_score', 0), reverse=True)[:limit]
    
    def get_events_by_venue(self, events: List[Dict], venue_name: str) -> List[Dict]:
        """Filter events by venue name."""
        return [event for event in events if venue_name.lower() in event.get('venue_name', '').lower()]
    
    def get_events_by_genre(self, events: List[Dict], genre: str) -> List[Dict]:
        """Filter events by music genre."""
        return [event for event in events 
                if genre.lower() in [g.lower() for g in event.get('detected_genres', [])]]
    
    async def close(self):
        """Clean up resources."""
        if self.tiktok_scraper:
            await self.tiktok_scraper.close_session()


async def main():
    """Example usage of the Social Scraper Manager."""
    
    # Initialize manager
    manager = SocialScraperManager()
    
    print("Starting comprehensive Copenhagen event scraping...")
    
    try:
        # Scrape all platforms
        all_events = await manager.scrape_all_platforms(
            days_back=7,
            max_events_per_platform=30,
            include_viral=True
        )
        
        # Save to JSON
        output_file = manager.save_events_to_json(all_events)
        
        # Display results
        print(f"\nScraping Results:")
        print(f"- Instagram venues: {len(all_events['instagram_venues'])} events")
        print(f"- Instagram viral: {len(all_events['instagram_viral'])} events")
        print(f"- TikTok: {len(all_events['tiktok'])} events")
        print(f"- Combined unique: {len(all_events['combined'])} events")
        
        # Show top events
        top_events = manager.get_top_events(all_events['combined'], limit=5)
        print(f"\nTop 5 Events by Engagement:")
        for i, event in enumerate(top_events, 1):
            print(f"{i}. {event['title']}")
            print(f"   Platform: {event['platform'].upper()}")
            print(f"   Venue: {event['venue_name']}")
            print(f"   Engagement Score: {event['engagement_score']:.1f}")
            print(f"   Genres: {', '.join(event['detected_genres'][:3])}")
            print()
        
        # Show events by popular venues
        popular_venues = ["Vega", "Rust", "Culture Box"]
        for venue in popular_venues:
            venue_events = manager.get_events_by_venue(all_events['combined'], venue)
            if venue_events:
                print(f"\nEvents at {venue}: {len(venue_events)}")
                for event in venue_events[:2]:
                    print(f"  - {event['title']}")
        
        print(f"\nFull data saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main scraping process: {e}")
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())