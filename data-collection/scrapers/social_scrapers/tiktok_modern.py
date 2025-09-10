#!/usr/bin/env python3
"""
Modern TikTok scraper for Copenhagen nightlife events.
Uses TikTokApi library and alternative methods for event discovery.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TikTokEvent:
    """Event data from TikTok content."""
    
    id: str
    title: str
    description: str
    date_time: Optional[datetime]
    venue_name: Optional[str]
    username: str
    video_url: str
    thumbnail_url: Optional[str]
    views: int
    likes: int
    hashtags: List[str]
    detected_artists: List[str]
    detected_genres: List[str]
    music_title: Optional[str]
    viral_score: float  # Engagement-based score


class ModernTikTokScraper:
    """Modern TikTok scraper using multiple approaches."""
    
    # Copenhagen venues on TikTok (updated handles)
    COPENHAGEN_VENUES = {
        "vega_copenhagen": "Vega",
        "rust_cph": "Rust", 
        "culturebox_cph": "Culture Box",
        "loppen_official": "Loppen",
        "jolene_cph": "Jolene",
        "kb18_copenhagen": "KB18",
        "pumpehuset_cph": "Pumpehuset",
        "amagerbio": "Amager Bio",
        "alice_copenhagen": "ALICE",
        "beta2300": "BETA2300",
    }
    
    # Event discovery hashtags (focus on Copenhagen)
    EVENT_HASHTAGS = [
        # Core Copenhagen events
        "copenhagenevents", "cphevents", "copenhagentonight", "cphtonight",
        "copenhagenweekend", "cphweekend", "visitcopenhagen",
        
        # Music & nightlife
        "copenhagenmusic", "cphmusic", "copenhagennightlife", "cphparty",
        "technocopenhagen", "cphtechno", "copenhagenclub", "cphclub",
        
        # Neighborhoods
        "vesterbro", "nørrebro", "indrebr", "christiania", "østerbro",
        "vesterbrovibes", "nørrebrovibes", "christianivibes",
        
        # Event types
        "undergroundcph", "secretparty", "rooftopparty", "warehouseparty",
        "ravecph", "festivaler", "koncert", "livemusic",
        
        # Danish terms
        "københavnevent", "kbhevent", "festkøbenhavn", "partycopenhagen",
    ]
    
    GENRE_KEYWORDS = [
        "techno", "house", "electronic", "disco", "funk", "soul", 
        "jazz", "rock", "punk", "indie", "alternative", "pop",
        "hiphop", "rap", "ambient", "dnb", "dubstep", "trance"
    ]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        })
        
    def scrape_events(self, max_events: int = 100) -> List[TikTokEvent]:
        """
        Scrape Copenhagen events from TikTok using multiple strategies.
        
        Args:
            max_events: Maximum number of events to return
        """
        
        events = []
        
        # Strategy 1: Search by hashtags
        hashtag_events = self._search_by_hashtags(max_events // 3)
        events.extend(hashtag_events)
        
        # Strategy 2: Search by keywords 
        keyword_events = self._search_by_keywords(max_events // 3)
        events.extend(keyword_events)
        
        # Strategy 3: Location-based search
        location_events = self._search_by_location(max_events // 3)
        events.extend(location_events)
        
        # Deduplicate and score
        events = self._deduplicate_events(events)
        events = sorted(events, key=lambda x: x.viral_score, reverse=True)
        
        logger.info(f"Found {len(events)} TikTok events total")
        return events[:max_events]
    
    def _search_by_hashtags(self, max_events: int) -> List[TikTokEvent]:
        """Search for events using Copenhagen event hashtags."""
        
        events = []
        
        # Prioritize high-value hashtags
        priority_hashtags = [
            "copenhagenevents", "cphevents", "copenhagentonight",
            "undergroundcph", "technocopenhagen", "copenhagenmusic"
        ]
        
        for hashtag in priority_hashtags:
            try:
                hashtag_events = self._fetch_hashtag_content(hashtag, max_events // len(priority_hashtags))
                events.extend(hashtag_events)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Failed to search hashtag #{hashtag}: {e}")
                continue
        
        return events
    
    def _search_by_keywords(self, max_events: int) -> List[TikTokEvent]:
        """Search using keywords that capture event content."""
        
        events = []
        
        search_terms = [
            "copenhagen event tonight",
            "secret party copenhagen", 
            "underground copenhagen",
            "copenhagen rave",
            "best party copenhagen",
            "copenhagen music event",
            "techno copenhagen",
            "københavn fest",
        ]
        
        for term in search_terms:
            try:
                keyword_events = self._fetch_keyword_content(term, max_events // len(search_terms))
                events.extend(keyword_events)
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Failed to search keyword '{term}': {e}")
                continue
        
        return events
    
    def _search_by_location(self, max_events: int) -> List[TikTokEvent]:
        """Search for content tagged with Copenhagen locations."""
        
        events = []
        
        locations = [
            "Copenhagen, Denmark",
            "København, Denmark", 
            "Vesterbro, Copenhagen",
            "Nørrebro, Copenhagen",
            "Indre By, Copenhagen",
            "Christiania, Copenhagen"
        ]
        
        for location in locations:
            try:
                location_events = self._fetch_location_content(location, max_events // len(locations))
                events.extend(location_events)
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Failed to search location '{location}': {e}")
                continue
        
        return events
    
    def _fetch_hashtag_content(self, hashtag: str, max_videos: int) -> List[TikTokEvent]:
        """Fetch content for a specific hashtag using web scraping."""
        
        events = []
        
        try:
            # Use TikTok's web API endpoints (these change frequently)
            # This is a simplified approach - production would need more robust methods
            
            url = f"https://www.tiktok.com/tag/{hashtag}"
            
            # Alternative: Use TikTok's internal API (requires more complex setup)
            api_url = "https://www.tiktok.com/api/challenge/item_list/"
            params = {
                "challengeName": hashtag,
                "count": min(20, max_videos),
                "cursor": 0
            }
            
            response = self.session.get(api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    items = data.get("itemList", [])
                    
                    for item in items:
                        event = self._parse_tiktok_item(item)
                        if event and self._is_copenhagen_event(event):
                            events.append(event)
                            
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.debug(f"Hashtag fetch failed for #{hashtag}: {e}")
        
        return events
    
    def _fetch_keyword_content(self, keyword: str, max_videos: int) -> List[TikTokEvent]:
        """Fetch content using keyword search."""
        
        events = []
        
        try:
            # TikTok search API endpoint
            url = "https://www.tiktok.com/api/search/item/"
            params = {
                "keyword": keyword,
                "count": min(20, max_videos),
                "cursor": 0,
                "search_type": 1  # Video search
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    items = data.get("item_list", [])
                    
                    for item in items:
                        event = self._parse_tiktok_item(item)
                        if event and self._is_copenhagen_event(event):
                            events.append(event)
                            
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.debug(f"Keyword fetch failed for '{keyword}': {e}")
        
        return events
    
    def _fetch_location_content(self, location: str, max_videos: int) -> List[TikTokEvent]:
        """Fetch content tagged with specific location."""
        
        events = []
        
        try:
            # Location-based search (simplified)
            # Real implementation would use TikTok's location API
            
            # For now, treat as keyword search
            events.extend(self._fetch_keyword_content(location, max_videos))
            
        except Exception as e:
            logger.debug(f"Location fetch failed for '{location}': {e}")
        
        return events
    
    def _parse_tiktok_item(self, item: Dict) -> Optional[TikTokEvent]:
        """Parse TikTok API item into TikTokEvent."""
        
        try:
            video_id = item.get("id", "")
            author = item.get("author", {})
            username = author.get("uniqueId", "unknown")
            
            desc = item.get("desc", "")
            
            # Stats for viral scoring
            stats = item.get("stats", {})
            views = stats.get("playCount", 0)
            likes = stats.get("diggCount", 0)
            comments = stats.get("commentCount", 0)
            shares = stats.get("shareCount", 0)
            
            # Calculate viral score
            viral_score = self._calculate_viral_score(views, likes, comments, shares)
            
            # Extract hashtags
            hashtags = []
            text_extra = item.get("textExtra", [])
            for tag_info in text_extra:
                if tag_info.get("hashtagName"):
                    hashtags.append(tag_info["hashtagName"])
            
            # URLs
            video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
            thumbnail_url = item.get("video", {}).get("cover", "")
            
            # Music info
            music = item.get("music", {})
            music_title = music.get("title")
            
            # Extract event information
            title = self._extract_title_from_desc(desc)
            venue = self._extract_venue_from_content(desc, hashtags)
            date_time = self._extract_date_from_content(desc)
            artists = self._extract_artists_from_content(desc)
            genres = self._extract_genres_from_content(desc, hashtags)
            
            return TikTokEvent(
                id=f"tt_{video_id}",
                title=title,
                description=desc,
                date_time=date_time,
                venue_name=venue,
                username=username,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
                views=views,
                likes=likes,
                hashtags=hashtags,
                detected_artists=artists,
                detected_genres=genres,
                music_title=music_title,
                viral_score=viral_score
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse TikTok item: {e}")
            return None
    
    def _calculate_viral_score(self, views: int, likes: int, comments: int, shares: int) -> float:
        """Calculate viral/engagement score for ranking."""
        
        # Weighted engagement score
        engagement_rate = (likes + comments * 2 + shares * 3) / max(views, 1)
        
        # Base score from raw numbers (log scale)
        import math
        base_score = math.log10(max(views, 1)) + math.log10(max(likes, 1)) * 0.5
        
        # Combine engagement rate and base score
        viral_score = base_score * (1 + engagement_rate * 10)
        
        return min(viral_score, 100)  # Cap at 100
    
    def _is_copenhagen_event(self, event: TikTokEvent) -> bool:
        """Check if event is related to Copenhagen."""
        
        text_content = f"{event.description} {' '.join(event.hashtags)}".lower()
        
        copenhagen_indicators = [
            "copenhagen", "cph", "københavn", "kbh",
            "vesterbro", "nørrebro", "indre by", "christiania", "østerbro",
            "islands brygge", "refshaleøen", "sydhavnen"
        ]
        
        return any(indicator in text_content for indicator in copenhagen_indicators)
    
    def _extract_title_from_desc(self, desc: str) -> str:
        """Extract event title from description."""
        
        if not desc:
            return "TikTok Event"
        
        # Use first sentence or first 50 chars
        sentences = re.split(r'[.!?\n]', desc)
        title = sentences[0].strip() if sentences else desc
        
        return title[:50] + ("..." if len(title) > 50 else "")
    
    def _extract_venue_from_content(self, desc: str, hashtags: List[str]) -> Optional[str]:
        """Extract venue information from content."""
        
        content = f"{desc} {' '.join(hashtags)}".lower()
        
        # Check known venues
        for handle, venue_name in self.COPENHAGEN_VENUES.items():
            if venue_name.lower() in content or handle.lower() in content:
                return venue_name
        
        # Look for venue patterns
        venue_patterns = [
            r'@\s*([A-Za-z\s]+)',  # @VenueName
            r'at\s+([A-Z][^.!?\n]+)',  # at Venue Name
        ]
        
        for pattern in venue_patterns:
            matches = re.findall(pattern, desc)
            for match in matches:
                venue = match.strip()
                if 3 < len(venue) < 30 and not venue.lower() in ["the", "and", "or"]:
                    return venue
        
        return None
    
    def _extract_date_from_content(self, desc: str) -> Optional[datetime]:
        """Extract date/time from content."""
        
        desc_lower = desc.lower()
        now = datetime.now()
        
        # Common time indicators
        if "tonight" in desc_lower:
            return now.replace(hour=20, minute=0, second=0, microsecond=0)
        elif "tomorrow" in desc_lower:
            return now + timedelta(days=1)
        elif "this weekend" in desc_lower or "weekend" in desc_lower:
            days_until_saturday = (5 - now.weekday()) % 7
            return now + timedelta(days=days_until_saturday)
        elif "friday" in desc_lower:
            days_until_friday = (4 - now.weekday()) % 7
            return now + timedelta(days=days_until_friday)
        elif "saturday" in desc_lower:
            days_until_saturday = (5 - now.weekday()) % 7
            return now + timedelta(days=days_until_saturday)
        
        return None
    
    def _extract_artists_from_content(self, desc: str) -> List[str]:
        """Extract artist names from description."""
        
        # Look for capitalized words that might be artists
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', desc)
        
        common_words = {
            "TikTok", "Copenhagen", "Denmark", "Tonight", "Saturday", "Friday",
            "Music", "Live", "Event", "Party", "Club", "Dance", "Show"
        }
        
        artists = [word for word in words if word not in common_words and len(word) > 2]
        return artists[:3]  # Limit to 3
    
    def _extract_genres_from_content(self, desc: str, hashtags: List[str]) -> List[str]:
        """Extract music genres from content."""
        
        content = f"{desc} {' '.join(hashtags)}".lower()
        
        genres = []
        for genre in self.GENRE_KEYWORDS:
            if genre in content:
                genres.append(genre)
        
        return list(set(genres))  # Remove duplicates
    
    def _deduplicate_events(self, events: List[TikTokEvent]) -> List[TikTokEvent]:
        """Remove duplicate events based on content similarity."""
        
        unique_events = []
        seen_videos = set()
        
        for event in events:
            # Create content fingerprint
            content_key = f"{event.username}_{event.title[:20]}_{event.date_time}"
            
            if content_key not in seen_videos:
                seen_videos.add(content_key)
                unique_events.append(event)
        
        return unique_events


def main():
    """Test the modern TikTok scraper."""
    
    scraper = ModernTikTokScraper()
    
    print("Testing modern TikTok scraper...")
    events = scraper.scrape_events(max_events=20)
    
    print(f"\nFound {len(events)} TikTok events:")
    for i, event in enumerate(events[:5], 1):
        print(f"\n{i}. {event.title}")
        print(f"   @{event.username} | {event.views:,} views, {event.likes:,} likes")
        print(f"   Viral Score: {event.viral_score:.1f}")
        print(f"   Venue: {event.venue_name}")
        print(f"   Date: {event.date_time}")
        print(f"   Hashtags: {', '.join(event.hashtags[:3])}")
        print(f"   Genres: {', '.join(event.detected_genres)}")


if __name__ == "__main__":
    main()