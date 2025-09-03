#!/usr/bin/env python3
"""
TikTok scraper for Copenhagen nightlife events.
Uses unofficial TikTok API to find event-related videos from venues.
"""

import requests
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import logging
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TikTokEvent:
    """Structured event data extracted from TikTok videos."""

    id: str
    title: str
    description: str
    date_time: Optional[datetime]
    venue_name: str
    venue_username: str
    video_url: str
    thumbnail_url: str
    views: int
    likes: int
    hashtags: List[str]
    detected_artists: List[str]
    detected_genres: List[str]
    music_title: Optional[str]


class TikTokEventScraper:
    """Scraper for events from Copenhagen venue TikTok accounts."""

    # Major Copenhagen venues and their TikTok handles
    COPENHAGEN_VENUES = {
        "vegacph": "Vega",
        "rustcph": "Rust",
        "cultureboxcph": "Culture Box",
        "loppencph": "Loppen",
        "jolenecph": "Jolene",
        "kb18cph": "KB18",
        "pumpehusetcph": "Pumpehuset",
        "amagerbio": "Amager Bio",
        "alicecph": "ALICE",
        "beta2300cph": "BETA2300",
        "bruscph": "BRUS",
        "chateaumotelcph": "Chateau Motel",
    }

    # Event-related hashtags to search for (viral/organic event discovery)
    EVENT_HASHTAGS = [
        # General Copenhagen events (viral discovery)
        "copenhagenevents",
        "cphevents",
        "copenhagentonight",
        "cphtonight",
        "copenhagennow",
        "cphweekend",
        "copenhagenweekend",
        "copenhagenfun",
        "cphlife",
        "copenhagenlife",
        "visitcopenhagen",
        "cphvibes",
        # Music and nightlife discovery
        "copenhagenmusic",
        "cphmusic",
        "copenhagennight",
        "cphnight",
        "copenhagennightlife",
        "cphparty",
        "copenhagenparty",
        "cphclub",
        "copenhagenclub",
        "klubaften",
        "fest",
        "technocopenhagen",
        "cphtechno",
        # Neighborhood-specific trending
        "vesterbro",
        "nørrebro",
        "indrebr",
        "christiania",
        "østerbro",
        "vesterbrovibes",
        "nørrebrovibes",
        "indrebynight",
        # Event types that go viral
        "undergroundcph",
        "secretparty",
        "rooftopparty",
        "warehouseparty",
        "ravecph",
        "festivaler",
        "koncert",
        "livemusic",
        "dj",
        # Trending/viral markers
        "fyp",
        "trending",
        "viral",
        "mustsee",
        "dontmiss",
        "tonight",
        "thissaturday",
        "weekend",
        "eventanbefalinger",
        "hidden gems",
    ]

    GENRE_KEYWORDS = [
        "techno",
        "house",
        "electronic",
        "disco",
        "funk",
        "soul",
        "jazz",
        "rock",
        "punk",
        "indie",
        "alternative",
        "pop",
        "hiphop",
        "rap",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.request_count = 0
        self.max_requests_per_minute = 30
        self.last_request_time = 0

    def scrape_venues(
        self,
        venue_usernames: List[str] = None,
        days_back: int = 30,
        max_videos_per_venue: int = 50,
    ) -> List[TikTokEvent]:
        """
        Scrape events from venue TikTok accounts.

        Args:
            venue_usernames: List of TikTok usernames to scrape
            days_back: How many days back to look for videos
            max_videos_per_venue: Maximum videos to check per venue
        """

        if not venue_usernames:
            venue_usernames = list(self.COPENHAGEN_VENUES.keys())

        all_events = []

        for username in venue_usernames:
            logger.info(f"Scraping TikTok account: @{username}")

            try:
                events = self._scrape_venue_videos(
                    username,
                    self.COPENHAGEN_VENUES.get(username, username),
                    days_back,
                    max_videos_per_venue,
                )

                all_events.extend(events)
                logger.info(f"Found {len(events)} events from @{username}")

                # Rate limiting
                self._rate_limit()

            except Exception as e:
                logger.error(f"Error scraping @{username}: {e}")
                continue

        # Search by viral/trending hashtags (prioritize general event discovery)
        viral_hashtags = [
            "copenhagenevents",
            "cphevents",
            "copenhagentonight",
            "cphtonight",
            "undergroundcph",
            "secretparty",
            "rooftopparty",
            "copenhagenmusic",
            "cphlife",
            "vesterbrovibes",
            "nørrebrovibes",
            "technocopenhagen",
        ]
        hashtag_events = self._scrape_by_hashtags(viral_hashtags, days_back)
        all_events.extend(hashtag_events)

        # Search for trending event content (not just hashtags)
        trending_events = self._scrape_trending_event_content(days_back)
        all_events.extend(trending_events)

        logger.info(f"Total events found: {len(all_events)}")
        return all_events

    def _scrape_venue_videos(
        self, username: str, venue_name: str, days_back: int, max_videos: int
    ) -> List[TikTokEvent]:
        """Scrape event videos from a single venue account."""

        events = []

        try:
            # Get user info and videos using unofficial API
            user_data = self._get_user_data(username)
            if not user_data:
                return events

            videos = self._get_user_videos(username, max_videos)
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for video in videos:
                try:
                    # Check if video is recent enough
                    video_date = datetime.fromtimestamp(video.get("createTime", 0))
                    if video_date < cutoff_date:
                        continue

                    # Check if video looks like an event
                    if self._is_event_video(video):
                        event = self._extract_event_data(video, venue_name, username)
                        if event:
                            events.append(event)

                except Exception as e:
                    logger.warning(f"Failed to parse video: {e}")
                    continue

                self._rate_limit()

        except Exception as e:
            logger.error(f"Error getting videos from @{username}: {e}")

        return events

    def _scrape_by_hashtags(
        self, hashtags: List[str], days_back: int, max_videos_per_hashtag: int = 30
    ) -> List[TikTokEvent]:
        """Search for events by hashtags."""

        events = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for hashtag in hashtags:
            logger.info(f"Searching hashtag: #{hashtag}")

            try:
                videos = self._search_hashtag(hashtag, max_videos_per_hashtag)

                for video in videos:
                    try:
                        video_date = datetime.fromtimestamp(video.get("createTime", 0))
                        if video_date < cutoff_date:
                            continue

                        if self._is_event_video(video):
                            # Try to determine venue from hashtags/description
                            venue_name = self._extract_venue_from_video(video)
                            username = video.get("author", {}).get(
                                "uniqueId", "unknown"
                            )

                            event = self._extract_event_data(
                                video, venue_name, username
                            )
                            if event:
                                events.append(event)

                    except Exception as e:
                        logger.warning(f"Failed to parse hashtag video: {e}")
                        continue

                self._rate_limit()

            except Exception as e:
                logger.error(f"Error searching hashtag #{hashtag}: {e}")
                continue

        return events

    def _scrape_trending_event_content(
        self, days_back: int, max_videos: int = 100
    ) -> List[TikTokEvent]:
        """Search for trending event content using keyword search."""

        events = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Search terms that capture viral event content
        search_terms = [
            "copenhagen event tonight",
            "secret party copenhagen",
            "underground copenhagen",
            "copenhagen rave",
            "københavn fest",
            "techno copenhagen tonight",
            "best party copenhagen",
            "warehouse party cph",
            "rooftop party copenhagen",
            "copenhagen music event",
        ]

        for term in search_terms:
            logger.info(f"Searching TikTok for: '{term}'")

            try:
                videos = self._search_by_keywords(term, max_videos // len(search_terms))

                for video in videos:
                    try:
                        video_date = datetime.fromtimestamp(video.get("createTime", 0))
                        if video_date < cutoff_date:
                            continue

                        if self._is_viral_event_content(video):
                            # Extract venue/location from viral content
                            venue_name = self._extract_venue_from_viral_content(video)
                            username = video.get("author", {}).get(
                                "uniqueId", "viral_user"
                            )

                            event = self._extract_event_data(
                                video, venue_name, username
                            )
                            if event:
                                # Mark as viral content
                                event.detected_genres.append("viral")
                                events.append(event)

                    except Exception as e:
                        logger.warning(f"Failed to parse viral content: {e}")
                        continue

                self._rate_limit()

            except Exception as e:
                logger.error(f"Error searching for '{term}': {e}")
                continue

        return events

    def _search_by_keywords(self, keywords: str, max_videos: int) -> List[Dict]:
        """Search TikTok videos by keywords."""

        videos = []
        cursor = 0
        count = min(30, max_videos)

        while len(videos) < max_videos:
            # Use TikTok's search API
            url = f"https://www.tiktok.com/api/search/item/"
            params = {
                "keyword": keywords,
                "count": count,
                "cursor": cursor,
                "search_type": 1,  # Video search
            }

            try:
                self._rate_limit()
                response = self.session.get(url, params=params)

                if response.status_code != 200:
                    break

                data = response.json()
                items = data.get("item_list", [])

                if not items:
                    break

                videos.extend(items)

                if not data.get("has_more", False):
                    break

                cursor = data.get("cursor", cursor + count)

            except Exception as e:
                logger.warning(f"Error in keyword search: {e}")
                break

        return videos[:max_videos]

    def _is_viral_event_content(self, video: Dict) -> bool:
        """Check if video is viral event content (not just venue self-promotion)."""

        desc = video.get("desc", "").lower()
        hashtags = [
            tag.get("hashtagName", "").lower() for tag in video.get("textExtra", [])
        ]
        author = video.get("author", {}).get("uniqueId", "").lower()

        # Combine all text
        all_text = f"{desc} {' '.join(hashtags)}"

        # Indicators of viral/organic event content
        viral_indicators = [
            "found this party",
            "stumbled upon",
            "discovered",
            "hidden gem",
            "secret location",
            "underground",
            "word of mouth",
            "invite only",
            "crazy night",
            "insane party",
            "best night ever",
            "you need to go",
            "dont miss",
            "trending",
            "viral",
            "everyone talking about",
            "packed",
            "sold out",
            "queue around the block",
            "legendary",
            "spontaneous",
            "popup",
            "last minute",
            "surprise event",
        ]

        # Event urgency/excitement indicators
        urgency_indicators = [
            "tonight",
            "right now",
            "happening now",
            "live",
            "currently",
            "omg",
            "insane",
            "wild",
            "unreal",
            "epic",
            "legendary",
        ]

        # Copenhagen event location indicators
        location_indicators = [
            "copenhagen",
            "cph",
            "københavn",
            "vesterbro",
            "nørrebro",
            "indre by",
            "christiania",
            "østerbro",
            "islands brygge",
            "refshaleøen",
            "papirøen",
            "sydhavnen",
        ]

        # Check for viral content patterns
        has_viral_language = any(
            indicator in all_text for indicator in viral_indicators
        )
        has_urgency = any(indicator in all_text for indicator in urgency_indicators)
        has_location = any(indicator in all_text for indicator in location_indicators)

        # Exclude obvious venue self-promotion
        venue_names = [venue.lower() for venue in self.COPENHAGEN_VENUES.values()]
        is_venue_promotion = any(venue in author for venue in venue_names)

        # High engagement indicates viral content
        stats = video.get("stats", {})
        like_count = stats.get("diggCount", 0)
        comment_count = stats.get("commentCount", 0)
        share_count = stats.get("shareCount", 0)

        high_engagement = like_count > 1000 or comment_count > 100 or share_count > 50

        # Must have location + (viral language OR urgency OR high engagement) AND not venue self-promotion
        return (
            has_location
            and (has_viral_language or has_urgency or high_engagement)
            and not is_venue_promotion
        )

    def _extract_venue_from_viral_content(self, video: Dict) -> str:
        """Extract venue information from viral event content."""

        desc = video.get("desc", "").lower()
        hashtags = [
            tag.get("hashtagName", "").lower() for tag in video.get("textExtra", [])
        ]
        all_text = f"{desc} {' '.join(hashtags)}"

        # Check for known venues mentioned
        for username, venue_name in self.COPENHAGEN_VENUES.items():
            if venue_name.lower() in all_text or username.lower() in all_text:
                return venue_name

        # Look for location patterns
        location_patterns = [
            r"at\s+([A-Za-z\s]+)",  # "at [venue name]"
            r"@\s*([A-Za-z\s]+)",  # "@[venue name]"
            r"(\w+\s*warehouse|warehouse\s*\w+)",  # warehouse parties
            r"rooftop\s+(\w+)",  # rooftop locations
            r"secret\s+location\s+(\w+)",  # secret locations
        ]

        for pattern in location_patterns:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                potential_venue = match.group(1).strip().title()
                if len(potential_venue) > 2 and len(potential_venue) < 30:
                    return potential_venue

        # Copenhagen neighborhoods as fallback venues
        neighborhoods = [
            "Vesterbro",
            "Nørrebro",
            "Indre By",
            "Christiania",
            "Østerbro",
            "Islands Brygge",
            "Refshaleøen",
        ]

        for neighborhood in neighborhoods:
            if neighborhood.lower() in all_text:
                return f"Event in {neighborhood}"

        return "Viral Copenhagen Event"

    def _get_user_data(self, username: str) -> Optional[Dict]:
        """Get TikTok user data using unofficial API."""

        # This uses a common TikTok web API endpoint
        url = f"https://www.tiktok.com/api/user/detail/?uniqueId={username}"

        try:
            self._rate_limit()
            response = self.session.get(url)

            if response.status_code == 200:
                data = response.json()
                return data.get("userInfo", {}).get("user", {})

        except Exception as e:
            logger.warning(f"Failed to get user data for @{username}: {e}")

        return None

    def _get_user_videos(self, username: str, max_videos: int) -> List[Dict]:
        """Get recent videos from a TikTok user."""

        videos = []
        cursor = 0
        count = min(30, max_videos)  # TikTok API limit

        while len(videos) < max_videos:
            url = f"https://www.tiktok.com/api/post/item_list/"
            params = {"uniqueId": username, "count": count, "cursor": cursor}

            try:
                self._rate_limit()
                response = self.session.get(url, params=params)

                if response.status_code != 200:
                    break

                data = response.json()
                items = data.get("itemList", [])

                if not items:
                    break

                videos.extend(items)

                # Check if there are more videos
                if not data.get("hasMore", False):
                    break

                cursor = data.get("cursor", cursor + count)

            except Exception as e:
                logger.warning(f"Error fetching videos: {e}")
                break

        return videos[:max_videos]

    def _search_hashtag(self, hashtag: str, max_videos: int) -> List[Dict]:
        """Search videos by hashtag."""

        videos = []
        cursor = 0
        count = min(30, max_videos)

        while len(videos) < max_videos:
            url = f"https://www.tiktok.com/api/challenge/item_list/"
            params = {"challengeName": hashtag, "count": count, "cursor": cursor}

            try:
                self._rate_limit()
                response = self.session.get(url, params=params)

                if response.status_code != 200:
                    break

                data = response.json()
                items = data.get("itemList", [])

                if not items:
                    break

                videos.extend(items)

                if not data.get("hasMore", False):
                    break

                cursor = data.get("cursor", cursor + count)

            except Exception as e:
                logger.warning(f"Error searching hashtag: {e}")
                break

        return videos[:max_videos]

    def _is_event_video(self, video: Dict) -> bool:
        """Determine if a TikTok video is likely about an event."""

        desc = video.get("desc", "").lower()
        hashtags = [
            tag.get("hashtagName", "").lower() for tag in video.get("textExtra", [])
        ]
        music_title = video.get("music", {}).get("title", "").lower()

        # Combine all text
        all_text = f"{desc} {' '.join(hashtags)} {music_title}"

        # Check for event keywords
        event_keywords = [
            "live",
            "concert",
            "show",
            "event",
            "tonight",
            "saturday",
            "friday",
            "party",
            "fest",
            "klub",
            "night",
            "aften",
            "weekend",
        ]

        has_event_keywords = any(keyword in all_text for keyword in event_keywords)

        # Check for venue names
        has_venue = any(
            venue.lower() in all_text for venue in self.COPENHAGEN_VENUES.values()
        )

        # Check for Copenhagen-related hashtags
        cph_hashtags = any(tag in hashtags for tag in self.EVENT_HASHTAGS)

        return has_event_keywords or has_venue or cph_hashtags

    def _extract_event_data(
        self, video: Dict, venue_name: str, username: str
    ) -> Optional[TikTokEvent]:
        """Extract structured event data from TikTok video."""

        try:
            video_id = video.get("id", "")
            desc = video.get("desc", "")

            # URLs
            video_url = f"https://www.tiktok.com/@{username}/video/{video_id}"
            thumbnail_url = video.get("video", {}).get("cover", "")

            # Stats
            stats = video.get("stats", {})
            views = stats.get("playCount", 0)
            likes = stats.get("diggCount", 0)

            # Hashtags
            hashtags = [
                tag.get("hashtagName", "") for tag in video.get("textExtra", [])
            ]

            # Music info
            music = video.get("music", {})
            music_title = music.get("title")

            # Extract artists and genres
            artists = self._extract_artists_from_text(desc)
            genres = self._extract_genres_from_text(f"{desc} {' '.join(hashtags)}")

            # Try to extract date (simplified - would need NLP for better results)
            date_time = self._extract_date_from_text(desc)

            # Create title from description or music
            title = desc[:100] if desc else (music_title or f"Event at {venue_name}")

            return TikTokEvent(
                id=f"tt_{video_id}",
                title=title,
                description=desc,
                date_time=date_time,
                venue_name=venue_name,
                venue_username=username,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
                views=views,
                likes=likes,
                hashtags=hashtags,
                detected_artists=artists,
                detected_genres=genres,
                music_title=music_title,
            )

        except Exception as e:
            logger.warning(f"Error extracting TikTok event data: {e}")
            return None

    def _extract_venue_from_video(self, video: Dict) -> str:
        """Try to determine venue from video hashtags/description."""

        desc = video.get("desc", "").lower()
        hashtags = [
            tag.get("hashtagName", "").lower() for tag in video.get("textExtra", [])
        ]
        all_text = f"{desc} {' '.join(hashtags)}"

        # Check against known venues
        for username, venue_name in self.COPENHAGEN_VENUES.items():
            if venue_name.lower() in all_text or username.lower() in all_text:
                return venue_name

        return "Unknown Venue"

    def _extract_artists_from_text(self, text: str) -> List[str]:
        """Extract potential artist names from text."""

        # Look for capitalized words that might be artist names
        words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)

        # Filter out common words
        common_words = {
            "TikTok",
            "Copenhagen",
            "Denmark",
            "Friday",
            "Saturday",
            "Sunday",
            "Live",
            "Music",
            "Tonight",
            "Event",
            "Party",
            "Club",
        }

        artists = [word for word in words if word not in common_words and len(word) > 2]
        return artists[:3]  # Limit to 3 potential artists

    def _extract_genres_from_text(self, text: str) -> List[str]:
        """Extract music genres mentioned in text."""

        text_lower = text.lower()
        genres = [genre for genre in self.GENRE_KEYWORDS if genre in text_lower]
        return list(set(genres))

    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Simplified date extraction - would use NLP transformer for better results."""

        text_lower = text.lower()
        now = datetime.now()

        if "tonight" in text_lower or "i aften" in text_lower:
            return now.replace(hour=20, minute=0, second=0, microsecond=0)
        elif "tomorrow" in text_lower:
            return now + timedelta(days=1)
        elif "friday" in text_lower or "fredag" in text_lower:
            days_ahead = (4 - now.weekday()) % 7
            return now + timedelta(days=days_ahead)
        elif "saturday" in text_lower or "lørdag" in text_lower:
            days_ahead = (5 - now.weekday()) % 7
            return now + timedelta(days=days_ahead)

        return None

    def _rate_limit(self):
        """Simple rate limiting to avoid getting blocked."""

        current_time = time.time()

        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time

        # If we've made too many requests, wait
        if self.request_count >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_count = 0
            self.last_request_time = time.time()

        self.request_count += 1
        time.sleep(2)  # Base delay between requests


def main():
    """Example usage of TikTokEventScraper."""

    scraper = TikTokEventScraper()

    # Test with a few venues
    test_venues = ["vegacph", "rustcph"]

    events = scraper.scrape_venues(
        venue_usernames=test_venues, days_back=14, max_videos_per_venue=20
    )

    print(f"Found {len(events)} events:")
    for event in events[:3]:
        print(f"- {event.title}")
        print(f"  @{event.venue_username} | {event.views} views, {event.likes} likes")
        print(f"  Hashtags: {', '.join(event.hashtags[:3])}")
        print()


if __name__ == "__main__":
    main()
