#!/usr/bin/env python3
"""
Instagram viral event discovery scraper for Copenhagen.
Focuses on trending hashtags and viral event content rather than just venue accounts.
"""

import instaloader
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import time
import logging
from pathlib import Path

from instagram import InstagramEventScraper, InstagramEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstagramViralEventScraper(InstagramEventScraper):
    """Enhanced Instagram scraper for viral event discovery."""

    # Viral event hashtags for Copenhagen
    VIRAL_HASHTAGS = [
        # General Copenhagen event discovery
        "copenhagenevents",
        "cphevents",
        "copenhagentonight",
        "cphtonight",
        "copenhagennow",
        "cphweekend",
        "copenhagenweekend",
        "copenhagen_events",
        "cphlife",
        "copenhagenlife",
        "visitcopenhagen",
        "cphvibes",
        # Underground/secret events
        "undergroundcph",
        "secretpartycopenhagen",
        "hiddencph",
        "cphunderground",
        "warehousepartycopenhagen",
        "rooftoppartycopenhagen",
        "secretlocation",
        "popupparty",
        "undergroundrave",
        "secretrave",
        "invite_only",
        # Trending/viral discovery
        "copenhagennightlife",
        "cphparty",
        "copenhagenparty",
        "nightoutcopenhagen",
        "cphnight",
        "copenhagennight",
        "partytime",
        "nightlife",
        "clubbing",
        "ravecph",
        "technikcph",
        "housemusiccph",
        "electronicmusiccph",
        # Neighborhood-specific viral content
        "vesterbrovibes",
        "vesterbronights",
        "nørrebrovibes",
        "nørrebronights",
        "indrebynight",
        "christianiavibes",
        "østerbronights",
        "islandsbrygge",
        "refshaleøen",
        "papirøen",
        "sydhavnen",
        "nordhavn",
        # Event type hashtags that go viral
        "technocopenhagen",
        "housecopenhagen",
        "jazzcopenhagen",
        "rockcopenhagen",
        "indiecopenhagen",
        "festivaler",
        "koncerter",
        "dj_set",
        "livemusic",
        "alternativecopenhagen",
        "experimentalmusic",
        "clubnight",
    ]

    # User types that often post viral event content (not venues)
    VIRAL_USER_TYPES = [
        "event_photographer",
        "party_photographer",
        "nightlife_blogger",
        "music_blogger",
        "copenhagen_influencer",
        "party_goer",
        "raver",
        "event_hunter",
        "underground_scout",
        "music_lover",
        "scene_reporter",
    ]

    def scrape_viral_events(
        self,
        days_back: int = 14,
        max_posts_per_hashtag: int = 50,
        min_engagement: int = 100,  # Minimum likes/comments for viral content
    ) -> List[InstagramEvent]:
        """
        Scrape viral event content from trending hashtags and user posts.

        Args:
            days_back: How many days back to look for posts
            max_posts_per_hashtag: Maximum posts to check per hashtag
            min_engagement: Minimum engagement (likes + comments) for viral content
        """

        all_events = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Search viral hashtags
        for hashtag in self.VIRAL_HASHTAGS:
            logger.info(f"Searching viral hashtag: #{hashtag}")

            try:
                hashtag_obj = instaloader.Hashtag.from_name(
                    self.loader.context, hashtag
                )
                events = self._scrape_hashtag_posts(
                    hashtag_obj,
                    hashtag,
                    cutoff_date,
                    max_posts_per_hashtag,
                    min_engagement,
                )

                all_events.extend(events)
                logger.info(f"Found {len(events)} viral events from #{hashtag}")

                # Rate limiting
                time.sleep(3)

            except instaloader.exceptions.ProfileNotExistsException:
                logger.warning(f"Hashtag #{hashtag} not found")
            except Exception as e:
                logger.error(f"Error scraping hashtag #{hashtag}: {e}")
                continue

        # Search for viral content from specific user types
        viral_user_events = self._scrape_viral_users(days_back, min_engagement)
        all_events.extend(viral_user_events)

        logger.info(f"Total viral events found: {len(all_events)}")
        return all_events

    def _scrape_hashtag_posts(
        self,
        hashtag_obj,
        hashtag_name: str,
        cutoff_date: datetime,
        max_posts: int,
        min_engagement: int,
    ) -> List[InstagramEvent]:
        """Scrape posts from a hashtag for viral event content."""

        events = []
        posts_checked = 0

        try:
            for post in hashtag_obj.get_posts():
                if posts_checked >= max_posts:
                    break

                # Skip old posts
                if post.date < cutoff_date:
                    break

                posts_checked += 1

                # Check engagement threshold
                engagement = post.likes + post.comments
                if engagement < min_engagement:
                    continue

                # Check if post is viral event content
                if self._is_viral_event_post(post):
                    venue_name = self._extract_venue_from_viral_post(post)
                    username = post.owner_username

                    event = self._extract_viral_event_data(
                        post, venue_name, username, hashtag_name
                    )
                    if event:
                        events.append(event)

                # Rate limiting
                time.sleep(2)

        except Exception as e:
            logger.error(f"Error getting posts from hashtag: {e}")

        return events

    def _scrape_viral_users(
        self, days_back: int, min_engagement: int, max_users: int = 20
    ) -> List[InstagramEvent]:
        """Search for viral event content from specific user types."""

        # This is a simplified implementation
        # In practice, you'd maintain a curated list of users who post viral event content

        viral_usernames = [
            # Example users (these would be real accounts in production)
            "copenhagen_nightlife_scout",
            "cph_underground",
            "party_hunter_cph",
            "copenhagen_raves",
            "cph_events_insider",
            "nightlife_copenhagen",
            "underground_cph",
            "techno_copenhagen",
            "house_copenhagen",
        ]

        events = []

        for username in viral_usernames[:max_users]:
            logger.info(f"Checking viral user: @{username}")

            try:
                profile = instaloader.Profile.from_username(
                    self.loader.context, username
                )
                user_events = self._scrape_viral_user_posts(
                    profile, days_back, min_engagement
                )
                events.extend(user_events)

                # Rate limiting
                time.sleep(3)

            except instaloader.exceptions.ProfileNotExistsException:
                logger.warning(f"Viral user @{username} not found")
            except Exception as e:
                logger.error(f"Error scraping viral user @{username}: {e}")
                continue

        return events

    def _scrape_viral_user_posts(
        self, profile, days_back: int, min_engagement: int, max_posts: int = 30
    ) -> List[InstagramEvent]:
        """Scrape posts from a viral user account."""

        events = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        posts_checked = 0

        try:
            for post in profile.get_posts():
                if posts_checked >= max_posts:
                    break

                if post.date < cutoff_date:
                    break

                posts_checked += 1

                # Check engagement threshold
                engagement = post.likes + post.comments
                if engagement < min_engagement:
                    continue

                # Check if post is about events
                if self._is_viral_event_post(post):
                    venue_name = self._extract_venue_from_viral_post(post)

                    event = self._extract_viral_event_data(
                        post, venue_name, profile.username, "viral_user"
                    )
                    if event:
                        events.append(event)

                time.sleep(1)

        except Exception as e:
            logger.error(f"Error getting posts from user: {e}")

        return events

    def _is_viral_event_post(self, post) -> bool:
        """Enhanced detection for viral event posts."""

        caption = post.caption.lower() if post.caption else ""
        hashtags = (
            [tag.lower() for tag in post.caption_hashtags]
            if post.caption_hashtags
            else []
        )

        # Viral event indicators
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
            "hemmelig location",
            "alle snakker om",
            "gå ikke glip af",
        ]

        # Event urgency indicators
        urgency_indicators = [
            "tonight",
            "right now",
            "happening now",
            "live",
            "currently",
            "i aften",
            "lige nu",
            "sker nu",
            "this weekend",
            "denne weekend",
        ]

        # Copenhagen location indicators
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
            "nordhavn",
        ]

        # Check for viral patterns
        has_viral_language = any(indicator in caption for indicator in viral_indicators)
        has_urgency = any(indicator in caption for indicator in urgency_indicators)
        has_location = any(indicator in caption for indicator in location_indicators)

        # Check hashtag patterns
        viral_hashtag_patterns = [
            "underground",
            "secret",
            "popup",
            "warehouse",
            "rooftop",
            "rave",
            "tonight",
            "weekend",
            "viral",
            "trending",
            "epic",
            "insane",
        ]

        has_viral_hashtags = any(
            any(pattern in hashtag for pattern in viral_hashtag_patterns)
            for hashtag in hashtags
        )

        # High engagement threshold (viral content)
        high_engagement = post.likes > 500 or post.comments > 50

        # Must have location + (viral indicators OR high engagement)
        return has_location and (
            has_viral_language or has_urgency or has_viral_hashtags or high_engagement
        )

    def _extract_venue_from_viral_post(self, post) -> str:
        """Extract venue from viral post content."""

        caption = post.caption.lower() if post.caption else ""

        # Check for known venues mentioned in caption
        for username, venue_name in self.COPENHAGEN_VENUES.items():
            if venue_name.lower() in caption or username.lower() in caption:
                return venue_name

        # Look for location patterns in viral content
        location_patterns = [
            r"at\s+([A-Za-z\s]+)",  # "at [venue name]"
            r"@\s*([A-Za-z\s]+)",  # "@[venue name]"
            r"(\w+\s*warehouse|warehouse\s*\w+)",  # warehouse parties
            r"rooftop\s+([A-Za-z\s]+)",  # rooftop locations
            r"secret\s+location\s+([A-Za-z\s]+)",  # secret locations
            r"popup\s+at\s+([A-Za-z\s]+)",  # popup events
        ]

        for pattern in location_patterns:
            matches = re.finditer(pattern, caption, re.IGNORECASE)
            for match in matches:
                potential_venue = match.group(1).strip().title()
                if len(potential_venue) > 2 and len(potential_venue) < 30:
                    return potential_venue

        # Check Instagram location tag
        if hasattr(post, "location") and post.location:
            return post.location.name

        # Fallback to neighborhood
        neighborhoods = [
            "Vesterbro",
            "Nørrebro",
            "Indre By",
            "Christiania",
            "Østerbro",
            "Islands Brygge",
            "Refshaleøen",
            "Nordhavn",
        ]

        for neighborhood in neighborhoods:
            if neighborhood.lower() in caption:
                return f"Event in {neighborhood}"

        return "Viral Copenhagen Event"

    def _extract_viral_event_data(
        self, post, venue_name: str, username: str, source_hashtag: str
    ) -> Optional[InstagramEvent]:
        """Extract event data from viral post with additional viral metadata."""

        # Use base extraction method
        event = self._extract_event_data(post, venue_name, username)

        if event:
            # Add viral-specific metadata
            event.detected_genres.append("viral")

            # Mark high-engagement content
            if post.likes > 1000:
                event.detected_genres.append("trending")

            # Add source information
            event.hashtags.append(f"source_{source_hashtag}")

            # Enhance title for viral content
            if "secret" in event.description.lower():
                event.title = f"🤫 {event.title}"
            elif "underground" in event.description.lower():
                event.title = f"🔥 {event.title}"
            elif post.likes > 1000:
                event.title = f"🌟 {event.title}"

        return event

    def find_trending_event_hashtags(self, days_back: int = 7) -> List[Tuple[str, int]]:
        """Find trending event-related hashtags in Copenhagen."""

        # This would analyze hashtag frequency and growth
        # Simplified implementation returning common viral hashtags

        trending_hashtags = [
            ("copenhagenevents", 1500),
            ("cphtonight", 1200),
            ("undergroundcph", 800),
            ("secretpartycopenhagen", 600),
            ("technocopenhagen", 900),
            ("vesterbrovibes", 400),
            ("warehousepartycopenhagen", 300),
            ("popupparty", 250),
        ]

        return trending_hashtags


def main():
    """Example usage of InstagramViralEventScraper."""

    scraper = InstagramViralEventScraper()

    # Find viral events
    viral_events = scraper.scrape_viral_events(
        days_back=7, max_posts_per_hashtag=20, min_engagement=100
    )

    print(f"Found {len(viral_events)} viral events:")
    for event in viral_events[:5]:
        print(f"- {event.title}")
        print(f"  @{event.venue_username} | {event.likes} likes")
        print(f"  Venue: {event.venue_name}")
        print(f"  Genres: {', '.join(event.detected_genres[:3])}")
        print()

    # Find trending hashtags
    trending = scraper.find_trending_event_hashtags()
    print(f"\nTrending event hashtags:")
    for hashtag, count in trending[:5]:
        print(f"- #{hashtag}: {count} posts")


if __name__ == "__main__":
    main()
