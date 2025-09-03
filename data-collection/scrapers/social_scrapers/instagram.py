#!/usr/bin/env python3
"""
Instagram event scraper for Copenhagen nightlife venues.
Uses instaloader to scrape venue accounts for event posts.
"""

import instaloader
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InstagramEvent:
    """Structured event data extracted from Instagram posts."""

    id: str
    title: str
    description: str
    date_time: Optional[datetime]
    venue_name: str
    venue_username: str
    post_url: str
    image_url: str
    likes: int
    comments: int
    hashtags: List[str]
    detected_artists: List[str]
    detected_genres: List[str]


class InstagramEventScraper:
    """Scraper for events from Copenhagen venue Instagram accounts."""

    # Major Copenhagen venues and their Instagram handles
    COPENHAGEN_VENUES = {
        "vega_copenhagen": "Vega",
        "rust_cph": "Rust",
        "culturebox_cph": "Culture Box",
        "loppen_official": "Loppen",
        "jolene_cph": "Jolene",
        "kb18_cph": "KB18",
        "pumpehuset": "Pumpehuset",
        "amager_bio": "Amager Bio",
        "alice_cph": "ALICE",
        "beta2300": "BETA2300",
        "brus_kbh": "BRUS",
        "chateau_motel": "Chateau Motel",
        "den_groenne_koncertsal": "Den Grønne Koncertsal",
        "dronninglouisebridge": "Dronning Louise Bro",
        "huset_kbh": "HUSET",
        "ideal_bar": "Ideal Bar",
        "jazzhouse_copenhagen": "Jazzhouse",
        "kbhk": "Københavns Klub",
        "nebbiolo_wine_bar": "Nebbiolo",
        "penthouse_cph": "Penthouse",
        "reffen_copenhagen": "Reffen",
        "studenterhuset_cph": "Studenterhuset",
        "the_Jane": "The Jane",
        "warehouse9_cph": "Warehouse9",
    }

    # Event-related keywords and patterns (expanded for viral discovery)
    EVENT_KEYWORDS = {
        "english": [
            # Basic event types
            "concert",
            "live",
            "show",
            "event",
            "party",
            "night",
            "dj",
            "performance",
            # Viral/trending language
            "underground",
            "secret",
            "popup",
            "warehouse",
            "rooftop",
            "rave",
            "festival",
            "tonight",
            "happening",
            "dont miss",
            "viral",
            "trending",
            "epic",
            "insane",
            "legendary",
            "packed",
            "sold out",
            "queue",
            "hidden gem",
            "discovered",
            # Engagement language
            "everyone talking",
            "word of mouth",
            "invite only",
            "last minute",
            "spontaneous",
            "surprise",
            "crazy night",
            "best party",
            "unreal",
        ],
        "danish": [
            # Basic Danish event terms
            "koncert",
            "optræden",
            "fest",
            "aften",
            "nat",
            "live",
            "show",
            # Viral Danish terms
            "hemmelig",
            "underground",
            "popup",
            "lager",
            "tag",
            "rave",
            "i aften",
            "sker nu",
            "gå glip",
            "viral",
            "trending",
            "episk",
            "legendarisk",
            "proppet",
            "udsolgt",
            "kø",
            "skjult perle",
            "alle snakker",
            "mund til mund",
            "kun inviterede",
            "sidste øjeblik",
        ],
    }

    GENRE_KEYWORDS = [
        "techno",
        "house",
        "electronic",
        "disco",
        "funk",
        "soul",
        "jazz",
        "blues",
        "rock",
        "punk",
        "indie",
        "alternative",
        "pop",
        "hip hop",
        "rap",
        "r&b",
        "ambient",
        "experimental",
        "classical",
        "folk",
        "acoustic",
        "reggae",
    ]

    # Date patterns (Danish and English)
    DATE_PATTERNS = [
        r"(\d{1,2})\.(\d{1,2})\.(\d{4})",  # DD.MM.YYYY
        r"(\d{1,2})/(\d{1,2})/(\d{4})",  # DD/MM/YYYY
        r"(\d{1,2})\s+(januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)",  # Danish months
        r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)",  # English months
        r"(mandag|tirsdag|onsdag|torsdag|fredag|lørdag|søndag)",  # Danish weekdays
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",  # English weekdays
    ]

    def __init__(self, username: str = None, password: str = None):
        """
        Initialize scraper with optional login credentials.
        Login increases rate limits but may be detected.
        """
        self.loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_comments=True,
            save_metadata=False,
            compress_json=False,
        )

        if username and password:
            try:
                self.loader.login(username, password)
                logger.info("Successfully logged into Instagram")
            except Exception as e:
                logger.warning(f"Instagram login failed: {e}")

    def scrape_venues(
        self,
        venue_usernames: List[str] = None,
        days_back: int = 30,
        max_posts_per_venue: int = 50,
    ) -> List[InstagramEvent]:
        """
        Scrape events from venue Instagram accounts.

        Args:
            venue_usernames: List of Instagram usernames to scrape
            days_back: How many days back to look for posts
            max_posts_per_venue: Maximum posts to check per venue
        """

        if not venue_usernames:
            venue_usernames = list(self.COPENHAGEN_VENUES.keys())

        all_events = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for username in venue_usernames:
            logger.info(f"Scraping Instagram account: @{username}")

            try:
                profile = instaloader.Profile.from_username(
                    self.loader.context, username
                )
                venue_name = self.COPENHAGEN_VENUES.get(username, username)

                events = self._scrape_venue_posts(
                    profile, venue_name, cutoff_date, max_posts_per_venue
                )

                all_events.extend(events)
                logger.info(f"Found {len(events)} events from @{username}")

                # Rate limiting
                time.sleep(2)

            except instaloader.exceptions.ProfileNotExistsException:
                logger.warning(f"Instagram profile @{username} not found")
            except instaloader.exceptions.LoginRequiredException:
                logger.warning(f"Login required to access @{username}")
            except Exception as e:
                logger.error(f"Error scraping @{username}: {e}")
                continue

        logger.info(f"Total events found: {len(all_events)}")
        return all_events

    def _scrape_venue_posts(
        self,
        profile: instaloader.Profile,
        venue_name: str,
        cutoff_date: datetime,
        max_posts: int,
    ) -> List[InstagramEvent]:
        """Scrape event posts from a single venue profile."""

        events = []
        posts_checked = 0

        try:
            for post in profile.get_posts():
                if posts_checked >= max_posts:
                    break

                # Skip old posts
                if post.date < cutoff_date:
                    break

                posts_checked += 1

                # Check if post looks like an event
                if self._is_event_post(post):
                    event = self._extract_event_data(post, venue_name, profile.username)
                    if event:
                        events.append(event)

                # Rate limiting
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error getting posts from {profile.username}: {e}")

        return events

    def _is_event_post(self, post) -> bool:
        """Determine if an Instagram post is likely about an event."""

        caption = post.caption.lower() if post.caption else ""

        # Check for event keywords
        keywords_found = any(
            keyword in caption
            for keyword_list in self.EVENT_KEYWORDS.values()
            for keyword in keyword_list
        )

        # Check for date patterns
        date_patterns_found = any(
            re.search(pattern, caption, re.IGNORECASE) for pattern in self.DATE_PATTERNS
        )

        # Check for music/event hashtags
        hashtags = (
            [tag.lower() for tag in post.caption_hashtags]
            if post.caption_hashtags
            else []
        )
        event_hashtags = any(
            any(
                keyword in tag
                for keyword in self.EVENT_KEYWORDS["english"]
                + self.EVENT_KEYWORDS["danish"]
            )
            for tag in hashtags
        )

        return keywords_found or date_patterns_found or event_hashtags

    def _extract_event_data(
        self, post, venue_name: str, venue_username: str
    ) -> Optional[InstagramEvent]:
        """Extract structured event data from Instagram post."""

        try:
            caption = post.caption if post.caption else ""

            # Extract basic info
            event_id = f"ig_{post.shortcode}"
            post_url = f"https://www.instagram.com/p/{post.shortcode}/"

            # Get image URL
            image_url = post.url

            # Extract title (first line of caption or first sentence)
            title_lines = caption.split("\n")
            title = title_lines[0][:100] if title_lines else f"Event at {venue_name}"

            # Extract hashtags
            hashtags = list(post.caption_hashtags) if post.caption_hashtags else []

            # Try to extract date/time
            extracted_date = self._extract_date_from_text(caption)

            # Extract potential artist names (capitalized words, excluding common words)
            artists = self._extract_artists_from_text(caption)

            # Extract genres from text and hashtags
            genres = self._extract_genres_from_text(caption + " " + " ".join(hashtags))

            return InstagramEvent(
                id=event_id,
                title=title,
                description=caption,
                date_time=extracted_date,
                venue_name=venue_name,
                venue_username=venue_username,
                post_url=post_url,
                image_url=image_url,
                likes=post.likes,
                comments=post.comments,
                hashtags=hashtags,
                detected_artists=artists,
                detected_genres=genres,
            )

        except Exception as e:
            logger.warning(f"Error extracting event data: {e}")
            return None

    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Attempt to extract date/time information from text."""

        # This is a simplified implementation
        # A more robust version would use NLP libraries like spaCy

        text_lower = text.lower()

        # Look for today/tomorrow/this weekend
        now = datetime.now()
        if "tonight" in text_lower or "i aften" in text_lower:
            return now.replace(hour=20, minute=0, second=0, microsecond=0)
        elif "tomorrow" in text_lower or "i morgen" in text_lower:
            return now + timedelta(days=1)

        # Look for specific date patterns
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                # This would need more sophisticated date parsing
                # For now, return a placeholder
                return now + timedelta(days=7)  # Assume next week

        return None

    def _extract_artists_from_text(self, text: str) -> List[str]:
        """Extract potential artist names from post text."""

        # Look for capitalized words that might be artist names
        words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)

        # Filter out common words
        common_words = {
            "Instagram",
            "Copenhagen",
            "Denmark",
            "Friday",
            "Saturday",
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Tonight",
            "Tomorrow",
            "Event",
            "Concert",
            "Live",
            "Music",
            "Dance",
            "Party",
        }

        artists = [word for word in words if word not in common_words and len(word) > 2]

        return artists[:5]  # Limit to 5 potential artists

    def _extract_genres_from_text(self, text: str) -> List[str]:
        """Extract music genres mentioned in text."""

        text_lower = text.lower()
        genres = []

        for genre in self.GENRE_KEYWORDS:
            if genre in text_lower:
                genres.append(genre)

        return list(set(genres))  # Remove duplicates


def main():
    """Example usage of InstagramEventScraper."""

    scraper = InstagramEventScraper()

    # Scrape a few major venues
    test_venues = ["vega_copenhagen", "rust_cph", "culturebox_cph"]

    events = scraper.scrape_venues(
        venue_usernames=test_venues, days_back=14, max_posts_per_venue=20
    )

    print(f"Found {len(events)} events:")
    for event in events[:3]:
        print(f"- {event.title}")
        print(f"  @{event.venue_username} | {event.likes} likes")
        print(f"  Artists: {', '.join(event.detected_artists[:3])}")
        print(f"  Genres: {', '.join(event.detected_genres[:3])}")
        print()


if __name__ == "__main__":
    main()
