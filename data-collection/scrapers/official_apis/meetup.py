#!/usr/bin/env python3
"""
Meetup.com API scraper for Copenhagen tech and social events.

API Documentation: https://www.meetup.com/api/

To get an API key:
1. Go to https://www.meetup.com/api/oauth/list/
2. Create a new OAuth consumer
3. Use the API key in your .env file
"""

import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MeetupEvent:
    """Structured event data from Meetup."""

    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    venue_name: str
    venue_address: Optional[str]
    venue_lat: Optional[float]
    venue_lon: Optional[float]
    url: str
    group_name: str
    rsvp_count: int
    is_free: bool
    fee_amount: Optional[float]
    image_url: Optional[str]


class MeetupScraper:
    """
    Scraper for Meetup events in Copenhagen area.

    Meetup is excellent for:
    - Tech meetups and conferences
    - Startup events
    - Social/networking events
    - Learning workshops
    """

    # Meetup GraphQL API endpoint
    GRAPHQL_URL = "https://api.meetup.com/gql"

    # Copenhagen coordinates
    COPENHAGEN_LAT = 55.6761
    COPENHAGEN_LON = 12.5683
    RADIUS_MILES = 15  # ~25km

    # Categories relevant to our topics
    CATEGORIES = {
        "tech": [292],      # Technology
        "social": [402],    # Social
        "sports": [511],    # Sports & Fitness
        "music": [395],     # Music
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Meetup scraper.

        Args:
            api_key: Meetup API key. If None, tries to get from environment.
        """
        self.api_key = api_key or os.getenv("MEETUP_API_KEY")
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
            })

    def search_events(
        self,
        categories: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[MeetupEvent]:
        """
        Search for events in Copenhagen.

        Args:
            categories: List of category names ('tech', 'social', 'sports', 'music')
            start_date: Start date filter
            end_date: End date filter
            max_results: Maximum number of events to return

        Returns:
            List of MeetupEvent objects
        """
        if not self.api_key:
            logger.warning("No Meetup API key provided. Set MEETUP_API_KEY environment variable.")
            return []

        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=60)

        events = []

        # If no categories specified, search all
        if not categories:
            categories = list(self.CATEGORIES.keys())

        for category in categories:
            category_ids = self.CATEGORIES.get(category, [])
            if not category_ids:
                continue

            try:
                category_events = self._fetch_events_by_category(
                    category_ids,
                    start_date,
                    end_date,
                    max_results // len(categories)
                )
                events.extend(category_events)
                logger.info(f"Fetched {len(category_events)} events for category '{category}'")

            except Exception as e:
                logger.error(f"Error fetching {category} events: {e}")

        return events[:max_results]

    def _fetch_events_by_category(
        self,
        category_ids: List[int],
        start_date: datetime,
        end_date: datetime,
        limit: int
    ) -> List[MeetupEvent]:
        """Fetch events for specific category IDs using REST API."""

        events = []

        # Use the REST API endpoint for searching events
        # Note: The GraphQL API requires more complex auth
        url = "https://api.meetup.com/find/upcoming_events"

        params = {
            "lat": self.COPENHAGEN_LAT,
            "lon": self.COPENHAGEN_LON,
            "radius": self.RADIUS_MILES,
            "start_date_range": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date_range": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "page": min(limit, 200),
            "order": "time",
        }

        try:
            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 401:
                logger.error("Meetup API authentication failed. Check your API key.")
                return []

            if response.status_code != 200:
                logger.error(f"Meetup API error: {response.status_code} - {response.text}")
                return []

            data = response.json()
            raw_events = data.get("events", [])

            for event_data in raw_events:
                try:
                    event = self._parse_event(event_data)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Error parsing event: {e}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")

        return events

    def _parse_event(self, data: Dict) -> Optional[MeetupEvent]:
        """Parse raw API response into MeetupEvent."""

        try:
            # Extract venue info
            venue = data.get("venue", {})
            venue_name = venue.get("name", "Online Event")
            venue_address = None
            if venue.get("address_1"):
                parts = [venue.get("address_1")]
                if venue.get("city"):
                    parts.append(venue.get("city"))
                venue_address = ", ".join(parts)

            # Parse dates
            start_time = datetime.fromtimestamp(data.get("time", 0) / 1000)
            duration_ms = data.get("duration", 0)
            end_time = None
            if duration_ms:
                end_time = datetime.fromtimestamp((data.get("time", 0) + duration_ms) / 1000)

            # Fee info
            fee = data.get("fee", {})
            is_free = fee.get("amount", 0) == 0 if fee else True
            fee_amount = fee.get("amount") if fee else None

            # Image
            image_url = None
            if data.get("featured_photo"):
                image_url = data["featured_photo"].get("highres_link") or data["featured_photo"].get("photo_link")

            return MeetupEvent(
                id=str(data.get("id")),
                title=data.get("name", "Untitled Event"),
                description=data.get("description", ""),
                start_time=start_time,
                end_time=end_time,
                venue_name=venue_name,
                venue_address=venue_address,
                venue_lat=venue.get("lat"),
                venue_lon=venue.get("lon"),
                url=data.get("link", ""),
                group_name=data.get("group", {}).get("name", ""),
                rsvp_count=data.get("yes_rsvp_count", 0),
                is_free=is_free,
                fee_amount=fee_amount,
                image_url=image_url,
            )

        except Exception as e:
            logger.warning(f"Error parsing Meetup event: {e}")
            return None

    def to_normalized_events(self, events: List[MeetupEvent]) -> List[Dict]:
        """
        Convert MeetupEvents to normalized format for the pipeline.

        Returns list of dicts compatible with ingest_events().
        """
        normalized = []

        for event in events:
            normalized.append({
                "title": event.title,
                "description": event.description,
                "date_time": event.start_time,
                "end_date_time": event.end_time,
                "venue_name": event.venue_name,
                "venue_address": event.venue_address,
                "venue_lat": event.venue_lat,
                "venue_lon": event.venue_lon,
                "source_id": event.id,
                "source_url": event.url,
                "price_min": 0 if event.is_free else event.fee_amount,
                "price_max": event.fee_amount if not event.is_free else None,
                "currency": "DKK",
                "image_url": event.image_url,
            })

        return normalized


def fetch_meetup_events(api_key: Optional[str] = None, max_results: int = 100) -> List[Dict]:
    """
    Convenience function to fetch Meetup events.

    Returns normalized event dictionaries ready for pipeline.
    """
    scraper = MeetupScraper(api_key)
    events = scraper.search_events(max_results=max_results)
    return scraper.to_normalized_events(events)


if __name__ == "__main__":
    # Test the scraper
    api_key = os.getenv("MEETUP_API_KEY")

    if not api_key:
        print("Set MEETUP_API_KEY environment variable to test")
        print("Get your key at: https://www.meetup.com/api/")
    else:
        events = fetch_meetup_events(api_key, max_results=10)
        print(f"Found {len(events)} events:")
        for event in events[:5]:
            print(f"  - {event['title']} at {event['venue_name']}")
