#!/usr/bin/env python3
"""
Luma.com scraper for Copenhagen tech and startup events.

Luma has no official API, so this uses web scraping.
No API key required.

Luma is excellent for:
- Tech meetups and conferences
- Startup events
- AI/ML events
- Developer conferences
"""

import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LumaEvent:
    """Structured event data from Luma."""

    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    venue_name: str
    venue_address: Optional[str]
    url: str
    host_name: str
    image_url: Optional[str]
    is_free: bool


class LumaScraper:
    """
    Scraper for Luma events in Copenhagen.

    Uses web scraping since Luma has no public API.
    Focuses on tech/startup events.
    """

    # Luma search URL
    BASE_URL = "https://lu.ma"

    # Copenhagen-related search terms
    SEARCH_TERMS = [
        "copenhagen",
        "denmark",
        "kobenhavn",
        "cph",
    ]

    # Tech-related search terms to combine
    TECH_TERMS = [
        "tech",
        "startup",
        "ai",
        "developer",
        "hackathon",
        "web3",
        "crypto",
        "data",
        "engineering",
    ]

    def __init__(self):
        """Initialize Luma scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/json",
        })

    def search_events(
        self,
        max_results: int = 50,
    ) -> List[LumaEvent]:
        """
        Search for tech events in Copenhagen.

        Args:
            max_results: Maximum number of events to return

        Returns:
            List of LumaEvent objects
        """
        events = []
        seen_ids = set()

        # Search with different term combinations
        for location in self.SEARCH_TERMS[:2]:  # Limit to avoid rate limiting
            for tech_term in self.TECH_TERMS[:5]:
                if len(events) >= max_results:
                    break

                query = f"{tech_term} {location}"
                try:
                    found_events = self._search_query(query)

                    for event in found_events:
                        if event.id not in seen_ids:
                            seen_ids.add(event.id)
                            events.append(event)

                except Exception as e:
                    logger.warning(f"Error searching '{query}': {e}")

        logger.info(f"Found {len(events)} unique Luma events")
        return events[:max_results]

    def _search_query(self, query: str) -> List[LumaEvent]:
        """Search Luma for a specific query."""

        events = []

        try:
            # Luma's discover page with search
            url = f"{self.BASE_URL}/discover"
            params = {"q": query}

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Luma returned status {response.status_code}")
                return []

            # Try to find event data in the page
            # Luma uses Next.js and embeds JSON data
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for Next.js data script
            for script in soup.find_all('script', type='application/json'):
                try:
                    data = json.loads(script.string)
                    events.extend(self._extract_events_from_json(data))
                except:
                    pass

            # Also try to find event links directly
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if '/event/' in href or href.startswith('/e/'):
                    event_id = href.split('/')[-1]
                    if event_id and len(event_id) > 5:
                        # Fetch individual event
                        event = self._fetch_event_page(event_id)
                        if event:
                            events.append(event)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")

        return events

    def _extract_events_from_json(self, data: Dict) -> List[LumaEvent]:
        """Extract events from Luma's embedded JSON data."""
        events = []

        # Navigate the nested structure to find events
        # This structure may change - handle gracefully
        try:
            if isinstance(data, dict):
                # Look for events in various possible locations
                for key in ['events', 'featured', 'results', 'data']:
                    if key in data:
                        items = data[key]
                        if isinstance(items, list):
                            for item in items:
                                event = self._parse_event_json(item)
                                if event:
                                    events.append(event)
        except Exception as e:
            logger.debug(f"Error extracting from JSON: {e}")

        return events

    def _parse_event_json(self, data: Dict) -> Optional[LumaEvent]:
        """Parse event from JSON data."""

        try:
            # Extract basic info
            event_id = data.get('api_id') or data.get('id') or data.get('slug')
            if not event_id:
                return None

            title = data.get('name') or data.get('title')
            if not title:
                return None

            # Parse start time
            start_str = data.get('start_at') or data.get('startTime')
            if not start_str:
                return None

            start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))

            # End time
            end_time = None
            end_str = data.get('end_at') or data.get('endTime')
            if end_str:
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))

            # Venue
            venue_data = data.get('geo_address_info') or data.get('venue') or {}
            venue_name = venue_data.get('full_address') or venue_data.get('name') or "Online Event"
            venue_address = venue_data.get('address')

            # Host
            host = data.get('hosts', [{}])[0] if data.get('hosts') else {}
            host_name = host.get('name', '')

            # Image
            cover = data.get('cover_url') or data.get('coverUrl')

            # URL
            url = f"{self.BASE_URL}/event/{event_id}"

            return LumaEvent(
                id=str(event_id),
                title=title,
                description=data.get('description', ''),
                start_time=start_time,
                end_time=end_time,
                venue_name=venue_name,
                venue_address=venue_address,
                url=url,
                host_name=host_name,
                image_url=cover,
                is_free=True,  # Most Luma events are free
            )

        except Exception as e:
            logger.debug(f"Error parsing Luma event: {e}")
            return None

    def _fetch_event_page(self, event_id: str) -> Optional[LumaEvent]:
        """Fetch and parse a single event page."""

        try:
            url = f"{self.BASE_URL}/event/{event_id}"
            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to extract from JSON-LD
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string)
                    if data.get('@type') == 'Event':
                        return LumaEvent(
                            id=event_id,
                            title=data.get('name', ''),
                            description=data.get('description', ''),
                            start_time=datetime.fromisoformat(data.get('startDate', '').replace('Z', '+00:00')),
                            end_time=datetime.fromisoformat(data.get('endDate', '').replace('Z', '+00:00')) if data.get('endDate') else None,
                            venue_name=data.get('location', {}).get('name', 'Online Event'),
                            venue_address=data.get('location', {}).get('address', {}).get('streetAddress'),
                            url=url,
                            host_name=data.get('organizer', {}).get('name', ''),
                            image_url=data.get('image'),
                            is_free=True,
                        )
                except:
                    pass

            # Fallback: extract from meta tags
            title = soup.find('meta', property='og:title')
            if title:
                return LumaEvent(
                    id=event_id,
                    title=title.get('content', ''),
                    description=soup.find('meta', property='og:description').get('content', '') if soup.find('meta', property='og:description') else '',
                    start_time=datetime.now(),  # Fallback
                    end_time=None,
                    venue_name="See event page",
                    venue_address=None,
                    url=url,
                    host_name="",
                    image_url=soup.find('meta', property='og:image').get('content') if soup.find('meta', property='og:image') else None,
                    is_free=True,
                )

        except Exception as e:
            logger.debug(f"Error fetching event page: {e}")

        return None

    def to_normalized_events(self, events: List[LumaEvent]) -> List[Dict]:
        """
        Convert LumaEvents to normalized format for the pipeline.

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
                "source_id": event.id,
                "source_url": event.url,
                "price_min": 0,  # Most Luma events are free
                "price_max": None,
                "currency": "DKK",
                "image_url": event.image_url,
            })

        return normalized


def fetch_luma_events(max_results: int = 50) -> List[Dict]:
    """
    Convenience function to fetch Luma events.

    Returns normalized event dictionaries ready for pipeline.
    No API key required.
    """
    scraper = LumaScraper()
    events = scraper.search_events(max_results=max_results)
    return scraper.to_normalized_events(events)


if __name__ == "__main__":
    # Test the scraper
    print("Fetching Luma events (no API key needed)...")
    events = fetch_luma_events(max_results=10)
    print(f"Found {len(events)} events:")
    for event in events[:5]:
        print(f"  - {event['title']}")
        print(f"    {event['source_url']}")
