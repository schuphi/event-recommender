#!/usr/bin/env python3
"""
TicketMaster Discovery API scraper for Copenhagen concerts and sports events.

API Documentation: https://developer.ticketmaster.com/products-and-docs/apis/discovery-api/v2/

To get an API key:
1. Go to https://developer.ticketmaster.com/
2. Create an account and register an app
3. Use the Consumer Key as your API key
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
class TicketMasterEvent:
    """Structured event data from TicketMaster."""

    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    venue_name: str
    venue_address: Optional[str]
    venue_lat: Optional[float]
    venue_lon: Optional[float]
    venue_city: str
    url: str
    price_min: Optional[float]
    price_max: Optional[float]
    currency: str
    image_url: Optional[str]
    segment: str  # Music, Sports, Arts & Theatre, etc.
    genre: Optional[str]


class TicketMasterScraper:
    """
    Scraper for TicketMaster events in Copenhagen/Denmark.

    TicketMaster is excellent for:
    - Major concerts and festivals
    - Sports events (football, handball, etc.)
    - Theatre and arts events
    """

    BASE_URL = "https://app.ticketmaster.com/discovery/v2"

    # Denmark country code
    COUNTRY_CODE = "DK"

    # Copenhagen DMA (Designated Market Area) - use city for filtering
    CITY = "Copenhagen"

    # Segment IDs for filtering
    SEGMENTS = {
        "music": "KZFzniwnSyZfZ7v7nJ",
        "sports": "KZFzniwnSyZfZ7v7nE",
        "arts": "KZFzniwnSyZfZ7v7na",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TicketMaster scraper.

        Args:
            api_key: TicketMaster API key. If None, tries to get from environment.
        """
        self.api_key = api_key or os.getenv("TICKETMASTER_API_KEY")
        self.session = requests.Session()

    def search_events(
        self,
        segments: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[TicketMasterEvent]:
        """
        Search for events in Copenhagen.

        Args:
            segments: List of segment names ('music', 'sports', 'arts')
            start_date: Start date filter
            end_date: End date filter
            max_results: Maximum number of events to return

        Returns:
            List of TicketMasterEvent objects
        """
        if not self.api_key:
            logger.warning("No TicketMaster API key provided. Set TICKETMASTER_API_KEY environment variable.")
            return []

        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=90)

        events = []

        # If no segments specified, search all
        if not segments:
            segments = list(self.SEGMENTS.keys())

        for segment in segments:
            segment_id = self.SEGMENTS.get(segment)
            if not segment_id:
                continue

            try:
                segment_events = self._fetch_events(
                    segment_id,
                    start_date,
                    end_date,
                    max_results // len(segments)
                )
                events.extend(segment_events)
                logger.info(f"Fetched {len(segment_events)} events for segment '{segment}'")

            except Exception as e:
                logger.error(f"Error fetching {segment} events: {e}")

        # Remove duplicates by ID
        seen_ids = set()
        unique_events = []
        for event in events:
            if event.id not in seen_ids:
                seen_ids.add(event.id)
                unique_events.append(event)

        return unique_events[:max_results]

    def _fetch_events(
        self,
        segment_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int
    ) -> List[TicketMasterEvent]:
        """Fetch events for specific segment."""

        events = []
        page = 0
        page_size = min(limit, 200)  # API max is 200

        while len(events) < limit:
            params = {
                "apikey": self.api_key,
                "countryCode": self.COUNTRY_CODE,
                "city": self.CITY,
                "segmentId": segment_id,
                "startDateTime": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endDateTime": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size": page_size,
                "page": page,
                "sort": "date,asc",
            }

            try:
                response = self.session.get(
                    f"{self.BASE_URL}/events.json",
                    params=params,
                    timeout=30
                )

                if response.status_code == 401:
                    logger.error("TicketMaster API authentication failed. Check your API key.")
                    break

                if response.status_code == 429:
                    logger.warning("TicketMaster API rate limit reached")
                    break

                if response.status_code != 200:
                    logger.error(f"TicketMaster API error: {response.status_code}")
                    break

                data = response.json()

                # Check if we have events
                if "_embedded" not in data or "events" not in data["_embedded"]:
                    break

                raw_events = data["_embedded"]["events"]

                for event_data in raw_events:
                    try:
                        event = self._parse_event(event_data)
                        if event:
                            events.append(event)
                    except Exception as e:
                        logger.warning(f"Error parsing event: {e}")

                # Check if there are more pages
                page_info = data.get("page", {})
                total_pages = page_info.get("totalPages", 1)

                if page >= total_pages - 1:
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                break

        return events

    def _parse_event(self, data: Dict) -> Optional[TicketMasterEvent]:
        """Parse raw API response into TicketMasterEvent."""

        try:
            # Extract venue info
            venues = data.get("_embedded", {}).get("venues", [])
            venue = venues[0] if venues else {}

            venue_name = venue.get("name", "Unknown Venue")
            venue_address = None
            if venue.get("address", {}).get("line1"):
                venue_address = venue["address"]["line1"]
                if venue.get("city", {}).get("name"):
                    venue_address += f", {venue['city']['name']}"

            venue_lat = None
            venue_lon = None
            if venue.get("location"):
                venue_lat = float(venue["location"].get("latitude", 0))
                venue_lon = float(venue["location"].get("longitude", 0))

            venue_city = venue.get("city", {}).get("name", "Copenhagen")

            # Parse dates
            dates = data.get("dates", {}).get("start", {})
            start_time = None
            if dates.get("dateTime"):
                start_time = datetime.fromisoformat(dates["dateTime"].replace("Z", "+00:00"))
            elif dates.get("localDate"):
                start_time = datetime.strptime(dates["localDate"], "%Y-%m-%d")

            if not start_time:
                return None

            # Price info
            price_ranges = data.get("priceRanges", [])
            price_min = None
            price_max = None
            currency = "DKK"

            if price_ranges:
                price_min = price_ranges[0].get("min")
                price_max = price_ranges[0].get("max")
                currency = price_ranges[0].get("currency", "DKK")

            # Image
            images = data.get("images", [])
            image_url = None
            if images:
                # Get highest resolution image
                sorted_images = sorted(images, key=lambda x: x.get("width", 0), reverse=True)
                image_url = sorted_images[0].get("url")

            # Classification (segment/genre)
            classifications = data.get("classifications", [])
            segment = "Other"
            genre = None
            if classifications:
                segment = classifications[0].get("segment", {}).get("name", "Other")
                genre = classifications[0].get("genre", {}).get("name")

            # Description
            description = data.get("info", "") or data.get("pleaseNote", "") or ""

            return TicketMasterEvent(
                id=data.get("id"),
                title=data.get("name", "Untitled Event"),
                description=description,
                start_time=start_time,
                end_time=None,  # TicketMaster doesn't always provide end time
                venue_name=venue_name,
                venue_address=venue_address,
                venue_lat=venue_lat,
                venue_lon=venue_lon,
                venue_city=venue_city,
                url=data.get("url", ""),
                price_min=price_min,
                price_max=price_max,
                currency=currency,
                image_url=image_url,
                segment=segment,
                genre=genre,
            )

        except Exception as e:
            logger.warning(f"Error parsing TicketMaster event: {e}")
            return None

    def to_normalized_events(self, events: List[TicketMasterEvent]) -> List[Dict]:
        """
        Convert TicketMasterEvents to normalized format for the pipeline.

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
                "price_min": event.price_min,
                "price_max": event.price_max,
                "currency": event.currency,
                "image_url": event.image_url,
            })

        return normalized


def fetch_ticketmaster_events(api_key: Optional[str] = None, max_results: int = 100) -> List[Dict]:
    """
    Convenience function to fetch TicketMaster events.

    Returns normalized event dictionaries ready for pipeline.
    """
    scraper = TicketMasterScraper(api_key)
    events = scraper.search_events(max_results=max_results)
    return scraper.to_normalized_events(events)


if __name__ == "__main__":
    # Test the scraper
    api_key = os.getenv("TICKETMASTER_API_KEY")

    if not api_key:
        print("Set TICKETMASTER_API_KEY environment variable to test")
        print("Get your key at: https://developer.ticketmaster.com/")
    else:
        events = fetch_ticketmaster_events(api_key, max_results=10)
        print(f"Found {len(events)} events:")
        for event in events[:5]:
            print(f"  - {event['title']} at {event['venue_name']}")
            if event.get('price_min'):
                print(f"    Price: {event['price_min']}-{event['price_max']} {event['currency']}")
