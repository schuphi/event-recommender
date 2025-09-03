#!/usr/bin/env python3
"""
Eventbrite API scraper for Copenhagen events.
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import h3
from dataclasses import dataclass
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EventbriteEvent:
    """Structured event data from Eventbrite."""

    id: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    price_min: Optional[float]
    price_max: Optional[float]
    venue_name: str
    venue_address: str
    venue_lat: float
    venue_lon: float
    url: str
    image_url: Optional[str]


class EventbriteScraper:
    """Scraper for Eventbrite events in Copenhagen area."""

    BASE_URL = "https://www.eventbriteapi.com/v3"

    # Copenhagen bounding box
    COPENHAGEN_BOUNDS = {
        "latitude": 55.6761,
        "longitude": 12.5683,
        "within": "25km",  # Radius around Copenhagen center
    }

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        )

    def search_events(
        self,
        categories: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 1000,
    ) -> List[EventbriteEvent]:
        """
        Search for events in Copenhagen.

        Args:
            categories: List of category IDs (e.g., ['103' for Music])
            start_date: Start date filter
            end_date: End date filter
            max_results: Maximum number of events to return
        """

        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=60)

        # Music and nightlife categories
        if not categories:
            categories = [
                "103",  # Music
                "110",  # Nightlife
                "105",  # Performing & Visual Arts
            ]

        params = {
            "location.latitude": self.COPENHAGEN_BOUNDS["latitude"],
            "location.longitude": self.COPENHAGEN_BOUNDS["longitude"],
            "location.within": self.COPENHAGEN_BOUNDS["within"],
            "start_date.range_start": start_date.isoformat(),
            "start_date.range_end": end_date.isoformat(),
            "categories": ",".join(categories),
            "sort_by": "date",
            "expand": "venue,ticket_availability,organizer",
            "page_size": min(50, max_results),
            "page": 1,
        }

        events = []
        total_fetched = 0

        while total_fetched < max_results:
            try:
                logger.info(f"Fetching Eventbrite page {params['page']}...")

                response = self.session.get(
                    f"{self.BASE_URL}/events/search/", params=params
                )
                response.raise_for_status()

                data = response.json()
                page_events = data.get("events", [])

                if not page_events:
                    break

                for event_data in page_events:
                    try:
                        event = self._parse_event(event_data)
                        if event:
                            events.append(event)
                            total_fetched += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse event {event_data.get('id', 'unknown')}: {e}"
                        )
                        continue

                # Check if there are more pages
                pagination = data.get("pagination", {})
                if pagination.get("has_more_items", False):
                    params["page"] += 1
                    time.sleep(0.5)  # Rate limiting
                else:
                    break

            except requests.RequestException as e:
                logger.error(f"API request failed: {e}")
                break

        logger.info(f"Fetched {len(events)} events from Eventbrite")
        return events

    def _parse_event(self, event_data: Dict) -> Optional[EventbriteEvent]:
        """Parse raw Eventbrite event data into structured format."""

        try:
            # Basic event info
            event_id = event_data["id"]
            title = event_data["name"]["text"]
            description = event_data.get("description", {}).get("text", "")

            # Dates
            start_time = datetime.fromisoformat(
                event_data["start"]["local"].replace("Z", "+00:00")
            )
            end_time = None
            if event_data.get("end", {}).get("local"):
                end_time = datetime.fromisoformat(
                    event_data["end"]["local"].replace("Z", "+00:00")
                )

            # Pricing
            price_min, price_max = self._extract_pricing(event_data)

            # Venue information
            venue = event_data.get("venue")
            if not venue:
                return None

            venue_name = venue.get("name", "Unknown Venue")
            venue_address = venue.get("address", {}).get("localized_area_display", "")

            # Coordinates
            venue_lat = venue.get("latitude")
            venue_lon = venue.get("longitude")

            if not venue_lat or not venue_lon:
                return None

            venue_lat = float(venue_lat)
            venue_lon = float(venue_lon)

            # URLs and images
            url = event_data["url"]
            image_url = None
            if event_data.get("logo"):
                image_url = event_data["logo"].get("url")

            return EventbriteEvent(
                id=event_id,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                price_min=price_min,
                price_max=price_max,
                venue_name=venue_name,
                venue_address=venue_address,
                venue_lat=venue_lat,
                venue_lon=venue_lon,
                url=url,
                image_url=image_url,
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing event: {e}")
            return None

    def _extract_pricing(
        self, event_data: Dict
    ) -> tuple[Optional[float], Optional[float]]:
        """Extract min/max pricing from event data."""

        try:
            if event_data.get("is_free", False):
                return 0.0, 0.0

            ticket_availability = event_data.get("ticket_availability")
            if not ticket_availability:
                return None, None

            prices = []
            for ticket_class in ticket_availability.get("ticket_classes", []):
                if ticket_class.get("cost"):
                    # Price in cents, convert to DKK (assuming USD, convert via ~6.5 rate)
                    price_usd = float(ticket_class["cost"]["display"]) / 100
                    price_dkk = price_usd * 6.5  # Rough conversion
                    prices.append(price_dkk)

            if prices:
                return min(prices), max(prices)

            return None, None

        except (KeyError, ValueError, TypeError):
            return None, None


def main():
    """Example usage of EventbriteScraper."""

    api_token = os.getenv("EVENTBRITE_API_TOKEN")
    if not api_token:
        logger.error("EVENTBRITE_API_TOKEN environment variable required")
        return

    scraper = EventbriteScraper(api_token)

    # Search for music events in next 30 days
    end_date = datetime.now() + timedelta(days=30)
    events = scraper.search_events(
        categories=["103"], end_date=end_date, max_results=100  # Music only
    )

    print(f"Found {len(events)} events:")
    for event in events[:5]:  # Show first 5
        print(f"- {event.title} at {event.venue_name} on {event.start_time.date()}")


if __name__ == "__main__":
    main()
