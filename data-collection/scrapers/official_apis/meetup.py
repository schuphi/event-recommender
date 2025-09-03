#!/usr/bin/env python3
"""
Meetup API scraper for Copenhagen events.
Note: Meetup deprecated their public API, using GraphQL endpoint used by their web app.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import logging

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
    venue_address: str
    venue_lat: float
    venue_lon: float
    url: str
    image_url: Optional[str]
    group_name: str
    attendee_count: int


class MeetupScraper:
    """Scraper for Meetup events in Copenhagen area."""

    # Using Meetup's internal GraphQL API
    GRAPHQL_URL = "https://www.meetup.com/gql"

    # Copenhagen coordinates
    COPENHAGEN_CENTER = (55.6761, 12.5683)
    SEARCH_RADIUS = 25  # km

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def search_events(
        self,
        topics: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None,
        max_results: int = 500,
    ) -> List[MeetupEvent]:
        """
        Search for events in Copenhagen.

        Args:
            topics: List of topic keywords (e.g., ['music', 'nightlife'])
            start_date: Start date filter
            end_date: End date filter
            max_results: Maximum number of events to return
        """

        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=60)

        if not topics:
            topics = ["music", "nightlife", "electronic", "concerts", "dancing", "dj"]

        all_events = []

        # Search by location
        events = self._search_by_location(start_date, end_date, max_results)
        all_events.extend(events)

        # Filter by topics if specified
        if topics:
            filtered_events = []
            for event in all_events:
                if self._matches_topics(event, topics):
                    filtered_events.append(event)
            all_events = filtered_events

        logger.info(f"Fetched {len(all_events)} events from Meetup")
        return all_events[:max_results]

    def _search_by_location(
        self, start_date: datetime, end_date: datetime, max_results: int
    ) -> List[MeetupEvent]:
        """Search events by geographic location."""

        # GraphQL query for events near Copenhagen
        query = """
        query($lat: Float!, $lon: Float!, $radius: Int!, $startDateTime: String, $endDateTime: String) {
          rankedEvents(
            filter: {
              lat: $lat
              lon: $lon
              radius: $radius
              startDateRange: $startDateTime
              endDateRange: $endDateTime
              source: EVENTS
              eventType: PHYSICAL
            }
          ) {
            edges {
              node {
                id
                title
                description
                dateTime
                endTime
                eventUrl
                images {
                  baseUrl
                }
                venue {
                  name
                  address
                  lat
                  lng
                }
                group {
                  name
                  urlname
                }
                going
                maxTickets
              }
            }
          }
        }
        """

        variables = {
            "lat": self.COPENHAGEN_CENTER[0],
            "lon": self.COPENHAGEN_CENTER[1],
            "radius": self.SEARCH_RADIUS,
            "startDateTime": start_date.isoformat(),
            "endDateTime": end_date.isoformat(),
        }

        events = []

        try:
            response = self.session.post(
                self.GRAPHQL_URL, json={"query": query, "variables": variables}
            )

            if response.status_code != 200:
                logger.warning(f"Meetup API returned status {response.status_code}")
                return events

            data = response.json()

            if "errors" in data:
                logger.warning(f"Meetup GraphQL errors: {data['errors']}")
                return events

            edges = data.get("data", {}).get("rankedEvents", {}).get("edges", [])

            for edge in edges:
                try:
                    event = self._parse_event(edge["node"])
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse Meetup event: {e}")
                    continue

        except requests.RequestException as e:
            logger.error(f"Meetup API request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

        return events

    def _parse_event(self, event_data: Dict) -> Optional[MeetupEvent]:
        """Parse raw Meetup event data into structured format."""

        try:
            event_id = event_data["id"]
            title = event_data.get("title", "Untitled Event")
            description = event_data.get("description", "")

            # Parse dates
            start_time = datetime.fromisoformat(
                event_data["dateTime"].replace("Z", "+00:00")
            )
            end_time = None
            if event_data.get("endTime"):
                end_time = datetime.fromisoformat(
                    event_data["endTime"].replace("Z", "+00:00")
                )

            # Venue information
            venue = event_data.get("venue", {})
            venue_name = venue.get("name", "TBD")
            venue_address = venue.get("address", "")
            venue_lat = venue.get("lat")
            venue_lon = venue.get("lng")

            if not venue_lat or not venue_lon:
                return None

            # URLs and images
            url = event_data.get("eventUrl", "")
            image_url = None
            images = event_data.get("images", [])
            if images:
                image_url = images[0].get("baseUrl")

            # Group info
            group = event_data.get("group", {})
            group_name = group.get("name", "Unknown Group")

            # Attendance
            attendee_count = event_data.get("going", 0)

            return MeetupEvent(
                id=event_id,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                venue_name=venue_name,
                venue_address=venue_address,
                venue_lat=float(venue_lat),
                venue_lon=float(venue_lon),
                url=url,
                image_url=image_url,
                group_name=group_name,
                attendee_count=attendee_count,
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing Meetup event: {e}")
            return None

    def _matches_topics(self, event: MeetupEvent, topics: List[str]) -> bool:
        """Check if event matches any of the specified topics."""

        text = f"{event.title} {event.description} {event.group_name}".lower()

        return any(topic.lower() in text for topic in topics)


def main():
    """Example usage of MeetupScraper."""

    scraper = MeetupScraper()

    # Search for music/nightlife events
    events = scraper.search_events(
        topics=["music", "electronic", "nightlife", "dj"], max_results=50
    )

    print(f"Found {len(events)} events:")
    for event in events[:5]:
        print(f"- {event.title} by {event.group_name} on {event.start_time.date()}")
        print(f"  {event.venue_name} - {event.attendee_count} going")


if __name__ == "__main__":
    main()
