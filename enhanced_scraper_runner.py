#!/usr/bin/env python3
"""
Enhanced scraper runner that combines venue website scraping with Eventbrite organization API.
This is the main scraper called by scheduler.py for comprehensive event collection.
"""

import os
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add the venue scrapers directory to Python path
sys.path.append(str(Path(__file__).parent / "data-collection" / "scrapers" / "venue_scrapers"))

from copenhagen_venues import CopenhagenVenueScraper, VenueEvent

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventbriteOrganizationScraper:
    """Scraper for specific Copenhagen organizations on Eventbrite."""

    def __init__(self):
        self.api_token = os.getenv('EVENTBRITE_API_TOKEN')
        self.base_url = 'https://www.eventbriteapi.com/v3'

        # Known Copenhagen venue organizations on Eventbrite
        # These would need to be populated with actual organization IDs
        self.copenhagen_organizations = {
            # Format: 'organization_name': 'organization_id'
            # Example: 'Vega Copenhagen': '123456789'
        }

    def scrape_organization_events(self, org_id: str, org_name: str) -> list:
        """Scrape events from a specific Eventbrite organization."""
        events = []

        if not self.api_token:
            logger.warning("No Eventbrite API token provided")
            return events

        try:
            url = f"{self.base_url}/organizations/{org_id}/events/"
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'Content-Type': 'application/json'
            }

            params = {
                'status': 'live,started,ended',
                'order_by': 'start_asc',
                'time_filter': 'current_future',
                'expand': 'venue,ticket_availability'
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            for event_data in data.get('events', []):
                try:
                    event = self._parse_eventbrite_event(event_data, org_name)
                    if event:
                        events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse Eventbrite event: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch events from organization {org_name}: {e}")

        return events

    def _parse_eventbrite_event(self, event_data: dict, org_name: str):
        """Parse Eventbrite event data into VenueEvent format."""
        try:
            # Extract basic info
            event_id = event_data.get('id')
            name = event_data.get('name', {}).get('text', '')
            description = event_data.get('description', {}).get('text', '')

            # Parse dates
            start_time = event_data.get('start', {}).get('local', '')
            end_time = event_data.get('end', {}).get('local', '')

            if start_time:
                start_datetime = datetime.fromisoformat(start_time.replace('Z', ''))
            else:
                return None

            end_datetime = None
            if end_time:
                end_datetime = datetime.fromisoformat(end_time.replace('Z', ''))

            # Extract venue info
            venue_data = event_data.get('venue')
            venue_name = venue_data.get('name', org_name) if venue_data else org_name
            venue_address = venue_data.get('address', {}).get('localized_address_display', '') if venue_data else ''

            # Default Copenhagen coordinates if no venue location
            lat = 55.6761
            lon = 12.5683

            if venue_data:
                lat = float(venue_data.get('latitude', lat))
                lon = float(venue_data.get('longitude', lon))

            # Extract pricing
            is_free = event_data.get('is_free', False)
            price_min = 0.0 if is_free else None
            price_max = None

            # Event URL
            event_url = event_data.get('url', '')

            return VenueEvent(
                title=name,
                description=description[:500] if description else name,  # Limit description length
                date_time=start_datetime,
                end_date_time=end_datetime,
                venue_name=venue_name,
                venue_address=venue_address,
                venue_lat=lat,
                venue_lon=lon,
                price_min=price_min,
                price_max=price_max,
                source=f'eventbrite_{org_name.lower().replace(" ", "_")}',
                ticket_url=event_url,
                artist=name
            )

        except Exception as e:
            logger.warning(f"Error parsing Eventbrite event: {e}")
            return None


def run_enhanced_scraping():
    """Run comprehensive event scraping from all sources."""
    logger.info("=== Starting Enhanced Event Scraping ===")

    # Initialize scrapers
    venue_scraper = CopenhagenVenueScraper()
    eventbrite_scraper = EventbriteOrganizationScraper()

    total_events = []

    # 1. Scrape venue websites
    logger.info("Scraping venue websites...")
    try:
        venue_events = venue_scraper.scrape_all_venues(days_ahead=90)
        total_events.extend(venue_events)
        logger.info(f"Found {len(venue_events)} events from venue websites")
    except Exception as e:
        logger.error(f"Failed to scrape venue websites: {e}")

    # 2. Scrape Eventbrite organizations (if API token available)
    logger.info("Scraping Eventbrite organizations...")
    eventbrite_total = 0

    for org_name, org_id in eventbrite_scraper.copenhagen_organizations.items():
        try:
            org_events = eventbrite_scraper.scrape_organization_events(org_id, org_name)
            total_events.extend(org_events)
            eventbrite_total += len(org_events)
            logger.info(f"Found {len(org_events)} events from {org_name}")
        except Exception as e:
            logger.warning(f"Failed to scrape {org_name}: {e}")

    logger.info(f"Found {eventbrite_total} events from Eventbrite organizations")

    # 3. Store all events in database
    if total_events:
        logger.info(f"Storing {len(total_events)} total events in database...")

        # Import the database storage class
        sys.path.append(str(Path(__file__).parent))
        from venue_scraper_runner import VenueScraperDatabase

        db = VenueScraperDatabase()
        result = db.store_scraped_events(total_events)
        return result
    else:
        logger.warning("No events found from any source")
        return {'success': False, 'error': 'No events found'}


def main():
    """Main entry point for enhanced scraping."""
    try:
        result = run_enhanced_scraping()

        if result['success']:
            print(f"SUCCESS: Enhanced scraping completed")
            print(f"- Stored: {result['stored_events']} new events")
            print(f"- Updated: {result['updated_events']} existing events")
            print(f"- Total database events: {result['total_events']}")

            # Show breakdown by source
            print("\nEvents by source:")
            for source, count in result['source_breakdown'].items():
                print(f"  - {source}: {count} events")

        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"FATAL ERROR in enhanced scraper: {e}")
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()