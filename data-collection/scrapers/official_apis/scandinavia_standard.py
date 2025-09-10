#!/usr/bin/env python3
"""
Scandinavia Standard event scraper for Copenhagen cultural events.
Scrapes curated monthly event roundups.
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
import logging
from urllib.parse import urljoin, urlparse
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScandinaviaStandardEvent:
    """Structured event data from Scandinavia Standard."""
    
    id: str
    title: str
    description: str
    date_time: Optional[datetime]
    end_date_time: Optional[datetime]
    venue_name: Optional[str]
    venue_address: Optional[str]
    price_info: Optional[str]
    event_type: Optional[str]
    source_url: str
    image_url: Optional[str]


class ScandinaviaStandardScraper:
    """Scraper for curated Copenhagen events from Scandinavia Standard."""
    
    BASE_URL = "https://scandinaviastandard.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    # Event type keywords for categorization
    EVENT_TYPES = {
        "music": ["concert", "live music", "jazz", "classical", "opera", "festival", "band", "orchestra"],
        "art": ["exhibition", "gallery", "museum", "art", "painting", "sculpture", "photography"],
        "theater": ["theater", "theatre", "play", "performance", "drama", "comedy"],
        "cultural": ["cultural", "heritage", "history", "tradition", "celebration"],
        "market": ["market", "flea", "vintage", "shopping", "craft"],
        "film": ["film", "cinema", "movie", "screening", "documentary"],
        "festival": ["festival", "fest", "celebration", "fair"],
        "nightlife": ["party", "club", "bar", "nightlife", "dj", "dance"]
    }
    
    # Month name mappings (English/Danish)
    MONTH_NAMES = {
        "january": 1, "januar": 1,
        "february": 2, "februar": 2,
        "march": 3, "marts": 3,
        "april": 4,
        "may": 5, "maj": 5,
        "june": 6, "juni": 6,
        "july": 7, "juli": 7,
        "august": 8,
        "september": 9,
        "october": 10, "oktober": 10,
        "november": 11,
        "december": 12
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def scrape_events(self, months_ahead: int = 3) -> List[ScandinaviaStandardEvent]:
        """
        Scrape events from Scandinavia Standard monthly roundups.
        
        Args:
            months_ahead: How many months ahead to scrape (current + future)
        """
        
        events = []
        current_date = datetime.now()
        
        for month_offset in range(months_ahead):
            target_date = current_date + timedelta(days=month_offset * 30)
            month_events = self._scrape_month_events(target_date.year, target_date.month)
            events.extend(month_events)
            
            # Be respectful with requests
            time.sleep(1)
        
        logger.info(f"Total events scraped from Scandinavia Standard: {len(events)}")
        return events
    
    def _scrape_month_events(self, year: int, month: int) -> List[ScandinaviaStandardEvent]:
        """Scrape events from a specific month's event roundup."""
        
        month_names = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
        month_name = month_names[month - 1]
        
        # Try common URL patterns
        url_patterns = [
            f"{self.BASE_URL}/whats-on-in-copenhagen-events-in-{month_name}-{year}/",
            f"{self.BASE_URL}/copenhagen-events-{month_name}-{year}/",
            f"{self.BASE_URL}/events-copenhagen-{month_name}-{year}/",
        ]
        
        for url in url_patterns:
            try:
                logger.info(f"Attempting to scrape: {url}")
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    events = self._parse_event_page(response.text, url)
                    if events:
                        logger.info(f"Found {len(events)} events for {month_name} {year}")
                        return events
                
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                continue
        
        logger.info(f"No event page found for {month_name} {year}")
        return []
    
    def _parse_event_page(self, html: str, source_url: str) -> List[ScandinaviaStandardEvent]:
        """Parse events from the HTML of an event roundup page."""
        
        soup = BeautifulSoup(html, 'html.parser')
        events = []
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'sidebar']):
            element.decompose()
        
        # Find the main content area
        main_content = self._find_main_content(soup)
        if not main_content:
            return events
        
        # Extract events using multiple strategies
        events.extend(self._extract_events_by_date_headers(main_content, source_url))
        events.extend(self._extract_events_by_text_blocks(main_content, source_url))
        
        # Remove duplicates based on title similarity
        events = self._deduplicate_events(events)
        
        return events
    
    def _find_main_content(self, soup: BeautifulSoup):
        """Find the main content area containing events."""
        
        # Try common content selectors
        selectors = [
            '.entry-content',
            '.post-content',
            '.article-content',
            '.content',
            'main',
            '.main',
            '#main',
            '[role="main"]'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                return content
        
        # Fallback to body
        return soup.find('body') or soup
    
    def _extract_events_by_date_headers(self, content, source_url: str) -> List[ScandinaviaStandardEvent]:
        """Extract events organized under date headers."""
        
        events = []
        current_date = None
        
        # Find all text nodes and headers
        elements = content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div'])
        
        for element in elements:
            text = element.get_text(strip=True)
            if not text or len(text) < 10:
                continue
            
            # Check if this is a date header
            date_match = self._extract_date_from_text(text)
            if date_match and self._is_likely_date_header(text):
                current_date = date_match
                continue
            
            # If we have a current date and this looks like an event
            if current_date and self._looks_like_event(text):
                event = self._create_event_from_text(text, current_date, source_url)
                if event:
                    events.append(event)
        
        return events
    
    def _extract_events_by_text_blocks(self, content, source_url: str) -> List[ScandinaviaStandardEvent]:
        """Extract events from text blocks with embedded dates."""
        
        events = []
        
        # Find paragraphs and divs that might contain events
        text_blocks = content.find_all(['p', 'div'], string=True)
        
        for block in text_blocks:
            text = block.get_text(strip=True)
            if len(text) < 50:  # Skip very short text
                continue
            
            # Check if this block contains event-like content
            if self._looks_like_event(text):
                # Try to extract date from within the text
                event_date = self._extract_date_from_text(text)
                
                event = self._create_event_from_text(text, event_date, source_url)
                if event:
                    events.append(event)
        
        return events
    
    def _extract_date_from_text(self, text: str) -> Optional[datetime]:
        """Extract date information from text."""
        
        text_lower = text.lower().strip()
        current_year = datetime.now().year
        
        # Pattern 1: "September 15" or "15 September"
        date_patterns = [
            r'(\d{1,2})\s+(?:of\s+)?([a-zA-Z]+)(?:\s+(\d{4}))?',  # 15 September 2025
            r'([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?',  # September 15th 2025
            r'(\d{1,2})[\.\/\-](\d{1,2})(?:[\.\/\-](\d{4}))?',  # 15.09.2025
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    groups = match.groups()
                    
                    # Handle different group arrangements
                    if groups[0].isdigit():  # Day first
                        day = int(groups[0])
                        month_str = groups[1]
                        year = int(groups[2]) if groups[2] else current_year
                    else:  # Month first
                        month_str = groups[0]
                        day = int(groups[1])
                        year = int(groups[2]) if groups[2] else current_year
                    
                    # Convert month name to number
                    month = self.MONTH_NAMES.get(month_str)
                    if month:
                        return datetime(year, month, day)
                    
                except (ValueError, IndexError):
                    continue
        
        # Pattern 2: Relative dates
        now = datetime.now()
        if 'today' in text_lower or 'tonight' in text_lower:
            return now
        elif 'tomorrow' in text_lower:
            return now + timedelta(days=1)
        elif 'this weekend' in text_lower:
            days_until_saturday = (5 - now.weekday()) % 7
            return now + timedelta(days=days_until_saturday)
        
        return None
    
    def _is_likely_date_header(self, text: str) -> bool:
        """Check if text is likely a date header."""
        
        text_lower = text.lower().strip()
        
        # Short text that starts with date patterns
        if len(text) < 50 and any(month in text_lower for month in self.MONTH_NAMES.keys()):
            return True
        
        # Contains weekday names
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                   'mandag', 'tirsdag', 'onsdag', 'torsdag', 'fredag', 'lørdag', 'søndag']
        if any(day in text_lower for day in weekdays):
            return True
        
        return False
    
    def _looks_like_event(self, text: str) -> bool:
        """Check if text describes an event."""
        
        text_lower = text.lower()
        
        # Must have minimum length
        if len(text) < 30:
            return False
        
        # Event indicators
        event_indicators = [
            'concert', 'exhibition', 'performance', 'show', 'festival', 'market',
            'screening', 'workshop', 'tour', 'opening', 'closing', 'premiere',
            'live', 'event', 'celebration', 'party', 'gathering', 'meeting',
            'koncert', 'udstilling', 'forestilling', 'festival', 'marked'
        ]
        
        # Location indicators
        location_indicators = [
            'at ', 'venue', 'location', 'address', 'copenhagen', 'københavn',
            'museum', 'gallery', 'theater', 'theatre', 'club', 'bar', 'hall',
            'center', 'centre', 'park', 'square', 'street'
        ]
        
        # Time indicators
        time_indicators = [
            'time', 'clock', 'am', 'pm', ':', 'from', 'until', 'to',
            'klokken', 'kl', 'fra', 'til'
        ]
        
        has_event = any(indicator in text_lower for indicator in event_indicators)
        has_location = any(indicator in text_lower for indicator in location_indicators)
        has_time = any(indicator in text_lower for indicator in time_indicators)
        
        return has_event and (has_location or has_time)
    
    def _create_event_from_text(self, text: str, event_date: Optional[datetime], source_url: str) -> Optional[ScandinaviaStandardEvent]:
        """Create an event object from extracted text."""
        
        if not text or len(text) < 20:
            return None
        
        # Extract title (first sentence or first line)
        sentences = re.split(r'[.!?]', text)
        title = sentences[0].strip()[:100] if sentences else text[:100]
        
        # Clean title
        title = re.sub(r'^[^\w]+', '', title)  # Remove leading non-word chars
        if not title:
            return None
        
        # Extract venue information
        venue_name = self._extract_venue_name(text)
        
        # Extract price information
        price_info = self._extract_price_info(text)
        
        # Determine event type
        event_type = self._categorize_event(text)
        
        # Generate unique ID
        event_id = f"ss_{hash(title + str(event_date) + (venue_name or ''))}"
        
        return ScandinaviaStandardEvent(
            id=event_id,
            title=title,
            description=text.strip(),
            date_time=event_date,
            end_date_time=None,
            venue_name=venue_name,
            venue_address=None,
            price_info=price_info,
            event_type=event_type,
            source_url=source_url,
            image_url=None
        )
    
    def _extract_venue_name(self, text: str) -> Optional[str]:
        """Extract venue name from event text."""
        
        # Common venue patterns
        venue_patterns = [
            r'at\s+([A-Z][^.!?]+?)(?:\s|,|\.)',  # "at Venue Name"
            r'venue:\s*([^.!?]+?)(?:\s|,|\.)',    # "Venue: Name"
            r'location:\s*([^.!?]+?)(?:\s|,|\.)',  # "Location: Name"
            r'(@\s*[A-Z][^.!?]+?)(?:\s|,|\.)',    # "@Venue Name"
        ]
        
        for pattern in venue_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                venue = match.group(1).strip()
                if len(venue) > 2 and len(venue) < 50:
                    return venue
        
        return None
    
    def _extract_price_info(self, text: str) -> Optional[str]:
        """Extract price information from text."""
        
        # Price patterns
        price_patterns = [
            r'(free|gratis)',
            r'(\d+\s*(?:kr|dkk|€|\$))',
            r'(price:\s*[^.!?]+)',
            r'(admission:\s*[^.!?]+)',
            r'(ticket[s]?:\s*[^.!?]+)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _categorize_event(self, text: str) -> Optional[str]:
        """Categorize event based on content."""
        
        text_lower = text.lower()
        
        for category, keywords in self.EVENT_TYPES.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "cultural"  # Default category
    
    def _deduplicate_events(self, events: List[ScandinaviaStandardEvent]) -> List[ScandinaviaStandardEvent]:
        """Remove duplicate events based on title similarity."""
        
        unique_events = []
        seen_titles = set()
        
        for event in events:
            # Create a normalized title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', event.title.lower()).strip()
            normalized_title = ' '.join(normalized_title.split())  # Normalize whitespace
            
            if normalized_title not in seen_titles and len(normalized_title) > 3:
                seen_titles.add(normalized_title)
                unique_events.append(event)
        
        return unique_events


def main():
    """Test the Scandinavia Standard scraper."""
    
    scraper = ScandinaviaStandardScraper()
    
    print("Testing Scandinavia Standard scraper...")
    events = scraper.scrape_events(months_ahead=2)
    
    print(f"\nFound {len(events)} events:")
    for i, event in enumerate(events[:5], 1):
        print(f"\n{i}. {event.title}")
        print(f"   Date: {event.date_time}")
        print(f"   Venue: {event.venue_name}")
        print(f"   Type: {event.event_type}")
        print(f"   Price: {event.price_info}")
        print(f"   Description: {event.description[:100]}...")


if __name__ == "__main__":
    main()