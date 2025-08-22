#!/usr/bin/env python3
"""
NLP utilities for better event data extraction using transformers.
Improves date/time extraction and artist/venue recognition.
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import logging
from dataclasses import dataclass

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, falling back to regex-based extraction")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedDateTime:
    """Extracted date/time information with confidence."""
    datetime: datetime
    confidence: float
    original_text: str

@dataclass
class ExtractedEntity:
    """Extracted named entity (artist, venue, etc.)."""
    text: str
    label: str
    confidence: float
    start: int
    end: int

class EventNLPExtractor:
    """Advanced NLP extraction for event information."""
    
    def __init__(self):
        self.ner_pipeline = None
        self.date_patterns = self._compile_date_patterns()
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use multilingual NER model that works with Danish
                self.ner_pipeline = pipeline(
                    "ner",
                    model="Babelscape/wikineural-multilingual-ner",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded multilingual NER model")
            except Exception as e:
                logger.warning(f"Failed to load NER model: {e}")
                self.ner_pipeline = None
    
    def extract_datetime(self, text: str) -> List[ExtractedDateTime]:
        """Extract date/time information from text using multiple approaches."""
        
        results = []
        
        # Try regex patterns first
        regex_results = self._extract_datetime_regex(text)
        results.extend(regex_results)
        
        # TODO: Add transformer-based temporal extraction
        # Could use models like "microsoft/DialoGPT-medium" fine-tuned for temporal extraction
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def extract_entities(self, text: str) -> Dict[str, List[ExtractedEntity]]:
        """Extract named entities (artists, venues, locations) from text."""
        
        entities = {
            'PERSON': [],  # Potential artists
            'ORG': [],     # Venues, organizations
            'LOC': [],     # Locations
            'MISC': []     # Other relevant entities
        }
        
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                
                for entity in ner_results:
                    label = entity['entity_group']
                    
                    extracted_entity = ExtractedEntity(
                        text=entity['word'],
                        label=label,
                        confidence=entity['score'],
                        start=entity['start'],
                        end=entity['end']
                    )
                    
                    if label in entities:
                        entities[label].append(extracted_entity)
                    else:
                        entities['MISC'].append(extracted_entity)
                        
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        # Fallback to regex-based extraction
        regex_entities = self._extract_entities_regex(text)
        for label, entity_list in regex_entities.items():
            entities[label].extend(entity_list)
        
        return entities
    
    def extract_artists(self, text: str, venue_context: str = "") -> List[str]:
        """Extract potential artist names from text."""
        
        entities = self.extract_entities(text)
        artists = []
        
        # Get persons from NER
        for person in entities['PERSON']:
            if person.confidence > 0.5:  # High confidence threshold
                artists.append(person.text)
        
        # Add regex-based extraction for names that might be missed
        regex_artists = self._extract_artists_regex(text, venue_context)
        artists.extend(regex_artists)
        
        # Remove duplicates and filter
        artists = list(set(artists))
        return self._filter_artist_names(artists)[:5]  # Top 5 candidates
    
    def extract_genres(self, text: str, hashtags: List[str] = None) -> List[str]:
        """Extract music genres from text and hashtags."""
        
        if hashtags is None:
            hashtags = []
        
        all_text = f"{text} {' '.join(hashtags)}".lower()
        
        # Comprehensive genre list
        genres = [
            # Electronic
            'techno', 'house', 'deep house', 'tech house', 'progressive house',
            'electronic', 'edm', 'trance', 'dubstep', 'drum and bass', 'dnb',
            'ambient', 'downtempo', 'chillout', 'minimal', 'electro',
            
            # Traditional
            'rock', 'pop', 'jazz', 'blues', 'classical', 'folk', 'country',
            'punk', 'metal', 'hardcore', 'indie', 'alternative',
            
            # Urban
            'hip hop', 'rap', 'r&b', 'soul', 'funk', 'reggae', 'ska',
            
            # Other
            'disco', 'latin', 'world', 'experimental', 'acoustic'
        ]
        
        found_genres = []
        for genre in genres:
            if genre in all_text:
                found_genres.append(genre)
        
        return list(set(found_genres))
    
    def _compile_date_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile regex patterns for date extraction."""
        
        patterns = [
            # Specific dates
            (re.compile(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', re.IGNORECASE), 'dd.mm.yyyy'),
            (re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})', re.IGNORECASE), 'dd/mm/yyyy'),
            (re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})', re.IGNORECASE), 'yyyy-mm-dd'),
            
            # Named dates (Danish)
            (re.compile(r'(\d{1,2})\.\s*(januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)', re.IGNORECASE), 'dd month'),
            
            # Named dates (English)
            (re.compile(r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)', re.IGNORECASE), 'dd month'),
            
            # Weekdays (Danish)
            (re.compile(r'\b(mandag|tirsdag|onsdag|torsdag|fredag|lørdag|søndag)\b', re.IGNORECASE), 'weekday_da'),
            
            # Weekdays (English)
            (re.compile(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.IGNORECASE), 'weekday_en'),
            
            # Relative dates
            (re.compile(r'\b(i dag|today)\b', re.IGNORECASE), 'today'),
            (re.compile(r'\b(i morgen|tomorrow)\b', re.IGNORECASE), 'tomorrow'),
            (re.compile(r'\b(i aften|tonight)\b', re.IGNORECASE), 'tonight'),
            (re.compile(r'\b(denne weekend|this weekend)\b', re.IGNORECASE), 'this_weekend'),
            
            # Times
            (re.compile(r'(\d{1,2}):(\d{2})', re.IGNORECASE), 'time'),
            (re.compile(r'(\d{1,2})\s*(pm|am)', re.IGNORECASE), 'time_ampm'),
        ]
        
        return patterns
    
    def _extract_datetime_regex(self, text: str) -> List[ExtractedDateTime]:
        """Extract dates using regex patterns."""
        
        results = []
        now = datetime.now()
        
        for pattern, pattern_type in self.date_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                try:
                    extracted_date = self._parse_date_match(match, pattern_type, now)
                    if extracted_date:
                        results.append(ExtractedDateTime(
                            datetime=extracted_date,
                            confidence=0.7,  # Medium confidence for regex
                            original_text=match.group(0)
                        ))
                except Exception as e:
                    logger.debug(f"Failed to parse date match: {e}")
                    continue
        
        return results
    
    def _parse_date_match(self, match: re.Match, pattern_type: str, now: datetime) -> Optional[datetime]:
        """Parse a regex match into a datetime object."""
        
        try:
            if pattern_type == 'dd.mm.yyyy':
                day, month, year = match.groups()
                return datetime(int(year), int(month), int(day))
            
            elif pattern_type == 'dd/mm/yyyy':
                day, month, year = match.groups()
                return datetime(int(year), int(month), int(day))
            
            elif pattern_type == 'yyyy-mm-dd':
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day))
            
            elif pattern_type in ['weekday_da', 'weekday_en']:
                weekday = match.group(1).lower()
                return self._get_next_weekday(weekday, now)
            
            elif pattern_type == 'today':
                return now.replace(hour=20, minute=0, second=0, microsecond=0)
            
            elif pattern_type == 'tomorrow':
                return now + timedelta(days=1)
            
            elif pattern_type == 'tonight':
                return now.replace(hour=20, minute=0, second=0, microsecond=0)
            
            elif pattern_type == 'this_weekend':
                days_until_friday = (4 - now.weekday()) % 7
                return now + timedelta(days=days_until_friday)
            
        except (ValueError, IndexError):
            return None
        
        return None
    
    def _get_next_weekday(self, weekday: str, from_date: datetime) -> datetime:
        """Get the next occurrence of a weekday."""
        
        weekday_map = {
            # Danish
            'mandag': 0, 'tirsdag': 1, 'onsdag': 2, 'torsdag': 3,
            'fredag': 4, 'lørdag': 5, 'søndag': 6,
            # English
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_weekday = weekday_map.get(weekday.lower())
        if target_weekday is None:
            return from_date
        
        days_ahead = (target_weekday - from_date.weekday()) % 7
        if days_ahead == 0:  # Today is the target weekday
            days_ahead = 7  # Get next week's occurrence
        
        return from_date + timedelta(days=days_ahead)
    
    def _extract_entities_regex(self, text: str) -> Dict[str, List[ExtractedEntity]]:
        """Fallback entity extraction using regex."""
        
        entities = {'PERSON': [], 'ORG': [], 'LOC': []}
        
        # Look for capitalized words that might be names
        name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        matches = name_pattern.finditer(text)
        
        for match in matches:
            entity = ExtractedEntity(
                text=match.group(0),
                label='PERSON',  # Default to person
                confidence=0.3,  # Low confidence for regex
                start=match.start(),
                end=match.end()
            )
            entities['PERSON'].append(entity)
        
        return entities
    
    def _extract_artists_regex(self, text: str, venue_context: str) -> List[str]:
        """Extract artist names using regex patterns."""
        
        # Look for patterns like "featuring X", "with X", "X live"
        artist_patterns = [
            r'featuring\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+live',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+concert',
        ]
        
        artists = []
        for pattern in artist_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                artist = match.group(1).strip()
                if len(artist) > 2:
                    artists.append(artist)
        
        return artists
    
    def _filter_artist_names(self, artists: List[str]) -> List[str]:
        """Filter out common words that are not artist names."""
        
        exclude_words = {
            'Live', 'Music', 'Concert', 'Event', 'Party', 'Tonight', 'Tomorrow',
            'Friday', 'Saturday', 'Sunday', 'Copenhagen', 'Denmark', 'Danish',
            'Fredag', 'Lørdag', 'Søndag', 'København', 'Danmark'
        }
        
        filtered = []
        for artist in artists:
            if artist not in exclude_words and len(artist) > 2:
                filtered.append(artist)
        
        return filtered

def main():
    """Example usage of EventNLPExtractor."""
    
    extractor = EventNLPExtractor()
    
    # Test text with Danish and English
    test_text = """
    Tonight featuring Kollektiv Turmstrasse live at Vega!
    Electronic techno night this Friday 15. december.
    Special guest DJ supporting with house music.
    """
    
    # Extract datetime
    dates = extractor.extract_datetime(test_text)
    print("Extracted dates:")
    for date in dates:
        print(f"- {date.datetime} (confidence: {date.confidence:.2f}) - '{date.original_text}'")
    
    # Extract artists
    artists = extractor.extract_artists(test_text, "Vega")
    print(f"\nExtracted artists: {artists}")
    
    # Extract genres
    genres = extractor.extract_genres(test_text, ["#techno", "#electronic"])
    print(f"Extracted genres: {genres}")

if __name__ == "__main__":
    main()