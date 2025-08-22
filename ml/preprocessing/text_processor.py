#!/usr/bin/env python3
"""
Text preprocessing utilities for event descriptions and metadata.
Handles multilingual text (Danish/English) and social media content.
"""

import re
import html
from typing import List, Dict, Optional, Tuple
import unicodedata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventTextProcessor:
    """Text preprocessing for event data."""
    
    def __init__(self):
        # Danish stopwords (common words to remove)
        self.danish_stopwords = {
            'og', 'i', 'det', 'at', 'en', 'til', 'er', 'som', 'på', 'de', 'med', 'han',
            'af', 'for', 'ikke', 'der', 'var', 'mig', 'sig', 'men', 'et', 'har', 'om',
            'vi', 'min', 'havde', 'ham', 'hun', 'nu', 'over', 'da', 'fra', 'du', 'ud',
            'sin', 'dem', 'os', 'op', 'man', 'hans', 'hvor', 'eller', 'hvad', 'skal',
            'selv', 'her', 'alle', 'vil', 'blev', 'kunne', 'ind', 'når', 'være', 'dog',
            'noget', 'bliver', 'år', 'før', 'første', 'samme', 'jeg', 'skulle', 'var'
        }
        
        # English stopwords (subset of common words)
        self.english_stopwords = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Social media patterns
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        
        # Event-specific patterns
        self.price_pattern = re.compile(r'(\d+)\s*(kr|dkk|euro|eur|$)', re.IGNORECASE)
        self.time_pattern = re.compile(r'(\d{1,2}):(\d{2})', re.IGNORECASE)
        
    def clean_event_text(
        self,
        text: str,
        remove_stopwords: bool = False,
        preserve_hashtags: bool = True,
        preserve_mentions: bool = False
    ) -> str:
        """
        Clean and normalize event text.
        
        Args:
            text: Raw text to clean
            remove_stopwords: Whether to remove stopwords
            preserve_hashtags: Whether to keep hashtags
            preserve_mentions: Whether to keep @mentions
        
        Returns:
            Cleaned text string
        """
        
        if not text:
            return ""
        
        # Start with the original text
        cleaned = text
        
        # HTML decode
        cleaned = html.unescape(cleaned)
        
        # Unicode normalization
        cleaned = unicodedata.normalize('NFKC', cleaned)
        
        # Remove URLs
        cleaned = self.url_pattern.sub('', cleaned)
        
        # Handle hashtags
        if preserve_hashtags:
            # Convert hashtags to regular words
            cleaned = re.sub(r'#(\w+)', r'\1', cleaned)
        else:
            # Remove hashtags completely
            cleaned = self.hashtag_pattern.sub('', cleaned)
        
        # Handle mentions
        if not preserve_mentions:
            cleaned = self.mention_pattern.sub('', cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[!@#$%^&*()_+={}\[\]|\\:";\'<>?,./~`-]{3,}', ' ', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Lowercase for consistency
        cleaned = cleaned.lower()
        
        # Remove stopwords if requested
        if remove_stopwords:
            cleaned = self._remove_stopwords(cleaned)
        
        return cleaned
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        if not text:
            return []
        
        hashtags = self.hashtag_pattern.findall(text)
        # Remove the # symbol and normalize
        return [tag[1:].lower() for tag in hashtags]
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        if not text:
            return []
        
        mentions = self.mention_pattern.findall(text)
        # Remove the @ symbol and normalize
        return [mention[1:].lower() for mention in mentions]
    
    def extract_prices(self, text: str) -> List[Tuple[int, str]]:
        """Extract price information from text."""
        if not text:
            return []
        
        prices = []
        matches = self.price_pattern.finditer(text)
        
        for match in matches:
            amount = int(match.group(1))
            currency = match.group(2).lower()
            prices.append((amount, currency))
        
        return prices
    
    def extract_times(self, text: str) -> List[str]:
        """Extract time mentions from text."""
        if not text:
            return []
        
        times = []
        matches = self.time_pattern.finditer(text)
        
        for match in matches:
            time_str = f"{match.group(1)}:{match.group(2)}"
            times.append(time_str)
        
        return times
    
    def normalize_venue_name(self, venue_name: str) -> str:
        """Normalize venue names for consistency."""
        if not venue_name:
            return ""
        
        # Remove common prefixes/suffixes
        normalized = venue_name.strip()
        
        # Remove "the" prefix
        if normalized.lower().startswith('the '):
            normalized = normalized[4:]
        
        # Remove location suffixes
        suffixes_to_remove = [
            ' copenhagen', ' cph', ' kb', ' københavn', ' denmark', ' dk'
        ]
        
        for suffix in suffixes_to_remove:
            if normalized.lower().endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Title case
        normalized = normalized.title()
        
        return normalized.strip()
    
    def normalize_artist_name(self, artist_name: str) -> str:
        """Normalize artist names for consistency."""
        if not artist_name:
            return ""
        
        # Basic cleaning
        normalized = artist_name.strip()
        
        # Remove extra quotes
        normalized = re.sub(r'^["\']|["\']$', '', normalized)
        
        # Handle "DJ " prefix
        if normalized.lower().startswith('dj '):
            # Keep DJ prefix but normalize
            normalized = 'DJ ' + normalized[3:].title()
        else:
            # Title case for regular artists
            normalized = normalized.title()
        
        return normalized.strip()
    
    def extract_genre_keywords(self, text: str) -> List[str]:
        """Extract potential music genre keywords from text."""
        if not text:
            return []
        
        # Comprehensive genre keywords
        genre_keywords = {
            # Electronic
            'techno', 'house', 'electronic', 'edm', 'trance', 'dubstep', 'dnb',
            'drum and bass', 'ambient', 'downtempo', 'minimal', 'electro',
            'deep house', 'tech house', 'progressive house', 'hardcore',
            
            # Traditional
            'rock', 'pop', 'jazz', 'blues', 'classical', 'folk', 'country',
            'punk', 'metal', 'indie', 'alternative', 'acoustic',
            
            # Urban
            'hip hop', 'rap', 'r&b', 'soul', 'funk', 'reggae', 'ska',
            
            # Other
            'disco', 'latin', 'world', 'experimental', 'singer-songwriter'
        }
        
        text_lower = text.lower()
        found_genres = []
        
        for genre in genre_keywords:
            if genre in text_lower:
                found_genres.append(genre)
        
        return found_genres
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove Danish and English stopwords from text."""
        
        words = text.split()
        all_stopwords = self.danish_stopwords | self.english_stopwords
        
        filtered_words = [
            word for word in words 
            if word.lower() not in all_stopwords and len(word) > 2
        ]
        
        return ' '.join(filtered_words)
    
    def process_social_media_post(self, post_text: str) -> Dict[str, any]:
        """
        Comprehensive processing of social media post text.
        
        Returns:
            Dictionary with extracted information
        """
        
        if not post_text:
            return {}
        
        result = {
            'cleaned_text': self.clean_event_text(post_text),
            'hashtags': self.extract_hashtags(post_text),
            'mentions': self.extract_mentions(post_text),
            'prices': self.extract_prices(post_text),
            'times': self.extract_times(post_text),
            'genres': self.extract_genre_keywords(post_text),
            'original_length': len(post_text),
            'cleaned_length': len(self.clean_event_text(post_text))
        }
        
        return result
    
    def prepare_for_embedding(self, event_data: Dict) -> str:
        """
        Prepare event data for text embedding generation.
        
        Args:
            event_data: Dictionary with event information
        
        Returns:
            Cleaned and formatted text for embedding
        """
        
        parts = []
        
        # Title
        title = event_data.get('title', '')
        if title:
            cleaned_title = self.clean_event_text(title, preserve_hashtags=True)
            parts.append(cleaned_title)
        
        # Description
        description = event_data.get('description', '')
        if description:
            cleaned_desc = self.clean_event_text(description, preserve_hashtags=True)
            # Truncate very long descriptions
            if len(cleaned_desc) > 300:
                cleaned_desc = cleaned_desc[:300] + "..."
            parts.append(cleaned_desc)
        
        # Venue
        venue = event_data.get('venue_name', '')
        if venue:
            normalized_venue = self.normalize_venue_name(venue)
            parts.append(f"venue: {normalized_venue}")
        
        # Artists
        artists = event_data.get('artists', [])
        if artists:
            normalized_artists = [self.normalize_artist_name(a) for a in artists[:3]]
            parts.append(f"artists: {', '.join(normalized_artists)}")
        
        # Genres
        genres = event_data.get('genres', [])
        if genres:
            parts.append(f"genres: {', '.join(genres)}")
        
        # Join all parts
        final_text = ' . '.join(filter(None, parts))
        
        return final_text

def main():
    """Example usage of EventTextProcessor."""
    
    processor = EventTextProcessor()
    
    # Test social media post processing
    test_post = """
    🎵 TONIGHT! Kollektiv Turmstrasse LIVE at Culture Box! 🔥
    Epic #techno night with underground vibes 💃
    Doors open 23:00 - 300kr entrance
    @culturebox_cph #copenhagennight #electronic
    https://example.com/tickets
    """
    
    result = processor.process_social_media_post(test_post)
    print("Social media post analysis:")
    for key, value in result.items():
        print(f"- {key}: {value}")
    
    # Test event text preparation
    event_data = {
        'title': 'Kollektiv Turmstrasse Live!!!',
        'description': 'Amazing techno night with the best underground electronic music in Copenhagen',
        'venue_name': 'Culture Box Copenhagen',
        'artists': ['Kollektiv Turmstrasse', 'DJ Support'],
        'genres': ['techno', 'electronic']
    }
    
    embedding_text = processor.prepare_for_embedding(event_data)
    print(f"\nPrepared for embedding:\n{embedding_text}")

if __name__ == "__main__":
    main()