#!/usr/bin/env python3
"""
Artist and genre enrichment system for event data.
Uses multiple APIs to enhance event data with accurate artist and genre information.
"""

import re
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from urllib.parse import quote_plus
import difflib
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArtistInfo:
    """Enhanced artist information."""
    name: str
    canonical_name: str
    genres: List[str]
    popularity_score: float
    spotify_id: Optional[str] = None
    lastfm_id: Optional[str] = None
    image_url: Optional[str] = None
    bio: Optional[str] = None
    similar_artists: List[str] = None

@dataclass
class EnrichmentResult:
    """Result of artist/genre enrichment."""
    original_artists: List[str]
    enriched_artists: List[ArtistInfo]
    enhanced_genres: List[str]
    confidence_score: float
    sources_used: List[str]

class ArtistGenreEnricher:
    """Multi-source artist and genre enrichment system."""
    
    def __init__(self, 
                 spotify_client_id: Optional[str] = None,
                 spotify_client_secret: Optional[str] = None,
                 lastfm_api_key: Optional[str] = None,
                 cache_dir: str = "data-collection/cache"):
        """
        Initialize enrichment system.
        
        Args:
            spotify_client_id: Spotify API client ID
            spotify_client_secret: Spotify API client secret  
            lastfm_api_key: Last.fm API key
            cache_dir: Directory for caching API responses
        """
        
        self.spotify_client_id = spotify_client_id
        self.spotify_client_secret = spotify_client_secret
        self.lastfm_api_key = lastfm_api_key
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API sessions
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Copenhagen-Event-Recommender/1.0'
        })
        
        # Authentication tokens
        self.spotify_token = None
        self.spotify_token_expires = None
        
        # Caches
        self.artist_cache = self._load_cache('artist_cache.pkl')
        self.genre_cache = self._load_cache('genre_cache.pkl')
        
        # Artist name variations and mappings
        self.artist_mappings = self._load_artist_mappings()
        
        # Genre classification system
        self.genre_classifier = GenreClassifier()
        
        # Rate limiting
        self.last_spotify_request = 0
        self.last_lastfm_request = 0
        self.spotify_rate_limit = 0.1  # 10 requests per second
        self.lastfm_rate_limit = 0.2   # 5 requests per second
    
    def enrich_event_artists(self, 
                           event_data: Dict,
                           deep_enrichment: bool = True) -> EnrichmentResult:
        """
        Enrich event with detailed artist and genre information.
        
        Args:
            event_data: Event data containing artists list
            deep_enrichment: Whether to perform deep API lookups
            
        Returns:
            EnrichmentResult with enhanced artist/genre data
        """
        
        original_artists = event_data.get('artists', [])
        if not original_artists:
            # Try to extract artists from title or description
            original_artists = self._extract_artists_from_text(event_data)
        
        if not original_artists:
            return EnrichmentResult(
                original_artists=[],
                enriched_artists=[],
                enhanced_genres=[],
                confidence_score=0.0,
                sources_used=[]
            )
        
        logger.info(f"Enriching {len(original_artists)} artists: {original_artists}")
        
        enriched_artists = []
        sources_used = set()
        
        for artist_name in original_artists:
            if not artist_name.strip():
                continue
            
            # Clean and normalize artist name
            cleaned_name = self._clean_artist_name(artist_name)
            canonical_name = self._get_canonical_artist_name(cleaned_name)
            
            # Check cache first
            cache_key = canonical_name.lower()
            if cache_key in self.artist_cache:
                logger.info(f"Using cached data for {canonical_name}")
                enriched_artists.append(self.artist_cache[cache_key])
                continue
            
            # Enrich artist information
            artist_info = None
            
            if deep_enrichment:
                # Try Spotify first
                if self.spotify_client_id and self.spotify_client_secret:
                    artist_info = self._enrich_with_spotify(canonical_name)
                    if artist_info:
                        sources_used.add('spotify')
                
                # Try Last.fm if Spotify fails or as supplement
                if not artist_info and self.lastfm_api_key:
                    artist_info = self._enrich_with_lastfm(canonical_name)
                    if artist_info:
                        sources_used.add('lastfm')
            
            # Fallback: create basic artist info
            if not artist_info:
                artist_info = self._create_basic_artist_info(canonical_name)
                sources_used.add('basic')
            
            # Cache the result
            self.artist_cache[cache_key] = artist_info
            enriched_artists.append(artist_info)
        
        # Enhance genres from all artists
        enhanced_genres = self._enhance_genres(enriched_artists, event_data)
        
        # Calculate confidence score
        confidence_score = self._calculate_enrichment_confidence(
            enriched_artists, sources_used
        )
        
        # Save cache
        self._save_cache('artist_cache.pkl', self.artist_cache)
        
        return EnrichmentResult(
            original_artists=original_artists,
            enriched_artists=enriched_artists,
            enhanced_genres=enhanced_genres,
            confidence_score=confidence_score,
            sources_used=list(sources_used)
        )
    
    def _extract_artists_from_text(self, event_data: Dict) -> List[str]:
        """Extract artist names from title or description."""
        
        artists = []
        
        # Check title and description for artist indicators
        text_sources = [
            event_data.get('title', ''),
            event_data.get('description', '')
        ]
        
        for text in text_sources:
            if not text:
                continue
            
            # Common patterns for artist mentions
            patterns = [
                r'featuring\s+([^,\n]+)',
                r'ft\.?\s+([^,\n]+)',
                r'with\s+([^,\n]+)',
                r'presents?\s+([^,\n]+)',
                r'live:\s*([^,\n]+)',
                r'dj\s+([^,\n]+)',
                r'by\s+([^,\n]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Clean up the match
                    artist_name = re.sub(r'\s*[(&].*$', '', match.strip())
                    if len(artist_name) > 2 and len(artist_name) < 50:
                        artists.append(artist_name)
        
        return list(set(artists))
    
    def _clean_artist_name(self, artist_name: str) -> str:
        """Clean and normalize artist name."""
        
        # Remove common suffixes and prefixes
        cleaned = artist_name.strip()
        
        # Remove common prefixes
        prefixes = ['DJ ', 'dj ', 'Dj ', 'MC ', 'mc ', 'Live: ', 'live: ']
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove common suffixes
        suffixes = [' (Live)', ' (DJ Set)', ' (live)', ' (dj set)', ' LIVE', ' live']
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _get_canonical_artist_name(self, artist_name: str) -> str:
        """Get canonical artist name from mappings."""
        
        artist_lower = artist_name.lower()
        
        # Check exact mappings
        if artist_lower in self.artist_mappings:
            return self.artist_mappings[artist_lower]
        
        # Check fuzzy matches
        for mapped_name, canonical in self.artist_mappings.items():
            if difflib.SequenceMatcher(None, artist_lower, mapped_name).ratio() > 0.9:
                return canonical
        
        return artist_name
    
    def _enrich_with_spotify(self, artist_name: str) -> Optional[ArtistInfo]:
        """Enrich artist using Spotify API."""
        
        try:
            # Get access token
            if not self._ensure_spotify_token():
                return None
            
            # Rate limiting
            self._rate_limit_spotify()
            
            # Search for artist
            search_url = "https://api.spotify.com/v1/search"
            params = {
                'q': artist_name,
                'type': 'artist',
                'limit': 1
            }
            
            headers = {'Authorization': f'Bearer {self.spotify_token}'}
            response = self.session.get(search_url, params=params, headers=headers)
            
            if response.status_code != 200:
                logger.warning(f"Spotify API error for {artist_name}: {response.status_code}")
                return None
            
            data = response.json()
            artists = data.get('artists', {}).get('items', [])
            
            if not artists:
                logger.info(f"No Spotify results for {artist_name}")
                return None
            
            artist_data = artists[0]
            
            # Extract information
            spotify_id = artist_data['id']
            canonical_name = artist_data['name']
            genres = artist_data.get('genres', [])
            popularity = artist_data.get('popularity', 0) / 100.0  # Normalize to 0-1
            
            image_url = None
            if artist_data.get('images'):
                image_url = artist_data['images'][0]['url']
            
            return ArtistInfo(
                name=artist_name,
                canonical_name=canonical_name,
                genres=genres,
                popularity_score=popularity,
                spotify_id=spotify_id,
                image_url=image_url
            )
            
        except Exception as e:
            logger.error(f"Spotify enrichment failed for {artist_name}: {e}")
            return None
    
    def _enrich_with_lastfm(self, artist_name: str) -> Optional[ArtistInfo]:
        """Enrich artist using Last.fm API."""
        
        try:
            # Rate limiting
            self._rate_limit_lastfm()
            
            # Get artist info
            url = "http://ws.audioscrobbler.com/2.0/"
            params = {
                'method': 'artist.getInfo',
                'artist': artist_name,
                'api_key': self.lastfm_api_key,
                'format': 'json'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Last.fm API error for {artist_name}: {response.status_code}")
                return None
            
            data = response.json()
            
            if 'error' in data:
                logger.info(f"Last.fm error for {artist_name}: {data['message']}")
                return None
            
            artist_data = data.get('artist', {})
            
            # Extract information
            canonical_name = artist_data.get('name', artist_name)
            bio = ''
            if artist_data.get('bio', {}).get('summary'):
                bio = artist_data['bio']['summary']
            
            # Get top tags as genres
            genres = []
            if artist_data.get('tags', {}).get('tag'):
                tags = artist_data['tags']['tag']
                if isinstance(tags, list):
                    genres = [tag['name'] for tag in tags[:5]]
                else:
                    genres = [tags['name']]
            
            # Calculate popularity from stats
            popularity = 0.0
            if artist_data.get('stats'):
                listeners = int(artist_data['stats'].get('listeners', 0))
                playcount = int(artist_data['stats'].get('playcount', 0))
                # Rough popularity calculation
                popularity = min(1.0, (listeners + playcount / 100) / 100000)
            
            return ArtistInfo(
                name=artist_name,
                canonical_name=canonical_name,
                genres=self.genre_classifier.clean_genres(genres),
                popularity_score=popularity,
                lastfm_id=canonical_name,
                bio=bio
            )
            
        except Exception as e:
            logger.error(f"Last.fm enrichment failed for {artist_name}: {e}")
            return None
    
    def _create_basic_artist_info(self, artist_name: str) -> ArtistInfo:
        """Create basic artist info when APIs fail."""
        
        # Attempt to classify genres from name patterns
        inferred_genres = self.genre_classifier.infer_genres_from_name(artist_name)
        
        return ArtistInfo(
            name=artist_name,
            canonical_name=artist_name,
            genres=inferred_genres,
            popularity_score=0.1  # Low default popularity
        )
    
    def _enhance_genres(self, 
                       enriched_artists: List[ArtistInfo], 
                       event_data: Dict) -> List[str]:
        """Enhance genres from artists and event context."""
        
        all_genres = []
        
        # Collect genres from artists
        for artist in enriched_artists:
            all_genres.extend(artist.genres)
        
        # Extract genres from event text
        text_genres = self.genre_classifier.extract_genres_from_text(
            event_data.get('title', '') + ' ' + event_data.get('description', '')
        )
        all_genres.extend(text_genres)
        
        # Count and prioritize genres
        genre_counts = Counter(all_genres)
        
        # Get most common genres, but also include rare specific genres
        enhanced_genres = []
        
        # Add most common genres first
        for genre, count in genre_counts.most_common(10):
            enhanced_genres.append(genre)
        
        # Clean and standardize genres
        enhanced_genres = self.genre_classifier.clean_and_standardize_genres(enhanced_genres)
        
        return enhanced_genres[:8]  # Limit to 8 genres max
    
    def _calculate_enrichment_confidence(self, 
                                       artists: List[ArtistInfo], 
                                       sources: Set[str]) -> float:
        """Calculate confidence score for enrichment."""
        
        if not artists:
            return 0.0
        
        # Base confidence from data sources
        source_confidence = 0.3
        if 'spotify' in sources:
            source_confidence += 0.4
        if 'lastfm' in sources:
            source_confidence += 0.2
        if 'basic' in sources:
            source_confidence += 0.1
        
        # Artist data quality
        avg_popularity = sum(artist.popularity_score for artist in artists) / len(artists)
        quality_confidence = avg_popularity * 0.3
        
        # Genre richness
        total_genres = sum(len(artist.genres) for artist in artists)
        genre_confidence = min(0.2, total_genres * 0.02)
        
        return min(1.0, source_confidence + quality_confidence + genre_confidence)
    
    def _ensure_spotify_token(self) -> bool:
        """Ensure valid Spotify access token."""
        
        if (self.spotify_token and self.spotify_token_expires and 
            datetime.now() < self.spotify_token_expires):
            return True
        
        # Get new token
        try:
            auth_url = "https://accounts.spotify.com/api/token"
            
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.spotify_client_id,
                'client_secret': self.spotify_client_secret
            }
            
            response = self.session.post(auth_url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                self.spotify_token = token_data['access_token']
                expires_in = token_data['expires_in']
                self.spotify_token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
                return True
            else:
                logger.error(f"Spotify auth failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Spotify token request failed: {e}")
            return False
    
    def _rate_limit_spotify(self):
        """Apply rate limiting for Spotify API."""
        now = time.time()
        time_since_last = now - self.last_spotify_request
        if time_since_last < self.spotify_rate_limit:
            time.sleep(self.spotify_rate_limit - time_since_last)
        self.last_spotify_request = time.time()
    
    def _rate_limit_lastfm(self):
        """Apply rate limiting for Last.fm API."""
        now = time.time()
        time_since_last = now - self.last_lastfm_request
        if time_since_last < self.lastfm_rate_limit:
            time.sleep(self.lastfm_rate_limit - time_since_last)
        self.last_lastfm_request = time.time()
    
    def _load_cache(self, filename: str) -> Dict:
        """Load cache from disk."""
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {filename}: {e}")
        return {}
    
    def _save_cache(self, filename: str, cache_data: Dict):
        """Save cache to disk."""
        cache_path = self.cache_dir / filename
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {filename}: {e}")
    
    def _load_artist_mappings(self) -> Dict[str, str]:
        """Load artist name mappings for consistency."""
        
        # Common artist name variations in Copenhagen electronic scene
        mappings = {
            'kollektiv turmstrasse': 'Kollektiv Turmstrasse',
            'kollektiv turmstraße': 'Kollektiv Turmstrasse',
            'agnes obel': 'Agnes Obel',
            'wh0': 'WhoMadeWho',
            'whomadewho': 'WhoMadeWho',
            'trentemøller': 'Trentemøller',
            'trentemoller': 'Trentemøller',
            'iceage': 'Iceage',
            'mø': 'MØ',
            'mo': 'MØ',
            'lukas graham': 'Lukas Graham',
            'christopher': 'Christopher',
            'medina': 'Medina',
            'basement jaxx': 'Basement Jaxx',
            'fatboy slim': 'Fatboy Slim',
            'carl cox': 'Carl Cox',
            'solomun': 'Solomun',
            'tale of us': 'Tale Of Us',
            'dixon': 'Dixon',
            'âme': 'Âme',
            'ame': 'Âme'
        }
        
        return mappings

class GenreClassifier:
    """Genre classification and standardization system."""
    
    def __init__(self):
        """Initialize genre classifier."""
        
        # Genre hierarchy and mappings
        self.genre_mappings = self._load_genre_mappings()
        self.genre_patterns = self._load_genre_patterns()
    
    def clean_genres(self, genres: List[str]) -> List[str]:
        """Clean and filter genres."""
        
        cleaned = []
        for genre in genres:
            if isinstance(genre, str) and len(genre.strip()) > 1:
                clean_genre = genre.strip().lower()
                # Remove common non-genre tags
                if clean_genre not in ['seen live', 'favorites', 'danish', 'copenhagen', 'denmark']:
                    cleaned.append(clean_genre)
        
        return cleaned
    
    def clean_and_standardize_genres(self, genres: List[str]) -> List[str]:
        """Clean, standardize, and deduplicate genres."""
        
        standardized = set()
        
        for genre in genres:
            if not isinstance(genre, str) or len(genre.strip()) < 2:
                continue
            
            genre_lower = genre.strip().lower()
            
            # Map to standard genre
            if genre_lower in self.genre_mappings:
                standardized.add(self.genre_mappings[genre_lower])
            else:
                # Keep original if not in mappings
                standardized.add(genre.strip().title())
        
        # Convert back to list and sort
        return sorted(list(standardized))
    
    def extract_genres_from_text(self, text: str) -> List[str]:
        """Extract genre indicators from text."""
        
        if not text:
            return []
        
        text_lower = text.lower()
        detected_genres = []
        
        for pattern, genre in self.genre_patterns.items():
            if pattern in text_lower:
                detected_genres.append(genre)
        
        return detected_genres
    
    def infer_genres_from_name(self, artist_name: str) -> List[str]:
        """Infer genres from artist name patterns."""
        
        name_lower = artist_name.lower()
        inferred = []
        
        # DJ/producer patterns
        if any(prefix in name_lower for prefix in ['dj ', 'mc ', 'producer ']):
            inferred.append('electronic')
        
        # Language/origin patterns
        if any(indicator in name_lower for indicator in ['kollektiv', 'møller', 'æ', 'ø', 'å']):
            inferred.append('danish')
        
        return inferred
    
    def _load_genre_mappings(self) -> Dict[str, str]:
        """Load genre standardization mappings."""
        
        return {
            # Electronic music
            'techno': 'Techno',
            'tech house': 'Tech House',
            'tech-house': 'Tech House',
            'deep house': 'Deep House',
            'progressive house': 'Progressive House',
            'minimal': 'Minimal Techno',
            'minimal techno': 'Minimal Techno',
            'electro': 'Electro',
            'electronica': 'Electronic',
            'electronic': 'Electronic',
            'ambient': 'Ambient',
            'trance': 'Trance',
            'drum and bass': 'Drum & Bass',
            'drum & bass': 'Drum & Bass',
            'dnb': 'Drum & Bass',
            'dubstep': 'Dubstep',
            'edm': 'EDM',
            
            # Other genres
            'indie': 'Indie',
            'indie pop': 'Indie Pop',
            'indie rock': 'Indie Rock',
            'alternative': 'Alternative',
            'rock': 'Rock',
            'pop': 'Pop',
            'hip hop': 'Hip Hop',
            'hip-hop': 'Hip Hop',
            'rap': 'Hip Hop',
            'jazz': 'Jazz',
            'classical': 'Classical',
            'folk': 'Folk',
            'experimental': 'Experimental',
            
            # Danish/Nordic
            'danish': 'Danish',
            'nordic': 'Nordic',
            'scandinavian': 'Scandinavian'
        }
    
    def _load_genre_patterns(self) -> Dict[str, str]:
        """Load text patterns for genre detection."""
        
        return {
            'techno': 'Techno',
            'house music': 'House',
            'house': 'House',
            'electronic music': 'Electronic',
            'dance music': 'Dance',
            'club music': 'Electronic',
            'underground': 'Underground',
            'rave': 'Electronic',
            'beats': 'Electronic',
            'dj set': 'Electronic',
            'live set': 'Electronic',
            'synthesizer': 'Electronic',
            'indie': 'Indie',
            'alternative': 'Alternative',
            'experimental': 'Experimental',
            'acoustic': 'Acoustic',
            'jazz': 'Jazz',
            'classical': 'Classical',
            'orchestra': 'Classical',
            'hip hop': 'Hip Hop',
            'rap': 'Hip Hop',
            'rock': 'Rock',
            'metal': 'Metal',
            'punk': 'Punk',
            'folk': 'Folk',
            'ambient': 'Ambient',
            'minimal': 'Minimal'
        }

def main():
    """Example usage of ArtistGenreEnricher."""
    
    # Initialize with API keys (would normally come from environment)
    enricher = ArtistGenreEnricher()
    
    # Test event data
    test_event = {
        'title': 'Kollektiv Turmstrasse Live at Culture Box',
        'description': 'Underground techno night with the German duo. Deep house and minimal beats all night.',
        'artists': ['Kollektiv Turmstrasse', 'Local DJ'],
        'venue_name': 'Culture Box'
    }
    
    # Enrich artists and genres
    result = enricher.enrich_event_artists(test_event, deep_enrichment=False)
    
    print(f"Original artists: {result.original_artists}")
    print(f"Enriched {len(result.enriched_artists)} artists:")
    
    for artist in result.enriched_artists:
        print(f"  - {artist.name} → {artist.canonical_name}")
        print(f"    Genres: {artist.genres}")
        print(f"    Popularity: {artist.popularity_score:.2f}")
    
    print(f"\nEnhanced genres: {result.enhanced_genres}")
    print(f"Confidence score: {result.confidence_score:.2f}")
    print(f"Sources used: {result.sources_used}")

if __name__ == "__main__":
    main()