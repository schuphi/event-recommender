#!/usr/bin/env python3
"""
Advanced duplicate detection system for event data.
Uses multiple similarity algorithms to detect duplicate events across sources.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import difflib
from collections import defaultdict
import numpy as np
from geopy.distance import geodesic
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DuplicateMatch:
    """Represents a duplicate match between two events."""
    event_id_1: str
    event_id_2: str
    similarity_score: float
    match_type: str  # 'exact', 'high_confidence', 'likely', 'possible'
    similarity_components: Dict[str, float]
    match_reasons: List[str]

@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    total_events: int
    unique_events: int
    duplicates_found: int
    duplicate_groups: List[List[str]]
    duplicate_matches: List[DuplicateMatch]
    confidence_scores: Dict[str, float]

class EventDuplicateDetector:
    """Advanced duplicate detection for events using multiple similarity measures."""
    
    def __init__(self):
        """Initialize duplicate detector."""
        
        # Similarity thresholds
        self.thresholds = {
            'exact': 0.95,
            'high_confidence': 0.85,
            'likely': 0.70,
            'possible': 0.55
        }
        
        # Component weights for final similarity score
        self.weights = {
            'title_similarity': 0.25,
            'venue_similarity': 0.20,
            'time_similarity': 0.20,
            'description_similarity': 0.15,
            'location_similarity': 0.10,
            'artist_similarity': 0.10
        }
        
        # Text vectorizer for content similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Cache for processed data
        self.processed_events = {}
        self.venue_clusters = defaultdict(list)
        
    def detect_duplicates(self, events: List[Dict]) -> DeduplicationResult:
        """
        Detect duplicates in a list of events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            DeduplicationResult with duplicate information
        """
        
        logger.info(f"Starting duplicate detection for {len(events)} events")
        
        if len(events) < 2:
            return DeduplicationResult(
                total_events=len(events),
                unique_events=len(events),
                duplicates_found=0,
                duplicate_groups=[],
                duplicate_matches=[],
                confidence_scores={}
            )
        
        # Step 1: Preprocess events
        processed_events = self._preprocess_events(events)
        
        # Step 2: Initial clustering by venue and time
        venue_time_clusters = self._cluster_by_venue_time(processed_events)
        
        # Step 3: Detailed similarity analysis within clusters
        duplicate_matches = []
        for cluster in venue_time_clusters:
            if len(cluster) > 1:
                cluster_matches = self._find_duplicates_in_cluster(cluster)
                duplicate_matches.extend(cluster_matches)
        
        # Step 4: Cross-cluster duplicate detection (for events with different venue names)
        cross_cluster_matches = self._find_cross_cluster_duplicates(processed_events)
        duplicate_matches.extend(cross_cluster_matches)
        
        # Step 5: Group duplicates into clusters
        duplicate_groups = self._group_duplicates(duplicate_matches)
        
        # Step 6: Calculate statistics
        unique_events = len(events) - sum(len(group) - 1 for group in duplicate_groups)
        duplicates_found = len(events) - unique_events
        
        # Step 7: Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(duplicate_matches)
        
        logger.info(f"Found {duplicates_found} duplicates in {len(duplicate_groups)} groups")
        
        return DeduplicationResult(
            total_events=len(events),
            unique_events=unique_events,
            duplicates_found=duplicates_found,
            duplicate_groups=duplicate_groups,
            duplicate_matches=duplicate_matches,
            confidence_scores=confidence_scores
        )
    
    def _preprocess_events(self, events: List[Dict]) -> List[Dict]:
        """Preprocess events for duplicate detection."""
        
        processed = []
        
        for i, event in enumerate(events):
            # Create a copy with normalized fields
            processed_event = event.copy()
            
            # Add index for tracking
            processed_event['_index'] = i
            processed_event['_id'] = event.get('id', f'event_{i}')
            
            # Normalize text fields
            processed_event['_title_normalized'] = self._normalize_text(event.get('title', ''))
            processed_event['_description_normalized'] = self._normalize_text(event.get('description', ''))
            processed_event['_venue_normalized'] = self._normalize_venue_name(event.get('venue_name', ''))
            
            # Normalize artists
            artists = event.get('artists', [])
            if isinstance(artists, list):
                processed_event['_artists_normalized'] = [
                    self._normalize_text(artist) for artist in artists if isinstance(artist, str)
                ]
            else:
                processed_event['_artists_normalized'] = []
            
            # Parse and normalize datetime
            start_time = event.get('start_time')
            if start_time:
                if isinstance(start_time, str):
                    try:
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    except ValueError:
                        start_time = None
                processed_event['_start_time_normalized'] = start_time
            
            # Normalize location
            lat = event.get('venue_lat')
            lon = event.get('venue_lon')
            if lat is not None and lon is not None:
                processed_event['_location'] = (float(lat), float(lon))
            else:
                processed_event['_location'] = None
            
            processed.append(processed_event)
        
        return processed
    
    def _cluster_by_venue_time(self, events: List[Dict]) -> List[List[Dict]]:
        """Cluster events by venue and time for efficient comparison."""
        
        clusters = defaultdict(list)
        
        for event in events:
            # Create cluster key from venue and date
            venue_key = event['_venue_normalized']
            
            start_time = event.get('_start_time_normalized')
            if start_time:
                date_key = start_time.strftime('%Y%m%d')
            else:
                date_key = 'unknown'
            
            cluster_key = f"{venue_key}_{date_key}"
            clusters[cluster_key].append(event)
        
        # Convert to list of clusters
        return list(clusters.values())
    
    def _find_duplicates_in_cluster(self, cluster: List[Dict]) -> List[DuplicateMatch]:
        """Find duplicates within a cluster of similar events."""
        
        matches = []
        
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                event1 = cluster[i]
                event2 = cluster[j]
                
                similarity_result = self._calculate_event_similarity(event1, event2)
                
                if similarity_result['total_score'] >= self.thresholds['possible']:
                    match_type = self._classify_match_confidence(similarity_result['total_score'])
                    
                    match = DuplicateMatch(
                        event_id_1=event1['_id'],
                        event_id_2=event2['_id'],
                        similarity_score=similarity_result['total_score'],
                        match_type=match_type,
                        similarity_components=similarity_result['components'],
                        match_reasons=self._generate_match_reasons(similarity_result)
                    )
                    
                    matches.append(match)
        
        return matches
    
    def _find_cross_cluster_duplicates(self, events: List[Dict]) -> List[DuplicateMatch]:
        """Find duplicates across different venue/time clusters."""
        
        matches = []
        
        # Focus on events that might have venue name variations
        potential_cross_matches = []
        
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1 = events[i]
                event2 = events[j]
                
                # Skip if they're already in the same venue cluster
                if event1['_venue_normalized'] == event2['_venue_normalized']:
                    continue
                
                # Quick pre-filter based on title similarity
                title_sim = self._text_similarity(
                    event1['_title_normalized'],
                    event2['_title_normalized']
                )
                
                if title_sim >= 0.6:  # Potential cross-cluster match
                    potential_cross_matches.append((event1, event2))
        
        # Detailed analysis for potential matches
        for event1, event2 in potential_cross_matches:
            similarity_result = self._calculate_event_similarity(event1, event2)
            
            # Higher threshold for cross-cluster matches to avoid false positives
            if similarity_result['total_score'] >= self.thresholds['likely']:
                match_type = self._classify_match_confidence(similarity_result['total_score'])
                
                match = DuplicateMatch(
                    event_id_1=event1['_id'],
                    event_id_2=event2['_id'],
                    similarity_score=similarity_result['total_score'],
                    match_type=match_type,
                    similarity_components=similarity_result['components'],
                    match_reasons=self._generate_match_reasons(similarity_result) + ['cross_venue_match']
                )
                
                matches.append(match)
        
        return matches
    
    def _calculate_event_similarity(self, event1: Dict, event2: Dict) -> Dict[str, any]:
        """Calculate comprehensive similarity between two events."""
        
        components = {}
        
        # Title similarity
        components['title_similarity'] = self._text_similarity(
            event1['_title_normalized'],
            event2['_title_normalized']
        )
        
        # Venue similarity
        components['venue_similarity'] = self._venue_similarity(
            event1['_venue_normalized'],
            event2['_venue_normalized']
        )
        
        # Time similarity
        components['time_similarity'] = self._time_similarity(
            event1.get('_start_time_normalized'),
            event2.get('_start_time_normalized')
        )
        
        # Description similarity
        components['description_similarity'] = self._text_similarity(
            event1['_description_normalized'],
            event2['_description_normalized']
        )
        
        # Location similarity
        components['location_similarity'] = self._location_similarity(
            event1.get('_location'),
            event2.get('_location')
        )
        
        # Artist similarity
        components['artist_similarity'] = self._artist_similarity(
            event1['_artists_normalized'],
            event2['_artists_normalized']
        )
        
        # Calculate weighted total score
        total_score = sum(
            components[component] * self.weights[component]
            for component in components
            if component in self.weights
        )
        
        return {
            'total_score': total_score,
            'components': components
        }
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        
        if not text1 or not text2:
            return 0.0
        
        # Combine multiple similarity measures
        
        # 1. Levenshtein distance (normalized)
        levenshtein_sim = 1.0 - (Levenshtein.distance(text1, text2) / max(len(text1), len(text2)))
        
        # 2. Sequence matcher ratio
        sequence_sim = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # 3. Jaccard similarity (word-based)
        words1 = set(text1.split())
        words2 = set(text2.split())
        if words1 or words2:
            jaccard_sim = len(words1 & words2) / len(words1 | words2)
        else:
            jaccard_sim = 0.0
        
        # Weighted combination
        similarity = (levenshtein_sim * 0.4 + sequence_sim * 0.4 + jaccard_sim * 0.2)
        
        return similarity
    
    def _venue_similarity(self, venue1: str, venue2: str) -> float:
        """Calculate venue name similarity with special handling for variations."""
        
        if not venue1 or not venue2:
            return 0.0
        
        # Exact match
        if venue1 == venue2:
            return 1.0
        
        # Basic text similarity
        base_sim = self._text_similarity(venue1, venue2)
        
        # Check for common venue variations
        variations = self._get_venue_variations(venue1, venue2)
        if variations:
            return max(base_sim, 0.9)  # High similarity for known variations
        
        # Check for contained names (e.g., "Vega" in "Store Vega")
        if venue1 in venue2 or venue2 in venue1:
            return max(base_sim, 0.8)
        
        return base_sim
    
    def _time_similarity(self, time1: Optional[datetime], time2: Optional[datetime]) -> float:
        """Calculate time similarity between events."""
        
        if not time1 or not time2:
            return 0.5  # Neutral score for missing times
        
        # Calculate time difference
        time_diff = abs((time1 - time2).total_seconds())
        
        # Same day bonus
        if time1.date() == time2.date():
            if time_diff <= 3600:  # Within 1 hour
                return 1.0
            elif time_diff <= 7200:  # Within 2 hours
                return 0.9
            elif time_diff <= 14400:  # Within 4 hours
                return 0.8
            else:
                return 0.6  # Same day but different times
        
        # Different days
        day_diff = abs((time1.date() - time2.date()).days)
        if day_diff == 1:
            return 0.3  # Adjacent days
        elif day_diff <= 7:
            return 0.1  # Same week
        else:
            return 0.0  # Too far apart
    
    def _location_similarity(self, loc1: Optional[Tuple], loc2: Optional[Tuple]) -> float:
        """Calculate geographic similarity between locations."""
        
        if not loc1 or not loc2:
            return 0.5  # Neutral score for missing locations
        
        try:
            distance = geodesic(loc1, loc2).kilometers
            
            if distance <= 0.1:  # Same location (100m tolerance)
                return 1.0
            elif distance <= 0.5:  # Very close
                return 0.9
            elif distance <= 1.0:  # Close
                return 0.7
            elif distance <= 2.0:  # Nearby
                return 0.5
            else:
                return 0.0  # Too far apart
                
        except Exception:
            return 0.0
    
    def _artist_similarity(self, artists1: List[str], artists2: List[str]) -> float:
        """Calculate artist lineup similarity."""
        
        if not artists1 and not artists2:
            return 1.0  # Both empty
        
        if not artists1 or not artists2:
            return 0.0  # One empty, one not
        
        # Convert to sets for comparison
        set1 = set(artists1)
        set2 = set(artists2)
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 1.0
        
        jaccard_sim = intersection / union
        
        # Fuzzy matching for similar artist names
        fuzzy_matches = 0
        for artist1 in artists1:
            for artist2 in artists2:
                if artist1 not in set2 and artist2 not in set1:  # Not exact matches
                    if self._text_similarity(artist1, artist2) >= 0.8:
                        fuzzy_matches += 1
                        break
        
        # Combine exact and fuzzy matches
        total_possible_matches = min(len(artists1), len(artists2))
        if total_possible_matches > 0:
            fuzzy_contribution = fuzzy_matches / total_possible_matches * 0.3
            return min(1.0, jaccard_sim + fuzzy_contribution)
        
        return jaccard_sim
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common noise words and punctuation
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove common filler words
        filler_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in filler_words]
        
        return ' '.join(words)
    
    def _normalize_venue_name(self, venue_name: str) -> str:
        """Normalize venue name for comparison."""
        
        if not isinstance(venue_name, str):
            return ""
        
        normalized = venue_name.lower().strip()
        
        # Remove common suffixes
        suffixes = [' copenhagen', ' cph', ' dk', ' denmark', ' venue', ' club', ' bar']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # Remove common prefixes
        prefixes = ['club ', 'venue ', 'bar ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Standardize common variations
        replacements = {
            'kultur box': 'culture box',
            'kulturhuset': 'kulturhus',
            'kb hallen': 'kb-hallen',
            'the standard': 'standard'
        }
        
        for old, new in replacements.items():
            if old in normalized:
                normalized = normalized.replace(old, new)
        
        return normalized
    
    def _get_venue_variations(self, venue1: str, venue2: str) -> bool:
        """Check if two venue names are known variations."""
        
        # Common venue variations in Copenhagen
        variations = [
            {'culture box', 'kulturbox', 'kultur box'},
            {'vega', 'store vega'},
            {'kb hallen', 'kb-hallen', 'kbhallen'},
            {'rust', 'rust nightclub'},
            {'the standard', 'standard'},
            {'loppen', 'loppen christiania'},
        ]
        
        venue1_norm = venue1.lower().strip()
        venue2_norm = venue2.lower().strip()
        
        for variation_set in variations:
            if venue1_norm in variation_set and venue2_norm in variation_set:
                return True
        
        return False
    
    def _classify_match_confidence(self, similarity_score: float) -> str:
        """Classify match confidence based on similarity score."""
        
        if similarity_score >= self.thresholds['exact']:
            return 'exact'
        elif similarity_score >= self.thresholds['high_confidence']:
            return 'high_confidence'
        elif similarity_score >= self.thresholds['likely']:
            return 'likely'
        else:
            return 'possible'
    
    def _generate_match_reasons(self, similarity_result: Dict) -> List[str]:
        """Generate human-readable reasons for the match."""
        
        reasons = []
        components = similarity_result['components']
        
        if components.get('title_similarity', 0) >= 0.8:
            reasons.append('very_similar_titles')
        elif components.get('title_similarity', 0) >= 0.6:
            reasons.append('similar_titles')
        
        if components.get('venue_similarity', 0) >= 0.9:
            reasons.append('same_venue')
        elif components.get('venue_similarity', 0) >= 0.7:
            reasons.append('similar_venue')
        
        if components.get('time_similarity', 0) >= 0.9:
            reasons.append('same_time')
        elif components.get('time_similarity', 0) >= 0.6:
            reasons.append('similar_time')
        
        if components.get('artist_similarity', 0) >= 0.8:
            reasons.append('same_artists')
        elif components.get('artist_similarity', 0) >= 0.5:
            reasons.append('similar_artists')
        
        if components.get('location_similarity', 0) >= 0.9:
            reasons.append('same_location')
        
        if not reasons:
            reasons.append('overall_similarity')
        
        return reasons
    
    def _group_duplicates(self, matches: List[DuplicateMatch]) -> List[List[str]]:
        """Group duplicate matches into connected components."""
        
        # Build graph of connections
        connections = defaultdict(set)
        all_events = set()
        
        for match in matches:
            # Only group high confidence matches to avoid false clusters
            if match.match_type in ['exact', 'high_confidence', 'likely']:
                connections[match.event_id_1].add(match.event_id_2)
                connections[match.event_id_2].add(match.event_id_1)
                all_events.add(match.event_id_1)
                all_events.add(match.event_id_2)
        
        # Find connected components using DFS
        visited = set()
        groups = []
        
        for event_id in all_events:
            if event_id not in visited:
                group = []
                stack = [event_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        
                        # Add connected events to stack
                        for connected in connections[current]:
                            if connected not in visited:
                                stack.append(connected)
                
                if len(group) > 1:  # Only add groups with multiple events
                    groups.append(sorted(group))
        
        return groups
    
    def _calculate_confidence_scores(self, matches: List[DuplicateMatch]) -> Dict[str, float]:
        """Calculate overall confidence scores for duplicate detection."""
        
        if not matches:
            return {'overall_confidence': 0.0}
        
        # Calculate metrics
        total_matches = len(matches)
        high_confidence_matches = sum(1 for m in matches if m.match_type in ['exact', 'high_confidence'])
        avg_similarity = sum(m.similarity_score for m in matches) / total_matches
        
        # Overall confidence based on match quality
        overall_confidence = (
            (high_confidence_matches / total_matches) * 0.6 +
            avg_similarity * 0.4
        )
        
        return {
            'overall_confidence': overall_confidence,
            'avg_similarity': avg_similarity,
            'high_confidence_ratio': high_confidence_matches / total_matches if total_matches > 0 else 0.0,
            'total_matches': total_matches
        }

def main():
    """Example usage of EventDuplicateDetector."""
    
    detector = EventDuplicateDetector()
    
    # Test data with duplicates
    test_events = [
        {
            'id': 'event_1',
            'title': 'Kollektiv Turmstrasse Live',
            'description': 'Electronic techno night',
            'venue_name': 'Culture Box',
            'venue_lat': 55.6826,
            'venue_lon': 12.5941,
            'start_time': '2024-12-01T20:00:00',
            'artists': ['Kollektiv Turmstrasse']
        },
        {
            'id': 'event_2',
            'title': 'Kollektiv Turmstrasse Live at Culture Box',
            'description': 'Underground techno with German duo',
            'venue_name': 'Kultur Box',
            'venue_lat': 55.6826,
            'venue_lon': 12.5941,
            'start_time': '2024-12-01T20:30:00',
            'artists': ['Kollektiv Turmstrasse']
        },
        {
            'id': 'event_3',
            'title': 'Different Event',
            'description': 'Completely different event',
            'venue_name': 'Vega',
            'venue_lat': 55.6667,
            'venue_lon': 12.5419,
            'start_time': '2024-12-01T19:00:00',
            'artists': ['Other Artist']
        }
    ]
    
    result = detector.detect_duplicates(test_events)
    
    print(f"Duplicate Detection Results:")
    print(f"Total events: {result.total_events}")
    print(f"Unique events: {result.unique_events}")
    print(f"Duplicates found: {result.duplicates_found}")
    print(f"Duplicate groups: {len(result.duplicate_groups)}")
    
    for i, group in enumerate(result.duplicate_groups):
        print(f"  Group {i+1}: {group}")
    
    print(f"\nDuplicate matches:")
    for match in result.duplicate_matches:
        print(f"  {match.event_id_1} ↔ {match.event_id_2}")
        print(f"    Similarity: {match.similarity_score:.3f} ({match.match_type})")
        print(f"    Reasons: {', '.join(match.match_reasons)}")

if __name__ == "__main__":
    main()