#!/usr/bin/env python3
"""
Content-based recommendation model using sentence transformers and structured features.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from pathlib import Path

from ..embeddings.content_embedder import ContentEmbedder, EventFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserPreferences:
    """User preferences for content-based recommendations."""
    preferred_genres: List[str]
    preferred_artists: List[str]
    preferred_venues: List[str]
    price_range: Tuple[float, float]  # (min, max)
    location_lat: Optional[float]
    location_lon: Optional[float]
    preferred_times: List[int]  # Hours of day (0-23)
    preferred_days: List[int]   # Days of week (0-6)

class ContentBasedRecommender:
    """Content-based recommendation system for events."""
    
    def __init__(
        self,
        embedder: ContentEmbedder = None,
        model_cache_dir: str = "ml/models/content_based"
    ):
        """
        Initialize content-based recommender.
        
        Args:
            embedder: ContentEmbedder instance, creates new if None
            model_cache_dir: Directory to cache model data
        """
        self.embedder = embedder or ContentEmbedder()
        self.cache_dir = Path(model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cached data
        self.event_features: List[EventFeatures] = []
        self.event_ids: List[str] = []
        self.genre_popularity: Dict[str, float] = {}
        self.artist_popularity: Dict[str, float] = {}
        self.venue_popularity: Dict[str, float] = {}
        
        # Model parameters
        self.similarity_weights = {
            'text': 0.5,      # Text embedding similarity
            'genre': 0.2,     # Genre overlap
            'artist': 0.1,    # Artist preference
            'venue': 0.1,     # Venue/location preference  
            'price': 0.05,    # Price compatibility
            'time': 0.05      # Time preference
        }
    
    def fit(self, events: List[Dict], user_interactions: List[Dict] = None):
        """
        Fit the content-based model on event data.
        
        Args:
            events: List of event dictionaries
            user_interactions: Optional interaction data for popularity scoring
        """
        logger.info(f"Fitting content-based model on {len(events)} events...")
        
        # Generate embeddings for all events
        self.event_features = self.embedder.encode_events(events)
        self.event_ids = [event.get('id', f'event_{i}') for i, event in enumerate(events)]
        
        # Calculate popularity metrics
        self._calculate_popularity_metrics(events, user_interactions)
        
        # Cache the fitted model
        self._save_model()
        
        logger.info("Content-based model fitted successfully")
    
    def _calculate_popularity_metrics(
        self, 
        events: List[Dict], 
        user_interactions: List[Dict] = None
    ):
        """Calculate popularity scores for genres, artists, and venues."""
        
        # Initialize counters
        genre_counts = {}
        artist_counts = {}
        venue_counts = {}
        
        # Count from events
        for event in events:
            # Genres
            for genre in event.get('genres', []):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Artists
            for artist in event.get('artists', []):
                artist_counts[artist] = artist_counts.get(artist, 0) + 1
            
            # Venues
            venue = event.get('venue_name', '')
            if venue:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        # Add interaction-based popularity if available
        if user_interactions:
            interaction_weights = {'like': 1.0, 'going': 2.0, 'went': 1.5, 'save': 0.5}
            
            for interaction in user_interactions:
                event_id = interaction.get('event_id')
                interaction_type = interaction.get('interaction_type')
                weight = interaction_weights.get(interaction_type, 0)
                
                # Find corresponding event
                try:
                    event_idx = self.event_ids.index(event_id)
                    event_features = self.event_features[event_idx]
                    
                    # Add weighted popularity
                    for genre in event_features.genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + weight
                    
                    for artist in event_features.artists:
                        artist_counts[artist] = artist_counts.get(artist, 0) + weight
                    
                    venue_counts[event_features.venue_name] = venue_counts.get(event_features.venue_name, 0) + weight
                    
                except ValueError:
                    continue
        
        # Normalize to probabilities
        total_genre = sum(genre_counts.values()) or 1
        total_artist = sum(artist_counts.values()) or 1
        total_venue = sum(venue_counts.values()) or 1
        
        self.genre_popularity = {k: v / total_genre for k, v in genre_counts.items()}
        self.artist_popularity = {k: v / total_artist for k, v in artist_counts.items()}
        self.venue_popularity = {k: v / total_venue for k, v in venue_counts.items()}
    
    def recommend(
        self,
        user_preferences: UserPreferences,
        candidate_event_ids: List[str] = None,
        num_recommendations: int = 10,
        diversity_factor: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations for a user.
        
        Args:
            user_preferences: User's preferences and constraints
            candidate_event_ids: Optional list of event IDs to consider
            num_recommendations: Number of recommendations to return
            diversity_factor: Factor to promote diversity (0 = pure similarity)
        
        Returns:
            List of (event_id, score) tuples sorted by score
        """
        
        if not self.event_features:
            logger.warning("Model not fitted. Call fit() first.")
            return []
        
        # Filter candidate events
        if candidate_event_ids:
            candidate_indices = []
            for event_id in candidate_event_ids:
                try:
                    idx = self.event_ids.index(event_id)
                    candidate_indices.append(idx)
                except ValueError:
                    continue
        else:
            candidate_indices = list(range(len(self.event_features)))
        
        if not candidate_indices:
            return []
        
        # Create synthetic "query" features from user preferences
        query_features = self._create_query_features(user_preferences)
        
        # Get candidate features
        candidate_features = [self.event_features[i] for i in candidate_indices]
        candidate_ids = [self.event_ids[i] for i in candidate_indices]
        
        # Compute content similarity scores
        content_scores = self.embedder.compute_similarity(
            query_features,
            candidate_features,
            weights=self.similarity_weights
        )
        
        # Add popularity boost
        popularity_scores = self._compute_popularity_scores(candidate_features)
        
        # Combine scores (0.8 content + 0.2 popularity)
        final_scores = 0.8 * content_scores + 0.2 * popularity_scores
        
        # Apply diversity if requested
        if diversity_factor > 0:
            final_scores = self._apply_diversity(
                final_scores, 
                candidate_features, 
                diversity_factor
            )
        
        # Sort and return top recommendations
        scored_events = list(zip(candidate_ids, final_scores))
        scored_events.sort(key=lambda x: x[1], reverse=True)
        
        return scored_events[:num_recommendations]
    
    def _create_query_features(self, user_prefs: UserPreferences) -> EventFeatures:
        """Create synthetic event features from user preferences."""
        
        # Create synthetic text from preferences
        text_parts = []
        
        if user_prefs.preferred_genres:
            text_parts.append(f"music genres: {' '.join(user_prefs.preferred_genres)}")
        
        if user_prefs.preferred_artists:
            text_parts.append(f"artists: {' '.join(user_prefs.preferred_artists)}")
        
        if user_prefs.preferred_venues:
            text_parts.append(f"venues: {' '.join(user_prefs.preferred_venues)}")
        
        synthetic_text = " . ".join(text_parts) if text_parts else "music event"
        
        # Generate embedding for synthetic text
        if self.embedder.sentence_model:
            embedding = self.embedder.sentence_model.encode([synthetic_text])[0]
        else:
            # Fallback: average embeddings of matching events
            embedding = np.zeros(self.embedder.embedding_dim)
            count = 0
            
            for ef in self.event_features:
                if any(genre in ef.genres for genre in user_prefs.preferred_genres):
                    embedding += ef.text_embedding
                    count += 1
            
            if count > 0:
                embedding /= count
        
        # Calculate H3 index from user location
        h3_index = ""
        if user_prefs.location_lat and user_prefs.location_lon:
            import h3
            h3_index = h3.geo_to_h3(user_prefs.location_lat, user_prefs.location_lon, 8)
        
        # Create datetime features from preferences
        datetime_features = {}
        if user_prefs.preferred_times:
            avg_hour = sum(user_prefs.preferred_times) / len(user_prefs.preferred_times)
            datetime_features['hour'] = avg_hour / 23.0
        
        if user_prefs.preferred_days:
            avg_day = sum(user_prefs.preferred_days) / len(user_prefs.preferred_days)
            datetime_features['day_of_week'] = avg_day / 6.0
        
        return EventFeatures(
            text_embedding=embedding,
            title="User Query",
            description=synthetic_text,
            genres=user_prefs.preferred_genres,
            artists=user_prefs.preferred_artists,
            venue_name=user_prefs.preferred_venues[0] if user_prefs.preferred_venues else "",
            venue_lat=user_prefs.location_lat or 0.0,
            venue_lon=user_prefs.location_lon or 0.0,
            h3_index=h3_index,
            price_min=user_prefs.price_range[0],
            price_max=user_prefs.price_range[1],
            popularity_score=0.0,
            datetime_features=datetime_features
        )
    
    def _compute_popularity_scores(self, candidate_features: List[EventFeatures]) -> np.ndarray:
        """Compute popularity-based scores for candidates."""
        
        scores = np.zeros(len(candidate_features))
        
        for i, ef in enumerate(candidate_features):
            popularity = 0.0
            
            # Genre popularity
            if ef.genres:
                genre_pop = np.mean([self.genre_popularity.get(g, 0) for g in ef.genres])
                popularity += genre_pop
            
            # Artist popularity
            if ef.artists:
                artist_pop = np.mean([self.artist_popularity.get(a, 0) for a in ef.artists])
                popularity += artist_pop
            
            # Venue popularity
            venue_pop = self.venue_popularity.get(ef.venue_name, 0)
            popularity += venue_pop
            
            # Event's own popularity score
            popularity += ef.popularity_score
            
            scores[i] = popularity / 4.0  # Average of 4 components
        
        return scores
    
    def _apply_diversity(
        self,
        scores: np.ndarray,
        candidate_features: List[EventFeatures],
        diversity_factor: float
    ) -> np.ndarray:
        """Apply diversity penalty to avoid too similar recommendations."""
        
        # Simple diversity: penalize events with very similar genres/artists
        adjusted_scores = scores.copy()
        
        for i in range(len(candidate_features)):
            for j in range(i + 1, len(candidate_features)):
                # Calculate genre overlap
                genres_i = set(candidate_features[i].genres)
                genres_j = set(candidate_features[j].genres)
                
                if genres_i and genres_j:
                    overlap = len(genres_i & genres_j) / len(genres_i | genres_j)
                    
                    # Apply penalty to lower-scored event
                    penalty = diversity_factor * overlap
                    if scores[i] < scores[j]:
                        adjusted_scores[i] -= penalty
                    else:
                        adjusted_scores[j] -= penalty
        
        return np.clip(adjusted_scores, 0, 1)
    
    def explain_recommendation(
        self,
        event_id: str,
        user_preferences: UserPreferences
    ) -> Dict[str, any]:
        """
        Explain why an event was recommended to a user.
        
        Args:
            event_id: ID of the recommended event
            user_preferences: User's preferences
        
        Returns:
            Dictionary with explanation components
        """
        
        try:
            event_idx = self.event_ids.index(event_id)
            event_features = self.event_features[event_idx]
        except ValueError:
            return {"error": "Event not found"}
        
        query_features = self._create_query_features(user_preferences)
        
        # Calculate individual similarity components
        explanations = {}
        
        # Genre match
        user_genres = set(user_preferences.preferred_genres)
        event_genres = set(event_features.genres)
        
        if user_genres and event_genres:
            genre_overlap = user_genres & event_genres
            explanations['genre_match'] = {
                'score': len(genre_overlap) / len(user_genres | event_genres),
                'matched_genres': list(genre_overlap),
                'user_genres': list(user_genres),
                'event_genres': list(event_genres)
            }
        
        # Artist match
        user_artists = set(user_preferences.preferred_artists)
        event_artists = set(event_features.artists)
        
        if user_artists and event_artists:
            artist_overlap = user_artists & event_artists
            explanations['artist_match'] = {
                'score': len(artist_overlap) / len(user_artists | event_artists),
                'matched_artists': list(artist_overlap)
            }
        
        # Location proximity
        if (user_preferences.location_lat and user_preferences.location_lon and
            event_features.venue_lat and event_features.venue_lon):
            
            from geopy.distance import geodesic
            distance = geodesic(
                (user_preferences.location_lat, user_preferences.location_lon),
                (event_features.venue_lat, event_features.venue_lon)
            ).kilometers
            
            explanations['location'] = {
                'distance_km': round(distance, 1),
                'venue_name': event_features.venue_name
            }
        
        # Price compatibility
        user_min, user_max = user_preferences.price_range
        event_min = event_features.price_min or 0
        event_max = event_features.price_max or event_min
        
        if event_max > 0:
            price_compatible = not (event_min > user_max or event_max < user_min)
            explanations['price'] = {
                'compatible': price_compatible,
                'user_range': f"{user_min}-{user_max} DKK",
                'event_range': f"{event_min}-{event_max} DKK"
            }
        
        # Overall similarity score
        similarity = self.embedder.compute_similarity(
            query_features, [event_features], weights=self.similarity_weights
        )[0]
        
        explanations['overall_similarity'] = round(similarity, 3)
        explanations['popularity_score'] = round(event_features.popularity_score, 3)
        
        return explanations
    
    def _save_model(self):
        """Save fitted model to cache."""
        
        model_data = {
            'event_ids': self.event_ids,
            'genre_popularity': self.genre_popularity,
            'artist_popularity': self.artist_popularity,
            'venue_popularity': self.venue_popularity,
            'similarity_weights': self.similarity_weights
        }
        
        model_path = self.cache_dir / "model_data.json"
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Save event features separately
        features_path = self.cache_dir / "event_features.json"
        self.embedder.save_embeddings(self.event_features, str(features_path))
        
        logger.info(f"Model saved to {self.cache_dir}")
    
    def load_model(self):
        """Load fitted model from cache."""
        
        model_path = self.cache_dir / "model_data.json"
        features_path = self.cache_dir / "event_features.json"
        
        if not (model_path.exists() and features_path.exists()):
            logger.warning("No cached model found")
            return False
        
        # Load model data
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        self.event_ids = model_data['event_ids']
        self.genre_popularity = model_data['genre_popularity']
        self.artist_popularity = model_data['artist_popularity']
        self.venue_popularity = model_data['venue_popularity']
        self.similarity_weights = model_data['similarity_weights']
        
        # Load event features
        self.event_features = self.embedder.load_embeddings(str(features_path))
        
        logger.info(f"Model loaded from {self.cache_dir}")
        return True

def main():
    """Example usage of ContentBasedRecommender."""
    
    # Sample events
    events = [
        {
            'id': 'event_1',
            'title': 'Techno Night at Culture Box',
            'description': 'Underground electronic music with top DJs',
            'genres': ['techno', 'electronic'],
            'artists': ['Kollektiv Turmstrasse'],
            'venue_name': 'Culture Box',
            'venue_lat': 55.6826,
            'venue_lon': 12.5941,
            'price_min': 200,
            'price_max': 300
        },
        {
            'id': 'event_2',
            'title': 'Indie Concert at Vega',
            'description': 'Intimate acoustic performance',
            'genres': ['indie', 'acoustic'],
            'artists': ['Agnes Obel'],
            'venue_name': 'Vega',
            'venue_lat': 55.6667,
            'venue_lon': 12.5419,
            'price_min': 400,
            'price_max': 600
        }
    ]
    
    # User preferences
    user_prefs = UserPreferences(
        preferred_genres=['techno', 'electronic'],
        preferred_artists=[],
        preferred_venues=['Culture Box'],
        price_range=(100, 400),
        location_lat=55.6761,
        location_lon=12.5683,
        preferred_times=[20, 21, 22],
        preferred_days=[4, 5, 6]  # Fri, Sat, Sun
    )
    
    # Initialize and fit recommender
    recommender = ContentBasedRecommender()
    recommender.fit(events)
    
    # Get recommendations
    recommendations = recommender.recommend(user_prefs, num_recommendations=5)
    
    print("Content-based recommendations:")
    for event_id, score in recommendations:
        print(f"- {event_id}: {score:.3f}")
        
        # Get explanation
        explanation = recommender.explain_recommendation(event_id, user_prefs)
        print(f"  Explanation: {explanation}")

if __name__ == "__main__":
    main()