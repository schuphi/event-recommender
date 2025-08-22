#!/usr/bin/env python3
"""
Hybrid recommendation system that combines content-based and collaborative filtering
with a neural re-ranker using structured features (distance, price, recency, popularity, diversity).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timedelta
import h3
from geopy.distance import geodesic

from .content_based import ContentBasedRecommender, UserPreferences
from .collaborative_filtering import CollaborativeFilteringRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EventCandidate:
    """Event candidate with all features for ranking."""
    event_id: str
    content_score: float
    cf_score: float
    distance_km: float
    price_score: float
    recency_score: float
    popularity_score: float
    diversity_score: float
    time_match_score: float
    final_score: float = 0.0

class HybridRankerMLP(nn.Module):
    """Multi-layer perceptron for hybrid ranking."""
    
    def __init__(
        self,
        input_dim: int = 8,  # Number of input features
        hidden_dims: List[int] = [32, 16],
        dropout: float = 0.2
    ):
        """
        Initialize hybrid ranker MLP.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super(HybridRankerMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single score)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output between 0 and 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            features: Input features [batch_size, input_dim]
        
        Returns:
            Predicted scores [batch_size, 1]
        """
        return self.network(features)

class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches."""
    
    def __init__(
        self,
        content_model: ContentBasedRecommender = None,
        cf_model: CollaborativeFilteringRecommender = None,
        model_cache_dir: str = "ml/models/hybrid"
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            content_model: Trained content-based model
            cf_model: Trained collaborative filtering model
            model_cache_dir: Directory to cache hybrid model
        """
        self.content_model = content_model
        self.cf_model = cf_model
        self.cache_dir = Path(model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Neural ranker
        self.ranker_model: Optional[HybridRankerMLP] = None
        
        # Feature weights for simple weighted combination (fallback)
        self.feature_weights = {
            'content': 0.35,
            'collaborative': 0.25,
            'distance': 0.15,
            'price': 0.05,
            'recency': 0.05,
            'popularity': 0.05,
            'diversity': 0.05,
            'time_match': 0.05
        }
        
        # Feature normalizers (learned from training data)
        self.feature_stats = {
            'distance_km': {'mean': 5.0, 'std': 3.0},
            'price': {'mean': 350.0, 'std': 200.0},
            'recency_days': {'mean': 7.0, 'std': 14.0},
            'popularity': {'mean': 0.5, 'std': 0.3}
        }
    
    def recommend(
        self,
        user_preferences: UserPreferences,
        user_id: str = None,
        candidate_event_ids: List[str] = None,
        events_data: List[Dict] = None,
        num_recommendations: int = 10,
        use_neural_ranker: bool = True,
        diversity_weight: float = 0.1
    ) -> List[Tuple[str, float, Dict]]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_preferences: User's preferences and context
            user_id: User ID for collaborative filtering (optional)
            candidate_event_ids: List of candidate event IDs
            events_data: Event metadata for feature extraction
            num_recommendations: Number of recommendations to return
            use_neural_ranker: Whether to use neural ranker (if trained)
            diversity_weight: Weight for diversity promotion
        
        Returns:
            List of (event_id, score, explanation) tuples
        """
        
        if not candidate_event_ids or not events_data:
            logger.warning("No candidate events provided")
            return []
        
        # Create event lookup
        events_lookup = {event['id']: event for event in events_data}
        
        # Get content-based scores
        content_scores = {}
        if self.content_model:
            try:
                content_recs = self.content_model.recommend(
                    user_preferences,
                    candidate_event_ids=candidate_event_ids,
                    num_recommendations=len(candidate_event_ids)
                )
                content_scores = {event_id: score for event_id, score in content_recs}
            except Exception as e:
                logger.warning(f"Content-based scoring failed: {e}")
        
        # Get collaborative filtering scores
        cf_scores = {}
        if self.cf_model and user_id:
            try:
                cf_recs = self.cf_model.recommend(
                    user_id,
                    candidate_event_ids=candidate_event_ids,
                    num_recommendations=len(candidate_event_ids),
                    exclude_seen=True
                )
                cf_scores = {event_id: score for event_id, score in cf_recs}
            except Exception as e:
                logger.warning(f"Collaborative filtering failed: {e}")
        
        # Extract structured features for all candidates
        candidates = []
        for event_id in candidate_event_ids:
            if event_id not in events_lookup:
                continue
                
            event_data = events_lookup[event_id]
            
            candidate = self._extract_candidate_features(
                event_id,
                event_data,
                user_preferences,
                content_scores.get(event_id, 0.0),
                cf_scores.get(event_id, 0.0)
            )
            
            candidates.append(candidate)
        
        if not candidates:
            return []
        
        # Apply diversity scoring
        candidates = self._apply_diversity_scoring(candidates, diversity_weight)
        
        # Rank candidates
        if use_neural_ranker and self.ranker_model:
            ranked_candidates = self._rank_with_neural_model(candidates)
        else:
            ranked_candidates = self._rank_with_weighted_combination(candidates)
        
        # Sort by final score and return top recommendations
        ranked_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Create explanations
        recommendations = []
        for candidate in ranked_candidates[:num_recommendations]:
            explanation = self._create_explanation(candidate, user_preferences)
            recommendations.append((candidate.event_id, candidate.final_score, explanation))
        
        return recommendations
    
    def _extract_candidate_features(
        self,
        event_id: str,
        event_data: Dict,
        user_prefs: UserPreferences,
        content_score: float,
        cf_score: float
    ) -> EventCandidate:
        """Extract all features for a candidate event."""
        
        # Distance feature
        distance_km = 0.0
        if (user_prefs.location_lat and user_prefs.location_lon and
            event_data.get('venue_lat') and event_data.get('venue_lon')):
            
            try:
                distance_km = geodesic(
                    (user_prefs.location_lat, user_prefs.location_lon),
                    (event_data['venue_lat'], event_data['venue_lon'])
                ).kilometers
            except Exception:
                distance_km = 10.0  # Default moderate distance
        
        # Price compatibility score
        price_score = self._calculate_price_score(event_data, user_prefs)
        
        # Recency score (how soon is the event)
        recency_score = self._calculate_recency_score(event_data)
        
        # Popularity score
        popularity_score = event_data.get('popularity_score', 0.0)
        
        # Time match score (hour/day preferences)
        time_match_score = self._calculate_time_match_score(event_data, user_prefs)
        
        return EventCandidate(
            event_id=event_id,
            content_score=content_score,
            cf_score=cf_score,
            distance_km=distance_km,
            price_score=price_score,
            recency_score=recency_score,
            popularity_score=popularity_score,
            diversity_score=0.0,  # Will be calculated later
            time_match_score=time_match_score
        )
    
    def _calculate_price_score(self, event_data: Dict, user_prefs: UserPreferences) -> float:
        """Calculate price compatibility score."""
        
        event_min = event_data.get('price_min', 0) or 0
        event_max = event_data.get('price_max', event_min) or event_min
        user_min, user_max = user_prefs.price_range
        
        if event_max == 0:  # Free event
            return 1.0 if user_max >= 0 else 0.5
        
        # Check if price ranges overlap
        if event_min > user_max or event_max < user_min:
            return 0.0  # No overlap
        
        # Calculate overlap ratio
        overlap_start = max(event_min, user_min)
        overlap_end = min(event_max, user_max)
        overlap_size = overlap_end - overlap_start
        
        event_range = event_max - event_min or 1
        user_range = user_max - user_min or 1
        
        # Score based on relative overlap
        score = overlap_size / min(event_range, user_range)
        return min(1.0, score)
    
    def _calculate_recency_score(self, event_data: Dict) -> float:
        """Calculate recency score (preference for nearer-term events)."""
        
        event_date = event_data.get('date_time')
        if not event_date:
            return 0.5  # Default score for unknown dates
        
        if isinstance(event_date, str):
            try:
                event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            except:
                return 0.5
        
        now = datetime.now()
        days_until = (event_date - now).days
        
        if days_until < 0:  # Past event
            return 0.0
        elif days_until <= 7:  # This week
            return 1.0
        elif days_until <= 30:  # This month
            return 0.8
        elif days_until <= 90:  # Next 3 months
            return 0.6
        else:  # Far future
            return 0.3
    
    def _calculate_time_match_score(self, event_data: Dict, user_prefs: UserPreferences) -> float:
        """Calculate how well event time matches user preferences."""
        
        event_date = event_data.get('date_time')
        if not event_date:
            return 0.5
        
        if isinstance(event_date, str):
            try:
                event_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            except:
                return 0.5
        
        score = 0.0
        
        # Hour preference
        if user_prefs.preferred_times:
            event_hour = event_date.hour
            hour_scores = [abs(event_hour - pref_hour) for pref_hour in user_prefs.preferred_times]
            best_hour_match = min(hour_scores)
            hour_score = max(0, 1.0 - best_hour_match / 12.0)  # Normalize by 12 hours
            score += hour_score * 0.5
        
        # Day preference
        if user_prefs.preferred_days:
            event_day = event_date.weekday()
            if event_day in user_prefs.preferred_days:
                score += 0.5
        
        return min(1.0, score)
    
    def _apply_diversity_scoring(
        self, 
        candidates: List[EventCandidate], 
        diversity_weight: float
    ) -> List[EventCandidate]:
        """Apply diversity scores to promote variety in recommendations."""
        
        if diversity_weight <= 0:
            return candidates
        
        # Simple diversity based on venue and genre variety
        # More sophisticated approaches could use genre embeddings, etc.
        
        venue_counts = {}
        for candidate in candidates:
            # This would need event data, simplified for now
            venue_counts[candidate.event_id] = venue_counts.get(candidate.event_id, 0) + 1
        
        # Assign diversity scores (higher = more diverse)
        for candidate in candidates:
            # Simple heuristic: events at less popular venues get higher diversity scores
            venue_popularity = venue_counts.get(candidate.event_id, 1)
            candidate.diversity_score = 1.0 / (1.0 + venue_popularity * 0.1)
        
        return candidates
    
    def _rank_with_neural_model(self, candidates: List[EventCandidate]) -> List[EventCandidate]:
        """Rank candidates using trained neural model."""
        
        if not self.ranker_model:
            return self._rank_with_weighted_combination(candidates)
        
        # Prepare features
        features = []
        for candidate in candidates:
            feature_vector = [
                candidate.content_score,
                candidate.cf_score,
                self._normalize_distance(candidate.distance_km),
                candidate.price_score,
                candidate.recency_score,
                candidate.popularity_score,
                candidate.diversity_score,
                candidate.time_match_score
            ]
            features.append(feature_vector)
        
        # Predict scores
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        self.ranker_model.eval()
        with torch.no_grad():
            scores = self.ranker_model(features_tensor).squeeze()
        
        # Update final scores
        for i, candidate in enumerate(candidates):
            candidate.final_score = scores[i].item()
        
        return candidates
    
    def _rank_with_weighted_combination(self, candidates: List[EventCandidate]) -> List[EventCandidate]:
        """Rank candidates using weighted combination of features."""
        
        for candidate in candidates:
            # Normalize distance
            normalized_distance = self._normalize_distance(candidate.distance_km)
            distance_score = 1.0 - normalized_distance  # Closer is better
            
            # Weighted combination
            final_score = (
                self.feature_weights['content'] * candidate.content_score +
                self.feature_weights['collaborative'] * candidate.cf_score +
                self.feature_weights['distance'] * distance_score +
                self.feature_weights['price'] * candidate.price_score +
                self.feature_weights['recency'] * candidate.recency_score +
                self.feature_weights['popularity'] * candidate.popularity_score +
                self.feature_weights['diversity'] * candidate.diversity_score +
                self.feature_weights['time_match'] * candidate.time_match_score
            )
            
            candidate.final_score = final_score
        
        return candidates
    
    def _normalize_distance(self, distance_km: float) -> float:
        """Normalize distance to 0-1 range."""
        # Sigmoid normalization: closer events get higher scores
        return 1.0 / (1.0 + distance_km / 5.0)  # 5km half-distance
    
    def _create_explanation(
        self, 
        candidate: EventCandidate, 
        user_prefs: UserPreferences
    ) -> Dict[str, any]:
        """Create explanation for why this event was recommended."""
        
        explanation = {
            'overall_score': round(candidate.final_score, 3),
            'components': {
                'content_similarity': round(candidate.content_score, 3),
                'collaborative_score': round(candidate.cf_score, 3),
                'distance_km': round(candidate.distance_km, 1),
                'price_compatibility': round(candidate.price_score, 3),
                'timing_score': round(candidate.recency_score, 3),
                'popularity': round(candidate.popularity_score, 3),
                'time_match': round(candidate.time_match_score, 3)
            },
            'reasons': []
        }
        
        # Add specific reasons
        if candidate.content_score > 0.7:
            explanation['reasons'].append("Strong content match with your preferences")
        
        if candidate.cf_score > 0.7:
            explanation['reasons'].append("Liked by users with similar tastes")
        
        if candidate.distance_km < 2.0:
            explanation['reasons'].append("Very close to your location")
        
        if candidate.price_score > 0.8:
            explanation['reasons'].append("Price fits your budget")
        
        if candidate.recency_score > 0.8:
            explanation['reasons'].append("Happening soon")
        
        if candidate.time_match_score > 0.7:
            explanation['reasons'].append("Matches your preferred times")
        
        return explanation
    
    def train_neural_ranker(
        self,
        training_data: List[Dict],
        num_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.001
    ):
        """
        Train the neural ranker on user interaction data.
        
        Args:
            training_data: List of training examples with features and labels
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        
        if len(training_data) < 50:
            logger.warning("Insufficient data for neural ranker training")
            return
        
        logger.info(f"Training neural ranker on {len(training_data)} examples...")
        
        # Initialize model
        self.ranker_model = HybridRankerMLP(input_dim=8)
        optimizer = optim.Adam(self.ranker_model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Prepare training data
        features = []
        labels = []
        
        for example in training_data:
            feature_vector = [
                example['content_score'],
                example['cf_score'],
                self._normalize_distance(example['distance_km']),
                example['price_score'],
                example['recency_score'],
                example['popularity_score'],
                example['diversity_score'],
                example['time_match_score']
            ]
            
            features.append(feature_vector)
            labels.append(example['label'])  # 1 for positive, 0 for negative
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Training loop
        self.ranker_model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for i in range(0, len(features), batch_size):
                batch_features = features_tensor[i:i+batch_size]
                batch_labels = labels_tensor[i:i+batch_size]
                
                # Forward pass
                predictions = self.ranker_model(batch_features)
                loss = criterion(predictions, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / (len(features) // batch_size + 1)
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Save trained model
        self._save_neural_ranker()
        logger.info("Neural ranker training completed")
    
    def _save_neural_ranker(self):
        """Save neural ranker model."""
        if self.ranker_model:
            model_path = self.cache_dir / "neural_ranker.pth"
            torch.save(self.ranker_model.state_dict(), model_path)
            
            # Save feature weights and stats
            config_path = self.cache_dir / "hybrid_config.json"
            config = {
                'feature_weights': self.feature_weights,
                'feature_stats': self.feature_stats
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def load_neural_ranker(self) -> bool:
        """Load neural ranker model."""
        model_path = self.cache_dir / "neural_ranker.pth"
        config_path = self.cache_dir / "hybrid_config.json"
        
        if not (model_path.exists() and config_path.exists()):
            return False
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.feature_weights = config['feature_weights']
        self.feature_stats = config['feature_stats']
        
        # Load model
        self.ranker_model = HybridRankerMLP(input_dim=8)
        self.ranker_model.load_state_dict(torch.load(model_path))
        self.ranker_model.eval()
        
        return True

def main():
    """Example usage of HybridRecommender."""
    
    # This would typically use trained content and CF models
    hybrid = HybridRecommender()
    
    # Sample user preferences
    user_prefs = UserPreferences(
        preferred_genres=['techno', 'electronic'],
        preferred_artists=[],
        preferred_venues=[],
        price_range=(100, 400),
        location_lat=55.6761,
        location_lon=12.5683,
        preferred_times=[20, 21, 22],
        preferred_days=[4, 5, 6]
    )
    
    # Sample event data
    events_data = [
        {
            'id': 'event1',
            'title': 'Techno Night',
            'venue_lat': 55.6826,
            'venue_lon': 12.5941,
            'price_min': 200,
            'price_max': 300,
            'date_time': datetime.now() + timedelta(days=3),
            'popularity_score': 0.8
        }
    ]
    
    # Get recommendations (would normally have trained models)
    recommendations = hybrid.recommend(
        user_prefs,
        candidate_event_ids=['event1'],
        events_data=events_data,
        num_recommendations=5,
        use_neural_ranker=False  # No trained ranker yet
    )
    
    print("Hybrid recommendations:")
    for event_id, score, explanation in recommendations:
        print(f"- {event_id}: {score:.3f}")
        print(f"  Reasons: {', '.join(explanation['reasons'])}")

if __name__ == "__main__":
    main()