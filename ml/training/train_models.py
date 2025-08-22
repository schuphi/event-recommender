#!/usr/bin/env python3
"""
Training script for both content-based and collaborative filtering models.
Handles data loading, preprocessing, training, and evaluation.
"""

import duckdb
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np

from ..models.content_based import ContentBasedRecommender
from ..models.collaborative_filtering import CollaborativeFilteringRecommender
from ..embeddings.content_embedder import ContentEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of recommendation models."""
    
    def __init__(
        self,
        db_path: str = "data/events.duckdb",
        models_dir: str = "ml/models"
    ):
        """
        Initialize model trainer.
        
        Args:
            db_path: Path to the DuckDB database
            models_dir: Directory to save trained models
        """
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.content_model: ContentBasedRecommender = None
        self.cf_model: CollaborativeFilteringRecommender = None
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load events and interactions from database.
        
        Returns:
            Tuple of (events, interactions)
        """
        logger.info("Loading data from database...")
        
        if not Path(self.db_path).exists():
            logger.error(f"Database not found: {self.db_path}")
            return [], []
        
        conn = duckdb.connect(self.db_path)
        
        # Load events with venue information
        events_query = """
        SELECT 
            e.id,
            e.title,
            e.description,
            e.date_time,
            e.price_min,
            e.price_max,
            e.artist_ids,
            e.popularity_score,
            e.h3_index,
            v.name as venue_name,
            v.lat as venue_lat,
            v.lon as venue_lon,
            v.neighborhood
        FROM events e
        JOIN venues v ON e.venue_id = v.id
        WHERE e.status = 'active'
        """
        
        events_result = conn.execute(events_query).fetchall()
        
        events = []
        for row in events_result:
            # Parse artist IDs (stored as JSON)
            artist_ids = json.loads(row[6]) if row[6] else []
            
            event = {
                'id': row[0],
                'title': row[1],
                'description': row[2] or '',
                'date_time': row[3],
                'price_min': row[4],
                'price_max': row[5],
                'artist_ids': artist_ids,
                'popularity_score': row[7] or 0.0,
                'h3_index': row[8],
                'venue_name': row[9],
                'venue_lat': row[10],
                'venue_lon': row[11],
                'neighborhood': row[12],
                'artists': [],  # Will be populated from artist_ids
                'genres': []    # Will be populated from artists
            }
            events.append(event)
        
        # Load artist information
        artists_query = "SELECT id, name, genres FROM artists"
        artists_result = conn.execute(artists_query).fetchall()
        
        artist_lookup = {}
        for row in artists_result:
            artist_id = row[0]
            artist_name = row[1]
            genres = json.loads(row[2]) if row[2] else []
            artist_lookup[artist_id] = {
                'name': artist_name,
                'genres': genres
            }
        
        # Populate artist names and genres in events
        for event in events:
            event_artists = []
            event_genres = set()
            
            for artist_id in event['artist_ids']:
                if artist_id in artist_lookup:
                    artist_info = artist_lookup[artist_id]
                    event_artists.append(artist_info['name'])
                    event_genres.update(artist_info['genres'])
            
            event['artists'] = event_artists
            event['genres'] = list(event_genres)
        
        # Load interactions
        interactions_query = """
        SELECT user_id, event_id, interaction_type, rating, timestamp
        FROM interactions
        ORDER BY timestamp
        """
        
        interactions_result = conn.execute(interactions_query).fetchall()
        
        interactions = []
        for row in interactions_result:
            interaction = {
                'user_id': row[0],
                'event_id': row[1],
                'interaction_type': row[2],
                'rating': row[3],
                'timestamp': row[4].timestamp() if row[4] else 0.0
            }
            interactions.append(interaction)
        
        conn.close()
        
        logger.info(f"Loaded {len(events)} events and {len(interactions)} interactions")
        return events, interactions
    
    def train_content_model(
        self,
        events: List[Dict],
        interactions: List[Dict] = None
    ) -> ContentBasedRecommender:
        """
        Train content-based recommendation model.
        
        Args:
            events: List of event dictionaries
            interactions: Optional interaction data for popularity
        
        Returns:
            Trained ContentBasedRecommender
        """
        logger.info("Training content-based model...")
        
        # Initialize content embedder and recommender
        embedder = ContentEmbedder(
            model_name="all-MiniLM-L6-v2",
            cache_dir=str(self.models_dir / "embeddings")
        )
        
        self.content_model = ContentBasedRecommender(
            embedder=embedder,
            model_cache_dir=str(self.models_dir / "content_based")
        )
        
        # Fit the model
        self.content_model.fit(events, interactions)
        
        logger.info("Content-based model training completed")
        return self.content_model
    
    def train_cf_model(
        self,
        interactions: List[Dict],
        embedding_dim: int = 64,
        num_epochs: int = 100
    ) -> CollaborativeFilteringRecommender:
        """
        Train collaborative filtering model.
        
        Args:
            interactions: List of interaction dictionaries
            embedding_dim: Embedding dimension for CF model
            num_epochs: Number of training epochs
        
        Returns:
            Trained CollaborativeFilteringRecommender
        """
        logger.info("Training collaborative filtering model...")
        
        if len(interactions) < 10:
            logger.warning("Insufficient interactions for collaborative filtering")
            return None
        
        self.cf_model = CollaborativeFilteringRecommender(
            embedding_dim=embedding_dim,
            model_cache_dir=str(self.models_dir / "collaborative")
        )
        
        # Fit the model
        self.cf_model.fit(
            interactions,
            num_epochs=num_epochs,
            batch_size=min(1024, len(interactions) // 4),
            negative_sampling_ratio=5,
            validation_split=0.1
        )
        
        logger.info("Collaborative filtering model training completed")
        return self.cf_model
    
    def evaluate_models(
        self,
        events: List[Dict],
        interactions: List[Dict],
        test_users: List[str] = None,
        k: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate trained models using held-out test data.
        
        Args:
            events: List of events
            interactions: List of interactions
            test_users: Optional list of users for testing
            k: Number of recommendations to evaluate (for Recall@k)
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating models...")
        
        if not self.content_model and not self.cf_model:
            logger.warning("No trained models to evaluate")
            return {}
        
        # Split interactions into train/test (80/20)
        interactions_sorted = sorted(interactions, key=lambda x: x['timestamp'])
        split_idx = int(len(interactions_sorted) * 0.8)
        train_interactions = interactions_sorted[:split_idx]
        test_interactions = interactions_sorted[split_idx:]
        
        # Group test interactions by user
        user_test_items = {}
        for interaction in test_interactions:
            user_id = interaction['user_id']
            event_id = interaction['event_id']
            
            # Only consider positive interactions for evaluation
            if interaction['interaction_type'] in ['like', 'going', 'went', 'save']:
                if user_id not in user_test_items:
                    user_test_items[user_id] = set()
                user_test_items[user_id].add(event_id)
        
        # Use provided test users or sample from available users
        if test_users is None:
            test_users = list(user_test_items.keys())[:50]  # Sample 50 users
        
        # Filter to users who have test items
        test_users = [u for u in test_users if u in user_test_items]
        
        if not test_users:
            logger.warning("No test users available for evaluation")
            return {}
        
        # Available events for recommendation
        event_ids = [event['id'] for event in events]
        
        results = {}
        
        # Evaluate content-based model
        if self.content_model:
            content_metrics = self._evaluate_model(
                self.content_model, 'content', test_users, user_test_items, event_ids, k
            )
            results['content_based'] = content_metrics
        
        # Evaluate collaborative filtering model
        if self.cf_model:
            cf_metrics = self._evaluate_model(
                self.cf_model, 'collaborative', test_users, user_test_items, event_ids, k
            )
            results['collaborative'] = cf_metrics
        
        logger.info("Model evaluation completed")
        return results
    
    def _evaluate_model(
        self,
        model,
        model_type: str,
        test_users: List[str],
        user_test_items: Dict[str, set],
        event_ids: List[str],
        k: int
    ) -> Dict[str, float]:
        """Evaluate a single model."""
        
        recall_scores = []
        precision_scores = []
        
        for user_id in test_users:
            if user_id not in user_test_items:
                continue
            
            test_items = user_test_items[user_id]
            
            try:
                if model_type == 'content':
                    # For content-based, we need user preferences
                    # Use a simple heuristic based on their interactions
                    user_prefs = self._infer_user_preferences(user_id)
                    if user_prefs is None:
                        continue
                    
                    recommendations = model.recommend(
                        user_prefs,
                        candidate_event_ids=event_ids,
                        num_recommendations=k
                    )
                
                elif model_type == 'collaborative':
                    recommendations = model.recommend(
                        user_id,
                        candidate_event_ids=event_ids,
                        num_recommendations=k,
                        exclude_seen=True
                    )
                
                # Extract recommended event IDs
                recommended_ids = {rec[0] for rec in recommendations}
                
                # Calculate metrics
                hits = len(recommended_ids & test_items)
                
                if len(test_items) > 0:
                    recall = hits / len(test_items)
                    recall_scores.append(recall)
                
                if len(recommended_ids) > 0:
                    precision = hits / len(recommended_ids)
                    precision_scores.append(precision)
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Calculate average metrics
        metrics = {}
        if recall_scores:
            metrics[f'recall@{k}'] = np.mean(recall_scores)
        if precision_scores:
            metrics[f'precision@{k}'] = np.mean(precision_scores)
        
        # F1 score
        if f'recall@{k}' in metrics and f'precision@{k}' in metrics:
            recall = metrics[f'recall@{k}']
            precision = metrics[f'precision@{k}']
            if recall + precision > 0:
                metrics[f'f1@{k}'] = 2 * (recall * precision) / (recall + precision)
        
        metrics['num_evaluated_users'] = len(test_users)
        
        return metrics
    
    def _infer_user_preferences(self, user_id: str):
        """Infer user preferences from their interaction history."""
        # This is a simplified implementation
        # In practice, you'd analyze the user's interaction history
        from ..models.content_based import UserPreferences
        
        # Return a default preference for evaluation
        return UserPreferences(
            preferred_genres=['electronic', 'techno'],
            preferred_artists=[],
            preferred_venues=[],
            price_range=(0, 1000),
            location_lat=55.6761,
            location_lon=12.5683,
            preferred_times=[20, 21, 22],
            preferred_days=[4, 5, 6]
        )
    
    def save_training_report(self, results: Dict):
        """Save training and evaluation results to a report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': [],
            'evaluation_results': results,
            'model_configs': {}
        }
        
        if self.content_model:
            report['models_trained'].append('content_based')
            report['model_configs']['content_based'] = {
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_dim': self.content_model.embedder.embedding_dim,
                'similarity_weights': self.content_model.similarity_weights
            }
        
        if self.cf_model:
            report['models_trained'].append('collaborative')
            report['model_configs']['collaborative'] = {
                'embedding_dim': self.cf_model.embedding_dim,
                'learning_rate': self.cf_model.learning_rate,
                'interaction_weights': self.cf_model.interaction_weights
            }
        
        report_path = self.models_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")

def main():
    """Main training script."""
    
    trainer = ModelTrainer()
    
    # Load data
    events, interactions = trainer.load_data()
    
    if not events:
        logger.error("No events found in database. Run data collection first.")
        return
    
    # Train models
    content_model = trainer.train_content_model(events, interactions)
    
    if interactions:
        cf_model = trainer.train_cf_model(interactions)
    else:
        logger.warning("No interactions found, skipping collaborative filtering")
        cf_model = None
    
    # Evaluate models
    if interactions:
        evaluation_results = trainer.evaluate_models(events, interactions)
        
        # Print results
        print("\nEvaluation Results:")
        for model_name, metrics in evaluation_results.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
    else:
        evaluation_results = {}
    
    # Save report
    trainer.save_training_report(evaluation_results)
    
    print("\nTraining completed! Models saved to ml/models/")

if __name__ == "__main__":
    main()