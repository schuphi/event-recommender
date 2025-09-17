#!/usr/bin/env python3
"""
Bayesian Personalized Ranking (BPR) collaborative filtering model using PyTorch.
Learns user-event preferences from implicit feedback (likes, going, saves).
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
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    """User-event interaction with implicit feedback."""
    user_id: str
    event_id: str
    interaction_type: str  # like, dislike, going, went, save
    rating: Optional[float]  # For 'went' interactions
    timestamp: float

class BPRModel(nn.Module):
    """Bayesian Personalized Ranking matrix factorization model."""
    
    def __init__(
        self,
        num_users: int,
        num_events: int,
        embedding_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize BPR model.
        
        Args:
            num_users: Number of unique users
            num_events: Number of unique events
            embedding_dim: Latent factor dimension
            dropout: Dropout rate for regularization
        """
        super(BPRModel, self).__init__()
        
        self.num_users = num_users
        self.num_events = num_events
        self.embedding_dim = embedding_dim
        
        # User and event embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.event_embeddings = nn.Embedding(num_events, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.event_bias = nn.Embedding(num_events, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with small random values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.1)
        nn.init.normal_(self.event_embeddings.weight, mean=0, std=0.1)
        nn.init.normal_(self.user_bias.weight, mean=0, std=0.01)
        nn.init.normal_(self.event_bias.weight, mean=0, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, event_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict user-event preferences.
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            event_ids: Tensor of event indices [batch_size]
        
        Returns:
            Predicted scores [batch_size]
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]
        event_emb = self.event_embeddings(event_ids)  # [batch_size, embedding_dim]
        
        # Apply dropout
        user_emb = self.dropout(user_emb)
        event_emb = self.dropout(event_emb)
        
        # Compute dot product
        scores = torch.sum(user_emb * event_emb, dim=1)  # [batch_size]
        
        # Add bias terms
        user_b = self.user_bias(user_ids).squeeze()  # [batch_size]
        event_b = self.event_bias(event_ids).squeeze()  # [batch_size]
        
        scores = scores + user_b + event_b + self.global_bias
        
        return scores
    
    def predict_user_events(self, user_id: int, event_ids: List[int]) -> torch.Tensor:
        """Predict scores for a user on multiple events."""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(event_ids), dtype=torch.long)
            event_tensor = torch.tensor(event_ids, dtype=torch.long)
            scores = self.forward(user_tensor, event_tensor)
        return scores

class CollaborativeFilteringRecommender:
    """BPR-based collaborative filtering recommender."""
    
    def __init__(
        self,
        embedding_dim: int = 64,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        model_cache_dir: str = "ml/models/collaborative"
    ):
        """
        Initialize collaborative filtering recommender.
        
        Args:
            embedding_dim: Latent factor dimension
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization strength
            model_cache_dir: Directory to cache trained models
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cache_dir = Path(model_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and mappings
        self.model: Optional[BPRModel] = None
        self.user_to_idx: Dict[str, int] = {}
        self.event_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.idx_to_event: Dict[int, str] = {}
        
        # Training data
        self.interactions: List[Interaction] = []
        self.user_items: Dict[int, Set[int]] = defaultdict(set)  # user_idx -> event_idx set
        
        # Interaction weights (how much to weight different interaction types)
        self.interaction_weights = {
            'like': 1.0,
            'going': 2.0,
            'went': 1.5,
            'save': 0.8,
            'dislike': -0.5  # Negative feedback
        }
    
    def fit(
        self,
        interactions: List[Dict],
        num_epochs: int = 100,
        batch_size: int = 1024,
        negative_sampling_ratio: int = 5,
        validation_split: float = 0.1
    ):
        """
        Fit the BPR model on interaction data.
        
        Args:
            interactions: List of interaction dictionaries
            num_epochs: Number of training epochs
            batch_size: Training batch size
            negative_sampling_ratio: Number of negative samples per positive
            validation_split: Fraction of data for validation
        """
        logger.info(f"Fitting collaborative filtering model on {len(interactions)} interactions...")
        
        # Parse and prepare interactions
        self._prepare_interactions(interactions)
        
        if len(self.user_to_idx) < 2 or len(self.event_to_idx) < 2:
            logger.warning("Insufficient data for collaborative filtering")
            return
        
        # Create model
        self.model = BPRModel(
            num_users=len(self.user_to_idx),
            num_events=len(self.event_to_idx),
            embedding_dim=self.embedding_dim
        )
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Prepare training data
        positive_pairs = self._get_positive_pairs()
        
        # Split into train/validation
        split_idx = int(len(positive_pairs) * (1 - validation_split))
        train_pairs = positive_pairs[:split_idx]
        val_pairs = positive_pairs[split_idx:]
        
        logger.info(f"Training on {len(train_pairs)} positive pairs, validating on {len(val_pairs)}")
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            random.shuffle(train_pairs)
            
            for i in range(0, len(train_pairs), batch_size):
                batch_pairs = train_pairs[i:i + batch_size]
                
                # Generate negative samples for this batch
                batch_triplets = self._generate_bpr_triplets(
                    batch_pairs, negative_sampling_ratio
                )
                
                if not batch_triplets:
                    continue
                
                # Convert to tensors
                users, pos_items, neg_items = zip(*batch_triplets)
                user_tensor = torch.tensor(users, dtype=torch.long)
                pos_tensor = torch.tensor(pos_items, dtype=torch.long)
                neg_tensor = torch.tensor(neg_items, dtype=torch.long)
                
                # Forward pass
                pos_scores = self.model(user_tensor, pos_tensor)
                neg_scores = self.model(user_tensor, neg_tensor)
                
                # BPR loss: positive items should be ranked higher than negative
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Log progress
            if epoch % 10 == 0 and num_batches > 0:
                avg_loss = epoch_loss / num_batches
                val_loss = self._compute_validation_loss(val_pairs)
                logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save trained model
        self._save_model()
        logger.info("Collaborative filtering model training completed")
    
    def _prepare_interactions(self, interactions: List[Dict]):
        """Parse and prepare interaction data."""
        self.interactions = []
        user_set = set()
        event_set = set()
        
        for interaction in interactions:
            user_id = interaction['user_id']
            event_id = interaction['event_id']
            interaction_type = interaction['interaction_type']
            timestamp = interaction.get('timestamp', 0.0)
            rating = interaction.get('rating')
            
            self.interactions.append(Interaction(
                user_id=user_id,
                event_id=event_id,
                interaction_type=interaction_type,
                rating=rating,
                timestamp=timestamp
            ))
            
            user_set.add(user_id)
            event_set.add(event_id)
        
        # Create user and event mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(user_set))}
        self.event_to_idx = {event: idx for idx, event in enumerate(sorted(event_set))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_event = {idx: event for event, idx in self.event_to_idx.items()}
        
        # Build user-item matrix
        self.user_items = defaultdict(set)
        
        for interaction in self.interactions:
            user_idx = self.user_to_idx[interaction.user_id]
            event_idx = self.event_to_idx[interaction.event_id]
            
            # Weight the interaction
            weight = self.interaction_weights.get(interaction.interaction_type, 0.5)
            
            # For 'went' interactions, use rating if available
            if interaction.interaction_type == 'went' and interaction.rating:
                weight *= (interaction.rating / 5.0)  # Normalize rating to 0-1
            
            # Only add positive interactions to user_items
            if weight > 0:
                self.user_items[user_idx].add(event_idx)
    
    def _get_positive_pairs(self) -> List[Tuple[int, int]]:
        """Get all positive user-event pairs."""
        positive_pairs = []
        
        for user_idx, event_indices in self.user_items.items():
            for event_idx in event_indices:
                positive_pairs.append((user_idx, event_idx))
        
        return positive_pairs
    
    def _generate_bpr_triplets(
        self, 
        positive_pairs: List[Tuple[int, int]], 
        negative_ratio: int
    ) -> List[Tuple[int, int, int]]:
        """Generate (user, positive_item, negative_item) triplets for BPR."""
        
        triplets = []
        
        for user_idx, pos_item_idx in positive_pairs:
            user_positive_items = self.user_items[user_idx]
            
            # Sample negative items for this user
            for _ in range(negative_ratio):
                # Sample random negative item
                neg_item_idx = random.randint(0, len(self.event_to_idx) - 1)
                
                # Ensure it's actually negative (not in user's positive items)
                attempts = 0
                while neg_item_idx in user_positive_items and attempts < 10:
                    neg_item_idx = random.randint(0, len(self.event_to_idx) - 1)
                    attempts += 1
                
                if neg_item_idx not in user_positive_items:
                    triplets.append((user_idx, pos_item_idx, neg_item_idx))
        
        return triplets
    
    def _compute_validation_loss(self, val_pairs: List[Tuple[int, int]]) -> float:
        """Compute validation loss on held-out data."""
        if not val_pairs:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_triplets = 0
        
        with torch.no_grad():
            # Generate some negative samples for validation
            val_triplets = self._generate_bpr_triplets(val_pairs, negative_sampling_ratio=3)
            
            if val_triplets:
                users, pos_items, neg_items = zip(*val_triplets)
                user_tensor = torch.tensor(users, dtype=torch.long)
                pos_tensor = torch.tensor(pos_items, dtype=torch.long)
                neg_tensor = torch.tensor(neg_items, dtype=torch.long)
                
                pos_scores = self.model(user_tensor, pos_tensor)
                neg_scores = self.model(user_tensor, neg_tensor)
                
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                total_loss = loss.item()
                num_triplets = len(val_triplets)
        
        self.model.train()
        return total_loss
    
    def recommend(
        self,
        user_id: str,
        candidate_event_ids: List[str] = None,
        num_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Generate collaborative filtering recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            candidate_event_ids: Optional list of candidate events
            num_recommendations: Number of recommendations to return
            exclude_seen: Whether to exclude events user has already interacted with
        
        Returns:
            List of (event_id, score) tuples sorted by score
        """
        
        if not self.model or user_id not in self.user_to_idx:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Determine candidate events
        if candidate_event_ids:
            candidate_indices = []
            candidate_ids = []
            for event_id in candidate_event_ids:
                if event_id in self.event_to_idx:
                    candidate_indices.append(self.event_to_idx[event_id])
                    candidate_ids.append(event_id)
        else:
            candidate_indices = list(range(len(self.event_to_idx)))
            candidate_ids = [self.idx_to_event[idx] for idx in candidate_indices]
        
        if not candidate_indices:
            return []
        
        # Exclude seen events if requested
        if exclude_seen:
            seen_events = self.user_items[user_idx]
            filtered_indices = []
            filtered_ids = []
            
            for idx, event_id in zip(candidate_indices, candidate_ids):
                if idx not in seen_events:
                    filtered_indices.append(idx)
                    filtered_ids.append(event_id)
            
            candidate_indices = filtered_indices
            candidate_ids = filtered_ids
        
        if not candidate_indices:
            return []
        
        # Predict scores
        scores = self.model.predict_user_events(user_idx, candidate_indices)
        scores = scores.numpy()
        
        # Sort by score
        scored_events = list(zip(candidate_ids, scores))
        scored_events.sort(key=lambda x: x[1], reverse=True)
        
        return scored_events[:num_recommendations]
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get learned embedding for a user."""
        if not self.model or user_id not in self.user_to_idx:
            return None
        
        user_idx = self.user_to_idx[user_id]
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.user_embeddings(torch.tensor([user_idx])).numpy()[0]
        return embedding
    
    def get_event_embedding(self, event_id: str) -> Optional[np.ndarray]:
        """Get learned embedding for an event."""
        if not self.model or event_id not in self.event_to_idx:
            return None
        
        event_idx = self.event_to_idx[event_id]
        self.model.eval()
        with torch.no_grad():
            embedding = self.model.event_embeddings(torch.tensor([event_idx])).numpy()[0]
        return embedding
    
    def get_similar_users(self, user_id: str, num_users: int = 10) -> List[Tuple[str, float]]:
        """Find users with similar preferences."""
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            return []
        
        similarities = []
        
        for other_user_id, other_user_idx in self.user_to_idx.items():
            if other_user_id == user_id:
                continue
            
            other_emb = self.get_user_embedding(other_user_id)
            if other_emb is not None:
                # Cosine similarity
                similarity = np.dot(user_emb, other_emb) / (
                    np.linalg.norm(user_emb) * np.linalg.norm(other_emb)
                )
                similarities.append((other_user_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_users]
    
    def _save_model(self):
        """Save trained model and mappings."""
        if not self.model:
            return
        
        # Save PyTorch model
        model_path = self.cache_dir / "bpr_model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save mappings and metadata
        metadata = {
            'user_to_idx': self.user_to_idx,
            'event_to_idx': self.event_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_event': self.idx_to_event,
            'embedding_dim': self.embedding_dim,
            'interaction_weights': self.interaction_weights,
            'num_users': len(self.user_to_idx),
            'num_events': len(self.event_to_idx)
        }
        
        metadata_path = self.cache_dir / "cf_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Collaborative filtering model saved to {self.cache_dir}")
    
    def load_model(self) -> bool:
        """Load trained model from cache."""
        model_path = self.cache_dir / "bpr_model.pth"
        metadata_path = self.cache_dir / "cf_metadata.json"
        
        if not (model_path.exists() and metadata_path.exists()):
            logger.warning("No cached collaborative filtering model found")
            return False
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.user_to_idx = metadata['user_to_idx']
        self.event_to_idx = metadata['event_to_idx']
        self.idx_to_user = {int(k): v for k, v in metadata['idx_to_user'].items()}
        self.idx_to_event = {int(k): v for k, v in metadata['idx_to_event'].items()}
        self.embedding_dim = metadata['embedding_dim']
        self.interaction_weights = metadata['interaction_weights']
        
        # Create and load model
        self.model = BPRModel(
            num_users=metadata['num_users'],
            num_events=metadata['num_events'],
            embedding_dim=self.embedding_dim
        )
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        logger.info(f"Collaborative filtering model loaded from {self.cache_dir}")
        return True

def main():
    """Example usage of CollaborativeFilteringRecommender."""
    
    # Sample interactions
    interactions = [
        {'user_id': 'user1', 'event_id': 'event1', 'interaction_type': 'like'},
        {'user_id': 'user1', 'event_id': 'event2', 'interaction_type': 'going'},
        {'user_id': 'user1', 'event_id': 'event3', 'interaction_type': 'went', 'rating': 4.0},
        {'user_id': 'user2', 'event_id': 'event1', 'interaction_type': 'like'},
        {'user_id': 'user2', 'event_id': 'event4', 'interaction_type': 'save'},
        {'user_id': 'user3', 'event_id': 'event2', 'interaction_type': 'like'},
        {'user_id': 'user3', 'event_id': 'event3', 'interaction_type': 'going'},
        {'user_id': 'user3', 'event_id': 'event4', 'interaction_type': 'dislike'},
    ]
    
    # Initialize and fit recommender
    recommender = CollaborativeFilteringRecommender()
    recommender.fit(interactions, num_epochs=50)
    
    # Get recommendations
    recommendations = recommender.recommend('user1', num_recommendations=3)
    
    print("Collaborative filtering recommendations for user1:")
    for event_id, score in recommendations:
        print(f"- {event_id}: {score:.3f}")
    
    # Find similar users
    similar_users = recommender.get_similar_users('user1', num_users=2)
    print(f"\nUsers similar to user1: {similar_users}")

if __name__ == "__main__":
    main()