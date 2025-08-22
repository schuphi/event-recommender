"""
Machine learning models for the Copenhagen Event Recommender.
"""

from .content_based import ContentBasedRecommender, UserPreferences
from .collaborative_filtering import CollaborativeFilteringRecommender, BPRModel, Interaction
from .hybrid_ranker import HybridRecommender, HybridRankerMLP, EventCandidate

__all__ = [
    'ContentBasedRecommender', 'UserPreferences',
    'CollaborativeFilteringRecommender', 'BPRModel', 'Interaction',
    'HybridRecommender', 'HybridRankerMLP', 'EventCandidate'
]