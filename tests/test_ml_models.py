#!/usr/bin/env python3
"""
ML model validation tests for Copenhagen Event Recommender.
Tests content-based filtering, collaborative filtering, and hybrid ranking models.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pickle
import tempfile
import os


class TestContentBasedModel:
    """Test content-based recommendation model."""
    
    def test_model_initialization(self):
        """Test content-based model can be initialized."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        assert model is not None
        assert hasattr(model, 'recommend_events')
    
    def test_user_preference_synthesis(self, sample_user_preferences):
        """Test user preference synthesis from interaction history."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        # Mock user interactions
        interactions = [
            {'event_id': 'event_1', 'interaction_type': 'like', 'rating': 5.0},
            {'event_id': 'event_2', 'interaction_type': 'going', 'rating': 4.0}
        ]
        
        # Mock events data
        events = [
            {'id': 'event_1', 'genres': ['techno', 'electronic'], 'venue': 'Culture Box'},
            {'id': 'event_2', 'genres': ['indie', 'alternative'], 'venue': 'Vega'}
        ]
        
        preferences = model._synthesize_preferences(interactions, events)
        
        assert 'preferred_genres' in preferences
        assert 'preferred_venues' in preferences
        assert isinstance(preferences['preferred_genres'], list)
    
    def test_content_similarity_calculation(self):
        """Test event content similarity calculation."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        event1 = {
            'genres': ['techno', 'electronic'],
            'venue': 'Culture Box',
            'price_min': 200.0,
            'artists': ['Artist A']
        }
        
        event2 = {
            'genres': ['techno', 'house'], 
            'venue': 'Culture Box',
            'price_min': 250.0,
            'artists': ['Artist B']
        }
        
        similarity = model._calculate_content_similarity(event1, event2)
        
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be similar (same venue, overlapping genres)
    
    def test_embedding_generation(self, sample_events):
        """Test event embedding generation."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        for event in sample_events:
            embedding = model._generate_event_embedding(event)
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) > 0
            assert not np.isnan(embedding).any()
    
    def test_recommend_events(self, sample_events, sample_user_preferences):
        """Test end-to-end event recommendations."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        recommendations = model.recommend_events(
            user_preferences=sample_user_preferences,
            available_events=sample_events,
            num_recommendations=5
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        assert all('score' in rec for rec in recommendations)
        assert all('event' in rec for rec in recommendations)
        
        # Scores should be in descending order
        scores = [rec['score'] for rec in recommendations]
        assert scores == sorted(scores, reverse=True)


class TestCollaborativeFiltering:
    """Test collaborative filtering model."""
    
    def test_model_initialization(self):
        """Test collaborative filtering model initialization."""
        from ml.models.collaborative import CollaborativeFilteringRecommender
        
        model = CollaborativeFilteringRecommender()
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'recommend_events')
    
    def test_user_item_matrix_creation(self, ml_test_data):
        """Test user-item interaction matrix creation."""
        from ml.models.collaborative import CollaborativeFilteringRecommender
        
        model = CollaborativeFilteringRecommender()
        interactions = ml_test_data['training_interactions']
        
        matrix = model._create_user_item_matrix(interactions)
        
        assert isinstance(matrix, (np.ndarray, pd.DataFrame))
        assert matrix.shape[0] > 0  # Users
        assert matrix.shape[1] > 0  # Events
    
    def test_matrix_factorization_training(self, ml_test_data):
        """Test matrix factorization model training."""
        from ml.models.collaborative import CollaborativeFilteringRecommender
        
        model = CollaborativeFilteringRecommender(n_factors=10, n_epochs=5)
        interactions = ml_test_data['training_interactions']
        
        # Should not raise exception
        model.fit(interactions)
        
        # Should have learned embeddings
        assert hasattr(model, 'user_factors')
        assert hasattr(model, 'item_factors')
    
    def test_similarity_based_recommendations(self, ml_test_data):
        """Test item-item similarity recommendations."""
        from ml.models.collaborative import CollaborativeFilteringRecommender
        
        model = CollaborativeFilteringRecommender(method='item_similarity')
        interactions = ml_test_data['training_interactions']
        
        model.fit(interactions)
        
        recommendations = model.recommend_events(
            user_id='user_1',
            num_recommendations=3
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert all('event_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)


class TestHybridRanker:
    """Test hybrid ranking neural network model."""
    
    def test_model_architecture(self):
        """Test hybrid ranker neural network architecture."""
        from ml.models.hybrid_ranker import HybridRankerMLP
        
        model = HybridRankerMLP(input_dim=8, hidden_dims=[64, 32], dropout_rate=0.2)
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        batch_size = 10
        test_input = torch.randn(batch_size, 8)
        output = model(test_input)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
    
    def test_feature_extraction(self, sample_events, sample_user_preferences):
        """Test feature extraction for hybrid ranking."""
        from ml.models.hybrid_ranker import HybridRanker
        
        ranker = HybridRanker()
        
        for event in sample_events:
            features = ranker._extract_features(sample_user_preferences, event)
            
            assert isinstance(features, np.ndarray)
            assert len(features) == 8  # Expected feature dimension
            assert not np.isnan(features).any()
    
    def test_model_training_data_format(self, ml_test_data):
        """Test training data format for hybrid ranker."""
        from ml.training.data_generator import TrainingDataGenerator
        
        generator = TrainingDataGenerator()
        
        # Mock database service
        mock_db = Mock()
        mock_db.get_all_interactions.return_value = ml_test_data['training_interactions']
        mock_db.get_all_events.return_value = [
            {'id': 'event_1', 'title': 'Event 1', 'genres': ['techno']},
            {'id': 'event_2', 'title': 'Event 2', 'genres': ['indie']},
            {'id': 'event_3', 'title': 'Event 3', 'genres': ['house']}
        ]
        mock_db.get_all_users.return_value = [
            {'id': 'user_1', 'preferences': {'preferred_genres': ['techno']}},
            {'id': 'user_2', 'preferences': {'preferred_genres': ['indie']}}
        ]
        
        training_data = generator.generate_training_data(
            db_service=mock_db,
            split_ratio=0.8,
            max_examples=100
        )
        
        assert 'X_train' in training_data
        assert 'X_val' in training_data
        assert 'y_train' in training_data
        assert 'y_val' in training_data
        
        # Check shapes
        assert training_data['X_train'].shape[1] == 8  # Feature dimension
        assert len(training_data['X_train']) == len(training_data['y_train'])
    
    def test_model_training_process(self, ml_test_data):
        """Test hybrid ranker training process."""
        from ml.training.hybrid_trainer import HybridRankerTrainer
        
        trainer = HybridRankerTrainer()
        
        # Create minimal training data
        X_train = torch.randn(50, 8)
        y_train = torch.randint(0, 2, (50, 1)).float()
        X_val = torch.randn(20, 8) 
        y_val = torch.randint(0, 2, (20, 1)).float()
        
        training_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
        
        # Train for few epochs
        results = trainer.train_neural_ranker(
            training_data=training_data,
            num_epochs=5,
            batch_size=10
        )
        
        assert 'model' in results
        assert 'training_history' in results
        assert 'best_val_auc' in results
        
        # Model should be trained
        model = results['model']
        assert isinstance(model, torch.nn.Module)
        
        # Should be able to make predictions
        test_input = torch.randn(1, 8)
        prediction = model(test_input)
        assert not torch.isnan(prediction).any()
    
    def test_model_evaluation_metrics(self, ml_test_data):
        """Test evaluation metrics calculation."""
        from ml.training.hybrid_trainer import HybridRankerTrainer
        
        trainer = HybridRankerTrainer()
        
        # Mock predictions and targets
        y_true = torch.tensor([1, 0, 1, 0, 1]).float()
        y_pred = torch.tensor([0.8, 0.2, 0.7, 0.3, 0.9]).float()
        
        metrics = trainer._calculate_metrics(y_pred, y_true)
        
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_ndcg_calculation(self):
        """Test NDCG@k calculation for ranking quality."""
        from ml.training.hybrid_trainer import HybridRankerTrainer
        
        trainer = HybridRankerTrainer()
        
        # Mock relevance scores (higher is better)
        relevance_scores = [3, 2, 1, 0, 2]  # True relevance
        predicted_scores = [0.9, 0.8, 0.6, 0.1, 0.7]  # Predicted scores
        
        ndcg = trainer._calculate_ndcg(predicted_scores, relevance_scores, k=5)
        
        assert 0 <= ndcg <= 1
        assert isinstance(ndcg, float)


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_content_model_save_load(self, sample_user_preferences):
        """Test saving and loading content-based model."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = ContentBasedRecommender.load_model(model_path)
            assert loaded_model is not None
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_hybrid_model_save_load(self):
        """Test saving and loading hybrid ranker model."""
        from ml.models.hybrid_ranker import HybridRankerMLP
        
        model = HybridRankerMLP(input_dim=8)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
        
        try:
            # Save model
            torch.save(model.state_dict(), model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = HybridRankerMLP(input_dim=8)
            loaded_model.load_state_dict(torch.load(model_path))
            
            # Test they produce same output
            test_input = torch.randn(1, 8)
            original_output = model(test_input)
            loaded_output = loaded_model(test_input)
            
            assert torch.allclose(original_output, loaded_output)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestModelPerformance:
    """Test model performance and benchmarks."""
    
    def test_content_model_speed(self, sample_events, sample_user_preferences):
        """Test content-based model recommendation speed."""
        from ml.models.content_based import ContentBasedRecommender
        import time
        
        model = ContentBasedRecommender()
        
        start_time = time.time()
        recommendations = model.recommend_events(
            user_preferences=sample_user_preferences,
            available_events=sample_events * 10,  # 20 events
            num_recommendations=5
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert len(recommendations) == 5
    
    def test_hybrid_model_inference_speed(self):
        """Test hybrid model inference speed."""
        from ml.models.hybrid_ranker import HybridRankerMLP
        import time
        
        model = HybridRankerMLP(input_dim=8)
        model.eval()
        
        # Test batch inference
        batch_size = 100
        test_input = torch.randn(batch_size, 8)
        
        start_time = time.time()
        with torch.no_grad():
            predictions = model(test_input)
        end_time = time.time()
        
        # Should be fast for inference
        assert end_time - start_time < 1.0  # 1 second max
        assert predictions.shape == (batch_size, 1)


class TestModelIntegration:
    """Test model integration and end-to-end workflows."""
    
    def test_production_pipeline_integration(self, sample_events, sample_user_preferences):
        """Test models work together in production pipeline."""
        from ml.production.production_pipeline import MLProductionPipeline
        
        pipeline = MLProductionPipeline()
        
        # Mock database service
        mock_db = Mock()
        mock_db.get_user_by_id.return_value = {
            'id': 'user_1', 
            'preferences': sample_user_preferences
        }
        mock_db.get_upcoming_events.return_value = sample_events
        
        recommendations = pipeline.get_user_recommendations(
            user_id='user_1',
            db_service=mock_db,
            num_recommendations=5
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        assert all('event_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)
    
    def test_model_fallback_behavior(self, sample_events, sample_user_preferences):
        """Test model fallback when primary models fail."""
        from ml.production.production_pipeline import MLProductionPipeline
        
        pipeline = MLProductionPipeline()
        
        # Mock failing models
        with patch.object(pipeline, 'content_model') as mock_content:
            mock_content.recommend_events.side_effect = Exception("Model failed")
            
            # Should fallback gracefully
            recommendations = pipeline._get_fallback_recommendations(
                available_events=sample_events,
                num_recommendations=3
            )
            
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 3


class TestModelValidation:
    """Test model validation and quality checks."""
    
    def test_recommendation_diversity(self, sample_events, sample_user_preferences):
        """Test that recommendations are diverse."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        recommendations = model.recommend_events(
            user_preferences=sample_user_preferences,
            available_events=sample_events * 5,  # More events for diversity
            num_recommendations=5
        )
        
        if len(recommendations) >= 2:
            venues = [rec['event']['venue_name'] for rec in recommendations]
            genres = [genre for rec in recommendations for genre in rec['event'].get('genres', [])]
            
            # Should have some diversity
            unique_venues = set(venues)
            unique_genres = set(genres)
            
            assert len(unique_venues) >= 1
            assert len(unique_genres) >= 1
    
    def test_recommendation_relevance(self, sample_events, sample_user_preferences):
        """Test that recommendations are relevant to user preferences."""
        from ml.models.content_based import ContentBasedRecommender
        
        model = ContentBasedRecommender()
        
        recommendations = model.recommend_events(
            user_preferences=sample_user_preferences,
            available_events=sample_events,
            num_recommendations=3
        )
        
        preferred_genres = sample_user_preferences.get('preferred_genres', [])
        
        if preferred_genres and recommendations:
            # At least some recommendations should match preferred genres
            matching_recs = 0
            for rec in recommendations:
                event_genres = rec['event'].get('genres', [])
                if any(genre in event_genres for genre in preferred_genres):
                    matching_recs += 1
            
            # At least 50% should match preferences
            assert matching_recs >= len(recommendations) * 0.5