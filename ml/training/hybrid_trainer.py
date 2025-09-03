#!/usr/bin/env python3
"""
Enhanced training infrastructure for hybrid ranker with proper evaluation metrics,
hyperparameter tuning, and production training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import optuna
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, ndcg_score
import matplotlib.pyplot as plt
from collections import defaultdict

from ..models.hybrid_ranker import HybridRankerMLP
from .data_generator import TrainingExample, TrainingDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDataset(Dataset):
    """PyTorch Dataset for hybrid ranker training."""
    
    def __init__(self, examples: List[TrainingExample]):
        """
        Initialize dataset from training examples.
        
        Args:
            examples: List of TrainingExample objects
        """
        self.examples = examples
        
        # Extract features and labels
        self.features = []
        self.labels = []
        
        for example in examples:
            feature_vector = [
                example.content_score,
                example.cf_score,
                self._normalize_distance(example.distance_km),
                example.price_score,
                example.recency_score,
                example.popularity_score,
                example.diversity_score,
                example.time_match_score
            ]
            self.features.append(feature_vector)
            self.labels.append(float(example.label))
    
    def _normalize_distance(self, distance_km: float) -> float:
        """Normalize distance to 0-1 range."""
        return 1.0 / (1.0 + distance_km / 5.0)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

class EarlyStopping:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class HybridRankerTrainer:
    """Enhanced trainer for hybrid ranker with evaluation and hyperparameter tuning."""
    
    def __init__(
        self,
        model_dir: str = "ml/models/hybrid",
        device: Optional[str] = None
    ):
        """
        Initialize hybrid ranker trainer.
        
        Args:
            model_dir: Directory to save trained models
            device: PyTorch device ('cpu', 'cuda', or None for auto)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Training state
        self.model = None
        self.best_model_state = None
        self.training_history = []
        
    def train(
        self,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Train hybrid ranker with given configuration.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            config: Training configuration
            
        Returns:
            Training results dictionary
        """
        # Default configuration
        default_config = {
            'hidden_dims': [64, 32],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 128,
            'num_epochs': 100,
            'patience': 15,
            'min_delta': 0.001,
        }
        
        config = {**default_config, **(config or {})}
        
        logger.info(f"Training hybrid ranker with config: {config}")
        
        # Create datasets and dataloaders
        train_dataset = HybridDataset(train_examples)
        val_dataset = HybridDataset(val_examples)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        self.model = HybridRankerMLP(
            input_dim=8,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Use BCELoss since we're doing binary classification
        criterion = nn.BCELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config['patience'], 
            min_delta=config['min_delta']
        )
        
        # Training loop
        self.training_history = []
        best_val_auc = 0.0
        
        for epoch in range(config['num_epochs']):
            # Training phase
            train_loss, train_auc = self._train_epoch(
                train_loader, optimizer, criterion
            )
            
            # Validation phase
            val_loss, val_auc, val_metrics = self._validate_epoch(
                val_loader, criterion
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record history
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                **val_metrics
            }
            self.training_history.append(epoch_history)
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.best_model_state = self.model.state_dict().copy()
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch < 10:
                logger.info(
                    f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                    f"Train AUC={train_auc:.4f}, Val Loss={val_loss:.4f}, "
                    f"Val AUC={val_auc:.4f}"
                )
            
            # Early stopping check
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        final_metrics = self._final_evaluation(val_examples)
        
        # Save model and results
        self._save_model(config, final_metrics)
        
        results = {
            'best_val_auc': best_val_auc,
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'config': config
        }
        
        logger.info(f"Training completed. Best validation AUC: {best_val_auc:.4f}")
        return results
    
    def _train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device).unsqueeze(1)
            
            # Forward pass
            predictions = self.model(batch_features)
            loss = criterion(predictions, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy().flatten())
            all_labels.extend(batch_labels.detach().cpu().numpy().flatten())
        
        avg_loss = total_loss / len(train_loader)
        auc_score = roc_auc_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else 0.5
        
        return avg_loss, auc_score
    
    def _validate_epoch(
        self, 
        val_loader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float, Dict[str, float]]:
        """Validate for one epoch."""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)
                
                # Forward pass
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_labels)
                
                # Track metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(batch_labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        if len(set(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_predictions)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
            pr_auc = auc(recall, precision)
            
            # Additional metrics
            predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
            accuracy = (predictions_binary == np.array(all_labels)).mean()
            
            metrics = {
                'pr_auc': pr_auc,
                'accuracy': accuracy
            }
        else:
            auc_score = 0.5
            metrics = {'pr_auc': 0.5, 'accuracy': 0.5}
        
        return avg_loss, auc_score, metrics
    
    def _final_evaluation(self, val_examples: List[TrainingExample]) -> Dict[str, float]:
        """Comprehensive final evaluation."""
        
        logger.info("Running final evaluation...")
        
        # Group by user for ranking metrics
        user_examples = defaultdict(list)
        for example in val_examples:
            user_examples[example.user_id].append(example)
        
        # Calculate predictions for all examples
        val_dataset = HybridDataset(val_examples)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                predictions = self.model(batch_features)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(batch_labels.numpy().flatten())
        
        # Basic classification metrics
        metrics = {}
        
        if len(set(all_labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_predictions)
            
            precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
            metrics['pr_auc'] = auc(recall, precision)
            
            predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
            metrics['accuracy'] = (predictions_binary == np.array(all_labels)).mean()
            metrics['precision'] = ((predictions_binary == 1) & (np.array(all_labels) == 1)).sum() / (predictions_binary == 1).sum() if (predictions_binary == 1).sum() > 0 else 0
            metrics['recall'] = ((predictions_binary == 1) & (np.array(all_labels) == 1)).sum() / (np.array(all_labels) == 1).sum() if (np.array(all_labels) == 1).sum() > 0 else 0
        
        # Ranking metrics (NDCG)
        ndcg_scores = []
        for user_id, user_exs in user_examples.items():
            if len(user_exs) < 2:
                continue
            
            # Get predictions for this user
            user_indices = [i for i, ex in enumerate(val_examples) if ex.user_id == user_id]
            user_predictions = [all_predictions[i] for i in user_indices]
            user_labels = [all_labels[i] for i in user_indices]
            
            # Calculate NDCG@10
            if len(set(user_labels)) > 1:
                try:
                    ndcg = ndcg_score([user_labels], [user_predictions], k=10)
                    ndcg_scores.append(ndcg)
                except:
                    continue
        
        if ndcg_scores:
            metrics['ndcg@10'] = np.mean(ndcg_scores)
            metrics['num_users_evaluated'] = len(ndcg_scores)
        
        logger.info(f"Final metrics: {metrics}")
        return metrics
    
    def hyperparameter_tuning(
        self,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        n_trials: int = 50,
        timeout: int = 3600  # 1 hour
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning using Optuna.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters and results
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        def objective(trial):
            """Optuna objective function."""
            
            # Sample hyperparameters
            config = {
                'hidden_dims': [
                    trial.suggest_int('hidden1', 32, 128),
                    trial.suggest_int('hidden2', 16, 64)
                ],
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'num_epochs': 50,  # Reduced for tuning
                'patience': 10,
                'min_delta': 0.001
            }
            
            try:
                # Train with this configuration
                results = self.train(train_examples, val_examples, config)
                
                # Return validation AUC as objective
                return results['best_val_auc']
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation AUC: {best_value:.4f}")
        
        # Train final model with best parameters
        best_config = {
            'hidden_dims': [best_params['hidden1'], best_params['hidden2']],
            'dropout': best_params['dropout'],
            'learning_rate': best_params['learning_rate'],
            'weight_decay': best_params['weight_decay'],
            'batch_size': best_params['batch_size'],
            'num_epochs': 100,  # Full training
            'patience': 15,
            'min_delta': 0.001
        }
        
        final_results = self.train(train_examples, val_examples, best_config)
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'final_results': final_results,
            'study': study
        }
    
    def _save_model(self, config: Dict[str, Any], metrics: Dict[str, float]):
        """Save trained model and metadata."""
        
        # Save model state
        model_path = self.model_dir / "hybrid_ranker.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save training metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'final_metrics': metrics,
            'model_architecture': {
                'input_dim': 8,
                'hidden_dims': config['hidden_dims'],
                'dropout': config['dropout']
            },
            'feature_names': [
                'content_score', 'cf_score', 'distance_km', 'price_score',
                'recency_score', 'popularity_score', 'diversity_score', 'time_match_score'
            ]
        }
        
        metadata_path = self.model_dir / "hybrid_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training history
        history_path = self.model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained model."""
        
        if model_path is None:
            model_path = self.model_dir / "hybrid_ranker.pth"
        
        metadata_path = self.model_dir / "hybrid_metadata.json"
        
        if not Path(model_path).exists() or not metadata_path.exists():
            logger.warning("Model files not found")
            return False
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        config = metadata['config']
        
        # Initialize model
        self.model = HybridRankerMLP(
            input_dim=8,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Load state
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return True
    
    def predict(self, features: List[List[float]]) -> np.ndarray:
        """Make predictions on new features."""
        
        if self.model is None:
            raise ValueError("No model loaded")
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        return predictions.cpu().numpy().flatten()

def main():
    """Main training script with hyperparameter tuning."""
    
    # Load training data
    data_dir = Path("ml/training/data")
    train_path = data_dir / "hybrid_ranker_train.json"
    val_path = data_dir / "hybrid_ranker_val.json"
    
    if not train_path.exists() or not val_path.exists():
        logger.error("Training data not found. Run data_generator.py first.")
        return
    
    # Load training examples
    def load_examples(file_path: Path) -> List[TrainingExample]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            example = TrainingExample(
                user_id=item['user_id'],
                event_id=item['event_id'],
                content_score=item['content_score'],
                cf_score=item['cf_score'],
                distance_km=item['distance_km'],
                price_score=item['price_score'],
                recency_score=item['recency_score'],
                popularity_score=item['popularity_score'],
                diversity_score=item['diversity_score'],
                time_match_score=item['time_match_score'],
                label=item['label'],
                interaction_type=item.get('interaction_type'),
                rating=item.get('rating')
            )
            examples.append(example)
        
        return examples
    
    train_examples = load_examples(train_path)
    val_examples = load_examples(val_path)
    
    logger.info(f"Loaded {len(train_examples)} train and {len(val_examples)} validation examples")
    
    # Initialize trainer
    trainer = HybridRankerTrainer()
    
    # Option 1: Train with default parameters
    logger.info("Training with default parameters...")
    results = trainer.train(train_examples, val_examples)
    
    # Option 2: Hyperparameter tuning (comment out if you want to skip)
    # logger.info("Starting hyperparameter tuning...")
    # tuning_results = trainer.hyperparameter_tuning(
    #     train_examples, val_examples, n_trials=20, timeout=1800
    # )
    
    print("\nTraining completed!")
    print(f"Best validation AUC: {results['best_val_auc']:.4f}")
    print(f"Final metrics: {results['final_metrics']}")

if __name__ == "__main__":
    main()