#!/usr/bin/env python3
"""
Production ML training pipeline that integrates all components:
data generation, model training, evaluation, and deployment.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

from .data_generator import TrainingDataGenerator
from .hybrid_trainer import HybridRankerTrainer
from .train_models import ModelTrainer
from ..models.content_based import ContentBasedRecommender
from ..models.collaborative_filtering import CollaborativeFilteringRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionMLPipeline:
    """Production ML pipeline for event recommender models."""
    
    def __init__(
        self,
        db_path: str = "data/events.duckdb",
        models_dir: str = "ml/models",
        data_dir: str = "ml/training/data"
    ):
        """
        Initialize production ML pipeline.
        
        Args:
            db_path: Path to DuckDB database
            models_dir: Directory for model storage
            data_dir: Directory for training data
        """
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_trainer = ModelTrainer(db_path=db_path, models_dir=str(models_dir))
        self.data_generator = TrainingDataGenerator(
            db_path=db_path, 
            output_dir=str(data_dir)
        )
        self.hybrid_trainer = HybridRankerTrainer(model_dir=str(models_dir / "hybrid"))
        
        # Track pipeline state
        self.pipeline_state = {
            'content_model_trained': False,
            'cf_model_trained': False,
            'training_data_generated': False,
            'hybrid_model_trained': False,
            'last_run_timestamp': None
        }
    
    def run_full_pipeline(
        self, 
        retrain_all: bool = False,
        tune_hyperparameters: bool = False,
        max_training_examples: int = 50000
    ) -> Dict[str, Any]:
        """
        Run the complete ML training pipeline.
        
        Args:
            retrain_all: Whether to retrain all models from scratch
            tune_hyperparameters: Whether to perform hyperparameter tuning
            max_training_examples: Maximum number of training examples
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("Starting production ML pipeline...")
        
        pipeline_start = datetime.now()
        results = {
            'pipeline_start': pipeline_start.isoformat(),
            'steps_completed': [],
            'errors': [],
            'models_trained': [],
            'final_metrics': {}
        }
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data...")
            events, interactions = self.model_trainer.load_data()
            
            if not events:
                raise ValueError("No events found in database")
            
            if len(interactions) < 50:
                logger.warning(f"Only {len(interactions)} interactions found. May affect model quality.")
            
            results['data_stats'] = {
                'num_events': len(events),
                'num_interactions': len(interactions)
            }
            results['steps_completed'].append('data_loading')
            
            # Step 2: Train content-based model
            if retrain_all or not self._content_model_exists():
                logger.info("Step 2: Training content-based model...")
                content_model = self.model_trainer.train_content_model(events, interactions)
                
                if content_model:
                    results['models_trained'].append('content_based')
                    self.pipeline_state['content_model_trained'] = True
                    logger.info("Content-based model training completed")
                else:
                    raise ValueError("Content-based model training failed")
            else:
                logger.info("Step 2: Loading existing content-based model...")
                content_model = self._load_content_model()
            
            results['steps_completed'].append('content_model')
            
            # Step 3: Train collaborative filtering model
            cf_model = None
            if len(interactions) >= 100:  # Need sufficient interactions
                if retrain_all or not self._cf_model_exists():
                    logger.info("Step 3: Training collaborative filtering model...")
                    cf_model = self.model_trainer.train_cf_model(interactions)
                    
                    if cf_model:
                        results['models_trained'].append('collaborative')
                        self.pipeline_state['cf_model_trained'] = True
                        logger.info("Collaborative filtering model training completed")
                else:
                    logger.info("Step 3: Loading existing collaborative filtering model...")
                    cf_model = self._load_cf_model()
                
                results['steps_completed'].append('cf_model')
            else:
                logger.info("Step 3: Skipping collaborative filtering (insufficient interactions)")
            
            # Step 4: Generate training data for hybrid ranker
            if retrain_all or not self._training_data_exists():
                logger.info("Step 4: Generating training data for hybrid ranker...")
                
                # Set models in data generator
                self.data_generator.content_model = content_model
                self.data_generator.cf_model = cf_model
                
                train_examples, val_examples = self.data_generator.generate_training_data(
                    max_examples=max_training_examples,
                    balance_users=True
                )
                
                if not train_examples:
                    raise ValueError("No training examples generated for hybrid ranker")
                
                self.data_generator.save_training_data(train_examples, val_examples)
                self.pipeline_state['training_data_generated'] = True
                
                results['training_data_stats'] = {
                    'num_train_examples': len(train_examples),
                    'num_val_examples': len(val_examples),
                    'positive_ratio': sum(ex.label for ex in train_examples) / len(train_examples)
                }
                logger.info(f"Generated {len(train_examples)} training examples")
            else:
                logger.info("Step 4: Loading existing training data...")
                train_examples, val_examples = self._load_training_data()
            
            results['steps_completed'].append('training_data_generation')
            
            # Step 5: Train hybrid ranker
            if retrain_all or not self._hybrid_model_exists():
                logger.info("Step 5: Training hybrid ranker...")
                
                if tune_hyperparameters:
                    logger.info("Performing hyperparameter tuning...")
                    tuning_results = self.hybrid_trainer.hyperparameter_tuning(
                        train_examples, val_examples, n_trials=30, timeout=3600
                    )
                    hybrid_results = tuning_results['final_results']
                    results['hyperparameter_tuning'] = {
                        'best_params': tuning_results['best_params'],
                        'best_value': tuning_results['best_value']
                    }
                else:
                    hybrid_results = self.hybrid_trainer.train(train_examples, val_examples)
                
                results['models_trained'].append('hybrid')
                results['final_metrics']['hybrid'] = hybrid_results['final_metrics']
                self.pipeline_state['hybrid_model_trained'] = True
                logger.info("Hybrid ranker training completed")
            else:
                logger.info("Step 5: Loading existing hybrid model...")
                self.hybrid_trainer.load_model()
            
            results['steps_completed'].append('hybrid_model')
            
            # Step 6: Comprehensive evaluation
            logger.info("Step 6: Running comprehensive evaluation...")
            evaluation_results = self.model_trainer.evaluate_models(
                events, interactions, k=10
            )
            
            results['final_metrics'].update(evaluation_results)
            results['steps_completed'].append('evaluation')
            
            # Step 7: Save pipeline state and results
            self.pipeline_state['last_run_timestamp'] = pipeline_start.isoformat()
            self._save_pipeline_state()
            
            # Save comprehensive results
            pipeline_end = datetime.now()
            results['pipeline_end'] = pipeline_end.isoformat()
            results['total_duration_minutes'] = (pipeline_end - pipeline_start).total_seconds() / 60
            
            self._save_pipeline_results(results)
            
            logger.info(f"Production ML pipeline completed in {results['total_duration_minutes']:.1f} minutes")
            logger.info(f"Models trained: {', '.join(results['models_trained'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed at step {len(results['steps_completed']) + 1}: {e}")
            results['errors'].append(str(e))
            results['pipeline_end'] = datetime.now().isoformat()
            self._save_pipeline_results(results)
            raise
    
    def _content_model_exists(self) -> bool:
        """Check if content-based model exists."""
        model_path = self.models_dir / "content_based" / "model_data.json"
        return model_path.exists()
    
    def _cf_model_exists(self) -> bool:
        """Check if collaborative filtering model exists."""
        model_path = self.models_dir / "collaborative"
        return model_path.exists() and any(model_path.glob("*.pkl"))
    
    def _hybrid_model_exists(self) -> bool:
        """Check if hybrid model exists."""
        model_path = self.models_dir / "hybrid" / "hybrid_ranker.pth"
        return model_path.exists()
    
    def _training_data_exists(self) -> bool:
        """Check if training data exists."""
        train_path = self.data_dir / "hybrid_ranker_train.json"
        val_path = self.data_dir / "hybrid_ranker_val.json"
        return train_path.exists() and val_path.exists()
    
    def _load_content_model(self) -> Optional[ContentBasedRecommender]:
        """Load existing content-based model."""
        try:
            from ..embeddings.content_embedder import ContentEmbedder
            embedder = ContentEmbedder(cache_dir=str(self.models_dir / "embeddings"))
            
            model = ContentBasedRecommender(
                embedder=embedder,
                model_cache_dir=str(self.models_dir / "content_based")
            )
            
            if model.load_model():
                return model
            return None
        except Exception as e:
            logger.warning(f"Failed to load content model: {e}")
            return None
    
    def _load_cf_model(self) -> Optional[CollaborativeFilteringRecommender]:
        """Load existing collaborative filtering model."""
        try:
            model = CollaborativeFilteringRecommender(
                model_cache_dir=str(self.models_dir / "collaborative")
            )
            
            if model.load_model():
                return model
            return None
        except Exception as e:
            logger.warning(f"Failed to load CF model: {e}")
            return None
    
    def _load_training_data(self):
        """Load existing training data."""
        train_path = self.data_dir / "hybrid_ranker_train.json"
        val_path = self.data_dir / "hybrid_ranker_val.json"
        
        def load_examples(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            from .data_generator import TrainingExample
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
        
        return train_examples, val_examples
    
    def _save_pipeline_state(self):
        """Save pipeline state to disk."""
        state_path = self.models_dir / "pipeline_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to disk."""
        results_path = self.models_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        state_path = self.models_dir / "pipeline_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
        return self.pipeline_state

def main():
    """Main CLI interface for production pipeline."""
    
    parser = argparse.ArgumentParser(description='Production ML Pipeline for Event Recommender')
    parser.add_argument('--db-path', default='data/events.duckdb', 
                       help='Path to DuckDB database')
    parser.add_argument('--models-dir', default='ml/models',
                       help='Directory for model storage')
    parser.add_argument('--retrain-all', action='store_true',
                       help='Retrain all models from scratch')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--max-examples', type=int, default=50000,
                       help='Maximum training examples for hybrid ranker')
    parser.add_argument('--status', action='store_true',
                       help='Show pipeline status and exit')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionMLPipeline(
        db_path=args.db_path,
        models_dir=args.models_dir
    )
    
    # Show status if requested
    if args.status:
        status = pipeline.get_pipeline_status()
        print("Pipeline Status:")
        print(json.dumps(status, indent=2))
        return
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            retrain_all=args.retrain_all,
            tune_hyperparameters=args.tune_hyperparameters,
            max_training_examples=args.max_examples
        )
        
        print("\n" + "="*60)
        print("PRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Duration: {results['total_duration_minutes']:.1f} minutes")
        print(f"Models trained: {', '.join(results['models_trained'])}")
        print(f"Steps completed: {', '.join(results['steps_completed'])}")
        
        if 'final_metrics' in results:
            print("\nFinal Metrics:")
            for model_name, metrics in results['final_metrics'].items():
                print(f"  {model_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())