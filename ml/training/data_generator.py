#!/usr/bin/env python3
"""
Training data generation pipeline for hybrid recommender.
Creates positive/negative samples and feature extraction for neural ranker training.
"""

import duckdb
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import random
from collections import defaultdict
from dataclasses import dataclass

from ml.models.content_based import ContentBasedRecommender, UserPreferences
from ml.models.collaborative_filtering import CollaborativeFilteringRecommender
from ml.models.hybrid_ranker import EventCandidate
from geopy.distance import geodesic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example for the hybrid ranker."""

    user_id: str
    event_id: str
    content_score: float
    cf_score: float
    distance_km: float
    price_score: float
    recency_score: float
    popularity_score: float
    diversity_score: float
    time_match_score: float
    label: int  # 1 for positive, 0 for negative
    interaction_type: Optional[str] = None
    rating: Optional[float] = None


class TrainingDataGenerator:
    """Generates training data for hybrid recommendation models."""

    def __init__(
        self,
        db_path: str = "data/events.duckdb",
        content_model: Optional[ContentBasedRecommender] = None,
        cf_model: Optional[CollaborativeFilteringRecommender] = None,
        output_dir: str = "ml/training/data",
    ):
        """
        Initialize training data generator.

        Args:
            db_path: Path to DuckDB database
            content_model: Trained content-based model (optional)
            cf_model: Trained collaborative filtering model (optional)
            output_dir: Directory to save training data
        """
        self.db_path = db_path
        self.content_model = content_model
        self.cf_model = cf_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.positive_interaction_types = {"like", "going", "went", "save"}
        self.negative_sampling_ratio = 5  # 5 negatives per positive
        self.min_interactions_per_user = 3
        self.positive_rating_threshold = 4.0

        # Cache
        self._events_cache = None
        self._users_cache = None
        self._interactions_cache = None

    def generate_training_data(
        self,
        split_ratio: float = 0.8,
        max_examples: int = 100000,
        balance_users: bool = True,
    ) -> Tuple[List[TrainingExample], List[TrainingExample]]:
        """
        Generate training and validation data for hybrid ranker.

        Args:
            split_ratio: Train/validation split ratio
            max_examples: Maximum number of training examples
            balance_users: Whether to balance examples across users

        Returns:
            Tuple of (train_examples, val_examples)
        """
        logger.info("Generating training data for hybrid ranker...")

        # Load data
        events, users, interactions = self._load_data()

        if len(interactions) < 100:
            logger.warning(
                f"Only {len(interactions)} interactions available. Need more data."
            )
            return [], []

        # Filter users with sufficient interactions
        user_interaction_counts = defaultdict(int)
        for interaction in interactions:
            user_interaction_counts[interaction["user_id"]] += 1

        valid_users = [
            user_id
            for user_id, count in user_interaction_counts.items()
            if count >= self.min_interactions_per_user
        ]

        logger.info(
            f"Found {len(valid_users)} users with >= {self.min_interactions_per_user} interactions"
        )

        if not valid_users:
            logger.error("No users with sufficient interactions")
            return [], []

        # Generate positive examples
        positive_examples = self._generate_positive_examples(
            interactions, events, users, valid_users
        )

        # Generate negative examples
        negative_examples = self._generate_negative_examples(
            positive_examples, events, users, valid_users
        )

        # Combine and shuffle
        all_examples = positive_examples + negative_examples
        random.shuffle(all_examples)

        # Limit total examples
        if len(all_examples) > max_examples:
            all_examples = all_examples[:max_examples]
            logger.info(f"Limited to {max_examples} examples")

        # Balance across users if requested
        if balance_users:
            all_examples = self._balance_examples_by_user(all_examples, valid_users)

        # Split train/validation
        split_idx = int(len(all_examples) * split_ratio)
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]

        logger.info(
            f"Generated {len(train_examples)} training and {len(val_examples)} validation examples"
        )
        logger.info(
            f"Positive examples: {sum(1 for ex in all_examples if ex.label == 1)}"
        )
        logger.info(
            f"Negative examples: {sum(1 for ex in all_examples if ex.label == 0)}"
        )

        return train_examples, val_examples

    def _load_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load events, users, and interactions from database."""

        if (
            self._events_cache is not None
            and self._users_cache is not None
            and self._interactions_cache is not None
        ):
            return self._events_cache, self._users_cache, self._interactions_cache

        logger.info("Loading data from database...")

        if not Path(self.db_path).exists():
            logger.error(f"Database not found: {self.db_path}")
            return [], [], []

        conn = duckdb.connect(self.db_path)

        # Load events with venue information
        events_query = """
        SELECT 
            e.id, e.title, e.description, e.date_time, e.end_date_time,
            e.price_min, e.price_max, e.artist_ids, e.popularity_score, e.status,
            v.name as venue_name, v.lat as venue_lat, v.lon as venue_lon, v.neighborhood
        FROM events e
        JOIN venues v ON e.venue_id = v.id
        WHERE e.status = 'active'
        """

        events_result = conn.execute(events_query).fetchall()
        events = []
        for row in events_result:
            artist_ids = json.loads(row[7]) if row[7] else []
            events.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2] or "",
                    "date_time": row[3],
                    "end_date_time": row[4],
                    "price_min": row[5],
                    "price_max": row[6],
                    "artist_ids": artist_ids,
                    "popularity_score": row[8] or 0.0,
                    "venue_name": row[10],
                    "venue_lat": row[11],
                    "venue_lon": row[12],
                    "neighborhood": row[13],
                }
            )

        # Load users with preferences
        users_query = """
        SELECT id, name, preferences, location_lat, location_lon, created_at
        FROM users
        """

        users_result = conn.execute(users_query).fetchall()
        users = []
        for row in users_result:
            preferences = json.loads(row[2]) if row[2] else {}
            users.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "preferences": preferences,
                    "location_lat": row[3],
                    "location_lon": row[4],
                    "created_at": row[5],
                }
            )

        # Load interactions
        interactions_query = """
        SELECT user_id, event_id, interaction_type, rating, timestamp
        FROM interactions
        WHERE timestamp >= ?
        ORDER BY timestamp
        """

        # Only use recent interactions (last 6 months) for training
        cutoff_date = datetime.now() - timedelta(days=180)
        interactions_result = conn.execute(interactions_query, [cutoff_date]).fetchall()

        interactions = []
        for row in interactions_result:
            interactions.append(
                {
                    "user_id": row[0],
                    "event_id": row[1],
                    "interaction_type": row[2],
                    "rating": row[3],
                    "timestamp": row[4],
                }
            )

        conn.close()

        # Cache results
        self._events_cache = events
        self._users_cache = users
        self._interactions_cache = interactions

        logger.info(
            f"Loaded {len(events)} events, {len(users)} users, {len(interactions)} interactions"
        )
        return events, users, interactions

    def _generate_positive_examples(
        self,
        interactions: List[Dict],
        events: List[Dict],
        users: List[Dict],
        valid_users: List[str],
    ) -> List[TrainingExample]:
        """Generate positive training examples from user interactions."""

        logger.info("Generating positive examples...")

        # Create lookup dictionaries
        event_lookup = {event["id"]: event for event in events}
        user_lookup = {user["id"]: user for user in users}

        positive_examples = []

        for interaction in interactions:
            user_id = interaction["user_id"]
            event_id = interaction["event_id"]

            # Skip if user not in valid users or event not found
            if user_id not in valid_users or event_id not in event_lookup:
                continue

            # Determine if this is a positive interaction
            is_positive = False

            # Check interaction type
            if interaction["interaction_type"] in self.positive_interaction_types:
                is_positive = True

            # Check rating (if available)
            if (
                interaction.get("rating")
                and interaction["rating"] >= self.positive_rating_threshold
            ):
                is_positive = True

            if not is_positive:
                continue

            # Extract features for this user-event pair
            try:
                example = self._create_training_example(
                    user_id=user_id,
                    event_id=event_id,
                    event_data=event_lookup[event_id],
                    user_data=user_lookup.get(user_id, {}),
                    label=1,
                    interaction=interaction,
                )
                positive_examples.append(example)

            except Exception as e:
                logger.warning(
                    f"Failed to create positive example for {user_id}, {event_id}: {e}"
                )
                continue

        logger.info(f"Generated {len(positive_examples)} positive examples")
        return positive_examples

    def _generate_negative_examples(
        self,
        positive_examples: List[TrainingExample],
        events: List[Dict],
        users: List[Dict],
        valid_users: List[str],
    ) -> List[TrainingExample]:
        """Generate negative training examples through negative sampling."""

        logger.info("Generating negative examples...")

        # Create lookup dictionaries
        event_lookup = {event["id"]: event for event in events}
        user_lookup = {user["id"]: user for user in users}

        # Track positive user-event pairs
        positive_pairs = {(ex.user_id, ex.event_id) for ex in positive_examples}

        # Group positive examples by user
        user_positive_events = defaultdict(set)
        for example in positive_examples:
            user_positive_events[example.user_id].add(example.event_id)

        negative_examples = []
        event_ids = list(event_lookup.keys())

        for user_id in valid_users:
            if user_id not in user_positive_events:
                continue

            positive_events = user_positive_events[user_id]
            num_negatives_needed = len(positive_events) * self.negative_sampling_ratio

            # Sample negative events for this user
            candidate_events = [eid for eid in event_ids if eid not in positive_events]

            if len(candidate_events) < num_negatives_needed:
                sampled_events = candidate_events
            else:
                sampled_events = random.sample(candidate_events, num_negatives_needed)

            for event_id in sampled_events:
                # Skip if this is actually a positive pair
                if (user_id, event_id) in positive_pairs:
                    continue

                try:
                    example = self._create_training_example(
                        user_id=user_id,
                        event_id=event_id,
                        event_data=event_lookup[event_id],
                        user_data=user_lookup.get(user_id, {}),
                        label=0,  # Negative label
                        interaction=None,
                    )
                    negative_examples.append(example)

                except Exception as e:
                    logger.warning(
                        f"Failed to create negative example for {user_id}, {event_id}: {e}"
                    )
                    continue

        logger.info(f"Generated {len(negative_examples)} negative examples")
        return negative_examples

    def _create_training_example(
        self,
        user_id: str,
        event_id: str,
        event_data: Dict,
        user_data: Dict,
        label: int,
        interaction: Optional[Dict] = None,
    ) -> TrainingExample:
        """Create a single training example with all features."""

        # Create user preferences from user data
        user_prefs = self._create_user_preferences(user_data)

        # Get content-based score
        content_score = 0.0
        if self.content_model:
            try:
                content_recs = self.content_model.recommend(
                    user_prefs, candidate_event_ids=[event_id], num_recommendations=1
                )
                if content_recs:
                    content_score = content_recs[0][1]
            except Exception as e:
                logger.warning(f"Content model failed for {user_id}, {event_id}: {e}")

        # Get collaborative filtering score
        cf_score = 0.0
        if self.cf_model:
            try:
                cf_recs = self.cf_model.recommend(
                    user_id, candidate_event_ids=[event_id], num_recommendations=1
                )
                if cf_recs:
                    cf_score = cf_recs[0][1]
            except Exception as e:
                logger.warning(f"CF model failed for {user_id}, {event_id}: {e}")

        # Calculate distance
        distance_km = 0.0
        if (
            user_prefs.location_lat
            and user_prefs.location_lon
            and event_data.get("venue_lat")
            and event_data.get("venue_lon")
        ):
            try:
                distance_km = geodesic(
                    (user_prefs.location_lat, user_prefs.location_lon),
                    (event_data["venue_lat"], event_data["venue_lon"]),
                ).kilometers
            except Exception:
                distance_km = 10.0  # Default

        # Calculate price score
        price_score = self._calculate_price_score(event_data, user_prefs)

        # Calculate recency score
        recency_score = self._calculate_recency_score(event_data)

        # Get popularity score
        popularity_score = event_data.get("popularity_score", 0.0)

        # Calculate time match score
        time_match_score = self._calculate_time_match_score(event_data, user_prefs)

        # Diversity score (simplified for training data generation)
        diversity_score = random.uniform(0.3, 0.8)  # Placeholder

        return TrainingExample(
            user_id=user_id,
            event_id=event_id,
            content_score=content_score,
            cf_score=cf_score,
            distance_km=distance_km,
            price_score=price_score,
            recency_score=recency_score,
            popularity_score=popularity_score,
            diversity_score=diversity_score,
            time_match_score=time_match_score,
            label=label,
            interaction_type=(
                interaction.get("interaction_type") if interaction else None
            ),
            rating=interaction.get("rating") if interaction else None,
        )

    def _create_user_preferences(self, user_data: Dict) -> UserPreferences:
        """Create UserPreferences from user data."""

        preferences = user_data.get("preferences", {})

        return UserPreferences(
            preferred_genres=preferences.get("preferred_genres", []),
            preferred_artists=preferences.get("preferred_artists", []),
            preferred_venues=preferences.get("preferred_venues", []),
            price_range=tuple(preferences.get("price_range", [0, 1000])),
            location_lat=user_data.get("location_lat"),
            location_lon=user_data.get("location_lon"),
            preferred_times=preferences.get("preferred_times", [20, 21, 22]),
            preferred_days=preferences.get("preferred_days", [4, 5, 6]),
        )

    def _calculate_price_score(
        self, event_data: Dict, user_prefs: UserPreferences
    ) -> float:
        """Calculate price compatibility score."""

        event_min = event_data.get("price_min", 0) or 0
        event_max = event_data.get("price_max", event_min) or event_min
        user_min, user_max = user_prefs.price_range

        if event_max == 0:  # Free event
            return 1.0 if user_max >= 0 else 0.5

        # Check if price ranges overlap
        if event_min > user_max or event_max < user_min:
            return 0.0

        # Calculate overlap ratio
        overlap_start = max(event_min, user_min)
        overlap_end = min(event_max, user_max)
        overlap_size = overlap_end - overlap_start

        event_range = event_max - event_min or 1
        user_range = user_max - user_min or 1

        score = overlap_size / min(event_range, user_range)
        return min(1.0, score)

    def _calculate_recency_score(self, event_data: Dict) -> float:
        """Calculate recency score."""

        event_date = event_data.get("date_time")
        if not event_date:
            return 0.5

        if isinstance(event_date, str):
            try:
                event_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
            except:
                return 0.5

        now = datetime.now()
        days_until = (event_date - now).days

        if days_until < 0:
            return 0.0
        elif days_until <= 7:
            return 1.0
        elif days_until <= 30:
            return 0.8
        elif days_until <= 90:
            return 0.6
        else:
            return 0.3

    def _calculate_time_match_score(
        self, event_data: Dict, user_prefs: UserPreferences
    ) -> float:
        """Calculate time match score."""

        event_date = event_data.get("date_time")
        if not event_date:
            return 0.5

        if isinstance(event_date, str):
            try:
                event_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
            except:
                return 0.5

        score = 0.0

        # Hour preference
        if user_prefs.preferred_times:
            event_hour = event_date.hour
            hour_scores = [
                abs(event_hour - pref_hour) for pref_hour in user_prefs.preferred_times
            ]
            best_hour_match = min(hour_scores)
            hour_score = max(0, 1.0 - best_hour_match / 12.0)
            score += hour_score * 0.5

        # Day preference
        if user_prefs.preferred_days:
            event_day = event_date.weekday()
            if event_day in user_prefs.preferred_days:
                score += 0.5

        return min(1.0, score)

    def _balance_examples_by_user(
        self, examples: List[TrainingExample], valid_users: List[str]
    ) -> List[TrainingExample]:
        """Balance examples across users to prevent overfitting to heavy users."""

        # Group examples by user
        user_examples = defaultdict(list)
        for example in examples:
            user_examples[example.user_id].append(example)

        # Calculate target examples per user
        total_examples = len(examples)
        target_per_user = max(
            10, total_examples // (len(valid_users) * 2)
        )  # At least 10, reasonable cap

        balanced_examples = []
        for user_id in valid_users:
            user_exs = user_examples[user_id]
            if len(user_exs) <= target_per_user:
                balanced_examples.extend(user_exs)
            else:
                # Sample maintaining positive/negative ratio
                positive_exs = [ex for ex in user_exs if ex.label == 1]
                negative_exs = [ex for ex in user_exs if ex.label == 0]

                pos_target = min(len(positive_exs), target_per_user // 2)
                neg_target = target_per_user - pos_target

                sampled_pos = (
                    random.sample(positive_exs, pos_target) if positive_exs else []
                )
                sampled_neg = (
                    random.sample(negative_exs, min(len(negative_exs), neg_target))
                    if negative_exs
                    else []
                )

                balanced_examples.extend(sampled_pos + sampled_neg)

        logger.info(
            f"Balanced from {len(examples)} to {len(balanced_examples)} examples"
        )
        return balanced_examples

    def save_training_data(
        self,
        train_examples: List[TrainingExample],
        val_examples: List[TrainingExample],
        prefix: str = "hybrid_ranker",
    ):
        """Save training data to files."""

        # Convert to dictionaries
        def example_to_dict(example: TrainingExample) -> Dict:
            return {
                "user_id": example.user_id,
                "event_id": example.event_id,
                "content_score": example.content_score,
                "cf_score": example.cf_score,
                "distance_km": example.distance_km,
                "price_score": example.price_score,
                "recency_score": example.recency_score,
                "popularity_score": example.popularity_score,
                "diversity_score": example.diversity_score,
                "time_match_score": example.time_match_score,
                "label": example.label,
                "interaction_type": example.interaction_type,
                "rating": example.rating,
            }

        train_data = [example_to_dict(ex) for ex in train_examples]
        val_data = [example_to_dict(ex) for ex in val_examples]

        # Save files
        train_path = self.output_dir / f"{prefix}_train.json"
        val_path = self.output_dir / f"{prefix}_val.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2)

        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2)

        logger.info(f"Saved training data to {train_path} and {val_path}")

        # Save metadata
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "positive_train": sum(1 for ex in train_examples if ex.label == 1),
            "negative_train": sum(1 for ex in train_examples if ex.label == 0),
            "positive_val": sum(1 for ex in val_examples if ex.label == 1),
            "negative_val": sum(1 for ex in val_examples if ex.label == 0),
            "feature_names": [
                "content_score",
                "cf_score",
                "distance_km",
                "price_score",
                "recency_score",
                "popularity_score",
                "diversity_score",
                "time_match_score",
            ],
            "parameters": {
                "negative_sampling_ratio": self.negative_sampling_ratio,
                "min_interactions_per_user": self.min_interactions_per_user,
                "positive_rating_threshold": self.positive_rating_threshold,
            },
        }

        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main function to generate training data."""

    generator = TrainingDataGenerator()

    # Generate training data
    train_examples, val_examples = generator.generate_training_data(
        split_ratio=0.8, max_examples=50000, balance_users=True
    )

    if not train_examples:
        logger.error("No training examples generated")
        return

    # Save training data
    generator.save_training_data(train_examples, val_examples)

    print(f"\nTraining data generation completed!")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(
        f"Positive ratio: {sum(ex.label for ex in train_examples) / len(train_examples):.2f}"
    )


if __name__ == "__main__":
    main()
