#!/usr/bin/env python3
"""
Custom Event-Event Collaborative Filter for single-user systems.
Solves the collaborative filtering paradox by learning event-event relationships
instead of user-user relationships.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
import json

# Import with absolute paths
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.models.responses import EventResponse

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic similarity")

logger = logging.getLogger(__name__)


@dataclass
class EventSimilarity:
    """Similarity between two events."""

    event_id_1: str
    event_id_2: str
    similarity_score: float
    similarity_reasons: List[str]
    similarity_type: str  # 'content', 'temporal', 'behavioral', 'venue'


@dataclass
class CollaborativeSignal:
    """A collaborative signal for an event pair."""

    signal_type: str
    strength: float
    explanation: str


class SingleUserCollaborativeFilter:
    """
    Event-Event Collaborative Filter designed for single-user systems.

    Core Concept: Instead of "users similar to you liked X",
    learn "when you like events with properties A, you also tend to like events with properties B"

    Strategies:
    1. Event Co-occurrence: Events you interact with in same time periods
    2. Feature Pattern Learning: Genre/venue/time patterns in your preferences
    3. Sequential Preference Learning: How your taste evolves
    4. Context-based Segmentation: Weekend you vs weekday you vs exploring you
    """

    def __init__(self):
        # Event similarity matrices
        self.event_content_similarity = {}
        self.event_temporal_patterns = {}
        self.event_behavioral_patterns = {}

        # User context patterns
        self.user_context_segments = {
            "weekend_explorer": [],
            "weekday_regular": [],
            "late_night_party": [],
            "early_evening_culture": [],
            "solo_discovery": [],
            "price_conscious": [],
            "premium_experience": [],
        }

        # Preference evolution tracking
        self.preference_evolution = {
            "genre_transitions": defaultdict(float),
            "venue_exploration_pattern": [],
            "price_tolerance_changes": [],
            "temporal_preference_shifts": [],
        }

        # Event-event collaborative signals
        self.event_cooccurrence_matrix = defaultdict(lambda: defaultdict(float))
        self.event_feature_similarity = {}

        # Initialize text vectorizer if sklearn available
        self.text_vectorizer = None
        if SKLEARN_AVAILABLE:
            self.text_vectorizer = TfidfVectorizer(
                max_features=100, stop_words="english", ngram_range=(1, 2)
            )

    def learn_from_interactions(
        self, user_interactions: List[Dict], events: List[EventResponse]
    ) -> None:
        """
        Learn collaborative patterns from user interaction history.

        Args:
            user_interactions: List of interaction dicts with event_id, type, rating, timestamp
            events: List of EventResponse objects
        """
        logger.info(
            f"Learning collaborative patterns from {len(user_interactions)} interactions"
        )

        # Build event lookup
        event_lookup = {event.event_id: event for event in events}

        # Learn different types of collaborative patterns
        self._learn_event_cooccurrence(user_interactions, event_lookup)
        self._learn_context_patterns(user_interactions, event_lookup)
        self._learn_preference_evolution(user_interactions, event_lookup)
        self._learn_feature_similarity(events)

        logger.info("Collaborative pattern learning completed")

    def _learn_event_cooccurrence(
        self, interactions: List[Dict], event_lookup: Dict[str, EventResponse]
    ) -> None:
        """Learn which events tend to be liked together."""

        # Group interactions by time windows
        time_windows = self._group_interactions_by_time_window(interactions)

        for window_interactions in time_windows:
            liked_events = [
                interaction["event_id"]
                for interaction in window_interactions
                if interaction.get("interaction_type") in ["like", "going", "went"]
                or (interaction.get("rating", 0) >= 4)
            ]

            # Create co-occurrence signals
            for event_a in liked_events:
                for event_b in liked_events:
                    if event_a != event_b:
                        # Increase co-occurrence strength
                        self.event_cooccurrence_matrix[event_a][event_b] += 1.0

        # Normalize co-occurrence matrix
        self._normalize_cooccurrence_matrix()

    def _learn_context_patterns(
        self, interactions: List[Dict], event_lookup: Dict[str, EventResponse]
    ) -> None:
        """Learn context-based preference patterns."""

        for interaction in interactions:
            event_id = interaction["event_id"]
            if event_id not in event_lookup:
                continue

            event = event_lookup[event_id]
            timestamp = interaction.get("timestamp", datetime.now())

            # Parse timestamp if it's a string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except:
                    timestamp = datetime.now()

            # Categorize interaction into context segments
            contexts = self._identify_interaction_contexts(
                event, timestamp, interaction
            )

            for context in contexts:
                if context in self.user_context_segments:
                    self.user_context_segments[context].append(
                        {
                            "event_id": event_id,
                            "interaction_type": interaction.get("interaction_type"),
                            "rating": interaction.get("rating"),
                            "timestamp": timestamp,
                        }
                    )

    def _identify_interaction_contexts(
        self, event: EventResponse, timestamp: datetime, interaction: Dict
    ) -> List[str]:
        """Identify which contexts this interaction belongs to."""

        contexts = []

        # Temporal contexts
        if timestamp.weekday() >= 5:  # Friday = 4, Saturday = 5, Sunday = 6
            contexts.append("weekend_explorer")
        else:
            contexts.append("weekday_regular")

        if timestamp.hour >= 22:
            contexts.append("late_night_party")
        elif timestamp.hour <= 20:
            contexts.append("early_evening_culture")

        # Price contexts
        avg_price = (event.price_min or 0 + event.price_max or 0) / 2
        if avg_price > 400:  # DKK
            contexts.append("premium_experience")
        elif avg_price < 200:
            contexts.append("price_conscious")

        return contexts

    def _learn_preference_evolution(
        self, interactions: List[Dict], event_lookup: Dict[str, EventResponse]
    ) -> None:
        """Learn how preferences evolve over time."""

        # Sort interactions by timestamp
        sorted_interactions = sorted(
            interactions, key=lambda x: x.get("timestamp", datetime.now())
        )

        # Track genre progression
        genre_sequence = []
        venue_sequence = []

        for interaction in sorted_interactions:
            event_id = interaction["event_id"]
            if event_id not in event_lookup:
                continue

            event = event_lookup[event_id]

            # Track genre evolution
            for genre in event.genres:
                genre_sequence.append(genre)

            # Track venue exploration
            venue_sequence.append(event.venue_name)

        # Learn transition patterns
        self._learn_genre_transitions(genre_sequence)
        self._learn_venue_exploration_pattern(venue_sequence)

    def _learn_genre_transitions(self, genre_sequence: List[str]) -> None:
        """Learn genre transition patterns."""

        for i in range(len(genre_sequence) - 1):
            current_genre = genre_sequence[i]
            next_genre = genre_sequence[i + 1]

            # Increase transition strength
            transition_key = f"{current_genre} -> {next_genre}"
            self.preference_evolution["genre_transitions"][transition_key] += 1.0

    def _learn_venue_exploration_pattern(self, venue_sequence: List[str]) -> None:
        """Learn venue exploration patterns."""

        # Detect exploration vs exploitation patterns
        venue_counts = Counter(venue_sequence)
        exploration_events = []

        for i, venue in enumerate(venue_sequence):
            if venue_counts[venue] == 1:  # First time at this venue
                exploration_events.append(i)

        self.preference_evolution["venue_exploration_pattern"] = exploration_events

    def _learn_feature_similarity(self, events: List[EventResponse]) -> None:
        """Learn content-based feature similarity between events."""

        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, skipping advanced similarity")
            return

        try:
            # Create text representations of events
            event_texts = []
            event_ids = []

            for event in events:
                # Combine title, description, genres, artists for text analysis
                text_parts = [event.title]

                if event.description:
                    text_parts.append(event.description)

                text_parts.extend(event.genres)
                text_parts.extend(event.artists)
                text_parts.append(event.venue_name)

                event_text = " ".join(text_parts)
                event_texts.append(event_text)
                event_ids.append(event.event_id)

            # Create TF-IDF vectors
            if len(event_texts) > 1:
                tfidf_matrix = self.text_vectorizer.fit_transform(event_texts)

                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(tfidf_matrix)

                # Store similarities
                for i, event_id_1 in enumerate(event_ids):
                    for j, event_id_2 in enumerate(event_ids):
                        if i != j:
                            similarity_score = similarity_matrix[i][j]
                            if (
                                similarity_score > 0.1
                            ):  # Only store significant similarities
                                self.event_feature_similarity[
                                    (event_id_1, event_id_2)
                                ] = similarity_score

                logger.info(f"Computed feature similarity for {len(event_ids)} events")

        except Exception as e:
            logger.warning(f"Failed to compute feature similarity: {e}")

    def get_collaborative_recommendations(
        self,
        seed_events: List[str],
        candidate_events: List[EventResponse],
        user_context: Optional[Dict] = None,
        num_recommendations: int = 10,
    ) -> List[Tuple[EventResponse, float, List[str]]]:
        """
        Generate collaborative recommendations based on seed events.

        Args:
            seed_events: Event IDs that the user has liked/interacted with
            candidate_events: Events to rank
            user_context: Current context (time, situation, etc.)
            num_recommendations: Number of recommendations to return

        Returns:
            List of (event, score, reasons) tuples
        """

        recommendations = []

        for candidate in candidate_events:
            if candidate.event_id in seed_events:
                continue  # Don't recommend events user already interacted with

            # Calculate collaborative score
            collab_score = 0.0
            reasons = []

            # 1. Co-occurrence score
            cooccur_score = self._calculate_cooccurrence_score(
                candidate.event_id, seed_events
            )
            if cooccur_score > 0:
                collab_score += cooccur_score * 0.4
                reasons.append(
                    f"Often enjoyed with your liked events ({cooccur_score:.2f})"
                )

            # 2. Feature similarity score
            feature_score = self._calculate_feature_similarity_score(
                candidate.event_id, seed_events
            )
            if feature_score > 0:
                collab_score += feature_score * 0.3
                reasons.append(
                    f"Similar content to your preferences ({feature_score:.2f})"
                )

            # 3. Context pattern score
            context_score = self._calculate_context_pattern_score(
                candidate, user_context
            )
            if context_score > 0:
                collab_score += context_score * 0.2
                reasons.append(
                    f"Matches your behavioral patterns ({context_score:.2f})"
                )

            # 4. Preference evolution score
            evolution_score = self._calculate_evolution_score(candidate, seed_events)
            if evolution_score > 0:
                collab_score += evolution_score * 0.1
                reasons.append(
                    f"Aligns with your taste evolution ({evolution_score:.2f})"
                )

            if collab_score > 0:
                recommendations.append((candidate, collab_score, reasons))

        # Sort by collaborative score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Return top N
        return recommendations[:num_recommendations]

    def _calculate_cooccurrence_score(
        self, event_id: str, seed_events: List[str]
    ) -> float:
        """Calculate co-occurrence score with seed events."""

        total_score = 0.0

        for seed_event in seed_events:
            if event_id in self.event_cooccurrence_matrix.get(seed_event, {}):
                total_score += self.event_cooccurrence_matrix[seed_event][event_id]

        # Normalize by number of seed events
        return total_score / len(seed_events) if seed_events else 0.0

    def _calculate_feature_similarity_score(
        self, event_id: str, seed_events: List[str]
    ) -> float:
        """Calculate content feature similarity with seed events."""

        if not SKLEARN_AVAILABLE:
            return 0.0

        total_score = 0.0
        count = 0

        for seed_event in seed_events:
            # Check both directions
            key1 = (event_id, seed_event)
            key2 = (seed_event, event_id)

            if key1 in self.event_feature_similarity:
                total_score += self.event_feature_similarity[key1]
                count += 1
            elif key2 in self.event_feature_similarity:
                total_score += self.event_feature_similarity[key2]
                count += 1

        return total_score / count if count > 0 else 0.0

    def _calculate_context_pattern_score(
        self, candidate: EventResponse, user_context: Optional[Dict]
    ) -> float:
        """Calculate context pattern matching score."""

        if not user_context:
            return 0.0

        score = 0.0

        # Check if candidate matches user's context patterns
        current_time = datetime.now()
        contexts = self._identify_interaction_contexts(
            candidate, current_time, user_context
        )

        for context in contexts:
            if context in self.user_context_segments:
                context_events = self.user_context_segments[context]
                if context_events:
                    # Give higher score if this context has positive interactions
                    positive_interactions = sum(
                        1
                        for interaction in context_events
                        if interaction.get("interaction_type")
                        in ["like", "going", "went"]
                        or interaction.get("rating", 0) >= 4
                    )

                    if positive_interactions > 0:
                        score += positive_interactions / len(context_events)

        return score

    def _calculate_evolution_score(
        self, candidate: EventResponse, seed_events: List[str]
    ) -> float:
        """Calculate score based on preference evolution patterns."""

        score = 0.0

        # Check genre transition patterns
        for genre in candidate.genres:
            # Look for genres that user typically transitions to
            for transition, strength in self.preference_evolution[
                "genre_transitions"
            ].items():
                if transition.endswith(f" -> {genre}"):
                    score += strength * 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _group_interactions_by_time_window(
        self, interactions: List[Dict], window_hours: int = 24
    ) -> List[List[Dict]]:
        """Group interactions by time windows."""

        # Sort by timestamp
        sorted_interactions = sorted(
            interactions, key=lambda x: x.get("timestamp", datetime.now())
        )

        windows = []
        current_window = []
        window_start = None

        for interaction in sorted_interactions:
            timestamp = interaction.get("timestamp", datetime.now())

            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except:
                    timestamp = datetime.now()

            if window_start is None:
                window_start = timestamp
                current_window = [interaction]
            elif (timestamp - window_start).total_seconds() <= window_hours * 3600:
                current_window.append(interaction)
            else:
                if current_window:
                    windows.append(current_window)
                window_start = timestamp
                current_window = [interaction]

        if current_window:
            windows.append(current_window)

        return windows

    def _normalize_cooccurrence_matrix(self) -> None:
        """Normalize co-occurrence matrix."""

        # Find max value for normalization
        max_value = 0.0
        for event_dict in self.event_cooccurrence_matrix.values():
            for score in event_dict.values():
                max_value = max(max_value, score)

        if max_value > 0:
            # Normalize all values to [0, 1]
            for event_a in self.event_cooccurrence_matrix:
                for event_b in self.event_cooccurrence_matrix[event_a]:
                    self.event_cooccurrence_matrix[event_a][event_b] /= max_value

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""

        stats = {
            "cooccurrence_pairs": sum(
                len(event_dict)
                for event_dict in self.event_cooccurrence_matrix.values()
            ),
            "feature_similarity_pairs": len(self.event_feature_similarity),
            "context_segments": {
                context: len(events)
                for context, events in self.user_context_segments.items()
            },
            "genre_transitions": len(self.preference_evolution["genre_transitions"]),
            "venue_exploration_events": len(
                self.preference_evolution["venue_exploration_pattern"]
            ),
        }

        return stats
