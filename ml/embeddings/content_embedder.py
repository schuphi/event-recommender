#!/usr/bin/env python3
"""
Content embedding system using sentence-transformers for event descriptions.
Creates semantic embeddings for content-based recommendations.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import json
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    import torch

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import h3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EventFeatures:
    """Structured features for an event."""

    text_embedding: np.ndarray
    title: str
    description: str
    genres: List[str]
    artists: List[str]
    venue_name: str
    venue_lat: float
    venue_lon: float
    h3_index: str
    price_min: Optional[float]
    price_max: Optional[float]
    popularity_score: float
    datetime_features: Dict[str, float]  # hour, day_of_week, etc.


class ContentEmbedder:
    """Content-based embedding system for events."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "ml/models/embeddings",
        embedding_dim: int = 384,
    ):
        """
        Initialize content embedder.

        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory to cache embeddings and models
            embedding_dim: Expected embedding dimension
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        # Initialize models
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.genre_vocab = set()
        self.artist_vocab = set()

        self._load_models()

    def _load_models(self):
        """Load or initialize sentence transformer and TF-IDF models."""

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_path = (
                    self.cache_dir
                    / f"sentence_model_{self.model_name.replace('/', '_')}"
                )

                if model_path.exists():
                    logger.info(f"Loading cached sentence model from {model_path}")
                    self.sentence_model = SentenceTransformer(str(model_path))
                else:
                    logger.info(f"Downloading sentence model: {self.model_name}")
                    self.sentence_model = SentenceTransformer(self.model_name)
                    self.sentence_model.save(str(model_path))

                # Verify embedding dimension
                test_embedding = self.sentence_model.encode(["test"])
                actual_dim = test_embedding.shape[1]
                if actual_dim != self.embedding_dim:
                    logger.warning(
                        f"Model embedding dim {actual_dim} != expected {self.embedding_dim}"
                    )
                    self.embedding_dim = actual_dim

            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None

        # Load TF-IDF vectorizer if available
        tfidf_path = self.cache_dir / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            try:
                with open(tfidf_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded cached TF-IDF vectorizer")
            except Exception as e:
                logger.warning(f"Failed to load TF-IDF vectorizer: {e}")

    def encode_events(self, events: List[Dict]) -> List[EventFeatures]:
        """
        Encode a list of events into feature representations.

        Args:
            events: List of event dictionaries with keys:
                - title, description, genres, artists, venue_name, etc.

        Returns:
            List of EventFeatures with embeddings and structured features
        """

        logger.info(f"Encoding {len(events)} events...")

        # Extract text for embedding
        texts = []
        for event in events:
            text = self._prepare_text_for_embedding(event)
            texts.append(text)

        # Generate embeddings
        if self.sentence_model:
            logger.info("Generating sentence transformer embeddings...")
            embeddings = self.sentence_model.encode(
                texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
            )
        else:
            logger.warning("Using TF-IDF fallback embeddings")
            embeddings = self._generate_tfidf_embeddings(texts)

        # Create EventFeatures objects
        event_features = []
        for i, event in enumerate(events):
            try:
                features = self._create_event_features(event, embeddings[i])
                event_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to create features for event {i}: {e}")
                continue

        logger.info(f"Successfully encoded {len(event_features)} events")
        return event_features

    def _prepare_text_for_embedding(self, event: Dict) -> str:
        """Prepare event text for embedding generation."""

        # Combine title, description, and metadata
        parts = []

        # Title and description
        title = event.get("title", "").strip()
        description = event.get("description", "").strip()

        if title:
            parts.append(title)
        if description:
            # Truncate very long descriptions
            desc_truncated = (
                description[:500] if len(description) > 500 else description
            )
            parts.append(desc_truncated)

        # Add venue context
        venue_name = event.get("venue_name", "").strip()
        if venue_name:
            parts.append(f"at {venue_name}")

        # Add genre information
        genres = event.get("genres", [])
        if genres:
            genre_text = " ".join(genres)
            parts.append(f"music: {genre_text}")

        # Add artist information
        artists = event.get("artists", [])
        if artists:
            artist_text = " ".join(artists[:3])  # Top 3 artists
            parts.append(f"featuring: {artist_text}")

        # Join all parts
        text = " . ".join(parts)

        # Clean up text
        text = self._clean_text(text)

        return text

    def _clean_text(self, text: str) -> str:
        """Clean text for better embedding quality."""

        import re

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove excessive punctuation
        text = re.sub(r"[!@#$%^&*()_+=\[\]{}|;:,.<>?/~`-]{3,}", " ", text)

        # Remove HTML tags if any
        text = re.sub(r"<[^>]+>", "", text)

        return text.strip()

    def _generate_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings as fallback."""

        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.embedding_dim,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )

            # Fit on texts
            self.tfidf_vectorizer.fit(texts)

            # Cache the vectorizer
            tfidf_path = self.cache_dir / "tfidf_vectorizer.pkl"
            with open(tfidf_path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

        # Transform texts
        embeddings = self.tfidf_vectorizer.transform(texts).toarray()

        # Pad or truncate to match embedding_dim
        if embeddings.shape[1] < self.embedding_dim:
            padding = np.zeros(
                (embeddings.shape[0], self.embedding_dim - embeddings.shape[1])
            )
            embeddings = np.concatenate([embeddings, padding], axis=1)
        elif embeddings.shape[1] > self.embedding_dim:
            embeddings = embeddings[:, : self.embedding_dim]

        return embeddings

    def _create_event_features(
        self, event: Dict, embedding: np.ndarray
    ) -> EventFeatures:
        """Create structured features for an event."""

        # Extract datetime features
        datetime_features = {}
        if "date_time" in event and event["date_time"]:
            dt = event["date_time"]
            if hasattr(dt, "hour"):  # datetime object
                datetime_features = {
                    "hour": dt.hour / 23.0,  # Normalize to 0-1
                    "day_of_week": dt.weekday() / 6.0,
                    "month": (dt.month - 1) / 11.0,
                    "is_weekend": float(dt.weekday() >= 5),
                }

        # Calculate H3 index if needed
        lat = event.get("venue_lat") or event.get("lat")
        lon = event.get("venue_lon") or event.get("lon")
        h3_index = event.get("h3_index", "")

        if not h3_index and lat is not None and lon is not None:
            h3_index = h3.geo_to_h3(float(lat), float(lon), 8)

        return EventFeatures(
            text_embedding=embedding,
            title=event.get("title", ""),
            description=event.get("description", ""),
            genres=event.get("genres", []),
            artists=event.get("artists", []),
            venue_name=event.get("venue_name", ""),
            venue_lat=float(lat) if lat is not None else 0.0,
            venue_lon=float(lon) if lon is not None else 0.0,
            h3_index=h3_index,
            price_min=event.get("price_min"),
            price_max=event.get("price_max"),
            popularity_score=event.get("popularity_score", 0.0),
            datetime_features=datetime_features,
        )

    def compute_similarity(
        self,
        query_features: EventFeatures,
        candidate_features: List[EventFeatures],
        weights: Dict[str, float] = None,
    ) -> np.ndarray:
        """
        Compute similarity scores between query and candidates.

        Args:
            query_features: Features for query event/user preferences
            candidate_features: List of candidate event features
            weights: Weights for different similarity components

        Returns:
            Array of similarity scores [0, 1]
        """

        if weights is None:
            weights = {
                "text": 0.6,
                "genre": 0.2,
                "venue": 0.1,
                "price": 0.05,
                "time": 0.05,
            }

        n_candidates = len(candidate_features)
        similarities = np.zeros(n_candidates)

        # Text similarity (main component)
        if weights["text"] > 0:
            query_emb = query_features.text_embedding.reshape(1, -1)
            candidate_embs = np.vstack([cf.text_embedding for cf in candidate_features])

            text_sims = cosine_similarity(query_emb, candidate_embs)[0]
            similarities += weights["text"] * text_sims

        # Genre similarity
        if weights["genre"] > 0:
            query_genres = set(query_features.genres)
            for i, cf in enumerate(candidate_features):
                candidate_genres = set(cf.genres)
                if query_genres and candidate_genres:
                    genre_sim = len(query_genres & candidate_genres) / len(
                        query_genres | candidate_genres
                    )
                    similarities[i] += weights["genre"] * genre_sim

        # Venue/location similarity (using H3)
        if weights["venue"] > 0 and query_features.h3_index:
            for i, cf in enumerate(candidate_features):
                if cf.h3_index:
                    # H3 neighbor similarity (same or adjacent cells get high score)
                    if query_features.h3_index == cf.h3_index:
                        venue_sim = 1.0
                    elif cf.h3_index in h3.k_ring(query_features.h3_index, 1):
                        venue_sim = 0.8
                    elif cf.h3_index in h3.k_ring(query_features.h3_index, 2):
                        venue_sim = 0.5
                    else:
                        venue_sim = 0.0

                    similarities[i] += weights["venue"] * venue_sim

        # Price similarity (if both have price info)
        if weights["price"] > 0:
            q_price = (query_features.price_min or 0) + (query_features.price_max or 0)
            if q_price > 0:
                for i, cf in enumerate(candidate_features):
                    c_price = (cf.price_min or 0) + (cf.price_max or 0)
                    if c_price > 0:
                        # Gaussian similarity around price
                        price_diff = abs(q_price - c_price) / max(q_price, c_price)
                        price_sim = np.exp(-price_diff * 2)  # Decay factor
                        similarities[i] += weights["price"] * price_sim

        # Time similarity (hour of day, day of week)
        if weights["time"] > 0 and query_features.datetime_features:
            q_hour = query_features.datetime_features.get("hour", 0)
            q_dow = query_features.datetime_features.get("day_of_week", 0)

            for i, cf in enumerate(candidate_features):
                if cf.datetime_features:
                    c_hour = cf.datetime_features.get("hour", 0)
                    c_dow = cf.datetime_features.get("day_of_week", 0)

                    # Hour similarity (cyclic)
                    hour_diff = min(abs(q_hour - c_hour), 1 - abs(q_hour - c_hour))
                    hour_sim = 1 - hour_diff

                    # Day of week similarity
                    dow_sim = 1 - abs(q_dow - c_dow)

                    time_sim = (hour_sim + dow_sim) / 2
                    similarities[i] += weights["time"] * time_sim

        # Ensure similarities are in [0, 1]
        similarities = np.clip(similarities, 0, 1)

        return similarities

    def save_embeddings(self, event_features: List[EventFeatures], filepath: str):
        """Save event embeddings to disk."""

        # Convert to serializable format
        data = []
        for ef in event_features:
            data.append(
                {
                    "text_embedding": ef.text_embedding.tolist(),
                    "title": ef.title,
                    "description": ef.description,
                    "genres": ef.genres,
                    "artists": ef.artists,
                    "venue_name": ef.venue_name,
                    "venue_lat": ef.venue_lat,
                    "venue_lon": ef.venue_lon,
                    "h3_index": ef.h3_index,
                    "price_min": ef.price_min,
                    "price_max": ef.price_max,
                    "popularity_score": ef.popularity_score,
                    "datetime_features": ef.datetime_features,
                }
            )

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(data)} event embeddings to {filepath}")

    def load_embeddings(self, filepath: str) -> List[EventFeatures]:
        """Load event embeddings from disk."""

        with open(filepath, "r") as f:
            data = json.load(f)

        event_features = []
        for item in data:
            ef = EventFeatures(
                text_embedding=np.array(item["text_embedding"]),
                title=item["title"],
                description=item["description"],
                genres=item["genres"],
                artists=item["artists"],
                venue_name=item["venue_name"],
                venue_lat=item["venue_lat"],
                venue_lon=item["venue_lon"],
                h3_index=item["h3_index"],
                price_min=item["price_min"],
                price_max=item["price_max"],
                popularity_score=item["popularity_score"],
                datetime_features=item["datetime_features"],
            )
            event_features.append(ef)

        logger.info(f"Loaded {len(event_features)} event embeddings from {filepath}")
        return event_features


def main():
    """Example usage of ContentEmbedder."""

    # Sample events
    events = [
        {
            "title": "Kollektiv Turmstrasse Live",
            "description": "Electronic techno night with underground vibes",
            "genres": ["techno", "electronic"],
            "artists": ["Kollektiv Turmstrasse"],
            "venue_name": "Culture Box",
            "venue_lat": 55.6826,
            "venue_lon": 12.5941,
            "price_min": 200,
            "price_max": 300,
            "popularity_score": 0.8,
        },
        {
            "title": "Agnes Obel Concert",
            "description": "Intimate indie performance with classical elements",
            "genres": ["indie", "alternative", "classical"],
            "artists": ["Agnes Obel"],
            "venue_name": "Vega",
            "venue_lat": 55.6667,
            "venue_lon": 12.5419,
            "price_min": 400,
            "price_max": 600,
            "popularity_score": 0.9,
        },
    ]

    # Initialize embedder
    embedder = ContentEmbedder()

    # Encode events
    event_features = embedder.encode_events(events)

    # Compute similarity between events
    if len(event_features) >= 2:
        similarities = embedder.compute_similarity(
            event_features[0], event_features[1:]
        )
        print(f"Similarity between events: {similarities[0]:.3f}")

    # Save embeddings
    embedder.save_embeddings(event_features, "test_embeddings.json")

    print(f"Encoded {len(event_features)} events successfully")


if __name__ == "__main__":
    main()
