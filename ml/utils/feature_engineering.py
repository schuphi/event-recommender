#!/usr/bin/env python3
"""
Feature engineering utilities for the hybrid recommendation system.
Extracts and processes structured features for ranking.
"""

import numpy as np
import h3
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for event recommendation."""

    def __init__(self):
        # Copenhagen-specific defaults
        self.copenhagen_center = (55.6761, 12.5683)
        self.avg_copenhagen_price = 350.0
        self.price_std = 200.0

        # Time-based features
        self.peak_hours = [19, 20, 21, 22, 23]  # Evening peak
        self.weekend_days = [4, 5, 6]  # Fri, Sat, Sun

    def extract_temporal_features(self, event_datetime: datetime) -> Dict[str, float]:
        """Extract time-based features from event datetime."""

        now = datetime.now()

        features = {}

        # Hour of day features
        hour = event_datetime.hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        features["is_peak_hour"] = float(hour in self.peak_hours)

        # Day of week features
        day_of_week = event_datetime.weekday()
        features["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        features["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
        features["is_weekend"] = float(day_of_week in self.weekend_days)

        # Month features (seasonality)
        month = event_datetime.month
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)

        # Days until event
        days_until = (event_datetime - now).days
        features["days_until"] = days_until
        features["days_until_log"] = np.log(max(1, days_until))
        features["is_this_week"] = float(0 <= days_until <= 7)
        features["is_this_month"] = float(0 <= days_until <= 30)

        return features

    def extract_geo_features(
        self,
        event_lat: float,
        event_lon: float,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
    ) -> Dict[str, float]:
        """Extract geographic features."""

        features = {}

        # Distance from Copenhagen center
        try:
            center_distance = geodesic(
                (event_lat, event_lon), self.copenhagen_center
            ).kilometers
            features["distance_from_center"] = center_distance
            features["is_city_center"] = float(center_distance < 3.0)
        except:
            features["distance_from_center"] = 0.0
            features["is_city_center"] = 1.0

        # Distance from user (if provided)
        if user_lat and user_lon:
            try:
                user_distance = geodesic(
                    (event_lat, event_lon), (user_lat, user_lon)
                ).kilometers
                features["distance_from_user"] = user_distance
                features["is_nearby"] = float(user_distance < 5.0)
                features["is_walkable"] = float(user_distance < 2.0)
            except:
                features["distance_from_user"] = 10.0
                features["is_nearby"] = 0.0
                features["is_walkable"] = 0.0

        # H3 geographic features
        try:
            h3_index = h3.geo_to_h3(event_lat, event_lon, 8)
            features["h3_index"] = h3_index

            # Get neighboring cells for spatial features
            neighbors = h3.k_ring(h3_index, 1)
            features["num_h3_neighbors"] = len(neighbors)
        except:
            features["h3_index"] = ""
            features["num_h3_neighbors"] = 0

        return features

    def extract_price_features(
        self,
        price_min: Optional[float],
        price_max: Optional[float],
        user_budget_min: Optional[float] = None,
        user_budget_max: Optional[float] = None,
    ) -> Dict[str, float]:
        """Extract price-related features."""

        features = {}

        # Basic price features
        price_min = price_min or 0.0
        price_max = price_max or price_min

        features["price_min"] = price_min
        features["price_max"] = price_max
        features["price_avg"] = (price_min + price_max) / 2
        features["price_range"] = price_max - price_min
        features["is_free"] = float(price_max == 0)

        # Normalized prices (relative to Copenhagen average)
        features["price_min_norm"] = price_min / self.avg_copenhagen_price
        features["price_max_norm"] = price_max / self.avg_copenhagen_price

        # Price categories
        features["is_budget"] = float(price_max < 200)
        features["is_mid_range"] = float(200 <= price_max <= 500)
        features["is_premium"] = float(price_max > 500)

        # User budget compatibility (if provided)
        if user_budget_min is not None and user_budget_max is not None:
            # Check if price ranges overlap
            overlap = max(
                0, min(price_max, user_budget_max) - max(price_min, user_budget_min)
            )
            user_range = user_budget_max - user_budget_min
            event_range = price_max - price_min or 1

            features["budget_overlap_ratio"] = overlap / min(user_range, event_range)
            features["is_affordable"] = float(price_min <= user_budget_max)
            features["is_within_budget"] = float(price_max <= user_budget_max)
            features["budget_ratio"] = (price_min + price_max) / 2 / user_budget_max

        return features

    def extract_popularity_features(
        self,
        popularity_score: float,
        view_count: Optional[int] = None,
        like_count: Optional[int] = None,
        going_count: Optional[int] = None,
    ) -> Dict[str, float]:
        """Extract popularity and engagement features."""

        features = {}

        # Basic popularity
        features["popularity_score"] = popularity_score
        features["popularity_norm"] = min(1.0, popularity_score)  # Ensure 0-1 range

        # Engagement metrics (if available)
        if view_count is not None:
            features["view_count"] = view_count
            features["view_count_log"] = np.log(max(1, view_count))

        if like_count is not None:
            features["like_count"] = like_count
            features["like_count_log"] = np.log(max(1, like_count))

        if going_count is not None:
            features["going_count"] = going_count
            features["going_count_log"] = np.log(max(1, going_count))

            # Engagement ratios
            if view_count and view_count > 0:
                features["like_rate"] = like_count / view_count
                features["going_rate"] = going_count / view_count

        # Popularity categories
        features["is_trending"] = float(popularity_score > 0.8)
        features["is_niche"] = float(popularity_score < 0.3)

        return features

    def extract_content_features(
        self,
        title: str,
        description: str,
        genres: List[str],
        artists: List[str],
        venue_name: str,
    ) -> Dict[str, float]:
        """Extract content-based features."""

        features = {}

        # Text length features
        features["title_length"] = len(title) if title else 0
        features["description_length"] = len(description) if description else 0
        features["has_description"] = float(bool(description))

        # Content richness
        features["num_genres"] = len(genres)
        features["num_artists"] = len(artists)
        features["has_genres"] = float(len(genres) > 0)
        features["has_artists"] = float(len(artists) > 0)

        # Genre-specific features (common Copenhagen genres)
        common_genres = [
            "techno",
            "house",
            "electronic",
            "indie",
            "rock",
            "jazz",
            "pop",
        ]

        for genre in common_genres:
            genre_key = f"genre_{genre}"
            features[genre_key] = float(genre.lower() in [g.lower() for g in genres])

        # Venue features
        features["venue_name_length"] = len(venue_name) if venue_name else 0

        # Well-known Copenhagen venues
        famous_venues = ["vega", "rust", "culture box", "loppen", "kb18", "pumpehuset"]

        venue_lower = venue_name.lower() if venue_name else ""
        features["is_famous_venue"] = float(
            any(venue in venue_lower for venue in famous_venues)
        )

        return features

    def extract_user_context_features(
        self,
        user_preferences: Dict,
        user_history: List[Dict] = None,
        current_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Extract user context features."""

        features = {}
        current_time = current_time or datetime.now()

        # User preference features
        preferred_genres = user_preferences.get("preferred_genres", [])
        preferred_artists = user_preferences.get("preferred_artists", [])

        features["num_preferred_genres"] = len(preferred_genres)
        features["num_preferred_artists"] = len(preferred_artists)
        features["has_genre_prefs"] = float(len(preferred_genres) > 0)
        features["has_artist_prefs"] = float(len(preferred_artists) > 0)

        # Budget features
        price_range = user_preferences.get("price_range", (0, 1000))
        features["user_budget_min"] = price_range[0]
        features["user_budget_max"] = price_range[1]
        features["user_budget_range"] = price_range[1] - price_range[0]
        features["is_budget_conscious"] = float(price_range[1] < 300)

        # Time preferences
        preferred_times = user_preferences.get("preferred_times", [])
        preferred_days = user_preferences.get("preferred_days", [])

        features["num_preferred_times"] = len(preferred_times)
        features["num_preferred_days"] = len(preferred_days)
        features["prefers_evenings"] = float(
            any(hour >= 18 for hour in preferred_times)
        )
        features["prefers_weekends"] = float(
            any(day in [5, 6] for day in preferred_days)  # Sat, Sun
        )

        # User activity features (if history provided)
        if user_history:
            features["user_activity_count"] = len(user_history)
            features["is_active_user"] = float(len(user_history) >= 10)

            # Recent activity
            recent_cutoff = current_time - timedelta(days=30)
            recent_activities = [
                h
                for h in user_history
                if h.get("timestamp", datetime.min) > recent_cutoff
            ]
            features["recent_activity_count"] = len(recent_activities)
            features["is_recently_active"] = float(len(recent_activities) >= 3)

        return features

    def extract_diversity_features(
        self, event_data: Dict, recommended_events: List[Dict]
    ) -> Dict[str, float]:
        """Extract diversity features relative to already recommended events."""

        features = {}

        if not recommended_events:
            features["diversity_score"] = 1.0
            return features

        # Genre diversity
        event_genres = set(event_data.get("genres", []))
        recommended_genres = set()
        for rec_event in recommended_events:
            recommended_genres.update(rec_event.get("genres", []))

        if recommended_genres:
            genre_overlap = len(event_genres & recommended_genres)
            genre_diversity = 1.0 - (genre_overlap / len(recommended_genres))
            features["genre_diversity"] = genre_diversity
        else:
            features["genre_diversity"] = 1.0

        # Venue diversity
        event_venue = event_data.get("venue_name", "").lower()
        recommended_venues = {
            rec_event.get("venue_name", "").lower() for rec_event in recommended_events
        }

        features["venue_diversity"] = float(event_venue not in recommended_venues)

        # Artist diversity
        event_artists = set(event_data.get("artists", []))
        recommended_artists = set()
        for rec_event in recommended_events:
            recommended_artists.update(rec_event.get("artists", []))

        if recommended_artists:
            artist_overlap = len(event_artists & recommended_artists)
            artist_diversity = 1.0 - (artist_overlap / len(recommended_artists))
            features["artist_diversity"] = artist_diversity
        else:
            features["artist_diversity"] = 1.0

        # Overall diversity score
        features["diversity_score"] = (
            features["genre_diversity"]
            + features["venue_diversity"]
            + features["artist_diversity"]
        ) / 3.0

        return features

    def normalize_features(
        self,
        features: Dict[str, float],
        feature_stats: Dict[str, Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Normalize features using statistics or default normalization."""

        normalized = features.copy()

        # Features that need normalization
        normalization_configs = {
            "distance_from_user": {"max": 20.0},  # Max 20km
            "distance_from_center": {"max": 15.0},  # Max 15km from center
            "days_until": {"max": 90.0},  # Max 90 days ahead
            "price_min": {"max": 1000.0},  # Max 1000 DKK
            "price_max": {"max": 1000.0},
            "view_count": {"log": True},  # Log normalization
            "like_count": {"log": True},
            "going_count": {"log": True},
        }

        for feature_name, value in features.items():
            if feature_name in normalization_configs:
                config = normalization_configs[feature_name]

                if config.get("log"):
                    normalized[feature_name] = (
                        np.log(max(1, value)) / 10.0
                    )  # Divide by 10 for scaling
                elif "max" in config:
                    normalized[feature_name] = min(1.0, value / config["max"])
                elif feature_stats and feature_name in feature_stats:
                    stats = feature_stats[feature_name]
                    mean = stats.get("mean", 0)
                    std = stats.get("std", 1)
                    normalized[feature_name] = (value - mean) / std

        return normalized


def main():
    """Example usage of FeatureEngineer."""

    engineer = FeatureEngineer()

    # Sample event data
    event_datetime = datetime.now() + timedelta(
        days=5, hours=3
    )  # 5 days from now, 3 hours later

    # Extract temporal features
    temporal_features = engineer.extract_temporal_features(event_datetime)
    print("Temporal features:", temporal_features)

    # Extract geo features
    geo_features = engineer.extract_geo_features(
        event_lat=55.6826,  # Culture Box
        event_lon=12.5941,
        user_lat=55.6761,  # Copenhagen center
        user_lon=12.5683,
    )
    print("Geo features:", geo_features)

    # Extract price features
    price_features = engineer.extract_price_features(
        price_min=200, price_max=300, user_budget_min=100, user_budget_max=400
    )
    print("Price features:", price_features)


if __name__ == "__main__":
    main()
