#!/usr/bin/env python3
"""
Configuration settings for the Copenhagen Event Recommender API.
"""

from pydantic import BaseSettings, Field
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        env="API_CORS_ORIGINS"
    )
    
    # Database Configuration
    DATABASE_URL: str = Field(default="data/events.duckdb", env="DATABASE_URL")
    DB_POOL_SIZE: int = Field(default=10, env="DB_POOL_SIZE")
    
    # ML Model Configuration
    SENTENCE_TRANSFORMER_MODEL: str = Field(
        default="all-MiniLM-L6-v2", 
        env="SENTENCE_TRANSFORMER_MODEL"
    )
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")
    H3_RESOLUTION: int = Field(default=8, env="H3_RESOLUTION")
    
    # Model Paths
    MODELS_DIR: str = Field(default="ml/models", env="MODELS_DIR")
    CONTENT_MODEL_DIR: str = Field(default="ml/models/content_based", env="CONTENT_MODEL_DIR")
    CF_MODEL_DIR: str = Field(default="ml/models/collaborative", env="CF_MODEL_DIR")
    HYBRID_MODEL_DIR: str = Field(default="ml/models/hybrid", env="HYBRID_MODEL_DIR")
    
    # Feature Engineering
    MAX_DISTANCE_KM: float = Field(default=50.0, env="MAX_DISTANCE_KM")
    DEFAULT_RECOMMENDATION_COUNT: int = Field(default=10, env="DEFAULT_RECOMMENDATION_COUNT")
    MAX_RECOMMENDATION_COUNT: int = Field(default=50, env="MAX_RECOMMENDATION_COUNT")
    
    # Caching
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour
    RECOMMENDATION_CACHE_TTL: int = Field(default=1800, env="RECOMMENDATION_CACHE_TTL")  # 30 minutes
    
    # Authentication (optional)
    JWT_SECRET_KEY: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_HOURS: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # External APIs
    EVENTBRITE_API_TOKEN: Optional[str] = Field(default=None, env="EVENTBRITE_API_TOKEN")
    SPOTIFY_CLIENT_ID: Optional[str] = Field(default=None, env="SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET: Optional[str] = Field(default=None, env="SPOTIFY_CLIENT_SECRET")
    LASTFM_API_KEY: Optional[str] = Field(default=None, env="LASTFM_API_KEY")
    
    # Social Media Scraping
    INSTAGRAM_USERNAME: Optional[str] = Field(default=None, env="INSTAGRAM_USERNAME")
    INSTAGRAM_PASSWORD: Optional[str] = Field(default=None, env="INSTAGRAM_PASSWORD")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Model Training
    AUTO_RETRAIN_HOURS: int = Field(default=24, env="AUTO_RETRAIN_HOURS")  # Retrain every 24 hours
    MIN_INTERACTIONS_FOR_CF: int = Field(default=100, env="MIN_INTERACTIONS_FOR_CF")
    TRAINING_BATCH_SIZE: int = Field(default=1024, env="TRAINING_BATCH_SIZE")
    
    # Content Generation
    MAX_DESCRIPTION_LENGTH: int = Field(default=500, env="MAX_DESCRIPTION_LENGTH")
    MAX_TITLE_LENGTH: int = Field(default=100, env="MAX_TITLE_LENGTH")
    
    # Geo Configuration
    COPENHAGEN_CENTER_LAT: float = Field(default=55.6761, env="COPENHAGEN_CENTER_LAT")
    COPENHAGEN_CENTER_LON: float = Field(default=12.5683, env="COPENHAGEN_CENTER_LON")
    COPENHAGEN_RADIUS_KM: float = Field(default=25.0, env="COPENHAGEN_RADIUS_KM")
    
    # Data Collection
    SCRAPING_ENABLED: bool = Field(default=False, env="SCRAPING_ENABLED")
    SCRAPING_SCHEDULE_CRON: str = Field(default="0 2 * * *", env="SCRAPING_SCHEDULE_CRON")  # Daily at 2 AM
    MAX_EVENTS_PER_SCRAPER: int = Field(default=1000, env="MAX_EVENTS_PER_SCRAPER")
    
    # Feature Flags
    ENABLE_COLLABORATIVE_FILTERING: bool = Field(default=True, env="ENABLE_COLLABORATIVE_FILTERING")
    ENABLE_VIRAL_DISCOVERY: bool = Field(default=True, env="ENABLE_VIRAL_DISCOVERY")
    ENABLE_NEURAL_RANKER: bool = Field(default=False, env="ENABLE_NEURAL_RANKER")
    ENABLE_COLD_START_ONBOARDING: bool = Field(default=True, env="ENABLE_COLD_START_ONBOARDING")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    RECOMMENDATION_TIMEOUT_SECONDS: int = Field(default=10, env="RECOMMENDATION_TIMEOUT_SECONDS")
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(self.API_CORS_ORIGINS, str):
            return [origin.strip() for origin in self.API_CORS_ORIGINS.split(",")]
        return self.API_CORS_ORIGINS
    
    @property
    def DATABASE_PATH(self) -> Path:
        """Get database path as Path object."""
        return Path(self.DATABASE_URL)
    
    @property
    def MODELS_PATH(self) -> Path:
        """Get models directory as Path object."""
        return Path(self.MODELS_DIR)
    
    def get_model_path(self, model_type: str) -> Path:
        """Get path for specific model type."""
        model_dirs = {
            "content": self.CONTENT_MODEL_DIR,
            "collaborative": self.CF_MODEL_DIR,
            "hybrid": self.HYBRID_MODEL_DIR
        }
        
        return Path(model_dirs.get(model_type, self.MODELS_DIR))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
settings = Settings()