#!/usr/bin/env python3
"""
LangChain-powered event recommendation service.
Replaces complex ML pipeline with simple, powerful LangChain components.
"""

from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json

# LangChain imports
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Database
import duckdb
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EventRecommendation:
    """A single event recommendation with explanation."""

    event_id: str
    score: float
    reasons: List[str]
    explanation: str


class RecommendationExplanation(BaseModel):
    """Structured explanation for why an event was recommended."""

    reasons: List[str] = Field(
        description="List of specific reasons for recommendation"
    )
    explanation: str = Field(description="Natural language explanation")
    confidence: float = Field(description="Confidence score 0-1")


class LangChainRecommender:
    """
    LangChain-powered recommendation system.

    Features:
    - Semantic search with sentence transformers
    - LLM-powered explanations
    - Vector similarity for content matching
    - RAG for contextual recommendations
    """

    def __init__(
        self,
        db_path: str = "data/events.duckdb",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        vector_store_path: str = "vector_store",
    ):
        """Initialize the LangChain recommender."""
        self.db_path = db_path
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)

        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)

        # Initialize LLM for explanations (optional)
        self.llm = None  # Skip LLM for now, use rule-based explanations

        # Vector store for semantic search
        self.vector_store: Optional[FAISS] = None
        self.event_documents: List[Document] = []

        # Load existing vector store or create new one
        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """Load existing vector store or create from database."""
        vector_store_file = self.vector_store_path / "faiss_index"

        if vector_store_file.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path), self.embeddings
                )
                logger.info("Loaded existing vector store")
                return
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")

        # Create new vector store from database
        self._build_vector_store()

    def _build_vector_store(self):
        """Build vector store from event database."""
        logger.info("Building vector store from events database...")

        try:
            # Connect to database
            conn = duckdb.connect(self.db_path)

            # Query events with venue info
            events = conn.execute(
                """
                SELECT 
                    e.id,
                    e.title,
                    e.description,
                    e.date_time,
                    e.price_min,
                    e.price_max,
                    v.name as venue_name,
                    v.neighborhood,
                    v.address,
                    COALESCE(e.description, '') as full_description
                FROM events e
                JOIN venues v ON e.venue_id = v.id
                WHERE e.status = 'active'
            """
            ).fetchall()

            if not events:
                logger.warning("No events found in database")
                return

            # Create documents for vector store
            documents = []
            for event in events:
                (
                    event_id,
                    title,
                    desc,
                    date_time,
                    price_min,
                    price_max,
                    venue,
                    neighborhood,
                    address,
                    full_desc,
                ) = event

                # Create rich text representation
                content = f"""
                Event: {title}
                Description: {full_desc or 'No description'}
                Venue: {venue} in {neighborhood}
                Address: {address}
                Date: {date_time}
                Price: {price_min}-{price_max} DKK
                """

                # Create document with metadata
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        "event_id": event_id,
                        "title": title,
                        "venue": venue,
                        "neighborhood": neighborhood,
                        "date_time": str(date_time),
                        "price_min": price_min,
                        "price_max": price_max,
                    },
                )
                documents.append(doc)

            # Create vector store
            if documents:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)

                # Save vector store
                self.vector_store.save_local(str(self.vector_store_path))
                self.event_documents = documents

                logger.info(f"Built vector store with {len(documents)} events")

            conn.close()

        except Exception as e:
            logger.error(f"Failed to build vector store: {e}")

    def get_recommendations(
        self,
        user_preferences: Dict,
        location_lat: float = None,
        location_lon: float = None,
        num_recommendations: int = 10,
    ) -> List[EventRecommendation]:
        """
        Get personalized recommendations using semantic search.

        Args:
            user_preferences: User preferences including genres, price range, etc.
            location_lat: User latitude
            location_lon: User longitude
            num_recommendations: Number of recommendations to return

        Returns:
            List of EventRecommendation objects
        """
        if not self.vector_store:
            logger.warning("Vector store not available")
            return []

        try:
            # Build search query from preferences
            search_query = self._build_search_query(user_preferences)

            # Semantic search for similar events
            similar_docs = self.vector_store.similarity_search_with_score(
                search_query, k=num_recommendations * 2  # Get more to filter/rank
            )

            recommendations = []
            for doc, similarity_score in similar_docs[:num_recommendations]:
                event_id = doc.metadata["event_id"]

                # Generate explanation
                explanation = self._generate_explanation(
                    doc, user_preferences, similarity_score
                )

                recommendation = EventRecommendation(
                    event_id=event_id,
                    score=1.0 - similarity_score,  # Convert distance to similarity
                    reasons=explanation.get("reasons", ["Semantic similarity"]),
                    explanation=explanation.get(
                        "explanation", "Matches your preferences"
                    ),
                )
                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def _build_search_query(self, preferences: Dict) -> str:
        """Build search query from user preferences."""
        query_parts = []

        # Add preferred genres
        if "preferred_genres" in preferences:
            genres = preferences["preferred_genres"]
            if genres:
                query_parts.append(f"Music genres: {', '.join(genres)}")

        # Add price preferences
        if "max_price" in preferences:
            query_parts.append(
                f"Affordable events under {preferences['max_price']} DKK"
            )

        # Add location preferences
        if "neighborhood" in preferences:
            query_parts.append(f"Events in {preferences['neighborhood']}")

        # Default query if no specific preferences
        if not query_parts:
            query_parts.append("Popular nightlife events in Copenhagen")

        return " ".join(query_parts)

    def _generate_explanation(
        self, event_doc: Document, user_preferences: Dict, similarity_score: float
    ) -> Dict:
        """Generate explanation for why event was recommended."""

        # Simple rule-based explanation (can be enhanced with LLM)
        reasons = []

        # Check genre match
        preferred_genres = user_preferences.get("preferred_genres", [])
        event_content = event_doc.page_content.lower()

        for genre in preferred_genres:
            if genre.lower() in event_content:
                reasons.append(f"Matches your {genre} preference")

        # Check price match
        max_price = user_preferences.get("max_price")
        event_price = event_doc.metadata.get("price_min", 0)
        if max_price and event_price <= max_price:
            reasons.append("Within your price range")

        # Check neighborhood
        preferred_area = user_preferences.get("neighborhood")
        event_area = event_doc.metadata.get("neighborhood")
        if preferred_area and preferred_area.lower() in (event_area or "").lower():
            reasons.append(f"Located in your preferred area: {event_area}")

        # Default reasons based on similarity
        if not reasons:
            if similarity_score < 0.3:
                reasons.append("Highly relevant to your interests")
            elif similarity_score < 0.5:
                reasons.append("Good match for your preferences")
            else:
                reasons.append("Recommended based on content similarity")

        explanation = f"This event is recommended because: {', '.join(reasons)}"

        return {
            "reasons": reasons,
            "explanation": explanation,
            "confidence": 1.0 - similarity_score,
        }

    def search_events(self, query: str, limit: int = 10) -> List[Dict]:
        """Search events using semantic similarity."""
        if not self.vector_store:
            return []

        try:
            # Semantic search
            results = self.vector_store.similarity_search_with_score(query, k=limit)

            search_results = []
            for doc, score in results:
                result = {
                    "id": doc.metadata["event_id"],
                    "title": doc.metadata["title"],
                    "relevance": 1.0 - score,
                    "match_type": "semantic",
                    "venue": doc.metadata.get("venue", ""),
                    "neighborhood": doc.metadata.get("neighborhood", ""),
                }
                search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return []

    def refresh_vector_store(self):
        """Refresh vector store with latest events."""
        logger.info("Refreshing vector store...")
        self._build_vector_store()


# Global instance
recommender = LangChainRecommender()
