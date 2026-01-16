"""
Topic classifier for event categorization.

Uses a two-tier approach:
1. Rule-based keyword matching (fast, deterministic)
2. Embedding similarity fallback (for ambiguous cases)
"""

import re
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Topic definitions with descriptions
TOPICS = {
    "tech": "Technology meetups, hackathons, conferences, startup events, OpenAI, robotics, AI, researchers",
    "nightlife": "Club nights, DJ sets, bar events, late-night parties",
    "music": "Concerts, live performances, festivals, acoustic sessions",
    "sports": "Matches, fitness classes, outdoor activities, tournaments, basketball",
}

# Keywords for rule-based classification (lowercase)
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "tech": [
        # Core tech terms
        "hackathon", "startup", "developer", "coding", "programming",
        "software", "tech", "technology", "engineering", "engineer",
        # AI/ML
        "ai", "artificial intelligence", "machine learning", "ml", "openai",
        "chatgpt", "llm", "deep learning", "neural", "data science",
        # Specific tech
        "python", "javascript", "react", "kubernetes", "cloud", "devops",
        "blockchain", "web3", "crypto", "robotics", "automation",
        # Events
        "meetup", "conference", "workshop", "seminar", "talk",
        "pitch", "demo day", "incubator", "accelerator",
        # Organizations
        "google", "microsoft", "amazon", "meta", "apple",
        "github", "aws", "azure",
        # Research
        "researcher", "phd", "academic", "university", "science",
    ],
    "nightlife": [
        # Club/party terms
        "club", "clubbing", "nightclub", "party", "parties",
        "rave", "afterparty", "after party", "late night",
        # DJ/electronic
        "dj", "techno", "house music", "electronic", "edm",
        "bass", "drum and bass", "dnb", "trance", "minimal",
        # Venue types
        "bar", "lounge", "rooftop", "underground",
        # Event descriptors
        "night", "midnight", "2am", "3am", "4am", "all night",
        "dance floor", "dancing",
    ],
    "music": [
        # Performance types
        "concert", "live music", "live performance", "gig", "show",
        "festival", "acoustic", "unplugged", "orchestra", "symphony",
        "choir", "recital", "opera", "musical",
        # Genres (non-electronic)
        "rock", "jazz", "blues", "folk", "indie", "pop",
        "classical", "hip hop", "rap", "r&b", "soul", "funk",
        "metal", "punk", "country", "reggae", "latin",
        # Music terms
        "band", "singer", "vocalist", "musician", "artist",
        "album", "tour", "release party",
    ],
    "sports": [
        # Team sports
        "football", "soccer", "basketball", "volleyball", "handball",
        "hockey", "baseball", "rugby", "cricket",
        # Individual sports
        "tennis", "badminton", "golf", "boxing", "mma", "wrestling",
        "swimming", "cycling", "running", "marathon", "triathlon",
        # Fitness
        "fitness", "gym", "workout", "training", "crossfit",
        "yoga", "pilates", "spinning", "aerobics", "zumba",
        # Outdoor
        "hiking", "climbing", "surfing", "skating", "skiing",
        "outdoor", "adventure", "expedition",
        # Events
        "match", "game", "tournament", "championship", "league",
        "competition", "race", "cup",
    ],
}

# Negative keywords - if present, reduce confidence for certain topics
NEGATIVE_KEYWORDS: Dict[str, List[str]] = {
    "tech": ["techno", "tech house", "tech-house"],  # Music genres, not tech
    "music": ["music festival coding"],  # Compound terms
}

# Venue type hints
VENUE_TOPIC_HINTS: Dict[str, str] = {
    "club": "nightlife",
    "nightclub": "nightlife",
    "bar": "nightlife",
    "concert_hall": "music",
    "theater": "music",
    "stadium": "sports",
    "arena": "sports",
    "gym": "sports",
    "coworking": "tech",
    "conference_center": "tech",
}


@dataclass
class ClassificationResult:
    """Result of topic classification."""
    topic: str
    confidence: float
    method: str  # "rule_based" or "embedding"
    matched_keywords: List[str]


class TopicClassifier:
    """
    Classifier for event topics.

    Uses rule-based keyword matching first, falls back to
    embedding similarity for ambiguous cases.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.6,
        use_embeddings: bool = True,
    ):
        """
        Initialize the classifier.

        Args:
            embedding_model_name: Sentence transformer model for embeddings
            confidence_threshold: Minimum confidence for rule-based classification
            use_embeddings: Whether to use embedding fallback
        """
        self.confidence_threshold = confidence_threshold
        self.use_embeddings = use_embeddings
        self._embedding_model = None
        self._topic_embeddings = None
        self._embedding_model_name = embedding_model_name

        # Compile keyword patterns for efficiency
        self._keyword_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for keyword matching."""
        patterns = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            patterns[topic] = [
                re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                for kw in keywords
            ]
        return patterns

    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self._embedding_model_name)

                # Pre-compute topic embeddings
                topic_texts = [
                    f"{topic}: {description}"
                    for topic, description in TOPICS.items()
                ]
                self._topic_embeddings = self._embedding_model.encode(topic_texts)
                self._topic_names = list(TOPICS.keys())

                logger.info("Loaded embedding model for topic classification")
            except ImportError:
                logger.warning("sentence-transformers not installed, embedding fallback disabled")
                self.use_embeddings = False
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.use_embeddings = False

    def classify(
        self,
        title: str,
        description: str = "",
        venue_type: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify an event into a topic.

        Args:
            title: Event title
            description: Event description
            venue_type: Optional venue type for hints

        Returns:
            ClassificationResult with topic, confidence, and method
        """
        text = f"{title} {description}".lower()

        # Try rule-based classification first
        result = self._rule_based_classify(text, venue_type)

        if result.confidence >= self.confidence_threshold:
            return result

        # Fall back to embedding similarity
        if self.use_embeddings:
            embedding_result = self._embedding_classify(title, description)

            # Use embedding result if it's more confident
            if embedding_result.confidence > result.confidence:
                return embedding_result

        # Return rule-based result even if low confidence
        return result

    def _rule_based_classify(
        self,
        text: str,
        venue_type: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify using keyword matching.

        Returns topic with highest keyword match count.
        """
        scores: Dict[str, float] = {topic: 0.0 for topic in TOPICS}
        matched: Dict[str, List[str]] = {topic: [] for topic in TOPICS}

        # Score each topic based on keyword matches
        for topic, patterns in self._keyword_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    keyword = pattern.pattern.replace(r'\b', '').replace('\\', '')
                    matched[topic].append(keyword)
                    scores[topic] += 1.0

        # Apply negative keyword penalties
        for topic, neg_keywords in NEGATIVE_KEYWORDS.items():
            for neg_kw in neg_keywords:
                if neg_kw.lower() in text:
                    scores[topic] -= 0.5

        # Apply venue type hints
        if venue_type and venue_type.lower() in VENUE_TOPIC_HINTS:
            hinted_topic = VENUE_TOPIC_HINTS[venue_type.lower()]
            scores[hinted_topic] += 0.5

        # Find best topic
        best_topic = max(scores, key=scores.get)
        best_score = scores[best_topic]

        # Calculate confidence (normalize by max possible score)
        total_keywords = sum(len(patterns) for patterns in self._keyword_patterns.values())
        max_reasonable_matches = 10  # Most events won't match more than 10 keywords
        confidence = min(best_score / max_reasonable_matches, 1.0)

        # If no matches, default to music with low confidence
        if best_score == 0:
            return ClassificationResult(
                topic="music",
                confidence=0.3,
                method="rule_based",
                matched_keywords=[],
            )

        return ClassificationResult(
            topic=best_topic,
            confidence=confidence,
            method="rule_based",
            matched_keywords=matched[best_topic],
        )

    def _embedding_classify(
        self,
        title: str,
        description: str,
    ) -> ClassificationResult:
        """
        Classify using embedding similarity.
        """
        self._load_embedding_model()

        if self._embedding_model is None:
            return ClassificationResult(
                topic="music",
                confidence=0.3,
                method="embedding",
                matched_keywords=[],
            )

        try:
            import numpy as np

            # Encode the event text
            event_text = f"{title}. {description}"
            event_embedding = self._embedding_model.encode([event_text])[0]

            # Compute cosine similarity with each topic
            similarities = []
            for topic_emb in self._topic_embeddings:
                sim = np.dot(event_embedding, topic_emb) / (
                    np.linalg.norm(event_embedding) * np.linalg.norm(topic_emb)
                )
                similarities.append(sim)

            # Find best match
            best_idx = np.argmax(similarities)
            best_topic = self._topic_names[best_idx]
            confidence = float(similarities[best_idx])

            # Normalize confidence to 0-1 range (cosine sim can be negative)
            confidence = (confidence + 1) / 2

            return ClassificationResult(
                topic=best_topic,
                confidence=confidence,
                method="embedding",
                matched_keywords=[],
            )

        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}")
            return ClassificationResult(
                topic="music",
                confidence=0.3,
                method="embedding",
                matched_keywords=[],
            )

    def classify_batch(
        self,
        events: List[Dict],
    ) -> List[ClassificationResult]:
        """
        Classify multiple events efficiently.

        Args:
            events: List of event dicts with 'title' and 'description' keys

        Returns:
            List of ClassificationResult objects
        """
        results = []

        for event in events:
            title = event.get("title", "")
            description = event.get("description", "")
            venue_type = event.get("venue_type")

            result = self.classify(title, description, venue_type)
            results.append(result)

        return results

    def get_topic_keywords(self, topic: str) -> List[str]:
        """Get keywords for a specific topic."""
        return TOPIC_KEYWORDS.get(topic, [])

    def suggest_tags(
        self,
        title: str,
        description: str,
        price_min: Optional[float] = None,
    ) -> List[str]:
        """
        Suggest secondary tags for an event.

        Args:
            title: Event title
            description: Event description
            price_min: Minimum price (for free tag)

        Returns:
            List of suggested tags
        """
        tags = []
        text = f"{title} {description}".lower()

        # Free events
        if price_min is None or price_min == 0:
            if any(word in text for word in ["free", "gratis", "no cover", "free entry"]):
                tags.append("free")

        # Outdoor events
        if any(word in text for word in ["outdoor", "outside", "park", "garden", "rooftop", "open air"]):
            tags.append("outdoor")

        # Family friendly
        if any(word in text for word in ["family", "kids", "children", "all ages"]):
            tags.append("family-friendly")

        # 18+ / Adult
        if any(word in text for word in ["18+", "21+", "adults only", "mature"]):
            tags.append("18+")

        # Language
        if any(word in text for word in ["english", "in english"]):
            tags.append("english")
        if any(word in text for word in ["danish", "på dansk", "dansk"]):
            tags.append("danish")

        # Recurring
        if any(word in text for word in ["every week", "weekly", "monthly", "recurring"]):
            tags.append("recurring")

        return tags


def main():
    """Test the classifier."""
    classifier = TopicClassifier(use_embeddings=False)  # Fast mode

    test_events = [
        {"title": "Copenhagen AI Meetup", "description": "Join us for talks on machine learning and OpenAI"},
        {"title": "Techno Night at Culture Box", "description": "Underground electronic music with top DJs"},
        {"title": "Jazz Concert at Vega", "description": "Live performance by the Copenhagen Jazz Quartet"},
        {"title": "FC Copenhagen vs Brondby", "description": "Danish Superliga football match at Parken"},
        {"title": "Startup Pitch Night", "description": "Watch startups pitch to investors"},
        {"title": "Outdoor Yoga in the Park", "description": "Free yoga session for all levels"},
    ]

    print("Topic Classification Results:")
    print("-" * 60)

    for event in test_events:
        result = classifier.classify(event["title"], event["description"])
        tags = classifier.suggest_tags(event["title"], event["description"])

        print(f"Title: {event['title']}")
        print(f"Topic: {result.topic} (confidence: {result.confidence:.2f})")
        print(f"Method: {result.method}")
        print(f"Keywords: {', '.join(result.matched_keywords[:5])}")
        print(f"Tags: {', '.join(tags)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
