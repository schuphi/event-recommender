#!/usr/bin/env python3
"""
Centralized viral event discovery system that aggregates trending content
from multiple social platforms and identifies the most promising events.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import Counter, defaultdict
import re

from .tiktok import TikTokEventScraper, TikTokEvent
from .instagram_viral import InstagramViralEventScraper, InstagramEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ViralEventSignal:
    """Signal indicating viral event activity."""
    platform: str
    event_id: str
    venue_name: str
    title: str
    description: str
    engagement_score: float
    viral_indicators: List[str]
    hashtags: List[str]
    timestamp: datetime
    location: Optional[str] = None

@dataclass
class TrendingEvent:
    """Aggregated trending event from multiple signals."""
    event_key: str  # Unique identifier
    title: str
    venue_name: str
    location: str
    total_engagement: float
    platform_signals: List[ViralEventSignal]
    confidence_score: float
    viral_score: float
    predicted_attendance: int
    trending_hashtags: List[str]

class ViralEventDiscoveryEngine:
    """Engine for discovering viral/trending events across platforms."""
    
    def __init__(self):
        self.tiktok_scraper = TikTokEventScraper()
        self.instagram_scraper = InstagramViralEventScraper()
        
        # Viral signal weights by platform
        self.platform_weights = {
            'tiktok': 1.5,      # TikTok has stronger viral signals
            'instagram': 1.2,   # Instagram good for event discovery
            'cross_platform': 2.0  # Events appearing on multiple platforms
        }
        
        # Engagement thresholds for viral classification
        self.viral_thresholds = {
            'tiktok': {'likes': 1000, 'comments': 100, 'shares': 50},
            'instagram': {'likes': 500, 'comments': 50}
        }
    
    def discover_viral_events(
        self,
        days_back: int = 3,
        min_viral_score: float = 0.6,
        max_events: int = 50
    ) -> List[TrendingEvent]:
        """
        Discover trending/viral events across all platforms.
        
        Args:
            days_back: How many days back to search
            min_viral_score: Minimum viral score threshold
            max_events: Maximum events to return
        
        Returns:
            List of trending events ranked by viral score
        """
        
        logger.info("Starting viral event discovery...")
        
        # Collect signals from all platforms
        all_signals = []
        
        # TikTok viral signals
        tiktok_signals = self._collect_tiktok_signals(days_back)
        all_signals.extend(tiktok_signals)
        logger.info(f"Collected {len(tiktok_signals)} TikTok signals")
        
        # Instagram viral signals
        instagram_signals = self._collect_instagram_signals(days_back)
        all_signals.extend(instagram_signals)
        logger.info(f"Collected {len(instagram_signals)} Instagram signals")
        
        # Aggregate signals into trending events
        trending_events = self._aggregate_signals_to_events(all_signals)
        
        # Calculate viral scores
        for event in trending_events:
            event.viral_score = self._calculate_viral_score(event)
            event.confidence_score = self._calculate_confidence_score(event)
            event.predicted_attendance = self._predict_attendance(event)
        
        # Filter and sort by viral score
        viral_events = [
            event for event in trending_events 
            if event.viral_score >= min_viral_score
        ]
        
        viral_events.sort(key=lambda x: x.viral_score, reverse=True)
        
        logger.info(f"Found {len(viral_events)} viral events")
        return viral_events[:max_events]
    
    def _collect_tiktok_signals(self, days_back: int) -> List[ViralEventSignal]:
        """Collect viral signals from TikTok."""
        
        signals = []
        
        try:
            # Get viral TikTok events
            tiktok_events = self.tiktok_scraper.scrape_venues(
                days_back=days_back,
                max_videos_per_venue=30
            )
            
            for event in tiktok_events:
                # Calculate engagement score
                engagement_score = self._calculate_tiktok_engagement(event)
                
                # Extract viral indicators
                viral_indicators = self._extract_viral_indicators(event.description)
                
                signal = ViralEventSignal(
                    platform='tiktok',
                    event_id=event.id,
                    venue_name=event.venue_name,
                    title=event.title,
                    description=event.description,
                    engagement_score=engagement_score,
                    viral_indicators=viral_indicators,
                    hashtags=event.hashtags,
                    timestamp=datetime.now(),
                    location=None
                )
                
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error collecting TikTok signals: {e}")
        
        return signals
    
    def _collect_instagram_signals(self, days_back: int) -> List[ViralEventSignal]:
        """Collect viral signals from Instagram."""
        
        signals = []
        
        try:
            # Get viral Instagram events
            instagram_events = self.instagram_scraper.scrape_viral_events(
                days_back=days_back,
                max_posts_per_hashtag=30,
                min_engagement=100
            )
            
            for event in instagram_events:
                # Calculate engagement score
                engagement_score = self._calculate_instagram_engagement(event)
                
                # Extract viral indicators
                viral_indicators = self._extract_viral_indicators(event.description)
                
                signal = ViralEventSignal(
                    platform='instagram',
                    event_id=event.id,
                    venue_name=event.venue_name,
                    title=event.title,
                    description=event.description,
                    engagement_score=engagement_score,
                    viral_indicators=viral_indicators,
                    hashtags=event.hashtags,
                    timestamp=datetime.now(),
                    location=None
                )
                
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"Error collecting Instagram signals: {e}")
        
        return signals
    
    def _calculate_tiktok_engagement(self, event: TikTokEvent) -> float:
        """Calculate normalized engagement score for TikTok event."""
        
        # Weighted engagement formula
        likes_score = min(1.0, event.likes / 10000)  # Max score at 10k likes
        views_score = min(1.0, event.views / 100000)  # Max score at 100k views
        
        # Combine with weights
        engagement_score = (likes_score * 0.6 + views_score * 0.4)
        
        return engagement_score
    
    def _calculate_instagram_engagement(self, event: InstagramEvent) -> float:
        """Calculate normalized engagement score for Instagram event."""
        
        # Weighted engagement formula
        likes_score = min(1.0, event.likes / 5000)  # Max score at 5k likes
        comments_score = min(1.0, event.comments / 500)  # Max score at 500 comments
        
        # Combine with weights
        engagement_score = (likes_score * 0.7 + comments_score * 0.3)
        
        return engagement_score
    
    def _extract_viral_indicators(self, text: str) -> List[str]:
        """Extract viral language indicators from text."""
        
        text_lower = text.lower()
        
        viral_patterns = {
            'urgency': ['tonight', 'right now', 'happening now', 'dont miss', 'last chance'],
            'exclusivity': ['secret', 'underground', 'invite only', 'hidden', 'exclusive'],
            'social_proof': ['everyone talking', 'viral', 'trending', 'packed', 'sold out'],
            'discovery': ['found this', 'discovered', 'stumbled upon', 'hidden gem'],
            'excitement': ['insane', 'epic', 'legendary', 'unreal', 'crazy', 'wild'],
            'location_specific': ['warehouse', 'rooftop', 'popup', 'secret location']
        }
        
        indicators = []
        for category, patterns in viral_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append(f"{category}:{pattern}")
        
        return indicators
    
    def _aggregate_signals_to_events(self, signals: List[ViralEventSignal]) -> List[TrendingEvent]:
        """Aggregate signals into unique trending events."""
        
        # Group signals by event (venue + approximate time)
        event_groups = defaultdict(list)
        
        for signal in signals:
            # Create event key for grouping
            event_key = self._create_event_key(signal)
            event_groups[event_key].append(signal)
        
        trending_events = []
        
        for event_key, grouped_signals in event_groups.items():
            # Aggregate signals into a single trending event
            trending_event = self._create_trending_event(event_key, grouped_signals)
            trending_events.append(trending_event)
        
        return trending_events
    
    def _create_event_key(self, signal: ViralEventSignal) -> str:
        """Create a unique key for grouping similar events."""
        
        # Normalize venue name
        venue_normalized = re.sub(r'[^a-z0-9]', '', signal.venue_name.lower())
        
        # Use date (not time) for grouping events on the same day
        date_key = signal.timestamp.strftime('%Y%m%d')
        
        return f"{venue_normalized}_{date_key}"
    
    def _create_trending_event(
        self, 
        event_key: str, 
        signals: List[ViralEventSignal]
    ) -> TrendingEvent:
        """Create a TrendingEvent from grouped signals."""
        
        # Use the signal with highest engagement as primary
        primary_signal = max(signals, key=lambda s: s.engagement_score)
        
        # Aggregate engagement scores
        total_engagement = sum(s.engagement_score for s in signals)
        
        # Combine all hashtags
        all_hashtags = []
        for signal in signals:
            all_hashtags.extend(signal.hashtags)
        
        # Find most common hashtags
        hashtag_counts = Counter(all_hashtags)
        trending_hashtags = [tag for tag, count in hashtag_counts.most_common(10)]
        
        return TrendingEvent(
            event_key=event_key,
            title=primary_signal.title,
            venue_name=primary_signal.venue_name,
            location=primary_signal.location or "Copenhagen",
            total_engagement=total_engagement,
            platform_signals=signals,
            confidence_score=0.0,  # Will be calculated later
            viral_score=0.0,       # Will be calculated later
            predicted_attendance=0, # Will be calculated later
            trending_hashtags=trending_hashtags
        )
    
    def _calculate_viral_score(self, event: TrendingEvent) -> float:
        """Calculate overall viral score for an event."""
        
        # Base score from engagement
        engagement_score = min(1.0, event.total_engagement / 2.0)
        
        # Platform diversity bonus
        platforms = set(signal.platform for signal in event.platform_signals)
        platform_bonus = len(platforms) * 0.1  # Bonus for multi-platform presence
        
        # Viral indicator bonus
        all_indicators = []
        for signal in event.platform_signals:
            all_indicators.extend(signal.viral_indicators)
        
        indicator_score = min(0.3, len(set(all_indicators)) * 0.05)
        
        # Hashtag trending bonus
        hashtag_score = min(0.2, len(event.trending_hashtags) * 0.02)
        
        # Combine scores
        viral_score = engagement_score + platform_bonus + indicator_score + hashtag_score
        
        return min(1.0, viral_score)
    
    def _calculate_confidence_score(self, event: TrendingEvent) -> float:
        """Calculate confidence score for event prediction."""
        
        # More signals = higher confidence
        signal_confidence = min(1.0, len(event.platform_signals) / 3.0)
        
        # Cross-platform presence increases confidence
        platforms = set(signal.platform for signal in event.platform_signals)
        platform_confidence = len(platforms) * 0.2
        
        # Recent signals increase confidence
        now = datetime.now()
        recent_signals = [
            s for s in event.platform_signals 
            if (now - s.timestamp).hours <= 24
        ]
        recency_confidence = min(0.5, len(recent_signals) * 0.1)
        
        return min(1.0, signal_confidence + platform_confidence + recency_confidence)
    
    def _predict_attendance(self, event: TrendingEvent) -> int:
        """Predict event attendance based on viral signals."""
        
        # Base prediction from engagement
        base_attendance = int(event.total_engagement * 100)
        
        # Platform multipliers
        platform_multiplier = 1.0
        platforms = set(signal.platform for signal in event.platform_signals)
        
        if 'tiktok' in platforms:
            platform_multiplier *= 1.5
        if 'instagram' in platforms:
            platform_multiplier *= 1.3
        if len(platforms) > 1:
            platform_multiplier *= 1.2  # Cross-platform bonus
        
        # Viral indicator multipliers
        indicator_types = set()
        for signal in event.platform_signals:
            for indicator in signal.viral_indicators:
                indicator_types.add(indicator.split(':')[0])
        
        if 'urgency' in indicator_types:
            platform_multiplier *= 1.3
        if 'exclusivity' in indicator_types:
            platform_multiplier *= 1.2
        if 'social_proof' in indicator_types:
            platform_multiplier *= 1.4
        
        predicted_attendance = int(base_attendance * platform_multiplier)
        
        # Cap at reasonable maximum
        return min(predicted_attendance, 2000)
    
    def get_trending_hashtags(self, days_back: int = 7) -> List[Tuple[str, int, float]]:
        """Get trending event hashtags with growth metrics."""
        
        # This would analyze hashtag frequency and growth over time
        # Simplified implementation returning estimated trending hashtags
        
        trending_hashtags = [
            ('copenhagenevents', 1500, 0.25),  # hashtag, count, growth_rate
            ('cphtonight', 1200, 0.35),
            ('undergroundcph', 800, 0.45),
            ('secretpartycopenhagen', 600, 0.60),
            ('technocopenhagen', 900, 0.20),
            ('vesterbrovibes', 400, 0.30),
            ('warehousepartycopenhagen', 300, 0.80),
            ('popupparty', 250, 0.90)
        ]
        
        return trending_hashtags
    
    def analyze_viral_patterns(self, events: List[TrendingEvent]) -> Dict[str, any]:
        """Analyze patterns in viral event discovery."""
        
        analysis = {
            'total_events': len(events),
            'avg_viral_score': sum(e.viral_score for e in events) / len(events) if events else 0,
            'platform_distribution': Counter(
                signal.platform 
                for event in events 
                for signal in event.platform_signals
            ),
            'top_venues': Counter(event.venue_name for event in events).most_common(10),
            'viral_indicators': Counter(
                indicator 
                for event in events 
                for signal in event.platform_signals 
                for indicator in signal.viral_indicators
            ).most_common(10),
            'trending_hashtags': Counter(
                hashtag 
                for event in events 
                for hashtag in event.trending_hashtags
            ).most_common(20)
        }
        
        return analysis

def main():
    """Example usage of ViralEventDiscoveryEngine."""
    
    engine = ViralEventDiscoveryEngine()
    
    # Discover viral events
    viral_events = engine.discover_viral_events(
        days_back=3,
        min_viral_score=0.5,
        max_events=20
    )
    
    print(f"Found {len(viral_events)} viral events:")
    for i, event in enumerate(viral_events[:5], 1):
        print(f"\n{i}. {event.title}")
        print(f"   Venue: {event.venue_name}")
        print(f"   Viral Score: {event.viral_score:.3f}")
        print(f"   Confidence: {event.confidence_score:.3f}")
        print(f"   Predicted Attendance: {event.predicted_attendance}")
        print(f"   Platforms: {', '.join(set(s.platform for s in event.platform_signals))}")
        print(f"   Top Hashtags: {', '.join(event.trending_hashtags[:5])}")
    
    # Analyze patterns
    analysis = engine.analyze_viral_patterns(viral_events)
    print(f"\n--- Viral Event Analysis ---")
    print(f"Platform Distribution: {dict(analysis['platform_distribution'])}")
    print(f"Top Viral Indicators: {analysis['viral_indicators'][:5]}")

if __name__ == "__main__":
    main()