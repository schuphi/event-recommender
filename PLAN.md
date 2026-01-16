# Event Recommender Refactor Plan

## Executive Summary

This plan transforms the event recommender from a music-focused, barely-working system into a topic-filterable, ML-powered recommendation engine. We keep the valuable ML infrastructure (content-based + collaborative filtering) while simplifying the API and adding proper topic categorization.

---

## Phase 1: Foundation - CLAUDE.md & Repo Structure

### 1.1 Create CLAUDE.md

A `CLAUDE.md` file provides persistent context for AI assistants working on this codebase. This is highly valuable for:
- Maintaining consistency across sessions
- Documenting architectural decisions
- Providing quick-start context
- Defining coding standards

**Contents:**
```
- Project overview and purpose
- Architecture diagram (text-based)
- Directory structure with explanations
- ML pipeline overview
- Key abstractions and patterns
- Development workflow
- Testing approach
- Common commands
- Known limitations / tech debt
```

### 1.2 Repo Structure Cleanup

**Current issues:**
- Root-level test files (`test_connection.py`, `test_network.py`, `debug_env.py`)
- Multiple requirements files with overlap
- Inconsistent module organization

**Proposed structure:**
```
event-recommender/
├── CLAUDE.md                    # AI assistant context
├── README.md                    # Human documentation
├── pyproject.toml              # Single source of truth for deps
├── backend/
│   ├── app/
│   │   ├── api/                # Renamed from routers/
│   │   │   ├── events.py       # Event endpoints
│   │   │   ├── recommendations.py
│   │   │   └── health.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── dependencies.py
│   │   ├── models/             # Pydantic models
│   │   ├── services/           # Business logic
│   │   └── main.py
│   └── tests/
├── ml/
│   ├── models/                 # ML model implementations
│   ├── training/               # Training scripts
│   ├── embeddings/             # Embedding utilities
│   └── classifiers/            # NEW: Topic classification
├── data_collection/            # Renamed, snake_case
│   ├── scrapers/
│   ├── classifiers/            # Auto-topic classification
│   └── pipelines/
├── database/
│   ├── schema.sql
│   └── migrations/
├── frontend/
│   └── src/
├── scripts/
│   └── dev/                    # Development utilities
└── tests/                      # Integration tests
```

---

## Phase 2: Topic Categorization System

### 2.1 Define Topic Taxonomy

**Primary Topics (mutually exclusive):**
```python
TOPICS = {
    "tech": "Technology meetups, hackathons, conferences, startup events, OpenAI, robotics, AI, researchers",
    "nightlife": "Club nights, DJ sets, bar events, late-night parties",
    "music": "Concerts, live performances, festivals, acoustic sessions",
    "sports": "Matches, fitness classes, outdoor activities, tournaments, basketball"
}


### 2.2 Database Schema Changes

```sql
-- Add to events table
ALTER TABLE events ADD COLUMN topic VARCHAR(50) NOT NULL DEFAULT 'social';
ALTER TABLE events ADD COLUMN tags TEXT[] DEFAULT '{}';
ALTER TABLE events ADD COLUMN is_free BOOLEAN DEFAULT FALSE;

-- Create topic index
CREATE INDEX idx_events_topic ON events(topic);

-- Remove unused columns
ALTER TABLE events DROP COLUMN IF EXISTS embedding;
ALTER TABLE events DROP COLUMN IF EXISTS content_features;
```

### 2.3 Topic Classification Pipeline

**Three-tier approach:**

1. **Rule-based (fast, cheap):**
   ```python
   KEYWORD_RULES = {
       "tech": ["hackathon", "startup", "developer", "coding", "tech", "AI", "data"],
       "nightlife": ["club", "DJ", "dance", "party", "rave", "techno night"],
       "music": ["concert", "live music", "band", "orchestra", "acoustic"],
       # ...
   }
   ```

2. **Embedding-based (medium, local):**
   - Use sentence-transformers to embed event title+description
   - Compare against topic prototype embeddings
   - Assign topic with highest cosine similarity

3. **LLM fallback (expensive, accurate):**
   - For ambiguous cases (similarity < 0.7 for all topics)
   - Single Claude API call per event: "Classify this event: {title} - {description}"
   - Cache results

**Implementation:**
```python
class TopicClassifier:
    def classify(self, title: str, description: str) -> tuple[str, float]:
        # Try rule-based first
        topic, confidence = self.rule_based_classify(title, description)
        if confidence > 0.8:
            return topic, confidence

        # Try embedding-based
        topic, confidence = self.embedding_classify(title, description)
        if confidence > 0.7:
            return topic, confidence

        # LLM fallback
        return self.llm_classify(title, description)
```

---

## Phase 3: Simplified API Design

### 3.1 New Endpoint Structure

**Remove:** Auth endpoints, admin endpoints, cold-start, model feedback
**Keep:** Events, recommendations (simplified), health

```python
# Events
GET  /events                    # List with filters
GET  /events/{id}               # Single event
GET  /events/topics             # Topic list with counts

# Recommendations
GET  /recommendations           # Personalized feed (session-based)
GET  /recommendations/similar/{event_id}  # Similar events

# Interactions (session-based, no auth)
POST /events/{id}/like          # Toggle like
POST /events/{id}/save          # Toggle save
GET  /saved                     # User's saved events

# System
GET  /health                    # Health check
```

### 3.2 Query Parameters

```python
# GET /events
class EventFilters:
    topics: list[str] = []       # Filter by topics
    tags: list[str] = []         # Filter by tags
    date_from: datetime = now()  # Default: upcoming only
    date_to: datetime = None     # Optional end date
    price_max: float = None      # Max price filter
    is_free: bool = None         # Free events only
    near_lat: float = None       # Location-based
    near_lon: float = None
    radius_km: float = 10        # Default 10km
    limit: int = 20
    offset: int = 0
```

### 3.3 Session-Based User Tracking

**No auth required.** Use session ID for personalization:

```python
# Session management via cookie or X-Session-ID header
async def get_session(request: Request) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id
```

**Interactions stored per session:**
- Likes (binary signal)
- Saves (for later)
- Views (implicit, optional)

---

## Phase 4: ML Pipeline Refinement

### 4.1 Keep and Improve

**Content-Based Recommender** (`ml/models/content_based.py`):
- Already well-implemented
- Add topic-aware features
- Wire up properly in recommendation_service

**Collaborative Filtering** (`ml/models/collaborative_filtering.py`):
- BPR model is solid
- Needs: periodic retraining, cold-start handling
- Add: topic co-occurrence patterns

### 4.2 Remove

- `langchain_recommender.py` - Over-engineered, LLM costs
- `hybrid_ranker.py` - Partially stubbed, adds complexity
- Heavy sentence-transformer dependency for recommendations (keep for classification)

### 4.3 New Recommendation Flow

```
┌─────────────────────────────────────────────────────┐
│                  Request: /recommendations          │
│  session_id, filters (topics, date, price, etc.)   │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│           1. Filter Candidates (SQL)                │
│  - Apply topic/date/price/location filters          │
│  - Return ~100 candidates                           │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│           2. Check User History                     │
│  - Has session interacted before?                   │
│  - If yes → personalized scoring                    │
│  - If no → popularity-based with diversity          │
└─────────────────────┬───────────────────────────────┘
                      ▼
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────────┐     ┌───────────────────┐
│  New User Path    │     │  Returning User   │
│                   │     │                   │
│ - Popularity      │     │ - Content-based   │
│ - Topic diversity │     │   (liked → sim)   │
│ - Time decay      │     │ - Collab filter   │
│ - Random boost    │     │   (if trained)    │
└─────────┬─────────┘     └─────────┬─────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              3. Score & Rank                        │
│  - Combine model scores                             │
│  - Apply diversity penalty (don't repeat topics)    │
│  - Boost upcoming (next 48h)                        │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              4. Return Top N                        │
│  - Include explanation (why recommended)            │
│  - Cache for session                                │
└─────────────────────────────────────────────────────┘
```

### 4.4 Scoring Formula

```python
def compute_recommendation_score(event, user_context, models):
    score = 0.0

    # Base: popularity (always available)
    score += 0.2 * event.popularity_score

    # Content similarity (if user has likes)
    if user_context.liked_events:
        content_sim = models.content.similarity_to_liked(event, user_context.liked_events)
        score += 0.4 * content_sim

    # Collaborative score (if model trained)
    if models.collaborative.is_trained:
        collab_score = models.collaborative.predict(user_context.session_id, event.id)
        score += 0.3 * collab_score

    # Topic preference (learned from interactions)
    topic_affinity = user_context.topic_preferences.get(event.topic, 0.5)
    score += 0.1 * topic_affinity

    # Time boost (upcoming events)
    hours_until = (event.datetime - now()).total_seconds() / 3600
    if hours_until < 48:
        score *= 1.2

    return score
```

### 4.5 Model Training Schedule

```python
# Automated retraining (APScheduler already in deps)
TRAINING_CONFIG = {
    "collaborative": {
        "min_interactions": 100,      # Don't train until enough data
        "retrain_interval": "daily",  # Retrain daily
        "incremental": True           # Incremental updates between full retrains
    },
    "content": {
        "retrain_on": "new_events",   # Retrain when new events added
        "cache_embeddings": True       # Cache for performance
    }
}
```

---

## Phase 5: Data Collection Improvements

### 5.1 Expand Sources

**Current:** Eventbrite, venue scrapers
**Add:**
- Meetup.com API (especially for Tech)
- Facebook Events (via scraping or official API)
- TicketMaster (Sports, Music)
- Manual submission endpoint
- Luma (tech)


### 5.2 Ingestion Pipeline

```python
class EventIngestionPipeline:
    def ingest(self, raw_event: dict, source: str) -> Event:
        # 1. Normalize to common schema
        event = self.normalize(raw_event, source)

        # 2. Classify topic
        event.topic, confidence = self.classifier.classify(
            event.title,
            event.description
        )

        # 3. Extract tags
        event.tags = self.extract_tags(event)

        # 4. Deduplicate
        existing = self.find_duplicate(event)
        if existing:
            return self.merge_events(existing, event)

        # 5. Geocode if needed
        if not event.venue_lat:
            event.venue_lat, event.venue_lon = self.geocode(event.venue_address)

        # 6. Store
        return self.db.insert_event(event)
```

---

## Phase 6: Frontend Updates

### 6.1 Topic-First Navigation

```
┌─────────────────────────────────────────────────────┐
│  ┌─────┐ ┌─────────┐ ┌───────┐ ┌───────┐ ┌──────┐  │
│  │ All │ │Nightlife│ │ Tech  │ │ Music │ │Sports│  │
│  └──●──┘ └─────────┘ └───────┘ └───────┘ └──────┘  │
│                                                     │
│  ┌─────────────────────────────────────────────────┐│
│  │ 🎵 Techno Night at Culture Box                  ││
│  │ Nightlife • Tonight 23:00 • 200 DKK             ││
│  │ [♡ Like] [⊕ Save]                               ││
│  └─────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────┐│
│  │ 💻 Copenhagen JS Meetup                         ││
│  │ Tech • Tomorrow 18:00 • Free                    ││
│  │ [♡ Like] [⊕ Save]                               ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### 6.2 Component Changes

- `TopicChips.tsx` - Horizontal scrollable topic filter
- `EventCard.tsx` - Add topic badge, update styling
- `FilterPanel.tsx` - Replace genres with topics
- `SavedEvents.tsx` - New page for saved events

---

## Phase 7: Implementation Order

### Sprint 1: Foundation (Est. effort: Medium)
1. [ ] Create CLAUDE.md
2. [ ] Clean up repo structure
3. [ ] Consolidate requirements into pyproject.toml
4. [ ] Database migration: add topic/tags columns

### Sprint 2: Topic System (Est. effort: Medium)
5. [ ] Implement TopicClassifier (rule-based + embedding)
6. [ ] Create classification pipeline
7. [ ] Backfill existing events with topics
8. [ ] Add topic filtering to API

### Sprint 3: API Simplification (Est. effort: Medium)
9. [ ] Create new simplified endpoints
10. [ ] Implement session-based interactions
11. [ ] Remove auth system
12. [ ] Update API documentation

### Sprint 4: ML Pipeline (Est. effort: High)
13. [ ] Wire up content-based recommender properly
14. [ ] Add topic-aware features
15. [ ] Implement new scoring formula
16. [ ] Add training scheduler

### Sprint 5: Frontend (Est. effort: Medium)
17. [ ] Topic navigation UI
18. [ ] Updated event cards
19. [ ] Saved events page
20. [ ] Filter panel updates

### Sprint 6: Data & Polish (Est. effort: Low)
21. [ ] Add new data sources
22. [ ] Improve scraper reliability
23. [ ] Performance optimization
24. [ ] End-to-end testing

---

## Files to Create/Modify

### New Files:
- `CLAUDE.md` - AI context documentation
- `ml/classifiers/topic_classifier.py` - Topic classification
- `backend/app/api/recommendations.py` - New recommendation endpoint
- `database/migrations/002_add_topics.sql` - Schema migration

### Files to Modify:
- `backend/app/main.py` - Simplified routes
- `backend/app/services/recommendation_service.py` - New flow
- `database/schema.sql` - Add topic/tags
- `frontend/src/components/EventCard.tsx` - Topic badges
- `frontend/src/components/EventFeed.tsx` - Topic filters

### Files to Remove:
- `backend/app/routers/auth.py` - No auth needed
- `backend/app/services/langchain_recommender.py` - Over-engineered
- `test_connection.py`, `test_network.py`, `debug_env.py` - Root clutter

---

## Success Metrics

1. **Recommendation Quality:**
   - Click-through rate on recommendations > 15%
   - Users interact with 3+ different topics

2. **Performance:**
   - API response time < 200ms (p95)
   - Classification accuracy > 85%

3. **Simplicity:**
   - API surface reduced from 26 endpoints to 8
   - Dependencies reduced by 30%

---

## Open Questions

1. **LLM for classification:** Use Claude API or local model?
   - Local (e.g., DistilBERT): Free, faster, less accurate

2. **Embedding model:** Keep sentence-transformers?
   - Pro: Good for semantic similarity
   - Con: Heavy dependency, slow startup

3. **Frontend framework:** Keep React or simplify?
   - Current React setup is fine

---

## Appendix: CLAUDE.md Draft

```markdown
# CLAUDE.md - Event Recommender

## Quick Start
- Backend: `uvicorn backend.app.main:app --reload`
- Frontend: `cd frontend && npm run dev`
- Database: DuckDB at `data/events/events.duckdb`

## Architecture
Event recommender for Copenhagen with topic-based filtering and ML-powered recommendations.

### Core Flow
1. Events scraped from Eventbrite, venues → classified by topic
2. Users browse by topic, interact (like/save)
3. Recommendations generated via content-based + collaborative filtering
4. No auth - sessions tracked via cookie/header

### Key Directories
- `backend/app/` - FastAPI application
- `ml/models/` - Recommendation models (content-based, collaborative)
- `ml/classifiers/` - Topic classification
- `data_collection/` - Scrapers
- `frontend/src/` - React UI

### ML Pipeline
- Content-based: Sentence embeddings + feature similarity
- Collaborative: BPR matrix factorization (PyTorch)
- Classification: Rule-based → embedding → LLM fallback

### Database
- DuckDB (dev) / PostgreSQL (prod)
- Key tables: events, venues, interactions
- H3 geo-indexing for location queries

### Coding Standards
- Python: Black, Ruff, type hints
- Frontend: TypeScript strict, Tailwind
- Tests: pytest with async support

### Common Tasks
- Add new event source: `data_collection/scrapers/`
- Modify recommendation logic: `backend/app/services/recommendation_service.py`
- Update ML models: `ml/models/`
- API changes: `backend/app/api/`
```
