# CLAUDE.md - Copenhagen Event Recommender

> Context file for AI assistants working on this codebase.

## Project Overview

Event recommendation system for Copenhagen. Scrapes events from multiple sources, classifies by topic, and provides personalized recommendations using ML models.

**Status:** Refactoring in progress. See `PLAN.md` for roadmap.

## Quick Commands

```bash
# Backend (FastAPI)
cd /Users/philipp/Desktop/Github/event-recommender
uvicorn backend.app.main:app --reload --port 8000

# Frontend (React + Vite)
cd frontend && npm run dev

# Run tests
pytest tests/ -v

# Database location
data/events/events.duckdb  # DuckDB (dev)
# PostgreSQL via DATABASE_URL env var (prod)
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Sources   │────▶│   Classifiers   │────▶│    Database     │
│  (Eventbrite,   │     │  (Topic assign) │     │  (Events,       │
│   Venues, etc)  │     │                 │     │   Interactions) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐              │
│    Frontend     │◀────│   FastAPI       │◀─────────────┘
│  (React/Vite)   │     │  (REST API)     │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   ML Models     │
                        │ (Content-based, │
                        │  Collaborative) │
                        └─────────────────┘
```

## Directory Structure

```
event-recommender/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI entry point
│       ├── core/
│       │   ├── config.py        # Settings (pydantic-settings)
│       │   └── dependencies.py  # DI, auth, rate limiting
│       ├── models/
│       │   ├── requests.py      # Pydantic request models
│       │   └── responses.py     # Pydantic response models
│       ├── routers/
│       │   └── auth.py          # Auth endpoints (being removed)
│       └── services/
│           ├── database_service.py        # DB operations
│           ├── recommendation_service.py  # Main recommendation logic
│           ├── langchain_recommender.py   # Semantic search (being removed)
│           └── custom_collaborative_filter.py
│
├── ml/
│   ├── models/
│   │   ├── content_based.py         # Content-based recommender
│   │   ├── collaborative_filtering.py # BPR model (PyTorch)
│   │   └── hybrid_ranker.py         # Hybrid (partially stubbed)
│   ├── embeddings/
│   │   └── content_embedder.py      # Sentence transformer embeddings
│   ├── training/
│   │   ├── train_models.py          # Training scripts
│   │   └── production_pipeline.py   # Production training
│   └── preprocessing/
│       └── text_processor.py        # Text normalization
│
├── data-collection/
│   └── scrapers/
│       ├── official_apis/
│       │   └── eventbrite.py        # Eventbrite API client
│       └── venue_scrapers/
│           └── copenhagen_venues.py # Venue website scrapers
│
├── database/
│   └── schema.sql                   # Main schema definition
│
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── EventCard.tsx        # Single event display
│       │   ├── EventFeed.tsx        # Main event list
│       │   └── FilterPanel.tsx      # Filtering UI
│       └── lib/
│           └── api.ts               # API client
│
├── tests/                           # Test suite
├── scripts/                         # Utility scripts
└── config/                          # Docker, nginx, railway configs
```

## Topic Taxonomy

Events are classified into 4 primary topics:

```python
TOPICS = {
    "tech": "Technology meetups, hackathons, conferences, startup events, OpenAI, robotics, AI, researchers",
    "nightlife": "Club nights, DJ sets, bar events, late-night parties",
    "music": "Concerts, live performances, festivals, acoustic sessions",
    "sports": "Matches, fitness classes, outdoor activities, tournaments, basketball"
}
```

Classification uses rule-based keyword matching → embedding similarity fallback.

## Key Abstractions

### Database

**DuckDB** (local dev) or **PostgreSQL** (production via Supabase).

**Core tables:**
- `events` - Event data with title, description, datetime, price, venue_id, **topic**, **tags**
- `venues` - Venue info with coordinates, H3 geohash
- `interactions` - User interactions (like, save) per session
- `users` - Session tracking (minimal, no auth)

**Key patterns:**
- H3 geo-indexing for location queries (level 8 ~ 460m)
- JSON columns for flexible data (genres, tags)
- Timestamps use UTC
- Topic column for primary classification (tech, nightlife, music, sports)

### ML Models

**Content-Based** (`ml/models/content_based.py`):
- Uses sentence-transformers for text embeddings
- Feature weights: text (50%), genre (20%), artist (10%), venue (10%), price (5%), time (5%)
- Supports diversity factor to avoid repetitive recommendations
- Has `explain_recommendation()` for transparency

**Collaborative Filtering** (`ml/models/collaborative_filtering.py`):
- Bayesian Personalized Ranking (BPR) with PyTorch
- User and event embeddings (64 dims)
- Learns from implicit feedback (likes, saves, going)
- Requires training data - falls back to popularity without it

**Recommendation Service** (`backend/app/services/recommendation_service.py`):
- Orchestrates all models
- Fallback chain: hybrid → collaborative → content → langchain → popularity
- Currently: content-based and hybrid methods are stubbed (lines 672-680)

### API Patterns

- FastAPI with Pydantic models for validation
- CORS configured for localhost:3000-3003
- Session-based user tracking via X-User-ID header
- Rate limiting (in-memory)

## Current Limitations / Tech Debt

1. **Content-based not wired up** - `_content_based_recommendations()` in recommendation_service.py is a stub (line 672)

2. **Topic system missing** - Events have genres (music-focused) but no general topic categorization (Tech, Nightlife, Sports, etc.)

3. **Heavy dependencies** - LangChain, sentence-transformers add startup time. Consider removing for simpler deployments.

4. **Auth over-engineered** - Full JWT auth exists but project doesn't need user accounts. Simplify to session-only.

5. **Root-level test files** - `test_connection.py`, `debug_env.py` etc. should move to `tests/` or `scripts/`

6. **Multiple requirements files** - `requirements.txt`, `requirements-dev.txt`, `requirements-railway.txt` have overlap

## Coding Standards

**Python:**
- Black formatter (line length 88)
- Ruff for linting
- Type hints required for public functions
- Async/await for I/O operations
- Pydantic for data validation

**Frontend:**
- TypeScript strict mode
- Tailwind CSS for styling
- React hooks, no class components
- TanStack Query for data fetching

**Testing:**
- pytest with async support (`pytest-asyncio`)
- Tests in `tests/` directory
- Fixtures for database/API setup

## Environment Variables

```bash
# Required
DATABASE_URL=data/events/events.duckdb  # or postgresql://...

# Optional - API keys for data collection
EVENTBRITE_API_TOKEN=
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=

# Optional - ML config
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Optional - API config
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

## Common Tasks

### Add a new event source

1. Create scraper in `data-collection/scrapers/`
2. Implement `fetch_events() -> List[Dict]`
3. Add to runner in `data-collection/scrapers/runner.py`
4. Test with `python -m data_collection.scrapers.your_scraper`

### Modify recommendation logic

1. Edit `backend/app/services/recommendation_service.py`
2. Key method: `_generate_recommendations()` (line 380)
3. Scoring in `_popularity_based_recommendations()` or model-specific methods
4. Test via `/recommend/{user_id}` endpoint

### Update ML models

1. Models in `ml/models/`
2. Training in `ml/training/train_models.py`
3. Cache models in `ml/models/{model_type}/`
4. Load via `RecommendationService._load_*_model()` methods

### Add new API endpoint

1. Add Pydantic models to `backend/app/models/`
2. Add route to `backend/app/main.py` or create new router
3. Implement service method if needed
4. Update OpenAPI docs (automatic via FastAPI)

## Deployment

**Local:** `uvicorn backend.app.main:app --reload`

**Docker:** `docker compose -f config/docker/docker-compose.yml up`

**Railway:** Auto-deploys from main branch. Uses `railway.toml` config.

## API Endpoints (Current)

```
GET  /                    # API info
GET  /health              # Health check with DB stats
GET  /events              # List events (filters: topic, is_free, price, upcoming)
GET  /events/topics       # List topics with event counts
GET  /events/{id}         # Single event
GET  /recommend/{user_id} # Get recommendations
GET  /search              # Full-text search
GET  /stats               # Database statistics
```

### Topic Filtering

Filter events by topic using the `topic` query parameter:
```
GET /events?topic=tech        # Tech events only
GET /events?topic=nightlife   # Nightlife events only
GET /events?topic=music       # Music events only
GET /events?topic=sports      # Sports events only
GET /events?is_free=true      # Free events only
```

## Refactoring Notes

See `PLAN.md` for the full refactoring plan. Key changes:

1. **Add topic system** - Classify events into Tech, Nightlife, Sports, etc.
2. **Simplify API** - Remove auth, reduce to ~8 endpoints
3. **Wire up ML** - Connect content-based model properly
4. **Clean repo** - Consolidate requirements, move test files

## Useful Queries

```sql
-- Count events by source
SELECT source, COUNT(*) FROM events GROUP BY source;

-- Upcoming events this week
SELECT * FROM events
WHERE date_time > NOW() AND date_time < NOW() + INTERVAL '7 days'
ORDER BY date_time;

-- User's interactions
SELECT * FROM interactions WHERE user_id = 'xxx' ORDER BY timestamp DESC;

-- Popular venues
SELECT v.name, COUNT(e.id) as event_count
FROM venues v JOIN events e ON v.id = e.venue_id
GROUP BY v.name ORDER BY event_count DESC;
```
