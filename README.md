# Copenhagen Event Recommender

A hybrid ML-powered event recommendation system for Copenhagen nightlife and music events, demonstrating modern ML engineering and product development skills.

## Features

- **Hybrid Recommendations**: Combines content-based (transformer embeddings) + collaborative filtering (PyTorch BPR)
- **Geo-aware**: H3 grid indexing for efficient geographic filtering and clustering
- **Real-time Data**: Daily scraping from official APIs and social sources
- **Interactive UX**: React frontend with event feed, filters, and user interactions
- **Cold Start Handling**: Quick preference collection + content-based fallback

## Tech Stack

**ML**: PyTorch, sentence-transformers, implicit collaborative filtering  
**Backend**: FastAPI, DuckDB, H3 geo-indexing  
**Frontend**: React (Next.js), TypeScript  
**Data**: Eventbrite, Billetto, Meetup, Instagram, TikTok APIs  
**Deployment**: Docker, single-container architecture  

## Project Structure

```
├── data-collection/          # Scrapers and data pipeline
├── ml/                      # ML models and training
├── backend/                 # FastAPI recommendation service  
├── frontend/                # React event discovery UI
├── database/                # Schema and migrations
├── evaluation/              # Metrics and synthetic data
└── docker/                  # Deployment configuration
```

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt
npm install

# Initialize database
python database/init_db.py

# Start services
docker-compose up
```

## Data Sources

- **Official APIs**: Eventbrite, Billetto, Meetup, Songkick, Ticketmaster
- **Social Media**: Instagram, TikTok event discovery
- **Venue Feeds**: RSS/ICS from Copenhagen venues
- **Enrichment**: Spotify/Last.fm artist metadata

## Models

1. **Content-based**: Sentence transformer embeddings + venue/artist features
2. **Collaborative Filtering**: BPR matrix factorization on user interactions  
3. **Hybrid Ranker**: Weighted combination (0.6 content + 0.4 CF) + neural reranker

## Evaluation

- Offline metrics: Recall@10, NDCG@10
- A/B comparisons: content-only vs CF-only vs hybrid
- Cold start performance analysis

---

*This project demonstrates end-to-end ML system design for portfolio/job applications.*