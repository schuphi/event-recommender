# Development Setup Guide

## Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Docker (optional, for containerized development)

## Quick Start

### Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the API
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Using Docker
```bash
docker compose -f config/docker/docker-compose.dev.yml up -d
```

## Project Structure

```
event-recommender/
├── backend/           # FastAPI application
├── frontend/          # React frontend
├── ml/               # Machine learning models
├── tests/            # Test suite
├── database/         # Database management
├── data-collection/  # Event data collection
├── scripts/          # Utility scripts
│   ├── scrapers/     # Data scraper runners
│   ├── deployment/   # Deployment scripts
│   └── testing/      # Test runners
├── config/           # Configuration files
│   ├── docker/       # Docker configurations
│   ├── railway/      # Railway deployment
│   └── nginx/        # Nginx configuration
├── data/             # Data storage
│   ├── events/       # Event database files
│   ├── vectors/      # Vector embeddings
│   └── cache/        # Cached data
└── docs/             # Documentation
    ├── api/          # API documentation
    ├── deployment/   # Deployment guides
    └── development/  # Development guides
```

## Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=data/events/events.duckdb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# External APIs (Optional)
EVENTBRITE_API_TOKEN=your_token
SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
```

## Running Tests

```bash
# All tests
python scripts/testing/run_tests.py all

# Specific test categories
python scripts/testing/run_tests.py api
python scripts/testing/run_tests.py ml
python scripts/testing/run_tests.py database
```

## Data Collection

```bash
# Run data scrapers
python scripts/scrapers/scraper_runner.py

# Enhanced scraping with all sources
python scripts/scrapers/enhanced_scraper_runner.py
```