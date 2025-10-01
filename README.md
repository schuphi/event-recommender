# Copenhagen Event Recommender

Event recommendation system for Copenhagen nightlife, featuring real venue data and ML-powered suggestions.

## Tech Stack

### Backend
- **API**: FastAPI with Pydantic models
- **Database**: DuckDB with H3 geo-indexing
- **ML**: Sentence Transformers, scikit-learn, PyTorch
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: BeautifulSoup, Requests
- **Task Scheduling**: APScheduler

### Frontend
- **Framework**: React with Vite
- **UI Components**: Radix UI primitives
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query
- **Forms**: React Hook Form with Zod validation

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Deployment**: Railway.app ready
- **Testing**: Pytest with async support
- **Code Quality**: Black, Ruff, MyPy

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Optional: Eventbrite API token

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python start_server.py
```

### Frontend Setup
```bash
# Install and run frontend
cd frontend
npm install
npm run dev
```

### Access Points
- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## Features

### Current
- Real Copenhagen venue events
- Event search and filtering
- Basic recommendations
- Web scraping pipeline for venue websites
- RESTful API with CORS support


## Project Structure

```
event-recommender/
├── backend/                 # FastAPI application
│   └── app/                 # Core API code
├── frontend/                # React frontend
│   └── src/                 # Source code
├── ml/                      # Machine learning models
├── tests/                   # Test suite
├── database/                # Database management
├── data-collection/         # Event scraping pipeline
├── scripts/                 # Utility scripts
│   ├── scrapers/            # Data scraper runners
│   ├── deployment/          # Deployment scripts
│   └── testing/             # Test runners
├── config/                  # Configuration files
│   ├── docker/              # Docker configurations
│   ├── railway/             # Railway deployment
│   └── nginx/               # Nginx configuration
├── data/                    # Data storage
│   ├── events/              # Event database files
│   ├── vectors/             # Vector embeddings
│   └── cache/               # Cached data
└── docs/                    # Documentation
    ├── api/                 # API documentation
    ├── deployment/          # Deployment guides
    └── development/         # Development guides
```

## Machine Learning Pipeline

ML components:

- **Content-Based**: Uses sentence-transformers for event similarity
- **Feature Engineering**: H3 geo-indexing, text embeddings
- **Vector Search**: FAISS for semantic similarity


## Data Collection

The system scrapes real event data from:
- Copenhagen venue websites
- Eventbrite API integration
- RSS/JSON feeds from venues

Scrapers run on a schedule to keep event data current.

## Testing

```bash
# Run all tests
python scripts/testing/run_tests.py all

# Run specific categories
python scripts/testing/run_tests.py api
python scripts/testing/run_tests.py database
```

## Deployment

### Docker
```bash
docker compose -f config/docker/docker-compose.yml up -d
```

### Railway.app
The project includes Railway configuration files for easy deployment.

### Manual
```bash
# Start backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Start frontend
cd frontend && npm run build && npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch  
3. Run tests: `python run_tests.py all`
4. Submit pull request

## License

MIT License
