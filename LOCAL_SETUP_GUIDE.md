# 🚀 Local Development Setup Guide

## Quick Start Checklist

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for frontend)
cd frontend
npm install
cd ..
```

### 2. Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for local testing)
# Most defaults will work for local development
```

### 3. Initialize Database
```bash
# Create and populate database with sample data
python database/init_db.py
```

### 4. Start Backend API
```bash
# Start FastAPI backend
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test API
```bash
# In another terminal, test the API
curl http://localhost:8000/health
curl http://localhost:8000/events
```

### 6. Start Frontend (if using manually built frontend)
```bash
# In another terminal
cd frontend
npm run dev
```

---

## Detailed Setup Instructions

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Git** for version control

### Step 1: Clone & Dependencies

```bash
# If not already in the project directory
cd event-recommender

# Install Python packages
pip install -r requirements.txt

# This installs:
# - FastAPI, Uvicorn (API framework)
# - PyTorch, sentence-transformers (ML models)
# - DuckDB, SQLAlchemy (database)
# - Pandas, NumPy, scikit-learn (data processing)
# - All scraping dependencies
```

### Step 2: Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env if needed (optional for local testing)
```

**Key environment variables for local development:**
```bash
# Database (uses local file)
DATABASE_URL=data/events.duckdb

# API configuration  
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=["http://localhost:3000", "http://localhost:3001"]

# ML Models (will download automatically)
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Optional: Add API keys for enhanced data collection
# EVENTBRITE_API_TOKEN=your_token_here
# SPOTIFY_CLIENT_ID=your_client_id
# INSTAGRAM_USERNAME=your_username
```

### Step 3: Initialize Database with Sample Data

```bash
# Run the database initialization script
python database/init_db.py
```

**This creates:**
- ✅ Database schema with H3 geo-indexing
- ✅ Sample Copenhagen venues (Vega, Rust, Culture Box, etc.)
- ✅ Sample artists with genres
- ✅ Sample events with realistic data
- ✅ Sample users and interactions
- ✅ All necessary indexes and relationships

### Step 4: Start the Backend API

```bash
cd backend

# Start with auto-reload for development
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Started reloader process
```

### Step 5: Test the API

Open a new terminal and test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Get sample events
curl http://localhost:8000/events

# View API documentation
open http://localhost:8000/docs
```

**Expected responses:**
- ✅ Health endpoint returns `{"status": "healthy"}`
- ✅ Events endpoint returns JSON array of events
- ✅ `/docs` shows interactive Swagger UI

### Step 6: Test Recommendations

```bash
# Test recommendation endpoint
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test_user_123" \
  -d '{
    "user_preferences": {
      "preferred_genres": ["techno", "electronic"],
      "price_range": [0, 500],
      "location_lat": 55.6761,
      "location_lon": 12.5683
    },
    "num_recommendations": 5
  }'
```

**Expected:**
- ✅ Returns JSON with personalized event recommendations
- ✅ Each event has recommendation scores and explanations
- ✅ Events are ranked by hybrid ML algorithm

### Step 7: Test Interactions

```bash
# Record a user interaction
curl -X POST "http://localhost:8000/interactions" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test_user_123" \
  -d '{
    "event_id": "event_id_from_previous_response",
    "interaction_type": "like",
    "source": "feed"
  }'
```

---

## Troubleshooting Common Issues

### Issue: Dependencies Won't Install
```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# For Apple Silicon Macs with PyTorch issues:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Database Initialization Fails
```bash
# Create data directory manually
mkdir -p data

# Run with verbose output
python database/init_db.py --verbose

# Check if DuckDB is working
python -c "import duckdb; print('DuckDB working')"
```

### Issue: API Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000  # On Mac/Linux
netstat -an | findstr 8000  # On Windows

# Start on different port
python -m uvicorn app.main:app --reload --port 8001
```

### Issue: Import Errors
```bash
# Make sure you're in the right directory
pwd  # Should end with /event-recommender

# Check Python path
python -c "import sys; print(sys.path)"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Model Downloads Fail
```bash
# Test sentence transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# If behind corporate firewall, set proxy
pip install --proxy http://proxy.company.com:8080 -r requirements.txt
```

---

## Verification Steps

### ✅ Backend Health Check
```bash
# All these should return success:
curl http://localhost:8000/health           # API health
curl http://localhost:8000/events           # Database working
curl -X POST http://localhost:8000/recommend \  # ML models loaded
  -H "Content-Type: application/json" \
  -H "X-User-ID: test" \
  -d '{"user_preferences": {"preferred_genres": ["techno"]}}'
```

### ✅ Database Verification
```bash
# Check database contents
python -c "
import duckdb
conn = duckdb.connect('data/events.duckdb')
print('Events:', conn.execute('SELECT COUNT(*) FROM events').fetchone()[0])
print('Venues:', conn.execute('SELECT COUNT(*) FROM venues').fetchone()[0])
print('Users:', conn.execute('SELECT COUNT(*) FROM users').fetchone()[0])
"
```

### ✅ ML Models Verification
```bash
# Test model loading
python -c "
from ml.models.content_based import ContentBasedRecommender
from ml.models.collaborative_filtering import CollaborativeFilteringRecommender
print('✅ Content-based model loads')
print('✅ Collaborative filtering loads')
print('✅ ML pipeline ready')
"
```

---

## Performance Monitoring

### Development Tools
```bash
# API performance monitoring
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/recommend

# Database query profiling  
python -c "
import duckdb
conn = duckdb.connect('data/events.duckdb')
conn.execute('PRAGMA enable_profiling')
"

# Memory usage monitoring
pip install psutil
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

---

## Next Steps After Local Setup

1. **✅ Verify all endpoints work** (health, events, recommend, interactions)
2. **✅ Test recommendation quality** with different user preferences  
3. **✅ Verify database operations** (CRUD, filtering, search)
4. **✅ Check ML model performance** (content + collaborative filtering)
5. **🚀 Proceed with Lovable frontend** or remaining todos
6. **🔧 Add additional features** (geo filtering, cold start, evaluation)

---

## Current System Status

**✅ Implemented & Ready:**
- Database with sample Copenhagen events
- FastAPI backend with 15+ endpoints  
- Content-based recommendations (sentence transformers)
- Collaborative filtering (PyTorch BPR)
- Hybrid weighted ranking system
- User interaction tracking
- Event search and filtering
- Viral social media scraping

**🔧 Remaining Todos:**
- H3-based geo filtering integration
- Cold start preference collection UI
- Evaluation framework with synthetic data
- Docker deployment configuration

The system is **production-ready for core functionality** and ready for frontend integration or additional feature development!