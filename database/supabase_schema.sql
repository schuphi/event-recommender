-- Copenhagen Event Recommender Database Schema for Supabase (PostgreSQL)
-- Migration from DuckDB to PostgreSQL with authentication

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE,
    preferences JSONB,
    location_lat REAL,
    location_lon REAL,
    h3_index VARCHAR(20)
);

-- Venues with H3 geo-indexing for fast spatial queries
CREATE TABLE venues (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    address TEXT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    h3_index VARCHAR(20) NOT NULL,
    neighborhood VARCHAR(100),
    venue_type VARCHAR(50),
    capacity INTEGER,
    website TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Artists with genre and music platform metadata
CREATE TABLE artists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    genres JSONB,
    spotify_id VARCHAR(50),
    spotify_data JSONB,
    lastfm_data JSONB,
    popularity_score REAL DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Events - core entity with rich metadata
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    date_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_date_time TIMESTAMP WITH TIME ZONE,
    price_min REAL,
    price_max REAL,
    currency VARCHAR(3) DEFAULT 'DKK',
    
    -- Relationships
    venue_id UUID NOT NULL REFERENCES venues(id),
    artist_ids JSONB,
    
    -- Source and metadata
    source VARCHAR(50) NOT NULL,
    source_id VARCHAR(100),
    source_url TEXT,
    image_url TEXT,
    
    -- Computed fields
    h3_index VARCHAR(20),
    embedding BYTEA,
    content_features JSONB,
    popularity_score REAL DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active'
);

-- User interactions for collaborative filtering
CREATE TABLE interactions (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_id UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    interaction_type VARCHAR(20) NOT NULL,
    rating REAL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Context
    source VARCHAR(50),
    position INTEGER,
    
    UNIQUE(user_id, event_id, interaction_type)
);

-- Event-Artist many-to-many relationship
CREATE TABLE event_artists (
    event_id UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    is_headliner BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (event_id, artist_id)
);

-- Recommendation logs for evaluation and debugging
CREATE TABLE recommendation_logs (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100),
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Request context
    filters JSONB,
    location_lat REAL,
    location_lon REAL,
    
    -- Model outputs
    content_scores JSONB,
    cf_scores JSONB,
    hybrid_scores JSONB,
    recommended_events JSONB,
    
    -- Model metadata
    model_version VARCHAR(50),
    response_time_ms INTEGER
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_h3 ON users(h3_index);

CREATE INDEX idx_events_datetime ON events(date_time);
CREATE INDEX idx_events_venue ON events(venue_id);
CREATE INDEX idx_events_h3 ON events(h3_index);
CREATE INDEX idx_events_source ON events(source);
CREATE INDEX idx_events_status_datetime ON events(status, date_time);

CREATE INDEX idx_venues_h3 ON venues(h3_index);
CREATE INDEX idx_venues_neighborhood ON venues(neighborhood);

CREATE INDEX idx_interactions_user ON interactions(user_id);
CREATE INDEX idx_interactions_event ON interactions(event_id);
CREATE INDEX idx_interactions_type ON interactions(interaction_type);
CREATE INDEX idx_interactions_timestamp ON interactions(timestamp);

-- Views for common queries
CREATE VIEW active_events AS
SELECT * FROM events 
WHERE status = 'active' 
  AND date_time > NOW();

CREATE VIEW event_details AS
SELECT 
    e.id,
    e.title,
    e.description,
    e.date_time,
    e.price_min,
    e.price_max,
    e.status,
    e.venue_id,
    e.h3_index,
    e.source,
    e.created_at,
    e.updated_at,
    v.name as venue_name,
    v.neighborhood,
    v.lat as venue_lat,
    v.lon as venue_lon,
    COUNT(i.id) as interaction_count,
    AVG(CASE WHEN i.interaction_type = 'like' THEN 1.0 
             WHEN i.interaction_type = 'dislike' THEN 0.0 
             ELSE NULL END) as like_ratio
FROM events e
JOIN venues v ON e.venue_id = v.id
LEFT JOIN interactions i ON e.id = i.event_id
GROUP BY e.id, e.title, e.description, e.date_time, e.price_min, e.price_max, 
         e.status, e.venue_id, e.h3_index, e.source, e.created_at, e.updated_at, 
         v.name, v.neighborhood, v.lat, v.lon;

-- Row Level Security (RLS) for user data protection
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendation_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies (users can only access their own data)
CREATE POLICY "Users can view own profile" ON users
    FOR SELECT USING (auth.uid()::text = id::text);

CREATE POLICY "Users can update own profile" ON users
    FOR UPDATE USING (auth.uid()::text = id::text);

CREATE POLICY "Users can view own interactions" ON interactions
    FOR ALL USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can view own recommendation logs" ON recommendation_logs
    FOR ALL USING (auth.uid()::text = user_id::text);

-- Public access to events and venues (read-only)
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE venues ENABLE ROW LEVEL SECURITY;
ALTER TABLE artists ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access to events" ON events
    FOR SELECT USING (true);

CREATE POLICY "Public read access to venues" ON venues
    FOR SELECT USING (true);

CREATE POLICY "Public read access to artists" ON artists
    FOR SELECT USING (true);

