-- Copenhagen Event Recommender Database Schema
-- Using DuckDB/SQLite compatible SQL with H3 geo-indexing

-- Users table for anonymous sessions + preferences
CREATE TABLE users (
    id TEXT PRIMARY KEY,  -- UUID for anonymous sessions
    name TEXT,           -- Optional display name
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSON,    -- Cold start preferences: genres, price_cap, radius
    location_lat REAL,   -- User's preferred location
    location_lon REAL,
    h3_index TEXT       -- H3 index for user location (level 8/9)
);

-- Venues with H3 geo-indexing for fast spatial queries
CREATE TABLE venues (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    h3_index TEXT NOT NULL,  -- H3 level 8 for ~460m resolution
    neighborhood TEXT,       -- Nørrebro, Vesterbro, etc.
    venue_type TEXT,        -- club, bar, concert_hall, outdoor
    capacity INTEGER,
    website TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Artists with genre and music platform metadata
CREATE TABLE artists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    genres JSON,            -- Array of genre strings
    spotify_id TEXT,
    spotify_data JSON,      -- Spotify artist metadata
    lastfm_data JSON,       -- Last.fm artist metadata  
    popularity_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events - core entity with rich metadata
CREATE TABLE events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    date_time TIMESTAMP NOT NULL,
    end_date_time TIMESTAMP,
    price_min REAL,
    price_max REAL,
    currency TEXT DEFAULT 'DKK',
    
    -- Relationships
    venue_id TEXT NOT NULL REFERENCES venues(id),
    artist_ids JSON,        -- Array of artist IDs
    
    -- Source and metadata
    source TEXT NOT NULL,   -- eventbrite, instagram, etc.
    source_id TEXT,        -- External ID from source
    source_url TEXT,
    image_url TEXT,
    
    -- Computed fields
    h3_index TEXT,         -- Inherited from venue
    embedding BLOB,        -- Sentence transformer embedding
    content_features JSON, -- Preprocessed features for ML
    popularity_score REAL DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    status TEXT DEFAULT 'active', -- active, cancelled, postponed
    
    FOREIGN KEY (venue_id) REFERENCES venues(id)
);

-- User interactions for collaborative filtering
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    event_id TEXT NOT NULL REFERENCES events(id),
    interaction_type TEXT NOT NULL, -- like, dislike, going, went, saved
    rating REAL,                    -- 1-5 rating for 'went' interactions
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Context
    source TEXT,           -- where interaction happened: feed, search, map
    position INTEGER,      -- position in recommendation list
    
    UNIQUE(user_id, event_id, interaction_type)
);

-- Event-Artist many-to-many relationship
CREATE TABLE event_artists (
    event_id TEXT NOT NULL REFERENCES events(id),
    artist_id TEXT NOT NULL REFERENCES artists(id),
    is_headliner BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (event_id, artist_id)
);

-- Recommendation logs for evaluation and debugging
CREATE TABLE recommendation_logs (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    session_id TEXT,
    request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Request context
    filters JSON,          -- Applied filters
    location_lat REAL,
    location_lon REAL,
    
    -- Model outputs
    content_scores JSON,   -- Event ID -> content score
    cf_scores JSON,        -- Event ID -> CF score  
    hybrid_scores JSON,    -- Event ID -> final hybrid score
    recommended_events JSON, -- Final ranked list
    
    -- Model metadata
    model_version TEXT,
    response_time_ms INTEGER
);

-- Indexes for performance
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

CREATE INDEX idx_users_h3 ON users(h3_index);

-- Views for common queries
CREATE VIEW active_events AS
SELECT * FROM events 
WHERE status = 'active' 
  AND date_time > CURRENT_TIMESTAMP;

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
GROUP BY e.id, e.title, e.description, e.date_time, e.price_min, e.price_max, e.status, e.venue_id, e.h3_index, e.source, e.created_at, e.updated_at, v.name, v.neighborhood, v.lat, v.lon;