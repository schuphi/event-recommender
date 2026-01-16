-- Migration: Add topic categorization to events
-- Run after initial schema creation

-- Add topic column (required, defaults to 'music' for existing events)
ALTER TABLE events ADD COLUMN topic VARCHAR(50) NOT NULL DEFAULT 'music';

-- Add tags array for secondary labels
ALTER TABLE events ADD COLUMN tags JSON DEFAULT '[]';

-- Add is_free convenience flag
ALTER TABLE events ADD COLUMN is_free BOOLEAN DEFAULT FALSE;

-- Create index for topic filtering
CREATE INDEX idx_events_topic ON events(topic);

-- Create composite index for common query pattern
CREATE INDEX idx_events_topic_datetime ON events(topic, date_time);

-- Update is_free based on existing price data
UPDATE events SET is_free = TRUE WHERE price_min IS NULL OR price_min = 0;

-- Remove unused columns (optional - run separately if needed)
-- ALTER TABLE events DROP COLUMN IF EXISTS embedding;
-- ALTER TABLE events DROP COLUMN IF EXISTS content_features;

-- View for topic statistics
CREATE VIEW topic_stats AS
SELECT
    topic,
    COUNT(*) as event_count,
    COUNT(CASE WHEN date_time > CURRENT_TIMESTAMP THEN 1 END) as upcoming_count,
    AVG(CASE WHEN price_min IS NOT NULL THEN price_min ELSE 0 END) as avg_price
FROM events
WHERE status = 'active'
GROUP BY topic;
