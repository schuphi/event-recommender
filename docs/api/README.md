# API Documentation

## Overview

The Copenhagen Event Recommender API provides endpoints for event discovery, search, and personalized recommendations.

**Base URL**: `http://localhost:8000`
**Interactive Docs**: `http://localhost:8000/docs`

## Authentication

Currently supports optional JWT authentication for personalized features. Anonymous access is allowed for basic functionality.

## Endpoints

### Health Check
```http
GET /health
```
Returns system health status and database statistics.

### Events
```http
GET /events?limit=10&price_max=500&genre=electronic
```
List events with optional filters:
- `limit`: Number of events to return
- `offset`: Pagination offset
- `price_min/price_max`: Price range in DKK
- `genre`: Filter by music genre
- `venue`: Filter by venue name
- `date_from/date_to`: Date range filters

### Event Details
```http
GET /events/{event_id}
```
Get detailed information for a specific event.

### Recommendations
```http
GET /recommend/{user_id}?limit=5
```
Get personalized event recommendations for a user.

### Search
```http
GET /search?query=electronic&limit=10
```
Search events by text query across titles, descriptions, and artist names.

### Statistics
```http
GET /stats
```
Get database and system statistics.

## Response Format

All responses follow this structure:
```json
{
  "data": [...],
  "meta": {
    "total": 100,
    "limit": 10,
    "offset": 0
  }
}
```

## Error Handling

API uses standard HTTP status codes:
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

Error response format:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter",
    "details": {...}
  }
}
```