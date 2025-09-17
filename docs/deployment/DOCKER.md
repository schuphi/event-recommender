# Docker Deployment Guide

## Overview

The application is containerized with Docker for consistent deployment across environments.

## Configuration Files

- `config/docker/docker-compose.yml` - Production configuration
- `config/docker/docker-compose.dev.yml` - Development configuration
- `config/docker/Dockerfile` - Main application image
- `config/docker/Dockerfile.railway` - Railway-specific image

## Development Deployment

```bash
# Start development environment
docker compose -f config/docker/docker-compose.dev.yml up -d

# View logs
docker compose -f config/docker/docker-compose.dev.yml logs -f

# Stop services
docker compose -f config/docker/docker-compose.dev.yml down
```

This provides:
- Hot reloading for development
- Debug port exposure
- Volume mounts for live code editing
- Development database

## Production Deployment

```bash
# Build and start production services
docker compose -f config/docker/docker-compose.yml up -d

# Scale API service
docker compose -f config/docker/docker-compose.yml up --scale api=3 -d
```

Production features:
- Optimized multi-stage builds
- Health checks
- Automatic restarts
- Production database persistence
- Nginx reverse proxy
- Monitoring with Grafana/Prometheus

## Environment Variables

Configure via `.env` file:

```bash
# Database
DATABASE_URL=data/events/events.duckdb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# External APIs
EVENTBRITE_API_TOKEN=your_token
```

## Health Checks

All services include health checks:
- API: `GET /health`
- Database: Connection test
- Frontend: Static file serving

## Volumes

Data persistence:
- `./data:/app/data` - Database and cached data
- `./logs:/app/logs` - Application logs

## Networking

Services communicate via Docker network:
- Frontend: Port 3000
- API: Port 8000
- Database: Internal only
- Nginx: Port 80/443

## Troubleshooting

```bash
# Check service status
docker compose ps

# View service logs
docker compose logs [service-name]

# Rebuild specific service
docker compose build [service-name]

# Reset all data
docker compose down -v
docker compose up -d
```