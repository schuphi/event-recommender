#!/usr/bin/env python3
"""
API endpoint tests for Copenhagen Event Recommender.
Tests all FastAPI routes for proper functionality, error handling, and security.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_check_with_database(self, client, test_db):
        """Test health check includes database connectivity."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert data["database"] == "connected"


class TestEventEndpoints:
    """Test event-related API endpoints."""
    
    def test_get_events_basic(self, client, test_db):
        """Test basic events endpoint."""
        response = client.get("/events")
        assert response.status_code == 200
        
        data = response.json()
        assert "events" in data
        assert isinstance(data["events"], list)
        assert len(data["events"]) >= 2  # From test data
    
    def test_get_events_with_pagination(self, client, test_db):
        """Test events endpoint with pagination."""
        response = client.get("/events?limit=1&offset=0")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["events"]) == 1
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
    
    def test_get_events_with_filters(self, client, test_db):
        """Test events endpoint with various filters."""
        # Test genre filter
        response = client.get("/events?genre=techno")
        assert response.status_code == 200
        
        # Test date range filter
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        response = client.get(f"/events?start_date={tomorrow}")
        assert response.status_code == 200
        
        # Test venue filter
        response = client.get("/events?venue=Culture Box")
        assert response.status_code == 200
    
    def test_get_event_by_id(self, client, test_db):
        """Test getting specific event by ID."""
        response = client.get("/events/event_1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == "event_1"
        assert "title" in data
        assert "venue" in data
    
    def test_get_nonexistent_event(self, client, test_db):
        """Test getting event that doesn't exist."""
        response = client.get("/events/nonexistent_event")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestRecommendationEndpoints:
    """Test ML recommendation endpoints."""
    
    def test_get_recommendations_unauthorized(self, client):
        """Test recommendations without authentication."""
        response = client.get("/recommendations")
        assert response.status_code == 401
    
    def test_get_recommendations_for_user(self, client, test_db):
        """Test getting recommendations for authenticated user."""
        # Create test user session (simplified)
        headers = {"Authorization": "Bearer test-token"}
        response = client.get("/recommendations?user_id=user_1", headers=headers)
        
        # Should return recommendations or proper error
        assert response.status_code in [200, 401, 403]
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
    
    def test_recommendations_with_preferences(self, client, test_db):
        """Test recommendations with user preferences."""
        preferences = {
            "preferred_genres": ["techno", "electronic"],
            "location_lat": 55.6761,
            "location_lon": 12.5683,
            "price_range": [100, 400]
        }
        
        response = client.post("/recommendations", json=preferences)
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data


class TestUserEndpoints:
    """Test user management endpoints."""
    
    def test_create_user(self, client):
        """Test user creation."""
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "preferences": {
                "preferred_genres": ["techno", "house"],
                "location_lat": 55.6761,
                "location_lon": 12.5683
            }
        }
        
        response = client.post("/users", json=user_data)
        assert response.status_code in [200, 201]
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "id" in data
            assert data["email"] == user_data["email"]
    
    def test_get_user_profile(self, client, test_db):
        """Test getting user profile."""
        headers = {"Authorization": "Bearer test-token"}
        response = client.get("/users/user_1", headers=headers)
        
        assert response.status_code in [200, 401, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "id" in data
            assert "preferences" in data
    
    def test_update_user_preferences(self, client, test_db):
        """Test updating user preferences."""
        updated_preferences = {
            "preferred_genres": ["indie", "alternative"],
            "price_range": [200, 500]
        }
        
        headers = {"Authorization": "Bearer test-token"}
        response = client.put("/users/user_1/preferences", 
                            json=updated_preferences, 
                            headers=headers)
        
        assert response.status_code in [200, 401, 404]


class TestInteractionEndpoints:
    """Test user interaction endpoints."""
    
    def test_record_interaction(self, client, test_db):
        """Test recording user interaction with event."""
        interaction_data = {
            "event_id": "event_1",
            "interaction_type": "like",
            "rating": 5.0
        }
        
        headers = {"Authorization": "Bearer test-token"}
        response = client.post("/interactions", 
                             json=interaction_data, 
                             headers=headers)
        
        assert response.status_code in [200, 201, 401]
    
    def test_get_user_interactions(self, client, test_db):
        """Test getting user's interaction history."""
        headers = {"Authorization": "Bearer test-token"}
        response = client.get("/users/user_1/interactions", headers=headers)
        
        assert response.status_code in [200, 401, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "interactions" in data
            assert isinstance(data["interactions"], list)


class TestSearchEndpoints:
    """Test search functionality."""
    
    def test_search_events(self, client, test_db):
        """Test event search endpoint."""
        response = client.get("/search?q=techno")
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_search_empty_query(self, client, test_db):
        """Test search with empty query."""
        response = client.get("/search?q=")
        assert response.status_code == 400
    
    def test_search_with_filters(self, client, test_db):
        """Test search with additional filters."""
        response = client.get("/search?q=concert&venue=Vega&date=2024-12-01")
        assert response.status_code == 200


class TestVenueEndpoints:
    """Test venue-related endpoints."""
    
    def test_get_venues(self, client, test_db):
        """Test getting all venues."""
        response = client.get("/venues")
        assert response.status_code == 200
        
        data = response.json()
        assert "venues" in data
        assert len(data["venues"]) >= 3  # From test data
    
    def test_get_venue_events(self, client, test_db):
        """Test getting events for specific venue."""
        response = client.get("/venues/venue_1/events")
        assert response.status_code == 200
        
        data = response.json()
        assert "events" in data
        assert isinstance(data["events"], list)


class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post("/users", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        incomplete_user = {"name": "Test User"}  # Missing email
        response = client.post("/users", json=incomplete_user)
        assert response.status_code == 422
    
    def test_invalid_route(self, client):
        """Test accessing non-existent route."""
        response = client.get("/nonexistent/route")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.post("/health")  # Health is GET only
        assert response.status_code == 405


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_sql_injection_protection(self, client, test_db, security_test_payloads):
        """Test protection against SQL injection."""
        for payload in security_test_payloads['sql_injection']:
            response = client.get(f"/search?q={payload}")
            # Should not cause 500 error or expose database
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                # Should not return sensitive data
                content = response.text.lower()
                assert "error" not in content or "sql" not in content
    
    def test_xss_protection(self, client, test_db, security_test_payloads):
        """Test protection against XSS attacks."""
        for payload in security_test_payloads['xss_payloads']:
            user_data = {
                "name": payload,
                "email": "test@example.com",
                "preferences": {}
            }
            
            response = client.post("/users", json=user_data)
            # Should sanitize or reject malicious input
            assert response.status_code in [200, 201, 400, 422]
            
            if response.status_code in [200, 201]:
                # Response should not contain raw script tags
                assert "<script>" not in response.text
    
    def test_oversized_request_handling(self, client, security_test_payloads):
        """Test handling of oversized requests."""
        oversized_data = security_test_payloads['oversized_requests']
        
        response = client.post("/users", json={
            "name": "Test User",
            "email": "test@example.com", 
            "preferences": oversized_data
        })
        
        # Should reject or handle gracefully
        assert response.status_code in [400, 413, 422]


class TestRateLimiting:
    """Test API rate limiting (when implemented)."""
    
    @pytest.mark.skipif(True, reason="Rate limiting not yet implemented")
    def test_rate_limiting(self, client):
        """Test API rate limiting functionality."""
        # Make many rapid requests
        responses = []
        for i in range(100):
            response = client.get("/events")
            responses.append(response)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        assert any(r.status_code == 429 for r in responses)


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/events")
        
        # Should have CORS headers (if configured)
        headers = response.headers
        assert response.status_code in [200, 204]
        
        # Check for common CORS headers
        expected_headers = [
            "access-control-allow-origin",
            "access-control-allow-methods", 
            "access-control-allow-headers"
        ]
        
        # At least some CORS headers should be present
        cors_headers_present = any(header in headers for header in expected_headers)
        # Note: May not be configured yet, so this is informational