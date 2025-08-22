#!/usr/bin/env python3
"""
FastAPI dependencies for the Copenhagen Event Recommender API.
"""

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Annotated
import jwt
from datetime import datetime, timedelta
import logging

from .config import settings
from ..services.database_service import DatabaseService

logger = logging.getLogger(__name__)

# Optional JWT authentication
security = HTTPBearer(auto_error=False)

class AuthenticationService:
    """Service for handling user authentication."""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expiration_hours = settings.JWT_EXPIRATION_HOURS
    
    def create_access_token(self, user_id: str, data: dict = None) -> str:
        """Create JWT access token."""
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY not configured")
        
        to_encode = {"sub": user_id}
        if data:
            to_encode.update(data)
        
        expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id."""
        if not self.secret_key:
            return None  # Authentication disabled
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            
            if user_id is None:
                return None
            
            return user_id
            
        except jwt.PyJWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None

# Global auth service
auth_service = AuthenticationService()

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_user_id: Optional[str] = Header(None)
) -> Optional[str]:
    """
    Get current user from JWT token or X-User-ID header.
    Returns None if no authentication provided (anonymous access allowed).
    """
    
    # Try JWT authentication first
    if credentials:
        user_id = auth_service.verify_token(credentials.credentials)
        if user_id:
            return user_id
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Fallback to X-User-ID header for anonymous users
    if x_user_id:
        return x_user_id
    
    # No authentication provided - anonymous access
    return None

async def get_authenticated_user(
    current_user: Optional[str] = Depends(get_current_user)
) -> str:
    """
    Require authenticated user (not anonymous).
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user

# Database service dependency
_db_service = None

def get_db_service() -> DatabaseService:
    """Get database service instance (singleton)."""
    global _db_service
    
    if _db_service is None:
        _db_service = DatabaseService()
    
    return _db_service

# Rate limiting dependency
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}  # user_id -> list of request timestamps
        self.max_requests = settings.RATE_LIMIT_REQUESTS_PER_MINUTE
        self.window_seconds = 60
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        now = datetime.utcnow()
        
        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                timestamp for timestamp in self.requests[user_id]
                if (now - timestamp).total_seconds() < self.window_seconds
            ]
        else:
            self.requests[user_id] = []
        
        # Check limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True

rate_limiter = RateLimiter()

async def check_rate_limit(
    current_user: Optional[str] = Depends(get_current_user),
    x_forwarded_for: Optional[str] = Header(None)
):
    """Check rate limits for user or IP."""
    
    # Use user_id if available, otherwise use IP
    identifier = current_user or x_forwarded_for or "anonymous"
    
    if not rate_limiter.is_allowed(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )

# Validation dependencies
def validate_location(lat: Optional[float], lon: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Validate latitude/longitude coordinates."""
    
    if lat is not None and (lat < -90 or lat > 90):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Latitude must be between -90 and 90"
        )
    
    if lon is not None and (lon < -180 or lon > 180):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Longitude must be between -180 and 180"
        )
    
    # Validate both are provided together
    if (lat is None) != (lon is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both latitude and longitude must be provided"
        )
    
    return lat, lon

def validate_copenhagen_location(lat: Optional[float], lon: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Validate location is within Copenhagen area."""
    
    lat, lon = validate_location(lat, lon)
    
    if lat is not None and lon is not None:
        # Check if location is roughly within Copenhagen area
        copenhagen_bounds = {
            'lat_min': 55.6150, 'lat_max': 55.7350,
            'lon_min': 12.4500, 'lon_max': 12.6500
        }
        
        if not (copenhagen_bounds['lat_min'] <= lat <= copenhagen_bounds['lat_max'] and
                copenhagen_bounds['lon_min'] <= lon <= copenhagen_bounds['lon_max']):
            logger.warning(f"Location outside Copenhagen: {lat}, {lon}")
            # Don't raise error, just log warning
    
    return lat, lon

# Pagination dependency
class PaginationParams:
    """Pagination parameters."""
    
    def __init__(
        self,
        page: int = 1,
        size: int = 20,
        max_size: int = 100
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page must be >= 1"
            )
        
        if size < 1 or size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Size must be between 1 and {max_size}"
            )
        
        self.page = page
        self.size = size
        self.offset = (page - 1) * size
        self.limit = size

def get_pagination(
    page: int = 1,
    size: int = 20
) -> PaginationParams:
    """Get pagination parameters."""
    return PaginationParams(page=page, size=size)

# Admin authentication dependency
async def require_admin(
    current_user: str = Depends(get_authenticated_user),
    db: DatabaseService = Depends(get_db_service)
) -> str:
    """Require admin privileges."""
    
    # Simple admin check - in production, implement proper role system
    admin_users = ["admin", "system"]  # Could be loaded from config or database
    
    if current_user not in admin_users:
        # Check if user has admin role in database
        user = await db.get_user(current_user)
        if not user or not user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
    
    return current_user

# Content validation
def validate_content_length(content: str, max_length: int, field_name: str) -> str:
    """Validate content length."""
    
    if len(content) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} cannot exceed {max_length} characters"
        )
    
    return content.strip()

# Cache dependencies
def get_cache_key(*args) -> str:
    """Generate cache key from arguments."""
    return ":".join(str(arg) for arg in args if arg is not None)

# Health check dependency
async def verify_system_health() -> bool:
    """Verify system health for critical operations."""
    
    try:
        # Check database
        db = get_db_service()
        db_healthy = await db.health_check()
        
        # Check models (simplified)
        models_healthy = True  # Would check model loading status
        
        return db_healthy and models_healthy
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return False

async def require_system_health():
    """Require system to be healthy."""
    
    if not await verify_system_health():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is currently unhealthy"
        )