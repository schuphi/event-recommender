#!/usr/bin/env python3
"""
Authentication service for user management with PostgreSQL.
"""

import asyncio
import asyncpg
import bcrypt
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.core.config import Settings
from backend.app.models.requests import UserRegisterRequest, UserLoginRequest
from backend.app.models.responses import UserResponse

logger = logging.getLogger(__name__)

class AuthService:
    """Service for user authentication and management."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db_pool: Optional[asyncpg.Pool] = None
    
    async def init_pool(self):
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.settings.DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close_pool(self):
        """Close database connection pool."""
        if self.db_pool:
            await self.db_pool.close()
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    async def create_user(self, user_data: UserRegisterRequest) -> UserResponse:
        """Create a new user."""
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
        
        user_id = str(uuid.uuid4())
        hashed_password = self._hash_password(user_data.password)
        
        async with self.db_pool.acquire() as conn:
            # Check if user already exists
            existing = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", 
                user_data.email
            )
            
            if existing:
                raise ValueError("User with this email already exists")
            
            # Create user
            await conn.execute("""
                INSERT INTO users (id, email, name, password_hash, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, user_data.email, user_data.name, hashed_password, datetime.utcnow())
            
            # Return user data
            return UserResponse(
                user_id=user_id,
                email=user_data.email,
                name=user_data.name,
                created_at=datetime.utcnow(),
                interaction_count=0
            )
    
    async def authenticate_user(self, login_data: UserLoginRequest) -> Optional[UserResponse]:
        """Authenticate user and return user data if valid."""
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.db_pool.acquire() as conn:
            user_row = await conn.fetchrow("""
                SELECT id, email, name, password_hash, created_at,
                       (SELECT COUNT(*) FROM interactions WHERE user_id = users.id) as interaction_count
                FROM users 
                WHERE email = $1
            """, login_data.email)
            
            if not user_row:
                return None
            
            # Verify password
            if not self._verify_password(login_data.password, user_row['password_hash']):
                return None
            
            # Update last active
            await conn.execute(
                "UPDATE users SET last_active = $1 WHERE id = $2",
                datetime.utcnow(), user_row['id']
            )
            
            return UserResponse(
                user_id=user_row['id'],
                email=user_row['email'],
                name=user_row['name'],
                created_at=user_row['created_at'],
                interaction_count=user_row['interaction_count']
            )
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        if not self.db_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.db_pool.acquire() as conn:
            user_row = await conn.fetchrow("""
                SELECT id, email, name, created_at, preferences, location_lat, location_lon, last_active,
                       (SELECT COUNT(*) FROM interactions WHERE user_id = users.id) as interaction_count
                FROM users 
                WHERE id = $1
            """, user_id)
            
            if not user_row:
                return None
            
            return UserResponse(
                user_id=user_row['id'],
                email=user_row['email'],
                name=user_row['name'],
                preferences=user_row['preferences'],
                location_lat=user_row['location_lat'],
                location_lon=user_row['location_lon'],
                created_at=user_row['created_at'],
                last_active=user_row['last_active'],
                interaction_count=user_row['interaction_count']
            )

# Global auth service instance
auth_service: Optional[AuthService] = None

async def get_auth_service() -> AuthService:
    """Get auth service instance."""
    global auth_service
    if not auth_service:
        from backend.app.core.config import settings
        auth_service = AuthService(settings)
        await auth_service.init_pool()
    return auth_service

