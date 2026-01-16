#!/usr/bin/env python3
"""
Authentication router for user registration and login.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.models.requests import UserRegisterRequest, UserLoginRequest
from backend.app.models.responses import AuthResponse, UserResponse
from backend.app.services.auth_service import get_auth_service, AuthService
from backend.app.core.dependencies import auth_service as jwt_service, get_authenticated_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> AuthResponse:
    """Register a new user."""
    try:
        # Create user
        user = await auth_service.create_user(user_data)
        
        # Generate JWT token
        access_token = jwt_service.create_access_token(
            user_id=user.user_id,
            data={"email": user.email, "name": user.name}
        )
        
        return AuthResponse(
            access_token=access_token,
            user=user
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=AuthResponse)
async def login_user(
    login_data: UserLoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> AuthResponse:
    """Login user."""
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(login_data)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Generate JWT token
        access_token = jwt_service.create_access_token(
            user_id=user.user_id,
            data={"email": user.email, "name": user.name}
        )
        
        return AuthResponse(
            access_token=access_token,
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user_id: str = Depends(get_authenticated_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """Get current user profile."""
    try:
        user = await auth_service.get_user_by_id(current_user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.get("/health")
async def auth_health_check(
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Health check for auth service."""
    try:
        # Simple DB connectivity check
        if auth_service.db_pool:
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": "2024-01-01T00:00:00Z"  # Will be replaced with actual timestamp
            }
        else:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "timestamp": "2024-01-01T00:00:00Z"
            }
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

