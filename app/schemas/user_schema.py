from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

class UserRegister(BaseModel):
    """Schema for validating user registration payload."""
    username: str
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=72)

class UserLogin(BaseModel):
    """Schema for validating user login payload (if not using OAuth2 form data directly)."""
    username: str
    password: str

class UserResponse(BaseModel):
    """Schema for serializing a User document for API responses."""
    username: str
    email: EmailStr
    tier: str
    created_at: datetime
    
    model_config = {
        "populate_by_name": True
    }

class TokenResponse(BaseModel):
    """Schema for the JWT token response."""
    access_token: str
    token_type: str = "bearer"
