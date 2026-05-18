from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class UserModel(BaseModel):
    """
    User entity representing a document in the MongoDB 'users' collection.
    """
    id: Optional[str] = Field(alias="_id", default=None)
    username: str
    email: EmailStr
    password_hash: str
    tier: str = "free"  
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }
