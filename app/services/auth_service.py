from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.user_schema import UserRegister, UserResponse
from app.utils.security import get_password_hash
from app.models.user_model import UserModel

class AuthService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["users"]

    async def register_user(self, user_data: UserRegister) -> UserResponse:
        existing_user = await self.collection.find_one({"username": user_data.username})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        existing_email = await self.collection.find_one({"email": user_data.email})
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        hashed_password = get_password_hash(user_data.password)
        new_user = UserModel(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password
        )
        
        user_dict = new_user.model_dump(by_alias=True, exclude_none=True)
        result = await self.collection.insert_one(user_dict)
        
        
        user_dict["_id"] = str(result.inserted_id)
        return UserResponse(**user_dict)
