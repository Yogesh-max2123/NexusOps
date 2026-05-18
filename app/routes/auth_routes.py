from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from app.schemas.user_schema import UserRegister, UserResponse, TokenResponse
from app.services.auth_service import AuthService
from app.database.mongodb import get_database
from app.utils.security import verify_password, create_access_token

router = APIRouter()

def get_auth_service(db = Depends(get_database)) -> AuthService:
    return AuthService(db)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, auth_service: AuthService = Depends(get_auth_service)):
    """Registers a new user."""
    return await auth_service.register_user(user_data)

@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_database)):
    """Authenticates a user and returns a JWT access token."""
    user = await db["users"].find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return TokenResponse(access_token=access_token, token_type="bearer")

@router.post("/logout")
async def logout():
    """Logs out the user. In a stateless JWT implementation, this can simply return success."""
    return {"message": "Successfully logged out"}
