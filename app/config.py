from pydantic_settings import BaseSettings
import logging
from typing import Optional

class Settings(BaseSettings):
    
    PROJECT_NAME: str = "ModelSmith"
    DEBUG: bool = True
    
    
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "modelsmith"
    REDIS_URL: str = "redis://localhost:6379"
    
   
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    
    GEMINI_API_KEY: str
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

  
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MLFLOW_BACKEND_STORE: str = "./mlruns"
    MLFLOW_ARTIFACT_STORE: str = "./mlartifacts"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  

settings = Settings()


def setup_logging():
    """Configure detailed logging for all operations"""
    
    logger = logging.getLogger("modelsmith")
    logger.setLevel(logging.DEBUG)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('modelsmith.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()