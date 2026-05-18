from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.mongodb import connect_to_mongo, close_mongo_connection
from app.routes.auth_routes import router as auth_router
from app.routes.submission_routes import router as submission_router
from app.config import settings
from app.routes import auth_routes, submission_routes
from app.routes import metrics_routes  

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    await connect_to_mongo()
    yield
    
    await close_mongo_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Scalable ML Platform API for Requirements Submission.",
    version="1.1.0",
    lifespan=lifespan
)

app.include_router(auth_routes.router, prefix="/auth", tags=["Authentication"])
app.include_router(submission_routes.router, prefix="/submit", tags=["Submissions"])
app.include_router(metrics_routes.router, prefix="/submit", tags=["Metrics"])  # ✅ ADD THIS

@app.get("/")
async def root():
    return {"message": "Welcome to the simplified ML Platform API"}
