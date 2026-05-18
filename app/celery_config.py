from celery import Celery
from app.config import settings

celery_app = Celery(
    "modelsmith_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['app.services.training_tasks'] 
)

celery_app.conf.update(
    result_backend=settings.REDIS_URL,
    result_extended=True,
    task_track_started=True
)


celery_app.autodiscover_tasks(['app.services'])