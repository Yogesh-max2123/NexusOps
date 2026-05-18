import json
import asyncio
from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
import logging

logger = logging.getLogger("modelsmith")

class MetricsStreamer:
    """Stream training metrics to database for real-time UI updates"""
    
    def __init__(self, submission_id: str, db: AsyncIOMotorDatabase):
        self.submission_id = submission_id
        self.db = db
        self.metrics_collection = db["training_metrics"]
        self.metrics_buffer = []
    
    async def log_metric(self, metric_name: str, metric_value: float, step: int = None):
        """Log a single metric"""
        
        metric_doc = {
            "submission_id": self.submission_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "step": step,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.metrics_buffer.append(metric_doc)
        
        
        if len(self.metrics_buffer) >= 5:
            await self.flush_metrics()
    
    async def log_phase_update(self, phase: str, details: str):
        """Log training phase updates"""
        
        phase_doc = {
            "submission_id": self.submission_id,
            "type": "phase_update",
            "phase": phase,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.metrics_collection.insert_one(phase_doc)
        logger.info(f" Phase logged: {phase} - {details}")
    
    async def flush_metrics(self):
        """Flush buffered metrics to database"""
        
        if not self.metrics_buffer:
            return
        
        try:
            await self.metrics_collection.insert_many(self.metrics_buffer)
            logger.debug(f" Flushed {len(self.metrics_buffer)} metrics to database")
            self.metrics_buffer = []
        except Exception as e:
            logger.error(f" Error flushing metrics: {str(e)}")
    
    async def get_latest_metrics(self, limit: int = 50):
        """Get latest metrics for this submission"""
        
        metrics = await self.metrics_collection.find({
            "submission_id": self.submission_id
        }).sort("timestamp", -1).limit(limit).to_list(None)
        
        return list(reversed(metrics))  
    
    async def get_metrics_by_type(self, metric_name: str):
        """Get all values for a specific metric"""
        
        metrics = await self.metrics_collection.find({
            "submission_id": self.submission_id,
            "metric_name": metric_name
        }).sort("step", 1).to_list(None)
        
        return metrics
    
    async def clear_metrics(self):
        """Clear all metrics for this submission"""
        
        result = await self.metrics_collection.delete_many({
            "submission_id": self.submission_id
        })
        
        logger.info(f" Cleared {result.deleted_count} metrics for {self.submission_id}")