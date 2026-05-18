import os
import json
from fastapi import APIRouter, Depends, HTTPException
from app.database.mongodb import get_database
from app.utils.dependencies import get_current_user
from app.services.submission_service import SubmissionService
from app.config import logger

router = APIRouter()

def get_submission_service(db = Depends(get_database)) -> SubmissionService:
    return SubmissionService(db)

# 1. LIVE METRICS ROUTE (UPDATED WITH STATS & SCATTER PLOT DATA )
@router.get("/{submission_id}/live-metrics")
async def get_live_metrics(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Fetch real-time training metrics from MongoDB in pure array format"""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
   
    config = sub.get("model_config_json", {})
    metrics_data = config.get("metrics", {})

    
    if not metrics_data or not metrics_data.get("epochs"):
         return {
            "epochs": [],
            "train_losses": [],
            "test_rmses": [],
            "final_score": None,
            "status": sub.get("status", "in_progress")
        }

    response_data = {
        "epochs": metrics_data.get("epochs", []),
        "train_losses": metrics_data.get("train_losses", []),
        "test_rmses": metrics_data.get("test_rmses", []),
        "test_maes": metrics_data.get("test_maes", []), 
        "test_r2s": metrics_data.get("test_r2s", []),  
        "final_score": config.get("final_score", None),
        "status": sub.get("status", "completed" if sub.get("status") == "completed" else "in_progress")
    }

    
    if config:
        
        response_data["execution_stats"] = config.get("execution_stats", {})
        response_data["best_params"] = config.get("best_params", {})
        
        
        scatter_data = config.get("scatter_plot_data", {})
        response_data["actual_targets"] = scatter_data.get("actual_targets", [])
        response_data["predicted_targets"] = scatter_data.get("predicted_targets", [])
        
        
        response_data["feature_importance"] = config.get("feature_importance", {}) 
        response_data["llm_executive_summary"] = config.get("llm_executive_summary", "")

    return response_data

# 2. DEBUG CONFIG - AUTH REQUIRED
@router.get("/{submission_id}/debug-config")
async def debug_config(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Debug endpoint - see what's saved in DB config"""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404)
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403)
    
    config = sub.get("model_config_json", {})
    return {
        "submission_id": submission_id,
        "status": sub.get("status"),
        "config_keys": list(config.keys()),
        "has_metrics": "metrics" in config,
        "metrics_if_exists": config.get("metrics", {})
    }

# 3. PUBLIC DEBUG CONFIG - NO AUTH 
@router.get("/{submission_id}/debug-config-public")
async def debug_config_public(
    submission_id: str,
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Public debug endpoint - NO AUTH REQUIRED"""
    try:
        sub = await submission_service.get_submission(submission_id)
        
        if not sub:
            return {"error": "Submission not found in Database"}
        
        config = sub.get("model_config_json", {})
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PUBLIC DEBUG CONFIG for {submission_id}")
        logger.info(f"Config Keys: {list(config.keys())}")
        logger.info(f"Has metrics: {'metrics' in config}")
        
        if "metrics" in config:
            metrics = config.get("metrics", {})
            logger.info(f"Metrics Keys: {list(metrics.keys())}")
            logger.info(f"Epochs: {len(metrics.get('epochs', []))} values")
            logger.info(f"Train Losses: {len(metrics.get('train_losses', []))} values")
            logger.info(f"Test RMSEs: {len(metrics.get('test_rmses', []))} values")
        logger.info(f"{'='*70}\n")
        
        return {
            "submission_id": submission_id,
            "status": sub.get("status"),
            "config_keys": list(config.keys()),
            "has_metrics": "metrics" in config,
            "metrics_structure": config.get("metrics", {}) if "metrics" in config else "NO METRICS SAVED YET",
            "full_config": config
        }
    
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        return {"error": f"Something broke: {str(e)}"}