from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException, Response
from app.schemas.submission_schema import SubmissionResponse
from app.services.submission_service import SubmissionService
from app.database.mongodb import get_database
from app.utils.dependencies import get_current_user
from app.config import logger
import os
import io
import json
import urllib.request
import zipfile
import sys
from pydantic import BaseModel


from celery import chord
from app.services.training_tasks import run_optuna_trials_task, finalize_model_training_task
from app.celery_config import celery_app 

router = APIRouter()

def get_submission_service(db = Depends(get_database)) -> SubmissionService:
    return SubmissionService(db)

@router.post("/", response_model=SubmissionResponse)
async def submit_job(
    target_column: str = Form(...),
    use_case: str = Form(...),
    requirement: str = Form(...),
    dataset: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Submits the model requirements and uploads the dataset securely."""
    user_id = str(current_user["_id"])
    logger.info(f"New submission received from user: {user_id}")
    logger.info(f"Target: {target_column}, Use Case: {use_case[:50]}...")
    
    return await submission_service.create_submission(
        user_id=user_id, 
        file=dataset, 
        target_column=target_column,
        use_case=use_case,
        requirement=requirement
    )

@router.get("/", response_model=list[SubmissionResponse])
async def get_my_submissions(
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Retrieves all submissions for the logged-in user."""
    user_id = str(current_user["_id"])
    return await submission_service.get_user_submissions(user_id)



async def run_training_pipeline(submission_id: str, service: SubmissionService, n_trials: int = 20, n_workers: int = 4):
    """Distributed background training pipeline using Celery and Redis"""
    
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"DISTRIBUTED TRAINING PIPELINE STARTED for submission: {submission_id}")
        logger.info(f"{'='*70}\n")
        
        # STEP 1: Mark as training
        logger.info("STEP 1: Updating status to 'training'...")
        await service.update_submission_status(submission_id, "training")
        logger.info("Status updated: pending -> training\n")
        
        # STEP 2: Get Submission metadata
        logger.info("STEP 2: Fetching submission metadata from MongoDB...")
        sub = await service.get_submission(submission_id)
        if not sub:
            logger.error(f"Submission {submission_id} not found!")
            return

        csv_url = sub["dataset_url"]
        target = sub["target_column"]
        use_case = sub["use_case"]
        requirement = sub["requirement"]
        
        logger.info(f"   Target: {target}")
        logger.info(f"   Use Case: {use_case[:50]}...")
        logger.info(f"   Dataset URL: {csv_url[:50]}...\n")
        
        # STEP 3: Download CSV from Cloudinary locally for Workers
        logger.info("STEP 3: Downloading CSV for Distributed Workers...")
        local_csv = f"{submission_id}.csv"
        try:
            import os 
            urllib.request.urlretrieve(csv_url, local_csv)
            local_csv_path = os.path.abspath(local_csv) 
            file_size = os.path.getsize(local_csv_path) / 1024  # KB
            logger.info(f"CSV downloaded successfully ({file_size:.2f} KB)\n")
        except Exception as e:
            logger.error(f"Failed to download CSV: {str(e)}")
            await service.update_submission_status(submission_id, "failed")
            return
        
        logger.info(" STEP 3.5: Generating Model Constraints via Gemini API (Once)...")
        from Model_Training.ML_Pipeline.InputState import InputStateBuilder
        from Model_Training.ML_Pipeline.ConstraintEngine import run_constraint_engine
        import os
        from dotenv import load_dotenv
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            state_builder = InputStateBuilder(api_key=api_key) 
            input_state = state_builder.build(
                dataset_path=local_csv_path,
                use_case=use_case,
                user_text=requirement,
                target_col=target
            )
            
            final_space = run_constraint_engine(input_state, api_key)
            logger.info("✅ Constraints generated successfully!")
            
        except Exception as e:
            logger.error(f"Failed to generate constraints: {e}") 
            await service.update_submission_status(submission_id, "failed")
            return
            
        
        logger.info("STEP 4: Dispatching Optuna Trials to Celery Workers...")
        logger.info("Architecture Shift: Subprocess -> Redis Message Broker")
        
        study_name = f"optuna_study_{submission_id}"
        
        
        total_trials = n_trials
        num_workers = n_workers 
        
        
        trials_per_worker = max(1, total_trials // num_workers)
        
       
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        celery_app.conf.update(
            result_backend=redis_url,
            result_extended=True
        )
        
        parallel_tasks = [
        run_optuna_trials_task.s(study_name, csv_url, target, final_space, trials_per_worker) 
        for _ in range(num_workers)
        ]
        

        callback_task = finalize_model_training_task.s(study_name, csv_url, target, submission_id)
        
        chord(parallel_tasks)(callback_task)
        
        logger.info(f"Distributed Tasks successfully sent to Redis Queue!")
        logger.info(f"Note: FastAPI is now free! Celery will handle the DB sync and artifacts.\n")
        
    except Exception as e:
        logger.error(f"\nEXCEPTION in dispatching training: {str(e)}", exc_info=True)
        await service.update_submission_status(submission_id, "failed")



class TrainingConfig(BaseModel):
    n_trials: int = 20
    n_workers: int = 4

@router.post("/{submission_id}/train")
async def trigger_training(
    submission_id: str, 
    config: TrainingConfig,  
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Triggers the background training pipeline."""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
   
    max_safe_workers = max(1, (os.cpu_count() or 4) - 1)
    
    
    final_workers = min(config.n_workers, max_safe_workers)
    
    logger.info(f"Training triggered for {submission_id} | Trials: {config.n_trials} | Workers: {final_workers}")
    
    
    background_tasks.add_task(
        run_training_pipeline, 
        submission_id, 
        submission_service, 
        config.n_trials,    
        final_workers       
    )
    
    return {
        "message": "Training started in distributed mode!",
        "trials": config.n_trials,
        "workers_allocated": final_workers
    }

@router.get("/{submission_id}/download")
async def download_model(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Downloads the trained model and config as a ZIP file."""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
        
    if sub.get("status") != "completed" or not sub.get("model_artifact"):
        raise HTTPException(status_code=400, detail="Model is not ready yet.")
    
    logger.info(f"Downloading model for submission: {submission_id}")
    
    zip_buffer = io.BytesIO()
    model_bytes = sub["model_artifact"]
    config_str = json.dumps(sub["model_config_json"], indent=4)
    target = sub["target_column"]
    safe_name = target.replace(" ", "_").lower()
    
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr(f"{safe_name}_best_model.pth", model_bytes)
        zip_file.writestr(f"{safe_name}_model_config.json", config_str)
        
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={safe_name}_model_artifacts.zip"}
    )


@router.post("/{submission_id}/predict")
async def predict_with_model(
    submission_id: str,
    test_data: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Feeds new sample data into the completed model and returns predictions."""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
        
    if sub.get("status") != "completed" or not sub.get("model_artifact"):
        raise HTTPException(status_code=400, detail="Model is not fully trained yet.")
        
    if test_data.filename and not test_data.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV sample data is supported.")
    
    logger.info(f"Running predictions for submission: {submission_id}")
    sample_bytes = await test_data.read()
    
    from app.utils.predict_utils import run_prediction_pipeline
    
    try:
        result = run_prediction_pipeline(submission_id, sub, sample_bytes)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/{submission_id}/mlflow-dashboard")
async def get_mlflow_dashboard(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Get MLflow dashboard URL for this submission"""
    
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    mlflow_run_id = sub.get("mlflow_run_id")
    
    if not mlflow_run_id:
        raise HTTPException(status_code=400, detail="MLflow tracking data not found for this submission")
    
    import os
    
    # Base URL dynamic tareeqe se nikalenge
    base_mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").rstrip('/')
    
    # Ab localhost ki jagah base_mlflow_url variable use karenge
    mlflow_url = f"{base_mlflow_url}/#/experiments/0/runs/{mlflow_run_id}"
    
    logger.info(f"Returning MLflow dashboard for {submission_id}")
    
    return {
        "mlflow_run_id": mlflow_run_id,
        "dashboard_url": mlflow_url,
        "local_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        "message": "Click dashboard_url to view training metrics, curves, and artifacts"
    }
    
@router.delete("/{submission_id}")
async def delete_submission_route(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Deletes a submission completely from DB."""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    success = await submission_service.delete_submission(submission_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete submission")
        
    return {"message": "Submission deleted successfully!"}

from bson import ObjectId
from fastapi import HTTPException


@router.get("/{submission_id}")
async def get_single_submission(submission_id: str, db=Depends(get_database)):
    try:
        
        record = await db["submissions"].find_one(
            {"_id": ObjectId(submission_id)},
            {"model_artifact": 0} 
        )
        
        if not record:
            raise HTTPException(status_code=404, detail="Submission not found")
            
        record["_id"] = str(record["_id"])
        
        return record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
from fastapi.responses import Response

@router.get("/{submission_id}/model-file")
async def download_model_weights(submission_id: str, db=Depends(get_database)):
    try:
       
        record = await db["submissions"].find_one(
            {"_id": ObjectId(submission_id)},
            {"model_artifact": 1, "target_col": 1}
        )
        
        if not record or "model_artifact" not in record:
            raise HTTPException(status_code=404, detail="Model weights (.pth) not found in database.")
            
        target = record.get("target_col", "model").replace(" ", "_").lower()
        
        
        return Response(
            content=record["model_artifact"], 
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{target}_weights.pth"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))