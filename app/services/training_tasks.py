import os
import sys
import json
import logging
import asyncio
import time
import random
from celery import shared_task
import optuna
import torch
import torch.nn as nn
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
modelsmith_root = os.path.abspath(os.path.join(current_dir, "../../"))

if modelsmith_root not in sys.path:
    sys.path.insert(0, modelsmith_root)

model_training_path = os.path.abspath(os.path.join(modelsmith_root, "Model_Training"))
if model_training_path not in sys.path:
    sys.path.insert(0, model_training_path)

logger = logging.getLogger(__name__)


from app.services.submission_service import SubmissionService
from app.database.mongodb import get_database

async def async_save_to_mongodb(submission_id, target_col, model_bytes, config, csv_path):
    try:
        import os
        from motor.motor_asyncio import AsyncIOMotorClient
        from bson import ObjectId  
        from dotenv import load_dotenv
        
        load_dotenv(override=True)
        mongo_uri = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        db_name = os.getenv("DATABASE_NAME", "modelsmith")
        
        
        client = AsyncIOMotorClient(mongo_uri)
        db = client[db_name]
        
        
        await db["submissions"].update_one(
            {"_id": ObjectId(submission_id)},
            {"$set": {
                "status": "completed",
                "model_file": model_bytes,
                "model_config_json": config
            }}
        )
        logger.info(f" MongoDB Update Success for {submission_id}")
        
       
        client.close()
        
    except Exception as e:
        logger.error(f" MongoDB Update Failed: {e}")

# TASK A: RUN OPTUNA TRIALS

from celery import shared_task

@shared_task(name="run_optuna_trials_task")
def run_optuna_trials_task(study_name, csv_url, target_col, final_space, n_trials=5, n_workers=1): 
    """TASK A: Distributed Optuna Worker"""
    import os
    import urllib.request
    import optuna
    import torch
    from dotenv import load_dotenv
    
    torch.set_num_threads(1)
    # Worker ke andar file download karwao
    local_csv_path = f"{study_name}_dataset.csv"
    urllib.request.urlretrieve(csv_url, local_csv_path)
    
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    from Model_Training.ML_Pipeline.PrepareDataset import prepare_datasets
    from Model_Training.ML_Pipeline.InputState import InputStateBuilder
    from Model_Training.OptunaOptimizer.MLP import create_objective
    from optuna.storages import JournalStorage, JournalRedisStorage

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    # Celery wala '?ssl_cert_reqs=CERT_NONE' flag hata kar clean URL banate hain
    clean_redis_url = redis_url.split('?')[0]  
    storage = JournalStorage(JournalRedisStorage(clean_redis_url))
    
    study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", load_if_exists=True)
    
    state_builder = InputStateBuilder(api_key=api_key)
    input_state = state_builder.build(
        dataset_path=local_csv_path,  # Fix: Used local_csv_path here
        use_case="dummy", 
        user_text="dummy", 
        target_col=target_col
    )
    
    input_state["search_space"] = final_space
    
    # Fix: Used local_csv_path here too
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_datasets(local_csv_path, target_col) 
    
    study.optimize(
        create_objective(X_train, y_train, X_val, y_val, input_state, final_space), 
        n_trials=n_trials,
        n_jobs=n_workers  
    )
    
    # Cleanup: (Optional but recommended) Delete the file after training to save space
    if os.path.exists(local_csv_path):
        os.remove(local_csv_path)
        
    return f"Completed {n_trials} trials using {n_workers} workers"


from celery import shared_task
import time
import json
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

@shared_task(name="finalize_model_training_task")
def finalize_model_training_task(results, study_name, incoming_path, target_col, submission_id):
    """TASK B: Final Training, Feature Importance, and Saving"""
    logger.info(f"\n[{submission_id}] >>> STARTING FINALIZATION TASK <<<")
    start_time = time.time()
    
    import os  
    import urllib.request
    import shutil
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from bson import ObjectId
    
    # --- OOM PROTECTION ---
    torch.set_num_threads(1)
    
    local_csv_path = f"{study_name}_final_dataset.csv"
    incoming_path_str = str(incoming_path).strip()
    
    try:
        if incoming_path_str.startswith("http"):
            logger.info("Detected Web URL. Downloading...")
            urllib.request.urlretrieve(incoming_path_str, local_csv_path)
        else:
            logger.info("Detected Local Path.")
            if os.path.exists(incoming_path_str):
                shutil.copy(incoming_path_str, local_csv_path)
            else:
                logger.warning("Local file missing! Rescuing via DB URL...")
                load_dotenv(override=True)
                client = MongoClient(os.getenv("MONGODB_URL"))
                sub = client["modelsmith"]["submissions"].find_one({"_id": ObjectId(submission_id)})
                client.close()
                db_url = sub.get("dataset_url", "")
                if db_url.startswith("http"):
                    urllib.request.urlretrieve(db_url, local_csv_path)
                else:
                    return "FAILED AT DOWNLOAD"
    except Exception as e:
        logger.error(f"[FATAL ERROR] Path Resolver Crashed: {e}")
        return "FAILED AT DOWNLOAD"
        
    # --- Yahan se aapka code normal (Step 1) chalu hoga ---
    # (Sirf dhyaan rakhna har jagah local_csv_path use ho)
    
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    from Model_Training.OptunaOptimizer.Train import train_and_evaluate_final_model
    from Model_Training.ML_Pipeline.InputState import InputStateBuilder
    from Model_Training.ML_Pipeline.PrepareDataset import prepare_datasets
    import optuna
    from optuna.storages import JournalStorage, JournalRedisStorage

    logger.info("[STEP 1/6] Connecting to Optuna Redis Storage to fetch best parameters...")
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        clean_redis_url = redis_url.split('?')[0]
        storage = JournalStorage(JournalRedisStorage(clean_redis_url))
        
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params
        logger.info(f"Best parameters fetched successfully: {len(best_params)} keys found.")
    except Exception as e:
        logger.error(f"[FATAL ERROR] Step 1 Failed - Could not load study: {e}")
        return "FAILED AT STEP 1"
    
    logger.info("[STEP 2/6] Building Input State and Preparing Datasets...")
    try:
        state_builder = InputStateBuilder(api_key=api_key)
        input_state = state_builder.build(
            dataset_path=local_csv_path, # <-- FIXED PATH
            use_case="dummy", 
            user_text="dummy", 
            target_col=target_col
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_datasets(local_csv_path, target_col) # <-- FIXED PATH
        logger.info(f"Data ready. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    except Exception as e:
        logger.error(f"[FATAL ERROR] Step 2 Failed - Data prep issue: {e}")
        return "FAILED AT STEP 2"
    
    logger.info("[STEP 3/6] Running Final Model Training (This may take a minute)...")
    try:
        model, final_score, metrics = train_and_evaluate_final_model(
            best_params, X_train, y_train, X_test, y_test, input_state,
            submission_id=submission_id
        )
        logger.info("Final model training completed successfully.")
    except Exception as e:
        logger.error(f"[FATAL ERROR] Step 3 Failed - Training crashed: {e}")
        return "FAILED AT STEP 3"
    
    logger.info("[STEP 4/6] Calculating Feature Dependencies...")
    feature_importance_dict = {}
    try:
        import pandas as pd  
        model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            preds_baseline = model(X_test_tensor)
            base_mse = criterion(preds_baseline, y_test_tensor.view_as(preds_baseline)).item()
            
            feat_names = input_state.get('preprocessed_feature_names', [])
            if not feat_names:
                try:
                    df_temp = pd.read_csv(local_csv_path) # <-- FIXED PATH (This would have crashed!)
                    cols = [c for c in df_temp.columns if c != target_col]
                    feat_names = cols if len(cols) == X_test.shape[1] else [f"Feature_{i}" for i in range(X_test.shape[1])]
                except:
                    feat_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
            
            for i, col_name in enumerate(feat_names):
                mse_drops = []
                for _ in range(3):
                    X_test_permuted = X_test.copy()
                    np.random.shuffle(X_test_permuted[:, i])
                    X_permuted_tensor = torch.FloatTensor(X_test_permuted)
                    preds_permuted = model(X_permuted_tensor)
                    permuted_mse = criterion(preds_permuted, y_test_tensor.view_as(preds_permuted)).item()
                    mse_drops.append(max(0, permuted_mse - base_mse)) 
                
                feature_importance_dict[col_name] = np.mean(mse_drops)

            max_val = max(feature_importance_dict.values()) if feature_importance_dict.values() else 1.0
            for k, v in feature_importance_dict.items():
                feature_importance_dict[k] = round(float(v) / float(max_val), 4)
            
            feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
            logger.info("Feature importance calculated successfully.")
    except Exception as e:
        logger.error(f"[WARNING] Step 4 Error - Failed to calculate feature importance: {e}")
        
    
    total_time_seconds = time.time() - start_time
    device = "CUDA/GPU" if torch.cuda.is_available() else "CPU"
    epochs_run = best_params.get("epochs", 1)
    time_per_epoch = total_time_seconds / epochs_run if epochs_run > 0 else total_time_seconds
    
    try:
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            preds_scaled = model(X_test_tensor).numpy()
            preds_original = scaler_y.inverse_transform(preds_scaled).flatten().tolist()
            actuals_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten().tolist()
    except Exception as e:
        logger.error(f"[WARNING] Could not generate scatter plot data: {e}")
        preds_original, actuals_original = [], []

    logger.info("[STEP 5/6] Generating Executive Insights (Safe REST API Call to Gemini)...")
    try:
        final_r2 = metrics.get("test_r2s", [])[-1] if metrics.get("test_r2s") else 0
        r2_pct = max(0, final_r2 * 100)
        final_rmse = metrics.get("test_rmses", [])[-1] if metrics.get("test_rmses") else 0
        
        top_features = list(feature_importance_dict.items())[:3]
        feat_str = ", ".join([f"{k} ({v*100:.0f}%)" for k, v in top_features])
        
        use_case = input_state.get("use_case", "Predicting target variable based on provided dataset.")
        
        prompt = f"""
        You are a Chief Data Scientist presenting a model's results to non-technical Business Stakeholders.
        Based STRICTLY on the facts provided below, write a 150-word Executive Summary. 
        DO NOT hallucinate, DO NOT add external knowledge, and DO NOT use heavy technical jargon. Make it simple for a layman to understand.

        --- FACTUAL DATA ---
        Target Variable Predicted: '{target_col}'
        Original Business Goal: {use_case}
        Model Reliability Score (R2): {r2_pct:.1f}%
        Error Margin (RMSE): {final_rmse:.4f}
        Top Predictive Factors: {feat_str}
        --------------------

        Write the summary following this structure seamlessly:
        1. Business Impact: What is the model predicting and how does it help the business goal?
        2. Key Drivers: Explain the top predictive factors in simple terms.
        3. Verdict: Based on the {r2_pct:.1f}% reliability score, state clearly if this model is ready to support business decisions.

        Output MUST be valid JSON only in this format: {{ "llm_summary": "Your paragraph here" }}
        """

        import requests
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing in environment variables.")

        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        resp = requests.post(gemini_url, json=payload, timeout=15)
        resp.raise_for_status()
        
        resp_data = resp.json()
        raw_text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        s_idx, e_idx = raw_text.find('{'), raw_text.rfind('}') + 1
        if s_idx != -1 and e_idx != 0:
            llm_json = json.loads(raw_text[s_idx:e_idx])
            llm_insights_paragraph = llm_json.get("llm_summary", "Insights generated but parsing failed.")
        else:
            llm_insights_paragraph = raw_text
            
        logger.info("✅ Brilliant Insights generated via safe REST call.")
        
    except Exception as e:
        logger.error(f"[WARNING] Safe Gemini call failed or timed out: {e}")
        top_feat_fallback = list(feature_importance_dict.items())[:2]
        fallback_str = ", ".join([f"{k}" for k, v in top_feat_fallback])
        llm_insights_paragraph = (
            f"Model execution finished with a production variance reliability score of {r2_pct:.1f}%. "
            f"Feature dependency evaluation identified '{fallback_str}' as primary drivers for target predictions. "
            f"The model is saved and ready for deployment."
        )

    logger.info("[STEP 6/6] Saving Config to Disk and Syncing to MongoDB...")
    model_save_path = ""
    config_save_path = ""
    try:
        safe_name = target_col.replace(" ", "_").lower()
        model_save_path = f"{safe_name}_best_model.pth"
        config_save_path = f"{safe_name}_model_config.json"
        
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model weights saved to {model_save_path}")
        
        deployment_config = {
            "input_state": input_state,
            "best_params": best_params,
            "final_score": float(final_score),
            "metrics": metrics,
            "execution_stats": {
                "total_time_seconds": round(total_time_seconds, 2),
                "time_per_epoch_seconds": round(time_per_epoch, 4),
                "hardware_used": device
            },
            "scatter_plot_data": {
                "actual_targets": actuals_original[:500], 
                "predicted_targets": preds_original[:500]
            },
            "feature_importance": feature_importance_dict,
            "llm_executive_summary": llm_insights_paragraph
        }
        
        with open(config_save_path, "w") as f:
            json.dump(deployment_config, f, indent=4)
        logger.info(f"Config JSON saved to {config_save_path}")
            
        logger.info("[STEP 6/6] Pushing updates to MongoDB Atlas (Synchronous Mode)...")
        from pymongo import MongoClient
        from bson import ObjectId
        
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        load_dotenv(os.path.join(root_dir, ".env"), override=True)
        
        mongo_uri = os.getenv("MONGODB_URL") 
        
        if not mongo_uri:
            logger.error(" CRITICAL: MONGODB_URL not found! Celery is blind.")
            return "FAILED - NO DB URL"
            
        client = MongoClient(mongo_uri)
        db = client["modelsmith"] 
            
        with open(model_save_path, "rb") as f:
            model_bytes = f.read()
            
        result = db["submissions"].update_one(
            {"_id": ObjectId(submission_id)},
            {"$set": {
                "status": "completed",
                "model_artifact": model_bytes,
                "model_config_json": deployment_config
            }}
        )
        
        if result.matched_count == 0:
            logger.error(f" CRITICAL ALERT: Connected to Atlas 'modelsmith', but ID {submission_id} NOT FOUND!")
        else:
            logger.info("✅ BINGO! MongoDB Status successfully updated to 'completed'!")
            
        client.close()
        logger.info(f"\n[{submission_id}] <<< PIPELINE COMPLETION SUCCESSFUL >>>")
        
    except Exception as e:
        logger.error(f"[FATAL ERROR] Step 6 Failed - Storage or DB sync crashed: {e}")
        return "FAILED AT STEP 6"

    # --- CLOUD DISK CLEANUP (Prevents Free Tier Disk Full Error) ---
    finally:
        logger.info("Cleaning up temporary local artifacts to save disk space...")
        try:
            if os.path.exists(local_csv_path): os.remove(local_csv_path)
            if model_save_path and os.path.exists(model_save_path): os.remove(model_save_path)
            if config_save_path and os.path.exists(config_save_path): os.remove(config_save_path)
            logger.info("Cleanup successful. Ready for next task.")
        except Exception as cleanup_err:
            logger.error(f"Could not delete some local files: {cleanup_err}")
            
    return "SUCCESS: Pipeline Complete"