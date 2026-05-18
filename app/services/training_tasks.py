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

@shared_task(name="run_optuna_trials_task")
def run_optuna_trials_task(study_name, csv_path, target_col, final_space, n_trials=5, n_workers=1): 
    """TASK A: Distributed Optuna Worker"""
    import os
    from dotenv import load_dotenv
    
     
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    from Model_Training.ML_Pipeline.PrepareDataset import prepare_datasets
    from Model_Training.ML_Pipeline.InputState import InputStateBuilder
    from Model_Training.OptunaOptimizer.MLP import create_objective
    from optuna.storages import JournalStorage, JournalRedisStorage

    import os
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    storage = JournalStorage(JournalRedisStorage(redis_url))
    
    
    study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", load_if_exists=True)
    
    state_builder = InputStateBuilder(api_key=api_key)
    input_state = state_builder.build(
        dataset_path=csv_path, 
        use_case="dummy", 
        user_text="dummy", 
        target_col=target_col
    )
    
    input_state["search_space"] = final_space
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_datasets(csv_path, target_col)
    
    
    study.optimize(
        create_objective(X_train, y_train, X_val, y_val, input_state, final_space), 
        n_trials=n_trials,
        n_jobs=n_workers  
    )
    return f"Completed {n_trials} trials using {n_workers} workers"


# TASK B: FINALIZE TRAINING & GENERATE AI INSIGHTS
@shared_task(name="finalize_model_training_task")
def finalize_model_training_task(results, study_name, csv_path, target_col, submission_id):
    """TASK B: Final Training, Feature Importance, and Saving"""
    logger.info(f"\n[{submission_id}] >>> STARTING FINALIZATION TASK <<<")
    start_time = time.time()
    
    import os  
    from dotenv import load_dotenv
    
    
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")

    from Model_Training.OptunaOptimizer.Train import train_and_evaluate_final_model
    from Model_Training.ML_Pipeline.InputState import InputStateBuilder
    from Model_Training.ML_Pipeline.PrepareDataset import prepare_datasets
    from optuna.storages import JournalStorage, JournalRedisStorage

    logger.info("[STEP 1/6] Connecting to Optuna Redis Storage to fetch best parameters...")
    try:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        storage = JournalStorage(JournalRedisStorage(redis_url))
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
            dataset_path=csv_path, 
            use_case="dummy", 
            user_text="dummy", 
            target_col=target_col
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_datasets(csv_path, target_col)
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
                    df_temp = pd.read_csv(csv_path)
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
        import json
        import os
        
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
        import os
        from dotenv import load_dotenv
        
    
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
        return "SUCCESS: Pipeline Complete"
    except Exception as e:
        logger.error(f"[FATAL ERROR] Step 6 Failed - Storage or DB sync crashed: {e}")
        return "FAILED AT STEP 6"