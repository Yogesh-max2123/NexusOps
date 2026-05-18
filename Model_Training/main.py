import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import os
import json
import torch
import optuna
import logging
from dotenv import load_dotenv
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("modelsmith")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ML_Pipeline.InputState import InputStateBuilder
from ML_Pipeline.ConstraintEngine import run_constraint_engine
from ML_Pipeline.PrepareDataset import prepare_datasets
from OptunaOptimizer.MLP import create_objective
from OptunaOptimizer.Train import train_and_evaluate_final_model
from OptunaOptimizer.SaveModel import test_saved_model

mlops_tracker = None
MLOPS_AVAILABLE = False

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.services.mlops_tracker import MLOpsTracker
    MLOPS_AVAILABLE = True
except:
    pass

load_dotenv()

def start_automl(csv_path, target, use_case, user_req, submission_id: str = None):
    """AutoML pipeline with metrics tracking"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(" GEMINI_API_KEY not found!")
        return

    mlops_tracker = None
    if MLOPS_AVAILABLE and submission_id:
        try:
            mlops_tracker = MLOpsTracker(submission_id, f"ModelSmith_{submission_id}")
            mlops_tracker.start_run(tags={
                "target": target,
                "use_case": use_case,
                "requirement": user_req
            })
            logger.info(f" MLflow: {mlops_tracker.get_run_url()}\n")
        except Exception as e:
            logger.warning(f"Could not init MLflow: {e}")

    logger.info(f"\n{'='*70}")
    logger.info(f" STARTING AUTOML PIPELINE")
    logger.info(f"{'='*70}\n")

    try:
        # PHASE 1
        logger.info("1️  PHASE 1: Input State Builder")
        builder = InputStateBuilder(api_key=api_key)
        input_state = builder.build(csv_path, use_case, user_req, target)
        
        logger.info(f"    Dataset: {input_state['dataset_meta']['num_samples']} samples\n")
        
        if mlops_tracker:
            try:
                mlops_tracker.log_input_state(input_state)
            except:
                pass

        # PHASE 2
        logger.info(" PHASE 2: Constraint Engine")
        final_space = run_constraint_engine(input_state, api_key)
        logger.info(f"   ✅ Search space generated\n")

        # PHASE 3
        logger.info(" PHASE 3: Data Preparation")
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = prepare_datasets(csv_path, target)
        logger.info(f"   ✅ Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}\n")

        # PHASE 4
        logger.info(" PHASE 4: Optuna Optimization")
        objective = create_objective(X_train, y_train, X_val, y_val, input_state, final_space)
        study = optuna.create_study(direction=input_state["objective"]["optuna_direction"])
        
        def trial_callback(study, trial):
            if mlops_tracker:
                try:
                    mlops_tracker.log_optuna_trial(
                        trial.number,
                        trial.value if trial.value else float('inf'),
                        trial.params,
                        trial.duration.total_seconds() if trial.duration else 0
                    )
                except:
                    pass
        
        study.optimize(objective, n_trials=10, callbacks=[trial_callback])
        logger.info(f" Best Trial: #{study.best_trial.number}, Value: {study.best_value:.4f}\n")

        # PHASE 5 - Training with metrics collection
        logger.info(" PHASE 5: Final Model Training")
        
        model, score, metrics = train_and_evaluate_final_model(
            study.best_params, X_train, y_train, X_test, y_test, input_state,
            submission_id=submission_id,
            mlops_tracker=mlops_tracker
        )

        # PHASE 6
        logger.info(" PHASE 6: Exporting Artifacts")
        safe_name = target.replace(" ", "_").lower()
        model_save_path = f"{safe_name}_best_model.pth"
        config_save_path = f"{safe_name}_model_config.json"

        torch.save(model.state_dict(), model_save_path)
        
        
        deployment_config = {
            "input_state": input_state,
            "best_params": study.best_params,
            "final_score": float(score),
            "metrics": metrics  
        }
        
        with open(config_save_path, "w") as f:
            json.dump(deployment_config, f, indent=4)
        
        logger.info(f"   ✅ Model saved: {model_save_path}")
        logger.info(f"   ✅ Config saved with {len(metrics.get('epochs', []))} epochs of metrics\n")
        
        if mlops_tracker:
            try:
                mlops_tracker.log_final_metrics({
                    "final_rmse": score,
                    "total_epochs": len(metrics.get("epochs", []))
                })
                mlops_tracker.log_model_weights(model, "best_model")
                mlops_tracker.end_run()
            except:
                pass

        # PHASE 7
        logger.info("7️⃣  PHASE 7: Model Verification")
        _ = test_saved_model(model_save_path, study.best_params, input_state, X_test, y_test)
        logger.info("   ✅ Verification passed\n")

        logger.info(f"{'='*70}")
        logger.info(f"🎉 PIPELINE COMPLETED - {len(metrics.get('epochs', []))} epochs recorded")
        logger.info(f"{'='*70}\n")

        return {
            "status": "success",
            "model_path": model_save_path,
            "config_path": config_save_path,
            "rmse": score,
            "metrics_recorded": len(metrics.get("epochs", []))
        }

    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        if mlops_tracker:
            try:
                mlops_tracker.end_run(status="FAILED")
            except:
                pass
        raise

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML Pipeline")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--use_case", type=str, required=True)
    parser.add_argument("--req", type=str, required=True)
    parser.add_argument("--submission_id", type=str, required=False)
    
    args = parser.parse_args()
    
    start_automl(
        csv_path=args.csv_path,
        target=args.target,
        use_case=args.use_case,
        user_req=args.req,
        submission_id=args.submission_id
    )