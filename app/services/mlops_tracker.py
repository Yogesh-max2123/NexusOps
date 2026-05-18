import mlflow
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger("modelsmith")

class MLOpsTracker:
    """Track model training with MLflow"""
    
    def __init__(self, submission_id: str, experiment_name: str):
        self.submission_id = submission_id
        self.experiment_name = experiment_name
        self.run_id = None
        self.run = None
        self.metrics_history = []
        
    def start_run(self, tags: Dict[str, str] = None):
        """Start MLflow experiment run"""
        
        logger.info(f" Starting MLflow tracking for {self.submission_id}")
        
        
        mlflow.set_experiment(self.experiment_name)
        
        self.run = mlflow.start_run()
        self.run_id = self.run.info.run_id
        
       
        default_tags = {
            "submission_id": self.submission_id,
            "timestamp": datetime.now().isoformat(),
            "platform": "ModelSmith"
        }
        
        if tags:
            default_tags.update(tags)
        
        mlflow.set_tags(default_tags)
        
        logger.info(f" MLflow run started: {self.run_id}")
        return self.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        
        
        loggable_params = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                loggable_params[key] = value
            else:
                loggable_params[key] = str(value)
        
        mlflow.log_params(loggable_params)
        logger.info(f" Logged {len(loggable_params)} parameters")
    
    def log_input_state(self, input_state: Dict):
        """Log dataset metadata"""
        
        metadata = {
            "num_samples": input_state["dataset_meta"]["num_samples"],
            "num_features": input_state["dataset_meta"]["num_features"],
            "problem_type": input_state["problem_type"],
            "target_type": input_state["dataset_meta"]["target_type"],
            "missing_ratio": input_state["dataset_meta"]["missing_ratio"],
            "noise_level": input_state["dataset_meta"]["noise_level"],
        }
        
        mlflow.log_params(metadata)
        logger.info(f" Logged input state metadata")
    
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Log single metric"""
        
        mlflow.log_metric(metric_name, value, step=step)
        
        self.metrics_history.append({
            "metric": metric_name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_optuna_trial(self, trial_number: int, trial_value: float, 
                        trial_params: Dict, trial_duration: float):
        """Log Optuna trial metrics"""
        
        logger.info(f"📊 Logging Optuna Trial {trial_number}")
        
        
        mlflow.log_metric("optuna_trial_value", trial_value, step=trial_number)
        mlflow.log_metric("optuna_trial_duration", trial_duration, step=trial_number)
        
        
        for param_name, param_value in trial_params.items():
            if isinstance(param_value, (int, float, bool)):
                mlflow.log_metric(f"trial_{trial_number}_{param_name}", float(param_value))
    
    def log_epoch_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                         test_rmse: float = None):
        """Log training epoch metrics"""
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        if test_rmse is not None:
            mlflow.log_metric("test_rmse", test_rmse, step=epoch)
        
        logger.debug(f" Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    def log_final_metrics(self, metrics: Dict[str, float]):
        """Log final model metrics"""
        
        logger.info(f" Logging final metrics")
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        logger.info(f" Logged {len(metrics)} final metrics")
    
    def log_model_architecture(self, model: torch.nn.Module):
        """Log model architecture"""
        
        
        model_str = str(model)
        
        
        with open("model_architecture.txt", "w") as f:
            f.write(model_str)
        
        mlflow.log_artifact("model_architecture.txt")
        
        logger.info(f" Logged model architecture")
    
    def log_model_weights(self, model: torch.nn.Module, model_name: str):
        """Log model weights as artifact"""
        
        
        model_path = f"{model_name}_artifact.pth"
        torch.save(model.state_dict(), model_path)
        
        
        mlflow.log_artifact(model_path)
        
        logger.info(f" Logged model weights: {model_path}")
    
    def log_config_json(self, config_dict: Dict):
        """Log configuration JSON"""
        
        
        config_path = "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        
        
        mlflow.log_artifact(config_path)
        
        logger.info(f" Logged model configuration")
    
    def end_run(self, status: str = "FINISHED"):
        """End MLflow run"""
        
        mlflow.end_run()
        logger.info(f" MLflow run ended with status: {status}")
    
    def get_metrics_history(self):
        """Get all tracked metrics"""
        return self.metrics_history
    
    def get_run_url(self):
        """Get MLflow UI URL for this run"""
        
        if not self.run_id:
            return None
        
        import os
        experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        # Base URL nikalo (with fallback to localhost)
        base_mlflow_url = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").rstrip('/')
        
        # Dynamic URL return karo
        return f"{base_mlflow_url}/#/experiments/{experiment_id}/runs/{self.run_id}"


class MLOpsRemoteLogger:
    """Send metrics to remote tracking (for real-time UI updates)"""
    
    def __init__(self, submission_id: str):
        self.submission_id = submission_id
        self.metrics_buffer = []
    
    def buffer_metric(self, metric_name: str, value: float, step: int = None):
        """Buffer metric for batch sending"""
        
        self.metrics_buffer.append({
            "metric_name": metric_name,
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
    
    def flush_to_database(self, db_client):
        """Send buffered metrics to MongoDB"""
        
        if not self.metrics_buffer:
            return
        
        
        db_client["training_metrics"].insert_many(self.metrics_buffer)
        
        logger.info(f" Flushed {len(self.metrics_buffer)} metrics to database")
        self.metrics_buffer = []