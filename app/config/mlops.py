import mlflow
import os
from pathlib import Path

class MLOpsConfig:
    """MLOps configuration for tracking experiments"""
    
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    MLFLOW_BACKEND_STORE = os.getenv("MLFLOW_BACKEND_STORE", "./mlruns")
    MLFLOW_ARTIFACT_STORE = os.getenv("MLFLOW_ARTIFACT_STORE", "./mlartifacts")
    
    Path(MLFLOW_BACKEND_STORE).mkdir(parents=True, exist_ok=True)
    Path(MLFLOW_ARTIFACT_STORE).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def setup_mlflow():
        """Initialize MLflow"""
        mlflow.set_tracking_uri(MLOpsConfig.MLFLOW_TRACKING_URI)
        mlflow.set_artifact_uri(MLOpsConfig.MLFLOW_ARTIFACT_STORE)
        mlflow.pytorch.autolog()
        return mlflow
    
mlflow_client = MLOpsConfig.setup_mlflow()