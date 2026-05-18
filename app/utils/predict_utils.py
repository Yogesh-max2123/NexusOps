import os
import io
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
import torch
import tempfile  # ✅ ADD THIS
from sklearn.preprocessing import StandardScaler

# Ensure Model_Training is in sys.path so we can import DynamicMLP
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
model_training_path = os.path.join(project_root, "Model_Training")
if model_training_path not in sys.path:
    sys.path.append(model_training_path)

from OptunaOptimizer.MLP import DynamicMLP
import logging

logger = logging.getLogger("modelsmith")

def run_prediction_pipeline(submission_id: str, submission: dict, sample_bytes: bytes) -> dict:
    """
    Given the model artifacts and sample data bytes, run prediction:
    1. Downloads original dataset to train scalers based on its data distribution.
    2. Aligns the sample dataset.
    3. Runs inference.
    """
    target_col = submission["target_column"]
    model_bytes = submission["model_artifact"]
    config_dict = submission["model_config_json"]
    dataset_url = submission["dataset_url"]

    temp_dir = None
    local_orig = None
    
    try:
        logger.info(f"🔮 Starting prediction pipeline for submission: {submission_id}")
        
        # 1. Read sample dataset
        logger.info("   📝 Reading sample test CSV...")
        df_sample = pd.read_csv(io.BytesIO(sample_bytes))
        logger.info(f"   ✅ Sample data loaded: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns")
        
        # 2. Download original dataset
        # ✅ FIX: Use tempfile to get system-specific temp directory
        logger.info("   📥 Downloading original dataset from Cloudinary...")
        temp_dir = tempfile.gettempdir()  # ✅ Windows: C:\Users\...\AppData\Local\Temp
                                          # ✅ Linux/Mac: /tmp
        local_orig = os.path.join(temp_dir, f"{submission_id}_orig.csv")
        
        logger.info(f"   Temp dir: {temp_dir}")
        logger.info(f"   Saving to: {local_orig}")
        
        try:
            urllib.request.urlretrieve(dataset_url, local_orig)
            logger.info(f"   ✅ Original dataset downloaded")
        except Exception as e:
            logger.error(f"   ❌ Failed to download dataset: {str(e)}")
            raise
        
        if not os.path.exists(local_orig):
            raise FileNotFoundError(f"Downloaded file not found at: {local_orig}")
        
        logger.info(f"   ✅ File exists at: {local_orig}")
        df_orig = pd.read_csv(local_orig)
        logger.info(f"   ✅ Original data loaded: {df_orig.shape[0]} rows")

        # 3. Preprocess Original Dataset (same logic as PrepareDataset.py)
        logger.info("   🔧 Preprocessing original dataset...")
        df_orig_prep, X_orig_cols = _preprocess_features(df_orig, target_col, is_training=True)
        X_orig = df_orig_prep.drop(columns=[target_col], errors='ignore').values
        y_orig = df_orig[target_col].values
        logger.info(f"   ✅ Original data preprocessed: {X_orig.shape}")

        logger.info("   📊 Fitting scalers on original data distribution...")
        scaler_X = StandardScaler()
        scaler_X.fit(X_orig)
        
        scaler_y = StandardScaler()
        scaler_y.fit(y_orig.reshape(-1, 1))
        logger.info("   ✅ Scalers fitted")

        # 4. Preprocess Sample Dataset
        logger.info("   🔧 Preprocessing sample test data...")
        df_sample_prep, _ = _preprocess_features(df_sample, target_col, is_training=False, expected_cols=X_orig_cols)
        
        # Scale test data
        X_sample = df_sample_prep.values
        X_sample_scaled = scaler_X.transform(X_sample)
        logger.info(f"   ✅ Sample data preprocessed and scaled: {X_sample_scaled.shape}")

        # 5. Load Model Architecture & Weights
        logger.info("   🤖 Loading trained model...")
        input_dim = X_sample_scaled.shape[1]
        input_state = config_dict["input_state"]
        best_params = config_dict["best_params"]
        output_dim = input_state["model_spec"]["output_dim"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"   Using device: {device}")
        
        model = DynamicMLP(input_dim, output_dim, best_params)

        with io.BytesIO(model_bytes) as m_bytes:
            # Load weights safely
            logger.info("   📦 Loading model weights...")
            state_dict = torch.load(m_bytes, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logger.info("   ✅ Model weights loaded")

        model.to(device)
        model.eval()

        # 6. Run Inference
        logger.info("   🔮 Running inference on test data...")
        X_tensor = torch.FloatTensor(X_sample_scaled).to(device)
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy().flatten()
        
        # Inverse transform
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        logger.info(f"   ✅ Predictions generated: {len(predictions)} samples")

        # Remove temporary original file
        logger.info("   🧹 Cleaning up temporary files...")
        if os.path.exists(local_orig):
            os.remove(local_orig)
            logger.info(f"   ✅ Deleted temp file: {local_orig}")

        # Compute testing metrics if actual ground truth is available
        metrics = None
        
        # Try to find target column (case-insensitive)
        actual_target_col = None
        for col in df_sample.columns:
            if col.lower().strip() == target_col.lower().strip():
                actual_target_col = col
                break
                
        if actual_target_col:
            logger.info("   📊 Computing evaluation metrics...")
            actuals = df_sample[actual_target_col].values
            # Safe valid idx check across all dtypes
            valid_idx = ~pd.isna(actuals)
            if valid_idx.any():
                problem_type = config_dict.get("input_state", {}).get("problem_type", "regression")
                
                if problem_type == "classification":
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    preds_rounded = np.round(predictions[valid_idx])
                    actuals_valid = actuals[valid_idx]
                    
                    is_multiclass = len(np.unique(actuals_valid)) > 2
                    avg_type = "macro" if is_multiclass else "binary"
                    
                    try:
                        acc = float(accuracy_score(actuals_valid, preds_rounded))
                        prec = float(precision_score(actuals_valid, preds_rounded, average=avg_type, zero_division=0))
                        rec = float(recall_score(actuals_valid, preds_rounded, average=avg_type, zero_division=0))
                        f1 = float(f1_score(actuals_valid, preds_rounded, average=avg_type, zero_division=0))
                        metrics = {
                            "Accuracy": acc,
                            "Precision": prec,
                            "Recall": rec,
                            "F1 Score": f1
                        }
                        logger.info(f"   ✅ Classification metrics computed: {metrics}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Classification metrics failed: {str(e)}")
                        problem_type = "regression" # Fallback if classification metrics fail
                
                if problem_type == "regression":
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    rmse = float(np.sqrt(mean_squared_error(actuals[valid_idx], predictions[valid_idx])))
                    mae = float(mean_absolute_error(actuals[valid_idx], predictions[valid_idx]))
                    r2 = float(r2_score(actuals[valid_idx], predictions[valid_idx]))
                    metrics = {
                        "RMSE (Root Mean Sq Error)": rmse,
                        "MAE (Mean Absolute Error)": mae,
                        "R² (Coefficient of Det.)": r2
                    }
                    logger.info(f"   ✅ Regression metrics computed: {metrics}")

        # Attach predictions to new DF for easy CSV reporting
        df_sample[f"Predicted_{target_col}"] = predictions

        logger.info(f"✅ Prediction pipeline completed successfully!")
        
        # Return CSV string
        return {"csv_data": df_sample.to_csv(index=False), "metrics": metrics}
    
    except Exception as e:
        logger.error(f"❌ Prediction pipeline failed: {str(e)}", exc_info=True)
        
        # Cleanup on error
        if local_orig and os.path.exists(local_orig):
            try:
                os.remove(local_orig)
                logger.info(f"   Cleaned up temp file on error")
            except:
                pass
        
        raise


def _preprocess_features(df: pd.DataFrame, target_col: str, is_training: bool, expected_cols: list = None):
    """Preprocess features for prediction"""
    
    # Expand Date Features
    for col in list(df.columns):
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Same check as PrepareDataset.py
                pd.to_datetime(df[col].dropna().head(), format='mixed')
                dt = pd.to_datetime(df[col], format='mixed', errors="coerce")
                df = df.drop(columns=[col])
                df.insert(0, f"{col}_year", dt.dt.year)
                df.insert(1, f"{col}_month", dt.dt.month)
                df.insert(2, f"{col}_day", dt.dt.day)
                df.insert(3, f"{col}_weekday", dt.dt.weekday)
            except (ValueError, TypeError, Exception):
                pass
                
    # Dummies
    cat_cols = [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    if not is_training and target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Median filling
    df = df.fillna(df.median(numeric_only=True))

    if is_training:
        cols = [c for c in df.columns if c != target_col]
        return df, cols
    else:
        # Reindex to match training columns
        df = df.reindex(columns=expected_cols, fill_value=0)
        return df, None