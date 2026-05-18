import os
import uuid
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class InputStateBuilder:
    
    GEMINI_MODEL = "gemini-2.5-flash"
    SCHEMA_VERSION = "1.2"

    # MODULE 1: INPUT STATE BUILDER

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY")
        genai.configure(api_key=key.strip())
        self.model = genai.GenerativeModel(self.GEMINI_MODEL)

    # DATE COLUMN DETECTION

    def _is_date_column(self, series: pd.Series) -> bool:
        """Detect if a column is a date/datetime string."""
        if pd.api.types.is_numeric_dtype(series):
            return False
        sample = series.dropna().head(20)
        try:
            pd.to_datetime(sample, format='mixed')
            return True
        except (ValueError, TypeError, Exception):
            return False

    def _expand_date_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Replace a date column with numeric year/month/day/weekday features."""
        dt = pd.to_datetime(df[col], format='mixed', errors="coerce")
        df = df.drop(columns=[col])
        df.insert(0, f"{col}_year", dt.dt.year)
        df.insert(1, f"{col}_month", dt.dt.month)
        df.insert(2, f"{col}_day", dt.dt.day)
        df.insert(3, f"{col}_weekday", dt.dt.weekday)
        return df

    # DATA ANALYSIS

    def _extract_data_facts(self, path: str, target_col: Optional[str] = None) -> Dict[str, Any]:

        if not os.path.isfile(path):
            raise ValueError(f"Dataset not found: {path}")

        df = pd.read_csv(path)

        date_cols_found = []
        for col in df.columns:
            if col == (target_col or df.columns[-1]):
                continue
            if self._is_date_column(df[col]):
                date_cols_found.append(col)
                df = self._expand_date_features(df, col)
                
        
        cat_cols = [c for c in df.columns if c != (target_col or df.columns[-1]) and not pd.api.types.is_numeric_dtype(df[c])]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

       
        df.columns = df.columns.str.strip()
        
        if target_col:
            target_col = target_col.strip()
            
        
        if target_col:
            if target_col not in df.columns:
                
                print(f"\nDEBUG: Actual columns in CSV are: {df.columns.tolist()}\n")
                raise ValueError(f"Target column '{target_col}' not found. Available cols: {df.columns.tolist()}")
            
            
            cols = [c for c in df.columns if c != target_col] + [target_col]
            df = df[cols]

        target = df.iloc[:, -1]
        feature_df = df.iloc[:, :-1]

        n_samples = len(df)
        n_features = feature_df.shape[1]

        
        if target.dtype == object:
            target_type = "categorical"
            n_classes = int(target.nunique())
        elif np.issubdtype(target.dtype, np.integer) and target.nunique() <= 20:
            target_type = "categorical"
            n_classes = int(target.nunique())
        else:
            target_type = "continuous"
            n_classes = 1

        
        col_missing = df.isnull().mean()
        missing_ratio = float(col_missing.mean())
        severely_missing = col_missing[col_missing > 0.5].to_dict()
        n_severely_missing = len([c for c in severely_missing if c != df.columns[-1]])

        
        numeric_features = feature_df.select_dtypes(include=[np.number])
        outlier_ratio = 0.0
        if not numeric_features.empty:
            std = numeric_features.std(ddof=0).replace(0, np.nan)
            z = np.abs((numeric_features - numeric_features.mean()) / std)
            outlier_ratio = float((z > 3).sum().sum() / z.size)

       
        categorical_cols_count = feature_df.select_dtypes(exclude=[np.number]).shape[1]
        continuous_cols_count = n_features - categorical_cols_count

        
        corr_matrix = None
        corr_level = "low"
        if not numeric_features.empty:
            corr_matrix = numeric_features.corr().abs()
            corr_score = float(corr_matrix.mean().mean())
            corr_level = "high" if corr_score > 0.7 else ("medium" if corr_score > 0.4 else "low")

       
        noise_level = "low"
        if not numeric_features.empty:
            cv = (numeric_features.std() / numeric_features.mean().replace(0, np.nan)).abs().dropna()
            mean_cv = float(cv.mean()) if not cv.empty else 0.0
            noise_level = "high" if mean_cv > 1 else ("medium" if mean_cv > 0.5 else "low")

       
        is_balanced = True
        if target_type == "categorical":
            counts = target.value_counts(normalize=True)
            is_balanced = bool(counts.min() > 0.1)

        
        complexity = "high" if n_samples > 100_000 else ("medium" if n_samples > 20_000 else "low")

        return {
            "data_type": "tabular",
            "num_samples": n_samples,
            "num_features": n_features,
            "num_classes": n_classes,
            "input_shape": [n_features],
            "output_shape": [n_classes],
            "missing_ratio": round(missing_ratio, 4),
            "n_severely_missing_features": n_severely_missing,
            "outlier_ratio": round(outlier_ratio, 4),
            "target_type": target_type,
            "is_balanced": is_balanced,
            "feature_type_breakdown": {
                "continuous": continuous_cols_count,
                "categorical": categorical_cols_count,
                "date_cols_expanded": date_cols_found,
            },
            "correlation_structure": corr_level,
            "noise_level": noise_level,
            "complexity_tier": complexity,
        }

    # MODEL SPEC

    def _derive_model_spec(self, problem_type: str, meta: Dict) -> Dict:
        if problem_type == "regression":
            return {"output_dim": 1, "activation": "linear", "loss": "mse"}
        if meta["num_classes"] <= 2:
            return {"output_dim": 1, "activation": "sigmoid", "loss": "binary_crossentropy"}
        return {
            "output_dim": meta["num_classes"],
            "activation": "softmax",
            "loss": "categorical_crossentropy",
        }

    # LLM CALL (USER INTENT)

    def _call_llm(self, use_case: str, user_text: str) -> Dict:
        prompt = f"""
    You are an ML system parser.

    Extract ONLY user intent.

    INPUT:
    USE_CASE: {use_case}
    USER: {user_text}

    OUTPUT JSON (no extra text, no markdown, strict JSON only):

    {{
    "constraints": {{
        "latency": "low | medium | high",
        "training_time": "short | medium | long",
        "compute": "cpu | gpu | any"
    }},
    "objective": {{
        "priority": "high_accuracy | low_latency | balanced"
    }},
    "preferences": {{
        "model_complexity": "low | medium | high",
        "interpretability": "required | not_required"
    }}
    }}
    """
        try:
            res = self.model.generate_content(prompt)
            text = res.text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return {
                "constraints": {"latency": "medium", "training_time": "medium", "compute": "cpu"},
                "objective": {"priority": "balanced"},
                "preferences": {"model_complexity": "medium", "interpretability": "not_required"},
            }

    # FINAL BUILD

    def build(
        self,
        dataset_path: str,
        use_case: str,
        user_text: str,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:

        data = self._extract_data_facts(dataset_path, target_col=target_col)

        problem_type = (
            "classification"
            if data["target_type"] == "categorical" and data["num_classes"] > 1
            else "regression"
        )

        llm = self._call_llm(use_case, user_text)

        if problem_type == "classification":
            metric = "f1_macro" if not data["is_balanced"] else "accuracy"
            direction = "maximize"
        else:
            metric = "rmse"
            direction = "minimize"

        model_spec = self._derive_model_spec(problem_type, data)

        return {
            "schema_version": self.SCHEMA_VERSION,
            "request_id": str(uuid.uuid4()),
            "random_seed": 42,
            "problem_type": problem_type,
            "data_type": data["data_type"],
            "dataset_meta": data,
            "model_spec": model_spec,
            "constraints": {
                **llm["constraints"],
                "max_inference_ms": None,
                "max_training_epochs": None,
                "deployment_target": "server_gpu" if llm["constraints"]["compute"] == "gpu" else "server_cpu",
                "max_model_size_mb": 1000.0,
                "memory_budget_gb": None,
            },
            "objective": {
                "priority": llm["objective"]["priority"],
                "primary_metric": metric,
                "secondary_metric": None,
                "optuna_direction": direction,
                "min_acceptable_score": None,
            },
            "preferences": llm["preferences"],
            "arch_hints": {
                "preferred_optimizer": None,
                "preferred_loss": None,
                "arch_family_override": None,
            },
            "data_expectations": {
                "handle_noise": data["noise_level"] != "low",
                "handle_imbalance": not data["is_balanced"],
                "handle_high_missing": data["n_severely_missing_features"] > 0,
            },
        }