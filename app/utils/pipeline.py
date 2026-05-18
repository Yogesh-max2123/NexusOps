import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SmartAutoPipeline:
    """Automated preprocessing pipeline."""
    
    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df.copy()
        self.target = target
        self.task_type = None

    def analyze(self):
        """Detect regression vs classification based on the target column."""
        if self.target and self.target in self.df.columns:
            target_dtype = self.df[self.target].dtype
            unique_vals = self.df[self.target].nunique()
            
            if pd.api.types.is_numeric_dtype(target_dtype) and unique_vals > 20:
                self.task_type = 'regression'
            else:
                self.task_type = 'classification'
        return self

    def clean_columns(self):
        """Normalize column names to lowercase with underscores."""
        self.df.columns = self.df.columns.str.lower().str.replace(r'\s+', '_', regex=True)
        if self.target:
            self.target = self.target.lower().replace(' ', '_')
        return self

    def handle_missing(self):
        """Imputes missing values: Median for numerical, Mode for categorical."""
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
                
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val[0])
        return self

    def encode(self):
        """Encodes categorical data (One-Hot for < 10 unique, Label for others). Skips target."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if self.target and col == self.target:
                continue
                
            unique_count = self.df[col].nunique()
            if unique_count < 10:
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
            else:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        return self

    def handle_outliers(self):
        """Uses IQR to cap outliers. Skips heavily skewed columns and the target."""
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.target and col == self.target:
                continue
                
            if abs(self.df[col].skew()) > 1.0:
                continue

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
        return self

    def scale(self):
        """Scales numerical data using StandardScaler. Skips target and encoded dummies."""
        cols_to_scale = [
            col for col in self.df.select_dtypes(include=[np.number]).columns
            if col != self.target and not set(self.df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0})
        ]
        
        if cols_to_scale:
            scaler = StandardScaler()
            self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
        return self

    def get_data(self) -> pd.DataFrame:
        """Returns the fully processed DataFrame."""
        return self.df
