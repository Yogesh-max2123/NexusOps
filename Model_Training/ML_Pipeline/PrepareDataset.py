import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_datasets(dataset_path: str, target_col: str):
    """
    Loads the dataset, handles basic preprocessing (like date expansion),
    and splits the data into Train, Validation, and Test sets.
    """
    print(f"Loading data from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # 1. Expand Date Features (with updated dtype check)
    for col in df.columns:
        if col == target_col:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_datetime(df[col].dropna().head(), format='mixed')
                dt = pd.to_datetime(df[col], format='mixed', errors="coerce")
                df = df.drop(columns=[col])
                df.insert(0, f"{col}_year", dt.dt.year)
                df.insert(1, f"{col}_month", dt.dt.month)
                df.insert(2, f"{col}_day", dt.dt.day)
                df.insert(3, f"{col}_weekday", dt.dt.weekday)
            except (ValueError, TypeError, Exception):
                pass
                
    # 2. Convert remaining text/categorical columns to numbers (Foolproof check)
    cat_cols = [c for c in df.columns if c != target_col and not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # 3. Fill Missing Values
    df = df.fillna(df.median(numeric_only=True))

    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # 5. Perform the 3-way Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_val = scaler.transform(X_val)         
    X_test = scaler.transform(X_test) 

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()      

    print(f"Data Split Complete!")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape:   {X_val.shape}")
    print(f"  Test shape:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y