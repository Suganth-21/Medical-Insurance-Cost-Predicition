import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def run_preprocessing():
    print("Starting advanced data preprocessing pipeline (54 columns)...")
    
    # 1. Load the raw data
    file_path = "data/raw/medical_insurance.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded massive dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    except FileNotFoundError:
        print(f"❌ Error: Cannot find {file_path}. Check your data folder!")
        return
        
    # 2. Clean the data (Drop duplicates and person_id)
    df = df.drop_duplicates()
    if 'person_id' in df.columns:
        df = df.drop(columns=['person_id']) # Drop ID as it has no predictive value
    
    # 3. Separate Features (X) and Target (y)
    target_col = "annual_medical_cost"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 4. Automatically identify Numerical and Categorical columns
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Detected {len(num_features)} numerical features and {len(cat_features)} categorical features.")

    # 5. Build Professional MLOps Pipelines for Transformation
    # For numbers: Fill missing with Median -> Scale
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # For categories/text: Fill missing with most frequent -> One-Hot Encode
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
        ]
    )
    
    # 6. Fit and Transform the data
    print("Imputing, encoding, and scaling features...")
    X_processed = preprocessor.fit_transform(X)
    
    # 7. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # 8. Save Processed Data
    os.makedirs("data/processed", exist_ok=True)
    
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)
    
    # Save the pipeline object for web deployment
    joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
    
    print("✅ Preprocessing complete! Train/Test arrays and preprocessor.pkl saved to data/processed/")

if __name__ == "__main__":
    run_preprocessing()