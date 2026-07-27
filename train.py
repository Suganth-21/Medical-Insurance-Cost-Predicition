import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_and_evaluate():
    print("🚀 Loading preprocessed dataset arrays...")
    
    # 1. Load preprocessed numpy arrays
    try:
        X_train = np.load("data/processed/X_train.npy")
        X_test = np.load("data/processed/X_test.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("❌ Error: Processed data arrays not found! Make sure you ran 'python main_preprocessing.py' first.")
        return

    print(f"Dataset shape -> Training set: {X_train.shape}, Test set: {X_test.shape}\n")

    # 2. Define the models specified in your presentation
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    best_model = None
    best_r2 = -float("inf")
    best_model_name = ""

    print("📊 Training and Evaluating Models...\n" + "="*50)

    # 3. Train and evaluate each model
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics specified in presentation (MAE, RMSE, R²)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        
        print(f"🔹 {name}")
        print(f"   - MAE:      ${mae:,.2f}")
        print(f"   - RMSE:     ${rmse:,.2f}")
        print(f"   - R² Score: {r2:.4f}\n")
        
        # Track the best performing model based on R² Score
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    # 4. Display Comparison Table
    results_df = pd.DataFrame(results)
    print("="*50)
    print("🏆 FINAL MODEL PERFORMANCE COMPARISON:")
    print(results_df.to_string(index=False))
    print("="*50)
    print(f"\n🌟 Best Model: {best_model_name} with R² Score of {best_r2:.4f}")

    # 5. Save the best model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved successfully to '{model_path}'!")

if __name__ == "__main__":
    train_and_evaluate()





























    