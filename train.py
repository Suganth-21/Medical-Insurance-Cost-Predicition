import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
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

    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Medical_Insurance_Cost_Prediction")

    # 2. Define the models specified in your presentation
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Regularized Random Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=10, 
            min_samples_leaf=4, 
            random_state=42
        )
    }

    results = []
    best_model = None
    best_r2 = -float("inf")
    best_model_name = ""

    print("📊 Training and Evaluating Models...\n" + "="*50)

    # 3. Train and evaluate each model
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log params & metrics
            mlflow.log_param("model_type", name)
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
                
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_rmse", rmse)
            
            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")

            results.append({
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "Train R2": round(train_r2, 4),
                "Test R2": round(test_r2, 4)
            })
            
            print(f"🔹 {name}")
            print(f"   - MAE:      ${mae:,.2f}")
            print(f"   - RMSE:     ${rmse:,.2f}")
            print(f"   - Train R²: {train_r2:.4f}")
            print(f"   - Test R²:  {test_r2:.4f}\n")
            
            # Track the best performing model based on Test R² Score
            if test_r2 > best_r2:
                best_r2 = test_r2
                best_model = model
                best_model_name = name

    # 4. Display Comparison Table
    results_df = pd.DataFrame(results)
    print("="*50)
    print("🏆 FINAL MODEL PERFORMANCE COMPARISON:")
    print(results_df.to_string(index=False))
    print("="*50)
    print(f"\n🌟 Best Model: {best_model_name} with Test R² Score of {best_r2:.4f}")

    # 5. Save the best model artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved successfully to '{model_path}'!")

if __name__ == "__main__":
    train_and_evaluate()