# Medical Insurance Cost Prediction MLOps Project

## Overview
This is a comprehensive healthcare ML application designed to predict a patient's annual medical insurance cost based on their demographic and health profiles. It leverages a fully functional MLOps pipeline using `scikit-learn` and `MLflow`, integrated with a polished `FastAPI` + Vanilla JS/HTML professional web interface.

The project demonstrates a production-grade machine learning lifecycle: data ingestion, preprocessing, robust modeling (Regularized Random Forests), artifact storage, and an interactive prediction API.

## Architecture

1. **Frontend**: A polished, responsive web interface built with HTML, Tailwind CSS, and Vanilla JS (`frontend/`). It gathers logically grouped patient data and asynchronously queries the API.
2. **Backend**: A robust REST API built with `FastAPI` (`backend/`). It handles input validation using Pydantic, applies the exact preprocessing state used during training, executes the model prediction, and serves the static frontend UI.
3. **ML Pipeline**: A robust scikit-learn pipeline (`src/preprocessing/`) handles Missing Value Imputation (Median/Most Frequent), StandardScaler, and OneHotEncoding for 52 features.
4. **Model Tracking**: `MLflow` tracks model parameters, metrics (R², MAE, RMSE), and artifacts across experiments (`train.py`), allowing for simple versioning and debugging.

## Dataset
The dataset originally contains 54 columns. 
- **Identifier**: `person_id` (Excluded from modeling)
- **Target**: `annual_medical_cost`
- **Features**: 52 numerical and categorical features representing clinical metrics (e.g., BMI, Systolic BP, Medical Conditions) and demographics (e.g., Age, Region, Income).

*Note: Models are rigorously regularized to prevent overfitting and leakage that previously skewed performance.*

## Project Structure
```text
.
├── backend/                  # FastAPI Application
│   ├── __init__.py
│   ├── main.py               # API Endpoints & Static Serving
│   ├── model_service.py      # Preprocessor and Model Loader 
│   └── schemas.py            # Pydantic validation schemas
├── data/                     
│   ├── processed/            # Processed numpy arrays & preprocessor artifact
│   └── raw/                  # Raw CSV dataset
├── frontend/                 # Professional Web UI
│   ├── index.html            
│   └── script.js             
├── models/                   # Saved model artifacts (best_model.pkl)
├── src/                      
│   └── preprocessing/        # Data engineering pipelines
│       └── main_preprocessing.py
├── tests/                    # Pytest test suite
│   └── test_pipeline.py
├── app.py                    # Legacy Streamlit UI (Optional)
├── train.py                  # Model training and MLflow tracking script
├── run_app.bat               # Windows startup script
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

## Setup & Installation
1. **Clone the repository and CD into it**:
   ```bash
   cd Medical-Insurance-Cost-Predicition
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Execution

### 1. Run Data Preprocessing
Generate the missing value imputations, scalers, and encoders.
```bash
python src/preprocessing/main_preprocessing.py
```

### 2. Train Models
Train the Baseline and Regularized Random Forest models, logging runs to MLflow.
```bash
python train.py
```

*(Optional)* View the MLflow Dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 3. Launch the Application
Start the FastAPI server and the Frontend web interface.
**On Windows**:
```cmd
run_app.bat
```

**Manual Startup**:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Then navigate to **http://localhost:8000/** in your browser.

*(Optional)* Legacy Streamlit interface can be launched via: `streamlit run app.py`

## Testing
Run the automated test suite utilizing `pytest` to guarantee the ML artifacts load properly and end-to-end inference works reliably:
```bash
pytest tests/
```

## Example Prediction payload
Endpoint: `POST /predict`
```json
{
  "age": 30,
  "sex": "Male",
  "region": "North",
  "dependents": 0,
  "income": 50000.0,
  "bmi": 25.0,
  "smoker": "Never",
  "systolic_bp": 120.0,
  "diastolic_bp": 80.0
}
```
Response:
```json
{
  "predicted_cost": 4125.68,
  "currency": "₹"
}
```