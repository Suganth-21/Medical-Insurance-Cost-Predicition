import joblib
import pandas as pd
import numpy as np
import pytest
import os

def test_model_and_preprocessor_exist():
    assert os.path.exists("models/best_model.pkl"), "Model file is missing."
    assert os.path.exists("data/processed/preprocessor.pkl"), "Preprocessor file is missing."

def test_inference_pipeline():
    # Load artifacts
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    
    # Create dummy input data that exactly matches the expected 41 columns
    dummy_input = pd.DataFrame([{
        "age": 30,
        "sex": "Male",
        "region": "North",
        "urban_rural": "Urban",
        "income": 50000.0,
        "education": "Bachelor",
        "marital_status": "Married",
        "employment_status": "Employed",
        "household_size": 2,
        "dependents": 0,
        "bmi": 25.0,
        "smoker": "Never",
        "alcohol_freq": "None",
        "visits_last_year": 1,
        "hospitalizations_last_3yrs": 0,
        "days_hospitalized_last_3yrs": 0,
        "medication_count": 0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "ldl": 100.0,
        "hba1c": 5.5,
        "plan_type": "PPO",
        "network_tier": "Bronze",
        "deductible": 1000.0,
        "copay": 20.0,
        "policy_term_years": 1,
        "policy_changes_last_2yrs": 0,
        "provider_quality": 3.0,
        "risk_score": 0.5,
        "chronic_count": 0,
        "hypertension": 0,
        "diabetes": 0,
        "asthma": 0,
        "copd": 0,
        "cardiovascular_disease": 0,
        "cancer_history": 0,
        "kidney_disease": 0,
        "liver_disease": 0,
        "arthritis": 0,
        "mental_health": 0,
        "is_high_risk": 0
    }])
    
    # Preprocess
    transformed_input = preprocessor.transform(dummy_input)
    assert transformed_input is not None
    assert transformed_input.shape[0] == 1
    
    # Predict
    prediction = model.predict(transformed_input)
    assert prediction is not None
    assert len(prediction) == 1
    assert isinstance(prediction[0], float) or isinstance(prediction[0], np.float64)
