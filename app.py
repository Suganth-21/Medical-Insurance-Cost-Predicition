import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Enter patient demographic and health information below to estimate medical insurance costs.")

# 2. Load Model and Preprocessor
@st.cache_resource
def load_artifacts():
    model_path = "models/best_model.pkl"
    preprocessor_path = "data/processed/preprocessor.pkl"
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
    st.success("✅ ML Model & Preprocessor loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()

st.divider()

# 3. User Input Form
st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Number of Children / Dependents", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker Status", options=["Never", "Former", "Current"])
    region = st.selectbox("Region", options=["North", "South", "East", "West"])

# 4. Predict Button & Inference Logic
st.divider()

if st.button("🚀 Estimate Insurance Cost", use_container_width=True):
    # Construct input dataframe with all 52 required features (using sensible defaults for missing fields)
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "region": region,
        "urban_rural": "Urban",
        "income": 50000.0,
        "education": "Bachelor",
        "marital_status": "Married",
        "employment_status": "Employed",
        "household_size": 2 + children,
        "dependents": children,
        "bmi": bmi,
        "smoker": smoker,
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
        "annual_premium": 1000.0,
        "monthly_premium": 100.0,
        "claims_count": 0,
        "avg_claim_amount": 0.0,
        "total_claims_paid": 0.0,
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
        "proc_imaging_count": 0,
        "proc_surgery_count": 0,
        "proc_physio_count": 0,
        "proc_consult_count": 0,
        "proc_lab_count": 0,
        "is_high_risk": 0,
        "had_major_procedure": 0
    }])
    
    try:
        # Preprocess features and predict
        transformed_input = preprocessor.transform(input_data)
        prediction = model.predict(transformed_input)[0]
        
        # Display Prediction Result
        st.markdown("### 💰 Estimated Insurance Premium")
        st.metric(label="Predicted Cost", value=f"₹{prediction:,.2f}")
        
    except Exception as err:
        st.error(f"Error during prediction calculation: {err}")