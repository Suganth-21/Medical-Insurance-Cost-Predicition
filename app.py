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
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Number of Children / Dependents", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker Status", options=["no", "yes"])
    region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])

# 4. Predict Button & Inference Logic
st.divider()

if st.button("🚀 Estimate Insurance Cost", use_container_width=True):
    # Construct input dataframe
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
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