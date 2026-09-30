import joblib
import pandas as pd
import os

class ModelService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "best_model.pkl")
        self.preprocessor_path = os.path.join(base_dir, "data", "processed", "preprocessor.pkl")
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        # Mapping from Indian States/UTs to underlying dataset regions (North, South, East, West, Central)
        self.state_to_region = {
            "Jammu and Kashmir": "North", "Ladakh": "North", "Himachal Pradesh": "North", 
            "Punjab": "North", "Chandigarh": "North", "Uttarakhand": "North", 
            "Haryana": "North", "Delhi": "North", "Uttar Pradesh": "North",
            "Andhra Pradesh": "South", "Karnataka": "South", "Kerala": "South", 
            "Tamil Nadu": "South", "Telangana": "South", "Lakshadweep": "South", 
            "Puducherry": "South", "Andaman and Nicobar Islands": "South",
            "Bihar": "East", "Jharkhand": "East", "Odisha": "East", 
            "West Bengal": "East", "Sikkim": "East", "Assam": "East", 
            "Arunachal Pradesh": "East", "Nagaland": "East", "Manipur": "East", 
            "Mizoram": "East", "Tripura": "East", "Meghalaya": "East",
            "Rajasthan": "West", "Gujarat": "West", "Maharashtra": "West", 
            "Goa": "West", "Dadra and Nagar Haveli and Daman and Diu": "West",
            "Madhya Pradesh": "Central", "Chhattisgarh": "Central"
        }

    def predict(self, input_data: dict) -> float:
        state = input_data.get("state", "Delhi")
        mapped_region = self.state_to_region.get(state, "North")
        
        # Calculate derived fields
        dependents = input_data.get("dependents", 0.0)
        marital_status = input_data.get("marital_status", "Single")
        household_size = 1.0 + dependents + (1.0 if marital_status == "Married" else 0.0)
        
        chronic_count = sum([
            input_data.get("hypertension", 0.0),
            input_data.get("diabetes", 0.0),
            input_data.get("asthma", 0.0),
            input_data.get("copd", 0.0),
            input_data.get("cardiovascular_disease", 0.0),
            input_data.get("cancer_history", 0.0),
            input_data.get("kidney_disease", 0.0),
            input_data.get("liver_disease", 0.0),
            input_data.get("arthritis", 0.0),
            input_data.get("mental_health", 0.0)
        ])
        
        full_features = {
            "age": input_data.get("age", 48.0),
            "sex": input_data.get("sex", "Female"),
            "region": mapped_region,
            "urban_rural": input_data.get("urban_rural", "Urban"),
            "income": input_data.get("income", 3004600.0),
            "education": input_data.get("education", "Bachelors"),
            "marital_status": marital_status,
            "employment_status": input_data.get("employment_status", "Employed"),
            "household_size": household_size,
            "dependents": dependents,
            "bmi": input_data.get("bmi", 27.0),
            "smoker": input_data.get("smoker", "Never"),
            "alcohol_freq": input_data.get("alcohol_freq", "Occasional"),
            "visits_last_year": input_data.get("visits_last_year", 2.0),
            "hospitalizations_last_3yrs": input_data.get("hospitalizations_last_3yrs", 0.0),
            "days_hospitalized_last_3yrs": input_data.get("days_hospitalized_last_3yrs", 0.0),
            "medication_count": input_data.get("medication_count", 1.0),
            "systolic_bp": input_data.get("systolic_bp", 117.0),
            "diastolic_bp": input_data.get("diastolic_bp", 73.0),
            "ldl": input_data.get("ldl", 120.0),
            "hba1c": input_data.get("hba1c", 5.44),
            "chronic_count": float(chronic_count),
            "hypertension": input_data.get("hypertension", 0.0),
            "diabetes": input_data.get("diabetes", 0.0),
            "asthma": input_data.get("asthma", 0.0),
            "copd": input_data.get("copd", 0.0),
            "cardiovascular_disease": input_data.get("cardiovascular_disease", 0.0),
            "cancer_history": input_data.get("cancer_history", 0.0),
            "kidney_disease": input_data.get("kidney_disease", 0.0),
            "liver_disease": input_data.get("liver_disease", 0.0),
            "arthritis": input_data.get("arthritis", 0.0),
            "mental_health": input_data.get("mental_health", 0.0),
            "is_high_risk": input_data.get("is_high_risk", 0.0),
            
            # The remaining non-UI dataset features (anchors)
            "policy_term_years": 6.0,
            "policy_changes_last_2yrs": 0.0,
            "provider_quality": 3.6
        }
        
        df = pd.DataFrame([full_features])
        expected_cols = self.preprocessor.feature_names_in_
        df = df[expected_cols]
        
        transformed = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed)[0]
        return float(prediction)

model_service = ModelService()
