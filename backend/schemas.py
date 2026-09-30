from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: str = Field(...)
    bmi: float = Field(..., ge=10.0, le=50.0)
    dependents: int = Field(..., ge=0, le=15)
    smoker: str = Field(...)
    state: str = Field(...)
    income: float = Field(default=3000000.0)
    systolic_bp: float = Field(default=120.0)
    diastolic_bp: float = Field(default=80.0)
    
    # New Demographics
    urban_rural: str = Field(default="Urban")
    education: str = Field(default="Bachelors")
    marital_status: str = Field(default="Single")
    employment_status: str = Field(default="Employed")
    alcohol_freq: str = Field(default="Never")
    
    # Medical History
    visits_last_year: float = Field(default=1.0)
    hospitalizations_last_3yrs: float = Field(default=0.0)
    days_hospitalized_last_3yrs: float = Field(default=0.0)
    medication_count: float = Field(default=0.0)
    
    # Vitals / Disease History
    ldl: float = Field(default=100.0)
    hba1c: float = Field(default=5.5)
    
    # Booleans / Checkboxes (passed as floats 1.0 or 0.0)
    hypertension: float = Field(default=0.0)
    diabetes: float = Field(default=0.0)
    is_high_risk: float = Field(default=0.0)
    asthma: float = Field(default=0.0)
    copd: float = Field(default=0.0)
    cardiovascular_disease: float = Field(default=0.0)
    cancer_history: float = Field(default=0.0)
    kidney_disease: float = Field(default=0.0)
    liver_disease: float = Field(default=0.0)
    arthritis: float = Field(default=0.0)
    mental_health: float = Field(default=0.0)

class PredictionResponse(BaseModel):
    predicted_cost: float
    currency: str = "₹"
