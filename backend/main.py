import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import ValidationError

from backend.schemas import PredictionRequest, PredictionResponse
from backend.model_service import model_service

app = FastAPI(title="Medical Insurance Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        cost = model_service.predict(request.model_dump())
        return PredictionResponse(predicted_cost=cost)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
