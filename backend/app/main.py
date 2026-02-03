from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from .schemas import TransactionInput, CreditCardInput, PredictionOutput
from math import radians, cos, sin, asin, sqrt

app = FastAPI(title="Fraud Simulator API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:8081"], # Vite default & alternatives
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Models
models = {}

@app.on_event("startup")
def load_models():
    model_dir = "backend/models"
    try:
        models["primary"] = joblib.load(os.path.join(model_dir, "model_primary.pkl"))
        models["encoder"] = joblib.load(os.path.join(model_dir, "label_encoder_type.pkl"))
        models["secondary_rf"] = joblib.load(os.path.join(model_dir, "model_secondary_rf.pkl"))
        # models["secondary_iso"] = joblib.load(os.path.join(model_dir, "model_secondary_iso.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

# --- Primary Model (Mobile Money) ---
@app.post("/predict/primary", response_model=PredictionOutput)
def predict_primary(data: TransactionInput):
    if "primary" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Feature Engineering (Must match training script)
    # 1. Type Encoding
    try:
        type_encoded = models["encoder"].transform([data.type])[0]
    except:
        # If unknown type (e.g., DEBIT which we filtered out?), default to CASH_OUT or similar safest
        type_encoded = 0 
    
    # 2. Balance Errors
    errorBalanceOrg = data.newbalanceOrig + data.amount - data.oldbalanceOrg
    errorBalanceDest = data.oldbalanceDest + data.amount - data.newbalanceDest
    
    features = pd.DataFrame([{
        'type': type_encoded,
        'amount': data.amount,
        'oldbalanceOrg': data.oldbalanceOrg,
        'newbalanceOrig': data.newbalanceOrig,
        'errorBalanceOrg': errorBalanceOrg,
        'errorBalanceDest': errorBalanceDest
    }])
    
    # Predict
    prob = models["primary"].predict_proba(features)[0][1]
    is_fraud = prob > 0.5
    
    return {
        "probability": float(prob),
        "is_fraud": bool(is_fraud),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
        "explanation": f"Balance Discrepancy: {errorBalanceOrg:.2f}"
    }

# --- Secondary Model (Credit Card) ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

@app.post("/predict/secondary", response_model=PredictionOutput)
def predict_secondary(data: CreditCardInput):
    if "secondary_rf" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Feature Engineering
    dist = haversine(data.long, data.lat, data.merch_long, data.merch_lat)
    
    try:
        data_dob = pd.to_datetime(data.dob)
        age = 2020 - data_dob.year
    except:
        age = 30 # default
        
    features = pd.DataFrame([{
        'amt': data.amt,
        'dist_to_merch': dist,
        'age': age,
        'city_pop': data.city_pop
    }])
    
    # Predict
    pred = models["secondary_rf"].predict(features)[0]
    prob = models["secondary_rf"].predict_proba(features)[0][1]
    
    return {
        "probability": float(prob),
        "is_fraud": bool(pred),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
        "explanation": f"Distance: {dist:.2f}km"
    }
