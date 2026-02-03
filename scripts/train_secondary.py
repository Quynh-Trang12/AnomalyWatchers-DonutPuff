import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from math import radians, cos, sin, asin, sqrt
import joblib
import os

# Config
DATA_PATH = "data/fraudTrain.csv" # Using Train for training
MODEL_DIR = "backend/models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "model_secondary_rf.pkl")
ISO_MODEL_PATH = os.path.join(MODEL_DIR, "model_secondary_iso.pkl")

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

def train():
    print("Loading Secondary Dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH).sample(n=100000, random_state=42) # Downsample for demo speed
    
    print("Feature Engineering...")
    # Geospatial
    df['dist_to_merch'] = haversine_vectorized(df['long'], df['lat'], df['merch_long'], df['merch_lat'])
    
    # Age
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = 2020 - df['dob'].dt.year

    # Features
    features = ['amt', 'dist_to_merch', 'age', 'city_pop']
    X = df[features]
    y = df['is_fraud']

    # 1. Random Forest (Supervised)
    print("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    
    print("RF Evaluation:")
    print(classification_report(y_test, rf.predict(X_test)))
    
    joblib.dump(rf, RF_MODEL_PATH)
    print(f"RF Model saved to {RF_MODEL_PATH}")

    # 2. Isolation Forest (Unsupervised/Clustering)
    print("Training Isolation Forest...")
    # We train on 'normal' behavior mostly, but here just unsupervised on X
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X)
    
    joblib.dump(iso, ISO_MODEL_PATH)
    print(f"Isolation Forest Model saved to {ISO_MODEL_PATH}")

if __name__ == "__main__":
    train()
