import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Config
DATA_PATH = "data/onlinefraud.csv"
MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_primary.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_type.pkl")

def train():
    print("Loading Primary Dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        return

    # Load only necessary columns for speed if dataset is huge
    df = pd.read_csv(DATA_PATH)
    
    # Feature Engineering
    print("Feature Engineering...")
    # 1. Select specific types
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    
    # 2. Balance Error
    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Preprocessing
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    
    # Features & Target
    X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg', 'errorBalanceDest']]
    y = df['isFraud']

    # Split
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train
    print("Training XGBoost...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    print("Evaluating...")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
