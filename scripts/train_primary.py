import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# Config
DATA_PATH = "data/onlinefraud.csv"
MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_primary.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_type.pkl")

def train():
    print("="*60)
    print("PRIMARY MODEL TRAINING PIPELINE (HD GRADE)")
    print("="*60)
    
    print("\n[Step 1] Loading Primary Dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        return

    # Load with optimized dtypes
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    
    # Feature Engineering
    print("\n[Step 2] Feature Engineering...")
    # 1. Select specific types
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    
    # 2. Balance Error Features
    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Preprocessing
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    
    # Features & Target
    X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg', 'errorBalanceDest']]
    y = df['isFraud']

    # Split
    print("\n[Step 3] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Critical Failure 2 Fix: Handling Class Imbalance with SMOTE
    print(f"Original Train Balance: {y_train.value_counts(normalize=True).to_dict()}")
    print("Implementing SMOTE (Synthetic Minority Over-sampling Technique)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled Train Balance: {y_train_resampled.value_counts(normalize=True).to_dict()}")

    # Critical Failure 1 Fix: Hyperparameter Tuning
    print("\n[Step 4] Hyperparameter Tuning (RandomizedSearchCV)...")
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    
    # User-defined HD Grid
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 10, 25, 50]  # Crucial for imbalance!
    }
    
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=3, 
        verbose=1,
        scoring='average_precision', # Optimize for AUPRC directly
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train_resampled, y_train_resampled)
    best_model = random_search.best_estimator_
    print(f"Best Params: {random_search.best_params_}")

    # Evaluate
    print("\n[Step 5] Evaluation...")
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    
    # Critical Failure 3 Fix: Metric Rigor (AUPRC & Confusion Matrix)
    auprc = average_precision_score(y_test, probs)
    print(f"AUPRC Score: {auprc:.4f}")
    
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
