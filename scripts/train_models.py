"""
AnomalyWatchers - ML Pipeline Training Script
Assignment 2: Tri-Model Fraud Detection Architecture

This script implements:
1. Data Engineering (Loading, Preprocessing, SMOTE)
2. Feature Engineering (Error features, Ratios)
3. Tri-Model Training (Logistic Regression, XGBoost, Isolation Forest)
4. Evaluation (AUPRC, Confusion Matrix, SHAP)
5. Model Serialization
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve,
    average_precision_score,
    f1_score,
    roc_auc_score
)

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    print("Installing XGBoost...")
    os.system(f"{sys.executable} -m pip install xgboost")
    import xgboost as xgb

# SMOTE for imbalance handling (optional)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Note: imbalanced-learn not available. Using class weights instead of SMOTE.")
    SMOTE_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = "data"
MODEL_DIR = "backend/models"
PRIMARY_DATASET = os.path.join(DATA_DIR, "onlinefraud.csv")

# Features matching frontend schema
FEATURE_COLUMNS = [
    'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'errorBalanceOrg', 'errorBalanceDest'
]
TARGET_COLUMN = 'isFraud'

# ==============================================================================
# PHASE 1: DATA ENGINEERING
# ==============================================================================

def load_primary_dataset(filepath: str, sample_frac: float = 0.1) -> pd.DataFrame:
    """
    Load the Rupak Roy (Paysim) dataset with memory optimization.
    Uses sampling for faster development iterations.
    """
    print(f"\n{'='*60}")
    print("PHASE 1: DATA ENGINEERING")
    print(f"{'='*60}")
    
    print(f"\n[1.1] Loading dataset: {filepath}")
    
    # Optimize dtypes for memory efficiency
    dtype_map = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'object',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }
    
    df = pd.read_csv(filepath, dtype=dtype_map)
    print(f"   Total records: {len(df):,}")
    print(f"   Memory usage: {df.memory_usage().sum() / 1e6:.1f} MB")
    
    # Sample for faster training (remove this for production)
    if sample_frac < 1.0:
        print(f"\n[1.2] Sampling {sample_frac*100:.0f}% of data for faster training...")
        # Stratified sampling to maintain fraud ratio
        df_fraud = df[df['isFraud'] == 1]
        df_legit = df[df['isFraud'] == 0].sample(frac=sample_frac, random_state=42)
        df = pd.concat([df_fraud, df_legit]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   Sampled records: {len(df):,}")
    
    # Show class imbalance
    fraud_rate = df['isFraud'].mean()
    print(f"\n[1.3] Class Distribution:")
    print(f"   Legitimate: {(1-fraud_rate)*100:.2f}%")
    print(f"   Fraudulent: {fraud_rate*100:.4f}%")
    
    return df

# ==============================================================================
# PHASE 2: FEATURE ENGINEERING
# ==============================================================================

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Create features matching the frontend schema.
    Implements the 'error' features that are critical for fraud detection.
    """
    print(f"\n{'='*60}")
    print("PHASE 2: FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    df = df.copy()
    
    # Filter to only transaction types where fraud occurs
    # (Fraud only happens in TRANSFER and CASH_OUT in Paysim)
    print("\n[2.1] Filtering to relevant transaction types...")
    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])]
    print(f"   Records after filtering: {len(df):,}")
    
    # Create Error Features (The "Secret Sauce")
    # In legitimate transactions, these should be ~0
    # In fraud, the balance math doesn't add up
    print("\n[2.2] Creating Error Features...")
    df['errorBalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # Label Encode Transaction Type
    print("\n[2.3] Encoding categorical features...")
    le_type = LabelEncoder()
    df['type'] = le_type.fit_transform(df['type'])
    print(f"   Type classes: {list(le_type.classes_)}")
    
    # Select only the features we need
    print("\n[2.4] Selecting final feature set...")
    features = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    
    # Check for missing values
    missing = features.isnull().sum().sum()
    print(f"   Missing values: {missing}")
    
    return features, le_type

# ==============================================================================
# PHASE 3: TRI-MODEL ARCHITECTURE
# ==============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train the Tri-Model Architecture:
    1. Logistic Regression (Baseline)
    2. XGBoost (Champion)
    3. Isolation Forest (Anomaly Detector)
    """
    print(f"\n{'='*60}")
    print("PHASE 3: TRI-MODEL ARCHITECTURE")
    print(f"{'='*60}")
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Model 1: Logistic Regression (Baseline)
    # -------------------------------------------------------------------------
    print("\n[3.1] Training Model 1: Logistic Regression (Baseline)")
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    results['logistic_regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'probabilities': y_prob_lr,
        'auprc': average_precision_score(y_test, y_prob_lr),
        'f1': f1_score(y_test, y_pred_lr)
    }
    print(f"   AUPRC: {results['logistic_regression']['auprc']:.4f}")
    print(f"   F1-Score: {results['logistic_regression']['f1']:.4f}")
    
    # -------------------------------------------------------------------------
    # Model 2: XGBoost (Champion)
    # -------------------------------------------------------------------------
    print("\n[3.2] Training Model 2: XGBoost (Champion)")
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    # Hyperparameter tuning with GridSearchCV
    xgb_params = {
        'max_depth': [4, 6],
        'learning_rate': [0.1, 0.3],
        'n_estimators': [100],
        'scale_pos_weight': [scale_pos_weight],
        'eval_metric': ['aucpr'],
        'use_label_encoder': [False]
    }
    
    xgb_model = xgb.XGBClassifier(random_state=42, verbosity=0)
    
    print("   Running GridSearchCV...")
    grid_search = GridSearchCV(
        xgb_model, xgb_params, 
        cv=3, scoring='f1', 
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
    
    results['xgboost'] = {
        'model': best_xgb,
        'predictions': y_pred_xgb,
        'probabilities': y_prob_xgb,
        'auprc': average_precision_score(y_test, y_prob_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'best_params': grid_search.best_params_
    }
    print(f"   Best Params: {grid_search.best_params_}")
    print(f"   AUPRC: {results['xgboost']['auprc']:.4f}")
    print(f"   F1-Score: {results['xgboost']['f1']:.4f}")
    
    # -------------------------------------------------------------------------
    # Model 3: Isolation Forest (Anomaly Detector)
    # -------------------------------------------------------------------------
    print("\n[3.3] Training Model 3: Isolation Forest (Anomaly Detector)")
    
    # Train only on legitimate transactions (unsupervised)
    X_legit = X_train[y_train == 0]
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,  # Expected fraud rate
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_legit)
    
    # Predict: -1 = anomaly (fraud), 1 = normal
    y_pred_iso = iso_forest.predict(X_test)
    y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert to 0/1
    
    results['isolation_forest'] = {
        'model': iso_forest,
        'predictions': y_pred_iso,
        'f1': f1_score(y_test, y_pred_iso)
    }
    print(f"   F1-Score: {results['isolation_forest']['f1']:.4f}")
    
    return results

# ==============================================================================
# PHASE 4: EVALUATION
# ==============================================================================

def evaluate_models(results: dict, y_test: np.ndarray):
    """
    Comprehensive model evaluation with metrics and confusion matrices.
    """
    print(f"\n{'='*60}")
    print("PHASE 4: EVALUATION")
    print(f"{'='*60}")
    
    print("\n[4.1] Model Comparison Summary:")
    print("-" * 50)
    print(f"{'Model':<25} {'AUPRC':<10} {'F1-Score':<10}")
    print("-" * 50)
    
    for name, res in results.items():
        auprc = res.get('auprc', 'N/A')
        f1 = res['f1']
        if isinstance(auprc, float):
            print(f"{name:<25} {auprc:<10.4f} {f1:<10.4f}")
        else:
            print(f"{name:<25} {auprc:<10} {f1:<10.4f}")
    
    # Champion model confusion matrix
    print("\n[4.2] Champion Model (XGBoost) Confusion Matrix:")
    cm = confusion_matrix(y_test, results['xgboost']['predictions'])
    print(f"   True Negatives:  {cm[0,0]:,}")
    print(f"   False Positives: {cm[0,1]:,} (Customer Insults)")
    print(f"   False Negatives: {cm[1,0]:,} (Missed Fraud - CRITICAL)")
    print(f"   True Positives:  {cm[1,1]:,}")
    
    # Classification Report
    print("\n[4.3] Classification Report (XGBoost):")
    print(classification_report(y_test, results['xgboost']['predictions'], 
                                target_names=['Legitimate', 'Fraud']))

# ==============================================================================
# PHASE 5: SERIALIZATION
# ==============================================================================

def save_models(results: dict, le_type: LabelEncoder, model_dir: str):
    """
    Serialize trained models for backend integration.
    """
    print(f"\n{'='*60}")
    print("PHASE 5: SERIALIZATION")
    print(f"{'='*60}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save XGBoost (Champion) as primary model
    primary_path = os.path.join(model_dir, "model_primary.pkl")
    joblib.dump(results['xgboost']['model'], primary_path)
    print(f"\n[5.1] Saved XGBoost model: {primary_path}")
    
    # Save Label Encoder
    encoder_path = os.path.join(model_dir, "label_encoder_type.pkl")
    joblib.dump(le_type, encoder_path)
    print(f"[5.2] Saved Label Encoder: {encoder_path}")
    
    # Save Logistic Regression (for explainability)
    lr_path = os.path.join(model_dir, "model_logistic.pkl")
    joblib.dump(results['logistic_regression']['model'], lr_path)
    print(f"[5.3] Saved Logistic Regression: {lr_path}")
    
    # Save Isolation Forest (for anomaly detection)
    iso_path = os.path.join(model_dir, "model_isolation_forest.pkl")
    joblib.dump(results['isolation_forest']['model'], iso_path)
    print(f"[5.4] Saved Isolation Forest: {iso_path}")
    
    print(f"\n✅ All models saved to: {model_dir}/")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Execute the complete ML Pipeline.
    """
    print("\n" + "="*60)
    print("ANOMALYWATCHERS - ML PIPELINE TRAINING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Phase 1: Load Data
    df = load_primary_dataset(PRIMARY_DATASET, sample_frac=1.0)
    
    # Phase 2: Feature Engineering
    features_df, le_type = engineer_features(df)
    
    # Prepare train/test split
    X = features_df[FEATURE_COLUMNS].values
    y = features_df[TARGET_COLUMN].values
    
    print(f"\n[2.5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {len(X_train):,}")
    print(f"   Test size: {len(X_test):,}")
    
    # Apply SMOTE for class balancing (only on training data)
    if SMOTE_AVAILABLE:
        print(f"\n[2.6] Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   Balanced train size: {len(X_train_balanced):,}")
        print(f"   Fraud ratio after SMOTE: {y_train_balanced.mean()*100:.2f}%")
    else:
        print(f"\n[2.6] Using class weights (SMOTE not available)...")
        X_train_balanced, y_train_balanced = X_train, y_train
        print(f"   Train size: {len(X_train_balanced):,}")
        print(f"   Using class_weight='balanced' in models")
    
    # Phase 3: Train Models
    results = train_models(X_train_balanced, X_test, y_train_balanced, y_test)
    
    # Phase 4: Evaluate
    evaluate_models(results, y_test)
    
    # Phase 5: Save Models
    save_models(results, le_type, MODEL_DIR)
    
    print("\n" + "="*60)
    print("✅ ML PIPELINE COMPLETE")
    print("="*60)
    print("\nNext Steps:")
    print("1. Restart the backend server to load the new models")
    print("2. Test with the frontend (High Risk preset)")
    print("3. Verify predictions match expected behavior")
    
    return results

if __name__ == "__main__":
    main()
