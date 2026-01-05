import sqlite3
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

from src.matching import get_candidates, calculate_features

# Configuration
DB_PATH = "data/clients.db"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "entity_resolution_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data_and_generate_training_set():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load Processed Data (for features)
    print("Loading processed data...")
    df_all = pd.read_sql("SELECT * FROM clients_processed", conn)
    # OPTIMIZATION: Pass DataFrame directly to calculate_features for parallel processing
    # df_dict = df_all.set_index('record_id').to_dict('index')
    
    # 2. Load Ground Truth (for labels)
    print("Loading ground truth...")
    df_truth = pd.read_sql("SELECT record_id, entity_id FROM clients", conn)
    truth_dict = df_truth.set_index('record_id')['entity_id'].to_dict()
    
    # 3. Generate Candidates (The Search Space)
    # We use the same blocking logic as the pipeline to ensure we train on relevant pairs
    pairs_df = get_candidates(conn)
    
    # 4. Calculate Features
    print("Calculating features for training...")
    features_df = calculate_features(pairs_df, df_all)
    
    # 5. Create Labels
    print("Generating labels...")
    def get_label(row):
        id_a = row['id_a']
        id_b = row['id_b']
        # Match if Entity IDs are identical
        return 1 if truth_dict.get(id_a) == truth_dict.get(id_b) else 0
        
    features_df['label'] = features_df.apply(get_label, axis=1)
    
    print(f"Training Data: {len(features_df)} pairs.")
    print(f"Class Balance: {features_df['label'].value_counts().to_dict()}")
    
    conn.close()
    return features_df

def train_model(df):
    # Feature Columns (exclude IDs and Label)
    # Removing city/address as they are artificially perfect predictors in synthetic data
    feature_cols = [
        'nid_score', 'email_score', 'phone_match', 
        'first_name_score', 'last_name_score', 
        'dob_match', 'year_match'
    ]
    
    X = df[feature_cols]
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost
    print("Training XGBoost Classifier...")
    # Scale_pos_weight helps with imbalanced datasets (which entity resolution usually is)
    # Estimate ratio: Negatives / Positives
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / pos_count if pos_count > 0 else 1
    
    # Use a more conservative configuration to prevent overfitting
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=3,              # Reduced depth (was 5)
        learning_rate=0.1,
        subsample=0.8,            # Train on 80% of data per tree
        colsample_bytree=0.8,     # Use 80% of features per tree
        reg_alpha=0.1,            # L1 Regularization
        reg_lambda=1.0,           # L2 Regularization
        scale_pos_weight=scale_weight,
        eval_metric='logloss'
    )
    
    # Cross-Validation
    print("Running 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV ROC-AUC Scores: {scores}")
    print(f"Mean CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    print("\nFeature Importance:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    return model

if __name__ == "__main__":
    # 1. Prepare Data
    train_df = load_data_and_generate_training_set()
    
    # 2. Train
    model = train_model(train_df)
    
    # 3. Save
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Done.")
