import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

def main():
    print("Fetching UCI Heart Disease dataset...")
    # fetch dataset (Heart Disease is ID 45)
    heart_disease = fetch_ucirepo(id=45)
    
    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    
    # Target is 0 (no disease) and 1-4 (presence of disease)
    # We binarize it for binary classification
    y = (y > 0).astype(int).values.ravel()
    
    # The requirement mentions handling missing values via imputation.
    # We'll create a pipeline that handles this.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing steps: Imputation -> Z-score normalization -> RFE
    # Estimator for RFE
    rfe_estimator = LogisticRegression(max_iter=1000)
    
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Statistical imputation
        ('scaler', StandardScaler()), # Z-score normalization
        ('rfe', RFE(estimator=rfe_estimator, n_features_to_select=10)) # Recursive Feature Elimination
    ])
    
    # Transform data
    print("Preprocessing data (Imputation, Scaling, RFE)...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_processed, y_train)
    
    # Evaluate Logistic Regression
    lr_preds = lr_model.predict(X_test_processed)
    lr_probs = lr_model.predict_proba(X_test_processed)[:, 1]
    
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    print(f"Logistic Regression - Accuracy: {lr_acc:.4f}, AUC-ROC: {lr_auc:.4f}")
    
    # Train SVM Baseline
    print("Training SVM baseline...")
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_processed, y_train)
    
    svm_preds = svm_model.predict(X_test_processed)
    svm_probs = svm_model.predict_proba(X_test_processed)[:, 1]
    
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_auc = roc_auc_score(y_test, svm_probs)
    
    print(f"SVM Baseline - Accuracy: {svm_acc:.4f}, AUC-ROC: {svm_auc:.4f}")
    print(f"Improvement over SVM: {(lr_acc - svm_acc)*100:.2f}% Accuracy")
    
    # Create the final pipeline object that includes preprocessing and the classifier
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lr_model)
    ])
    
    # Save the pipeline
    model_path = 'model_pipeline.pkl'
    joblib.dump(final_pipeline, model_path)
    print(f"Model pipeline saved to {model_path}")
    
    # Save feature names for reference
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Feature names saved to feature_names.pkl")

if __name__ == '__main__':
    main()
