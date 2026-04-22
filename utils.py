import joblib
import pandas as pd
import numpy as np

def load_models():
    """Loads and returns all trained models and the scaler."""
    try:
        scaler = joblib.load('models/scaler.pkl')
        clf_eligibility = joblib.load('models/clf_eligibility.pkl')
        reg_amount = joblib.load('models/reg_amount.pkl')
        reg_rate = joblib.load('models/reg_rate.pkl')
        reg_tenure = joblib.load('models/reg_tenure.pkl')
        return scaler, clf_eligibility, reg_amount, reg_rate, reg_tenure
    except FileNotFoundError:
        return None, None, None, None, None

def make_prediction(scaler, clf, reg_amt, reg_rate, reg_ten, input_data):
    """
    Takes user input, scales it, and returns predictions.
    input_data format: dict with keys: 'Credit Score', 'Age', 'Annual Income', 'Previous Loan Taken'
    """
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Map categorical
    df_input['Previous Loan Taken'] = df_input['Previous Loan Taken'].map({'Yes': 1, 'No': 0})
    
    # Scale numerical
    X_scaled = scaler.transform(df_input)
    
    # 1. Predict Eligibility
    eligibility_prob = clf.predict_proba(X_scaled)[0][1] # Probability of class 1
    is_eligible = clf.predict(X_scaled)[0]
    
    results = {
        'is_eligible': bool(is_eligible),
        'eligibility_probability': float(eligibility_prob)
    }
    
    if is_eligible:
        # Predict regression targets
        amount = reg_amt.predict(X_scaled)[0]
        rate = reg_rate.predict(X_scaled)[0]
        tenure = reg_ten.predict(X_scaled)[0]
        
        # Format values
        results['amount'] = max(1000, round(amount / 100) * 100) # Minimum 1000, rounded to nearest 100
        results['rate'] = max(1.0, round(rate, 2)) # Minimum 1.0%
        results['tenure'] = max(12, int(round(tenure / 12) * 12)) # Minimum 12, snap to nearest 12
        
    return results
