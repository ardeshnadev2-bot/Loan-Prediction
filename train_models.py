import os
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error, mean_squared_error
import numpy as np

from preprocess import load_data, preprocess_data

def train_and_evaluate():
    """Trains the models, evaluates them, and saves the best ones."""
    
    print("Loading data...")
    df = load_data()
    if df is None:
        return
        
    print("Preprocessing data...")
    data = preprocess_data(df)
    
    # Unpack data
    X_train_c, X_test_c, y_train_c, y_test_c = data['classification']
    X_train_r, X_test_r, y_train_amt, y_test_amt, y_train_rate, y_test_rate, y_train_ten, y_test_ten = data['regression']
    scaler = data['scaler']
    
    os.makedirs('models', exist_ok=True)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved Scaler.")
    
    # ---------------------------------------------------------
    # 1. Classification (Eligibility)
    # ---------------------------------------------------------
    print("\n--- Training Eligibility Classification Model ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    # Evaluate
    y_pred_c = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test_c, y_pred_c))
    
    # Save Model
    joblib.dump(clf, 'models/clf_eligibility.pkl')
    print("Saved Eligibility Model.")
    
    # ---------------------------------------------------------
    # 2. Regression (Recommended Amount)
    # ---------------------------------------------------------
    print("\n--- Training Recommended Amount Regression Model ---")
    reg_amt = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_amt.fit(X_train_r, y_train_amt)
    
    # Evaluate
    y_pred_amt = reg_amt.predict(X_test_r)
    print(f"R2 Score: {r2_score(y_test_amt, y_pred_amt):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_amt, y_pred_amt):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_amt, y_pred_amt)):.2f}")
    
    # Save Model
    joblib.dump(reg_amt, 'models/reg_amount.pkl')
    print("Saved Amount Model.")
    
    # ---------------------------------------------------------
    # 3. Regression (Interest Rate)
    # ---------------------------------------------------------
    print("\n--- Training Interest Rate Regression Model ---")
    reg_rate = LinearRegression() # Linear Regression works well for this target based on our generation logic
    reg_rate.fit(X_train_r, y_train_rate)
    
    # Evaluate
    y_pred_rate = reg_rate.predict(X_test_r)
    print(f"R2 Score: {r2_score(y_test_rate, y_pred_rate):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_rate, y_pred_rate):.2f}")
    
    # Save Model
    joblib.dump(reg_rate, 'models/reg_rate.pkl')
    print("Saved Interest Rate Model.")
    
    # ---------------------------------------------------------
    # 4. Regression (Tenure)
    # ---------------------------------------------------------
    print("\n--- Training Tenure Regression Model ---")
    reg_ten = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_ten.fit(X_train_r, y_train_ten)
    
    # Evaluate
    y_pred_ten = reg_ten.predict(X_test_r)
    print(f"R2 Score: {r2_score(y_test_ten, y_pred_ten):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_ten, y_pred_ten):.2f}")
    
    # Save Model
    joblib.dump(reg_ten, 'models/reg_tenure.pkl')
    print("Saved Tenure Model.")
    
    print("\nAll models trained and saved successfully in the 'models/' directory.")

if __name__ == "__main__":
    train_and_evaluate()
