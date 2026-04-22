import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath='data/loan_data.csv'):
    """Loads the dataset."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run generate_data.py first.")
        return None

def preprocess_data(df):
    """
    Preprocesses the data for training.
    Separates into classification features/target and regression features/targets.
    """
    # 1. Clean data (handle missing values if any, though synthetic has none)
    df = df.dropna()
    
    # 2. Encode Categorical Features
    # 'Previous Loan Taken' (Yes/No)
    df['Previous Loan Taken'] = df['Previous Loan Taken'].map({'Yes': 1, 'No': 0})
    
    # 'Eligible' (Eligible/Not Eligible)
    # We will use Eligible (1/0) as target for classification
    # For regression, we only train on Eligible == 1 rows
    df['Eligible_Target'] = df['Eligible'].map({'Eligible': 1, 'Not Eligible': 0})
    
    # Feature columns
    X_cols = ['Credit Score', 'Age', 'Annual Income', 'Previous Loan Taken']
    
    # Classification split
    X_class = df[X_cols]
    y_class = df['Eligible_Target']
    
    # Scale numerical features (Age, Income, Credit Score)
    scaler = StandardScaler()
    X_class_scaled = scaler.fit_transform(X_class)
    
    # Train/test split for classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Regression split (only eligible customers)
    df_eligible = df[df['Eligible_Target'] == 1]
    X_reg = df_eligible[X_cols]
    
    # Regression targets
    y_amount = df_eligible['Recommended Amount']
    y_rate = df_eligible['Interest Rate']
    y_tenure = df_eligible['Tenure (Months)']
    
    # Scale regression features
    # Note: We reuse the same scaler fitted on the entire dataset for consistency in production
    X_reg_scaled = scaler.transform(X_reg)
    
    # Train/test split for regression
    X_train_r, X_test_r, y_train_amt, y_test_amt, y_train_rate, y_test_rate, y_train_ten, y_test_ten = train_test_split(
        X_reg_scaled, y_amount, y_rate, y_tenure, test_size=0.2, random_state=42
    )
    
    return {
        'classification': (X_train_c, X_test_c, y_train_c, y_test_c),
        'regression': (X_train_r, X_test_r, y_train_amt, y_test_amt, y_train_rate, y_test_rate, y_train_ten, y_test_ten),
        'scaler': scaler,
        'feature_names': X_cols
    }
