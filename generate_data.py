import pandas as pd
import numpy as np
import os

def generate_synthetic_loan_data(num_samples=1000, random_state=42):
    """
    Generates a synthetic bank loan dataset.
    Features: Credit Score, Age, Annual Income, Previous Loan Taken
    Targets: Eligible (Classification), Recommended Amount (Regression), 
             Interest Rate (Regression), Tenure (Regression)
    """
    np.random.seed(random_state)
    
    # 1. Generate Features
    # Credit Score: 300 to 850, normally distributed around 650
    credit_scores = np.random.normal(loc=650, scale=100, size=num_samples)
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    # Age: 18 to 80
    ages = np.random.randint(18, 81, size=num_samples)
    
    # Annual Income: log-normal distribution, 10k to 200k+
    incomes = np.random.lognormal(mean=10.8, sigma=0.6, size=num_samples)
    incomes = np.clip(incomes, 10000, 300000).astype(int)
    
    # Previous Loan Taken (Yes=1, No=0)
    prev_loans = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.4, 0.6])
    
    # 2. Generate Targets
    
    # Target 1: Eligibility (Classification)
    # Higher chance if Credit Score > 600 and Income > 30000
    eligibility_scores = (credit_scores / 850) * 0.5 + (incomes / 300000) * 0.4 + (ages / 80) * 0.1
    # Add some randomness
    eligibility_scores += np.random.normal(0, 0.1, size=num_samples)
    
    # Threshold for eligibility
    eligible = (eligibility_scores > 0.45).astype(int)
    # Convert to Yes/No
    eligible_str = np.where(eligible == 1, 'Eligible', 'Not Eligible')
    
    # Target 2: Recommended Amount (Regression)
    # Around 20% to 40% of Annual Income, depending on credit score. If not eligible, amount is 0.
    base_amount_ratio = 0.2 + ((credit_scores - 300) / 550) * 0.2
    recommended_amounts = incomes * base_amount_ratio
    # Add noise
    recommended_amounts += np.random.normal(0, 5000, size=num_samples)
    recommended_amounts = np.clip(recommended_amounts, 1000, 100000).astype(int)
    # Ensure divisible by 100 for realistic loan amounts
    recommended_amounts = (recommended_amounts // 100) * 100
    # Set to 0 if not eligible
    recommended_amounts = np.where(eligible == 1, recommended_amounts, 0)
    
    # Target 3: Interest Rate (Regression)
    # Base rate of 15% minus up to 10% based on credit score. Range ~5% to 15%
    base_rate = 15.0 - ((credit_scores - 300) / 550) * 10.0
    # Add noise
    interest_rates = base_rate + np.random.normal(0, 0.5, size=num_samples)
    interest_rates = np.clip(interest_rates, 3.5, 20.0)
    interest_rates = np.round(interest_rates, 2)
    # Set to 0 if not eligible
    interest_rates = np.where(eligible == 1, interest_rates, 0.0)
    
    # Target 4: Tenure in months (Regression)
    # Standard tenures: 12, 24, 36, 48, 60. Longer tenure for higher amounts.
    tenures_months = 12 + (recommended_amounts / 100000) * 48
    # Add noise
    tenures_months += np.random.normal(0, 6, size=num_samples)
    # Snap to nearest 12 months for realism
    tenures_months = np.clip(np.round(tenures_months / 12) * 12, 12, 60).astype(int)
    # Set to 0 if not eligible
    tenures_months = np.where(eligible == 1, tenures_months, 0)
    
    # Create DataFrame
    data = {
        'Credit Score': credit_scores,
        'Age': ages,
        'Annual Income': incomes,
        'Previous Loan Taken': prev_loans,
        'Eligible': eligible_str,
        'Recommended Amount': recommended_amounts,
        'Interest Rate': interest_rates,
        'Tenure (Months)': tenures_months
    }
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    filepath = os.path.join('data', 'loan_data.csv')
    df.to_csv(filepath, index=False)
    print(f"Generated {num_samples} samples and saved to {filepath}")
    
    return df

if __name__ == "__main__":
    generate_synthetic_loan_data(num_samples=2500)
