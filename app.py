import streamlit as st
import pandas as pd
import time
from utils import load_models, make_prediction

# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
# We want to make it look premium
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .success-text {
        color: #28a745;
    }
    .danger-text {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏦 Smart Bank Loan Predictor")
    st.markdown("Enter your details in the sidebar to check your loan eligibility and get personalized loan recommendations.")
    
    # Load Models
    scaler, clf, reg_amt, reg_rate, reg_ten = load_models()
    
    if scaler is None:
        st.error("⚠️ Models not found! Please run `python train_models.py` first to generate the models.")
        st.stop()

    # --- Sidebar Inputs ---
    st.sidebar.header("Applicant Details")
    
    credit_score = st.sidebar.slider(
        "Credit Score", 
        min_value=300, max_value=850, value=650, step=1,
        help="A number between 300-850 that depicts a consumer's creditworthiness."
    )
    
    age = st.sidebar.slider(
        "Age", 
        min_value=18, max_value=80, value=35, step=1
    )
    
    annual_income = st.sidebar.number_input(
        "Annual Income ($)", 
        min_value=10000, max_value=1000000, value=60000, step=1000,
        help="Your total yearly income before taxes."
    )
    
    prev_loan = st.sidebar.radio(
        "Previous Loan Taken?", 
        options=["Yes", "No"], index=1,
        help="Have you taken a loan from our bank before?"
    )
    
    predict_btn = st.sidebar.button("Predict Eligibility 🚀", use_container_width=True, type="primary")
    
    # --- Main Content Area ---
    if predict_btn:
        with st.spinner("Analyzing application..."):
            time.sleep(1) # Add a slight delay for effect
            
            input_data = {
                'Credit Score': credit_score,
                'Age': age,
                'Annual Income': annual_income,
                'Previous Loan Taken': prev_loan
            }
            
            # Make Prediction
            results = make_prediction(scaler, clf, reg_amt, reg_rate, reg_ten, input_data)
            
            st.markdown("---")
            st.subheader("Results")
            
            if results['is_eligible']:
                st.success(f"🎉 **Congratulations!** You are **Eligible** for a loan. (Confidence: {results['eligibility_probability']:.1%})")
                
                st.markdown("### Recommended Loan Details")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Recommended Amount</div>
                        <div class="metric-value">${results['amount']:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Estimated Interest Rate</div>
                        <div class="metric-value">{results['rate']}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Recommended Tenure</div>
                        <div class="metric-value">{results['tenure']} Months</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.info("💡 **Tip:** These recommendations are based on your credit score and annual income. A higher credit score usually yields lower interest rates.")
            else:
                st.error(f"❌ **Unfortunately**, you are **Not Eligible** for a loan at this time. (Probability: {results['eligibility_probability']:.1%})")
                st.warning("💡 **Tip:** Try increasing your credit score or applying with a higher annual income. A credit score above 600 significantly improves your chances.")
                
    else:
        # Initial Dashboard View
        st.markdown("---")
        st.info("👈 Please fill out the form in the sidebar and click **Predict Eligibility**.")
        
        # Educational section
        st.markdown("### How it works")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **What we look at:**
            - **Credit Score:** The most important factor. Higher is better.
            - **Annual Income:** Determines how much you can borrow.
            - **Age & History:** Helps us understand your financial maturity.
            """)
        with col2:
            st.markdown("""
            **What we provide:**
            - Instant eligibility decision.
            - Data-driven recommended loan amount.
            - Fair interest rates tailored to your profile.
            """)

if __name__ == "__main__":
    main()
