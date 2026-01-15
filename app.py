import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Page Config ---
st.set_page_config(page_title="Advanced Customer Churn Predictor", layout="wide")

# --- Title and Description ---
st.title("🚀 Advanced Customer Churn Prediction")
st.write("""
This application uses Machine Learning to predict customer retention. 
By including contract types and security services, this version provides much higher accuracy.
""")

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    # Loading split dataset files
    df1 = pd.read_csv('Telco-Customer_1.csv')
    df2 = pd.read_csv('Telco-Customer_2.csv')
    df3 = pd.read_csv('Telco-Customer_3.csv')
    df4 = pd.read_csv('Telco-Customer_4.csv')
    
    # Merging all files
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    # Cleaning: Converting TotalCharges to numeric and dropping NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# --- Model Training with Enhanced Features ---
@st.cache_resource
def train_model(data):
    # Mapping categorical text to numbers for training
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['gender'] = data['gender'].map({'Female': 1, 'Male': 0})
    
    # Contract Mapping: Month-to-month=0, One year=1, Two year=2
    data['Contract'] = data['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    
    # OnlineSecurity Mapping: Yes=1, No/No internet=0
    data['OnlineSecurity'] = data['OnlineSecurity'].map({'No': 0, 'Yes': 1, 'No internet service': 0})

    # Selecting the top most important features
    features = ['tenure', 'MonthlyCharges', 'gender', 'Contract', 'OnlineSecurity']
    X = data[features]
    y = data['Churn']
    
    # Training the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Initialize training
model = train_model(df.copy())

# --- User Input Sidebar ---
st.sidebar.header("Customer Profile Settings")

def user_input_features():
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    tenure = st.sidebar.slider("Tenure (Months with company)", 0, 72, 12)
    
    st.sidebar.subheader("Service Details")
    contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
    security = st.sidebar.selectbox("Online Security Service", ("Yes", "No"))
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
    
    # Converting inputs to numeric format to match the model's training order
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'gender': 1 if gender == "Female" else 0,
        'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
        'OnlineSecurity': 1 if security == "Yes" else 0
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main Dashboard Display ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Customer Profile")
    st.write(input_df)

with col2:
    st.subheader("Prediction Analysis")
    if st.button("Predict Churn Risk"):
        # Making the prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"🚨 **High Risk!** This customer is likely to CHURN.")
        else:
            st.success(f"✅ **Safe!** This customer is likely to STAY.")
            
        # Showing the confidence level
        st.metric("Model Confidence", f"{np.max(prediction_proba)*100:.1f}%")

st.markdown("---")
st.subheader("📊 Why is this version more accurate?")
st.write("""
- **Contract Influence:** Data shows that customers on **Month-to-month** contracts have the highest churn rates.
- **Service Loyalty:** Customers who subscribe to additional services like **Online Security** are statistically more likely to stay.
- **Tenure:** Long-term customers (high tenure) show much stronger loyalty.
""")

st.info("Built with ❤️ for Customer Churn Analysis Portfolio")