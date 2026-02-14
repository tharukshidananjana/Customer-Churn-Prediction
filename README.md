# 📊 Customer Churn Prediction App

This is a **Machine Learning Web Application** built with Python and Streamlit to predict the likelihood of a customer leaving a service (Churn). It uses the Telco Customer dataset to analyze patterns and provide real-time risk assessments.

## 🚀 Live Demo :- https://telco-churn-analyzer.streamlit.app/

You can interact with the AI model by adjusting customer parameters such as **Tenure**, **Contract Type**, and **Monthly Charges** to see the prediction results instantly.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Streamlit (for the Web Interface)
- **ML Library:** Scikit-learn (Random Forest Classifier)
- **Data Handling:** Pandas & NumPy
- **Visualization:** Matplotlib & Seaborn

## 📂 Dataset Information
The project uses a combined dataset of Telco customers, merged from multiple sources to ensure a comprehensive analysis. Key features used for prediction include:
- **Tenure:** Number of months the customer has stayed with the company.
- **Contract:** The contract term (Month-to-month, One year, Two year).
- **Monthly Charges:** The amount charged to the customer monthly.
- **Online Security:** Whether the customer has online security or not.
- **Gender:** Customer's gender.

## 🧠 Model Logic & Insights
During the Exploratory Data Analysis (EDA), we identified that:
1. **Contract Type** is the strongest predictor; Month-to-month customers have the highest risk of leaving.
2. **Tenure** has an inverse relationship with Churn; long-term customers are more loyal.
3. **Online Security** subscribers tend to stay longer with the service.



## 🔧 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/tharukshidananjana/Customer-Churn-Prediction.git](https://github.com/tharukshidananjana/Customer-Churn-Prediction.git)
