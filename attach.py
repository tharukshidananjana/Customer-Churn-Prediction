import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- STEP 1: Data Loading & Merging ---
# Loading the split dataset files
df1 = pd.read_csv('Telco-Customer_1.csv')
df2 = pd.read_csv('Telco-Customer_2.csv')
df3 = pd.read_csv('Telco-Customer_3.csv')
df4 = pd.read_csv('Telco-Customer_4.csv')

# Merging all dataframes into one single dataframe
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Converting TotalCharges to a numeric format (handling empty strings as NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# --- STEP 2: Data Cleaning ---
# Removing rows with missing values (NaN) to ensure a clean dataset
df.dropna(inplace=True)

# Dropping 'customerID' as it is a unique identifier and doesn't help in prediction
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)


# --- STEP 3: Exploratory Data Analysis (Visualization) ---
# Visualizing how many customers stayed vs. how many left
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='magma')
plt.title('Distribution of Customer Churn')
plt.show()

# Visualizing Churn based on the type of Contract
plt.figure(figsize=(10, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='viridis')
plt.title('Churn Count by Contract Type')
plt.show()


# --- STEP 4: Feature Encoding ---
# Converting categorical text into numerical values for the Machine Learning model

# 1. Binary Encoding (Mapping Yes/No and Gender to 1 and 0)
binary_mapping = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
cols_to_map = ['Churn', 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

for col in cols_to_map:
    # Check if the column is still a string/object before mapping to avoid errors
    if df[col].dtype == 'O': 
        df[col] = df[col].map(binary_mapping)

# 2. One-Hot Encoding (Creating dummy columns for features with multiple categories)
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaymentMethod']

df_final = pd.get_dummies(df, columns=categorical_cols)

print("\n--- Preprocessing Complete ---")
print(f"Total features after encoding: {df_final.shape[1]}")


# --- STEP 5: Model Training & Evaluation ---
# Splitting the data into Features (X) and Target Label (y)
X = df_final.drop(['Churn'], axis=1)
y = df_final['Churn']

# Dividing the dataset: 80% for Training and 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

print("\n--- Training the AI Model ---")
rf_model.fit(X_train, y_train)

# Making predictions on the unseen test data
y_pred = rf_model.predict(X_test)

# Displaying accuracy and the classification report
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --- STEP 6: Feature Importance Analysis ---
# Identifying which factors are the biggest drivers of customer churn
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:] # Taking top 10 important features



plt.figure(figsize=(10, 6))
plt.title('Top 10 Factors Influencing Customer Churn')
plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance Score')
plt.show()