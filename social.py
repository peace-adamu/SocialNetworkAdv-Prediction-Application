import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from sklearn.preprocessing import LabelEncoder

# Load model and preprocessing objects
model = joblib.load("xgb_smote_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("encoder.joblib")  # This should be a fitted LabelEncoder object

# Prediction functions
def predict_purchase(age, salary, gender):
    try:
        gender_encoded = 1 if gender.lower() == 'male' else 0
        input_data = np.array([[age, salary, gender_encoded]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return prediction
    except Exception as e:
        return f"Error: {e}"


def batch_predict(dataframe):
    try:
        dataframe['Gender'] = label_encoder.transform(dataframe['Gender'])
        scaled_data = scaler.transform(dataframe[['Age', 'EstimatedSalary', 'Gender']])
        predictions = model.predict(scaled_data)
        dataframe['Prediction'] = predictions
        return dataframe
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI Configuration
st.set_page_config(page_title="Social Network Purchase Predictor", layout="wide")

# App Introduction
st.title("🛒 Social Network Purchase Predictor")
st.markdown("""
Welcome to the **Social Network Purchase Predictor** app!

This tool uses machine learning to predict whether a user will purchase a product based on:
- **Age**
- **Estimated Salary**
- **Gender**

👉 Use the **sidebar** to choose between:
- **🔍 Single Prediction**: Enter a user's information manually
- **📂 Batch Prediction**: Upload a CSV file with multiple users
""")

# Sidebar Navigation
st.sidebar.title("🔧 App Modes")
mode = st.sidebar.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction"])

# Single Prediction Mode
if mode == "Single Prediction":
    st.subheader("🔍 Single User Prediction")
    st.markdown("Enter user details to check if they are likely to purchase.")

    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    salary = st.number_input("Estimated Salary", min_value=1000, max_value=1000000, step=1000)
    gender = st.selectbox("Gender", options=["Male", "Female"])

    if st.button("Predict"):
        result = predict_purchase(age, salary, gender)
        if result == 1:
            st.success("✅ The user is likely to PURCHASE.")
        elif result == 0:
            st.info("❌ The user is NOT likely to purchase.")
        else:
            st.error(result)

# Batch Prediction Mode
elif mode == "Batch Prediction":
    st.subheader("📂 Batch Prediction")
    st.markdown("Upload a CSV file with columns: `Age`, `EstimatedSalary`, and `Gender`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data", df.head())

        result_df = batch_predict(df)
        if isinstance(result_df, pd.DataFrame):
            st.write("### Predictions", result_df)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        else:
            st.error(result_df)
