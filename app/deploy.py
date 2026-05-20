# app/deploy.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add project root to path so we can import src if needed, but not strictly required
sys.path.append(str(Path(__file__).resolve().parent.parent))

MODEL_PATH = "models/best_model.pkl"

# Ranges derived from training data (you would compute these from actual data and store in config)
VALID_RANGES = {
    "age": (0, 100),
    "avg_glucose_level": (55, 300),
    "bmi": (10, 65)
}
WARNING_RANGES = {
    "age": (20, 85),
    "avg_glucose_level": (70, 200),
    "bmi": (18.5, 45)
}

def main():
    st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
    st.title("🧠 Stroke Risk Prediction")
    st.markdown("This app predicts the probability of stroke based on patient data.")

    # Load model
    @st.cache_resource
    def load_model():
        if not Path(MODEL_PATH).exists():
            st.error(f"Model file not found at {MODEL_PATH}. Please train first.")
            st.stop()
        return joblib.load(MODEL_PATH)
    
    pipeline = load_model()

    with st.form("input_form"):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1], format_func=lambda x: "Yes" if x else "No")
            heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1], format_func=lambda x: "Yes" if x else "No")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"])

        with col2:
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=65.0, value=25.0)
            smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

        submitted = st.form_submit_button("Predict Stroke Risk")

    if submitted:
        # Input validation warnings
        warnings = []
        if age < WARNING_RANGES["age"][0] or age > WARNING_RANGES["age"][1]:
            warnings.append(f"Age {age} is outside typical range ({WARNING_RANGES['age'][0]}-{WARNING_RANGES['age'][1]}).")
        if avg_glucose < WARNING_RANGES["avg_glucose_level"][0] or avg_glucose > WARNING_RANGES["avg_glucose_level"][1]:
            warnings.append(f"Glucose {avg_glucose} is outside typical range ({WARNING_RANGES['avg_glucose_level'][0]}-{WARNING_RANGES['avg_glucose_level'][1]}).")
        if bmi < WARNING_RANGES["bmi"][0] or bmi > WARNING_RANGES["bmi"][1]:
            warnings.append(f"BMI {bmi} is outside typical range ({WARNING_RANGES['bmi'][0]}-{WARNING_RANGES['bmi'][1]}).")
        
        for w in warnings:
            st.warning(w)

        # Build input DataFrame with exactly the original column names (as used in training)
        input_data = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "residence_type": residence_type,
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }])

        # Get prediction probability
        proba = pipeline.predict_proba(input_data)[0, 1]
        risk_level = "High" if proba >= 0.5 else "Low"

        st.subheader("Result")
        st.metric("Stroke Probability", f"{proba:.2%}")
        st.write(f"Risk Level: **{risk_level}**")

        if risk_level == "High":
            st.error("Patient is at high risk of stroke. Please consult a healthcare professional.")
        else:
            st.success("Patient is at low risk of stroke.")

if __name__ == "__main__":
    main()