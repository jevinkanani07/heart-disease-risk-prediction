import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# ================= LOAD ARTIFACTS =================
model = joblib.load("logistic_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ================= UI =================
st.title("❤️ Heart Disease Prediction By Jevin")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ================= PREDICTION =================
if st.button("Predict"):

    # -------- RAW INPUT --------
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1
    }

    # -------- DATAFRAME --------
    input_df = pd.DataFrame([raw_input])

    # -------- HANDLE MISSING COLUMNS --------
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # -------- SCALE --------
    input_scaled = scaler.transform(input_df)

    # -------- PREDICT --------
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # ================= RESULT =================
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(
            f"""
            **Risk Level: HIGH RISK**

            **Probability: {probability:.2f}%**

            ⚠️ This indicates a higher likelihood of heart disease.
            """
        )
    else:
        st.success(
            f"""
            **Risk Level: LOW RISK**

            **Probability: {probability:.2f}%**

            ✅ This indicates a lower likelihood of heart disease.
            """
        )

st.markdown("---")
st.markdown("**Developed by Jevin Kanani**  \nData Science | Machine Learning")




