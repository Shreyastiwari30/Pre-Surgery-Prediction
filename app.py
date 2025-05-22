import streamlit as st
import joblib
import numpy as np

# Load models
model_suitability = joblib.load("model_suitability.pkl")
model_mortality = joblib.load("model_mortality.pkl")
model_recovery = joblib.load("model_recovery.pkl")
model_stay = joblib.load("model_stay.pkl")

# App Configuration
st.set_page_config(
    page_title="Pre-Surgery Analyzer",
    layout="centered",
    page_icon="🧠"
)

# Custom CSS for background and card
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1588776814546-ec5d03c3b218');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    h1, h2, h3, h4 {
        color: #1f2f57;
    }

    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.title("🧠 Pre-Surgery Analyzer")
    st.markdown("""
    Welcome to the **Pre-Surgery Analyzer** – an AI-powered tool to assist in evaluating a patient's readiness for surgery.

    🔍 Enter patient details below to predict:
    - ✅ **Surgery Suitability**
    - ⚠️ **Mortality Risk**
    - ⏱️ **Recovery Time**
    - 🏥 **Hospital Stay Duration**
    """)
    st.markdown("---")

    # Form for input
    with st.form("prediction_form"):
        st.subheader("📋 Patient Medical Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("🎂 Age", 18, 90, 40)
            blood_pressure = st.slider("🩸 Blood Pressure (mmHg)", 90, 180, 120)
            heart_rate = st.slider("❤️ Heart Rate (bpm)", 60, 120, 80)
            oxygen_saturation = st.slider("🌬️ Oxygen Saturation (%)", 85.0, 100.0, 95.0)
            hemoglobin = st.slider("🧪 Hemoglobin (g/dL)", 10.0, 18.0, 14.0)
            wbc = st.slider("🧫 WBC Count (×10⁹/L)", 4.0, 11.0, 6.0)

        with col2:
            platelet = st.slider("🩻 Platelet Count (×10⁹/L)", 150.0, 450.0, 250.0)
            sugar = st.slider("🍬 Blood Sugar (mg/dL)", 70, 200, 100)
            bmi = st.slider("⚖️ BMI", 18.5, 35.0, 25.0)
            conditions = st.slider("📋 Prior Conditions", 0, 4, 1)
            smoking = st.radio("🚬 Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            alcohol = st.radio("🍷 Alcohol Use", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        submitted = st.form_submit_button("🚀 Predict Now")

    # Prediction and Output
    if submitted:
        input_data = np.array([
            age, blood_pressure, heart_rate, oxygen_saturation, hemoglobin,
            wbc, platelet, sugar, bmi, conditions, smoking, alcohol
        ]).reshape(1, -1)

        # Predictions
        suitability = model_suitability.predict(input_data)[0]
        mortality = model_mortality.predict(input_data)[0]
        recovery_days = model_recovery.predict(input_data)[0]
        hospital_days = model_stay.predict(input_data)[0]

        st.success("✅ Prediction Successful!")

        st.markdown("---")
        st.subheader("🔎 Prediction Results")

        st.markdown(f"**Surgery Suitability:** {'🟢 Suitable' if suitability == 1 else '🔴 Not Suitable'}")
        st.markdown(f"**Mortality Risk:** {'⚠️ High Risk' if mortality == 1 else '✅ Low Risk'}")
        st.markdown(f"**Estimated Recovery Time:** ⏱️ {round(recovery_days)} days")
        st.markdown(f"**Estimated Hospital Stay:** 🏥 {round(hospital_days)} days")

        st.markdown("---")
        st.caption("*Note: Predictions are based on synthetic data for demonstration purposes.*")
