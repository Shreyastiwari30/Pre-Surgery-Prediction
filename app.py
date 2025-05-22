import streamlit as st
import joblib
import numpy as np
import sklearn

# Load models
model_suitability = joblib.load("model_suitability.pkl")
model_mortality = joblib.load("model_mortality.pkl")
model_recovery = joblib.load("model_recovery.pkl")
model_stay = joblib.load("model_stay.pkl")

# Page configuration
st.set_page_config(page_title="Pre-Surgery Analyzer", page_icon="🧠", layout="centered")

# Header
st.title("🧠 Pre-Surgery Analyzer")
st.markdown(
    """
    <style>
    .big-font {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    "<div class='big-font'>Enter patient details to predict:</div>",
    unsafe_allow_html=True
)

st.markdown("""
- 🟢 Surgery Suitability  
- ⚰️ Mortality Risk  
- ⏱️ Recovery Time  
- 🏥 Hospital Stay Duration  
---
""")

# Input form
with st.form("prediction_form"):
    st.subheader("🩺 Patient Medical Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age", 18, 90, 40)
        blood_pressure = st.slider("🩸 Blood Pressure (mmHg)", 90, 180, 120)
        heart_rate = st.slider("❤️ Heart Rate (bpm)", 60, 120, 80)
        oxygen_saturation = st.slider("🫁 Oxygen Saturation (%)", 85.0, 100.0, 95.0)
        hemoglobin = st.slider("🧬 Hemoglobin (g/dL)", 10.0, 18.0, 14.0)
        wbc = st.slider("🧪 WBC Count (×10⁹/L)", 4.0, 11.0, 6.0)

    with col2:
        platelets = st.slider("🧫 Platelet Count (×10⁹/L)", 150.0, 450.0, 250.0)
        sugar = st.slider("🍬 Blood Sugar (mg/dL)", 70, 200, 100)
        bmi = st.slider("⚖️ BMI", 18.5, 35.0, 25.0)
        conditions = st.slider("📋 Prior Conditions", 0, 4, 1)
        smoking = st.radio("🚬 Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        alcohol = st.radio("🍷 Alcohol Use", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    with st.spinner("⏳ Predicting... Please wait"):
        features = np.array([
            age, blood_pressure, heart_rate, oxygen_saturation, hemoglobin,
            wbc, platelets, sugar, bmi, conditions, smoking, alcohol
        ]).reshape(1, -1)

        suitability = model_suitability.predict(features)[0]
        mortality = model_mortality.predict(features)[0]
        recovery_days = model_recovery.predict(features)[0]
        stay_days = model_stay.predict(features)[0]

    st.success("✅ Prediction Complete")

    # Results
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    st.markdown(f"**🟢 Surgery Suitability:** {'✅ Suitable' if suitability == 1 else '❌ Not Suitable'}")
    st.markdown(f"**⚰️ Mortality Risk:** {'⚠️ High Risk' if mortality == 1 else '🟢 Low Risk'}")
    st.markdown(f"**⏱️ Estimated Recovery Time:** {round(recovery_days)} days")
    st.markdown(f"**🏥 Estimated Hospital Stay:** {round(stay_days)} days")

    st.progress(min(1.0 - mortality, 1.0), text="🧬 Survival Likelihood")
    st.markdown("---")
st.caption("*Note: This is a synthetic project for educational/demo purposes.*\n\n Made by Shreyas Tiwari | 2025 ")
