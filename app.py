import streamlit as st
import pandas as pd
import joblib
from utils import get_health_tips

st.set_page_config("Health Risk Predictor", layout="centered")

st.title("🏥 Multi-Disease Health Risk Predictor")
st.markdown("Enter your health info to check risk levels:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
bmi = st.slider("BMI", 10.0, 50.0, 22.0)
smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
married = st.selectbox("Ever Married", ["Yes", "No"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])
work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

label_encoders = joblib.load("models/label_encoders.pkl")
features = ["age", "bmi", "gender", "smoking_status", "ever_married", "Residence_type", "work_type"]

input_dict = {
    "age": age,
    "bmi": bmi,
    "gender": label_encoders["gender"].transform([gender])[0],
    "smoking_status": label_encoders["smoking_status"].transform([smoking])[0],
    "ever_married": label_encoders["ever_married"].transform([married])[0],
    "Residence_type": label_encoders["Residence_type"].transform([residence])[0],
    "work_type": label_encoders["work_type"].transform([work])[0]
}

input_df = pd.DataFrame([input_dict])[features]


models = {
    "heart": joblib.load("models/heart_model.pkl"),
    "hypertension": joblib.load("models/hypertension_model.pkl"),
    "diabetes": joblib.load("models/diabetes_model.pkl")
}

if st.button("🔍 Predict Health Risks"):
    st.subheader("🫀 Heart Disease")
    heart = models["heart"].predict(input_df)[0]
    st.write(f"**Risk:** {heart}")
    st.info(get_health_tips("heart", heart))

    st.subheader("💢 Hypertension")
    hyper = models["hypertension"].predict(input_df)[0]
    st.write(f"**Risk:** {hyper}")
    st.info(get_health_tips("hypertension", hyper))

    st.subheader("🩸 Diabetes")
    diab = models["diabetes"].predict(input_df)[0]
    st.write(f"**Risk:** {diab}")
    st.info(get_health_tips("diabetes", diab))
