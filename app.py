import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ========== Utility ==========
def get_health_tips(disease, level):
    tips = {
        "heart": {
            "Low": "Keep up healthy habits and regular checkups.",
            "High": "Consult a cardiologist and manage cholesterol/stress."
        },
        "hypertension": {
            "Low": "Maintain a low-sodium diet and regular activity.",
            "High": "Monitor BP, reduce salt, and follow medical advice."
        },
        "diabetes": {
            "Low": "Maintain your sugar levels and healthy weight.",
            "Medium": "Cut back on sugars, walk daily.",
            "High": "Seek medical guidance and manage diet strictly."
        }
    }
    return tips[disease].get(level, "No advice available.")

# ========== Train Models If Needed ==========
@st.cache_resource
def train_models():
    df = pd.read_csv("data/healthcare.csv")
    df = df.drop(columns=["id", "stroke"])
    df = df.dropna()

    # Add target labels
    df["diabetes_risk"] = df["avg_glucose_level"].apply(
        lambda x: "Low" if x < 140 else "Medium" if x < 200 else "High")
    df["heart_risk"] = df["heart_disease"].map({0: "Low", 1: "High"})
    df["hypertension_risk"] = df["hypertension"].map({0: "Low", 1: "High"})

    df = df.drop(columns=["heart_disease", "hypertension", "avg_glucose_level"])

    features = ["age", "bmi", "gender", "smoking_status", "ever_married", "Residence_type", "work_type"]
    label_encoders = {}

    X = df[features].copy()
    for col in X.select_dtypes(include='object'):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    models = {}
    for target, name in [
        ("heart_risk", "heart_model"),
        ("hypertension_risk", "hypertension_model"),
        ("diabetes_risk", "diabetes_model")
    ]:
        y = df[target]
        model = RandomForestClassifier()
        model.fit(X, y)
        models[name] = model

    return models, label_encoders, features

# ========== Load or Train ==========
models, label_encoders, features = train_models()

# ========== Streamlit UI ==========
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

# ========== Prepare Input ==========
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

# ========== Predict ==========
if st.button("🔍 Predict Health Risks"):
    st.subheader("🫀 Heart Disease")
    heart = models["heart_model"].predict(input_df)[0]
    st.write(f"**Risk:** {heart}")
    st.info(get_health_tips("heart", heart))

    st.subheader("💢 Hypertension")
    hyper = models["hypertension_model"].predict(input_df)[0]
    st.write(f"**Risk:** {hyper}")
    st.info(get_health_tips("hypertension", hyper))

    st.subheader("🩸 Diabetes")
    diab = models["diabetes_model"].predict(input_df)[0]
    st.write(f"**Risk:** {diab}")
    st.info(get_health_tips("diabetes", diab))
