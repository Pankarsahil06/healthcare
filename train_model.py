import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load and clean original dataset
df = pd.read_csv("data/healthcare.csv")

# Drop missing values and unnecessary columns
df = df.drop(columns=["id", "stroke"])
df = df.dropna()

# Derive risk columns
df["diabetes_risk"] = df["avg_glucose_level"].apply(
    lambda x: "Low" if x < 140 else "Medium" if x < 200 else "High"
)
df["heart_risk"] = df["heart_disease"].map({0: "Low", 1: "High"})
df["hypertension_risk"] = df["hypertension"].map({0: "Low", 1: "High"})

# Drop raw target columns after encoding risk labels
df = df.drop(columns=["heart_disease", "hypertension", "avg_glucose_level"])

# Define features and encode categorical columns
features = ["age", "bmi", "gender", "smoking_status", "ever_married", "Residence_type", "work_type"]
label_encoders = {}

X = df[features].copy()
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train models
def train_model(target, model_name):
    y = df[target]
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, f"models/{model_name}.pkl")

train_model("heart_risk", "heart_model")
train_model("hypertension_risk", "hypertension_model")
train_model("diabetes_risk", "diabetes_model")
joblib.dump(label_encoders, "models/label_encoders.pkl")
