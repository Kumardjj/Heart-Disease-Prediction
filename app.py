import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction System")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/DHEERAJ/Desktop/heart disease prediction using machine learning/heart_data.csv")
    return df

df = load_data()

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler

model, scaler = train_model()

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Patient Details")

user_input = {}
for col in X.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.number_input(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

input_df = pd.DataFrame([user_input])

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Prediction Result")

if st.button("Predict Heart Disease"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric("Risk Probability", f"{probability*100:.2f}%")
