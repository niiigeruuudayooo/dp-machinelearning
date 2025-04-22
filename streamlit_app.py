# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Load and Prepare Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("combined_pollution_health_data.csv")
    df = df.dropna()

    # Rename for easier reference
    df = df.rename(columns={
        "Asthma emergency department visits due to PM2.5": "asthma_rate",
        "Respiratory hospitalizations due to PM2.5 (age 20+)": "respiratory_rate",
        "Fine particles (PM 2.5)": "PM2.5",
        "Nitrogen dioxide (NO2)": "NO2",
        "Ozone (O3)": "O3",
        "Boiler Emissions- Total SO2 Emissions": "SO2"
    })

    return df

df = load_data()

# ---- Train Models ----
X = df[["PM2.5", "NO2", "O3", "SO2"]]
y_asthma = df["asthma_rate"]
y_resp = df["respiratory_rate"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_a_train, y_a_test = train_test_split(X_scaled, y_asthma, test_size=0.2, random_state=42)
_, _, y_r_train, y_r_test = train_test_split(X_scaled, y_resp, test_size=0.2, random_state=42)

asthma_model = RandomForestRegressor(n_estimators=100, random_state=42)
asthma_model.fit(X_train, y_a_train)

resp_model = RandomForestRegressor(n_estimators=100, random_state=42)
resp_model.fit(X_train, y_r_train)

# ---- Streamlit App UI ----
st.title("🏥 NYC Health Impact Predictor")
st.markdown("Enter pollutant levels to predict asthma and respiratory hospitalization rates in NYC neighborhoods.")

# Sidebar inputs
pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 30.0, 10.0)
no2 = st.slider("NO2 (ppb)", 0.0, 60.0, 20.0)
o3 = st.slider("O3 (ppb)", 0.0, 60.0, 25.0)
so2 = st.slider("SO2 (Boiler Emissions)", 0.0, 100.0, 20.0)

# Prediction
user_input = np.array([[pm25, no2, o3, so2]])
user_input_scaled = scaler.transform(user_input)

asthma_pred = asthma_model.predict(user_input_scaled)[0]
resp_pred = resp_model.predict(user_input_scaled)[0]

# Results
st.subheader("📈 Predicted Health Outcomes:")
st.metric("Asthma ED Visit Rate", f"{asthma_pred:.2f}")
st.metric("Respiratory Hospitalization Rate", f"{resp_pred:.2f}")