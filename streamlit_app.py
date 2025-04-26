import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset (merged_health_data.csv)
@st.cache_data
def load_data():
    return pd.read_csv("merged_health_data.csv")

df = load_data()

# Drop missing values
df = df.dropna()

# Title and intro
st.title("🌫️ PM2.5 Health Impact Prediction")
st.write("Predict asthma ED visits and respiratory hospitalizations based on PM2.5 levels.")

# Display a sample of the data
st.subheader("📊 Data Sample")
st.dataframe(df.head())

# Prepare training data
X = df[["Asthma emergency department visits due to PM2.5"]]  # This will be target soon
y_asthma = df["Asthma emergency department visits due to PM2.5"]
y_resp = df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

# Use PM2.5 as a feature
X_pm25 = df[["Asthma emergency department visits due to PM2.5"]].rename(
    columns={"Asthma emergency department visits due to PM2.5": "PM2.5"}
)

# Train models
asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)

asthma_model.fit(X_pm25, y_asthma)
resp_model.fit(X_pm25, y_resp)

# User input section
st.subheader("🧪 Input PM2.5 Level")
pm25_input = st.slider("PM2.5 Concentration (µg/m³)", min_value=0.0, max_value=150.0, value=20.0)

# Predict button
if st.button("Predict"):
    input_data = [[pm25_input]]
    pred_asthma = asthma_model.predict(input_data)[0]
    pred_resp = resp_model.predict(input_data)[0]

    st.success(f"🫁 Predicted Asthma ED Visits: **{pred_asthma:.2f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{pred_resp:.2f}** per 10,000")

    # Optional: chart to compare input vs predicted output
    st.subheader("📈 Prediction Summary")
    st.bar_chart({
        "Health Outcome": ["Asthma ED Visits", "Respiratory Hospitalizations"],
        "Prediction": [pred_asthma, pred_resp]
    })