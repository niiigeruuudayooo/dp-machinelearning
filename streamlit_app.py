import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the full health dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_health_data.csv")

df = load_data()

# Drop NAs (in case any)
df = df.dropna()

# Display sample
st.title("🫁 NYC Health Prediction from SO2 Emissions")
st.write("### Sample of dataset")
st.dataframe(df.head())

# Feature and targets
X = df[["Boiler Emissions- Total SO2 Emissions"]]
y_asthma = df["Asthma emergency department visits due to PM2.5"]
y_resp = df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

# Train models
asthma_model = RandomForestRegressor(random_state=0)
resp_model = RandomForestRegressor(random_state=0)

asthma_model.fit(X, y_asthma)
resp_model.fit(X, y_resp)

# User Input
st.subheader("🔢 Enter SO2 Emission Level (tons/year)")
so2_input = st.number_input("Boiler Emissions", min_value=0.0, value=1.0)

# Predict on click
if st.button("Predict Health Rates"):
    input_val = [[so2_input]]
    asthma_pred = asthma_model.predict(input_val)[0]
    resp_pred = resp_model.predict(input_val)[0]

    st.success(f"🌬️ Predicted Asthma ED Visits Rate: **{asthma_pred:.2f}** per 10,000")
    st.success(f"🫁 Predicted Respiratory Hospitalization Rate: **{resp_pred:.2f}** per 10,000")