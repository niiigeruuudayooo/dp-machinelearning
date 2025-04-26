import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load merged health dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_health_data.csv")

df = load_data().dropna()

st.title("🌫️ PM2.5 Health Impact Prediction + Surrogate Pollutant Test")
st.write("""
This app predicts:
- 🫁 Asthma emergency department visits  
- 🏥 Respiratory hospitalizations  

based on PM2.5 levels — and optionally tests SO2, NO2, or O3 as substitutes.
""")

# Corrected feature and targets
X_pm25 = df[["PM2.5"]]  # Assuming "PM2.5" is the column with pollutant levels
y_asthma = df["Asthma emergency department visits due to PM2.5"]
y_resp = df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

# Train model
asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)
asthma_model.fit(X_pm25, y_asthma)
resp_model.fit(X_pm25, y_resp)

# User input
st.subheader("🧪 Select Input Type")
input_type = st.selectbox("Use PM2.5 or test SO2/NO2/O3 as substitutes?", ["PM2.5", "SO2", "NO2", "O3"])

# Dynamically adjust slider ranges based on pollutant (example ranges)
pollutant_ranges = {
    "PM2.5": (0.0, 150.0, 20.0),
    "SO2": (0.0, 100.0, 15.0),
    "NO2": (0.0, 200.0, 30.0),
    "O3": (0.0, 300.0, 50.0)
}
min_val, max_val, default_val = pollutant_ranges[input_type]

pollutant_value = st.slider(
    f"{input_type} Level (µg/m³)", 
    min_value=min_val, 
    max_value=max_val, 
    value=default_val
)

# Predict
if st.button("🔮 Predict Health Impact"):
    # Treat surrogate pollutant value as PM2.5 input
    input_data = [[pollutant_value]]
    pred_asthma = asthma_model.predict(input_data)[0]
    pred_resp = resp_model.predict(input_data)[0]

    st.success(f"🫁 Predicted Asthma ED Visits (using {input_type}): **{pred_asthma:.2f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalizations (using {input_type}): **{pred_resp:.2f}** per 10,000")

    # Chart
    st.subheader("📈 Prediction Summary")
    st.bar_chart({
        "Health Outcome": ["Asthma ED Visits", "Respiratory Hospitalizations"],
        "Prediction": [pred_asthma, pred_resp]
    })

# Optional: Show dataset
with st.expander("🗂️ View Data Sample"):
    st.dataframe(df.head())