import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

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

# Targets
y_asthma = df["Asthma emergency department visits due to PM2.5"]
y_resp = df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

# PM2.5 as feature
X_pm25 = df[["Asthma emergency department visits due to PM2.5"]].rename(
    columns={"Asthma emergency department visits due to PM2.5": "PM2.5"}
)

# Train model
asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)
asthma_model.fit(X_pm25, y_asthma)
resp_model.fit(X_pm25, y_resp)

# Initialize scaler
scaler = MinMaxScaler()

# User input
st.subheader("🧪 Select Input Type")
input_type = st.selectbox("Use PM2.5 or test SO2/NO2/O3 as substitutes?", ["PM2.5", "SO2", "NO2", "O3"])

# Get user input value
pollutant_value = st.slider(f"{input_type} Level (µg/m³ or tons/year)", min_value=0.0, max_value=150.0, value=20.0)

# Scale the input value
scaled_value = scaler.fit_transform([[pollutant_value]])

# Predict
if st.button("🔮 Predict Health Impact"):
    input_data = [[scaled_value[0][0]]]  # Use the scaled value for prediction
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