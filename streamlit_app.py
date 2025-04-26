import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load all datasets
@st.cache_data
def load_data():
    health_data = pd.read_csv("merged_health_data.csv")
    no2_data = pd.read_csv("nitrogen_dioxide.csv")
    o3_data = pd.read_csv("ozone.csv")
    return health_data, no2_data, o3_data

health_df, no2_df, o3_df = load_data()
health_df = health_df.dropna()

st.title("🌍 Cross-Pollutant Health Impact Analysis")
st.write("""
Test if PM2.5-trained models respond similarly to other pollutants:
1. Model trained on PM2.5 health relationships
2. Apply to NO2/O3 levels as PM2.5 substitutes
""")

# Model setup
st.subheader("🧪 Select Pollutant Type")
pollutant_type = st.selectbox(
    "Choose pollutant to test:",
    options=["PM2.5", "NO2", "O3"]
)

# Initialize variables
current_data = health_df
unit = "µg/m³"
input_col = "PM2.5"

if pollutant_type == "NO2":
    current_data = no2_df.rename(columns={"data_value": "NO2"})
    current_data = current_data[current_data['name'] == "Nitrogen dioxide (NO2)"]
    unit = "ppb"
    input_col = "NO2"
elif pollutant_type == "O3":
    current_data = o3_df.rename(columns={"data_value": "O3"})
    current_data = current_data[current_data['name'] == "Ozone (O3)"]
    unit = "ppb"
    input_col = "O3"

# Get available locations and dates
locations = current_data['geo_place_name'].unique()
dates = current_data['start_date'].unique()

# Location/date selection
col1, col2 = st.columns(2)
with col1:
    selected_location = st.selectbox("📍 Select area", locations)
with col2:
    selected_date = st.selectbox("📅 Select date", dates)

# Get pollution range
filtered_data = current_data[
    (current_data['geo_place_name'] == selected_location) &
    (current_data['start_date'] == selected_date)
]

# Train PM2.5 model (from original data)
pm25_X = health_df[["PM2.5"]]
pm25_asthma = health_df["Asthma emergency department visits due to PM2.5"]
pm25_resp = health_df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

asthma_model = RandomForestRegressor(random_state=42).fit(pm25_X, pm25_asthma)
resp_model = RandomForestRegressor(random_state=42).fit(pm25_X, pm25_resp)

# Prediction interface
st.subheader("🔮 Prediction Simulation")
base_value = filtered_data[input_col].values[0] if not filtered_data.empty else 0

pollutant_value = st.slider(
    f"Adjust {pollutant_type} level ({unit})",
    min_value=0.0,
    max_value=50.0 if pollutant_type != "PM2.5" else 150.0,
    value=float(base_value),
    step=0.1
)

if st.button("Run Cross-Pollutant Prediction"):
    # Treat current pollutant value as PM2.5 equivalent
    input_data = [[pollutant_value]]
    
    asthma_pred = asthma_model.predict(input_data)[0]
    resp_pred = resp_model.predict(input_data)[0]
    
    # Display results
    st.success(f"🫁 Predicted Asthma ED Visits (PM2.5-equivalent model): **{asthma_pred:.1f}**/10k")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{resp_pred:.1f}**/10k")
    
    # Comparison visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual PM2.5 relationship
    ax.scatter(pm25_X, pm25_asthma, alpha=0.3, label="Actual PM2.5 Asthma Cases")
    ax.scatter(pm25_X, pm25_resp, alpha=0.3, label="Actual PM2.5 Resp. Cases")
    
    # Plot current prediction
    ax.scatter(pollutant_value, asthma_pred, s=200, marker="X", label="Current Asthma Prediction")
    ax.scatter(pollutant_value, resp_pred, s=200, marker="X", label="Current Resp. Prediction")
    
    ax.set_xlabel(f"{pollutant_type} Level ({unit})")
    ax.set_ylabel("Health Outcomes per 10k")
    ax.set_title(f"PM2.5 Model Response to {pollutant_type}")
    ax.legend()
    
    st.pyplot(fig)

# Data disclaimer
st.markdown("""
---
**🔍 Analysis Note:**  
This simulation assumes {pollutant_type} levels would impact health outcomes  
through the same biological pathways as PM2.5. Actual relationships may differ.
""")