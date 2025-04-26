import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load datasets
@st.cache_data
def load_data():
    merged_health = pd.read_csv("merged_health_data.csv")
    pm25 = pd.read_csv("fine_particles.csv")
    so2 = pd.read_csv("boiler_emissions.csv")
    no2 = pd.read_csv("nitrogen_dioxide.csv")
    o3 = pd.read_csv("ozone.csv")
    return merged_health, pm25, so2, no2, o3

merged_health, pm25, so2, no2, o3 = load_data()

# Filter pollutant datasets
pm25 = pm25[pm25["name"] == "Fine particles (PM 2.5)"]
so2 = so2[so2["name"] == "Boiler Emissions- Total SO2 Emissions"]
no2 = no2[no2["name"] == "Nitrogen Dioxide (NO2)"]
o3 = o3[o3["name"] == "Ozone (O3)"]

# Drop NA in health data
merged_health = merged_health.dropna()

# Prepare training data using PM2.5 as X, health as y
X_pm25 = merged_health[["Fine particles (PM2.5)"]]
y_asthma = merged_health["Asthma emergency department visits due to PM2.5"]
y_resp = merged_health["Respiratory hospitalizations due to PM2.5 (age 20+)"]

# Train models
asthma_model = RandomForestRegressor(random_state=0).fit(X_pm25, y_asthma)
resp_model = RandomForestRegressor(random_state=0).fit(X_pm25, y_resp)

# Streamlit UI
st.title("🌆 NYC Health Predictions Based on Pollutant Levels")
st.markdown("Models are trained on PM2.5 data and applied to other pollutants for comparison.")

# Show SO2 predictions
st.subheader("🔻 Predicted Health Rates from SO2")
so2_vals = so2[["geo_place_name", "start_date", "data_value"]].dropna()
so2_vals["asthma_pred"] = asthma_model.predict(so2_vals[["data_value"]])
so2_vals["resp_pred"] = resp_model.predict(so2_vals[["data_value"]])
st.dataframe(so2_vals.head())

# Show NO2 predictions
st.subheader("🔻 Predicted Health Rates from NO2")
no2_vals = no2[["geo_place_name", "start_date", "data_value"]].dropna()
no2_vals["asthma_pred"] = asthma_model.predict(no2_vals[["data_value"]])
no2_vals["resp_pred"] = resp_model.predict(no2_vals[["data_value"]])
st.dataframe(no2_vals.head())

# Show O3 predictions
st.subheader("🔻 Predicted Health Rates from O3")
o3_vals = o3[["geo_place_name", "start_date", "data_value"]].dropna()
o3_vals["asthma_pred"] = asthma_model.predict(o3_vals[["data_value"]])
o3_vals["resp_pred"] = resp_model.predict(o3_vals[["data_value"]])
st.dataframe(o3_vals.head())