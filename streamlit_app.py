import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="Air Quality Health Impact", layout="wide")

# Load datasets
@st.cache_data
def load_pm25():
    return pd.read_csv("fine_particles.csv")

@st.cache_data
def load_so2():
    return pd.read_csv("boiler_emissions.csv")

@st.cache_data
def load_o3():
    return pd.read_csv("ozone.csv")

@st.cache_data
def load_no2():
    return pd.read_csv("nitrogen_dioxide.csv")

df_pm25 = load_pm25().dropna()
df_so2 = load_so2().dropna()
df_o3 = load_o3().dropna()
df_no2 = load_no2().dropna()

# Display title
st.title("🌆 Predict Health Impacts of Pollutants Using PM2.5 Model")

# PM2.5-based model
X_pm25 = df_pm25[["PM2.5"]]
y_asthma = df_pm25["Asthma emergency department visits due to PM2.5"]
y_resp = df_pm25["Respiratory hospitalizations due to PM2.5 (age 20+)"]

asthma_model = RandomForestRegressor(random_state=0)
resp_model = RandomForestRegressor(random_state=0)
asthma_model.fit(X_pm25, y_asthma)
resp_model.fit(X_pm25, y_resp)

st.subheader("🔍 Enter PM2.5 value to predict health outcomes")
pm25_input = st.number_input("PM2.5 concentration (µg/m³)", min_value=0.0, value=10.0)

if st.button("Predict"):
    pred_asthma = asthma_model.predict([[pm25_input]])[0]
    pred_resp = resp_model.predict([[pm25_input]])[0]

    st.success(f"🫁 Predicted Asthma ED Visits: **{pred_asthma:.2f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{pred_resp:.2f}** per 10,000")

# Function to apply PM2.5 model to another pollutant
def predict_and_plot(df, pollutant_name, feature_col):
    X = df[[feature_col]]
    pred_asthma = asthma_model.predict(X)
    pred_resp = resp_model.predict(X)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(df["geo_place_name"], pred_asthma)
    ax[0].set_title(f"Asthma ED Visit Predictions using {pollutant_name}")
    ax[0].tick_params(axis='x', rotation=90)

    ax[1].bar(df["geo_place_name"], pred_resp, color='orange')
    ax[1].set_title(f"Respiratory Hospitalization Predictions using {pollutant_name}")
    ax[1].tick_params(axis='x', rotation=90)

    st.pyplot(fig)

# Show predictions for SO2, O3, NO2 using PM2.5-trained model
st.header("📊 Predict Health Impacts Using Other Pollutants")

with st.expander("💨 SO2 (Boiler Emissions)"):
    predict_and_plot(df_so2, "SO2", "Boiler Emissions- Total SO2 Emissions")

with st.expander("☁️ O3 (Ozone)"):
    predict_and_plot(df_o3, "O3", "Ozone")

with st.expander("🌫️ NO2 (Nitrogen Dioxide)"):
    predict_and_plot(df_no2, "NO2", "NO2")