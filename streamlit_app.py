import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="NYC Health Predictor", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("merged_health_data.csv")
    df.columns = df.columns.str.strip()
    return df.dropna()

df = load_data()

# Page title and description
st.title("🫁 NYC Health Rate Prediction")
st.write("This app predicts asthma emergency visits and respiratory hospitalizations based on pollutant levels (SO2, PM2.5, O3, NO2).")

# Sample data
st.write("### 📊 Sample Data")
st.dataframe(df.head())

# Column names
SO2_col = "Boiler Emissions- Total SO2 Emissions"
PM25_col = "PM2.5"
O3_col = "Ozone"
NO2_col = "NO2"
asthma_col = "Asthma emergency department visits due to PM2.5"
resp_col = "Respiratory hospitalizations due to PM2.5 (age 20+)"

# Function to train a model for a given feature
def train_model(feature_col):
    X = df[[feature_col]]
    y_asthma = df[asthma_col]
    y_resp = df[resp_col]
    model_asthma = RandomForestRegressor(random_state=0).fit(X, y_asthma)
    model_resp = RandomForestRegressor(random_state=0).fit(X, y_resp)
    return model_asthma, model_resp

# Train models per pollutant
so2_asthma, so2_resp = train_model(SO2_col)
pm25_asthma, pm25_resp = train_model(PM25_col)
o3_asthma, o3_resp = train_model(O3_col)
no2_asthma, no2_resp = train_model(NO2_col)

# Input sliders
st.subheader("🔧 Enter Pollution Levels")

so2_input = st.slider("SO2 Emissions (tons/year)", min_value=0.0, max_value=100.0, value=1.0)
pm25_input = st.slider("PM2.5 Concentration (µg/m³)", min_value=0.0, max_value=50.0, value=10.0)
o3_input = st.slider("Ozone (ppb)", min_value=0.0, max_value=100.0, value=30.0)
no2_input = st.slider("NO2 (ppb)", min_value=0.0, max_value=100.0, value=20.0)

# Predict on button
if st.button("Predict All Health Rates"):
    preds = {
        "Pollutant": [],
        "Asthma Rate": [],
        "Respiratory Rate": []
    }

    def add_prediction(pollutant, input_val, model_a, model_r):
        preds["Pollutant"].append(pollutant)
        preds["Asthma Rate"].append(model_a.predict([[input_val]])[0])
        preds["Respiratory Rate"].append(model_r.predict([[input_val]])[0])

    add_prediction("SO2", so2_input, so2_asthma, so2_resp)
    add_prediction("PM2.5", pm25_input, pm25_asthma, pm25_resp)
    add_prediction("Ozone", o3_input, o3_asthma, o3_resp)
    add_prediction("NO2", no2_input, no2_asthma, no2_resp)

    pred_df = pd.DataFrame(preds)

    st.success("✅ Predictions Generated!")
    st.write("### 📈 Predicted Health Rates by Pollutant")
    st.dataframe(pred_df)

    st.bar_chart(pred_df.set_index("Pollutant"))