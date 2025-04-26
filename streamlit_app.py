import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="NYC Health Predictor", layout="centered")

@st.cache_data
def load_all():
    def clean_columns(df):
        df.columns = df.columns.str.strip()
        return df.dropna()

    so2_df = pd.read_csv("boiler_emissions.csv")
    no2_df = pd.read_csv("nitrogen_dioxide.csv")
    o3_df = pd.read_csv("ozone.csv")
    pm25_df = pd.read_csv("fine_particles.csv")

    return clean_columns(so2_df), clean_columns(no2_df), clean_columns(o3_df), clean_columns(pm25_df)

so2_df, no2_df, o3_df, pm25_df = load_all()

# Sidebar selection
st.sidebar.title("🌫️ Select Pollutant")
pollutant = st.sidebar.selectbox("Choose pollutant to predict health impact from:", ["SO2", "NO2", "O3", "PM2.5"])

# Load selected data
if pollutant == "SO2":
    df = so2_df
    X_col = "Boiler Emissions- Total SO2 Emissions"
elif pollutant == "NO2":
    df = no2_df
    X_col = "NO2"
elif pollutant == "O3":
    df = o3_df
    X_col = "Ozone"
elif pollutant == "PM2.5":
    df = pm25_df
    X_col = "PM2.5"

# Show data sample
st.title("🩺 NYC Health Rate Predictor")
st.write(f"### Sample data ({pollutant})")
st.dataframe(df[[X_col, "Asthma emergency department visits due to PM2.5", "Respiratory hospitalizations due to PM2.5 (age 20+)"]].head())

# Train models
X = df[[X_col]]
y_asthma = df["Asthma emergency department visits due to PM2.5"]
y_resp = df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)

asthma_model.fit(X, y_asthma)
resp_model.fit(X, y_resp)

# Input
st.subheader(f"📥 Enter {pollutant} Level")
pollutant_input = st.number_input(f"{pollutant} Level", min_value=0.0, value=1.0)

# Predict
if st.button("🔮 Predict Health Outcomes"):
    input_val = [[pollutant_input]]
    asthma_pred = asthma_model.predict(input_val)[0]
    resp_pred = resp_model.predict(input_val)[0]

    st.success(f"🫁 Predicted Asthma ED Visits Rate: **{asthma_pred:.2f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalization Rate: **{resp_pred:.2f}** per 10,000")

    # Chart
    st.write("### 📊 Prediction Comparison")
    chart_data = pd.DataFrame({
        "Health Outcome": ["Asthma", "Respiratory"],
        "Rate per 10,000": [asthma_pred, resp_pred]
    })
    st.bar_chart(chart_data.set_index("Health Outcome"))