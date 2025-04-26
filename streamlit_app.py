import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load and cache the data
@st.cache_data
def load_data():
    return pd.read_csv("merged_health_data.csv")

df = load_data()
df = df.dropna()

# App title
st.title("🌆 NYC Health Risk Predictor from Air Pollution")
st.write("Use air quality indicators to predict asthma and respiratory hospitalization rates.")

# Select features and targets
features = ['Boiler Emissions- Total SO2 Emissions', 'PM2.5', 'Ozone', 'NO2']
target_asthma = 'Asthma emergency department visits due to PM2.5'
target_respiratory = 'Respiratory hospitalizations due to PM2.5 (age 20+)'

X = df[features]
y_asthma = df[target_asthma]
y_respiratory = df[target_respiratory]

# Train models
asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)
asthma_model.fit(X, y_asthma)
resp_model.fit(X, y_respiratory)

# Sidebar for input
st.sidebar.header("🔧 Enter Pollution Values")
so2 = st.sidebar.slider("SO2 Emissions (tons/year)", 0.0, 60.0, 10.0)
pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 20.0, 8.0)
o3 = st.sidebar.slider("Ozone (O3) (ppb)", 0.0, 40.0, 15.0)
no2 = st.sidebar.slider("NO2 (ppb)", 0.0, 50.0, 20.0)

input_data = pd.DataFrame([[so2, pm25, o3, no2]], columns=features)

# Predict
if st.button("📈 Predict Health Outcomes"):
    pred_asthma = asthma_model.predict(input_data)[0]
    pred_resp = resp_model.predict(input_data)[0]

    st.success(f"🫁 Predicted Asthma ED Visits Rate: **{pred_asthma:.2f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalization Rate: **{pred_resp:.2f}** per 10,000")

    # Visualization
    fig, ax = plt.subplots()
    ax.bar(['Asthma ED Visits', 'Respiratory Hospitalizations'], [pred_asthma, pred_resp], color=['skyblue', 'salmon'])
    ax.set_ylabel("Rate per 10,000 people")
    ax.set_title("📊 Predicted Health Outcomes")
    st.pyplot(fig)