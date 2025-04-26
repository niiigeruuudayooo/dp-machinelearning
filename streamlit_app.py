import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load and merge data
@st.cache_data
def load_data():
    # Load base health data
    health_df = pd.read_csv("merged_health_data.csv")

    # Load pollutant datasets
    o3 = pd.read_csv("ozone.csv").rename(columns={"data_value": "O3"})
    no2 = pd.read_csv("nitrogen_dioxide.csv").rename(columns={"data_value": "NO2"})
    pm25 = pd.read_csv("fine_particles.csv").rename(columns={"data_value": "PM2.5"})

    # Merge pollutant data into health dataset
    for pollutant_df in [o3, no2, pm25]:
        health_df = health_df.merge(
            pollutant_df[["geo_place_name", "start_date", pollutant_df.columns[-1]]],
            on=["geo_place_name", "start_date"],
            how="left"
        )

    return health_df.dropna()

# Load full dataset
df = load_data()

# Title and preview
st.title("🫁 Predicting Health Impacts from Air Pollutants")
st.markdown("This tool uses PM2.5 to train a model predicting asthma and respiratory rates, then simulates predictions using other pollutants (O3, NO2, SO2).")
st.write("### Sample Data")
st.dataframe(df.head())

# Train models based on PM2.5 only
X_pm25 = df[["PM2.5"]]
y_asthma = df["Asthma emergency department visits due to PM2.5"]
y_resp = df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

asthma_model = RandomForestRegressor(random_state=0)
resp_model = RandomForestRegressor(random_state=0)

asthma_model.fit(X_pm25, y_asthma)
resp_model.fit(X_pm25, y_resp)

# User inputs
st.subheader("🔢 Enter Pollutant Levels (μg/m³ or tons/year)")
pm25_input = st.slider("PM2.5", 0.0, 50.0, 10.0)
no2_input = st.slider("NO2", 0.0, 50.0, 10.0)
o3_input = st.slider("O3", 0.0, 50.0, 10.0)
so2_input = st.slider("SO2 (tons/year)", 0.0, 50.0, 10.0)

# Predict and visualize
if st.button("🚀 Predict Health Outcomes"):
    def predict_from(val, label):
        asthma = asthma_model.predict([[val]])[0]
        resp = resp_model.predict([[val]])[0]
        st.write(f"**{label}** → Asthma: `{asthma:.2f}`, Respiratory: `{resp:.2f}`")
        return {"Pollutant": label, "Asthma": asthma, "Respiratory": resp}

    st.markdown("### 📊 Prediction Results")
    results = [
        predict_from(pm25_input, "PM2.5"),
        predict_from(no2_input, "NO2"),
        predict_from(o3_input, "O3"),
        predict_from(so2_input, "SO2"),
    ]

    # Chart
    results_df = pd.DataFrame(results)
    st.bar_chart(results_df.set_index("Pollutant"))