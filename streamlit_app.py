import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    asthma_df = pd.read_csv("asthma_ed_visits.csv")
    respiratory_df = pd.read_csv("respiratory_hosp.csv")
    boiler_df = pd.read_csv("boiler_emissions.csv")
    pm25_df = pd.read_csv("fine_particles.csv")
    ozone_df = pd.read_csv("ozone.csv")
    no2_df = pd.read_csv("nitrogen_dioxide.csv")
    return asthma_df, respiratory_df, boiler_df, pm25_df, ozone_df, no2_df

asthma_df, respiratory_df, boiler_df, pm25_df, ozone_df, no2_df = load_data()

# Helper to get the latest value per geo_place_name
def get_latest_value(df, value_col_name):
    return df.groupby("geo_place_name")["data_value"].mean().reset_index().rename(columns={"data_value": value_col_name})

# Prepare individual datasets (ignore time)
asthma = get_latest_value(asthma_df, "Asthma ED Visits")
resp = get_latest_value(respiratory_df, "Resp. Hospitalizations")
boiler = get_latest_value(boiler_df, "SO2 Emissions")
pm25 = get_latest_value(pm25_df, "PM2.5")
ozone = get_latest_value(ozone_df, "Ozone")
no2 = get_latest_value(no2_df, "NO2")

# Merge everything on geo_place_name
merged = asthma.merge(resp, on="geo_place_name", how="outer")
merged = merged.merge(boiler, on="geo_place_name", how="outer")
merged = merged.merge(pm25, on="geo_place_name", how="outer")
merged = merged.merge(ozone, on="geo_place_name", how="outer")
merged = merged.merge(no2, on="geo_place_name", how="outer")

# Handle missing values
merged.dropna(inplace=True)

# Streamlit app
st.title("NYC Air Quality & Health Outcomes")
st.write("This app displays average pollution and health data per location (ignoring time).")

st.dataframe(merged)

# Optional: select location to explore
location = st.selectbox("Select a location to explore", merged["geo_place_name"].unique())
row = merged[merged["geo_place_name"] == location].squeeze()

st.subheader(f"📍 Data for: {location}")
st.write({
    "Asthma ED Visits": row["Asthma ED Visits"],
    "Resp. Hospitalizations": row["Resp. Hospitalizations"],
    "SO2 Emissions": row["SO2 Emissions"],
    "PM2.5": row["PM2.5"],
    "Ozone": row["Ozone"],
    "NO2": row["NO2"]
})