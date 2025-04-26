import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load merged health dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_health_data.csv")

df = load_data().dropna()

st.title("🌫️ Surrogate Pollutant Impact Analysis")
st.write("""
This app tests if other pollutants correlate with PM2.5 health outcomes:
- 🫁 Asthma ED visits (labeled as PM2.5-related)
- � Hospitalizations (labeled as PM2.5-related)
""")

# Prepare data
st.subheader("📊 Your Data Structure")
st.write("Available columns:", list(df.columns))

# Use available pollutants as features
available_pollutants = [
    'Boiler Emissions- Total SO2 Emissions',
    # Add other pollutant columns if present in your data:
    # 'NO2_Emissions', 
    # 'O3_Levels'
]

# Check which pollutants are actually present
existing_pollutants = [col for col in available_pollutants if col in df.columns]

# Targets remain PM2.5-related outcomes
target_asthma = 'Asthma emergency department visits due to PM2.5'
target_resp = 'Respiratory hospitalizations due to PM2.5 (age 20+)'

# Model setup
st.subheader("🧪 Select Analysis Mode")
selected_pollutant = st.selectbox(
    "Choose pollutant to test as PM2.5 surrogate:",
    options=existing_pollutants
)

# Train model using selected pollutant
X = df[[selected_pollutant]]
y_asthma = df[target_asthma]
y_resp = df[target_resp]

# Model training
asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)
asthma_model.fit(X, y_asthma)
resp_model.fit(X, y_resp)

# User input
st.subheader("📈 Set Pollution Level")
pollutant_value = st.slider(
    f"{selected_pollutant} Level",
    min_value=float(df[selected_pollutant].min()),
    max_value=float(df[selected_pollutant].max()),
    value=float(df[selected_pollutant].median())
)

# Prediction
if st.button("🔮 Predict PM2.5-related Outcomes"):
    input_data = [[pollutant_value]]
    
    asthma_pred = asthma_model.predict(input_data)[0]
    resp_pred = resp_model.predict(input_data)[0]
    
    st.success(f"🫁 Predicted Asthma ED Visits: **{asthma_pred:.2f}** per 10k")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{resp_pred:.2f}** per 10k")
    
    # Show correlation
corr_asthma = df[[selected_pollutant, target_asthma]].corr().iloc[0,1]
corr_resp = df[[selected_pollutant, target_resp]].corr().iloc[0,1]

st.subheader("🔗 Correlation Analysis")
st.write(f"Correlation between {selected_pollutant} and:")
st.write(f"- Asthma ED Visits: **{corr_asthma:.2f}**")
st.write(f"- Respiratory Hospitalizations: **{corr_resp:.2f}**")