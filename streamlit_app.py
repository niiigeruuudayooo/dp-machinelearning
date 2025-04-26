import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load datasets
@st.cache_data
def load_data():
    health = pd.read_csv("merged_health_data.csv")
    no2 = pd.read_csv("nitrogen_dioxide.csv")
    o3 = pd.read_csv("ozone.csv")
    return health, no2, o3

health_df, no2_df, o3_df = load_data()
health_df = health_df.dropna()

st.title("🌍 Cross-Pollutant Health Impact Analyzer")
st.write("""
Explore how different pollutants might correlate with PM2.5-related health outcomes.
""")

# Model training on BOILER EMISSIONS (the only available pollutant in health data)
st.subheader("🔧 Base Model Setup")
base_pollutant = 'Boiler Emissions- Total SO2 Emissions'  # Using existing pollutant

# Prepare training data
X = health_df[[base_pollutant]]
y_asthma = health_df["Asthma emergency department visits due to PM2.5"]
y_resp = health_df["Respiratory hospitalizations due to PM2.5 (age 20+)"]

# Cache model training
@st.cache_data
def train_models(X, y_asthma, y_resp):
    asthma_model = RandomForestRegressor(random_state=42).fit(X, y_asthma)
    resp_model = RandomForestRegressor(random_state=42).fit(X, y_resp)
    return asthma_model, resp_model

asthma_model, resp_model = train_models(X, y_asthma, y_resp)

# Prediction interface
st.subheader("🧪 Simulation Parameters")
pollutant_type = st.selectbox("Choose pollutant to test:", ["SO2", "NO2", "O3"])

# Configure based on selected pollutant
if pollutant_type == "SO2":
    min_val = float(health_df[base_pollutant].min())
    max_val = float(health_df[base_pollutant].max())
    default = float(health_df[base_pollutant].median())
    unit = 'tons/year'
elif pollutant_type == "NO2":
    min_val = no2_df['data_value'].min()
    max_val = no2_df['data_value'].max()
    default = no2_df['data_value'].median()
    unit = 'ppb'
elif pollutant_type == "O3":
    min_val = o3_df['data_value'].min()
    max_val = o3_df['data_value'].max()
    default = o3_df['data_value'].median()
    unit = 'ppb'

# Get user input
value = st.slider(
    f"{pollutant_type} Level ({unit})",
    min_value=min_val,
    max_value=max_val,
    value=default,
    step=0.1
)

if st.button("Predict Health Outcomes"):
    # Make predictions using the SO2-trained model
    input_data = pd.DataFrame({base_pollutant: [value]})
    
    asthma_pred = asthma_model.predict(input_data)[0]
    resp_pred = resp_model.predict(input_data)[0]
    
    # Display results
    st.success(f"🫁 Predicted Asthma ED Visits: **{asthma_pred:.1f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{resp_pred:.1f}** per 10,000")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual SO2 relationship
    ax.scatter(health_df[base_pollutant], y_asthma, alpha=0.3, 
               label="Actual SO₂ vs Asthma ED Visits", color='blue')
    ax.scatter(health_df[base_pollutant], y_resp, alpha=0.3,
               label="Actual SO₂ vs Respiratory Hospitalizations", color='green')
    
    # Plot current prediction
    ax.scatter(value, asthma_pred, s=200, marker="X", 
               label=f"Predicted Asthma (Input: {pollutant_type})", color='red')
    ax.scatter(value, resp_pred, s=200, marker="X",
               label=f"Predicted Respiratory (Input: {pollutant_type})", color='orange')
    
    ax.set_xlabel(f"{pollutant_type} Level ({unit})")
    ax.set_ylabel("Health Outcomes per 10,000")
    ax.set_title(f"Health Impact Projection using {pollutant_type}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)

st.markdown("""
---
**🔍 Analysis Note:**  
Model trained on boiler SO₂ emissions health relationships.  
Other pollutants tested as potential substitutes through concentration comparison.
""")