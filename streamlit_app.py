import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load merged health dataset
@st.cache_data
def load_data():
    return pd.read_csv("merged_health_data.csv")

df = load_data().dropna()

st.title("🌫️ Surrogate Pollutant Impact Analysis")
st.write("""
This app tests if other pollutants correlate with PM2.5 health outcomes:
- 🫁 Asthma ED visits (per 10,000 population)
- 🏥 Respiratory hospitalizations (per 10,000 population)
""")

# Use available pollutants as features with units
available_pollutants = {
    'Boiler Emissions- Total SO2 Emissions': 'tons/year',
    # Add other pollutants with units as needed:
    # 'NO2_Emissions': 'ppm',
    # 'O3_Levels': 'ppb'
}

# Check which pollutants are actually present
existing_pollutants = [col for col in available_pollutants.keys() if col in df.columns]

# Targets with units
target_asthma = ('Asthma emergency department visits due to PM2.5', 'per 10,000 pop')
target_resp = ('Respiratory hospitalizations due to PM2.5 (age 20+)', 'per 10,000 pop')

# Model setup
st.subheader("🧪 Select Analysis Mode")
selected_pollutant = st.selectbox(
    "Choose pollutant to test as PM2.5 surrogate:",
    options=existing_pollutants
)

# Get units for selected pollutant
pollutant_unit = available_pollutants[selected_pollutant]

# Train model using selected pollutant
X = df[[selected_pollutant]]
y_asthma = df[target_asthma[0]]
y_resp = df[target_resp[0]]

# Model training
asthma_model = RandomForestRegressor(random_state=42)
resp_model = RandomForestRegressor(random_state=42)
asthma_model.fit(X, y_asthma)
resp_model.fit(X, y_resp)

# User input with units
st.subheader("📈 Set Pollution Level")
pollutant_value = st.slider(
    f"{selected_pollutant} ({pollutant_unit})",
    min_value=float(df[selected_pollutant].min()),
    max_value=float(df[selected_pollutant].max()),
    value=float(df[selected_pollutant].median()),
    help=f"Range: {df[selected_pollutant].min():.2f} to {df[selected_pollutant].max():.2f} {pollutant_unit}"
)

# Prediction and plots
if st.button("🔮 Predict PM2.5-related Outcomes"):
    input_data = [[pollutant_value]]
    
    # Get predictions
    asthma_pred = asthma_model.predict(input_data)[0]
    resp_pred = resp_model.predict(input_data)[0]
    
    # Generate actual vs predicted plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Asthma plot
    ax1.scatter(y_asthma, asthma_model.predict(X), alpha=0.5)
    ax1.plot([y_asthma.min(), y_asthma.max()], [y_asthma.min(), y_asthma.max()], 'k--')
    ax1.set_xlabel(f'Actual {target_asthma[1]}')
    ax1.set_ylabel(f'Predicted {target_asthma[1]}')
    ax1.set_title('Asthma ED Visits: Actual vs Predicted')
    
    # Respiratory plot
    ax2.scatter(y_resp, resp_model.predict(X), alpha=0.5)
    ax2.plot([y_resp.min(), y_resp.max()], [y_resp.min(), y_resp.max()], 'k--')
    ax2.set_xlabel(f'Actual {target_resp[1]}')
    ax2.set_ylabel(f'Predicted {target_resp[1]}')
    ax2.set_title('Respiratory Hospitalizations: Actual vs Predicted')
    
    # Display results
    st.success(f"🫁 Predicted Asthma ED Visits: **{asthma_pred:.2f}** {target_asthma[1]}")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{resp_pred:.2f}** {target_resp[1]}")
    
    # Show plots
    st.subheader("📊 Model Performance Visualization")
    st.pyplot(fig)
    
    # Show correlation
    corr_asthma = df[[selected_pollutant, target_asthma[0]].corr().iloc[0,1]
    corr_resp = df[[selected_pollutant, target_resp[0]].corr().iloc[0,1]
    
    st.subheader("🔗 Correlation Analysis")
    st.write(f"Correlation between {selected_pollutant} ({pollutant_unit}) and:")
    st.write(f"- Asthma ED Visits ({target_asthma[1]}): **{corr_asthma:.2f}**")
    st.write(f"- Respiratory Hospitalizations ({target_resp[1]}): **{corr_resp:.2f}**")

# Data preview with units
with st.expander("🗂️ View Raw Data with Units"):
    display_df = df.rename(columns={
        selected_pollutant: f"{selected_pollutant} ({pollutant_unit})",
        target_asthma[0]: f"Asthma ED Visits {target_asthma[1]}",
        target_resp[0]: f"Respiratory Hosp. {target_resp[1]}"
    })
    st.dataframe(display_df)