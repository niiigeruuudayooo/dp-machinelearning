import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

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
st.markdown("""
Explore how different pollutants might correlate with PM2.5-related health outcomes.
*Models trained on boiler SO₂ emissions data.*
""")

# Model training configuration
base_pollutant = 'Boiler Emissions- Total SO2 Emissions'
unit_dict = {'SO2': 'tons/year', 'NO2': 'ppb', 'O3': 'ppb'}

# Prepare training data
X = health_df[[base_pollutant]]
y_asthma = health_df["Asthma emergency department visits due to PM2.5"]
y_resp = health_df["Respiratory hospitalizations due to PM2.5 (age 20+)"]
so2_range = (X.min()[0], X.max()[0])

# Cache model training with performance metrics
@st.cache_data
def train_models(X, y_asthma, y_resp):
    asthma_model = RandomForestRegressor(random_state=42).fit(X, y_asthma)
    resp_model = RandomForestRegressor(random_state=42).fit(X, y_resp)
    
    # Calculate performance metrics
    asthma_r2 = asthma_model.score(X, y_asthma)
    resp_r2 = resp_model.score(X, y_resp)
    asthma_mae = -cross_val_score(asthma_model, X, y_asthma, 
                                scoring='neg_mean_absolute_error').mean()
    resp_mae = -cross_val_score(resp_model, X, y_resp, 
                              scoring='neg_mean_absolute_error').mean()
    
    return asthma_model, resp_model, asthma_r2, resp_r2, asthma_mae, resp_mae

asthma_model, resp_model, asthma_r2, resp_r2, asthma_mae, resp_mae = train_models(X, y_asthma, y_resp)

# Model performance report
with st.expander("📊 Model Performance Report (SO₂ Basis)"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Asthma Model", 
                f"R²: {asthma_r2:.2f}", 
                f"MAE: {asthma_mae:.1f} cases/10k")
    with col2:
        st.metric("Respiratory Model", 
                f"R²: {resp_r2:.2f}", 
                f"MAE: {resp_mae:.1f} cases/10k")
    st.caption("Based on 5-fold cross validation")

# Simulation interface
st.subheader("🧪 Pollution Scenario Setup")
col1, col2 = st.columns([2, 1])
with col1:
    pollutant_type = st.selectbox("Select pollutant:", ["SO2", "NO2", "O3"])
    unit = unit_dict[pollutant_type]

# Configure pollutant parameters
if pollutant_type == "SO2":
    data_source = health_df
    param_col = base_pollutant
else:
    data_source = no2_df if pollutant_type == "NO2" else o3_df
    param_col = 'data_value'
    data_source = data_source[data_source['name'].str.contains(pollutant_type)]

min_val = float(data_source[param_col].min())
max_val = float(data_source[param_col].max())
default = float(data_source[param_col].median())

# Interactive slider
with col2:
    value = st.slider(
        f"{pollutant_type} Level ({unit})",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=0.1,
        help="Adjust pollutant concentration level"
    )

# Validation warnings
if pollutant_type != "SO2":
    st.warning(f"""
    ⚠️ Comparative Analysis: 
    Showing {pollutant_type} levels as if they had the same concentration-health relationship as SO₂.
    """)
    
    if value < so2_range[0] or value > so2_range[1]:
        st.error(f"""
        🚨 Interpretation Caution:  
        Input value ({value:.1f} {unit}) outside SO₂ training range ({so2_range[0]:.1f}-{so2_range[1]:.1f} tons/year).
        """)

# Prediction and visualization
if st.button("Run Health Impact Simulation"):
    input_data = pd.DataFrame({base_pollutant: [value]})
    
    asthma_pred = asthma_model.predict(input_data)[0]
    resp_pred = resp_model.predict(input_data)[0]
    
    # Results display
    st.success(f"🫁 Predicted Asthma ED Visits: **{asthma_pred:.1f}** per 10,000")
    st.success(f"🏥 Predicted Respiratory Hospitalizations: **{resp_pred:.1f}** per 10,000")
    
    # Enhanced visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Actual relationships
    ax.scatter(X, y_asthma, alpha=0.4, c='#1f77b4', s=60, 
              label='Actual Asthma Cases')
    ax.scatter(X, y_resp, alpha=0.4, c='#2ca02c', s=60, 
              label='Actual Respiratory Cases')
    
    # Prediction markers
    pred_color = '#d62728' if pollutant_type != "SO2" else '#9467bd'
    ax.scatter(value, asthma_pred, s=400, marker="*", 
              edgecolor='black', linewidth=1, c=pred_color,
              label=f'Asthma Prediction ({pollutant_type})')
    ax.scatter(value, resp_pred, s=400, marker="P", 
              edgecolor='black', linewidth=1, c=pred_color,
              label=f'Respiratory Prediction ({pollutant_type})')
    
    # Reference elements
    ax.axvspan(so2_range[0], so2_range[1], alpha=0.1, color='gray',
              label='SO₂ Training Range')
    ax.axhline(y_asthma.median(), color='#1f77b4', linestyle=':', 
              linewidth=1.5, label='Asthma Median')
    ax.axhline(y_resp.median(), color='#2ca02c', linestyle=':', 
              linewidth=1.5, label='Respiratory Median')
    
    # Annotations
    ax.annotate(f'{asthma_pred:.1f}', (value, asthma_pred),
               xytext=(15,-15), textcoords='offset points',
               arrowprops=dict(arrowstyle="->", color='black'),
               fontsize=10, weight='bold')
    ax.annotate(f'{resp_pred:.1f}', (value, resp_pred),
               xytext=(15,15), textcoords='offset points',
               arrowprops=dict(arrowstyle="->", color='black'),
               fontsize=10, weight='bold')
    
    # Styling
    ax.set_xlabel(f"{pollutant_type} Concentration ({unit})", fontsize=12)
    ax.set_ylabel("Health Outcomes per 10,000 Population", fontsize=12)
    ax.set_title(f"Health Impact Projection: {pollutant_type} vs. SO₂ Model", 
               fontsize=14, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig)

# Data exploration section
with st.expander("🔍 Explore Source Datasets"):
    tab1, tab2, tab3 = st.tabs(["Health Data", "NO₂ Data", "O₃ Data"])
    
    with tab1:
        st.write("**Health Outcomes & SO₂ Emissions**")
        styled_health = health_df.style.format({
            base_pollutant: "{:.1f} tons/yr",
            y_asthma.name: "{:.1f} cases",
            y_resp.name: "{:.1f} cases"
        }).background_gradient(subset=[base_pollutant], cmap='Oranges')
        st.dataframe(styled_health, height=300)
    
    with tab2:
        st.write("**Nitrogen Dioxide Measurements**")
        styled_no2 = no2_df.style.format({
            'data_value': "{:.1f} ppb"
        }).background_gradient(subset=['data_value'], cmap='Blues')
        st.dataframe(styled_no2, height=300)
    
    with tab3:
        st.write("**Ozone Measurements**")
        styled_o3 = o3_df.style.format({
            'data_value': "{:.1f} ppb"
        }).background_gradient(subset=['data_value'], cmap='Greens')
        st.dataframe(styled_o3, height=300)

# Scientific context footer
st.markdown(f"""
---
**🔬 Scientific Context**  
1. **Model Basis**: Trained on SO₂ emissions ({so2_range[0]:.1f}-{so2_range[1]:.1f} tons/yr)  
2. **Comparison Approach**: Direct concentration substitution  
3. **Key Assumption**: Linear concentration-response relationship holds across pollutants  
4. **Limitations**: Does not account for:  
   - Pollutant-specific toxicity profiles  
   - Synergistic effects between pollutants  
   - Population vulnerability differences  

*For research purposes only - consult epidemiological studies for causal inference.*
""")