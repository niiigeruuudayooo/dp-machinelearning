import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="NYC Health Predictor", layout="centered")

# Load cleaned and merged health dataset
@st.cache_data
def load_data():
    df = pd.read_csv("merged_health_data.csv")
    df.columns = df.columns.str.strip()  # clean column names
    return df.dropna()

df = load_data()

# Show dataset sample
st.title("🫁 NYC Health Rate Prediction")
st.write("This app predicts asthma emergency visits and respiratory hospitalizations based on SO2 boiler emissions.")
st.write("### 📊 Sample Data")
st.dataframe(df.head())

# Show actual column names for verification
st.write("Columns loaded:", list(df.columns))

# Select features and targets
X_col = "Boiler Emissions- Total SO2 Emissions"
y_asthma_col = "Asthma emergency department visits due to PM2.5"
y_resp_col = "Respiratory hospitalizations due to PM2.5 (age 20+)"

# Features and labels
X = df[[X_col]]
y_asthma = df[y_asthma_col]
y_resp = df[y_resp_col]

# Train models
asthma_model = RandomForestRegressor(random_state=0)
resp_model = RandomForestRegressor(random_state=0)
asthma_model.fit(X, y_asthma)
resp_model.fit(X, y_resp)

# User input
st.subheader("🔢 Enter SO2 Emission Level (tons/year)")
so2_input = st.number_input("Boiler Emissions", min_value=0.0, value=1.0)

# Predict and show results
if st.button("Predict Health Rates"):
    input_val = [[so2_input]]
    asthma_pred = asthma_model.predict(input_val)[0]
    resp_pred = resp_model.predict(input_val)[0]

    st.success(f"🌬️ Predicted Asthma ED Visits Rate: **{asthma_pred:.2f}** per 10,000")
    st.success(f"🫁 Predicted Respiratory Hospitalization Rate: **{resp_pred:.2f}** per 10,000")

    # Optional: show bar chart
    st.subheader("📈 Prediction Chart")
    pred_df = pd.DataFrame({
        "Health Metric": ["Asthma ED Visits", "Respiratory Hospitalizations"],
        "Predicted Rate": [asthma_pred, resp_pred]
    })
    st.bar_chart(pred_df.set_index("Health Metric"))