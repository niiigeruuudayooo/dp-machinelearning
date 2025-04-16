import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("🌫️ Air Quality & Health Predictor (K-Means)")

st.info("This app clusters air quality data and predicts health impacts like asthma and respiratory disease rates.")

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("your_cleaned_air_health_data.csv")
    return df

df = load_data()

with st.expander("View Raw Data"):
    st.dataframe(df)

# Feature selection
features = ['NO2', 'O3', 'PM2.5', 'Boiler_Emissions']
X = df[features]
health = df[['Asthma_Rate', 'Respiratory_Disease_Rate']]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means
k = 3  # you can make this adjustable via st.slider
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Sidebar for user input
st.sidebar.header("📥 Input Air Quality Data")
user_input = {
    'NO2': st.sidebar.slider("NO2 (ppb)", float(df.NO2.min()), float(df.NO2.max()), float(df.NO2.mean())),
    'O3': st.sidebar.slider("O3 (ppb)", float(df.O3.min()), float(df.O3.max()), float(df.O3.mean())),
    'PM2.5': st.sidebar.slider("PM2.5 (µg/m³)", float(df['PM2.5'].min()), float(df['PM2.5'].max()), float(df['PM2.5'].mean())),
    'Boiler_Emissions': st.sidebar.slider("Boiler Emissions (tons)", float(df['Boiler_Emissions'].min()), float(df['Boiler_Emissions'].max()), float(df['Boiler_Emissions'].mean()))
}
input_df = pd.DataFrame(user_input, index=[0])
input_scaled = scaler.transform(input_df)

# Predict cluster
predicted_cluster = kmeans.predict(input_scaled)[0]
st.success(f"Predicted Cluster: {predicted_cluster}")

# Show typical health rates in this cluster
cluster_avg = df[df['Cluster'] == predicted_cluster][['Asthma_Rate', 'Respiratory_Disease_Rate']].mean()
st.subheader("🩺 Estimated Health Impact")
st.write(f"**Estimated Asthma Rate:** {cluster_avg['Asthma_Rate']:.2f}")
st.write(f"**Estimated Respiratory Disease Rate:** {cluster_avg['Respiratory_Disease_Rate']:.2f}")

# Optional: Show cluster chart
with st.expander("📊 Cluster Visualization (PCA)"):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    fig, ax = plt.subplots()
    for c in range(k):
        cluster_data = pca_df[pca_df['Cluster'] == c]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f"Cluster {c}")
    ax.set_title("Clusters (PCA Projection)")
    ax.legend()
    st.pyplot(fig)
