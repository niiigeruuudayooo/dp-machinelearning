import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Air Quality Clustering & Health Outcome Prediction")

@st.cache_data
def load_and_prepare_data():
    # File mapping
    files = [
        'asthma_ed_visits.csv',
        'respiratory_hosp.csv',
        'boiler_emissions.csv',
        'fine_particles.csv',
        'ozone.csv',
        'nitrogen_dioxide.csv'
    ]
    
    # Load and concatenate
    all_data = []
    for f in files:
        df = pd.read_csv(f, usecols=['name', 'geo_place_name', 'start_date', 'data_value'])
        df['geo_place_name'] = df['geo_place_name'].str.lower().str.strip()
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date
        all_data.append(df)
    
    merged = pd.concat(all_data)

    # Pivot so each measurement is a column
    pivot = merged.pivot_table(
        index=['geo_place_name', 'start_date'],
        columns='name',
        values='data_value'
    ).reset_index()

    # Drop rows with missing values
    pivot.dropna(inplace=True)

    return pivot

df = load_and_prepare_data()

st.subheader("Select air quality features to include in the clustering:")

# Feature options from the actual column names
feature_options = [
    'Nitrogen dioxide (NO2)',
    'Ozone (O3)',
    'Fine particles (PM 2.5)',
    'Boiler Emissions- Total SO2 Emissions'
]

selected_features = st.multiselect("Choose features:", feature_options, default=feature_options)

if not selected_features:
    st.warning("Please select at least one air quality feature.")
else:
    # Select features + outcomes
    X = df[selected_features]
    outcomes = [
        'Asthma emergency department visits due to PM2.5',
        'Respiratory hospitalizations due to PM2.5 (age 20+)',
    ]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run KMeans
    k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)
    model = KMeans(n_clusters=k, random_state=0)
    df['cluster'] = model.fit_predict(X_scaled)

    st.subheader("Clustered Data")
    st.dataframe(df[['geo_place_name', 'start_date', 'cluster'] + selected_features + outcomes])

    # Visualize cluster centers
    st.subheader("Cluster Centers")
    centers = pd.DataFrame(model.cluster_centers_, columns=selected_features)
    st.write(centers)

    # Plot clusters
    st.subheader("Cluster Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='cluster', ax=ax)
    st.pyplot(fig)