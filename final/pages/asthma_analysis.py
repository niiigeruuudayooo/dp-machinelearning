# asthma_dashboard_map.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from neo4j import GraphDatabase

# --------------- MongoDB Connection -----------------
mongo_client = MongoClient('mongodb://localhost:27017/')
mongo_db = mongo_client['air_quality']
mongo_collection = mongo_db['pollution_asthma']

# --------------- Neo4j Connection -----------------
neo4j_driver = GraphDatabase.driver('bolt://localhost:7687', auth=("neo4j", "neo4j1234"))

def get_neo4j_coordinates():
    with neo4j_driver.session() as session:
        results = session.run("""
            MATCH (b:Borough)
            RETURN b.name AS name, b.lat AS lat, b.lon AS lon
            ORDER BY b.name
        """)
        data = results.data()
    return pd.DataFrame(data)

# --------------- Load Data -----------------

# Exclude the _id field directly in the MongoDB query
data = pd.DataFrame(list(mongo_collection.find({}, {'_id': 0})))
# data = data.drop(columns=['_id'])

# Load coordinates from Neo4j
coords_df = get_neo4j_coordinates()

# --- Add these lines for debugging ---
st.subheader("Debugging Neo4j Data")
st.write("Columns found in coords_df:", coords_df.columns.tolist())
st.write("First 5 rows of coords_df:")
st.dataframe(coords_df.head())
# --- End of debugging lines ---

# Merge the data
merged_data = pd.merge(data, coords_df, left_on='Borough', right_on='name', how='left')

# --------------- Streamlit Frontend -----------------
# st.set_page_config(page_title="Borough Pollution & Asthma Dashboard", layout="wide")
st.title("🌍 Borough Pollution & Asthma Dashboard")

# Sidebar Selection
borough_list = merged_data['Borough'].sort_values().unique()
selected_borough = st.sidebar.selectbox("Select a Borough", borough_list)

# Filtered Data
filtered_data = merged_data[merged_data['Borough'] == selected_borough]

# Check if data exists
if not filtered_data.empty:
    st.subheader(f"📄 Data for {selected_borough}")
    st.dataframe(filtered_data[['Date', 'NO2', 'O3', 'PM2.5', 'Asthma_Visits']])

    # ------- Pollution Trend Bar Chart --------
    filtered_data_melted = filtered_data.melt(
        id_vars=['Date'],
        value_vars=['NO2', 'O3', 'PM2.5', 'Asthma_Visits'],
        var_name='Indicator',
        value_name='Measurement'
    )

    fig = px.bar(
        filtered_data_melted,
        x='Date',
        y='Measurement',
        color='Indicator',
        barmode='group',
        title=f"Pollution and Asthma Trends in {selected_borough}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------- Location Map --------
    st.subheader("🗺️ Location on Map")

    fig_map = px.scatter_mapbox(
        filtered_data,
        lat='lat_y',
        lon='lon_y',
        hover_name='Borough',
        hover_data={'NO2': True, 'O3': True, 'PM2.5': True, 'Asthma_Visits': True},
        color_discrete_sequence=["red"],
        zoom=10,
        height=500
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("⚠️ No data available for the selected borough.")

# --------------- Cleanup -----------------
neo4j_driver.close()
mongo_client.close()
