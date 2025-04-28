# pages/📊_Regression_Prediction.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import traceback

# --- Configuration & Caching ---
st.set_page_configi(layout="wide", page_title="Prediction Analysis")

# --- Constants ---
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads all necessary CSV files into pandas DataFrames from the 'data' folder."""
    files = {
        'boiler': 'Boiler_Emissions-_Total_SO2_Emissions.csv',
        'pm25': 'Fine_particles_PM_2.5.csv',
        'no2': 'Nitrogen_dioxide_NO2.csv',
        'o3': 'Ozone_O3.csv',
        'asthma': 'Asthma_emergency_department_visits_due_to_PM2.5.csv',
        'resp': 'Respiratory_hospitalizations_due_to_PM2.5_age_20+.csv'
    }
    datasets = {}
    all_loaded = True
    
    for name, filename in files.items():
        try:
            filepath = os.path.join(BASE_PATH, filename)
            datasets[name] = pd.read_csv(filepath)
            
            required_cols = ['Geo Place Name', 'Start_Date', 'Data Value'] if name not in ['asthma', 'resp'] else ['geo_place_name', 'year', 'data_value']
            if not all(col in datasets[name].columns for col in required_cols):
                st.warning(f"File '{filename}' might be missing expected columns ({', '.join(required_cols)}). Check file format.")
        except FileNotFoundError:
            st.error(f"Error: File not found at {filepath}. Please ensure '{filename}' is in the 'data' folder.")
            datasets[name] = None
            all_loaded = False
        except Exception as e:
            st.error(f"An error occurred loading {filename}: {e}")
            datasets[name] = None
            all_loaded = False
            
    if not all_loaded:
        st.warning("Some data files could not be loaded or had potential issues.")
    return datasets

# --- Data Preprocessing with Imputation ---
@st.cache_data
def preprocess_data(datasets):
    """Cleans, standardizes, merges, imputes, and interpolates the data."""
    if not datasets or any(df is None for df in datasets.values()):
        st.error("Preprocessing cannot proceed because some datasets failed to load.")
        return None, None, None, None

    try:
        # --- Rename and Format Dates ---
        dfs_to_process = {
            'boiler': (datasets.get('boiler'), 'boiler_emissions'),
            'pm25': (datasets.get('pm25'), 'PM25'),
            'no2': (datasets.get('no2'), 'NO2'),
            'o3': (datasets.get('o3'), 'O3'),
            'asthma': (datasets.get('asthma'), 'asthma_rate'),
            'resp': (datasets.get('resp'), 'respiratory_rate')
        }
        processed_dfs = {}
        all_places = set()
        all_dates = set()
        pollutant_cols = []
        health_cols = []

        for name, (df, value_col) in dfs_to_process.items():
            if df is None:
                processed_dfs[name] = pd.DataFrame()
                continue

            temp_df = df.copy()
            
            # Standardize column names
            if 'Data Value' in temp_df.columns and 'Geo Place Name' in temp_df.columns:
                temp_df = temp_df.rename(columns={'Data Value': value_col, 'Geo Place Name': 'Geo Place Name'})
            elif 'data_value' in temp_df.columns and 'geo_place_name' in temp_df.columns:
                temp_df = temp_df.rename(columns={'data_value': value_col, 'geo_place_name': 'Geo Place Name'})
            else:
                processed_dfs[name] = pd.DataFrame()
                continue

            # Standardize date columns
            if 'Start_Date' in temp_df.columns:
                temp_df = temp_df.rename(columns={'Start_Date': 'Date'})
                temp_df['Date_temp'] = pd.to_datetime(temp_df['Date'], errors='coerce')
            elif 'year' in temp_df.columns:
                temp_df = temp_df.rename(columns={'year': 'Year'})
                temp_df['Date_temp'] = pd.to_datetime(temp_df['Year'].astype(str) + '-01-01', errors='coerce')
            else:
                processed_dfs[name] = pd.DataFrame()
                continue

            # Ensure value column is numeric
            temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors='coerce')
            
            # Drop rows where essential info is missing
            temp_df = temp_df.dropna(subset=['Date_temp', 'Geo Place Name', value_col])
            if temp_df.empty:
                processed_dfs[name] = pd.DataFrame()
                continue

            temp_df['YearMonth'] = temp_df['Date_temp'].dt.to_period("M").astype(str)
            temp_df['Year'] = temp_df['Date_temp'].dt.year

            # Aggregate potential duplicates
            group_cols = ['Geo Place Name', 'YearMonth', 'Year', 'Date_temp']
            df_cleaned = temp_df.groupby(group_cols, as_index=False)[value_col].mean()

            processed_dfs[name] = df_cleaned
            all_places.update(df_cleaned['Geo Place Name'].unique())
            all_dates.update(df_cleaned['YearMonth'].unique())

            # Add value_col to the correct list
            if name in ['asthma', 'resp']:
                if value_col not in health_cols: health_cols.append(value_col)
            else:
                if value_col not in pollutant_cols: pollutant_cols.append(value_col)

        # --- Create full grid & Merge ---
        if not all_places or not all_dates:
            st.error("Could not create a combined grid. Check data consistency (places/dates).")
            return None, None, None, None

        all_places = sorted(list(all_places))
        all_dates = sorted(list(all_dates))
        full_grid = pd.MultiIndex.from_product(
            [all_places, all_dates], names=["Geo Place Name", "YearMonth"]
        ).to_frame(index=False)

        merged_data = full_grid.copy()

        # Merge pollutants first
        for name, df in processed_dfs.items():
            if name not in ['asthma', 'resp'] and not df.empty:
                value_col = dfs_to_process[name][1]
                if value_col in df.columns:
                    merge_cols = ['Geo Place Name', 'YearMonth', value_col]
                    merged_data = pd.merge(merged_data, df[merge_cols],
                                         on=["Geo Place Name", "YearMonth"], how="left")

        # Interpolate pollutants (forward/backward fill within each location)
        merged_data['Date_temp'] = pd.to_datetime(merged_data['YearMonth'], errors='coerce')
        merged_data = merged_data.sort_values(by=['Geo Place Name', 'Date_temp'])
        for col in pollutant_cols:
            if col in merged_data.columns:
                merged_data[col] = merged_data.groupby('Geo Place Name')[col].transform(lambda x: x.ffill().bfill())
                if merged_data[col].isnull().any():
                    overall_fill = merged_data[col].mean()
                    merged_data[col].fillna(overall_fill, inplace=True)

        # Merge Health Data (using Year)
        merged_data['Year'] = merged_data['Date_temp'].dt.year
        final_df = merged_data

        for name in ['asthma', 'resp']:
            if name in processed_dfs and not processed_dfs[name].empty:
                value_col = dfs_to_process[name][1]
                if value_col in processed_dfs[name].columns:
                    health_avg = processed_dfs[name].groupby(['Geo Place Name', 'Year'], as_index=False)[value_col].mean()
                    final_df = pd.merge(final_df, health_avg, on=['Geo Place Name', 'Year'], how='left')

        # Impute missing values for both pollutants and health outcomes
        impute_cols = pollutant_cols + health_cols
        for col in impute_cols:
            if col in final_df.columns:
                final_df[col] = final_df.groupby('Geo Place Name')[col].transform(lambda x: x.fillna(x.mean()))
                if final_df[col].isnull().any():
                    overall_mean = final_df[col].mean()
                    final_df[col].fillna(overall_mean, inplace=True)

        # Prepare data for modeling
        model_cols = pollutant_cols + health_cols
        model_data = final_df.groupby('Geo Place Name')[model_cols].mean()
        model_data = model_data.dropna(subset=model_cols)

        if model_data.empty:
            return final_df, None, None, None

        # Separate features (X) and targets (y)
        X_model = model_data[pollutant_cols]
        y_asthma_model = model_data['asthma_rate'] if 'asthma_rate' in model_data.columns else None
        y_resp_model = model_data['respiratory_rate'] if 'respiratory_rate' in model_data.columns else None

        return final_df, X_model, y_asthma_model, y_resp_model

    except Exception as e:
        st.error(f"An error occurred during data preprocessing: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None

# --- Load and preprocess data ---
datasets = load_data()
final_data, X_model_data, y_asthma_model_data, y_resp_model_data = preprocess_data(datasets)

# --- Train Models ---
model_asthma = None
model_resp = None
trained_features = []

if X_model_data is not None and not X_model_data.empty:
    trained_features = X_model_data.columns.tolist()

    # Train Asthma Model
    if y_asthma_model_data is not None and not y_asthma_model_data.empty:
        if len(X_model_data) == len(y_asthma_model_data):
            if not X_model_data.index.equals(y_asthma_model_data.index):
                y_asthma_model_data = y_asthma_model_data.reindex(X_model_data.index)
            
            if not (X_model_data.isnull().values.any() or y_asthma_model_data.isnull().any()):
                try:
                    model_asthma = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
                    model_asthma.fit(X_model_data, y_asthma_model_data)
                except Exception as e:
                    st.error(f"Could not train Asthma model: {e}")
                    model_asthma = None

    # Train Respiratory Model
    if y_resp_model_data is not None and not y_resp_model_data.empty:
        if len(X_model_data) == len(y_resp_model_data):
            if not X_model_data.index.equals(y_resp_model_data.index):
                y_resp_model_data = y_resp_model_data.reindex(X_model_data.index)
            
            if not (X_model_data.isnull().values.any() or y_resp_model_data.isnull().any()):
                try:
                    model_resp = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
                    model_resp.fit(X_model_data, y_resp_model_data)
                except Exception as e:
                    st.error(f"Could not train Respiratory model: {e}")
                    model_resp = None

# --- Page Title ---
st.title("📊 Health Outcome Prediction Analysis")
st.markdown("Predict health outcomes based on average pollutant levels.")

# --- Check if data processing was successful ---
if final_data is not None and not final_data.empty:
    # --- Sidebar Filters ---
    st.sidebar.header("⚙️ Filter Options")
    locations = sorted(final_data['Geo Place Name'].unique())
    if not locations:
        st.error("No locations found in the final processed data.")
        st.stop()

    # Default to first location if NYC not found or locations list is empty
    default_ix = 0
    if 'NYC' in locations:
        default_ix = locations.index('NYC')
    elif locations:
        default_ix = 0

    selected_location = st.sidebar.selectbox("Select Location:", options=locations, index=default_ix)

    # Use features the model was actually trained on
    available_pollutants_for_display = trained_features if trained_features else \
                                     [col for col in ['boiler_emissions', 'PM25', 'NO2', 'O3'] if col in final_data.columns]

    health_outcomes = {'Asthma ED Visits Rate': 'asthma_rate',
                     'Respiratory Hospitalization Rate (Age 20+)': 'respiratory_rate'}
    selected_health_outcome_label = st.sidebar.radio(
        "Select Health Outcome for Prediction:",
        options=list(health_outcomes.keys()),
        index=0
    )
    selected_health_outcome_col = health_outcomes[selected_health_outcome_label]

    # Filter data for the selected location
    filtered_by_location = final_data[final_data['Geo Place Name'] == selected_location].copy()

    # --- Health Outcome Prediction Section ---
    st.header(f"⚕️ Health Outcome Prediction for: {selected_location}")
    st.markdown(f"Predicting average **{selected_health_outcome_label}** based on average pollutant levels using a Random Forest model.")

    # Get average pollutant data for the selected location for prediction
    location_avg_data_for_pred = None
    if X_model_data is not None and selected_location in X_model_data.index:
        location_avg_data_for_pred = X_model_data.loc[[selected_location]]
    elif not filtered_by_location.empty and available_pollutants_for_display:
        location_avg_data_for_pred = filtered_by_location[available_pollutants_for_display].mean().to_frame().T
        location_avg_data_for_pred.index = [selected_location]
        if trained_features:
            fill_val = 0
            location_avg_data_for_pred = location_avg_data_for_pred.reindex(columns=trained_features, fill_value=fill_val)
    else:
        st.warning(f"Could not retrieve or calculate average pollutant data for {selected_location}.")

    # Select the correct model
    model_to_use = model_asthma if selected_health_outcome_col == 'asthma_rate' else model_resp

    # Get actual average health value for comparison
    actual_value = None
    if not filtered_by_location.empty and selected_health_outcome_col in filtered_by_location.columns:
        actual_value = filtered_by_location[selected_health_outcome_col].mean(skipna=True)

    # Perform prediction
    if model_to_use is not None and location_avg_data_for_pred is not None and not location_avg_data_for_pred.empty:
        if trained_features:
            fill_val_pred = 0
            location_avg_data_aligned = location_avg_data_for_pred.reindex(columns=trained_features, fill_value=fill_val_pred)

            if not location_avg_data_aligned.isnull().values.any():
                try:
                    prediction = model_to_use.predict(location_avg_data_aligned)[0]

                    pred_col, actual_col = st.columns(2)
                    with pred_col:
                        st.metric(label=f"Predicted Avg. {selected_health_outcome_label}", value=f"{prediction:.2f}")
                    with actual_col:
                        if actual_value is not None and not np.isnan(actual_value):
                            st.metric(
                                label=f"Actual Avg. {selected_health_outcome_label}",
                                value=f"{actual_value:.2f}",
                                delta=f"{prediction - actual_value:.2f} (Pred - Actual)",
                                delta_color="normal"
                            )
                        else:
                            st.info(f"Actual average data for {selected_health_outcome_label} not available or calculable in {selected_location}.")

                    # Feature Importance
                    if hasattr(model_to_use, 'feature_importances_') and trained_features:
                        st.subheader("Pollutant Importance for Prediction")
                        importance_df = pd.DataFrame({
                            'Feature': trained_features,
                            'Importance': model_to_use.feature_importances_
                        }).sort_values(by='Importance', ascending=False)

                        # Map internal names to display names for the plot
                        pollutant_display_names_map = {
                            'boiler_emissions': 'Boiler Emissions',
                            'PM25': 'PM₂.₅',
                            'NO2': 'NO₂',
                            'O3': 'O₃'
                        }
                        importance_df['Feature_Display'] = importance_df['Feature'].apply(lambda x: pollutant_display_names_map.get(x, x))

                        fig_imp, ax_imp = plt.subplots(figsize=(10, max(4, len(trained_features) * 0.5)))
                        sns.barplot(x='Importance', y='Feature_Display', data=importance_df, ax=ax_imp, palette='viridis')
                        ax_imp.set_title(f"Pollutant Importance for Predicting {selected_health_outcome_label}")
                        ax_imp.set_xlabel("Importance Score")
                        ax_imp.set_ylabel("Pollutant")
                        plt.tight_layout()
                        st.pyplot(fig_imp)
                    else:
                        st.info("Feature importance data is not available for this model.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Prediction input data contains NaN values after alignment and fill.")
        else:
            st.error("Model was not trained or feature list is missing. Cannot align prediction data.")
    elif model_to_use is None:
        st.error(f"Prediction model for **{selected_health_outcome_label}** is not available.")
    elif location_avg_data_for_pred is None or location_avg_data_for_pred.empty:
        st.warning(f"Cannot make prediction: Average pollutant data for **{selected_location}** could not be prepared or retrieved.")
    else:
        st.warning(f"Cannot make prediction for an unknown reason. Check data and model status.")

elif final_data is None:
    st.error("Data could not be loaded or processed successfully. Cannot display analysis.")
    st.info("Please ensure all required CSV files are present in the 'data' folder and have the expected columns/formats.")
else:
    st.error("Processed data is empty. Cannot display analysis.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Analysis based on NYC Open Data.")
