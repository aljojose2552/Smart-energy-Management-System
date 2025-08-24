import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. Data Loading and Processing Logic ---
@st.cache_data
def load_and_process_data(uploaded_file, efficiency_measures, demand_response):
    """
    Loads the user's data and processes it based on selected simulation measures.
    This function replaces the previous data simulation with actual data processing.
    """
    if uploaded_file is None:
        return None, None
    
    # Read the uploaded CSV file into a DataFrame
    combined_df = pd.read_csv(uploaded_file)
    
    # --- AUTOMATICALLY CLEAN UP COLUMN NAMES ---
    # This removes any leading/trailing whitespace from column names.
    combined_df.columns = combined_df.columns.str.strip()

    # --- DIAGNOSTIC STEP: Print all column names to the console ---
    st.write("---")
    st.write("### Diagnostic: Columns in your uploaded file")
    st.write(combined_df.columns.tolist())
    st.write("---")
    
    # --- VERIFY REQUIRED COLUMNS EXIST ---
    # The 'hour' column will be created from 'date', so we check for 'date' instead.
    required_cols = ['From grid(Wh)', 'Consumption(Wh)', 'date', 'household']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]

    if missing_cols:
        st.error(f"Error: The following required columns are missing from your file: {', '.join(missing_cols)}. Please check your data.")
        return None, None
    
    # IOT SCHEDULING I.E WE SIMULATE DEMAND RESPONSE
    # It detects peak hours for example when average Consumption(Wh) is high
    # it suggests deferring flexible loads (e.g EV charging, laundry) to off-peak hours
    
    # create a copy of our original data
    df_scaled = combined_df.copy()

    # --- ADD 'hour' COLUMN BY EXTRACTING IT FROM 'date' ---
    # This step is crucial since the original data doesn't have an explicit 'hour' column
    df_scaled['date'] = pd.to_datetime(df_scaled['date'])
    df_scaled['hour'] = df_scaled['date'].dt.hour

    # Apply efficiency measures if enabled
    if efficiency_measures:
        # A simple placeholder: reduce total consumption by 10%
        # You can add more complex logic here
        df_scaled['From grid(Wh)'] *= 0.9

    # Apply demand response if enabled
    if demand_response:
        # first we classify peak & off-peak hours based on average consumptions
        hourly_avg = df_scaled.groupby('hour')['Consumption(Wh)'].mean()
        threshold = hourly_avg.quantile(0.75) # here we simply say that 25% is for peak hours
        
        # labeling hours as peak/off-peak
        df_scaled['demand_period'] = df_scaled['hour'].apply(lambda x:'Peak' if hourly_avg[x] >= threshold else 'Off-peak')
        
        # we simulate adjusted Grid Usage for 20% & 30% load shift during peak hours
        df_scaled['Adjusted_20%'] = df_scaled.apply(
            lambda row: row['From grid(Wh)'] * 0.8 if row['demand_period'] == 'Peak' else row['From grid(Wh)'],
            axis=1
        )
        df_scaled['Adjusted_30%'] = df_scaled.apply(
            lambda row: row['From grid(Wh)'] * 0.7 if row['demand_period'] == 'Peak' else row['From grid(Wh)'],
            axis=1
        )
    else:
        # If demand response is not enabled, the adjusted usage is the same as original
        df_scaled['Adjusted_20%'] = df_scaled['From grid(Wh)']
        df_scaled['Adjusted_30%'] = df_scaled['From grid(Wh)']
    
    # we can now aggregate original & adjusted usage per household
    grid_summary_scaled = df_scaled.groupby('household').agg({
        'From grid(Wh)':'sum',
        'Adjusted_20%':'sum',
        'Adjusted_30%':'sum'
    }).rename(columns={
        'From grid(Wh)': 'Original Grid Use(Wh)',
        'Adjusted_20%':'Grid Use after 20% Shift',
        'Adjusted_30%':'Grid Use after 30% Shift'
    })
    
    ## calculating % saving 7 absolute savings
    grid_summary_scaled['Savings_20% (Wh)'] = grid_summary_scaled['Original Grid Use(Wh)']- grid_summary_scaled['Grid Use after 20% Shift']
    grid_summary_scaled['Savings_30% (Wh)'] = grid_summary_scaled['Original Grid Use(Wh)']- grid_summary_scaled['Grid Use after 30% Shift']
    
    grid_summary_scaled['Savings_20% (%)'] = grid_summary_scaled['Savings_20% (Wh)']/ grid_summary_scaled['Original Grid Use(Wh)']*100
    grid_summary_scaled['Savings_30% (%)'] = grid_summary_scaled['Savings_30% (Wh)']/ grid_summary_scaled['Original Grid Use(Wh)']*100
    
    # display results
    grid_summary_scaled = grid_summary_scaled.round(2)

    return df_scaled, grid_summary_scaled

# --- 2. Streamlit UI Layout and Widgets ---
st.set_page_config(layout="wide", page_title="Smart Energy Simulation")

st.title("🏡 Smart Energy Management: Interactive Simulation")
st.markdown("Upload your data and use the controls to see the results of efficiency and demand response measures.")

# Initialize session state for results if not already present
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
    st.session_state.df_scaled = None

# --- Left Panel (Sidebar) ---
with st.sidebar:
    st.header("Data & Scenario Setup")
    uploaded_file = st.file_uploader("Upload your data file (CSV)", type="csv")

    efficiency_measures = st.checkbox("Apply Energy Efficiency Measures", value=False)
    demand_response = st.checkbox("Enable Demand Response", value=False)
    
    if st.button("Run Analysis"):
        if uploaded_file is not None:
            st.write("Running analysis...")
            df_scaled, grid_summary_scaled = load_and_process_data(uploaded_file, efficiency_measures, demand_response)
            st.session_state.df_scaled = df_scaled
            st.session_state.simulation_results = grid_summary_scaled
        else:
            st.error("Please upload a CSV file to run the analysis.")

# --- Main Page (Center and Right Panels) ---
# Only display results if a simulation has been run
if st.session_state.simulation_results is not None:
    grid_summary_scaled = st.session_state.simulation_results
    df_scaled = st.session_state.df_scaled
    
    if grid_summary_scaled is not None:
        # Create two columns for the main layout
        col1, col2 = st.columns([2, 1])

        # --- Center Panel (Visualizations) ---
        with col1:
            st.header("Main Visualizations")
            
            st.subheader("Grid Usage and Savings by Household")
            
            # Plotly bar chart for grid usage
            fig1 = px.bar(
                grid_summary_scaled.reset_index(), 
                x='household', 
                y=['Original Grid Use(Wh)', 'Grid Use after 20% Shift', 'Grid Use after 30% Shift'],
                barmode='group',
                title='Combined Grid Use For IoT Scheduling and Demand Response Saving',
                labels={'value': 'Grid Use (Wh)', 'household': 'Household'}
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Savings Percentage by Household")
            # Plotly line chart for savings percentage
            fig2 = px.line(
                grid_summary_scaled.reset_index(),
                x='household',
                y=['Savings_20% (%)', 'Savings_30% (%)'],
                markers=True,
                title='Savings from IoT Scheduling and Demand Response',
                labels={'value': 'Savings (%)', 'household': 'Household'}
            )
            st.plotly_chart(fig2, use_container_width=True)

        # --- Right Panel (Results & Insights) ---
        with col2:
            st.header("Results & Insights")
            
            st.subheader("Key Metrics")
            
            # Use pandas to calculate overall metrics
            overall_original_use = grid_summary_scaled['Original Grid Use(Wh)'].sum()
            overall_savings_20 = grid_summary_scaled['Savings_20% (Wh)'].sum()
            overall_savings_30 = grid_summary_scaled['Savings_30% (Wh)'].sum()
            
            metric_cols = st.columns(2)
            
            with metric_cols[0]:
                st.metric(label="Total Original Grid Use", value=f"{overall_original_use:.2f} Wh")
                st.metric(label="Total Savings (20% Shift)", value=f"{overall_savings_20:.2f} Wh")

            with metric_cols[1]:
                st.metric(label="Total Adjusted Use (20%)", value=f"{(overall_original_use - overall_savings_20):.2f} Wh")
                st.metric(label="Total Savings (30% Shift)", value=f"{overall_savings_30:.2f} Wh")
                
            st.subheader("Aggregated Grid Usage and Savings")
            st.dataframe(grid_summary_scaled)
            
            # Download option
            st.download_button(
                label="Download Results as CSV",
                data=grid_summary_scaled.to_csv().encode('utf-8'),
                file_name='energy_analysis_results.csv',
                mime='text/csv'
            )
