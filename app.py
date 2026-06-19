import streamlit as st
import pandas as pd
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

st.set_page_config(page_title="ParkFlow AI", layout="wide")

st.title("🚨 ParkFlow AI — Parking Congestion Impact Command Center")
st.markdown("Analyzing physical vehicle properties and temporal constraints to route BTP teams strategically.")
st.markdown("---")

processed_file = "data/processed_traffic_impact.csv"

if not os.path.exists(processed_file):
    st.error("❌ Processed file missing.")
    st.info("Execute `python src/train_pipeline.py` via your terminal to build data models first.")
else:
    df = pd.read_csv(processed_file)

    st.sidebar.header("🕹️ Filter Options")
    selected_hour = st.sidebar.slider("Hour of Day (24h)", 0, 23, 5)
    min_choke = st.sidebar.slider("Minimum Choke Priority Score", 1.0, 10.0, 2.0)

    # Filter dataframe
    filtered_df = df[(df['hour'] == selected_hour) & (df['choke_score'] >= min_choke)]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📍 Geospatial Bottleneck Heatmap")
        
        # Center map dynamically around your data points
        center = [12.9716, 77.5946]
        if not filtered_df.empty:
            center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
            
        m = folium.Map(location=center, zoom_start=13, tiles="CartoDB Positron")
        
        # Convert coordinates into HeatMap compatible matrix
        heat_data = filtered_df[['latitude', 'longitude', 'choke_score']].dropna().values.tolist()
        if heat_data:
            HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m)
        
        st_folium(m, width=700, height=500, returned_objects=[])

    with col2:
        st.subheader("⚠️ Clear-Zone Dispatch Priorities")
        
        if not filtered_df.empty:
            display_table = filtered_df.sort_values(by="choke_score", ascending=False)[
                ['hotspot_id', 'vehicle_type', 'violation_type', 'status', 'choke_score']
            ].head(10)
            
            display_table.columns = ['Zone ID', 'Vehicle Type', 'Violation Type', 'Verification Status', 'Choke Score']
            st.dataframe(
                display_table.style.background_gradient(cmap="OrRd", subset=['Choke Score']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No parking hazards currently present for this configuration criteria.")