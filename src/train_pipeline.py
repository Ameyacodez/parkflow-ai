import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from xgboost import XGBRegressor

def run_ml_pipeline():
    print("⏳ Loading real HackerEarth dataset sample...")
    
    csv_path = "data/hacker_earth_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing '{csv_path}'. Please place your dataset file inside the data/ folder.")
        
    # Read CSV without headers because your raw snippet has no column header row
    df = pd.read_csv(csv_path, header=None)
    
    # Assign specific column indexes based strictly on your data structure
    # Column 0: ID, Col 1: Lat, Col 2: Lon, Col 4: Vehicle Type, Col 7: Violation, Col 9: Timestamp, Col 22: Status
    df = df.rename(columns={
        1: 'latitude',
        2: 'longitude',
        4: 'vehicle_type',
        7: 'violation_type',
        9: 'timestamp',
        22: 'status'
    })
    
    # Clean and parse coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Convert Timestamp to extract temporal features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour.fillna(12).astype(int)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.fillna(0).astype(int)

    print("📍 Grouping spatial coordinates into Hotspots (DBSCAN)...")
    # Spatial Clustering: Group violations within ~150 meters of each other
    coords = df[['latitude', 'longitude']].values
    db = DBSCAN(eps=0.15/6371.0088, min_samples=2, metric='haversine').fit(np.radians(coords))
    df['hotspot_id'] = db.labels_

    print("🧠 Engineering Traffic Choke Impact Proxy...")
    # Map vehicle size weights (larger vehicle = worse choke impact)
    vehicle_weights = {'MAXI-CAB': 3.0, 'VAN': 2.5, 'CAR': 2.0, 'PASSENGER AUTO': 1.5, 'SCOOTER': 1.0, 'MOTOR CYCLE': 1.0}
    df['vehicle_weight'] = df['vehicle_type'].map(vehicle_weights).fillna(1.0)
    
    # Map operational validation weights (Approved violations are guaranteed bottlenecks)
    status_weights = {'approved': 1.5, 'rejected': 0.5}
    df['status_weight'] = df['status'].map(status_weights).fillna(1.0)
    
    # Map peak traffic hour multipliers for Bengaluru
    # Rush hours (8-11 AM, 5-9 PM) generate much higher congestion deltas
    df['time_multiplier'] = df['hour'].apply(lambda h: 2.0 if (8 <= h <= 11 or 17 <= h <= 21) else 1.0)
    
    # Target Formula: Quantified base Congestion Impact Score
    df['congestion_impact'] = df['vehicle_weight'] * df['status_weight'] * df['time_multiplier']

    # Train an XGBoost Regressor to predict congestion impact based on location and time features
    X = df[['latitude', 'longitude', 'hour', 'day_of_week', 'hotspot_id']]
    y = df['congestion_impact']
    
    print("🤖 Training Prediction Regressor Engine...")
    model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # Apply predictions and normalize to a 1-10 scale for enforcement dispatchers
    df['predicted_impact'] = model.predict(X)
    max_val = df['predicted_impact'].max() if df['predicted_impact'].max() > 0 else 1
    df['choke_score'] = ((df['predicted_impact'] / max_val) * 10).clip(1, 10).round(1)

    # Save output for frontend dashboard visualization
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_traffic_impact.csv", index=False)
    print("✅ Successfully generated: data/processed_traffic_impact.csv")

if __name__ == "__main__":
    run_ml_pipeline()