import os
import glob
import pandas as pd
import numpy as np
import h3
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

# --- CONFIGURATION ---
# POINT TO THE SHARED GEOLIFE DATA FOLDER
DATA_PATH = "/workspace/Geolife/data"
H3_RES = 9 
TRIP_SPLIT_MINUTES = 20 

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in meters between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000 
    return c * r

def parse_plt(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=6, header=None, 
                         names=['lat', 'lon', 'zero', 'alt', 'days_num', 'date_str', 'time_str'])
        df['timestamp'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'])
        return df[['lat', 'lon', 'timestamp', 'alt']]
    except Exception:
        return pd.DataFrame()

def load_labels(user_dir):
    label_path = os.path.join(user_dir, 'labels.txt')
    if not os.path.exists(label_path):
        return None
    try:
        df_labels = pd.read_csv(label_path, sep='\t', header=0)
        df_labels['Start Time'] = pd.to_datetime(df_labels['Start Time'])
        df_labels['End Time'] = pd.to_datetime(df_labels['End Time'])
        return df_labels
    except Exception:
        return None

def process_user(user_id):
    user_dir = os.path.join(DATA_PATH, user_id)
    traj_dir = os.path.join(user_dir, 'Trajectory')
    
    if not os.path.exists(traj_dir):
        return []

    labels_df = load_labels(user_dir)
    processed_trips = []
    plt_files = glob.glob(os.path.join(traj_dir, "*.plt"))
    
    for plt_file in plt_files:
        df = parse_plt(plt_file)
        if df.empty: continue

        df['dt'] = df['timestamp'].diff().dt.total_seconds() / 60.0
        split_indices = df[df['dt'] > TRIP_SPLIT_MINUTES].index
        
        trips = []
        start_idx = 0
        for idx in split_indices:
            trips.append(df.iloc[start_idx:idx])
            start_idx = idx
        trips.append(df.iloc[start_idx:])

        for i, trip in enumerate(trips):
            if len(trip) < 10: continue
            
            # --- SPEED CALCULATION ---
            duration_sec = (trip['timestamp'].iloc[-1] - trip['timestamp'].iloc[0]).total_seconds()
            if duration_sec == 0: continue

            lats = trip['lat'].values
            lons = trip['lon'].values
            dist_meters = 0.0
            for k in range(len(trip)-1):
                dist_meters += haversine(lons[k], lats[k], lons[k+1], lats[k+1])
            
            avg_speed = dist_meters / duration_sec
            
            # H3 Mapping
            h3_seq = [h3.latlng_to_cell(r.lat, r.lon, H3_RES) for _, r in trip.iterrows()]
            h3_compressed = [x for k, x in enumerate(h3_seq) if k == 0 or x != h3_seq[k-1]]
            
            if len(h3_compressed) < 2: continue

            mode = "unlabeled"
            if labels_df is not None:
                trip_start = trip['timestamp'].iloc[0]
                trip_end = trip['timestamp'].iloc[-1]
                mask = (labels_df['Start Time'] <= trip_start) & (labels_df['End Time'] >= trip_end)
                matching_labels = labels_df[mask]
                if not matching_labels.empty:
                    mode = matching_labels.iloc[0]['Transportation Mode']

            processed_trips.append({
                'user_id': user_id,
                'trip_id': f"{user_id}_{os.path.basename(plt_file)}_{i}",
                'h3_sequence': h3_compressed,
                'raw_points_count': len(trip),
                'avg_speed_mps': avg_speed,    # <--- THIS IS THE KEY FIELD
                'mode': mode
            })
            
    return processed_trips

if __name__ == "__main__":
    all_data = []
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path not found: {DATA_PATH}")
        exit(1)
        
    user_folders = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    print(f"Found {len(user_folders)} users. Updating data with SPEED...")
    
    # Process only a subset for speed if needed, but doing all is better for the paper
    # user_folders = user_folders[:20] 
    
    for user_id in tqdm(user_folders):
        user_trips = process_user(user_id)
        all_data.extend(user_trips)
        
    final_df = pd.DataFrame(all_data)
    # Save to current directory
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geolife_graph_ready.pkl")
    final_df.to_pickle(output_path)
    print(f"Done! Saved to {output_path}")
