import torch
import pandas as pd
import numpy as np
import h3
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

INPUT_FILE = "geolife_graph_ready.pkl"
OUTPUT_FILE = "geolife_graph.pt"

def build_graph():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 01_preprocess.py first.")
        return

    print("Loading processed data...")
    try:
        df = pd.read_pickle(INPUT_FILE)
    except Exception:
        import pickle
        with open(INPUT_FILE, 'rb') as f:
            df = pickle.load(f)
    
    # 1. Region Nodes
    print("Indexing regions...")
    all_hexes = set()
    for seq in df['h3_sequence']:
        all_hexes.update(seq)
    
    hex_to_id = {h: i for i, h in enumerate(all_hexes)}
    id_to_hex = {i: h for h, i in hex_to_id.items()}
    num_regions = len(all_hexes)
    
    region_features = []
    for i in range(num_regions):
        lat, lon = h3.cell_to_latlng(id_to_hex[i])
        region_features.append([lat, lon])
    
    region_features = torch.tensor(region_features, dtype=torch.float)
    if num_regions > 1:
        region_features = (region_features - region_features.mean(0)) / (region_features.std(0) + 1e-6)

    # 2. Trajectory Nodes (NOW WITH SPEED)
    num_trajectories = len(df)
    
    # Feature 1: Duration (Points count) - Log normalized
    feat_duration = torch.tensor(df['raw_points_count'].values, dtype=torch.float).unsqueeze(1)
    feat_duration = torch.log(feat_duration + 1)
    
    # Feature 2: Speed (Meters/Sec) - Normalized
    speeds = df['avg_speed_mps'].values
    # Handle crazy outliers (GPS errors giving 10000 m/s)
    speeds = np.clip(speeds, 0, 120) # Cap at 120 m/s (~430 km/h) (High Speed Train limit)
    feat_speed = torch.tensor(speeds, dtype=torch.float).unsqueeze(1)
    # Simple normalization
    feat_speed = (feat_speed - feat_speed.mean()) / (feat_speed.std() + 1e-6)
    
    # Combine Features: [Duration, Speed]
    traj_features = torch.cat([feat_duration, feat_speed], dim=1)
    print(f"Trajectory Feature Shape: {traj_features.shape} (Should be [N, 2])")

    # 3. Edges
    print("Building edges...")
    traj_src = []
    region_dst = []
    
    for traj_idx, row in tqdm(df.iterrows(), total=num_trajectories, desc="Linking"):
        for h3_hex in row['h3_sequence']:
            if h3_hex in hex_to_id:
                traj_src.append(traj_idx)
                region_dst.append(hex_to_id[h3_hex])
                
    traj_to_region_edge_index = torch.tensor([traj_src, region_dst], dtype=torch.long)

    # Neighbors
    region_src = []
    region_dst_spatial = []
    for h_str, h_id in tqdm(hex_to_id.items(), desc="Neighbors"):
        neighbors = h3.grid_disk(h_str, 1)
        for n in neighbors:
            if n in hex_to_id and n != h_str:
                region_src.append(h_id)
                region_dst_spatial.append(hex_to_id[n])
    region_edge_index = torch.tensor([region_src, region_dst_spatial], dtype=torch.long)

    # 4. Labels
    print("Encoding labels...")
    le = LabelEncoder()
    known_mask = df['mode'] != 'unlabeled'
    y_encoded = le.fit_transform(df.loc[known_mask, 'mode'])
    print(f"Classes: {le.classes_}")
    
    y_full = torch.full((num_trajectories,), -1, dtype=torch.long)
    indices = torch.tensor(known_mask.values.nonzero()[0])
    y_full[indices] = torch.tensor(y_encoded, dtype=torch.long)

    train_mask = torch.zeros(num_trajectories, dtype=torch.bool)
    test_mask = torch.zeros(num_trajectories, dtype=torch.bool)
    labeled_indices = np.where(known_mask)[0]
    np.random.shuffle(labeled_indices)
    split = int(len(labeled_indices) * 0.8)
    train_mask[labeled_indices[:split]] = True
    test_mask[labeled_indices[split:]] = True

    # 5. Assemble
    data = HeteroData()
    data['region'].x = region_features
    data['region'].num_nodes = num_regions
    data['trajectory'].x = traj_features
    data['trajectory'].y = y_full
    data['trajectory'].train_mask = train_mask
    data['trajectory'].test_mask = test_mask
    data['trajectory'].num_nodes = num_trajectories
    
    data['trajectory', 'visits', 'region'].edge_index = traj_to_region_edge_index
    data['region', 'rev_visits', 'trajectory'].edge_index = torch.flip(traj_to_region_edge_index, [0])
    data['region', 'neighbors', 'region'].edge_index = region_edge_index

    print("Graph Re-Built with SPEED Features!")
    torch.save(data, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_graph()
