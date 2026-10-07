import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import folium
import h3
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(BASE_DIR, "geolife_graph.pt")
DATA_FILE = os.path.join(BASE_DIR, "geolife_graph_ready.pkl")
EMBEDDING_FILE = os.path.join(BASE_DIR, "trajectory_embeddings.npy")
MAP_OUTPUT = os.path.join(BASE_DIR, "inference_map.html")
MATRIX_OUTPUT = os.path.join(BASE_DIR, "confusion_matrix.png")

# Class names based on your dataset's label encoder
CLASSES = ['airplane', 'bike', 'boat', 'bus', 'car', 'run', 'subway', 'taxi', 'train', 'walk']

def visualize_results():
    if not os.path.exists(EMBEDDING_FILE):
        print(f"Error: {EMBEDDING_FILE} not found. Run 03_train_and_evaluate.py first.")
        return

    print("Loading data for visualization...")
    df = pd.read_pickle(DATA_FILE)
    data = torch.load(GRAPH_FILE)
    embeddings = np.load(EMBEDDING_FILE)
    
    # ------------------------------------------------
    # VISUAL 2: INFERENCE MAP (Similarity Search)
    # ------------------------------------------------
    print("Generating Similarity Inference Map...")
    
    # Pick a "Walk" trip to query
    # Find indices where mode is 'walk'
    walk_indices = df[df['mode'] == 'walk'].index
    if len(walk_indices) > 0:
        query_idx = walk_indices[0] # Pick the first walking trip
    else:
        query_idx = 0 # Fallback
        
    query_vec = embeddings[query_idx].reshape(1, -1)
    
    # Find Top 3 Similar UNLABELED trips
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    
    # Sort
    sorted_indices = sim_scores.argsort()[::-1]
    
    # Filter for unlabeled
    found_trips = []
    for idx in sorted_indices:
        if idx == query_idx: continue
        if df.iloc[idx]['mode'] == 'unlabeled':
            found_trips.append(idx)
        if len(found_trips) >= 3:
            break
            
    # Plot on Map
    # Center map on query start
    start_hex = df.iloc[query_idx]['h3_sequence'][0]
    lat, lon = h3.cell_to_latlng(start_hex)
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles='CartoDB positron')
    
    # Helper to plot trip
    def plot_trip(idx, color, label, opacity=1.0):
        seq = df.iloc[idx]['h3_sequence']
        points = [h3.cell_to_latlng(h) for h in seq]
        folium.PolyLine(points, color=color, weight=5, opacity=opacity, tooltip=label).add_to(m)
        # Start Marker
        folium.CircleMarker(points[0], radius=5, color=color, fill=True).add_to(m)

    # Plot Query (Green)
    plot_trip(query_idx, 'green', f"QUERY: Walk (ID {query_idx})")
    
    # Plot Inferred (Blue dashed)
    for i, match_idx in enumerate(found_trips):
        score = sim_scores[match_idx]
        plot_trip(match_idx, 'blue', f"INFERRED: Unlabeled (ID {match_idx}, Sim: {score:.2f})", opacity=0.6)

    m.save(MAP_OUTPUT)
    print(f"Saved {MAP_OUTPUT}")

if __name__ == "__main__":
    visualize_results()
