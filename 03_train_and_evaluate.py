import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import folium
import h3
import os

# --- CONFIGURATION ---
# Ensure we look for files in the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(BASE_DIR, "geolife_graph.pt")
DATA_FILE = os.path.join(BASE_DIR, "geolife_graph_ready.pkl")
SLIDE1 = os.path.join(BASE_DIR, "Slide1_ConfusionMatrix.png")
SLIDE2 = os.path.join(BASE_DIR, "Slide2_Speed_Analysis.png")
SLIDE3 = os.path.join(BASE_DIR, "Slide3_Life_Patterns.html")
MODEL_FILE = os.path.join(BASE_DIR, "gnn_model.pth")
EMBEDDING_FILE = os.path.join(BASE_DIR, "trajectory_embeddings.npy")

CLASSES = ['airplane', 'bike', 'boat', 'bus', 'car', 'run', 'subway', 'taxi', 'train', 'walk']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 64
EPOCHS = 60
LR = 0.01

# --- 1. DEFINE MODEL ---
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_region_features, num_traj_features):
        super().__init__()
        self.region_lin = Linear(num_region_features, hidden_channels)
        self.traj_lin = Linear(num_traj_features, hidden_channels)

        self.conv1 = HeteroConv({
            ('region', 'neighbors', 'region'): SAGEConv(hidden_channels, hidden_channels),
            ('region', 'rev_visits', 'trajectory'): SAGEConv((-1, hidden_channels), hidden_channels),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('region', 'neighbors', 'region'): SAGEConv(hidden_channels, hidden_channels),
            ('region', 'rev_visits', 'trajectory'): SAGEConv((-1, hidden_channels), hidden_channels),
        }, aggr='mean')

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict['region'] = self.region_lin(x_dict['region']).relu()
        x_dict['trajectory'] = self.traj_lin(x_dict['trajectory']).relu()
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict['trajectory']

    def decode(self, z):
        return self.classifier(z)

# --- 2. TRAIN AND GENERATE DATA ---
def run_pipeline():
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: {GRAPH_FILE} not found. Run 02_build_graph.py first.")
        return

    print("Loading Graph Data...")
    data = torch.load(GRAPH_FILE)
    data = data.to(DEVICE)

    print("Initializing Model...")
    model = HeteroGNN(
        hidden_channels=HIDDEN_DIM, 
        out_channels=len(CLASSES),
        num_region_features=data['region'].x.shape[1],
        num_traj_features=data['trajectory'].x.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("Training Model (to generate fresh predictions)...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        z_traj = model(data.x_dict, data.edge_index_dict)
        out = model.decode(z_traj)
        mask = data['trajectory'].train_mask
        loss = F.cross_entropy(out[mask], data['trajectory'].y[mask])
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

    print("Training Complete. Saving Model and Embeddings...")
    torch.save(model.state_dict(), MODEL_FILE)
    
    model.eval()
    with torch.no_grad():
        z_traj = model(data.x_dict, data.edge_index_dict)
        out = model.decode(z_traj)
        preds = out.argmax(dim=1).cpu().numpy()
        embeddings = z_traj.cpu().numpy()
        
    np.save(EMBEDDING_FILE, embeddings)
    
    # Get True Labels for Confusion Matrix
    test_mask = data['trajectory'].test_mask.cpu().numpy()
    y_true = data['trajectory'].y.cpu().numpy()[test_mask]
    y_pred_test = preds[test_mask]

    # --- 3. GENERATE VISUALS ---
    
    # SLIDE 1: CONFUSION MATRIX
    print(f"Generating Slide 1: {SLIDE1}")
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Slide 1: Classification Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(SLIDE1)
    plt.close()

    # SLIDE 2: SPEED ANALYSIS
    print(f"Generating Slide 2: {SLIDE2}")
    try:
        df = pd.read_pickle(DATA_FILE)
        plot_df = df[df['mode'] != 'unlabeled'].copy()
        plt.figure(figsize=(12, 6))
        
        # Check if we have speed data (we should)
        if 'avg_speed_mps' in plot_df.columns:
            # Order likely by median speed
            order = plot_df.groupby(["mode"])["avg_speed_mps"].median().sort_values().index
            sns.boxplot(x='mode', y='avg_speed_mps', data=plot_df, order=order, palette="coolwarm", showfliers=False)
            plt.title('Slide 2: Velocity Profiles by Mode')
            plt.ylabel('Speed (m/s)')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(SLIDE2)
            plt.close()
        else:
             print("Warning: 'avg_speed_mps' not in data. Skipping Slide 2.")
    except Exception as e:
        print(f"Could not load DataFrame for Slide 2: {e}")

    # SLIDE 3: LIFE PATTERNS MAP
    print(f"Generating Slide 3: {SLIDE3}")
    try:
        unlabeled_mask = df['mode'] == 'unlabeled'
        if unlabeled_mask.sum() > 0:
            un_emb = embeddings[unlabeled_mask]
            
            # Cluster the UNLABELED data
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(un_emb)
            
            m = folium.Map(location=[39.9042, 116.4074], zoom_start=11, tiles='CartoDB dark_matter')
            colors = ['#FF0000', '#00FF00', '#0000FF'] # Red, Green, Blue
            
            # Plot sample from each cluster
            for c_id in range(3):
                c_idxs = np.where(clusters == c_id)[0]
                # Pick 20 random trips per cluster
                sample_idxs = np.random.choice(c_idxs, min(20, len(c_idxs)), replace=False)
                
                # Convert back to global dataframe index
                global_idxs = df.index[unlabeled_mask][sample_idxs]
                
                for idx in global_idxs:
                    seq = df.iloc[idx]['h3_sequence']
                    points = [h3.cell_to_latlng(h) for h in seq]
                    if len(points) > 1:
                        folium.PolyLine(points, color=colors[c_id], weight=2, opacity=0.6).add_to(m)

            legend_html = '''
             <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 90px; 
             background-color:rgba(255,255,255,0.8); z-index:9999; font-size:14px; border:2px solid grey; padding:5px; border-radius: 5px;">
             <b>Life Patterns (Unsupervised)</b><br>
             <i style="background:#FF0000;width:10px;height:10px;display:inline-block;border-radius:50%"></i> Pattern A (Ex. Commute)<br>
             <i style="background:#00FF00;width:10px;height:10px;display:inline-block;border-radius:50%"></i> Pattern B (Ex. Leisure)<br>
             <i style="background:#0000FF;width:10px;height:10px;display:inline-block;border-radius:50%"></i> Pattern C (Ex. Transit)
             </div>
             '''
            m.get_root().html.add_child(folium.Element(legend_html))
            m.save(SLIDE3)
        else:
            print("No unlabeled data found for Slide 3.")
    except Exception as e:
        print(f"Error generating Slide 3: {e}")

    print("\n--- DONE! ALL FILES CREATED ---")
    print(f"1. {SLIDE1}")
    print(f"2. {SLIDE2}")
    print(f"3. {SLIDE3}")

if __name__ == "__main__":
    run_pipeline()
