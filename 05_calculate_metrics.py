import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_FILE = os.path.join(BASE_DIR, "geolife_graph.pt")
MODEL_FILE = os.path.join(BASE_DIR, "gnn_model.pth")

CLASSES = ['airplane', 'bike', 'boat', 'bus', 'car', 'run', 'subway', 'taxi', 'train', 'walk']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 64

# --- 1. DEFINE MODEL (Must match training script) ---
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

def calculate_metrics():
    if not os.path.exists(GRAPH_FILE) or not os.path.exists(MODEL_FILE):
        print("Files not found. Please run previous scripts first.")
        return

    print("Loading Graph Data...")
    data = torch.load(GRAPH_FILE)
    data = data.to(DEVICE)

    print("Loading Model...")
    model = HeteroGNN(
        hidden_channels=HIDDEN_DIM, 
        out_channels=len(CLASSES),
        num_region_features=data['region'].x.shape[1],
        num_traj_features=data['trajectory'].x.shape[1]
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    print("Running Evaluation...")
    with torch.no_grad():
        z_traj = model(data.x_dict, data.edge_index_dict)
        out = model.decode(z_traj)
        preds = out.argmax(dim=1).cpu().numpy()
    
    # Get True Labels
    test_mask = data['trajectory'].test_mask.cpu().numpy()
    y_true = data['trajectory'].y.cpu().numpy()[test_mask]
    y_pred = preds[test_mask]

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    # Filter out classes that might not be in the test set to avoid warnings
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [CLASSES[i] for i in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == "__main__":
    calculate_metrics()
