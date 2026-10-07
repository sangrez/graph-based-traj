
import os
import pandas as pd
import h3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import math
import shutil
from collections import Counter

# ================= CONFIGURATION =================
DATASET_ROOT = './Geolife/data' 
PROCESSED_DIR = './geolife_processed_full' # Folder for the 400k graph files

H3_RES = 9 
LIMIT_FILES = None   # None = PROCESS EVERYTHING (17,000+ files)
BATCH_SIZE = 64      
HIDDEN_DIM = 128
PRETRAIN_EPOCHS = 15
FINETUNE_EPOCHS = 30
NUM_WORKERS = 4      # Parallel loading from disk (set to 0 if on Windows and it crashes)
EDGE_DIM = 6         # dist, dt, speed, acceleration, heading_change, speed_ratio

# SLIDING WINDOW SETTINGS
WINDOW_SIZE = 100    
STRIDE = 50          
# =================================================

MODE_MAPPING = {'walk': 0, 'bike': 1, 'bus': 2, 'car': 3, 'taxi': 3, 'subway': 4, 'train': 4}
CLASS_NAMES = ['Walk', 'Bike', 'Bus', 'Car', 'Rail']
BEIJING_BBOX = (39.75, 40.10, 116.15, 116.60)

# --- CLEANUP (Optional: Only if you want to restart from zero) ---
# if os.path.exists(PROCESSED_DIR): shutil.rmtree(PROCESSED_DIR)

# --- HELPER FUNCTIONS ---
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = (lon2 - lon1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.degrees(np.arctan2(x,y))
    return (brng + 360) % 360

def get_traffic_features(df):
    if len(df) < 5: return None
    coords = list(zip(df['lat'], df['lon']))
    timestamps = df['timestamp'].values
    
    dists, speeds, bearings = [], [], []
    for i in range(1, len(coords)):
        try:
            d = geodesic(coords[i-1], coords[i]).meters
            t = max(1.0, timestamps[i] - timestamps[i-1])
            s = d / t
            if s > 60: continue 
            b = calculate_bearing(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
            dists.append(d)
            speeds.append(s)
            bearings.append(b)
        except: continue
            
    if not speeds: return None

    v85 = np.percentile(speeds, 85)
    stop_rate = sum(1 for s in speeds if s < 1.0) / len(speeds)
    speed_var = np.var(speeds)
    bearing_diffs = [abs(bearings[i] - bearings[i-1]) for i in range(1, len(bearings))]
    bearing_diffs = [d if d <= 180 else 360-d for d in bearing_diffs]
    hcr = np.mean(bearing_diffs) if bearing_diffs else 0
    try:
        beeline = geodesic(coords[0], coords[-1]).meters
        sinuosity = beeline / sum(dists) if sum(dists) > 0 else 0
    except: sinuosity = 0
    
    # Use log-scaling instead of hard clipping to preserve information for fast modes
    accel_vals = [abs(speeds[i] - speeds[i-1]) / max(1.0, timestamps[i+1] - timestamps[i]) 
                  for i in range(1, min(len(speeds), len(timestamps)-1))]
    mean_accel = np.mean(accel_vals) if accel_vals else 0
    
    return torch.tensor([
        np.log1p(v85) / np.log1p(60),           # log-scaled, no hard clip
        stop_rate, 
        np.log1p(speed_var) / np.log1p(100),     # log-scaled variance
        min(hcr / 90.0, 1.0), 
        sinuosity,
        np.log1p(mean_accel) / np.log1p(10),     # mean acceleration
    ], dtype=torch.float).unsqueeze(0)

def process_sequence_graph(df):
    h3_seq = [h3.latlng_to_cell(lat, lon, H3_RES) for lat, lon in zip(df['lat'], df['lon'])]
    timestamps = df['timestamp'].values
    coords = list(zip(df['lat'], df['lon']))
    
    nodes, edge_attrs = [h3_seq[0]], []
    prev_speed = 0
    prev_bearing = 0
    curr_node, curr_time, curr_coord = h3_seq[0], timestamps[0], coords[0]
    
    for i in range(1, len(h3_seq)):
        if h3_seq[i] != curr_node or (timestamps[i] - curr_time > 30):
            try:
                dist = geodesic(curr_coord, coords[i]).meters
                dt = max(1.0, float(timestamps[i] - curr_time))
                speed = dist / dt
                bearing = calculate_bearing(curr_coord[0], curr_coord[1], coords[i][0], coords[i][1])
                accel = (speed - prev_speed) / dt
                hdg_change = abs(bearing - prev_bearing)
                if hdg_change > 180: hdg_change = 360 - hdg_change
                speed_ratio = speed / max(prev_speed, 0.1)
                edge_attrs.append([
                    min(dist/1000, 1), min(dt/120, 1), min(speed/35, 1),
                    np.clip(accel / 5.0, -1, 1),       # acceleration (normalized)
                    hdg_change / 180.0,                 # heading change (0-1)
                    min(speed_ratio / 5.0, 1.0),        # speed ratio
                ])
                nodes.append(h3_seq[i])
                prev_speed = speed
                prev_bearing = bearing
                curr_node, curr_time, curr_coord = h3_seq[i], timestamps[i], coords[i]
            except: continue
    return nodes, edge_attrs

# --- DISK-BASED DATASET ---
class DiskBasedDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        # We manually set the processed dir
        self.custom_processed_dir = PROCESSED_DIR
        super(DiskBasedDataset, self).__init__(root, transform, pre_transform)
        
        # Load metadata if exists
        if os.path.exists(os.path.join(self.custom_processed_dir, 'vocab.pt')):
            self.vocab = torch.load(os.path.join(self.custom_processed_dir, 'vocab.pt'))
            self.metadata = torch.load(os.path.join(self.custom_processed_dir, 'metadata.pt'))
            self.total_len = self.metadata['len']

    @property
    def processed_dir(self):
        return self.custom_processed_dir

    @property
    def processed_file_names(self):
        # If vocab exists, we assume processed.
        return ['vocab.pt', 'metadata.pt']

    def len(self):
        return self.total_len

    def get(self, idx):
        # Load ONE graph from disk
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        print(f"--- STARTING FULL DATASET PROCESSING ---")
        print(f"Saving files to: {self.processed_dir}")
        
        all_hexagons = set()
        
        users = sorted([u for u in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, u))])
        
        # We will process in TWO PASSES
        # Pass 1: Build Vocabulary (Fast scan)
        # Pass 2: Build Graphs & Save (Detailed scan)
        
        # Actually, let's do 1 pass and update vocab dynamically, then map later? 
        # No, mapping later is hard with disk storage. 
        # We will do 1 pass, collect Hex Strings, and then SAVE with Strings. 
        # Then, in __init__, we load and convert String->Int on the fly? Too slow.
        # BEST: Just build vocab first.
        
        # --- PASS 1: BUILD VOCAB (Sampled) ---
        print("Pass 1/2: Building Vocabulary (Scanning for Hexagons)...")
        # We sample 20% of files to get a good vocab coverage quickly, or scan all if needed.
        # For production, we scan ALL.
        
        for user_id in tqdm(users):
            traj_dir = os.path.join(DATASET_ROOT, user_id, 'Trajectory')
            if not os.path.exists(traj_dir): continue
            
            files = os.listdir(traj_dir)
            if LIMIT_FILES and len(files) > LIMIT_FILES: files = files[:LIMIT_FILES]
            
            for plt_file in files:
                if not plt_file.endswith('.plt'): continue
                try:
                    # Quick read just for lat/lon
                    df = pd.read_csv(os.path.join(traj_dir, plt_file), skiprows=6, header=None, usecols=[0,1], names=['lat','lon'])
                    # BBox Filter
                    df = df[(df['lat'] >= BEIJING_BBOX[0]) & (df['lat'] <= BEIJING_BBOX[1])]
                    if df.empty: continue
                    
                    seq = [h3.latlng_to_cell(lat, lon, H3_RES) for lat, lon in zip(df['lat'], df['lon'])]
                    all_hexagons.update(seq)
                except: continue

        vocab = {h: i for i, h in enumerate(sorted(list(all_hexagons)))}
        torch.save(vocab, os.path.join(self.processed_dir, 'vocab.pt'))
        print(f"Vocab Size: {len(vocab)}")

        # --- PASS 2: GENERATE AND SAVE GRAPHS ---
        print("Pass 2/2: Generating Graphs & Saving to Disk...")
        
        graph_idx = 0
        
        for user_id in tqdm(users):
            user_dir = os.path.join(DATASET_ROOT, user_id)
            traj_dir = os.path.join(user_dir, 'Trajectory')
            
            # Load Labels
            labels = []
            label_path = os.path.join(user_dir, 'labels.txt')
            if os.path.exists(label_path):
                try:
                    ldf = pd.read_csv(label_path, sep='\t')
                    ldf['Start Time'] = pd.to_datetime(ldf['Start Time'])
                    ldf['End Time'] = pd.to_datetime(ldf['End Time'])
                    for _, r in ldf.iterrows():
                        m = r['Transportation Mode'].lower()
                        if m in MODE_MAPPING:
                            labels.append({'start': r['Start Time'], 'end': r['End Time'], 'mode': MODE_MAPPING[m]})
                except: pass
            
            if not os.path.exists(traj_dir): continue
            
            files = os.listdir(traj_dir)
            if LIMIT_FILES and len(files) > LIMIT_FILES: files = files[:LIMIT_FILES]

            for plt_file in files:
                if not plt_file.endswith('.plt'): continue
                try:
                    df_full = pd.read_csv(os.path.join(traj_dir, plt_file), skiprows=6, header=None, usecols=[0,1,5,6], names=['lat','lon','date','time'])
                except: continue
                
                # Cleanup
                df_full = df_full[(df_full['lat'] >= -90) & (df_full['lat'] <= 90)]
                df_full = df_full[df_full.apply(lambda x: (BEIJING_BBOX[0]<x['lat']<BEIJING_BBOX[1]) and (BEIJING_BBOX[2]<x['lon']<BEIJING_BBOX[3]), axis=1)]
                if len(df_full) < WINDOW_SIZE: continue

                df_full['dt'] = pd.to_datetime(df_full['date'] + ' ' + df_full['time'])
                df_full['timestamp'] = df_full['dt'].astype(np.int64) // 10**9
                
                # SLIDING WINDOW
                for start_idx in range(0, len(df_full) - WINDOW_SIZE, STRIDE):
                    df_chunk = df_full.iloc[start_idx : start_idx + WINDOW_SIZE].copy()
                    
                    chunk_start = df_chunk.iloc[0]['dt']
                    chunk_end = df_chunk.iloc[-1]['dt']
                    
                    mode = -1
                    for l in labels:
                        if l['start'] <= chunk_start and chunk_end <= l['end']:
                            mode = l['mode']
                            break
                    
                    # Features
                    nodes, edge_feats = process_sequence_graph(df_chunk)
                    if len(nodes) < 5: continue
                    global_feats = get_traffic_features(df_chunk)
                    if global_feats is None: continue

                    # Tensors
                    # Note: Using .get() on vocab is safer if new hexagons appear in Pass 2 (rare but possible)
                    # We map unknown hexes to 0
                    x = torch.tensor([vocab.get(h, 0) for h in nodes], dtype=torch.long)
                    src, dst = list(range(len(nodes)-1)), list(range(1, len(nodes)))
                    edge_index = torch.tensor([src, dst], dtype=torch.long)
                    
                    if len(edge_feats) != len(nodes)-1: continue
                    edge_attr = torch.tensor(edge_feats, dtype=torch.float)
                    
                    # Bidirectional edges for better message passing
                    rev_edge_index = torch.tensor([dst, src], dtype=torch.long)
                    edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
                    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
                    
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([int(user_id)]))
                    data.mode_label = torch.tensor([mode], dtype=torch.long)
                    data.traffic = global_feats
                    
                    # SAVE TO DISK
                    torch.save(data, os.path.join(self.processed_dir, f'data_{graph_idx}.pt'))
                    graph_idx += 1
        
        # Save Metadata
        self.total_len = graph_idx
        torch.save({'len': graph_idx}, os.path.join(self.processed_dir, 'metadata.pt'))
        print(f"DONE! Saved {graph_idx} graphs to disk.")

# --- MODEL ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # per-class weight tensor
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = (1-pt)**self.gamma * CE_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        return F_loss

class TrafficBusterGNN(nn.Module):
    def __init__(self, num_nodes, num_users, num_modes, embed_dim=128, edge_dim=EDGE_DIM, wide_dim=6):
        super(TrafficBusterGNN, self).__init__()
        self.edge_dim = edge_dim
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        # Project raw edge features to expected dim if they don't match
        self.edge_proj = nn.Linear(edge_dim, edge_dim) if edge_dim == EDGE_DIM else None
        self.conv1 = GATv2Conv(embed_dim, embed_dim, heads=4, concat=False, edge_dim=edge_dim, dropout=0.3)
        self.conv2 = GATv2Conv(embed_dim, embed_dim, heads=2, concat=False, edge_dim=edge_dim, dropout=0.3)
        self.conv3 = GATv2Conv(embed_dim, embed_dim, heads=1, concat=False, edge_dim=edge_dim, dropout=0.3)
        self.wide_net = nn.Sequential(nn.Linear(wide_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        combined_dim = embed_dim + 64
        self.head_user = nn.Linear(combined_dim, num_users)
        self.head_mode = nn.Sequential(
            nn.Linear(combined_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_modes)
        )

    def get_rep(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding(x)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr=edge_attr)) + x
        x = F.elu(self.conv3(x, edge_index, edge_attr=edge_attr)) + x
        z_graph = global_max_pool(x, batch)
        u = data.traffic.view(z_graph.size(0), -1)
        z_wide = self.wide_net(u)
        return torch.cat([z_graph, z_wide], dim=1)

    def forward(self, data, task='user'):
        z = self.get_rep(data)
        if task == 'user': return self.head_user(z)
        return self.head_mode(z)

# --- EXECUTION ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    
    # 1. Initialize Dataset
    dataset = DiskBasedDataset(root='.')
    
    num_locs = len(dataset.vocab)
    num_users = 182 
    
    print(f"Total Graphs: {len(dataset)}")
    
    # Detect actual edge_dim and wide_dim from the first data sample
    sample_data = torch.load(os.path.join(PROCESSED_DIR, 'data_0.pt'))
    actual_edge_dim = sample_data.edge_attr.shape[1]
    actual_wide_dim = sample_data.traffic.shape[1] if sample_data.traffic.dim() > 1 else sample_data.traffic.shape[0]
    print(f"Detected edge_dim={actual_edge_dim}, wide_dim={actual_wide_dim}")
    if actual_edge_dim != EDGE_DIM:
        print(f"WARNING: Data has {actual_edge_dim} edge features but EDGE_DIM={EDGE_DIM}. Using data's dimension.")
        print(f"  To use new features, delete '{PROCESSED_DIR}/' and re-run to reprocess.")
    
    # --- Stratified Train / Val / Test Split ---
    # Scan all labeled indices and their mode labels
    print("Scanning labels for stratified split...")
    labeled_indices = []
    unlabeled_indices = []
    label_list = []
    for i in tqdm(range(len(dataset)), desc="Scanning labels"):
        d = torch.load(os.path.join(PROCESSED_DIR, f'data_{i}.pt'))
        mode = d.mode_label.item()
        if mode != -1:
            labeled_indices.append(i)
            label_list.append(mode)
        else:
            unlabeled_indices.append(i)
    
    print(f"Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
    
    # Stratified split: 80% train, 10% val, 10% test
    from sklearn.model_selection import train_test_split
    labeled_indices = np.array(labeled_indices)
    label_list = np.array(label_list)
    
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        labeled_indices, label_list, test_size=0.2, random_state=42, stratify=label_list
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Train class distribution: {dict(Counter(train_labels))}")
    
    # Compute per-class weights (inverse frequency) for Focal Loss
    class_counts = Counter(train_labels)
    total_labeled = len(train_labels)
    num_classes = 5
    class_weights = torch.zeros(num_classes)
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_weights[c] = total_labeled / (num_classes * class_counts[c])
        else:
            class_weights[c] = 1.0
    print(f"Class weights: {class_weights.tolist()}")
    
    # Build subset loaders
    # For pretrain, use ALL data (labeled + unlabeled) since user ID is always available
    all_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # For finetune, only use labeled train indices
    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_subset = torch.utils.data.Subset(dataset, test_idx.tolist())
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = TrafficBusterGNN(num_locs, num_users, num_classes, HIDDEN_DIM, edge_dim=actual_edge_dim, wide_dim=actual_wide_dim).to(device)
    
    # ========================
    # PHASE 1: PRETRAIN (User ID)
    # ========================
    print("\n--- PHASE 1: Pretraining (User ID) ---")
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    # Cosine schedule for pretrain
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS, eta_min=1e-5)
    
    model.train()
    for epoch in range(PRETRAIN_EPOCHS):
        loss_sum = 0
        count = 0
        for batch in tqdm(all_loader, desc=f"Pretrain {epoch+1}/{PRETRAIN_EPOCHS}"):
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch, task='user')
            loss = crit(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            count += 1
        pretrain_scheduler.step()
        print(f"Pretrain Epoch {epoch+1} Loss: {loss_sum/count:.4f} | LR: {pretrain_scheduler.get_last_lr()[0]:.6f}")

    # ========================
    # PHASE 2: FINETUNE (Mode Classification)
    # ========================
    print("\n--- PHASE 2: Finetuning (Mode) ---")
    
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit_focal = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    
    # Cosine annealing with warm restarts
    finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6)
    
    best_val_f1 = 0.0
    patience = 8
    patience_counter = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        # --- Train ---
        model.train()
        loss_sum = 0
        count = 0
        
        for batch in tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{FINETUNE_EPOCHS}"):
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch, task='mode')
            loss = crit_focal(out, batch.mode_label.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            count += 1
        
        finetune_scheduler.step()
        avg_loss = loss_sum / max(1, count)
        
        # --- Validate ---
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch, task='mode')
                preds = out.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch.mode_label.flatten().cpu().numpy())
        
        val_report = classification_report(val_true, val_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        val_f1 = val_report['macro avg']['f1-score']
        val_acc = val_report['accuracy']
        
        print(f"FT Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | LR: {finetune_scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping on validation macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best model saved (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break
    
    # ========================
    # PHASE 3: EVALUATE ON TEST SET
    # ========================
    print("\n--- PHASE 3: Test Evaluation ---")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch, task='mode')
            preds = out.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch.mode_label.flatten().cpu().numpy())
    
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    print(classification_report(test_true, test_preds, target_names=CLASS_NAMES, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig('test_confusion.png', dpi=150)
    print("Saved test_confusion.png")
    
    torch.save(model.state_dict(), 'production_model.pth')
    print("Production model saved.")
