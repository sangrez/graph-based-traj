"""
7-Class version matching the paper's classification:
  Walk, Bike, Bus, Car, Subway, Taxi, Train

Improvements over V1:
  1. Time-of-day features (sin/cos hour + day-of-week) — key Taxi discriminator
  2. Taxi-specific brief-stop rate (30-120s curb stops from pickup/dropoff)
  3. Velocity entropy (irregular for Taxi/Bus, smooth for Train/Car)
  4. Learnable attention-weighted pooling instead of global max pooling
  5. HIDDEN_DIM=256, 4-layer GATv2 with residuals
  6. Bidirectional + skip-1 edges (node i->i+2) for larger receptive field
  7. Label smoothing (0.1) on Focal Loss
  8. Warmup + cosine LR for pretrain; cosine-warm-restart for finetune

Direct comparison with Table 2 of:
  Tsanakas et al., FGCS 2024 (S0167739X23004867)
"""

import os
import pandas as pd
import h3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
import torch_geometric
from torch_geometric.nn import GATv2Conv, global_max_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import math
from collections import Counter

# ================= CONFIGURATION =================
DATASET_ROOT = './Geolife/data'
PROCESSED_DIR = './geolife_processed_7class_v2'

H3_RES = 9
LIMIT_FILES = None
BATCH_SIZE = 64
HIDDEN_DIM = 256
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 40
NUM_WORKERS = 4
EDGE_DIM = 6
WIDE_DIM = 12   # 6 physics + 4 time-of-day + 2 stop/entropy

WINDOW_SIZE = 100
STRIDE = 50
# =================================================

MODE_MAPPING = {'walk': 0, 'bike': 1, 'bus': 2, 'car': 3, 'subway': 4, 'taxi': 5., 'train': 6}
CLASS_NAMES = ['Walk', 'Bike', 'Bus', 'Car', 'Subway', 'Taxi', 'Train']
NUM_CLASSES = 7
BEIJING_BBOX = (39.75, 40.10, 116.15, 116.60)


def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon)))
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def get_traffic_features(df, chunk_dt):
    """
    Returns a 12-dim feature vector:
      [0]  log-scaled v85
      [1]  stop_rate (general, speed < 1 m/s)
      [2]  log-scaled speed variance
      [3]  heading change rate
      [4]  straightness ratio (beeline/path, inverse sinuosity, range [0,1])
      [5]  log-scaled mean acceleration
      [6]  sin(2*pi*hour/24)     <- TIME OF DAY
      [7]  cos(2*pi*hour/24)
      [8]  sin(2*pi*dow/7)       <- DAY OF WEEK
      [9]  cos(2*pi*dow/7)
      [10] brief_stop_rate       <- TAXI PICKUP SIGNATURE (stops 30-120s)
      [11] velocity_entropy      <- irregular speed distribution
    """
    if len(df) < 5:
        return None

    coords = list(zip(df['lat'], df['lon']))
    timestamps = df['timestamp'].values

    dists, speeds, bearings, stop_durations = [], [], [], []
    filtered_ts = []   # timestamps aligned with the filtered speeds list
    i = 0
    while i < len(coords) - 1:
        try:
            d = geodesic(coords[i], coords[i+1]).meters
            t = max(1.0, float(timestamps[i+1] - timestamps[i]))
            s = d / t
            if s > 60:
                i += 1
                continue
            b = calculate_bearing(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
            dists.append(d)
            speeds.append(s)
            bearings.append(b)
            filtered_ts.append(float(timestamps[i]))   # record start-of-segment timestamp
            # Brief taxi-stop: low speed, 30–180s (avoids counting GPS logging gaps)
            if s < 0.5 and 30 <= t <= 180:
                stop_durations.append(t)
        except Exception:
            pass
        i += 1

    if not speeds:
        return None

    v85 = np.percentile(speeds, 85)
    stop_rate = sum(1 for s in speeds if s < 1.0) / len(speeds)
    speed_var = np.var(speeds)
    bearing_diffs = [abs(bearings[j] - bearings[j-1]) for j in range(1, len(bearings))]
    bearing_diffs = [d if d <= 180 else 360 - d for d in bearing_diffs]
    hcr = np.mean(bearing_diffs) if bearing_diffs else 0.0

    try:
        beeline = geodesic(coords[0], coords[-1]).meters
        # Straightness ratio (beeline / path length), range [0,1].
        # 1 = perfectly straight; <1 = curved/backtracking.
        # Note: inverse of the conventional sinuosity definition.
        sinuosity = beeline / max(sum(dists), 1e-6)
    except Exception:
        sinuosity = 0.0

    # Use filtered_ts so time deltas match the filtered speed segments exactly
    accel_vals = [abs(speeds[j] - speeds[j-1]) / max(1.0, filtered_ts[j] - filtered_ts[j-1])
                  for j in range(1, len(speeds))]
    mean_accel = np.mean(accel_vals) if accel_vals else 0.0

    hour = chunk_dt.hour + chunk_dt.minute / 60.0
    dow = chunk_dt.dayofweek
    sin_hour = math.sin(2 * math.pi * hour / 24)
    cos_hour = math.cos(2 * math.pi * hour / 24)
    sin_dow  = math.sin(2 * math.pi * dow / 7)
    cos_dow  = math.cos(2 * math.pi * dow / 7)

    total_trip_time = max(float(timestamps[-1] - timestamps[0]), 1.0)
    brief_stop_rate = sum(min(t, 120) for t in stop_durations) / total_trip_time
    brief_stop_rate = float(min(brief_stop_rate, 1.0))

    if len(speeds) > 1:
        hist, _ = np.histogram(speeds, bins=10, range=(0, 30), density=False)
        hist = hist.astype(float) + 1e-8
        hist /= hist.sum()
        vel_entropy = float(-np.sum(hist * np.log2(hist)) / np.log2(10))
    else:
        vel_entropy = 0.0

    return torch.tensor([
        float(np.log1p(v85) / np.log1p(60)),
        float(stop_rate),
        float(np.log1p(speed_var) / np.log1p(100)),
        float(min(hcr / 90.0, 1.0)),
        float(sinuosity),
        float(np.log1p(mean_accel) / np.log1p(10)),
        sin_hour,
        cos_hour,
        sin_dow,
        cos_dow,
        brief_stop_rate,
        vel_entropy,
    ], dtype=torch.float).unsqueeze(0)


def process_sequence_graph(df):
    h3_seq = [h3.latlng_to_cell(lat, lon, H3_RES) for lat, lon in zip(df['lat'], df['lon'])]
    timestamps = df['timestamp'].values
    coords = list(zip(df['lat'], df['lon']))

    nodes, edge_attrs = [h3_seq[0]], []
    prev_speed = 0.0
    prev_bearing = 0.0
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
                if hdg_change > 180:
                    hdg_change = 360 - hdg_change
                speed_ratio = speed / max(prev_speed, 0.1)
                edge_attrs.append([
                    min(dist / 1000, 1.0),
                    min(dt / 120, 1.0),
                    min(speed / 35, 1.0),
                    float(np.clip(accel / 5.0, -1, 1)),
                    hdg_change / 180.0,
                    min(speed_ratio / 5.0, 1.0),
                ])
                nodes.append(h3_seq[i])
                prev_speed = speed
                prev_bearing = bearing
                curr_node, curr_time, curr_coord = h3_seq[i], timestamps[i], coords[i]
            except Exception:
                continue

    return nodes, edge_attrs


def build_edge_tensors(nodes, edge_feats):
    """
    Sequential + reverse + skip-1 + skip-1-reverse edges.
    """
    n = len(nodes)
    if n < 2:
        return None, None

    src, dst, attrs = [], [], []
    for i in range(n - 1):
        src.append(i);   dst.append(i+1); attrs.append(edge_feats[i])
        src.append(i+1); dst.append(i);   attrs.append(edge_feats[i])
    for i in range(n - 2):
        src.append(i);   dst.append(i+2); attrs.append(edge_feats[i])
        src.append(i+2); dst.append(i);   attrs.append(edge_feats[i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(attrs, dtype=torch.float)
    return edge_index, edge_attr


class DiskBasedDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.custom_processed_dir = PROCESSED_DIR
        super(DiskBasedDataset, self).__init__(root, transform, pre_transform)
        if os.path.exists(os.path.join(self.custom_processed_dir, 'vocab.pt')):
            self.vocab = torch.load(os.path.join(self.custom_processed_dir, 'vocab.pt'))
            self.metadata = torch.load(os.path.join(self.custom_processed_dir, 'metadata.pt'))
            self.total_len = self.metadata['len']

    @property
    def processed_dir(self):
        return self.custom_processed_dir

    @property
    def processed_file_names(self):
        return ['vocab.pt', 'metadata.pt']

    def len(self):
        return self.total_len

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        print("=== PROCESSING 7-CLASS V2 DATASET ===")

        all_hexagons = set()
        users = sorted([u for u in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, u))])

        print("Pass 1/2: Building vocabulary...")
        for user_id in tqdm(users):
            traj_dir = os.path.join(DATASET_ROOT, user_id, 'Trajectory')
            if not os.path.exists(traj_dir):
                continue
            files = os.listdir(traj_dir)
            if LIMIT_FILES:
                files = files[:LIMIT_FILES]
            for plt_file in files:
                if not plt_file.endswith('.plt'):
                    continue
                try:
                    df = pd.read_csv(os.path.join(traj_dir, plt_file), skiprows=6,
                                     header=None, usecols=[0, 1], names=['lat', 'lon'])
                    df = df[(df['lat'] >= BEIJING_BBOX[0]) & (df['lat'] <= BEIJING_BBOX[1])]
                    if df.empty:
                        continue
                    all_hexagons.update(
                        h3.latlng_to_cell(lat, lon, H3_RES)
                        for lat, lon in zip(df['lat'], df['lon'])
                    )
                except Exception:
                    continue

        vocab = {h: i for i, h in enumerate(sorted(all_hexagons))}
        torch.save(vocab, os.path.join(self.processed_dir, 'vocab.pt'))
        print(f"Vocabulary size: {len(vocab)}")

        print("Pass 2/2: Generating graphs...")
        graph_idx = 0

        for user_id in tqdm(users):
            user_dir = os.path.join(DATASET_ROOT, user_id)
            traj_dir = os.path.join(user_dir, 'Trajectory')

            labels = []
            label_path = os.path.join(user_dir, 'labels.txt')
            if os.path.exists(label_path):
                try:
                    ldf = pd.read_csv(label_path, sep='\t')
                    ldf['Start Time'] = pd.to_datetime(ldf['Start Time'])
                    ldf['End Time'] = pd.to_datetime(ldf['End Time'])
                    for _, row in ldf.iterrows():
                        m = row['Transportation Mode'].lower()
                        if m in MODE_MAPPING:
                            labels.append({
                                'start': row['Start Time'],
                                'end': row['End Time'],
                                'mode': MODE_MAPPING[m]
                            })
                except Exception:
                    pass

            if not os.path.exists(traj_dir):
                continue
            files = os.listdir(traj_dir)
            if LIMIT_FILES:
                files = files[:LIMIT_FILES]

            for plt_file in files:
                if not plt_file.endswith('.plt'):
                    continue
                try:
                    df_full = pd.read_csv(
                        os.path.join(traj_dir, plt_file), skiprows=6,
                        header=None, usecols=[0, 1, 5, 6],
                        names=['lat', 'lon', 'date', 'time']
                    )
                except Exception:
                    continue

                df_full = df_full[(df_full['lat'] >= -90) & (df_full['lat'] <= 90)]
                df_full = df_full[df_full.apply(
                    lambda r: (BEIJING_BBOX[0] < r['lat'] < BEIJING_BBOX[1]) and
                              (BEIJING_BBOX[2] < r['lon'] < BEIJING_BBOX[3]), axis=1
                )]
                if len(df_full) < WINDOW_SIZE:
                    continue

                df_full['dt'] = pd.to_datetime(df_full['date'] + ' ' + df_full['time'])
                df_full['timestamp'] = df_full['dt'].astype(np.int64) // 10 ** 9

                for start_idx in range(0, len(df_full) - WINDOW_SIZE, STRIDE):
                    df_chunk = df_full.iloc[start_idx:start_idx + WINDOW_SIZE].copy()
                    chunk_start_dt = df_chunk.iloc[0]['dt']
                    chunk_end_dt   = df_chunk.iloc[-1]['dt']

                    mode = -1
                    for lbl in labels:
                        if lbl['start'] <= chunk_start_dt and chunk_end_dt <= lbl['end']:
                            mode = lbl['mode']
                            break

                    nodes, edge_feats = process_sequence_graph(df_chunk)
                    if len(nodes) < 5 or len(edge_feats) != len(nodes) - 1:
                        continue

                    global_feats = get_traffic_features(df_chunk, chunk_start_dt)
                    if global_feats is None:
                        continue

                    x = torch.tensor([vocab.get(h, 0) for h in nodes], dtype=torch.long)
                    edge_index, edge_attr = build_edge_tensors(nodes, edge_feats)
                    if edge_index is None:
                        continue

                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=torch.tensor([int(user_id)]))
                    data.mode_label = torch.tensor([mode], dtype=torch.long)
                    data.traffic = global_feats

                    torch.save(data, os.path.join(self.processed_dir, f'data_{graph_idx}.pt'))
                    graph_idx += 1

        self.total_len = graph_idx
        torch.save({'len': graph_idx}, os.path.join(self.processed_dir, 'metadata.pt'))
        print(f"Done. Saved {graph_idx} graphs.")


class AttentionPool(nn.Module):
    """Learnable soft-attention aggregation over node embeddings per graph."""

    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x, batch):
        scores = self.gate(x)               # [N, 1]
        alpha  = pyg_softmax(scores, batch) # per-graph softmax, [N, 1]
        return global_add_pool(alpha * x, batch)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_prob = F.log_softmax(inputs, dim=1)
        ce_loss = -(smooth_targets * log_prob).sum(dim=1)
        pt = torch.exp(-ce_loss)
        f_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            f_loss = alpha_t * f_loss
        return f_loss.mean()


class TrafficBusterGNN(nn.Module):
    def __init__(self, num_nodes, num_users, num_modes,
                 embed_dim=256, edge_dim=EDGE_DIM, wide_dim=WIDE_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)

        # 4-layer GATv2 with residual connections
        self.conv1 = GATv2Conv(embed_dim, embed_dim, heads=4, concat=False,
                               edge_dim=edge_dim, dropout=0.3)
        self.conv2 = GATv2Conv(embed_dim, embed_dim, heads=4, concat=False,
                               edge_dim=edge_dim, dropout=0.3)
        self.conv3 = GATv2Conv(embed_dim, embed_dim, heads=2, concat=False,
                               edge_dim=edge_dim, dropout=0.25)
        self.conv4 = GATv2Conv(embed_dim, embed_dim, heads=1, concat=False,
                               edge_dim=edge_dim, dropout=0.2)

        self.attn_pool = AttentionPool(embed_dim)

        # Wide branch for physics + temporal
        self.wide_net = nn.Sequential(
            nn.Linear(wide_dim, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 128),     nn.BatchNorm1d(128), nn.GELU(),
        )

        combined = embed_dim + 128

        self.head_user = nn.Linear(combined, num_users)

        self.head_mode = nn.Sequential(
            nn.Linear(combined, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256),      nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),      nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_modes),
        )

    def get_rep(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding(x)
        r = x
        x = F.gelu(self.conv1(x, ei, edge_attr=ea))
        x = F.gelu(self.conv2(x, ei, edge_attr=ea)) + r      # residual
        r2 = x
        x = F.gelu(self.conv3(x, ei, edge_attr=ea))
        x = F.gelu(self.conv4(x, ei, edge_attr=ea)) + r2     # residual

        # Combine attention pool + max pool
        z_attn = self.attn_pool(x, batch)
        z_max  = global_max_pool(x, batch)
        z_graph = z_attn + z_max

        u = data.traffic.view(z_graph.size(0), -1)
        z_wide = self.wide_net(u)
        return torch.cat([z_graph, z_wide], dim=1)

    def forward(self, data, task='user'):
        z = self.get_rep(data)
        if task == 'user':
            return self.head_user(z)
        return self.head_mode(z)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = DiskBasedDataset(root='.')
    num_locs  = len(dataset.vocab)
    num_users = 182
    print(f"Total Graphs: {len(dataset)}")

    sample = torch.load(os.path.join(PROCESSED_DIR, 'data_0.pt'))
    actual_edge_dim = sample.edge_attr.shape[1]
    actual_wide_dim = sample.traffic.shape[1] if sample.traffic.dim() > 1 else sample.traffic.shape[0]
    print(f"Detected edge_dim={actual_edge_dim}, wide_dim={actual_wide_dim}")
    if actual_wide_dim != WIDE_DIM:
        print(f"WARNING: Expected WIDE_DIM={WIDE_DIM} but data has {actual_wide_dim}.")
        print("         Delete the processed directory and re-run to generate new features.")

    # Stratified split
    print("Scanning labels for stratified split...")
    labeled_indices, label_list = [], []
    for i in tqdm(range(len(dataset)), desc="Scanning"):
        d = torch.load(os.path.join(PROCESSED_DIR, f'data_{i}.pt'))
        m = d.mode_label.item()
        if m != -1:
            labeled_indices.append(i)
            label_list.append(m)

    print(f"Labeled samples: {len(labeled_indices)}")
    print(f"Class distribution: {dict(sorted(Counter(label_list).items()))}")

    from sklearn.model_selection import train_test_split
    labeled_indices = np.array(labeled_indices)
    label_list      = np.array(label_list)

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        labeled_indices, label_list, test_size=0.20, random_state=42, stratify=label_list
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    class_counts  = Counter(train_labels.tolist())
    total_labeled = len(train_labels)
    class_weights = torch.tensor([
        total_labeled / (NUM_CLASSES * max(class_counts[c], 1))
        for c in range(NUM_CLASSES)
    ])
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    all_loader   = DataLoader(dataset,
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx.tolist()),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(torch.utils.data.Subset(dataset, val_idx.tolist()),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(torch.utils.data.Subset(dataset, test_idx.tolist()),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = TrafficBusterGNN(
        num_locs, num_users, NUM_CLASSES,
        embed_dim=HIDDEN_DIM,
        edge_dim=actual_edge_dim,
        wide_dim=actual_wide_dim,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===========================
    # PHASE 1: PRETRAIN (user ID)
    # ===========================
    print("\n--- PHASE 1: Pretraining ---")
    opt_pre  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit_ce  = nn.CrossEntropyLoss()

    def warmup_cosine(epoch):
        warmup = 2
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(PRETRAIN_EPOCHS - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched_pre = torch.optim.lr_scheduler.LambdaLR(opt_pre, warmup_cosine)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in tqdm(all_loader, desc=f"Pretrain {epoch+1}/{PRETRAIN_EPOCHS}", leave=False):
            batch = batch.to(device)
            opt_pre.zero_grad()
            out  = model(batch, task='user')
            loss = crit_ce(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_pre.step()
            loss_sum += loss.item()
            count    += 1
        sched_pre.step()
        lr = sched_pre.get_last_lr()[0]
        print(f"Pretrain {epoch+1:>3}/{PRETRAIN_EPOCHS} | Loss: {loss_sum/count:.4f} | LR: {lr:.6f}")

    # ===========================
    # PHASE 2: FINETUNE (7-class)
    # ===========================
    print("\n--- PHASE 2: Finetuning ---")
    opt_ft = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    focal  = FocalLoss(alpha=class_weights, gamma=2.5, smoothing=0.1)
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_ft, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_val_f1, patience, patience_counter = 0.0, 10, 0

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"FT {epoch+1}/{FINETUNE_EPOCHS}", leave=False):
            batch = batch.to(device)
            opt_ft.zero_grad()
            out  = model(batch, task='mode')
            loss = focal(out, batch.mode_label.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_ft.step()
            loss_sum += loss.item()
            count    += 1
        sched_ft.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch, task='mode').argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch.mode_label.flatten().cpu().numpy())

        rep    = classification_report(val_true, val_preds, target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        val_f1  = rep['macro avg']['f1-score']
        val_acc = rep['accuracy']
        lr      = sched_ft.get_last_lr()[0]
        print(f"FT {epoch+1:>3}/{FINETUNE_EPOCHS} | Loss: {loss_sum/count:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | LR: {lr:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_7class_v2.pth')
            print(f"  -> Best saved (val F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stop at epoch {epoch+1}")
                break

    # ===========================
    # PHASE 3: TEST
    # ===========================
    print("\n--- PHASE 3: Test Evaluation ---")
    model.load_state_dict(torch.load('best_model_7class_v2.pth'))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch, task='mode').argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch.mode_label.flatten().cpu().numpy())

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(test_true, test_preds, target_names=CLASS_NAMES, zero_division=0))

    paper_f1 = {
        'Walk': 0.8982, 'Bike': 0.9112, 'Bus': 0.8722,
        'Car': 0.7240,  'Subway': 0.7734, 'Taxi': 0.7137, 'Train': 0.9553
    }
    our_rep = classification_report(test_true, test_preds, target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)

    print("=" * 60)
    print("HEAD-TO-HEAD vs PAPER (Table 2, Tsanakas et al. FGCS 2024)")
    print("=" * 60)
    print(f"{'Class':<10} {'Paper F1':>10} {'Ours F1':>10} {'Delta':>10}")
    print("-" * 44)
    p_f1s, o_f1s = [], []
    for cls in CLASS_NAMES:
        pf = paper_f1[cls]
        of = our_rep[cls]['f1-score']
        p_f1s.append(pf)
        o_f1s.append(of)
        d = of - pf
        sign = '+' if d >= 0 else ''
        print(f"{cls:<10} {pf:>10.4f} {of:>10.4f} {sign}{d:>9.4f}")
    print("-" * 44)
    d_mac = np.mean(o_f1s) - np.mean(p_f1s)
    sign  = '+' if d_mac >= 0 else ''
    print(f"{'MacroAvg':<10} {np.mean(p_f1s):>10.4f} {np.mean(o_f1s):>10.4f} {sign}{d_mac:>9.4f}")
    print(f"\nPaper Accuracy: 88.73%  |  Paper Balanced Acc: 81.69%")
    print(f"Ours  Accuracy: {our_rep['accuracy']*100:.2f}%")

    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('7-Class Confusion Matrix v2')
    plt.tight_layout()
    plt.savefig('test_confusion_7class_v2.png', dpi=150)
    print("Saved test_confusion_7class_v2.png")

    torch.save(model.state_dict(), 'production_model_7class_v2.pth')
    print("Production model saved.")
