import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import SAGEConv, global_max_pool
from tqdm import tqdm


CLASS_NAMES = ['Walk', 'Bike', 'Bus', 'Car', 'Subway', 'Taxi', 'Train']
NUM_CLASSES = len(CLASS_NAMES)
RANDOM_STATE = 42


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_macro_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': p,
        'recall': r,
        'f1': f1,
    }


class DiskDataset:
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.meta = torch.load(os.path.join(processed_dir, 'metadata.pt'), map_location='cpu')
        self.vocab = torch.load(os.path.join(processed_dir, 'vocab.pt'), map_location='cpu')
        self.length = self.meta['len']

    def __len__(self):
        return self.length

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), map_location='cpu')


def get_split_cache_path(processed_dir):
    return os.path.join(processed_dir, 'baseline_split_cache_7class.pt')


def build_stratified_split(dataset, cache=True):
    cache_path = get_split_cache_path(dataset.processed_dir)
    if cache and os.path.exists(cache_path):
        cache_obj = torch.load(cache_path)
        return (
            cache_obj['train_idx'],
            cache_obj['val_idx'],
            cache_obj['test_idx'],
            cache_obj['train_y'],
            cache_obj['val_y'],
            cache_obj['test_y'],
        )

    all_idx, all_y = [], []
    for idx in tqdm(range(len(dataset)), desc='Scanning labels'):
        d = dataset.get(idx)
        y = int(d.mode_label.item())
        if y != -1:
            all_idx.append(idx)
            all_y.append(y)

    all_idx = np.array(all_idx)
    all_y = np.array(all_y)

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        all_idx,
        all_y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=all_y,
    )
    val_idx, test_idx, val_y, test_y = train_test_split(
        temp_idx,
        temp_y,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_y,
    )

    if cache:
        torch.save(
            {
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'train_y': train_y,
                'val_y': val_y,
                'test_y': test_y,
            },
            cache_path,
        )

    return train_idx, val_idx, test_idx, train_y, val_y, test_y


def maybe_subsample(indices, labels, max_samples=None):
    if max_samples is None or len(indices) <= max_samples:
        return indices, labels
    keep_idx, _, keep_y, _ = train_test_split(
        indices,
        labels,
        train_size=max_samples,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return keep_idx, keep_y


def extract_ordered_forward_edge_sequence(data):
    """
    Extracts ordered trajectory sequence from graph edge features.
    For each step i->i+1, use one corresponding edge feature row.
    """
    src = data.edge_index[0].cpu().numpy()
    dst = data.edge_index[1].cpu().numpy()
    ea = data.edge_attr.cpu().numpy().astype(np.float32)

    mask = (dst - src) == 1
    f_src = src[mask]
    f_ea = ea[mask]
    if len(f_src) == 0:
        return np.zeros((1, ea.shape[1]), dtype=np.float32)

    order = np.argsort(f_src)
    seq = f_ea[order]
    return seq


def load_view(dataset, indices):
    wide_list, seq_list, y_list, graph_list = [], [], [], []
    max_len = 0

    for idx in tqdm(indices, desc='Loading split', leave=False):
        d = dataset.get(int(idx))
        y = int(d.mode_label.item())
        y_list.append(y)
        wide_list.append(d.traffic.view(-1).numpy().astype(np.float32))
        seq = extract_ordered_forward_edge_sequence(d)
        seq_list.append(seq)
        max_len = max(max_len, seq.shape[0])
        graph_list.append(d)

    seq_pad = []
    for s in seq_list:
        p = np.zeros((max_len, s.shape[1]), dtype=np.float32)
        p[: min(max_len, s.shape[0])] = s[:max_len]
        seq_pad.append(p)

    return {
        'wide': np.stack(wide_list),
        'seq': np.stack(seq_pad),
        'len': np.array([min(max_len, s.shape[0]) for s in seq_list]),
        'y': np.array(y_list),
        'graphs': graph_list,
    }


class SequenceTensorDataset(Dataset):
    def __init__(self, seq, lens, y):
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.lens = torch.tensor(lens, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx], self.lens[idx], self.y[idx]


class LSTMBaseline(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, seq, lens):
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1])


class CNNBaseline(nn.Module):
    def __init__(self, in_dim, n_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, seq, lens):
        del lens
        x = seq.transpose(1, 2)
        return self.fc(self.conv(x))


class GraphListDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]


class GraphSAGEBaseline(nn.Module):
    def __init__(self, n_nodes, wide_dim, embed_dim=128, n_classes=NUM_CLASSES):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, embed_dim)
        self.conv1 = SAGEConv(embed_dim, embed_dim)
        self.conv2 = SAGEConv(embed_dim, embed_dim)
        self.conv3 = SAGEConv(embed_dim, embed_dim)
        self.wide = nn.Sequential(
            nn.Linear(wide_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, data):
        x = self.emb(data.x)
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index)) + x
        x = F.relu(self.conv3(x, data.edge_index)) + x
        z_graph = global_max_pool(x, data.batch)
        z_wide = self.wide(data.traffic.view(z_graph.size(0), -1))
        return self.head(torch.cat([z_graph, z_wide], dim=1))


def train_seq_model(model, tr, va, class_w, epochs, bs, device):
    tr_loader = DataLoader(SequenceTensorDataset(tr['seq'], tr['len'], tr['y']), batch_size=bs, shuffle=True)
    va_loader = DataLoader(SequenceTensorDataset(va['seq'], va['len'], va['y']), batch_size=bs, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_w.to(device))
    best, best_f1, bad, patience = None, -1.0, 0, 5

    for _ in range(epochs):
        model.train()
        for seq, lens, y in tr_loader:
            seq, lens, y = seq.to(device), lens.to(device), y.to(device)
            opt.zero_grad()
            logits = model(seq, lens)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        yp, yt = predict_seq(model, va_loader, device)
        val_f1 = compute_macro_metrics(yt, yp)['f1']
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best is not None:
        model.load_state_dict(best)
    return model


def predict_seq(model, loader, device):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for seq, lens, y in loader:
            seq, lens = seq.to(device), lens.to(device)
            out = model(seq, lens).argmax(dim=1).cpu().numpy()
            pred.extend(out)
            true.extend(y.numpy())
    return np.array(pred), np.array(true)


def train_gnn(model, tr_graphs, va_graphs, class_w, epochs, bs, device):
    tr_loader = PyGDataLoader(GraphListDataset(tr_graphs), batch_size=bs, shuffle=True)
    va_loader = PyGDataLoader(GraphListDataset(va_graphs), batch_size=bs, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_w.to(device))
    best, best_f1, bad, patience = None, -1.0, 0, 5

    for _ in range(epochs):
        model.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            logits = model(b)
            loss = crit(logits, b.mode_label.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        yp, yt = predict_gnn(model, va_loader, device)
        val_f1 = compute_macro_metrics(yt, yp)['f1']
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best is not None:
        model.load_state_dict(best)
    return model


def predict_gnn(model, loader, device):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b).argmax(dim=1).cpu().numpy()
            pred.extend(out)
            true.extend(b.mode_label.view(-1).cpu().numpy())
    return np.array(pred), np.array(true)


def main():
    parser = argparse.ArgumentParser(description='Run 7-class baseline comparison (RF/LSTM/CNN/GNN).')
    parser.add_argument('--processed-dir', default='./geolife_processed_7class_v2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gnn-batch-size', type=int, default=64)
    parser.add_argument('--rf-trees', type=int, default=400)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--output', default='comparison_results_7class.csv')
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = DiskDataset(args.processed_dir)
    sample = dataset.get(0)
    wide_dim = sample.traffic.view(-1).shape[0]
    seq_dim = sample.edge_attr.shape[1]

    tr_idx, va_idx, te_idx, tr_y, va_y, te_y = build_stratified_split(dataset)
    tr_idx, tr_y = maybe_subsample(tr_idx, tr_y, args.max_train_samples)

    print(f'Train/Val/Test: {len(tr_idx)}/{len(va_idx)}/{len(te_idx)}')
    print(f'Train class distribution: {dict(sorted(Counter(tr_y.tolist()).items()))}')

    tr = load_view(dataset, tr_idx)
    va = load_view(dataset, va_idx)
    te = load_view(dataset, te_idx)

    class_counts = Counter(tr['y'].tolist())
    class_w = torch.tensor(
        [len(tr['y']) / (NUM_CLASSES * max(class_counts[c], 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32,
    )

    rows = []

    # RF baseline
    rf = RandomForestClassifier(
        n_estimators=args.rf_trees,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced_subsample',
    )
    rf.fit(tr['wide'], tr['y'])
    rf_pred = rf.predict(te['wide'])
    rf_metrics = compute_macro_metrics(te['y'], rf_pred)
    rows.append({'Method': 'Random Forest', **rf_metrics})
    print('RF:', rf_metrics)

    # LSTM baseline
    lstm = LSTMBaseline(seq_dim).to(device)
    lstm = train_seq_model(lstm, tr, va, class_w, args.epochs, args.batch_size, device)
    te_loader_seq = DataLoader(SequenceTensorDataset(te['seq'], te['len'], te['y']), batch_size=args.batch_size, shuffle=False)
    yp, yt = predict_seq(lstm, te_loader_seq, device)
    lstm_metrics = compute_macro_metrics(yt, yp)
    rows.append({'Method': 'LSTM', **lstm_metrics})
    print('LSTM:', lstm_metrics)

    # CNN baseline
    cnn = CNNBaseline(seq_dim).to(device)
    cnn = train_seq_model(cnn, tr, va, class_w, args.epochs, args.batch_size, device)
    yp, yt = predict_seq(cnn, te_loader_seq, device)
    cnn_metrics = compute_macro_metrics(yt, yp)
    rows.append({'Method': 'CNN', **cnn_metrics})
    print('CNN:', cnn_metrics)

    # GNN (non-attention) baseline
    gnn = GraphSAGEBaseline(len(dataset.vocab), wide_dim).to(device)
    gnn = train_gnn(gnn, tr['graphs'], va['graphs'], class_w, args.epochs, args.gnn_batch_size, device)
    te_loader_gnn = PyGDataLoader(GraphListDataset(te['graphs']), batch_size=args.gnn_batch_size, shuffle=False)
    yp, yt = predict_gnn(gnn, te_loader_gnn, device)
    gnn_metrics = compute_macro_metrics(yt, yp)
    rows.append({'Method': 'GNN (GraphSAGE)', **gnn_metrics})
    print('GNN:', gnn_metrics)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f'\nSaved: {args.output}')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()