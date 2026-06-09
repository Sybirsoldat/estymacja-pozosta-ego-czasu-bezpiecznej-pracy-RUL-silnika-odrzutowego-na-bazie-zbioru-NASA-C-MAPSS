"""
shared.py — wspólne funkcje, modele, konfiguracja
Importowane przez 10a_train, 10b_plots, 10c_analysis, 10d_robustness
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  KONFIGURACJA                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
RUL_CLIP = 125
SEED = 42
K_FOLDS = 3
XGB_TRIALS = 50
LSTM_TRIALS = 40
ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 314, 555, 888, 1337, 2077]

TRANSFER_MAP = {
    "FD001": "FD003", "FD002": "FD004",
    "FD003": "FD001", "FD004": "FD002",
}
AUG_NOISE_STD = 0.02
AUG_COPIES = 2

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
DS_INFO = {
    "FD001": {"conditions": 1, "faults": 1},
    "FD002": {"conditions": 6, "faults": 1},
    "FD003": {"conditions": 1, "faults": 2},
    "FD004": {"conditions": 6, "faults": 2},
}
COLORS = {"XGBoost": "#E64A19", "LSTM": "#2E7D32"}

RESULTS_DIR = "./results_optuna_v3"
PLOT_DIR = "./plots_optuna_v3"
ANALYSIS_DIR = "./plots_analysis"
ROBUST_DIR = "./plots_robustness"

for d in [RESULTS_DIR, PLOT_DIR, ANALYSIS_DIR, ROBUST_DIR]:
    os.makedirs(d, exist_ok=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ZALEŻNOŚCI                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def install_if_missing(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pip_name or pkg, "-q"])

install_if_missing("xgboost")
install_if_missing("optuna")
install_if_missing("torch")

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
XGB_DEVICE = "cpu"
XGB_TREE = "hist"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  METRYKI                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return sum(
        (np.exp(-di / 13) - 1) if di < 0 else (np.exp(di / 10) - 1)
        for di in d)

def evaluate(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae_metric(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "NASA Score": nasa_score(y_true, y_pred),
    }

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PREPROCESSING                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def find_data_dir(root, dataset_id="FD001"):
    if not root or not os.path.exists(root):
        return None
    if os.path.isfile(os.path.join(root, f"train_{dataset_id}.txt")):
        return root
    for dirpath, _, filenames in os.walk(root):
        if f"train_{dataset_id}.txt" in filenames:
            return dirpath
    return None

def get_data_path():
    for cand in ["./data", "./CMAPSSData", "./CMaps",
                 os.path.expanduser("~/.cache/kagglehub")]:
        f = find_data_dir(cand)
        if f:
            return f
    try:
        install_if_missing("kagglehub")
        import kagglehub
        dl = kagglehub.dataset_download("behrad3d/nasa-cmaps")
        return find_data_dir(dl)
    except Exception as e:
        print(f"  [!] {e}")
        return None

def normalize_per_condition(train_df, test_df, sensor_cols, op_cols, n_cond):
    if n_cond <= 1:
        sc = MinMaxScaler(feature_range=(0, 1))
        train_df[sensor_cols] = sc.fit_transform(train_df[sensor_cols])
        test_df[sensor_cols] = sc.transform(test_df[sensor_cols])
        return train_df, test_df
    km = KMeans(n_clusters=n_cond, random_state=SEED, n_init=10)
    train_df["op_cl"] = km.fit_predict(train_df[op_cols].values)
    test_df["op_cl"] = km.predict(test_df[op_cols].values)
    scalers = {}
    for cid in range(n_cond):
        mask = train_df["op_cl"] == cid
        if mask.sum() == 0: continue
        sc = MinMaxScaler(feature_range=(0, 1))
        train_df.loc[mask, sensor_cols] = sc.fit_transform(
            train_df.loc[mask, sensor_cols])
        scalers[cid] = sc
    for cid in test_df["op_cl"].unique():
        mask = test_df["op_cl"] == cid
        sc = scalers.get(cid, scalers[min(scalers.keys(),
                         key=lambda c: abs(c - cid))])
        test_df.loc[mask, sensor_cols] = sc.transform(
            test_df.loc[mask, sensor_cols])
    train_df[sensor_cols] = train_df[sensor_cols].clip(0, 1)
    test_df[sensor_cols] = test_df[sensor_cols].clip(0, 1)
    train_df.drop("op_cl", axis=1, inplace=True)
    test_df.drop("op_cl", axis=1, inplace=True)
    return train_df, test_df

def create_enhanced_features(df, sensor_cols, windows=[5, 10, 20]):
    frames = []
    for uid in df["unit_id"].unique():
        u = df[df["unit_id"] == uid].sort_values("cycle").copy()
        for w in windows:
            for s in sensor_cols:
                r = u[s].rolling(w, min_periods=1)
                u[f"{s}_mean_{w}"] = r.mean()
                u[f"{s}_std_{w}"] = r.std().fillna(0)
                u[f"{s}_min_{w}"] = r.min()
                u[f"{s}_max_{w}"] = r.max()
                u[f"{s}_trend_{w}"] = u[s].diff(w).fillna(0)
                u[f"{s}_ema_{w}"] = u[s].ewm(span=max(w, 2), min_periods=1).mean()
                u[f"{s}_p10_{w}"] = r.quantile(0.1)
                u[f"{s}_p90_{w}"] = r.quantile(0.9)
                mv = u[f"{s}_mean_{w}"]
                u[f"{s}_cv_{w}"] = np.where(mv.abs() > 1e-8,
                                             u[f"{s}_std_{w}"] / mv.abs(), 0)
        sl = [s for s in sensor_cols if s.startswith("sensor_")]
        for i in range(min(len(sl), 5)):
            for j in range(i + 1, min(len(sl), 5)):
                d = u[sl[j]].abs()
                u[f"ratio_{sl[i]}_{sl[j]}"] = np.where(d > 1e-8, u[sl[i]] / d, 0)
        frames.append(u)
    return pd.concat(frames, ignore_index=True)

def build_seqs(df, feat_cols, sl):
    X, y = [], []
    for uid in df["unit_id"].unique():
        u = df[df["unit_id"] == uid].sort_values("cycle")
        d, lab = u[feat_cols].values, u["RUL"].values
        if len(d) < sl:
            pad = np.zeros((sl - len(d), len(feat_cols)))
            d = np.vstack([pad, d])
            lab = np.concatenate([np.full(sl - len(lab), lab[0]), lab])
        for i in range(len(d) - sl + 1):
            X.append(d[i:i + sl])
            y.append(lab[i + sl - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def build_test_seqs(df, feat_cols, sl):
    X, y = [], []
    for uid in df["unit_id"].unique():
        u = df[df["unit_id"] == uid].sort_values("cycle")
        d = u[feat_cols].values
        rul = u["RUL"].values[-1]
        if len(d) < sl:
            pad = np.zeros((sl - len(d), len(feat_cols)))
            d = np.vstack([pad, d])
        X.append(d[-sl:])
        y.append(rul)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def load_and_preprocess(path, dataset_id):
    train_df = pd.read_csv(os.path.join(path, f"train_{dataset_id}.txt"),
                           sep=r"\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(os.path.join(path, f"test_{dataset_id}.txt"),
                          sep=r"\s+", header=None, names=COLUMNS)
    rul_df = pd.read_csv(os.path.join(path, f"RUL_{dataset_id}.txt"),
                         sep=r"\s+", header=None, names=["RUL"])
    mc = train_df.groupby("unit_id")["cycle"].max().reset_index()
    mc.columns = ["unit_id", "max_cycle"]
    train_df = train_df.merge(mc, on="unit_id")
    train_df["RUL"] = (train_df["max_cycle"] - train_df["cycle"]).clip(upper=RUL_CLIP)
    train_df.drop("max_cycle", axis=1, inplace=True)
    mct = test_df.groupby("unit_id")["cycle"].max().reset_index()
    mct.columns = ["unit_id", "max_cycle"]
    rul_df["unit_id"] = range(1, len(rul_df) + 1)
    mct = mct.merge(rul_df, on="unit_id")
    mct["total_life"] = mct["max_cycle"] + mct["RUL"]
    test_df = test_df.merge(mct[["unit_id", "total_life"]], on="unit_id")
    test_df["RUL"] = (test_df["total_life"] - test_df["cycle"]).clip(upper=RUL_CLIP)
    test_df.drop("total_life", axis=1, inplace=True)
    sensor_all = [f"sensor_{i}" for i in range(1, 22)]
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    var = train_df[sensor_all].var()
    keep = var[var > var.median() * 0.01].index.tolist()
    info = DS_INFO[dataset_id]
    train_df, test_df = normalize_per_condition(
        train_df, test_df, keep, op_cols, info["conditions"])
    if info["conditions"] > 1:
        osc = MinMaxScaler()
        train_df[op_cols] = osc.fit_transform(train_df[op_cols])
        test_df[op_cols] = osc.transform(test_df[op_cols])
        xgb_base = keep + op_cols
    else:
        xgb_base = keep
    return train_df, test_df, xgb_base, keep, sensor_all, op_cols

def augment_sequences(X, y, noise_std=0.02, n_copies=2):
    rng = np.random.RandomState(SEED)
    X_aug, y_aug = [X], [y]
    for _ in range(n_copies):
        noise = rng.normal(0, noise_std, X.shape).astype(np.float32)
        X_noisy = np.clip(X + noise, 0, 1)
        X_jittered = X_noisy.copy()
        n_s, seq_l, n_f = X_jittered.shape
        swap_mask = rng.random((n_s, seq_l - 1)) < 0.1
        for i in range(n_s):
            for t in range(seq_l - 1):
                if swap_mask[i, t]:
                    X_jittered[i, t], X_jittered[i, t + 1] = \
                        X_jittered[i, t + 1].copy(), X_jittered[i, t].copy()
        X_aug.append(X_jittered)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

def kfold_unit_splits(train_df, k=3, seed=42):
    uids = train_df["unit_id"].unique().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(uids)
    fold_size = len(uids) // k
    folds = []
    for i in range(k):
        val_uids = uids[i * fold_size:] if i == k - 1 \
            else uids[i * fold_size:(i + 1) * fold_size]
        tr_uids = np.setdiff1d(uids, val_uids)
        folds.append((tr_uids, val_uids))
    return folds

def clean_inf(df, cols):
    for col in cols:
        arr = df[col].values
        arr[~np.isfinite(arr)] = 0
        df[col] = arr

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL LSTM                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FlexLSTM(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dense=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, dense)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)

def train_lstm_quick(X_tr, y_tr, X_val, y_val, n_features, bp,
                     epochs=50, seed=42):
    """Szybki trening LSTM z podanymi parametrami. Zwraca model na CPU."""
    torch.manual_seed(seed)
    model = FlexLSTM(n_features, bp.get("hidden", 64),
                     bp.get("n_layers", 1), bp.get("dense", 32),
                     bp.get("dropout", 0.3)).cpu()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=bp.get("lr", 0.003))
    for ep in range(epochs):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            nn.MSELoss()(model(Xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD / SAVE                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def save_results(all_results, path=None):
    p = path or os.path.join(RESULTS_DIR, "optuna_v3_results.pkl")
    with open(p, "wb") as f:
        pickle.dump(all_results, f)
    print(f"  [✓] Zapisano: {p}")

def load_results(path=None):
    p = path or os.path.join(RESULTS_DIR, "optuna_v3_results.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

def save_preprocessed(preprocessed, path=None):
    p = path or os.path.join(RESULTS_DIR, "preprocessed.pkl")
    with open(p, "wb") as f:
        pickle.dump(preprocessed, f)

def load_preprocessed(path=None):
    p = path or os.path.join(RESULTS_DIR, "preprocessed.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

def save_studies(all_studies, path=None):
    p = path or os.path.join(RESULTS_DIR, "studies.pkl")
    with open(p, "wb") as f:
        pickle.dump(all_studies, f)

def load_studies(path=None):
    p = path or os.path.join(RESULTS_DIR, "studies.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)