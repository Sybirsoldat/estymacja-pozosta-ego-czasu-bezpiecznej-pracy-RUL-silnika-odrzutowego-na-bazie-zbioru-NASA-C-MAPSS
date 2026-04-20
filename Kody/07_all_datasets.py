"""
=============================================================================
NASA C-MAPSS — Ewaluacja na wszystkich zbiorach FD001–FD004 (Krok 7)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Pełny pipeline:  preprocessing → XGBoost → LSTM → CNN-LSTM
na każdym z 4 podzbiorów C-MAPSS, z automatyczną selekcją sensorów.

FD001: 1 warunek, 1 tryb awarii   (100 silników, prosty)
FD002: 6 warunków, 1 tryb awarii  (260 silników, średni)
FD003: 1 warunek, 2 tryby awarii  (100 silników, trudny)
FD004: 6 warunków, 2 tryby awarii (248 silników, najtrudniejszy)
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
SEQUENCE_LENGTH = 30
SEED = 42

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Informacje o zbiorach
DS_INFO = {
    "FD001": {"conditions": 1, "faults": 1, "train_engines": 100},
    "FD002": {"conditions": 6, "faults": 1, "train_engines": 260},
    "FD003": {"conditions": 1, "faults": 2, "train_engines": 100},
    "FD004": {"conditions": 6, "faults": 2, "train_engines": 248},
}

PLOT_DIR = "./plots_all"
RESULTS_DIR = "./results_all"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  INSTALACJA ZALEŻNOŚCI                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "-q"])
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FUNKCJE POMOCNICZE                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def find_data_dir(root, dataset_id):
    """Szuka folderu z plikami train_FDxxx.txt (rekurencyjnie)."""
    if not root or not os.path.exists(root):
        return None
    if os.path.isfile(os.path.join(root, f"train_{dataset_id}.txt")):
        return root
    for dirpath, _, filenames in os.walk(root):
        if f"train_{dataset_id}.txt" in filenames:
            return dirpath
    return None


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = 0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13) - 1
        else:
            score += np.exp(di / 10) - 1
    return score


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

def preprocess_dataset(path, dataset_id):
    """
    Pełny preprocessing dla jednego zbioru.
    Automatyczna selekcja sensorów na podstawie wariancji.
    Zwraca dane XGBoost (2D) i sekwencyjne (3D).
    """
    print(f"\n  Preprocessing {dataset_id}...")

    # --- Wczytanie ---
    train_df = pd.read_csv(
        os.path.join(path, f"train_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=COLUMNS
    )
    test_df = pd.read_csv(
        os.path.join(path, f"test_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=COLUMNS
    )
    rul_df = pd.read_csv(
        os.path.join(path, f"RUL_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )

    # --- RUL ---
    max_cycles = train_df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="unit_id")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop("max_cycle", axis=1, inplace=True)

    max_cycles_test = test_df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles_test.columns = ["unit_id", "max_cycle"]
    rul_df["unit_id"] = range(1, len(rul_df) + 1)
    max_cycles_test = max_cycles_test.merge(rul_df, on="unit_id")
    max_cycles_test["total_life"] = max_cycles_test["max_cycle"] + max_cycles_test["RUL"]
    test_df = test_df.merge(max_cycles_test[["unit_id", "total_life"]], on="unit_id")
    test_df["RUL"] = test_df["total_life"] - test_df["cycle"]
    test_df.drop("total_life", axis=1, inplace=True)

    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CLIP)
    test_df["RUL"] = test_df["RUL"].clip(upper=RUL_CLIP)

    # --- Automatyczna selekcja sensorów ---
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]

    # Usuń sensory z zerową lub bardzo niską wariancją
    variances = train_df[sensor_cols].var()
    threshold = variances.median() * 0.01  # < 1% mediany wariancji → usunąć
    keep_sensors = variances[variances > threshold].index.tolist()

    # Op settings: zachowaj jeśli wiele warunków operacyjnych
    info = DS_INFO[dataset_id]
    if info["conditions"] > 1:
        # Dla FD002/FD004: op_settings są różnorodne, normalizuj je
        feature_cols = keep_sensors + op_cols
    else:
        # Dla FD001/FD003: op_settings stałe, pomiń
        feature_cols = keep_sensors

    dropped = set(sensor_cols) - set(keep_sensors)
    print(f"    Silniki: train={train_df['unit_id'].nunique()}, "
          f"test={test_df['unit_id'].nunique()}")
    print(f"    Warunki: {info['conditions']}, Tryby awarii: {info['faults']}")
    print(f"    Usunięte sensory ({len(dropped)}): {sorted(dropped)}")
    print(f"    Cechy wejściowe ({len(feature_cols)}): {feature_cols}")

    # --- Normalizacja ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # --- Train/Val split (po unit_id) ---
    unit_ids = train_df["unit_id"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(unit_ids)
    split_idx = int(len(unit_ids) * 0.8)
    train_units = unit_ids[:split_idx]
    val_units = unit_ids[split_idx:]

    # --- XGBoost: feature engineering ---
    def create_features(df, sensor_cols, windows=[5, 10, 20]):
        frames = []
        for uid in df["unit_id"].unique():
            unit = df[df["unit_id"] == uid].sort_values("cycle").copy()
            for w in windows:
                for s in sensor_cols:
                    unit[f"{s}_mean_{w}"] = unit[s].rolling(w, min_periods=1).mean()
                    unit[f"{s}_std_{w}"] = unit[s].rolling(w, min_periods=1).std().fillna(0)
                    unit[f"{s}_min_{w}"] = unit[s].rolling(w, min_periods=1).min()
                    unit[f"{s}_max_{w}"] = unit[s].rolling(w, min_periods=1).max()
                    unit[f"{s}_trend_{w}"] = unit[s].diff(w).fillna(0)
            frames.append(unit)
        return pd.concat(frames, ignore_index=True)

    train_xgb = create_features(train_df, feature_cols)
    test_xgb = create_features(test_df, feature_cols)

    xgb_feat_cols = [c for c in train_xgb.columns
                     if c not in ["unit_id", "cycle", "RUL"]]

    train_mask = train_xgb["unit_id"].isin(train_units)
    val_mask = train_xgb["unit_id"].isin(val_units)

    test_last = test_xgb.groupby("unit_id").last().reset_index()

    xgb_data = {
        "X_train": train_xgb[train_mask][xgb_feat_cols].values,
        "y_train": train_xgb[train_mask]["RUL"].values,
        "X_val": train_xgb[val_mask][xgb_feat_cols].values,
        "y_val": train_xgb[val_mask]["RUL"].values,
        "X_test": test_last[xgb_feat_cols].values,
        "y_test": test_last["RUL"].values,
    }

    # --- Sekwencje LSTM/CNN-LSTM ---
    def create_sequences(df, feat_cols, seq_len):
        X, y = [], []
        for uid in df["unit_id"].unique():
            unit = df[df["unit_id"] == uid].sort_values("cycle")
            data = unit[feat_cols].values
            labels = unit["RUL"].values
            if len(data) < seq_len:
                padding = np.zeros((seq_len - len(data), len(feat_cols)))
                data = np.vstack([padding, data])
                labels = np.concatenate([np.full(seq_len - len(labels), labels[0]), labels])
            for i in range(len(data) - seq_len + 1):
                X.append(data[i:i + seq_len])
                y.append(labels[i + seq_len - 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def create_test_sequences(df, feat_cols, seq_len):
        X, y = [], []
        for uid in df["unit_id"].unique():
            unit = df[df["unit_id"] == uid].sort_values("cycle")
            data = unit[feat_cols].values
            rul = unit["RUL"].values[-1]
            if len(data) < seq_len:
                padding = np.zeros((seq_len - len(data), len(feat_cols)))
                data = np.vstack([padding, data])
            X.append(data[-seq_len:])
            y.append(rul)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    train_sub = train_df[train_df["unit_id"].isin(train_units)]
    val_sub = train_df[train_df["unit_id"].isin(val_units)]

    X_tr_seq, y_tr_seq = create_sequences(train_sub, feature_cols, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(val_sub, feature_cols, SEQUENCE_LENGTH)
    X_te_seq, y_te_seq = create_test_sequences(test_df, feature_cols, SEQUENCE_LENGTH)

    seq_data = {
        "X_train": X_tr_seq, "y_train": y_tr_seq,
        "X_val": X_val_seq, "y_val": y_val_seq,
        "X_test": X_te_seq, "y_test": y_te_seq,
    }

    n_features = len(feature_cols)
    print(f"    XGBoost features: {len(xgb_feat_cols)}, "
          f"Sequences: {X_tr_seq.shape}")

    return xgb_data, seq_data, n_features


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODELE                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# --- XGBoost ---
def train_xgboost(data):
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror",
        random_state=SEED, n_jobs=-1, early_stopping_rounds=30,
    )
    t0 = time.time()
    model.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=0,
    )
    train_time = time.time() - t0

    y_pred = np.clip(model.predict(data["X_test"]), 0, RUL_CLIP)
    metrics = evaluate(data["y_test"], y_pred)
    return metrics, y_pred, train_time


# --- LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, n_features, h1=64, h2=32, dense=32, drop=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(h2, dense)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# --- CNN-LSTM ---
class CNNLSTMModel(nn.Module):
    def __init__(self, n_features, c1=32, c2=64, ks=3, h1=64, h2=32,
                 dense=32, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, c1, ks, padding=ks // 2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, ks, padding=ks // 2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(c2, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(h2, dense)
        self.fc2 = nn.Linear(dense, 1)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm1(out)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def train_pytorch_model(model_class, seq_data, n_features,
                        epochs=100, patience=15, lr=0.001, batch_size=256):
    """Trening LSTM lub CNN-LSTM z early stopping."""
    model = model_class(n_features).to(device)

    X_tr = torch.FloatTensor(seq_data["X_train"]).to(device)
    y_tr = torch.FloatTensor(seq_data["y_train"]).to(device)
    X_val = torch.FloatTensor(seq_data["X_val"]).to(device)
    y_val = torch.FloatTensor(seq_data["y_val"]).to(device)
    X_te = torch.FloatTensor(seq_data["X_test"]).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_losses.append(criterion(model(Xb), yb).item())
        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()
    y_pred = np.clip(y_pred, 0, RUL_CLIP)

    metrics = evaluate(seq_data["y_test"], y_pred)
    return metrics, y_pred, train_time


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GŁÓWNA PĘTLA — WSZYSTKIE ZBIORY                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("NASA C-MAPSS — EWALUACJA NA WSZYSTKICH ZBIORACH (FD001–FD004)")
print("=" * 70)

# Znajdź dane
data_path = None
try:
    import kagglehub
    data_path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
except Exception:
    pass

actual_path = find_data_dir(data_path, "FD001")
if actual_path is None:
    for candidate in ["./data", "./CMAPSSData", "./CMaps"]:
        found = find_data_dir(candidate, "FD001")
        if found:
            actual_path = found
            break

if actual_path is None:
    print("BŁĄD: Nie znaleziono danych C-MAPSS!")
    sys.exit(1)

print(f"[✓] Dane: {actual_path}")
print(f"[✓] Device: {device}")

# Wyniki zbiorcze
all_results = {}

for ds_id in DATASETS:
    print(f"\n{'=' * 70}")
    print(f"  ZBIÓR: {ds_id} — {DS_INFO[ds_id]['conditions']} warunków, "
          f"{DS_INFO[ds_id]['faults']} tryby awarii")
    print(f"{'=' * 70}")

    # Preprocessing
    xgb_data, seq_data, n_features = preprocess_dataset(actual_path, ds_id)

    ds_results = {"y_test": xgb_data["y_test"]}

    # --- XGBoost ---
    print(f"\n  [1/3] XGBoost...")
    metrics_xgb, pred_xgb, time_xgb = train_xgboost(xgb_data)
    ds_results["XGBoost"] = {
        "metrics": metrics_xgb, "y_pred": pred_xgb, "time": time_xgb
    }
    print(f"    RMSE={metrics_xgb['RMSE']:.2f}  "
          f"NASA={metrics_xgb['NASA Score']:.0f}  "
          f"t={time_xgb:.1f}s")

    # --- LSTM ---
    print(f"  [2/3] LSTM...")
    metrics_lstm, pred_lstm, time_lstm = train_pytorch_model(
        LSTMModel, seq_data, n_features
    )
    ds_results["LSTM"] = {
        "metrics": metrics_lstm, "y_pred": pred_lstm, "time": time_lstm
    }
    print(f"    RMSE={metrics_lstm['RMSE']:.2f}  "
          f"NASA={metrics_lstm['NASA Score']:.0f}  "
          f"t={time_lstm:.1f}s")

    # --- CNN-LSTM ---
    print(f"  [3/3] CNN-LSTM...")
    metrics_cnn, pred_cnn, time_cnn = train_pytorch_model(
        CNNLSTMModel, seq_data, n_features
    )
    ds_results["CNN-LSTM"] = {
        "metrics": metrics_cnn, "y_pred": pred_cnn, "time": time_cnn
    }
    print(f"    RMSE={metrics_cnn['RMSE']:.2f}  "
          f"NASA={metrics_cnn['NASA Score']:.0f}  "
          f"t={time_cnn:.1f}s")

    all_results[ds_id] = ds_results

# Zapis wyników
with open(os.path.join(RESULTS_DIR, "all_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)
print(f"\n[✓] Wyniki zapisane: {RESULTS_DIR}/all_results.pkl")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WIZUALIZACJE ZBIORCZE                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("WIZUALIZACJE ZBIORCZE")
print(f"{'=' * 70}")

MODEL_NAMES = ["XGBoost", "LSTM", "CNN-LSTM"]
MODEL_COLORS = {"XGBoost": "#FF5722", "LSTM": "#4CAF50", "CNN-LSTM": "#E91E63"}
DS_LABELS = {
    "FD001": "FD001\n(1 war., 1 aw.)",
    "FD002": "FD002\n(6 war., 1 aw.)",
    "FD003": "FD003\n(1 war., 2 aw.)",
    "FD004": "FD004\n(6 war., 2 aw.)",
}

# ── Wykres 1: RMSE per zbiór per model ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics_list = [
    ("RMSE", "RMSE (↓ lepiej)"),
    ("MAE", "MAE (↓ lepiej)"),
    ("R²", "R² (↑ lepiej)"),
    ("NASA Score", "NASA Score (↓ lepiej)"),
]

for idx, (metric, title) in enumerate(metrics_list):
    ax = axes[idx // 2][idx % 2]

    x = np.arange(len(DATASETS))
    width = 0.25

    for i, model in enumerate(MODEL_NAMES):
        values = [all_results[ds][model]["metrics"][metric] for ds in DATASETS]
        bars = ax.bar(x + i * width, values, width, label=model,
                      color=MODEL_COLORS[model], alpha=0.85)

        for bar, val in zip(bars, values):
            if metric == "NASA Score":
                label = f"{val:,.0f}"
            elif metric == "R²":
                label = f"{val:.3f}"
            else:
                label = f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)

plt.suptitle("Porównanie modeli na wszystkich zbiorach C-MAPSS",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/36_all_datasets_metrics.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 36_all_datasets_metrics.png")

# ── Wykres 2: Heatmap RMSE ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (metric, cmap, title) in enumerate([
    ("RMSE", "YlOrRd", "RMSE (↓ lepiej)"),
    ("NASA Score", "YlOrRd", "NASA Score (↓ lepiej)"),
]):
    ax = axes[ax_idx]
    data_matrix = np.array([
        [all_results[ds][model]["metrics"][metric]
         for ds in DATASETS]
        for model in MODEL_NAMES
    ])

    im = ax.imshow(data_matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels(MODEL_NAMES, fontsize=10)
    ax.set_title(title, fontsize=12)

    # Wartości w komórkach
    for i in range(len(MODEL_NAMES)):
        for j in range(len(DATASETS)):
            val = data_matrix[i, j]
            text = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            # Najlepszy w kolumnie — pogrubiony
            col_vals = data_matrix[:, j]
            is_best = (val == col_vals.min())
            weight = "bold" if is_best else "normal"
            color = "white" if val > np.median(data_matrix) else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, fontweight=weight, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Heatmap wyników — wszystkie zbiory C-MAPSS",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/37_all_datasets_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 37_all_datasets_heatmap.png")

# ── Wykres 3: Czas treningu ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(DATASETS))
width = 0.25

for i, model in enumerate(MODEL_NAMES):
    times = [all_results[ds][model]["time"] for ds in DATASETS]
    bars = ax.bar(x + i * width, times, width, label=model,
                  color=MODEL_COLORS[model], alpha=0.85)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.0f}s", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels([DS_LABELS[ds] for ds in DATASETS])
ax.set_ylabel("Czas treningu (s)")
ax.set_title("Czas treningu per zbiór per model", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/38_all_datasets_time.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 38_all_datasets_time.png")

# ── Wykres 4: Predicted vs Actual — 4×3 grid ────────────────────────────────
fig, axes = plt.subplots(len(DATASETS), len(MODEL_NAMES),
                         figsize=(5 * len(MODEL_NAMES), 5 * len(DATASETS)))

for row, ds in enumerate(DATASETS):
    y_true = all_results[ds]["y_test"]
    for col, model in enumerate(MODEL_NAMES):
        ax = axes[row][col]
        y_pred = all_results[ds][model]["y_pred"]
        met = all_results[ds][model]["metrics"]

        ax.scatter(y_true, y_pred, alpha=0.5, s=20,
                   c=MODEL_COLORS[model], edgecolors="none")
        ax.plot([0, RUL_CLIP], [0, RUL_CLIP], "k--", lw=1, alpha=0.4)
        ax.fill_between([0, RUL_CLIP], [-15, RUL_CLIP - 15],
                        [15, RUL_CLIP + 15], alpha=0.08, color="green")

        ax.set_xlim(-5, RUL_CLIP + 5)
        ax.set_ylim(-5, RUL_CLIP + 5)
        ax.set_aspect("equal")

        ax.set_title(f"{ds} — {model}\n"
                     f"RMSE={met['RMSE']:.1f}  NASA={met['NASA Score']:,.0f}",
                     fontsize=9)

        if col == 0:
            ax.set_ylabel("Pred RUL")
        if row == len(DATASETS) - 1:
            ax.set_xlabel("Actual RUL")

plt.suptitle("Predicted vs Actual — wszystkie zbiory × modele",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/39_all_datasets_scatter.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 39_all_datasets_scatter.png")

# ── Wykres 5: Najlepszy model per zbiór ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

best_per_ds = {}
for ds in DATASETS:
    scores = {m: all_results[ds][m]["metrics"]["RMSE"] for m in MODEL_NAMES}
    best = min(scores, key=scores.get)
    best_per_ds[ds] = best

# Tabela jako tekst
cell_text = []
for ds in DATASETS:
    row = []
    for model in MODEL_NAMES:
        met = all_results[ds][model]["metrics"]
        txt = f"RMSE={met['RMSE']:.1f}\nNASA={met['NASA Score']:,.0f}"
        row.append(txt)
    cell_text.append(row)

ax.axis("off")
table = ax.table(
    cellText=cell_text,
    rowLabels=[f"{ds}\n({DS_INFO[ds]['conditions']} war., "
               f"{DS_INFO[ds]['faults']} aw.)" for ds in DATASETS],
    colLabels=MODEL_NAMES,
    cellLoc="center", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.5)

# Koloruj nagłówki
for j in range(len(MODEL_NAMES)):
    table[0, j].set_facecolor(MODEL_COLORS[MODEL_NAMES[j]])
    table[0, j].set_text_props(color="white", fontweight="bold")

# Koloruj najlepsze komórki
for i, ds in enumerate(DATASETS):
    best_model = best_per_ds[ds]
    j = MODEL_NAMES.index(best_model)
    table[i + 1, j].set_facecolor("#C8E6C9")  # zielone tło
    table[i + 1, j].set_text_props(fontweight="bold")

ax.set_title("Podsumowanie — najlepszy model per zbiór (★ = najniższy RMSE)",
             fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/40_all_datasets_summary.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 40_all_datasets_summary.png")

# ── Wykres 6: Wzrost trudności — RMSE vs złożoność zbioru ────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

complexity = {"FD001": 1, "FD002": 2, "FD003": 3, "FD004": 4}
x_labels = ["FD001\n(prosty)", "FD002\n(średni)", "FD003\n(trudny)",
            "FD004\n(najtrudniejszy)"]

for model in MODEL_NAMES:
    rmse_values = [all_results[ds][model]["metrics"]["RMSE"] for ds in DATASETS]
    ax.plot(range(4), rmse_values, "o-", label=model, lw=2, markersize=8,
            color=MODEL_COLORS[model])

ax.set_xticks(range(4))
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel("RMSE (test)", fontsize=12)
ax.set_title("Jak złożoność zbioru wpływa na RMSE?", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/41_complexity_vs_rmse.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 41_complexity_vs_rmse.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE KOŃCOWE                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PODSUMOWANIE KOŃCOWE — WSZYSTKIE ZBIORY")
print(f"{'=' * 70}\n")

# Tabela w konsoli
header = f"  {'Zbiór':<8} │ {'Warunki':>7} │"
for m in MODEL_NAMES:
    header += f" {m:>20} │"
print(header)
print("  " + "─" * 8 + "─┼" + "─" * 9 + "┼" + ("─" * 22 + "┼") * 3)

for ds in DATASETS:
    info = DS_INFO[ds]
    row = f"  {ds:<8} │ {info['conditions']}w, {info['faults']}f  │"
    rmses = {m: all_results[ds][m]["metrics"]["RMSE"] for m in MODEL_NAMES}
    best_m = min(rmses, key=rmses.get)
    for m in MODEL_NAMES:
        met = all_results[ds][m]["metrics"]
        star = "★" if m == best_m else " "
        row += f" {star} RMSE={met['RMSE']:>5.1f} N={met['NASA Score']:>5.0f} │"
    print(row)

print(f"\n  Najlepszy model per zbiór (wg RMSE):")
for ds in DATASETS:
    rmses = {m: all_results[ds][m]["metrics"]["RMSE"] for m in MODEL_NAMES}
    best = min(rmses, key=rmses.get)
    print(f"    {ds}: {best} (RMSE={rmses[best]:.2f})")

total_time = sum(
    all_results[ds][m]["time"]
    for ds in DATASETS for m in MODEL_NAMES
)
print(f"\n  Łączny czas treningu: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Wykresy: {PLOT_DIR}/36–41")
print(f"  Wyniki:  {RESULTS_DIR}/all_results.pkl")
print()