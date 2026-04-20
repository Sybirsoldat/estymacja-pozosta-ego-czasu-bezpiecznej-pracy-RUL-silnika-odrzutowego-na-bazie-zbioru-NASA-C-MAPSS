"""
=============================================================================
NASA C-MAPSS — Optymalizacja hiperparametrów XGBoost + LSTM (Krok 9)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Optuna stroi hiperparametry obu modeli na każdym zbiorze FD001–FD004:

  XGBoost (Optuna 25 prób):
    ✦ max_depth, learning_rate, subsample, colsample_bytree
    ✦ min_child_weight, reg_alpha, reg_lambda, gamma
    ✦ Rozszerzone cechy (EMA, percentyle, ratio)

  LSTM (Optuna 20 prób):
    ✦ Liczba warstw (1–3), rozmiar hidden (32–256)
    ✦ Dropout (0.1–0.5), learning rate (1e-4–1e-2)
    ✦ Batch size (64–512), sequence length (20–50)
    ✦ Huber Loss (delta stroiwany) vs MSE
    ✦ Ensemble: 3 modele z różnymi seedami → uśrednienie

  Preprocessing:
    ✦ Normalizacja per warunek operacyjny (KMeans) — z kroku 8
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
XGB_OPTUNA_TRIALS = 25
LSTM_OPTUNA_TRIALS = 20
ENSEMBLE_SEEDS = [42, 123, 7]  # 3 seedy do ensemble

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

PLOT_DIR = "./plots_optuna"
RESULTS_DIR = "./results_optuna"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ZALEŻNOŚCI                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            pip_name or package, "-q"
        ])

install_if_missing("xgboost")
install_if_missing("optuna")
install_if_missing("torch")
install_if_missing("kagglehub")

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    XGB_DEVICE, XGB_TREE = "cuda", "hist"
else:
    XGB_DEVICE, XGB_TREE = "cpu", "hist"


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
    score = 0
    for di in d:
        score += (np.exp(-di / 13) - 1) if di < 0 else (np.exp(di / 10) - 1)
    return score

def evaluate(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae_metric(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "NASA Score": nasa_score(y_true, y_pred),
    }

def find_data_dir(root, dataset_id):
    if not root or not os.path.exists(root):
        return None
    if os.path.isfile(os.path.join(root, f"train_{dataset_id}.txt")):
        return root
    for dirpath, _, filenames in os.walk(root):
        if f"train_{dataset_id}.txt" in filenames:
            return dirpath
    return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PREPROCESSING (z kroku 8 — normalizacja per warunek operacyjny)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def normalize_per_condition(train_df, test_df, sensor_cols, op_cols, n_cond):
    """Normalizacja per klaster operacyjny (KMeans) — kluczowe dla FD002/FD004."""
    if n_cond <= 1:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
        test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
        return train_df, test_df

    kmeans = KMeans(n_clusters=n_cond, random_state=SEED, n_init=10)
    train_df["op_cluster"] = kmeans.fit_predict(train_df[op_cols].values)
    test_df["op_cluster"] = kmeans.predict(test_df[op_cols].values)

    scalers = {}
    for cid in range(n_cond):
        mask = train_df["op_cluster"] == cid
        if mask.sum() == 0:
            continue
        sc = MinMaxScaler(feature_range=(0, 1))
        train_df.loc[mask, sensor_cols] = sc.fit_transform(
            train_df.loc[mask, sensor_cols])
        scalers[cid] = sc

    for cid in test_df["op_cluster"].unique():
        mask = test_df["op_cluster"] == cid
        sc = scalers.get(cid, scalers[min(scalers.keys(),
                         key=lambda c: abs(c - cid))])
        test_df.loc[mask, sensor_cols] = sc.transform(
            test_df.loc[mask, sensor_cols])

    train_df[sensor_cols] = train_df[sensor_cols].clip(0, 1)
    test_df[sensor_cols] = test_df[sensor_cols].clip(0, 1)
    train_df.drop("op_cluster", axis=1, inplace=True)
    test_df.drop("op_cluster", axis=1, inplace=True)
    return train_df, test_df


def create_enhanced_features(df, sensor_cols, windows=[5, 10, 20]):
    """Rozszerzone cechy XGBoost: EMA, percentyle, CV, ratio."""
    frames = []
    for uid in df["unit_id"].unique():
        unit = df[df["unit_id"] == uid].sort_values("cycle").copy()
        for w in windows:
            for s in sensor_cols:
                rolling = unit[s].rolling(w, min_periods=1)
                unit[f"{s}_mean_{w}"] = rolling.mean()
                unit[f"{s}_std_{w}"] = rolling.std().fillna(0)
                unit[f"{s}_min_{w}"] = rolling.min()
                unit[f"{s}_max_{w}"] = rolling.max()
                unit[f"{s}_trend_{w}"] = unit[s].diff(w).fillna(0)
                unit[f"{s}_ema_{w}"] = unit[s].ewm(
                    span=max(w, 2), min_periods=1).mean()
                unit[f"{s}_p10_{w}"] = rolling.quantile(0.1)
                unit[f"{s}_p90_{w}"] = rolling.quantile(0.9)
                mean_v = unit[f"{s}_mean_{w}"]
                unit[f"{s}_cv_{w}"] = np.where(
                    mean_v.abs() > 1e-8,
                    unit[f"{s}_std_{w}"] / mean_v.abs(), 0)
        # Ratio między top sensorami
        slist = [s for s in sensor_cols if s.startswith("sensor_")]
        for i in range(min(len(slist), 5)):
            for j in range(i + 1, min(len(slist), 5)):
                denom = unit[slist[j]].abs()
                unit[f"ratio_{slist[i]}_{slist[j]}"] = np.where(
                    denom > 1e-8, unit[slist[i]] / denom, 0)
        frames.append(unit)
    return pd.concat(frames, ignore_index=True)


def preprocess_dataset(path, dataset_id, seq_length=30):
    """Pełny preprocessing: RUL, selekcja, normalizacja per warunek, cechy."""
    train_df = pd.read_csv(
        os.path.join(path, f"train_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(
        os.path.join(path, f"test_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=COLUMNS)
    rul_df = pd.read_csv(
        os.path.join(path, f"RUL_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=["RUL"])

    # RUL
    mc = train_df.groupby("unit_id")["cycle"].max().reset_index()
    mc.columns = ["unit_id", "max_cycle"]
    train_df = train_df.merge(mc, on="unit_id")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop("max_cycle", axis=1, inplace=True)

    mct = test_df.groupby("unit_id")["cycle"].max().reset_index()
    mct.columns = ["unit_id", "max_cycle"]
    rul_df["unit_id"] = range(1, len(rul_df) + 1)
    mct = mct.merge(rul_df, on="unit_id")
    mct["total_life"] = mct["max_cycle"] + mct["RUL"]
    test_df = test_df.merge(mct[["unit_id", "total_life"]], on="unit_id")
    test_df["RUL"] = test_df["total_life"] - test_df["cycle"]
    test_df.drop("total_life", axis=1, inplace=True)

    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CLIP)
    test_df["RUL"] = test_df["RUL"].clip(upper=RUL_CLIP)

    # Selekcja sensorów
    sensor_all = [f"sensor_{i}" for i in range(1, 22)]
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    var = train_df[sensor_all].var()
    keep = var[var > var.median() * 0.01].index.tolist()
    info = DS_INFO[dataset_id]

    # Normalizacja per warunek
    train_df, test_df = normalize_per_condition(
        train_df, test_df, keep, op_cols, info["conditions"])

    seq_feat = keep
    xgb_base = keep + op_cols if info["conditions"] > 1 else keep
    if info["conditions"] > 1:
        osc = MinMaxScaler()
        train_df[op_cols] = osc.fit_transform(train_df[op_cols])
        test_df[op_cols] = osc.transform(test_df[op_cols])

    # Train/Val split
    uids = train_df["unit_id"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(uids)
    split = int(len(uids) * 0.8)
    tr_units, val_units = uids[:split], uids[split:]

    # XGBoost features
    tr_xgb = create_enhanced_features(train_df, xgb_base)
    te_xgb = create_enhanced_features(test_df, xgb_base)
    feat_cols = [c for c in tr_xgb.columns
                 if c not in ["unit_id", "cycle", "RUL"] + op_cols + sensor_all]
    tr_xgb[feat_cols] = tr_xgb[feat_cols].replace(
        [np.inf, -np.inf], 0).fillna(0)
    te_xgb[feat_cols] = te_xgb[feat_cols].replace(
        [np.inf, -np.inf], 0).fillna(0)

    tr_m = tr_xgb["unit_id"].isin(tr_units)
    val_m = tr_xgb["unit_id"].isin(val_units)
    te_last = te_xgb.groupby("unit_id").last().reset_index()

    xgb_data = {
        "X_train": tr_xgb[tr_m][feat_cols].values.astype(np.float32),
        "y_train": tr_xgb[tr_m]["RUL"].values.astype(np.float32),
        "X_val": tr_xgb[val_m][feat_cols].values.astype(np.float32),
        "y_val": tr_xgb[val_m]["RUL"].values.astype(np.float32),
        "X_test": te_last[feat_cols].values.astype(np.float32),
        "y_test": te_last["RUL"].values.astype(np.float32),
    }

    # Sekwencje — budujemy dla różnych seq_length
    def build_seqs(df, feat_cols, sl):
        X, y = [], []
        for uid in df["unit_id"].unique():
            u = df[df["unit_id"] == uid].sort_values("cycle")
            d, l = u[feat_cols].values, u["RUL"].values
            if len(d) < sl:
                pad = np.zeros((sl - len(d), len(feat_cols)))
                d = np.vstack([pad, d])
                l = np.concatenate([np.full(sl - len(l), l[0]), l])
            for i in range(len(d) - sl + 1):
                X.append(d[i:i + sl])
                y.append(l[i + sl - 1])
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

    # Przechowaj dane surowe do budowy sekwencji z różnym seq_length
    raw_data = {
        "train_sub": train_df[train_df["unit_id"].isin(tr_units)],
        "val_sub": train_df[train_df["unit_id"].isin(val_units)],
        "test_df": test_df,
        "seq_feat": seq_feat,
        "build_seqs": build_seqs,
        "build_test_seqs": build_test_seqs,
    }

    n_features = len(seq_feat)
    print(f"    Sensory: {n_features}, XGBoost feat: {len(feat_cols)}")
    return xgb_data, raw_data, n_features


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  XGBoost + OPTUNA                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def optimize_xgboost(data, n_trials=25):
    """XGBoost z Optuna — automatyczny dobór hiperparametrów."""

    def objective(trial):
        p = {
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_w", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "objective": "reg:squarederror",
            "tree_method": XGB_TREE, "device": XGB_DEVICE,
            "random_state": SEED, "n_jobs": -1,
            "early_stopping_rounds": 30,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(data["X_train"], data["y_train"],
              eval_set=[(data["X_val"], data["y_val"])], verbose=0)
        pred = np.clip(m.predict(data["X_val"]), 0, RUL_CLIP)
        return rmse(data["y_val"], pred)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Trenuj finalny model
    bp = study.best_params
    final_p = {
        "n_estimators": 1000,
        "max_depth": bp["max_depth"],
        "learning_rate": bp["lr"],
        "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample"],
        "min_child_weight": bp["min_child_w"],
        "reg_alpha": bp["reg_alpha"],
        "reg_lambda": bp["reg_lambda"],
        "gamma": bp["gamma"],
        "objective": "reg:squarederror",
        "tree_method": XGB_TREE, "device": XGB_DEVICE,
        "random_state": SEED, "n_jobs": -1,
        "early_stopping_rounds": 30,
    }
    t0 = time.time()
    model = xgb.XGBRegressor(**final_p)
    model.fit(data["X_train"], data["y_train"],
              eval_set=[(data["X_val"], data["y_val"])], verbose=0)
    train_time = time.time() - t0

    y_pred = np.clip(model.predict(data["X_test"]), 0, RUL_CLIP)
    metrics = evaluate(data["y_test"], y_pred)
    return metrics, y_pred, train_time, bp


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LSTM — czysty, parametryzowalny                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FlexLSTM(nn.Module):
    """
    Czysty LSTM z konfigurowalnymi hiperparametrami.
    Parametry strojone przez Optuna:
      - n_layers: 1–3
      - hidden_size: 32–256
      - dense_size: 16–128
      - dropout: 0.1–0.5
    """
    def __init__(self, n_features, hidden_size=64, n_layers=2,
                 dense_size=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # [batch, seq, hidden]
        out = out[:, -1, :]            # ostatni timestep
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def train_lstm_single(seq_data, n_features, hidden_size, n_layers,
                      dense_size, dropout, lr, batch_size, use_huber,
                      huber_delta, epochs=100, patience=15, seed=42):
    """Trenuje jeden model LSTM z podanymi hiperparametrami."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FlexLSTM(n_features, hidden_size, n_layers,
                     dense_size, dropout).to(device)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    if use_huber:
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl.append(nn.MSELoss()(model(Xb), yb).item())
        avg = np.mean(vl)

        if avg < best_val:
            best_val = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()
    return np.clip(y_pred, 0, RUL_CLIP), best_val


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LSTM + OPTUNA                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def optimize_lstm(raw_data, n_features, n_trials=20):
    """
    Optuna stroi hiperparametry czystego LSTM:
    - hidden_size, n_layers, dense_size, dropout
    - learning_rate, batch_size, sequence_length
    - loss function: MSE vs Huber (z delta)
    Po znalezieniu najlepszych parametrów: ensemble 3 modeli.
    """
    build_seqs = raw_data["build_seqs"]
    build_test_seqs = raw_data["build_test_seqs"]
    seq_feat = raw_data["seq_feat"]
    train_sub = raw_data["train_sub"]
    val_sub = raw_data["val_sub"]
    test_df = raw_data["test_df"]

    # Cache sekwencji dla różnych seq_length
    seq_cache = {}

    def get_seq_data(sl):
        if sl not in seq_cache:
            Xtr, ytr = build_seqs(train_sub, seq_feat, sl)
            Xv, yv = build_seqs(val_sub, seq_feat, sl)
            Xte, yte = build_test_seqs(test_df, seq_feat, sl)
            seq_cache[sl] = {
                "X_train": Xtr, "y_train": ytr,
                "X_val": Xv, "y_val": yv,
                "X_test": Xte, "y_test": yte,
            }
        return seq_cache[sl]

    def objective(trial):
        hidden = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        layers = trial.suggest_int("n_layers", 1, 3)
        dense = trial.suggest_categorical("dense_size", [16, 32, 64, 128])
        drop = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        sl = trial.suggest_categorical("seq_length", [20, 30, 40, 50])
        use_huber = trial.suggest_categorical("use_huber", [True, False])
        huber_delta = trial.suggest_float("huber_delta", 1.0, 20.0) \
            if use_huber else 1.0

        sd = get_seq_data(sl)
        _, val_loss = train_lstm_single(
            sd, n_features, hidden, layers, dense, drop,
            lr, bs, use_huber, huber_delta,
            epochs=60, patience=10, seed=SEED
        )
        return val_loss

    print(f"    Optuna LSTM: {n_trials} prób...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best_sl = bp["seq_length"]
    best_sd = get_seq_data(best_sl)

    loss_str = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) \
        if bp["use_huber"] else "MSE"
    print(f"    Best: hidden={bp['hidden_size']}, layers={bp['n_layers']}, "
          f"dense={bp['dense_size']}, drop={bp['dropout']:.2f}")
    print(f"           lr={bp['lr']:.5f}, batch={bp['batch_size']}, "
          f"seq_len={bp['seq_length']}, loss={loss_str}")

    # ── Ensemble: trenuj 3 modele z różnymi seedami, uśrednij ────────────
    print(f"    Ensemble: trenowanie {len(ENSEMBLE_SEEDS)} modeli...")
    t0 = time.time()
    preds = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        pred, _ = train_lstm_single(
            best_sd, n_features,
            bp["hidden_size"], bp["n_layers"], bp["dense_size"],
            bp["dropout"], bp["lr"], bp["batch_size"],
            bp["use_huber"], bp.get("huber_delta", 1.0),
            epochs=100, patience=15, seed=seed
        )
        preds.append(pred)
        print(f"      Model {i + 1}/{len(ENSEMBLE_SEEDS)}: "
              f"RMSE={rmse(best_sd['y_test'], pred):.2f}")

    train_time = time.time() - t0

    # Uśrednij predykcje ensemble
    y_pred_ensemble = np.mean(preds, axis=0)
    y_pred_ensemble = np.clip(y_pred_ensemble, 0, RUL_CLIP)
    metrics = evaluate(best_sd["y_test"], y_pred_ensemble)

    # Dla porównania — metryki pojedynczego najlepszego modelu
    single_metrics = [evaluate(best_sd["y_test"], p) for p in preds]
    best_single_rmse = min(m["RMSE"] for m in single_metrics)

    print(f"    Ensemble RMSE={metrics['RMSE']:.2f} "
          f"(best single={best_single_rmse:.2f})")

    total_params = sum(p.numel() for p in FlexLSTM(
        n_features, bp["hidden_size"], bp["n_layers"],
        bp["dense_size"], bp["dropout"]).parameters())

    return metrics, y_pred_ensemble, train_time, bp, total_params


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ZNAJDŹ / POBIERZ DANE                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("NASA C-MAPSS — OPTYMALIZACJA HIPERPARAMETRÓW (Krok 9)")
print("=" * 70)
print(f"  XGBoost: Optuna ({XGB_OPTUNA_TRIALS} prób)")
print(f"  LSTM:    Optuna ({LSTM_OPTUNA_TRIALS} prób) + Ensemble ({len(ENSEMBLE_SEEDS)} modele)")
print(f"  Device:  {device}" + (f" ({gpu_name})" if USE_GPU else ""))

data_path = None
for candidate in ["./data", "./CMAPSSData", "./CMaps",
                   os.path.expanduser("~/.cache/kagglehub")]:
    found = find_data_dir(candidate, "FD001")
    if found:
        data_path = found
        break

if data_path is None:
    print("\n  [i] Pobieram dane z Kaggle...")
    try:
        import kagglehub
        dl = kagglehub.dataset_download("behrad3d/nasa-cmaps")
        data_path = find_data_dir(dl, "FD001")
    except Exception as e:
        print(f"  [!] Błąd: {e}")
        print(f"  [!] Pobierz ręcznie: kaggle.com/datasets/behrad3d/nasa-cmaps")
        sys.exit(1)

if data_path is None:
    print("BŁĄD: Nie znaleziono plików C-MAPSS!")
    sys.exit(1)

print(f"  Dane:    {data_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GŁÓWNA PĘTLA                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

all_results = {}

for ds_id in DATASETS:
    print(f"\n{'=' * 70}")
    print(f"  {ds_id} — {DS_INFO[ds_id]['conditions']} warunków, "
          f"{DS_INFO[ds_id]['faults']} tryby awarii")
    print(f"{'=' * 70}")

    xgb_data, raw_data, n_feat = preprocess_dataset(data_path, ds_id)
    ds_res = {"y_test": xgb_data["y_test"]}

    # --- XGBoost ---
    print(f"\n  [1/2] XGBoost + Optuna ({XGB_OPTUNA_TRIALS} prób)...")
    m_xgb, p_xgb, t_xgb, bp_xgb = optimize_xgboost(
        xgb_data, n_trials=XGB_OPTUNA_TRIALS)
    ds_res["XGBoost"] = {
        "metrics": m_xgb, "y_pred": p_xgb,
        "time": t_xgb, "best_params": bp_xgb,
    }
    print(f"    ★ RMSE={m_xgb['RMSE']:.2f}  NASA={m_xgb['NASA Score']:.0f}  "
          f"t={t_xgb:.1f}s")

    # --- LSTM ---
    print(f"\n  [2/2] LSTM + Optuna ({LSTM_OPTUNA_TRIALS} prób) + Ensemble...")
    m_lstm, p_lstm, t_lstm, bp_lstm, n_params = optimize_lstm(
        raw_data, n_feat, n_trials=LSTM_OPTUNA_TRIALS)
    ds_res["LSTM"] = {
        "metrics": m_lstm, "y_pred": p_lstm,
        "time": t_lstm, "best_params": bp_lstm,
        "n_params": n_params,
    }
    print(f"    ★ RMSE={m_lstm['RMSE']:.2f}  NASA={m_lstm['NASA Score']:.0f}  "
          f"t={t_lstm:.1f}s  params={n_params:,}")

    all_results[ds_id] = ds_res

# Zapis
with open(os.path.join(RESULTS_DIR, "optuna_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WCZYTANIE WYNIKÓW v1 DO PORÓWNANIA                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

v1 = None
for p in ["./results_all/all_results.pkl", "../results_all/all_results.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            v1 = pickle.load(f)
        print(f"\n[✓] Wyniki v1 załadowane: {p}")
        break


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WIZUALIZACJE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("WIZUALIZACJE")
print(f"{'=' * 70}")

C = {"XGBoost v1": "#FFAB91", "LSTM v1": "#A5D6A7",
     "XGBoost": "#E64A19", "LSTM": "#2E7D32"}

# ── 47: Porównanie RMSE & NASA v1 vs Optuna ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ai, (metric, title) in enumerate([
    ("RMSE", "RMSE (↓ lepiej)"), ("NASA Score", "NASA Score (↓ lepiej)")
]):
    ax = axes[ai]
    x = np.arange(len(DATASETS))
    w = 0.18
    off = 0
    if v1:
        for mv1 in ["XGBoost", "LSTM"]:
            vals = [v1[ds][mv1]["metrics"][metric] for ds in DATASETS]
            ax.bar(x + off * w, vals, w, label=f"{mv1} v1",
                   color=C[f"{mv1} v1"], alpha=0.7, edgecolor="white")
            off += 1
    for m9 in ["XGBoost", "LSTM"]:
        vals = [all_results[ds][m9]["metrics"][metric] for ds in DATASETS]
        bars = ax.bar(x + off * w, vals, w, label=f"{m9} Optuna",
                      color=C[m9], alpha=0.9, edgecolor="white")
        for xi, val in zip(x + off * w, vals):
            fmt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            ax.text(xi, val, fmt, ha="center", va="bottom",
                    fontsize=7, rotation=45)
        off += 1
    ax.set_xticks(x + w * (off - 1) / 2)
    ax.set_xticklabels(DATASETS)
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.suptitle("v1 vs Optuna — XGBoost & LSTM",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/47_optuna_vs_v1.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 47_optuna_vs_v1.png")

# ── 48: Heatmap Optuna ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
models_9 = ["XGBoost", "LSTM"]
for ai, (metric, title) in enumerate([
    ("RMSE", "RMSE Optuna (↓)"), ("NASA Score", "NASA Score Optuna (↓)")
]):
    ax = axes[ai]
    mat = np.array([[all_results[ds][m]["metrics"][metric]
                     for ds in DATASETS] for m in models_9])
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(len(models_9)))
    ax.set_yticklabels(["XGBoost Optuna", "LSTM Optuna"])
    ax.set_title(title)
    for i in range(len(models_9)):
        for j in range(len(DATASETS)):
            val = mat[i, j]
            txt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            best = val == mat[:, j].min()
            color = "white" if val > np.median(mat) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=12, fontweight="bold" if best else "normal",
                    color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Heatmap — modele po Optuna", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/48_optuna_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 48_optuna_heatmap.png")

# ── 49: Scatter 4×2 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(DATASETS), 2,
                         figsize=(10, 5 * len(DATASETS)))
for row, ds in enumerate(DATASETS):
    yt = all_results[ds]["y_test"]
    for col, (m, color) in enumerate([("XGBoost", "#E64A19"),
                                       ("LSTM", "#2E7D32")]):
        ax = axes[row][col]
        yp = all_results[ds][m]["y_pred"]
        met = all_results[ds][m]["metrics"]
        ax.scatter(yt, yp, alpha=0.5, s=20, c=color, edgecolors="none")
        ax.plot([0, RUL_CLIP], [0, RUL_CLIP], "k--", lw=1, alpha=0.4)
        ax.fill_between([0, RUL_CLIP], [-15, RUL_CLIP - 15],
                        [15, RUL_CLIP + 15], alpha=0.08, color="green")
        ax.set_xlim(-5, RUL_CLIP + 5)
        ax.set_ylim(-5, RUL_CLIP + 5)
        ax.set_aspect("equal")
        ax.set_title(f"{ds} — {m} Optuna\n"
                     f"RMSE={met['RMSE']:.1f}  NASA={met['NASA Score']:,.0f}",
                     fontsize=10)
        if col == 0: ax.set_ylabel("Pred RUL")
        if row == len(DATASETS) - 1: ax.set_xlabel("Actual RUL")

plt.suptitle("Predicted vs Actual — Optuna", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/49_optuna_scatter.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 49_optuna_scatter.png")

# ── 50: Poprawa v1→Optuna (%) ────────────────────────────────────────────────
if v1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ai, (metric, title) in enumerate([
        ("RMSE", "Zmiana RMSE (v1→Optuna)"),
        ("NASA Score", "Zmiana NASA Score (v1→Optuna)")
    ]):
        ax = axes[ai]
        x = np.arange(len(DATASETS))
        w = 0.35
        for mi, m in enumerate(["XGBoost", "LSTM"]):
            v1_vals = [v1[ds][m]["metrics"][metric] for ds in DATASETS]
            v9_vals = [all_results[ds][m]["metrics"][metric] for ds in DATASETS]
            delta = [(v9 - v1v) / v1v * 100
                     for v1v, v9 in zip(v1_vals, v9_vals)]
            bars = ax.bar(x + mi * w - w / 2, delta, w,
                          label=m, color=C[m], alpha=0.85)
            for bar in bars:
                val = bar.get_height()
                clr = "#2E7D32" if val < 0 else "#F44336"
                ax.text(bar.get_x() + bar.get_width() / 2, val,
                        f"{val:+.1f}%", ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=9, color=clr, fontweight="bold")
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel("Zmiana (%)")
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Poprawa po Optuna (ujemne = lepiej)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/50_optuna_improvement.png", bbox_inches="tight")
    plt.close()
    print(f"  [✓] 50_optuna_improvement.png")

# ── 51: Najlepsze hiperparametry LSTM per zbiór ──────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

cols = ["Zbiór", "hidden", "layers", "dense", "dropout", "lr",
        "batch", "seq_len", "loss", "RMSE", "NASA"]
rows = []
for ds in DATASETS:
    bp = all_results[ds]["LSTM"]["best_params"]
    met = all_results[ds]["LSTM"]["metrics"]
    loss_str = f"Huber(δ={bp.get('huber_delta', 1):.1f})" \
        if bp["use_huber"] else "MSE"
    rows.append([
        ds, bp["hidden_size"], bp["n_layers"], bp["dense_size"],
        f"{bp['dropout']:.2f}", f"{bp['lr']:.5f}",
        bp["batch_size"], bp["seq_length"], loss_str,
        f"{met['RMSE']:.1f}", f"{met['NASA Score']:,.0f}"
    ])

table = ax.table(cellText=rows, colLabels=cols,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 2.0)
for j in range(len(cols)):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Najlepsze hiperparametry LSTM per zbiór (Optuna)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/51_lstm_best_params.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 51_lstm_best_params.png")

# ── 52: Tabela zbiorcza ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis("off")

col_labels = ["Zbiór"]
if v1: col_labels += ["XGBoost v1", "LSTM v1"]
col_labels += ["XGBoost Optuna", "LSTM Optuna"]

row_data = []
for ds in DATASETS:
    row = [f"{ds}\n({DS_INFO[ds]['conditions']}w, {DS_INFO[ds]['faults']}f)"]
    if v1:
        for m in ["XGBoost", "LSTM"]:
            met = v1[ds][m]["metrics"]
            row.append(f"RMSE={met['RMSE']:.1f}\nNASA={met['NASA Score']:,.0f}")
    for m in ["XGBoost", "LSTM"]:
        met = all_results[ds][m]["metrics"]
        row.append(f"RMSE={met['RMSE']:.1f}\nNASA={met['NASA Score']:,.0f}")
    row_data.append(row)

table = ax.table(cellText=row_data, colLabels=col_labels,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 2.8)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Podświetl najlepszy per wiersz
for i, ds in enumerate(DATASETS):
    rmses = {}
    if v1:
        rmses["XGBoost v1"] = v1[ds]["XGBoost"]["metrics"]["RMSE"]
        rmses["LSTM v1"] = v1[ds]["LSTM"]["metrics"]["RMSE"]
    rmses["XGBoost Optuna"] = all_results[ds]["XGBoost"]["metrics"]["RMSE"]
    rmses["LSTM Optuna"] = all_results[ds]["LSTM"]["metrics"]["RMSE"]
    best = min(rmses, key=rmses.get)
    if best in col_labels:
        table[i + 1, col_labels.index(best)].set_facecolor("#C8E6C9")
        table[i + 1, col_labels.index(best)].set_text_props(fontweight="bold")

ax.set_title("Porównanie v1 vs Optuna — pełna tabela",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/52_optuna_full_table.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 52_optuna_full_table.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PODSUMOWANIE — Optuna (Krok 9)")
print(f"{'=' * 70}\n")

header = f"  {'Zbiór':<8} │ {'XGBoost Optuna':>20} │ {'LSTM Optuna':>20} │"
if v1:
    header += f" {'Δ XGB':>8} │ {'Δ LSTM':>8} │"
print(header)
print("  " + "─" * len(header))

for ds in DATASETS:
    mx = all_results[ds]["XGBoost"]["metrics"]
    ml = all_results[ds]["LSTM"]["metrics"]
    row = f"  {ds:<8} │ RMSE={mx['RMSE']:>5.1f} N={mx['NASA Score']:>5.0f} │"
    row += f" RMSE={ml['RMSE']:>5.1f} N={ml['NASA Score']:>5.0f} │"
    if v1:
        dx = (mx["RMSE"] - v1[ds]["XGBoost"]["metrics"]["RMSE"]) \
            / v1[ds]["XGBoost"]["metrics"]["RMSE"] * 100
        dl = (ml["RMSE"] - v1[ds]["LSTM"]["metrics"]["RMSE"]) \
            / v1[ds]["LSTM"]["metrics"]["RMSE"] * 100
        row += f" {dx:>+6.1f}% │ {dl:>+6.1f}% │"
    print(row)

total_t = sum(all_results[ds][m]["time"]
              for ds in DATASETS for m in ["XGBoost", "LSTM"])
print(f"\n  Łączny czas: {total_t:.0f}s ({total_t / 60:.1f} min)")
print(f"  Wykresy: {PLOT_DIR}/47–52")
print(f"  Wyniki:  {RESULTS_DIR}/optuna_results.pkl")

# Najlepsze hiperparametry LSTM
print(f"\n  Najlepsze hiperparametry LSTM per zbiór:")
for ds in DATASETS:
    bp = all_results[ds]["LSTM"]["best_params"]
    loss = f"Huber(δ={bp.get('huber_delta', 1):.1f})" \
        if bp["use_huber"] else "MSE"
    print(f"    {ds}: {bp['n_layers']}×LSTM({bp['hidden_size']}) "
          f"→ Dense({bp['dense_size']}) "
          f"| drop={bp['dropout']:.2f} lr={bp['lr']:.5f} "
          f"bs={bp['batch_size']} seq={bp['seq_length']} loss={loss}")

print()