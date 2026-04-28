"""
=============================================================================
NASA C-MAPSS — Optymalizacja v2 z walidacją krzyżową (Krok 9 v2)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Poprawki vs v1 (09_optimized.py):
  ✦ K-Fold CV (3 foldy) zamiast jednego splita — stabilniejsza ocena
  ✦ Optuna MedianPruner — ucina złe konfiguracje po kilku epokach
  ✦ Zawężona przestrzeń LSTM (na bazie wyników v1)
  ✦ Dłuższy trening w fazie szukania (80 epok vs 60)
  ✦ Ensemble 5 modeli z najlepszymi parametrami
  ✦ Więcej prób: XGBoost 30, LSTM 25
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
K_FOLDS = 3
XGB_TRIALS = 30
LSTM_TRIALS = 25
ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99]  # 5 modeli ensemble

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

PLOT_DIR = "./plots_optuna_v2"
RESULTS_DIR = "./results_optuna_v2"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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
install_if_missing("kagglehub")

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
XGB_TREE = "hist"

if USE_GPU:
    print(f"[✓] GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")


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
        for di in d
    )

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
# ║  PREPROCESSING                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

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
        if mask.sum() == 0:
            continue
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


def preprocess_full(path, dataset_id):
    """Wczytanie i normalizacja. Zwraca train_df, test_df, cechy."""
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

    seq_feat = keep
    if info["conditions"] > 1:
        osc = MinMaxScaler()
        train_df[op_cols] = osc.fit_transform(train_df[op_cols])
        test_df[op_cols] = osc.transform(test_df[op_cols])
        xgb_base = keep + op_cols
    else:
        xgb_base = keep

    print(f"    Sensory: {len(keep)}, Warunki: {info['conditions']}")
    return train_df, test_df, xgb_base, seq_feat, sensor_all, op_cols


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  K-FOLD GENERATOR (po unit_id)                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def kfold_unit_splits(train_df, k=3, seed=42):
    """Generuje K foldów splittowanych po unit_id (nie po wierszach)."""
    uids = train_df["unit_id"].unique().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(uids)
    fold_size = len(uids) // k
    folds = []
    for i in range(k):
        if i < k - 1:
            val_uids = uids[i * fold_size:(i + 1) * fold_size]
        else:
            val_uids = uids[i * fold_size:]
        tr_uids = np.setdiff1d(uids, val_uids)
        folds.append((tr_uids, val_uids))
    return folds


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  XGBoost + OPTUNA + K-Fold CV                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def optimize_xgboost_cv(train_df, test_df, xgb_base, sensor_all, op_cols,
                        n_trials=30):
    """XGBoost z Optuna + K-Fold CV."""
    print(f"    Feature engineering...")
    tr_xgb = create_enhanced_features(train_df, xgb_base)
    te_xgb = create_enhanced_features(test_df, xgb_base)

    feat_cols = [c for c in tr_xgb.columns
                 if c not in ["unit_id", "cycle", "RUL"] + op_cols + sensor_all]
    tr_xgb[feat_cols] = tr_xgb[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
    te_xgb[feat_cols] = te_xgb[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)

    te_last = te_xgb.groupby("unit_id").last().reset_index()
    X_test = te_last[feat_cols].values.astype(np.float32)
    y_test = te_last["RUL"].values.astype(np.float32)

    # K-Fold
    folds = kfold_unit_splits(train_df, k=K_FOLDS, seed=SEED)

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
        scores = []
        for tr_uids, val_uids in folds:
            tr_m = tr_xgb["unit_id"].isin(tr_uids)
            val_m = tr_xgb["unit_id"].isin(val_uids)
            m = xgb.XGBRegressor(**p)
            m.fit(tr_xgb[tr_m][feat_cols].values,
                  tr_xgb[tr_m]["RUL"].values,
                  eval_set=[(tr_xgb[val_m][feat_cols].values,
                             tr_xgb[val_m]["RUL"].values)],
                  verbose=0)
            pred = np.clip(m.predict(tr_xgb[val_m][feat_cols].values), 0, RUL_CLIP)
            scores.append(rmse(tr_xgb[val_m]["RUL"].values, pred))
        return np.mean(scores)

    print(f"    Optuna XGBoost: {n_trials} prób × {K_FOLDS} foldów...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Finalny model: trenuj na PEŁNYM train
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
    }

    # Użyj ostatniego folda jako eval_set dla early stopping
    tr_uids, val_uids = folds[-1]
    tr_m = tr_xgb["unit_id"].isin(tr_uids)
    val_m = tr_xgb["unit_id"].isin(val_uids)

    t0 = time.time()
    final_p["early_stopping_rounds"] = 30
    model = xgb.XGBRegressor(**final_p)
    model.fit(tr_xgb[tr_m][feat_cols].values, tr_xgb[tr_m]["RUL"].values,
              eval_set=[(tr_xgb[val_m][feat_cols].values,
                         tr_xgb[val_m]["RUL"].values)], verbose=0)
    train_time = time.time() - t0

    y_pred = np.clip(model.predict(X_test), 0, RUL_CLIP)
    metrics = evaluate(y_test, y_pred)

    print(f"    Best CV RMSE: {study.best_value:.2f}")
    print(f"    depth={bp['max_depth']}, lr={bp['lr']:.4f}, "
          f"sub={bp['subsample']:.2f}, gamma={bp['gamma']:.2f}")

    return metrics, y_pred, train_time, bp, y_test


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LSTM — czysty, z pruningiem                                             ║
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


def train_lstm_once(X_tr, y_tr, X_val, y_val, n_features,
                    hidden, n_layers, dense, dropout, lr, batch_size,
                    use_huber, huber_delta,
                    epochs=80, patience=15, seed=42, trial=None):
    """Trenuj LSTM. Opcjonalny trial do pruningu Optuna."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FlexLSTM(n_features, hidden, n_layers, dense, dropout).to(device)

    Xt = torch.FloatTensor(X_tr).to(device)
    yt = torch.FloatTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    loader = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=huber_delta) if use_huber else nn.MSELoss()
    mse_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_mse = mse_fn(model(Xv), yv).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # Pruning: raportuj co 10 epok
        if trial is not None and epoch % 10 == 0:
            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def optimize_lstm_cv(train_df, test_df, seq_feat, n_features, n_trials=25):
    """LSTM z Optuna + K-Fold CV + MedianPruner + Ensemble."""

    folds = kfold_unit_splits(train_df, k=K_FOLDS, seed=SEED)

    # Cache sekwencji per (fold_idx, seq_length)
    seq_cache = {}

    def get_fold_data(fold_idx, sl):
        key = (fold_idx, sl)
        if key not in seq_cache:
            tr_uids, val_uids = folds[fold_idx]
            tr_sub = train_df[train_df["unit_id"].isin(tr_uids)]
            val_sub = train_df[train_df["unit_id"].isin(val_uids)]
            Xtr, ytr = build_seqs(tr_sub, seq_feat, sl)
            Xv, yv = build_seqs(val_sub, seq_feat, sl)
            seq_cache[key] = (Xtr, ytr, Xv, yv)
        return seq_cache[key]

    # Cache testu per seq_length
    test_seq_cache = {}

    def get_test_data(sl):
        if sl not in test_seq_cache:
            Xte, yte = build_test_seqs(test_df, seq_feat, sl)
            test_seq_cache[sl] = (Xte, yte)
        return test_seq_cache[sl]

    def objective(trial):
        # Zawężona przestrzeń na bazie wyników v1
        hidden = trial.suggest_categorical("hidden", [64, 128])
        n_layers = trial.suggest_int("n_layers", 1, 2)
        dense = trial.suggest_categorical("dense", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.15, 0.45)
        lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        sl = trial.suggest_categorical("seq_length", [30, 40])
        use_huber = trial.suggest_categorical("use_huber", [True, False])
        huber_delta = trial.suggest_float("huber_delta", 5.0, 20.0) \
            if use_huber else 1.0

        # K-Fold CV
        fold_scores = []
        for fi in range(K_FOLDS):
            Xtr, ytr, Xv, yv = get_fold_data(fi, sl)
            _, val_loss = train_lstm_once(
                Xtr, ytr, Xv, yv, n_features,
                hidden, n_layers, dense, dropout, lr, batch_size,
                use_huber, huber_delta,
                epochs=80, patience=12, seed=SEED, trial=trial
            )
            fold_scores.append(val_loss)

        return np.mean(fold_scores)

    print(f"    Optuna LSTM: {n_trials} prób × {K_FOLDS} foldów (z pruningiem)...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    best_sl = bp["seq_length"]
    loss_name = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) \
        if bp["use_huber"] else "MSE"

    print(f"    Best CV loss: {study.best_value:.4f} "
          f"(RMSE~{np.sqrt(study.best_value):.2f})")
    print(f"    hidden={bp['hidden']}, layers={bp['n_layers']}, "
          f"dense={bp['dense']}, drop={bp['dropout']:.2f}")
    print(f"    lr={bp['lr']:.5f}, batch={bp['batch_size']}, "
          f"seq={bp['seq_length']}, loss={loss_name}")
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    print(f"    Pruned: {n_pruned}/{n_trials} prób")

    # ── Ensemble: 5 modeli na pełnym train (last fold = val for stopping) ──
    Xte, yte = get_test_data(best_sl)
    tr_uids_full, val_uids_last = folds[-1]
    # Trening na fold 0+1 units, val na fold 2 (dla early stopping)
    tr_sub = train_df[train_df["unit_id"].isin(tr_uids_full)]
    val_sub = train_df[train_df["unit_id"].isin(val_uids_last)]
    Xtr_full, ytr_full = build_seqs(tr_sub, seq_feat, best_sl)
    Xv_last, yv_last = build_seqs(val_sub, seq_feat, best_sl)

    print(f"    Ensemble: {len(ENSEMBLE_SEEDS)} modeli...")
    t0 = time.time()
    preds = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        model, _ = train_lstm_once(
            Xtr_full, ytr_full, Xv_last, yv_last, n_features,
            bp["hidden"], bp["n_layers"], bp["dense"], bp["dropout"],
            bp["lr"], bp["batch_size"], bp["use_huber"],
            bp.get("huber_delta", 1.0),
            epochs=100, patience=15, seed=seed
        )
        model.eval()
        with torch.no_grad():
            Xte_t = torch.FloatTensor(Xte).to(device)
            pred = model(Xte_t).cpu().numpy()
        pred = np.clip(pred, 0, RUL_CLIP)
        preds.append(pred)
        r = rmse(yte, pred)
        print(f"      Model {i + 1}/{len(ENSEMBLE_SEEDS)}: RMSE={r:.2f}")

    train_time = time.time() - t0

    y_ensemble = np.clip(np.mean(preds, axis=0), 0, RUL_CLIP)
    metrics = evaluate(yte, y_ensemble)

    best_single = min(rmse(yte, p) for p in preds)
    print(f"    Ensemble RMSE={metrics['RMSE']:.2f} "
          f"(best single={best_single:.2f})")

    n_params = sum(p.numel() for p in FlexLSTM(
        n_features, bp["hidden"], bp["n_layers"],
        bp["dense"], bp["dropout"]).parameters())

    return metrics, y_ensemble, train_time, bp, n_params, yte


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ZNAJDŹ / POBIERZ DANE                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("NASA C-MAPSS — OPTYMALIZACJA v2 (K-Fold CV + Pruning)")
print("=" * 70)
print(f"  XGBoost: Optuna {XGB_TRIALS} prób × {K_FOLDS} foldów")
print(f"  LSTM:    Optuna {LSTM_TRIALS} prób × {K_FOLDS} foldów + "
      f"Ensemble {len(ENSEMBLE_SEEDS)} modeli")
print(f"  Device:  {device}" + (f" (GPU)" if USE_GPU else ""))

data_path = None
for cand in ["./data", "./CMAPSSData", "./CMaps",
             os.path.expanduser("~/.cache/kagglehub")]:
    f = find_data_dir(cand, "FD001")
    if f:
        data_path = f
        break

if data_path is None:
    print("\n  [i] Pobieram dane z Kaggle...")
    try:
        import kagglehub
        dl = kagglehub.dataset_download("behrad3d/nasa-cmaps")
        data_path = find_data_dir(dl, "FD001")
    except Exception as e:
        print(f"  [!] {e}")
        sys.exit(1)

print(f"  Dane:    {data_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GŁÓWNA PĘTLA                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

all_results = {}

for ds_id in DATASETS:
    print(f"\n{'=' * 70}")
    print(f"  {ds_id} — {DS_INFO[ds_id]['conditions']} war., "
          f"{DS_INFO[ds_id]['faults']} aw.")
    print(f"{'=' * 70}")

    train_df, test_df, xgb_base, seq_feat, sensor_all, op_cols = \
        preprocess_full(data_path, ds_id)

    ds_res = {}

    # XGBoost
    print(f"\n  [1/2] XGBoost + Optuna + {K_FOLDS}-Fold CV...")
    m_xgb, p_xgb, t_xgb, bp_xgb, y_test = optimize_xgboost_cv(
        train_df, test_df, xgb_base, sensor_all, op_cols, n_trials=XGB_TRIALS)
    ds_res["XGBoost"] = {
        "metrics": m_xgb, "y_pred": p_xgb,
        "time": t_xgb, "best_params": bp_xgb}
    ds_res["y_test"] = y_test
    print(f"    ★ RMSE={m_xgb['RMSE']:.2f}  NASA={m_xgb['NASA Score']:,.0f}")

    # LSTM
    print(f"\n  [2/2] LSTM + Optuna + {K_FOLDS}-Fold CV + Ensemble...")
    m_lstm, p_lstm, t_lstm, bp_lstm, n_params, y_test_lstm = optimize_lstm_cv(
        train_df, test_df, seq_feat, len(seq_feat), n_trials=LSTM_TRIALS)
    ds_res["LSTM"] = {
        "metrics": m_lstm, "y_pred": p_lstm,
        "time": t_lstm, "best_params": bp_lstm, "n_params": n_params}
    print(f"    ★ RMSE={m_lstm['RMSE']:.2f}  NASA={m_lstm['NASA Score']:,.0f}")

    all_results[ds_id] = ds_res

with open(os.path.join(RESULTS_DIR, "optuna_v2_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WCZYTANIE v1                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

v1 = None
for p in ["./results_all/all_results.pkl", "../results_all/all_results.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            v1 = pickle.load(f)
        print(f"\n[✓] Wyniki v1: {p}")
        break


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WIZUALIZACJE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("WIZUALIZACJE")
print(f"{'=' * 70}")

C = {"XGBoost v1": "#FFAB91", "LSTM v1": "#A5D6A7",
     "XGBoost": "#E64A19", "LSTM": "#2E7D32"}

# ── 47v2: Porównanie v1 vs Optuna v2 ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ai, (metric, title) in enumerate([
    ("RMSE", "RMSE (↓ lepiej)"), ("NASA Score", "NASA Score (↓ lepiej)")
]):
    ax = axes[ai]
    x = np.arange(len(DATASETS))
    w, off = 0.18, 0
    if v1:
        for mv in ["XGBoost", "LSTM"]:
            vals = [v1[ds][mv]["metrics"][metric] for ds in DATASETS]
            ax.bar(x + off * w, vals, w, label=f"{mv} v1",
                   color=C[f"{mv} v1"], alpha=0.7, edgecolor="white")
            off += 1
    for m in ["XGBoost", "LSTM"]:
        vals = [all_results[ds][m]["metrics"][metric] for ds in DATASETS]
        bars = ax.bar(x + off * w, vals, w, label=f"{m} Optuna v2",
                      color=C[m], alpha=0.9, edgecolor="white")
        for xi, val in zip(x + off * w, vals):
            fmt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            ax.text(xi, val, fmt, ha="center", va="bottom",
                    fontsize=7, rotation=45)
        off += 1
    ax.set_xticks(x + w * (off - 1) / 2)
    ax.set_xticklabels(DATASETS)
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.suptitle("v1 vs Optuna v2 (K-Fold CV + Pruning + Ensemble)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/47v2_vs_v1.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 47v2_vs_v1.png")

# ── 48v2: Heatmap ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ai, (metric, title) in enumerate([
    ("RMSE", "RMSE Optuna v2 (↓)"), ("NASA Score", "NASA Score Optuna v2 (↓)")
]):
    ax = axes[ai]
    mat = np.array([[all_results[ds][m]["metrics"][metric]
                     for ds in DATASETS] for m in ["XGBoost", "LSTM"]])
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["XGBoost", "LSTM"])
    ax.set_title(title)
    for i in range(2):
        for j in range(len(DATASETS)):
            val = mat[i, j]
            txt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            best = val == mat[:, j].min()
            clr = "white" if val > np.median(mat) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    fontweight="bold" if best else "normal", color=clr)
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle("Heatmap — Optuna v2", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/48v2_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 48v2_heatmap.png")

# ── 49v2: Scatter 4×2 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 5 * len(DATASETS)))
for row, ds in enumerate(DATASETS):
    yt = all_results[ds]["y_test"]
    for col, (m, clr) in enumerate([("XGBoost", "#E64A19"), ("LSTM", "#2E7D32")]):
        ax = axes[row][col]
        yp = all_results[ds][m]["y_pred"]
        met = all_results[ds][m]["metrics"]
        ax.scatter(yt, yp, alpha=0.5, s=20, c=clr, edgecolors="none")
        ax.plot([0, RUL_CLIP], [0, RUL_CLIP], "k--", lw=1, alpha=0.4)
        ax.fill_between([0, RUL_CLIP], [-15, RUL_CLIP - 15],
                        [15, RUL_CLIP + 15], alpha=0.08, color="green")
        ax.set_xlim(-5, RUL_CLIP + 5); ax.set_ylim(-5, RUL_CLIP + 5)
        ax.set_aspect("equal")
        ax.set_title(f"{ds} — {m}\nRMSE={met['RMSE']:.1f}  "
                     f"NASA={met['NASA Score']:,.0f}", fontsize=10)
        if col == 0: ax.set_ylabel("Pred RUL")
        if row == len(DATASETS) - 1: ax.set_xlabel("Actual RUL")
plt.suptitle("Predicted vs Actual — Optuna v2", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/49v2_scatter.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 49v2_scatter.png")

# ── 50v2: Delta v1→v2 ───────────────────────────────────────────────────────
if v1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ai, (metric, title) in enumerate([
        ("RMSE", "Zmiana RMSE (v1→Optuna v2)"),
        ("NASA Score", "Zmiana NASA Score (v1→Optuna v2)")
    ]):
        ax = axes[ai]
        x = np.arange(len(DATASETS))
        w = 0.35
        for mi, m in enumerate(["XGBoost", "LSTM"]):
            v1v = [v1[ds][m]["metrics"][metric] for ds in DATASETS]
            v2v = [all_results[ds][m]["metrics"][metric] for ds in DATASETS]
            delta = [(new - old) / old * 100 for old, new in zip(v1v, v2v)]
            bars = ax.bar(x + mi * w - w / 2, delta, w,
                          label=m, color=C[m], alpha=0.85)
            for bar in bars:
                val = bar.get_height()
                c2 = "#2E7D32" if val < 0 else "#F44336"
                ax.text(bar.get_x() + bar.get_width() / 2, val,
                        f"{val:+.1f}%", ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=9, color=c2, fontweight="bold")
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(DATASETS)
        ax.set_ylabel("Zmiana (%)"); ax.set_title(title); ax.legend()
    plt.suptitle("Poprawa po Optuna v2 (ujemne = lepiej)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/50v2_improvement.png", bbox_inches="tight")
    plt.close()
    print(f"  [✓] 50v2_improvement.png")

# ── 51v2: Tabela hiperparametrów LSTM ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
cols = ["Zbiór", "hidden", "layers", "dense", "dropout", "lr",
        "batch", "seq_len", "loss", "RMSE", "NASA"]
rows = []
for ds in DATASETS:
    bp = all_results[ds]["LSTM"]["best_params"]
    met = all_results[ds]["LSTM"]["metrics"]
    ln = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) \
        if bp["use_huber"] else "MSE"
    rows.append([ds, bp["hidden"], bp["n_layers"], bp["dense"],
                 f"{bp['dropout']:.2f}", f"{bp['lr']:.5f}",
                 bp["batch_size"], bp["seq_length"], ln,
                 f"{met['RMSE']:.1f}", f"{met['NASA Score']:,.0f}"])
table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 2.0)
for j in range(len(cols)):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Najlepsze hiperparametry LSTM per zbiór (Optuna v2, K-Fold CV)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/51v2_lstm_params.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 51v2_lstm_params.png")

# ── 52v2: Pełna tabela ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis("off")
cl = ["Zbiór"]
if v1: cl += ["XGBoost v1", "LSTM v1"]
cl += ["XGBoost Optuna v2", "LSTM Optuna v2"]
rd = []
for ds in DATASETS:
    row = [f"{ds}\n({DS_INFO[ds]['conditions']}w, {DS_INFO[ds]['faults']}f)"]
    if v1:
        for m in ["XGBoost", "LSTM"]:
            met = v1[ds][m]["metrics"]
            row.append(f"RMSE={met['RMSE']:.1f}\nNASA={met['NASA Score']:,.0f}")
    for m in ["XGBoost", "LSTM"]:
        met = all_results[ds][m]["metrics"]
        row.append(f"RMSE={met['RMSE']:.1f}\nNASA={met['NASA Score']:,.0f}")
    rd.append(row)
table = ax.table(cellText=rd, colLabels=cl, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.1, 2.8)
for j in range(len(cl)):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i, ds in enumerate(DATASETS):
    rmses = {}
    if v1:
        rmses["XGBoost v1"] = v1[ds]["XGBoost"]["metrics"]["RMSE"]
        rmses["LSTM v1"] = v1[ds]["LSTM"]["metrics"]["RMSE"]
    rmses["XGBoost Optuna v2"] = all_results[ds]["XGBoost"]["metrics"]["RMSE"]
    rmses["LSTM Optuna v2"] = all_results[ds]["LSTM"]["metrics"]["RMSE"]
    best = min(rmses, key=rmses.get)
    if best in cl:
        table[i + 1, cl.index(best)].set_facecolor("#C8E6C9")
        table[i + 1, cl.index(best)].set_text_props(fontweight="bold")
ax.set_title("Porównanie v1 vs Optuna v2 — pełna tabela",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/52v2_full_table.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 52v2_full_table.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PODSUMOWANIE — Optuna v2 (K-Fold CV + Pruning + Ensemble)")
print(f"{'=' * 70}\n")

header = f"  {'Zbiór':<8} │ {'XGBoost':>20} │ {'LSTM Ensemble':>20} │"
if v1: header += f" {'Δ XGB':>8} │ {'Δ LSTM':>8} │"
print(header)
print("  " + "─" * len(header))

for ds in DATASETS:
    mx = all_results[ds]["XGBoost"]["metrics"]
    ml = all_results[ds]["LSTM"]["metrics"]
    row = f"  {ds:<8} │ RMSE={mx['RMSE']:>5.1f} N={mx['NASA Score']:>6.0f} │"
    row += f" RMSE={ml['RMSE']:>5.1f} N={ml['NASA Score']:>6.0f} │"
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
print(f"  Wykresy: {PLOT_DIR}/47v2–52v2")
print(f"  Wyniki:  {RESULTS_DIR}/optuna_v2_results.pkl")

print(f"\n  Hiperparametry LSTM:")
for ds in DATASETS:
    bp = all_results[ds]["LSTM"]["best_params"]
    ln = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) \
        if bp["use_huber"] else "MSE"
    print(f"    {ds}: {bp['n_layers']}×LSTM({bp['hidden']}) "
          f"→ Dense({bp['dense']}) | drop={bp['dropout']:.2f} "
          f"lr={bp['lr']:.5f} bs={bp['batch_size']} "
          f"seq={bp['seq_length']} loss={ln}")
print()