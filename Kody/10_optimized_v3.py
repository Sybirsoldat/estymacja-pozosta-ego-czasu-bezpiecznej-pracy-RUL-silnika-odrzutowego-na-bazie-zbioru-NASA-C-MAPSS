"""
=============================================================================
NASA C-MAPSS — Optymalizacja v3 (Krok 10)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Poprawki vs v2:
  ✦ Więcej prób Optuna: XGBoost 50, LSTM 40
  ✦ Ensemble 10 modeli (vs 5)
  ✦ Data augmentation: szum Gaussowski + time jitter na sekwencjach
  ✦ OneCycleLR scheduler (vs CosineAnnealing)
  ✦ Transfer learning: pretrain na większym zbiorze, fine-tune na docelowym
    FD001 ← pretrain FD003 (ten sam warunek, 2 tryby awarii)
    FD002 ← pretrain FD004 (te same warunki, 2 tryby awarii)
    FD003 ← pretrain FD001 (ten sam warunek, 1 tryb awarii)
    FD004 ← pretrain FD002 (te same warunki, 1 tryb awarii)
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
XGB_TRIALS = 50
LSTM_TRIALS = 40
ENSEMBLE_SEEDS = [42, 123, 7, 2024, 99, 314, 555, 888, 1337, 2077]  # 10

# Transfer learning: pretrain_source → fine-tune target
# Logika: ten sam warunek operacyjny, ale inny tryb awarii
TRANSFER_MAP = {
    "FD001": "FD003",  # 1 warunek → 1 warunek (2 awarie dają szerszą wiedzę)
    "FD002": "FD004",  # 6 warunków → 6 warunków
    "FD003": "FD001",  # 1 warunek → 1 warunek
    "FD004": "FD002",  # 6 warunków → 6 warunków
}

# Augmentation
AUG_NOISE_STD = 0.02    # σ szumu Gaussowskiego
AUG_COPIES = 2          # ile augmentowanych kopii na oryginał

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

PLOT_DIR = "./plots_optuna_v3"
RESULTS_DIR = "./results_optuna_v3"
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
XGB_DEVICE = "cpu"  # XGBoost na CPU — GPU rezerwujemy dla LSTM
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
        for di in d)

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


def load_and_preprocess(path, dataset_id):
    """Wczytaj i przetwórz jeden zbiór."""
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DATA AUGMENTATION                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def augment_sequences(X, y, noise_std=0.02, n_copies=2):
    """
    Data augmentation na sekwencjach:
    1. Szum Gaussowski — dodaje losowy szum do sensorów
       (symuluje szum pomiarowy w prawdziwych danych)
    2. Time jitter — losowo przesuwa wartości o ±1 timestep
       (symuluje niedokładność synchronizacji sensorów)
    Nie augmentujemy labeli (RUL jest deterministyczny).
    """
    rng = np.random.RandomState(SEED)
    X_aug, y_aug = [X], [y]

    for copy_i in range(n_copies):
        # Szum Gaussowski
        noise = rng.normal(0, noise_std, X.shape).astype(np.float32)
        X_noisy = np.clip(X + noise, 0, 1)

        # Time jitter — losowo zamieniamy sąsiednie timestepy (10% szans)
        X_jittered = X_noisy.copy()
        n_samples, seq_len, n_feat = X_jittered.shape
        swap_mask = rng.random((n_samples, seq_len - 1)) < 0.1
        for i in range(n_samples):
            for t in range(seq_len - 1):
                if swap_mask[i, t]:
                    X_jittered[i, t], X_jittered[i, t + 1] = \
                        X_jittered[i, t + 1].copy(), X_jittered[i, t].copy()

        X_aug.append(X_jittered)
        y_aug.append(y)

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  K-FOLD GENERATOR                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  XGBoost + OPTUNA + K-Fold CV                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def optimize_xgboost_cv(train_df, test_df, xgb_base, sensor_all, op_cols,
                        n_trials=50):
    print(f"    Feature engineering...")
    tr_xgb = create_enhanced_features(train_df, xgb_base)
    te_xgb = create_enhanced_features(test_df, xgb_base)

    feat_cols = [c for c in tr_xgb.columns
                 if c not in ["unit_id", "cycle", "RUL"] + op_cols + sensor_all]
    for col in feat_cols:
        arr = tr_xgb[col].values
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0
        tr_xgb[col] = arr

        arr = te_xgb[col].values
        mask = ~np.isfinite(arr)
        if mask.any():
            arr[mask] = 0
        te_xgb[col] = arr

    te_last = te_xgb.groupby("unit_id").last().reset_index()
    X_test = te_last[feat_cols].values.astype(np.float32)
    y_test = te_last["RUL"].values.astype(np.float32)

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
                             tr_xgb[val_m]["RUL"].values)], verbose=0)
            pred = np.clip(m.predict(tr_xgb[val_m][feat_cols].values), 0, RUL_CLIP)
            scores.append(rmse(tr_xgb[val_m]["RUL"].values, pred))
        return np.mean(scores)

    print(f"    Optuna XGBoost: {n_trials} prób × {K_FOLDS} foldów...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    final_p = {
        "n_estimators": 1000, "max_depth": bp["max_depth"],
        "learning_rate": bp["lr"], "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample"],
        "min_child_weight": bp["min_child_w"],
        "reg_alpha": bp["reg_alpha"], "reg_lambda": bp["reg_lambda"],
        "gamma": bp["gamma"], "objective": "reg:squarederror",
        "tree_method": XGB_TREE, "device": XGB_DEVICE,
        "random_state": SEED, "n_jobs": -1, "early_stopping_rounds": 30,
    }

    tr_uids, val_uids = folds[-1]
    tr_m = tr_xgb["unit_id"].isin(tr_uids)
    val_m = tr_xgb["unit_id"].isin(val_uids)

    t0 = time.time()
    model = xgb.XGBRegressor(**final_p)
    model.fit(tr_xgb[tr_m][feat_cols].values, tr_xgb[tr_m]["RUL"].values,
              eval_set=[(tr_xgb[val_m][feat_cols].values,
                         tr_xgb[val_m]["RUL"].values)], verbose=0)
    train_time = time.time() - t0

    y_pred = np.clip(model.predict(X_test), 0, RUL_CLIP)
    metrics = evaluate(y_test, y_pred)

    print(f"    Best CV RMSE: {study.best_value:.2f}")
    return metrics, y_pred, train_time, bp, y_test, study


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LSTM + OneCycleLR                                                       ║
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
                    epochs=80, patience=15, seed=42, trial=None,
                    use_augmentation=False, pretrained_state=None):
    """
    Trenuj LSTM z:
    - OneCycleLR (vs CosineAnnealing w v2)
    - Opcjonalna augmentacja danych
    - Opcjonalny pretrained_state (transfer learning)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Augmentacja
    if use_augmentation:
        X_tr, y_tr = augment_sequences(X_tr, y_tr, AUG_NOISE_STD, AUG_COPIES)

    model = FlexLSTM(n_features, hidden, n_layers, dense, dropout).to(device)

    # Transfer learning: załaduj wagi z pretrenowanego modelu
    if pretrained_state is not None:
        try:
            model.load_state_dict(pretrained_state, strict=False)
        except RuntimeError:
            pass  # Architektura się różni — pomiń

    Xt = torch.FloatTensor(X_tr).to(device)
    yt = torch.FloatTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    loader = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # OneCycleLR — agresywnie rośnie LR w pierwszej połowie,
    # potem powoli obniża. Lepsze zbieżność niż CosineAnnealing
    steps_per_epoch = len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3, div_factor=10, final_div_factor=100
    )

    criterion = nn.HuberLoss(delta=huber_delta) if use_huber else nn.MSELoss()
    mse_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    val_loader_eval = DataLoader(TensorDataset(Xv, yv),
                                 batch_size=batch_size, shuffle=False)

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
        val_mse_sum, val_count = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader_eval:
                pred = model(Xb)
                val_mse_sum += mse_fn(pred, yb).item() * len(yb)
                val_count += len(yb)
        val_mse = val_mse_sum / val_count

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if trial is not None and epoch % 10 == 0:
            trial.report(val_mse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LSTM + OPTUNA + K-Fold CV + Transfer Learning + Augmentation             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def optimize_lstm_cv(train_df, test_df, seq_feat, n_features,
                     n_trials=40, pretrained_state=None):

    folds = kfold_unit_splits(train_df, k=K_FOLDS, seed=SEED)
    seq_cache = {}
    test_seq_cache = {}

    def get_fold_data(fold_idx, sl):
        key = (fold_idx, sl)
        if key not in seq_cache:
            tr_uids, val_uids = folds[fold_idx]
            tr_sub = train_df[train_df["unit_id"].isin(tr_uids)]
            val_sub = train_df[train_df["unit_id"].isin(val_uids)]
            seq_cache[key] = (*build_seqs(tr_sub, seq_feat, sl),
                              *build_seqs(val_sub, seq_feat, sl))
        return seq_cache[key]

    def get_test_data(sl):
        if sl not in test_seq_cache:
            test_seq_cache[sl] = build_test_seqs(test_df, seq_feat, sl)
        return test_seq_cache[sl]

    def objective(trial):
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

        fold_scores = []
        for fi in range(K_FOLDS):
            Xtr, ytr, Xv, yv = get_fold_data(fi, sl)
            _, val_loss = train_lstm_once(
                Xtr, ytr, Xv, yv, n_features,
                hidden, n_layers, dense, dropout, lr, batch_size,
                use_huber, huber_delta,
                epochs=80, patience=12, seed=SEED, trial=trial,
                use_augmentation=False,
                pretrained_state=pretrained_state
            )
            fold_scores.append(val_loss)
            if USE_GPU:
                torch.cuda.empty_cache()
        return np.mean(fold_scores)

    print(f"    Optuna LSTM: {n_trials} prób × {K_FOLDS} foldów "
          f"(pruning, {'z transfer' if pretrained_state else 'bez transfer'})...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=pruner
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(RuntimeError, torch.cuda.OutOfMemoryError))

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
    print(f"    Pruned: {n_pruned}/{n_trials}")

    # ── Ensemble 10 modeli z augmentacją + transfer learning ──
    Xte, yte = get_test_data(best_sl)
    tr_uids_full, val_uids_last = folds[-1]
    tr_sub = train_df[train_df["unit_id"].isin(tr_uids_full)]
    val_sub = train_df[train_df["unit_id"].isin(val_uids_last)]
    Xtr_full, ytr_full = build_seqs(tr_sub, seq_feat, best_sl)
    Xv_last, yv_last = build_seqs(val_sub, seq_feat, best_sl)

    print(f"    Ensemble: {len(ENSEMBLE_SEEDS)} modeli "
          f"(augmentacja + {'transfer' if pretrained_state else 'scratch'})...")
    t0 = time.time()
    preds = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        model, _ = train_lstm_once(
            Xtr_full, ytr_full, Xv_last, yv_last, n_features,
            bp["hidden"], bp["n_layers"], bp["dense"], bp["dropout"],
            bp["lr"], bp["batch_size"], bp["use_huber"],
            bp.get("huber_delta", 1.0),
            epochs=120, patience=18, seed=seed,
            use_augmentation=True,
            pretrained_state=pretrained_state
        )
        model.eval()
        te_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte).to(device),
                               torch.zeros(len(Xte)).to(device)),
                               batch_size=256, shuffle=False)
        pred_parts = []
        with torch.no_grad():
            for Xb, _ in te_loader:
                pred_parts.append(model(Xb).cpu().numpy())
        pred = np.clip(np.concatenate(pred_parts), 0, RUL_CLIP)
        preds.append(pred)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"      Model {i + 1}/{len(ENSEMBLE_SEEDS)}: "
                  f"RMSE={rmse(yte, pred):.2f}")
        del model
        if USE_GPU:
            torch.cuda.empty_cache()

    train_time = time.time() - t0
    y_ensemble = np.clip(np.mean(preds, axis=0), 0, RUL_CLIP)
    metrics = evaluate(yte, y_ensemble)

    best_single = min(rmse(yte, p) for p in preds)
    print(f"    Ensemble RMSE={metrics['RMSE']:.2f} "
          f"(best single={best_single:.2f})")

    n_params = sum(p.numel() for p in FlexLSTM(
        n_features, bp["hidden"], bp["n_layers"],
        bp["dense"], bp["dropout"]).parameters())

    return metrics, y_ensemble, train_time, bp, n_params, yte, study


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TRANSFER LEARNING: PRETRAIN MODEL                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def pretrain_lstm(train_df, seq_feat, n_features, bp, sl):
    """
    Pretrenuj LSTM na source dataset.
    Zwraca state_dict do użycia jako inicjalizacja na target.
    """
    folds = kfold_unit_splits(train_df, k=K_FOLDS, seed=SEED)
    tr_uids, val_uids = folds[-1]
    tr_sub = train_df[train_df["unit_id"].isin(tr_uids)]
    val_sub = train_df[train_df["unit_id"].isin(val_uids)]
    Xtr, ytr = build_seqs(tr_sub, seq_feat, sl)
    Xv, yv = build_seqs(val_sub, seq_feat, sl)

    model, _ = train_lstm_once(
        Xtr, ytr, Xv, yv, n_features,
        bp["hidden"], bp["n_layers"], bp["dense"], bp["dropout"],
        bp["lr"], bp["batch_size"], bp.get("use_huber", False),
        bp.get("huber_delta", 1.0),
        epochs=80, patience=12, seed=SEED,
        use_augmentation=True
    )
    return {k: v.clone() for k, v in model.state_dict().items()}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ZNAJDŹ / POBIERZ DANE                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("NASA C-MAPSS — OPTYMALIZACJA v3 (Krok 10)")
print("=" * 70)
print(f"  XGBoost: Optuna {XGB_TRIALS} prób × {K_FOLDS} foldów")
print(f"  LSTM:    Optuna {LSTM_TRIALS} prób × {K_FOLDS} foldów")
print(f"           + Ensemble {len(ENSEMBLE_SEEDS)} + Augmentacja + Transfer")
print(f"  Device:  {device}" + (" (GPU)" if USE_GPU else ""))

data_path = None
for cand in ["./data", "./CMAPSSData", "./CMaps",
             os.path.expanduser("~/.cache/kagglehub")]:
    f = find_data_dir(cand, "FD001")
    if f:
        data_path = f
        break
if data_path is None:
    try:
        import kagglehub
        dl = kagglehub.dataset_download("behrad3d/nasa-cmaps")
        data_path = find_data_dir(dl, "FD001")
    except Exception as e:
        print(f"  [!] {e}")
        sys.exit(1)
print(f"  Dane:    {data_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FAZA 1: PREPROCESSING WSZYSTKICH ZBIORÓW                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("FAZA 1: Preprocessing")
print(f"{'=' * 70}")

preprocessed = {}
for ds_id in DATASETS:
    print(f"\n  {ds_id}...")
    train_df, test_df, xgb_base, seq_feat, sensor_all, op_cols = \
        load_and_preprocess(data_path, ds_id)
    preprocessed[ds_id] = {
        "train_df": train_df, "test_df": test_df,
        "xgb_base": xgb_base, "seq_feat": seq_feat,
        "sensor_all": sensor_all, "op_cols": op_cols,
        "n_features": len(seq_feat),
    }
    print(f"    Sensory: {len(seq_feat)}, XGBoost base: {len(xgb_base)}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FAZA 2: XGBoost (niezależny od transfer learning)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("FAZA 2: XGBoost + Optuna")
print(f"{'=' * 70}")

all_results = {}
all_studies = {}  # Optuna study objects per dataset per model

for ds_id in DATASETS:
    print(f"\n  ── {ds_id} ──")
    pp = preprocessed[ds_id]
    m_xgb, p_xgb, t_xgb, bp_xgb, y_test, study_xgb = optimize_xgboost_cv(
        pp["train_df"], pp["test_df"], pp["xgb_base"],
        pp["sensor_all"], pp["op_cols"], n_trials=XGB_TRIALS)

    all_results[ds_id] = {
        "y_test": y_test,
        "XGBoost": {"metrics": m_xgb, "y_pred": p_xgb,
                     "time": t_xgb, "best_params": bp_xgb},
    }
    all_studies[ds_id] = {"XGBoost": study_xgb}
    print(f"    ★ RMSE={m_xgb['RMSE']:.2f}  NASA={m_xgb['NASA Score']:,.0f}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FAZA 3: LSTM — pretrain sources, potem fine-tune targets                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("FAZA 3: LSTM + Optuna + Transfer Learning + Augmentacja")
print(f"{'=' * 70}")

# Krok 3a: Optuna na każdym zbiorze BEZ transfer (żeby znaleźć hiperparametry)
# Potem pretrain source i fine-tune target z tymi parametrami

# Kolejność: najpierw zbiory-źródła, potem targety
# FD001↔FD003 (1 warunek), FD002↔FD004 (6 warunków)
process_order = ["FD001", "FD003", "FD002", "FD004"]

pretrained_states = {}

for ds_id in process_order:
    print(f"\n  ── {ds_id} ──")
    pp = preprocessed[ds_id]
    n_feat = pp["n_features"]
    source_id = TRANSFER_MAP[ds_id]

    # Sprawdź czy mamy pretrained z source
    # Sensory muszą się zgadzać (ta sama liczba features)
    source_pp = preprocessed[source_id]
    can_transfer = (source_id in pretrained_states and
                    source_pp["n_features"] == n_feat)

    pt_state = pretrained_states.get(source_id) if can_transfer else None

    if can_transfer:
        print(f"    Transfer learning: {source_id} → {ds_id}")
    else:
        print(f"    Bez transfer (source {source_id}: "
              f"{'brak' if source_id not in pretrained_states else 'inna architektura'})")

    m_lstm, p_lstm, t_lstm, bp_lstm, n_params, y_test_lstm, study_lstm = \
        optimize_lstm_cv(
            pp["train_df"], pp["test_df"], pp["seq_feat"], n_feat,
            n_trials=LSTM_TRIALS, pretrained_state=pt_state)

    all_results[ds_id]["LSTM"] = {
        "metrics": m_lstm, "y_pred": p_lstm,
        "time": t_lstm, "best_params": bp_lstm, "n_params": n_params,
    }
    all_studies[ds_id]["LSTM"] = study_lstm
    print(f"    ★ RMSE={m_lstm['RMSE']:.2f}  NASA={m_lstm['NASA Score']:,.0f}")

    # Pretrain ten model dla przyszłego transfer
    print(f"    Pretraining {ds_id} jako source...")
    pretrained_states[ds_id] = pretrain_lstm(
        pp["train_df"], pp["seq_feat"], n_feat, bp_lstm, bp_lstm["seq_length"])


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ZAPIS                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

with open(os.path.join(RESULTS_DIR, "optuna_v3_results.pkl"), "wb") as f:
    pickle.dump(all_results, f)

# Wczytaj v1 i v2
v1, v2 = None, None
for p in ["./results_all/all_results.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            v1 = pickle.load(f)
        break
for p in ["./results_optuna_v2/optuna_v2_results.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f:
            v2 = pickle.load(f)
        break


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WIZUALIZACJE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("WIZUALIZACJE")
print(f"{'=' * 70}")

C = {"XGBoost v1": "#FFAB91", "LSTM v1": "#A5D6A7",
     "XGBoost v2": "#FF8A65", "LSTM v2": "#66BB6A",
     "XGBoost v3": "#E64A19", "LSTM v3": "#2E7D32"}

# ── 53: Porównanie v1 vs v2 vs v3 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ai, (metric, title) in enumerate([
    ("RMSE", "RMSE (↓ lepiej)"), ("NASA Score", "NASA Score (↓ lepiej)")
]):
    ax = axes[ai]
    x = np.arange(len(DATASETS))
    w, off = 0.13, 0
    if v1:
        for m in ["XGBoost", "LSTM"]:
            vals = [v1[ds][m]["metrics"][metric] for ds in DATASETS]
            ax.bar(x + off * w, vals, w, label=f"{m} v1",
                   color=C[f"{m} v1"], alpha=0.6, edgecolor="white")
            off += 1
    if v2:
        for m in ["XGBoost", "LSTM"]:
            vals = [v2[ds][m]["metrics"][metric] for ds in DATASETS]
            ax.bar(x + off * w, vals, w, label=f"{m} v2",
                   color=C[f"{m} v2"], alpha=0.75, edgecolor="white")
            off += 1
    for m in ["XGBoost", "LSTM"]:
        vals = [all_results[ds][m]["metrics"][metric] for ds in DATASETS]
        bars = ax.bar(x + off * w, vals, w, label=f"{m} v3",
                      color=C[f"{m} v3"], alpha=0.95, edgecolor="white")
        for xi, val in zip(x + off * w, vals):
            fmt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            ax.text(xi, val, fmt, ha="center", va="bottom",
                    fontsize=6, rotation=45)
        off += 1
    ax.set_xticks(x + w * (off - 1) / 2)
    ax.set_xticklabels(DATASETS)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)

plt.suptitle("v1 vs v2 vs v3 — pełna ewolucja modeli",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/53_v1_v2_v3_comparison.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 53_v1_v2_v3_comparison.png")

# ── 54: Heatmap v3 ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ai, (metric, title) in enumerate([
    ("RMSE", "RMSE v3 (↓)"), ("NASA Score", "NASA Score v3 (↓)")
]):
    ax = axes[ai]
    mat = np.array([[all_results[ds][m]["metrics"][metric]
                     for ds in DATASETS] for m in ["XGBoost", "LSTM"]])
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["XGBoost v3", "LSTM v3"])
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
plt.suptitle("Heatmap — v3", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/54_v3_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 54_v3_heatmap.png")

# ── 55: Scatter 4×2 ─────────────────────────────────────────────────────────
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
        ax.set_title(f"{ds} — {m} v3\nRMSE={met['RMSE']:.1f}  "
                     f"NASA={met['NASA Score']:,.0f}", fontsize=10)
        if col == 0: ax.set_ylabel("Pred RUL")
        if row == len(DATASETS) - 1: ax.set_xlabel("Actual RUL")
plt.suptitle("Predicted vs Actual — v3", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/55_v3_scatter.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 55_v3_scatter.png")

# ── 56: LSTM params ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
cols = ["Zbiór", "hidden", "layers", "dense", "dropout", "lr",
        "batch", "seq_len", "loss", "transfer", "RMSE", "NASA"]
rows = []
for ds in DATASETS:
    bp = all_results[ds]["LSTM"]["best_params"]
    met = all_results[ds]["LSTM"]["metrics"]
    ln = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) \
        if bp.get("use_huber", False) else "MSE"
    src = TRANSFER_MAP[ds]
    src_pp = preprocessed[src]
    has_tf = src_pp["n_features"] == preprocessed[ds]["n_features"]
    tf_str = f"← {src}" if has_tf else "—"
    rows.append([ds, bp["hidden"], bp["n_layers"], bp["dense"],
                 f"{bp['dropout']:.2f}", f"{bp['lr']:.5f}",
                 bp["batch_size"], bp["seq_length"], ln, tf_str,
                 f"{met['RMSE']:.1f}", f"{met['NASA Score']:,.0f}"])
table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.05, 2.0)
for j in range(len(cols)):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Hiperparametry LSTM v3 (Optuna + Transfer + Augmentacja)",
             fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/56_v3_lstm_params.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 56_v3_lstm_params.png")

# ── 57: Pełna tabela v1/v2/v3 ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 6))
ax.axis("off")
cl = ["Zbiór"]
if v1: cl += ["XGB v1", "LSTM v1"]
if v2: cl += ["XGB v2", "LSTM v2"]
cl += ["XGB v3", "LSTM v3"]
rd = []
for ds in DATASETS:
    row = [f"{ds}\n({DS_INFO[ds]['conditions']}w, {DS_INFO[ds]['faults']}f)"]
    if v1:
        for m in ["XGBoost", "LSTM"]:
            met = v1[ds][m]["metrics"]
            row.append(f"R={met['RMSE']:.1f}\nN={met['NASA Score']:,.0f}")
    if v2:
        for m in ["XGBoost", "LSTM"]:
            met = v2[ds][m]["metrics"]
            row.append(f"R={met['RMSE']:.1f}\nN={met['NASA Score']:,.0f}")
    for m in ["XGBoost", "LSTM"]:
        met = all_results[ds][m]["metrics"]
        row.append(f"R={met['RMSE']:.1f}\nN={met['NASA Score']:,.0f}")
    rd.append(row)
table = ax.table(cellText=rd, colLabels=cl, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.05, 2.8)
for j in range(len(cl)):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Podświetl najlepszy per wiersz
for i, ds in enumerate(DATASETS):
    rmses = {}
    if v1:
        rmses["XGB v1"] = v1[ds]["XGBoost"]["metrics"]["RMSE"]
        rmses["LSTM v1"] = v1[ds]["LSTM"]["metrics"]["RMSE"]
    if v2:
        rmses["XGB v2"] = v2[ds]["XGBoost"]["metrics"]["RMSE"]
        rmses["LSTM v2"] = v2[ds]["LSTM"]["metrics"]["RMSE"]
    rmses["XGB v3"] = all_results[ds]["XGBoost"]["metrics"]["RMSE"]
    rmses["LSTM v3"] = all_results[ds]["LSTM"]["metrics"]["RMSE"]
    best = min(rmses, key=rmses.get)
    if best in cl:
        table[i + 1, cl.index(best)].set_facecolor("#C8E6C9")
        table[i + 1, cl.index(best)].set_text_props(fontweight="bold")

ax.set_title("Ewolucja modeli: v1 → v2 → v3",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/57_v3_full_evolution.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 57_v3_full_evolution.png")

# ── 58: Parallel Coordinates — XGBoost & LSTM per zbiór ──────────────────────
print(f"\n  Optuna — Parallel Coordinates & Importance...")

for ds_id in DATASETS:
    study_xgb = all_studies[ds_id]["XGBoost"]
    study_lstm = all_studies[ds_id]["LSTM"]

    # ── XGBoost Parallel Coordinates ──
    trials_xgb = [t for t in study_xgb.trials
                  if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials_xgb) >= 3:
        xgb_params = ["max_depth", "lr", "subsample", "colsample",
                       "min_child_w", "reg_alpha", "reg_lambda", "gamma"]

        fig, ax = plt.subplots(figsize=(16, 6))
        param_values = {p: [] for p in xgb_params}
        obj_values = []
        for t in trials_xgb:
            obj_values.append(t.value)
            for p in xgb_params:
                param_values[p].append(t.params.get(p, 0))

        n_pp = len(xgb_params) + 1
        x_ticks = range(n_pp)
        norm_data = []
        for p in xgb_params:
            vals = np.array(param_values[p], dtype=float)
            mn, mx = vals.min(), vals.max()
            norm_data.append((vals - mn) / (mx - mn + 1e-10))
        obj_arr = np.array(obj_values)
        obj_mn, obj_mx = obj_arr.min(), obj_arr.max()
        norm_obj = (obj_arr - obj_mn) / (obj_mx - obj_mn + 1e-10)
        norm_data.append(norm_obj)

        cmap = plt.cm.RdYlGn_r
        for i in range(len(trials_xgb)):
            color = cmap(norm_obj[i])
            alpha = 0.8 if obj_values[i] <= np.percentile(obj_values, 20) else 0.15
            lw = 2.0 if obj_values[i] <= np.percentile(obj_values, 20) else 0.5
            ax.plot(x_ticks, [nd[i] for nd in norm_data],
                    color=color, alpha=alpha, lw=lw)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xgb_params + ["RMSE_CV"], rotation=30, ha="right")
        ax.set_ylabel("Znormalizowana wartość")
        ax.set_title(f"Parallel Coordinates — XGBoost {ds_id} "
                     f"({len(trials_xgb)} prób, zielone = najlepsze)")
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(obj_mn, obj_mx))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="CV RMSE", shrink=0.8)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/58_parallel_xgb_{ds_id}.png",
                    bbox_inches="tight")
        plt.close()

    # ── LSTM Parallel Coordinates ──
    trials_lstm = [t for t in study_lstm.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials_lstm) >= 3:
        lstm_params = ["hidden", "n_layers", "dense", "dropout",
                        "lr", "batch_size", "seq_length", "use_huber"]

        fig, ax = plt.subplots(figsize=(14, 6))
        param_values = {p: [] for p in lstm_params}
        obj_values = []
        for t in trials_lstm:
            obj_values.append(t.value)
            for p in lstm_params:
                val = t.params.get(p, 0)
                if isinstance(val, bool):
                    val = int(val)
                param_values[p].append(val)

        n_pp = len(lstm_params) + 1
        x_ticks = range(n_pp)
        norm_data = []
        for p in lstm_params:
            vals = np.array(param_values[p], dtype=float)
            mn, mx = vals.min(), vals.max()
            norm_data.append((vals - mn) / (mx - mn + 1e-10))
        obj_arr = np.array(obj_values)
        obj_mn, obj_mx = obj_arr.min(), obj_arr.max()
        norm_obj = (obj_arr - obj_mn) / (obj_mx - obj_mn + 1e-10)
        norm_data.append(norm_obj)

        cmap = plt.cm.RdYlGn_r
        for i in range(len(trials_lstm)):
            color = cmap(norm_obj[i])
            alpha = 0.8 if obj_values[i] <= np.percentile(obj_values, 20) else 0.15
            lw = 2.0 if obj_values[i] <= np.percentile(obj_values, 20) else 0.5
            ax.plot(x_ticks, [nd[i] for nd in norm_data],
                    color=color, alpha=alpha, lw=lw)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(lstm_params + ["MSE_CV"], rotation=30, ha="right")
        ax.set_ylabel("Znormalizowana wartość")
        ax.set_title(f"Parallel Coordinates — LSTM {ds_id} "
                     f"({len(trials_lstm)} prób)")
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(obj_mn, obj_mx))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="CV MSE", shrink=0.8)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/58_parallel_lstm_{ds_id}.png",
                    bbox_inches="tight")
        plt.close()

print(f"  [✓] 58_parallel_xgb/lstm_FD001–FD004.png")

# ── 59: Ważność hiperparametrów ──────────────────────────────────────────────

for ds_id in DATASETS:
    study_xgb = all_studies[ds_id]["XGBoost"]
    study_lstm = all_studies[ds_id]["LSTM"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # XGBoost importance
    ax = axes[0]
    trials_xgb = [t for t in study_xgb.trials
                  if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials_xgb) >= 5:
        xgb_params = list(trials_xgb[0].params.keys())
        importances = {}
        obj_vals = np.array([t.value for t in trials_xgb])
        for p in xgb_params:
            p_vals = np.array([t.params[p] for t in trials_xgb], dtype=float)
            if p_vals.std() > 1e-10:
                corr = abs(np.corrcoef(p_vals, obj_vals)[0, 1])
                importances[p] = corr if not np.isnan(corr) else 0
            else:
                importances[p] = 0
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]
        bars = ax.barh(range(len(names)), values, color="#E64A19", alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("|Korelacja z RMSE|")
        ax.set_title(f"XGBoost — ważność hiperparametrów\n{ds_id}")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    # LSTM importance
    ax = axes[1]
    trials_lstm = [t for t in study_lstm.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials_lstm) >= 5:
        common_params = set(trials_lstm[0].params.keys())
        for t in trials_lstm:
            common_params &= set(t.params.keys())
        lstm_params = sorted(common_params)
        importances = {}
        obj_vals = np.array([t.value for t in trials_lstm])
        for p in lstm_params:
            p_vals = np.array([t.params[p] for t in trials_lstm], dtype=float)
            if p_vals.std() > 1e-10:
                corr = abs(np.corrcoef(p_vals, obj_vals)[0, 1])
                importances[p] = corr if not np.isnan(corr) else 0
            else:
                importances[p] = 0
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]
        bars = ax.barh(range(len(names)), values, color="#2E7D32", alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("|Korelacja z MSE|")
        ax.set_title(f"LSTM — ważność hiperparametrów\n{ds_id}")
        ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.suptitle(f"Ważność hiperparametrów — {ds_id}", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/59_importance_{ds_id}.png", bbox_inches="tight")
    plt.close()

print(f"  [✓] 59_importance_FD001–FD004.png")

# ── 60: Optimization History ─────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds_id in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]
    for model_name, color in [("XGBoost", "#E64A19"), ("LSTM", "#2E7D32")]:
        study = all_studies[ds_id][model_name]
        trials = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE]
        vals = [t.value for t in trials]
        best_so_far = []
        current_best = float("inf")
        for v in vals:
            current_best = min(current_best, v)
            best_so_far.append(current_best)
        label_suffix = " (RMSE)" if model_name == "XGBoost" else " (MSE)"
        ax.plot(range(1, len(vals) + 1), vals, "o", alpha=0.3,
                color=color, markersize=3)
        ax.plot(range(1, len(best_so_far) + 1), best_so_far, "-",
                color=color, lw=2, label=f"{model_name}{label_suffix}")
    ax.set_xlabel("Nr próby Optuna")
    ax.set_ylabel("Wartość objective (CV)")
    ax.set_title(f"{ds_id}")
    ax.legend(fontsize=9)

plt.suptitle("Optimization History — jak objective spada z kolejnymi próbami",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/60_optuna_history.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 60_optuna_history.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PODSUMOWANIE — v3")
print(f"{'=' * 70}\n")

header = f"  {'Zbiór':<8} │ {'XGBoost v3':>20} │ {'LSTM v3':>20} │"
if v2: header += f" {'Δ XGB v2→v3':>12} │ {'Δ LSTM v2→v3':>13} │"
print(header)
print("  " + "─" * len(header))

for ds in DATASETS:
    mx = all_results[ds]["XGBoost"]["metrics"]
    ml = all_results[ds]["LSTM"]["metrics"]
    row = f"  {ds:<8} │ RMSE={mx['RMSE']:>5.1f} N={mx['NASA Score']:>6.0f} │"
    row += f" RMSE={ml['RMSE']:>5.1f} N={ml['NASA Score']:>6.0f} │"
    if v2:
        dx = (mx["RMSE"] - v2[ds]["XGBoost"]["metrics"]["RMSE"]) \
            / v2[ds]["XGBoost"]["metrics"]["RMSE"] * 100
        dl = (ml["RMSE"] - v2[ds]["LSTM"]["metrics"]["RMSE"]) \
            / v2[ds]["LSTM"]["metrics"]["RMSE"] * 100
        row += f" {dx:>+10.1f}% │ {dl:>+11.1f}% │"
    print(row)

total_t = sum(all_results[ds][m]["time"]
              for ds in DATASETS for m in ["XGBoost", "LSTM"])
print(f"\n  Łączny czas: {total_t:.0f}s ({total_t / 60:.1f} min)")
print(f"  Wykresy: {PLOT_DIR}/53–60")
print(f"  Wyniki:  {RESULTS_DIR}/optuna_v3_results.pkl")
print()