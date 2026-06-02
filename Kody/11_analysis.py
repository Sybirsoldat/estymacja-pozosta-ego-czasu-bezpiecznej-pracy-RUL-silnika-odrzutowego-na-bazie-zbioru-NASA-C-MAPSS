"""
=============================================================================
NASA C-MAPSS — Analiza i walidacja wyników (Krok 11)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Ładuje wyniki z kroków 9v2 / 10v3 i generuje:
  1. Accuracy Window (% predykcji w ±5, ±10, ±15, ±20 cykli)
  2. Analiza rezyduów (błąd vs rzeczywisty RUL)
  3. Test Wilcoxona (istotność statystyczna XGBoost vs LSTM)
  4. SHAP values (interpretowalność XGBoost)
  5. Gradient saliency (interpretowalność LSTM)
  6. Analiza kosztowa (minimalizacja łącznego kosztu)
  7. Krzywe degradacji (predykcja RUL cykl po cyklu)
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
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy import stats

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

PLOT_DIR = "./plots_analysis"
os.makedirs(PLOT_DIR, exist_ok=True)

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
install_if_missing("shap")
install_if_missing("torch")
install_if_missing("kagglehub")

import xgboost as xgb
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  WCZYTANIE WYNIKÓW                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("NASA C-MAPSS — ANALIZA I WALIDACJA WYNIKÓW (Krok 11)")
print("=" * 70)

# Szukaj najnowszych wyników
results = None
source = None
for path, name in [
    ("./results_optuna_v3/optuna_v3_results.pkl", "v3"),
    ("./results_optuna_v2/optuna_v2_results.pkl", "v2"),
]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            results = pickle.load(f)
        source = name
        print(f"[✓] Wyniki: {path} ({name})")
        break

if results is None:
    print("BŁĄD: Brak wyników! Uruchom najpierw 09_optimized_v2.py lub 10_optimized_v3.py")
    sys.exit(1)

# Sprawdź dostępne modele
models_available = []
for ds in DATASETS:
    for m in ["XGBoost", "LSTM"]:
        if m in results[ds]:
            models_available.append(m)
            break
models_available = list(set(models_available))
print(f"  Modele: {models_available}")
print(f"  Zbiory: {DATASETS}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  METRYKI                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return sum(
        (np.exp(-di / 13) - 1) if di < 0 else (np.exp(di / 10) - 1)
        for di in d)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. ACCURACY WINDOW                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("1. Accuracy Window — % predykcji w ±N cykli")
print(f"{'=' * 70}")

windows = [5, 10, 15, 20]
C = {"XGBoost": "#E64A19", "LSTM": "#2E7D32"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]
    yt = results[ds]["y_test"]

    for mi, m in enumerate(["XGBoost", "LSTM"]):
        if m not in results[ds]:
            continue
        yp = results[ds][m]["y_pred"]
        errors = np.abs(yp - yt)
        accuracies = [100 * np.mean(errors <= w) for w in windows]

        x = np.arange(len(windows))
        width = 0.35
        bars = ax.bar(x + mi * width, accuracies, width,
                      label=m, color=C[m], alpha=0.85)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{acc:.0f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"±{w}" for w in windows])
    ax.set_ylabel("% predykcji w oknie")
    ax.set_title(f"{ds} ({DS_INFO[ds]['conditions']}w, {DS_INFO[ds]['faults']}f)")
    ax.set_ylim(0, 105)
    ax.legend()

    # Tabelka w konsoli
    print(f"\n  {ds}:")
    for m in ["XGBoost", "LSTM"]:
        if m not in results[ds]:
            continue
        yp = results[ds][m]["y_pred"]
        errors = np.abs(yp - yt)
        accs = [f"±{w}: {100 * np.mean(errors <= w):.0f}%" for w in windows]
        print(f"    {m:>8}: {', '.join(accs)}")

plt.suptitle("Accuracy Window — % predykcji w ±N cykli od prawdy",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/61_accuracy_window.png", bbox_inches="tight")
plt.close()
print(f"\n  [✓] 61_accuracy_window.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. ANALIZA REZYDUÓW                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("2. Analiza rezyduów — błąd vs rzeczywisty RUL")
print(f"{'=' * 70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]
    yt = results[ds]["y_test"]

    for m in ["XGBoost", "LSTM"]:
        if m not in results[ds]:
            continue
        yp = results[ds][m]["y_pred"]
        residuals = yp - yt  # pozytywne = za późno, negatywne = za wcześnie

        ax.scatter(yt, residuals, alpha=0.5, s=25, color=C[m],
                   label=m, edgecolors="none")

    ax.axhline(0, color="black", lw=1.5, ls="--", alpha=0.5)
    ax.axhline(15, color="red", lw=1, ls=":", alpha=0.4, label="+15 (za późno)")
    ax.axhline(-15, color="green", lw=1, ls=":", alpha=0.4, label="-15 (za wcześnie)")
    ax.fill_between([0, RUL_CLIP], -15, 15, alpha=0.05, color="green")
    ax.set_xlabel("Rzeczywisty RUL")
    ax.set_ylabel("Błąd (predicted − actual)")
    ax.set_title(f"{ds}")
    ax.legend(fontsize=8)

plt.suptitle("Analiza rezyduów — czy model systematycznie się myli?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/62_residual_analysis.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 62_residual_analysis.png")

# Analiza tendencji: czy błąd rośnie z RUL?
for ds in DATASETS:
    yt = results[ds]["y_test"]
    for m in ["XGBoost", "LSTM"]:
        if m not in results[ds]:
            continue
        yp = results[ds][m]["y_pred"]
        residuals = yp - yt
        corr, pval = stats.pearsonr(yt, residuals)
        trend = "za późno przy wysokim RUL" if corr > 0.1 else \
                "za wcześnie przy wysokim RUL" if corr < -0.1 else "brak trendu"
        print(f"  {ds} {m:>8}: korelacja(RUL, błąd)={corr:+.3f} "
              f"(p={pval:.4f}) → {trend}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. TEST WILCOXONA                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("3. Test Wilcoxona — XGBoost vs LSTM istotność statystyczna")
print(f"{'=' * 70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]
    yt = results[ds]["y_test"]

    if "XGBoost" not in results[ds] or "LSTM" not in results[ds]:
        continue

    err_xgb = np.abs(results[ds]["XGBoost"]["y_pred"] - yt)
    err_lstm = np.abs(results[ds]["LSTM"]["y_pred"] - yt)

    # Wilcoxon signed-rank test (sparowane próbki)
    stat, pval = stats.wilcoxon(err_xgb, err_lstm)
    sig = "★ TAK" if pval < 0.05 else "NIE"

    # Violin plot porównawczy
    parts = ax.violinplot([err_xgb, err_lstm], positions=[0, 1],
                          showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([C["XGBoost"], C["LSTM"]][i])
        pc.set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["XGBoost", "LSTM"])
    ax.set_ylabel("|Błąd predykcji|")

    better = "LSTM" if np.median(err_lstm) < np.median(err_xgb) else "XGBoost"
    ax.set_title(f"{ds}: Wilcoxon p={pval:.4f} → {sig}\n"
                 f"Lepszy: {better} (mediana |błędu|: "
                 f"XGB={np.median(err_xgb):.1f}, LSTM={np.median(err_lstm):.1f})")

    print(f"  {ds}: Wilcoxon stat={stat:.0f}, p={pval:.4f} → "
          f"różnica {sig} istotna (α=0.05), lepszy: {better}")

plt.suptitle("Test Wilcoxona — czy różnica XGBoost vs LSTM jest istotna?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/63_wilcoxon_test.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 63_wilcoxon_test.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. ANALIZA KOSZTOWA                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("4. Analiza kosztowa — minimalizacja łącznego kosztu")
print(f"{'=' * 70}")

# Scenariusz: koszt wczesnej wymiany = 1 (strata żywotności),
# koszt awarii = 10 (katastrofa)
COST_EARLY = 1    # koszt za każdy cykl wczesnej wymiany
COST_LATE = 10    # koszt za każdy cykl spóźnionej wymiany

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]
    yt = results[ds]["y_test"]

    costs = {}
    for m in ["XGBoost", "LSTM"]:
        if m not in results[ds]:
            continue
        yp = results[ds][m]["y_pred"]
        d = yp - yt  # d>0 = za późno, d<0 = za wcześnie

        early_cost = np.sum(np.abs(d[d < 0])) * COST_EARLY
        late_cost = np.sum(d[d >= 0]) * COST_LATE
        total_cost = early_cost + late_cost
        n_early = np.sum(d < 0)
        n_late = np.sum(d >= 0)

        costs[m] = {
            "early": early_cost, "late": late_cost,
            "total": total_cost, "n_early": n_early, "n_late": n_late
        }

    # Stacked bar chart
    models = list(costs.keys())
    early_vals = [costs[m]["early"] for m in models]
    late_vals = [costs[m]["late"] for m in models]

    x = np.arange(len(models))
    ax.bar(x, early_vals, 0.5, label="Wczesna wymiana (×1)",
           color="#4CAF50", alpha=0.8)
    ax.bar(x, late_vals, 0.5, bottom=early_vals,
           label="Spóźniona awaria (×10)", color="#F44336", alpha=0.8)

    for i, m in enumerate(models):
        total = costs[m]["total"]
        ax.text(i, total + max(total * 0.02, 50),
                f"Σ={total:,.0f}", ha="center", fontweight="bold")
        ax.text(i, early_vals[i] / 2,
                f"{costs[m]['n_early']} wcz.", ha="center",
                fontsize=8, color="white")
        ax.text(i, early_vals[i] + late_vals[i] / 2,
                f"{costs[m]['n_late']} późn.", ha="center",
                fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Łączny koszt")
    ax.set_title(f"{ds}")
    ax.legend(fontsize=8)

    best_m = min(costs, key=lambda m: costs[m]["total"])
    print(f"  {ds}: Najtańszy={best_m} "
          f"(koszt={costs[best_m]['total']:,.0f}, "
          f"wczesne={costs[best_m]['n_early']}, "
          f"późne={costs[best_m]['n_late']})")

plt.suptitle(f"Analiza kosztowa — wczesna wymiana ×{COST_EARLY} vs "
             f"awaria ×{COST_LATE}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/64_cost_analysis.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 64_cost_analysis.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. SHAP VALUES — XGBoost                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("5. SHAP Values — interpretowalność XGBoost")
print(f"{'=' * 70}")

# Potrzebujemy przetrenować model — ładujemy dane i trenujemy z best params

def find_data_dir(root, dataset_id):
    if not root or not os.path.exists(root):
        return None
    if os.path.isfile(os.path.join(root, f"train_{dataset_id}.txt")):
        return root
    for dirpath, _, filenames in os.walk(root):
        if f"train_{dataset_id}.txt" in filenames:
            return dirpath
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

# Znajdź dane
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
    except Exception:
        pass

if data_path is not None:
    # SHAP na FD001 (najprostszy, najczytelniejsze wykresy)
    for ds_shap in ["FD001"]:
        if "XGBoost" not in results[ds_shap]:
            continue
        bp = results[ds_shap]["XGBoost"].get("best_params", {})
        if not bp:
            print(f"  [!] Brak best_params dla {ds_shap}, pomijam SHAP")
            continue

        print(f"  Trenowanie XGBoost {ds_shap} dla SHAP...")

        # Quick preprocessing
        train_df = pd.read_csv(os.path.join(data_path, f"train_{ds_shap}.txt"),
                               sep=r"\s+", header=None, names=COLUMNS)
        test_df = pd.read_csv(os.path.join(data_path, f"test_{ds_shap}.txt"),
                              sep=r"\s+", header=None, names=COLUMNS)
        rul_df = pd.read_csv(os.path.join(data_path, f"RUL_{ds_shap}.txt"),
                             sep=r"\s+", header=None, names=["RUL"])

        mc = train_df.groupby("unit_id")["cycle"].max().reset_index()
        mc.columns = ["unit_id", "max_cycle"]
        train_df = train_df.merge(mc, on="unit_id")
        train_df["RUL"] = (train_df["max_cycle"] - train_df["cycle"]).clip(
            upper=RUL_CLIP)
        train_df.drop("max_cycle", axis=1, inplace=True)

        mct = test_df.groupby("unit_id")["cycle"].max().reset_index()
        mct.columns = ["unit_id", "max_cycle"]
        rul_df["unit_id"] = range(1, len(rul_df) + 1)
        mct = mct.merge(rul_df, on="unit_id")
        mct["total_life"] = mct["max_cycle"] + mct["RUL"]
        test_df = test_df.merge(mct[["unit_id", "total_life"]], on="unit_id")
        test_df["RUL"] = (test_df["total_life"] - test_df["cycle"]).clip(
            upper=RUL_CLIP)
        test_df.drop("total_life", axis=1, inplace=True)

        sensor_all = [f"sensor_{i}" for i in range(1, 22)]
        op_cols = [f"op_setting_{i}" for i in range(1, 4)]
        var = train_df[sensor_all].var()
        keep = var[var > var.median() * 0.01].index.tolist()
        info = DS_INFO[ds_shap]

        train_df, test_df = normalize_per_condition(
            train_df, test_df, keep, op_cols, info["conditions"])

        xgb_base = keep
        # Simple features for SHAP (bez full engineering — czytelniejsze nazwy)
        for w in [10]:
            for uid in train_df["unit_id"].unique():
                mask = train_df["unit_id"] == uid
                for s in keep:
                    train_df.loc[mask, f"{s}_mean_{w}"] = \
                        train_df.loc[mask, s].rolling(w, min_periods=1).mean()
                    train_df.loc[mask, f"{s}_std_{w}"] = \
                        train_df.loc[mask, s].rolling(w, min_periods=1).std().fillna(0)

            for uid in test_df["unit_id"].unique():
                mask = test_df["unit_id"] == uid
                for s in keep:
                    test_df.loc[mask, f"{s}_mean_{w}"] = \
                        test_df.loc[mask, s].rolling(w, min_periods=1).mean()
                    test_df.loc[mask, f"{s}_std_{w}"] = \
                        test_df.loc[mask, s].rolling(w, min_periods=1).std().fillna(0)

        feat_cols = [c for c in train_df.columns
                     if c not in ["unit_id", "cycle", "RUL"] + op_cols + sensor_all]
        for col in feat_cols:
            arr = train_df[col].values
            mask = ~np.isfinite(arr)
            if mask.any():
                arr[mask] = 0
            train_df[col] = arr
            arr = test_df[col].values
            mask = ~np.isfinite(arr)
            if mask.any():
                arr[mask] = 0
            test_df[col] = arr

        uids = train_df["unit_id"].unique()
        rng = np.random.RandomState(SEED)
        rng.shuffle(uids)
        split = int(len(uids) * 0.8)
        tr_m = train_df["unit_id"].isin(uids[:split])
        val_m = train_df["unit_id"].isin(uids[split:])

        te_last = test_df.groupby("unit_id").last().reset_index()

        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=bp.get("max_depth", 4),
            learning_rate=bp.get("lr", 0.03),
            subsample=bp.get("subsample", 0.7),
            colsample_bytree=bp.get("colsample", 0.7),
            base_score=0.5,  # explicit — fix SHAP compatibility
            random_state=SEED, n_jobs=-1, early_stopping_rounds=30,
        )
        model.fit(
            train_df[tr_m][feat_cols].values,
            train_df[tr_m]["RUL"].values,
            eval_set=[(train_df[val_m][feat_cols].values,
                       train_df[val_m]["RUL"].values)],
            verbose=0
        )

        # SHAP
        try:
            X_test_shap = te_last[feat_cols].values.astype(np.float32)
            # shap.Explainer — kompatybilny z nowszym XGBoost
            explainer = shap.Explainer(model, X_test_shap)
            shap_values = explainer(X_test_shap)

            # Beeswarm plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.beeswarm(shap_values, max_display=20, show=False)
            plt.title(f"SHAP — XGBoost {ds_shap}\n"
                      f"(jak cechy wpływają na predykcję RUL)")
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/65_shap_beeswarm_{ds_shap}.png",
                        bbox_inches="tight")
            plt.close()

            # Bar plot (mean |SHAP|)
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.bar(shap_values, max_display=20, show=False)
            plt.title(f"SHAP — średnia |wpływu| na predykcję ({ds_shap})")
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/65_shap_bar_{ds_shap}.png",
                        bbox_inches="tight")
            plt.close()
            print(f"  [✓] 65_shap_beeswarm/bar_{ds_shap}.png")

        except Exception as e:
            print(f"  [!] SHAP error: {e}")
            print(f"  [!] Spróbuj: pip install shap --upgrade")

else:
    print("  [!] Brak danych — pomijam SHAP i saliency")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. GRADIENT SALIENCY — LSTM                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("6. Gradient Saliency — interpretowalność LSTM")
print(f"{'=' * 70}")

if data_path is not None:
    class FlexLSTM(nn.Module):
        def __init__(self, n_features, hidden=64, n_layers=2, dense=32,
                     dropout=0.3):
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

    for ds_sal in ["FD001"]:
        if "LSTM" not in results[ds_sal]:
            continue
        bp = results[ds_sal]["LSTM"].get("best_params", {})
        if not bp:
            continue

        print(f"  Trenowanie LSTM {ds_sal} dla saliency...")

        # Sekwencje
        seq_feat = keep  # z wcześniejszego preprocessingu
        n_features = len(seq_feat)
        sl = bp.get("seq_length", 40)

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

        tr_sub = train_df[train_df["unit_id"].isin(uids[:split])]
        val_sub = train_df[train_df["unit_id"].isin(uids[split:])]
        Xtr, ytr = build_seqs(tr_sub, seq_feat, sl)
        Xv, yv = build_seqs(val_sub, seq_feat, sl)
        Xte, yte = build_test_seqs(test_df, seq_feat, sl)

        # Trenuj model
        torch.manual_seed(SEED)
        model = FlexLSTM(n_features, bp.get("hidden", 64),
                         bp.get("n_layers", 1), bp.get("dense", 32),
                         bp.get("dropout", 0.3)).to(device)

        Xt = torch.FloatTensor(Xtr).to(device)
        yt_t = torch.FloatTensor(ytr).to(device)
        loader = DataLoader(TensorDataset(Xt, yt_t), batch_size=128,
                            shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=bp.get("lr", 0.003))
        criterion = nn.MSELoss()

        for epoch in range(50):
            model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Gradient saliency na danych testowych
        # cuDNN nie pozwala na backward w eval mode — przenosimy na CPU
        model_cpu = model.cpu()
        model_cpu.eval()
        X_test_t = torch.FloatTensor(Xte)  # CPU
        X_test_t.requires_grad_(True)

        output = model_cpu(X_test_t)
        output.sum().backward()

        # |gradient| → ważność per timestep per sensor
        saliency = X_test_t.grad.abs().cpu().numpy()  # [n_test, seq_len, features]
        avg_saliency = saliency.mean(axis=0)  # [seq_len, features]

        # Heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(avg_saliency.T, aspect="auto", cmap="hot",
                       interpolation="nearest")
        ax.set_xlabel("Timestep (cykl w sekwencji)")
        ax.set_ylabel("Sensor")
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(seq_feat, fontsize=8)
        ax.set_title(f"Gradient Saliency — LSTM {ds_sal}\n"
                     f"(jaśniejsze = model zwraca większą uwagę)")
        plt.colorbar(im, ax=ax, label="|gradient|", shrink=0.8)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/66_saliency_heatmap_{ds_sal}.png",
                    bbox_inches="tight")
        plt.close()

        # Ważność per sensor (uśredniona po timestepach)
        sensor_importance = avg_saliency.mean(axis=0)
        sorted_idx = np.argsort(sensor_importance)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(n_features),
                sensor_importance[sorted_idx], color="#2E7D32", alpha=0.85)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([seq_feat[i] for i in sorted_idx], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Średni |gradient|")
        ax.set_title(f"Ważność sensorów wg LSTM ({ds_sal})")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/66_saliency_sensors_{ds_sal}.png",
                    bbox_inches="tight")
        plt.close()

        # Ważność per timestep (uśredniona po sensorach)
        timestep_importance = avg_saliency.mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(sl), timestep_importance, color="#2E7D32", alpha=0.85)
        ax.set_xlabel("Timestep (0 = najstarszy, {} = najnowszy)".format(sl - 1))
        ax.set_ylabel("Średni |gradient|")
        ax.set_title(f"Ważność timestepów wg LSTM ({ds_sal})\n"
                     f"(czy model patrzy na ostatnie cykle?)")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/66_saliency_timesteps_{ds_sal}.png",
                    bbox_inches="tight")
        plt.close()
        print(f"  [✓] 66_saliency_heatmap/sensors/timesteps_{ds_sal}.png")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. KRZYWE DEGRADACJI                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("7. Krzywe degradacji — predykcja RUL cykl po cyklu")
print(f"{'=' * 70}")

if data_path is not None and "LSTM" in results.get("FD001", {}):
    ds_deg = "FD001"
    bp = results[ds_deg]["LSTM"].get("best_params", {})
    sl = bp.get("seq_length", 40)

    # Predykcja cykl po cyklu — dla wybranych silników testowych
    # Potrzebujemy pełne sekwencje testowe (nie tylko ostatnią)
    test_df_deg = pd.read_csv(
        os.path.join(data_path, f"test_{ds_deg}.txt"),
        sep=r"\s+", header=None, names=COLUMNS)
    rul_df_deg = pd.read_csv(
        os.path.join(data_path, f"RUL_{ds_deg}.txt"),
        sep=r"\s+", header=None, names=["RUL"])

    mct = test_df_deg.groupby("unit_id")["cycle"].max().reset_index()
    mct.columns = ["unit_id", "max_cycle"]
    rul_df_deg["unit_id"] = range(1, len(rul_df_deg) + 1)
    mct = mct.merge(rul_df_deg, on="unit_id")
    mct["total_life"] = mct["max_cycle"] + mct["RUL"]
    test_df_deg = test_df_deg.merge(mct[["unit_id", "total_life"]],
                                     on="unit_id")
    test_df_deg["RUL"] = (test_df_deg["total_life"]
                          - test_df_deg["cycle"]).clip(upper=RUL_CLIP)
    test_df_deg.drop("total_life", axis=1, inplace=True)

    sensor_all = [f"sensor_{i}" for i in range(1, 22)]
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    var = test_df_deg[sensor_all].var()
    keep_deg = var[var > var.median() * 0.01].index.tolist()

    sc = MinMaxScaler()
    test_df_deg[keep_deg] = sc.fit_transform(test_df_deg[keep_deg])

    # Wybierz 6 silników z różnymi RUL
    test_ruls = test_df_deg.groupby("unit_id")["RUL"].min().sort_values()
    selected_units = list(test_ruls.iloc[::len(test_ruls) // 6].index[:6])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, uid in enumerate(selected_units):
        ax = axes[i]
        unit = test_df_deg[test_df_deg["unit_id"] == uid].sort_values("cycle")
        actual_rul = unit["RUL"].values
        cycles = unit["cycle"].values

        # Predykcja cykl po cyklu z LSTM
        data_seq = unit[keep_deg].values
        preds_per_cycle = []

        model.eval()
        model_deg = model.cpu()  # CPU dla predykcji cykl-po-cyklu
        for t in range(len(data_seq)):
            if t < sl:
                # Padding
                pad = np.zeros((sl - t - 1, len(keep_deg)))
                seq = np.vstack([pad, data_seq[:t + 1]])
            else:
                seq = data_seq[t - sl + 1:t + 1]

            seq_t = torch.FloatTensor(seq).unsqueeze(0)  # CPU
            with torch.no_grad():
                pred = model_deg(seq_t).cpu().item()
            preds_per_cycle.append(np.clip(pred, 0, RUL_CLIP))

        ax.plot(cycles, actual_rul, "b-", lw=2, label="Rzeczywisty RUL")
        ax.plot(cycles, preds_per_cycle, "r-", lw=1.5, alpha=0.8,
                label="Predykcja LSTM")
        ax.fill_between(cycles,
                        np.array(preds_per_cycle) - 15,
                        np.array(preds_per_cycle) + 15,
                        alpha=0.1, color="red")

        final_err = abs(preds_per_cycle[-1] - actual_rul[-1])
        ax.set_title(f"Silnik {uid} (RUL={actual_rul[-1]:.0f}, "
                     f"err={final_err:.0f})")
        ax.set_xlabel("Cykl")
        ax.set_ylabel("RUL")
        ax.legend(fontsize=8)

    plt.suptitle(f"Krzywe degradacji — LSTM {ds_deg}\n"
                 f"(predykcja RUL cykl po cyklu)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/67_degradation_curves.png", bbox_inches="tight")
    plt.close()
    print(f"  [✓] 67_degradation_curves.png")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'=' * 70}")
print("PODSUMOWANIE — Analiza i walidacja")
print(f"{'=' * 70}")
print(f"""
  Wygenerowane wykresy ({PLOT_DIR}/):
    61 — Accuracy Window (% w ±5/10/15/20 cykli)
    62 — Analiza rezyduów (trend błędu vs RUL)
    63 — Test Wilcoxona (istotność XGB vs LSTM)
    64 — Analiza kosztowa (wczesna wymiana vs awaria)
    65 — SHAP values (beeswarm + bar) — XGBoost
    66 — Gradient saliency (heatmap + sensory + timestepy) — LSTM
    67 — Krzywe degradacji (predykcja RUL cykl po cyklu)
""")