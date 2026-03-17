"""
=============================================================================
NASA C-MAPSS — Preprocessing (Krok 2)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Przygotowuje dane dla 3 modeli:
  1. XGBoost (baseline)  — ręczne feature engineering → tabela 2D
  2. LSTM                — sliding window → sekwencje 3D
  3. CNN-LSTM (hybrid)   — sliding window → sekwencje 3D
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. ŁADOWANIE DANYCH                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("PREPROCESSING — NASA C-MAPSS (FD001)")
print("=" * 70)

# --- Nazwy kolumn ---
COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# --- Parametry preprocessingu ---
RUL_CLIP = 125          # Piece-wise linear RUL cap
SEQUENCE_LENGTH = 30    # Długość okna czasowego dla LSTM / CNN-LSTM
DATASET_ID = "FD001"    # Podzbiór (zmień na FD002-FD004 jeśli chcesz)


def find_data_dir(root):
    """Szuka folderu zawierającego train_FD001.txt (rekurencyjnie)."""
    if not root or not os.path.exists(root):
        return None
    if os.path.isfile(os.path.join(root, f"train_{DATASET_ID}.txt")):
        return root
    for dirpath, dirnames, filenames in os.walk(root):
        if f"train_{DATASET_ID}.txt" in filenames:
            return dirpath
    return None


# Szukaj danych
path = None
try:
    import kagglehub
    path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
except ImportError:
    print("[!] Brak kagglehub, instaluję...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
    try:
        import kagglehub
        path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    except Exception as e:
        print(f"[!] Kaggle nie działa: {e}")
except Exception as e:
    print(f"[!] Błąd: {e}")

# Szukaj rekurencyjnie
actual_path = find_data_dir(path)
if actual_path is None:
    for candidate in ["./data", "./CMAPSSData", "./CMaps"]:
        found = find_data_dir(candidate)
        if found:
            actual_path = found
            break

if actual_path is None:
    print("BŁĄD: Nie znaleziono danych! Pobierz z Kaggle lub umieść w ./data/")
    sys.exit(1)

path = actual_path
print(f"[✓] Dane znalezione: {path}")

# Wczytanie
train_df = pd.read_csv(
    os.path.join(path, f"train_{DATASET_ID}.txt"),
    sep=r"\s+", header=None, names=COLUMNS
)
test_df = pd.read_csv(
    os.path.join(path, f"test_{DATASET_ID}.txt"),
    sep=r"\s+", header=None, names=COLUMNS
)
rul_df = pd.read_csv(
    os.path.join(path, f"RUL_{DATASET_ID}.txt"),
    sep=r"\s+", header=None, names=["RUL"]
)

print(f"[✓] Train: {train_df.shape}, Test: {test_df.shape}, RUL: {rul_df.shape}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. OBLICZENIE RUL                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 2. Obliczanie RUL ---")

# TRAIN: RUL = max_cycle - current_cycle
max_cycles_train = train_df.groupby("unit_id")["cycle"].max().reset_index()
max_cycles_train.columns = ["unit_id", "max_cycle"]
train_df = train_df.merge(max_cycles_train, on="unit_id")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
train_df.drop("max_cycle", axis=1, inplace=True)

# TEST: RUL dla ostatniego cyklu pochodzi z pliku RUL_FD001.txt
# Dla wcześniejszych cykli: RUL_last + (max_cycle_in_test - current_cycle)
max_cycles_test = test_df.groupby("unit_id")["cycle"].max().reset_index()
max_cycles_test.columns = ["unit_id", "max_cycle"]
rul_df["unit_id"] = range(1, len(rul_df) + 1)
max_cycles_test = max_cycles_test.merge(rul_df, on="unit_id")
# RUL_last = rul z pliku, więc prawdziwy total_life = max_cycle + RUL_last
max_cycles_test["total_life"] = max_cycles_test["max_cycle"] + max_cycles_test["RUL"]

test_df = test_df.merge(max_cycles_test[["unit_id", "total_life"]], on="unit_id")
test_df["RUL"] = test_df["total_life"] - test_df["cycle"]
test_df.drop("total_life", axis=1, inplace=True)

# Clipping RUL
train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CLIP)
test_df["RUL"] = test_df["RUL"].clip(upper=RUL_CLIP)

print(f"  Train RUL: min={train_df['RUL'].min()}, max={train_df['RUL'].max()}")
print(f"  Test  RUL: min={test_df['RUL'].min()}, max={test_df['RUL'].max()}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. SELEKCJA CECH (na podstawie EDA)                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 3. Selekcja cech ---")

# Czujniki stałe / niemal stałe w FD001 (z EDA)
DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

# Op settings w FD001 są praktycznie stałe
DROP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Redundantne czujniki (korelacja s9-s14 = 0.96, s20-s21 = 0.69)
DROP_REDUNDANT = ["sensor_14"]  # zachowujemy s9, usuwamy s14

DROP_COLS = DROP_SENSORS + DROP_SETTINGS + DROP_REDUNDANT

# Zachowane cechy
SENSOR_FEATURES = [c for c in train_df.columns
                   if c.startswith("sensor_") and c not in DROP_COLS]

print(f"  Usunięte kolumny ({len(DROP_COLS)}): {DROP_COLS}")
print(f"  Zachowane sensory ({len(SENSOR_FEATURES)}): {SENSOR_FEATURES}")

# Usunięcie zbędnych kolumn
train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns],
              inplace=True, errors="ignore")
test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns],
             inplace=True, errors="ignore")

# Dodanie czujników z niską zmiennością ale potencjalnie przydatnych
# sensor_2, sensor_8, sensor_13 — niska zmienność ale nie zerowa
# Zostawiamy je na wszelki wypadek, model sam zdecyduje
DROP_LOW_VAR = ["sensor_2", "sensor_8", "sensor_13"]
train_df.drop(columns=[c for c in DROP_LOW_VAR if c in train_df.columns],
              inplace=True, errors="ignore")
test_df.drop(columns=[c for c in DROP_LOW_VAR if c in test_df.columns],
             inplace=True, errors="ignore")

SENSOR_FEATURES = [c for c in SENSOR_FEATURES if c not in DROP_LOW_VAR]
print(f"  Finalne sensory ({len(SENSOR_FEATURES)}): {SENSOR_FEATURES}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. NORMALIZACJA (MinMaxScaler)                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 4. Normalizacja ---")

scaler = MinMaxScaler(feature_range=(0, 1))

# WAŻNE: fit TYLKO na train, transform na obu!
train_df[SENSOR_FEATURES] = scaler.fit_transform(train_df[SENSOR_FEATURES])
test_df[SENSOR_FEATURES] = scaler.transform(test_df[SENSOR_FEATURES])

print(f"  MinMaxScaler fitted na train ({len(SENSOR_FEATURES)} features)")
print(f"  Zakres po normalizacji: [{train_df[SENSOR_FEATURES].min().min():.3f}, "
      f"{train_df[SENSOR_FEATURES].max().max():.3f}]")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5A. FEATURE ENGINEERING — XGBoost (tabela 2D)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 5A. Feature engineering dla XGBoost ---")


def create_handcrafted_features(df, sensor_cols, window_sizes=[5, 10, 20]):
    """
    Tworzy ręczne cechy z okien czasowych:
    - Średnia, std, min, max, trend (nachylenie regresji liniowej)
    - Dla każdego okna i każdego sensora
    - Plus: aktualny cykl, aktualne wartości sensorów
    """
    result_frames = []

    for uid in df["unit_id"].unique():
        unit = df[df["unit_id"] == uid].sort_values("cycle").copy()

        for w in window_sizes:
            for s in sensor_cols:
                # Rolling statistics
                unit[f"{s}_mean_{w}"] = unit[s].rolling(w, min_periods=1).mean()
                unit[f"{s}_std_{w}"] = unit[s].rolling(w, min_periods=1).std().fillna(0)
                unit[f"{s}_min_{w}"] = unit[s].rolling(w, min_periods=1).min()
                unit[f"{s}_max_{w}"] = unit[s].rolling(w, min_periods=1).max()

                # Trend (różnica między końcem a początkiem okna)
                unit[f"{s}_trend_{w}"] = unit[s].diff(w).fillna(0)

        # Dodaj cycle jako cechę (znormalizowany)
        result_frames.append(unit)

    result = pd.concat(result_frames, ignore_index=True)
    return result


train_xgb = create_handcrafted_features(train_df, SENSOR_FEATURES)
test_xgb = create_handcrafted_features(test_df, SENSOR_FEATURES)

# Cechy do modelu XGBoost (wszystko oprócz unit_id, cycle, RUL)
XGB_FEATURE_COLS = [c for c in train_xgb.columns
                    if c not in ["unit_id", "cycle", "RUL"]]

X_train_xgb = train_xgb[XGB_FEATURE_COLS].values
y_train_xgb = train_xgb["RUL"].values

# Dla testu XGBoost — ewaluujemy na OSTATNIM cyklu każdego silnika
test_xgb_last = test_xgb.groupby("unit_id").last().reset_index()
X_test_xgb = test_xgb_last[XGB_FEATURE_COLS].values
y_test_xgb = test_xgb_last["RUL"].values

print(f"  Train: X={X_train_xgb.shape}, y={y_train_xgb.shape}")
print(f"  Test (ostatni cykl): X={X_test_xgb.shape}, y={y_test_xgb.shape}")
print(f"  Liczba cech: {len(XGB_FEATURE_COLS)}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5B. SLIDING WINDOW — LSTM / CNN-LSTM (sekwencje 3D)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n--- 5B. Tworzenie sekwencji (window={SEQUENCE_LENGTH}) ---")


def create_sequences(df, sensor_cols, sequence_length):
    """
    Tworzy sekwencje 3D: [samples, timesteps, features]
    Dla każdego silnika: sliding window po cyklach.
    Label = RUL ostatniego cyklu w oknie.
    """
    X, y = [], []

    for uid in df["unit_id"].unique():
        unit = df[df["unit_id"] == uid].sort_values("cycle")
        data = unit[sensor_cols].values
        labels = unit["RUL"].values

        # Jeśli silnik ma mniej cykli niż okno — padding zerami
        if len(data) < sequence_length:
            padding = np.zeros((sequence_length - len(data), len(sensor_cols)))
            data = np.vstack([padding, data])
            labels = np.concatenate([
                np.full(sequence_length - len(labels), labels[0]),
                labels
            ])

        # Sliding window
        for i in range(len(data) - sequence_length + 1):
            X.append(data[i : i + sequence_length])
            y.append(labels[i + sequence_length - 1])

    return np.array(X), np.array(y)


def create_test_sequences(df, sensor_cols, sequence_length):
    """
    Dla testu: tworzy JEDNĄ sekwencję na silnik
    (ostatnie sequence_length cykli) — bo ewaluujemy RUL na końcu.
    """
    X, y = [], []

    for uid in df["unit_id"].unique():
        unit = df[df["unit_id"] == uid].sort_values("cycle")
        data = unit[sensor_cols].values
        rul = unit["RUL"].values[-1]  # RUL ostatniego cyklu

        # Padding jeśli za krótkie
        if len(data) < sequence_length:
            padding = np.zeros((sequence_length - len(data), len(sensor_cols)))
            data = np.vstack([padding, data])

        # Bierzemy ostatnie sequence_length kroków
        X.append(data[-sequence_length:])
        y.append(rul)

    return np.array(X), np.array(y)


# Tworzenie sekwencji
X_train_seq, y_train_seq = create_sequences(
    train_df, SENSOR_FEATURES, SEQUENCE_LENGTH
)
X_test_seq, y_test_seq = create_test_sequences(
    test_df, SENSOR_FEATURES, SEQUENCE_LENGTH
)

print(f"  Train sequences: X={X_train_seq.shape}, y={y_train_seq.shape}")
print(f"  Test sequences:  X={X_test_seq.shape}, y={y_test_seq.shape}")
print(f"  Kształt: [samples, timesteps={SEQUENCE_LENGTH}, features={len(SENSOR_FEATURES)}]")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. WALIDACJA — podział train na train/val                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 6. Podział train/validation ---")

# Dzielimy po unit_id (nie losowo!) żeby nie mieszać danych z tego samego silnika
unit_ids = train_df["unit_id"].unique()
np.random.seed(42)
np.random.shuffle(unit_ids)

val_ratio = 0.2
split_idx = int(len(unit_ids) * (1 - val_ratio))
train_units = unit_ids[:split_idx]
val_units = unit_ids[split_idx:]

print(f"  Train units: {len(train_units)}, Val units: {len(val_units)}")

# --- XGBoost split ---
train_mask_xgb = train_xgb["unit_id"].isin(train_units)
val_mask_xgb = train_xgb["unit_id"].isin(val_units)

X_train_xgb_split = train_xgb[train_mask_xgb][XGB_FEATURE_COLS].values
y_train_xgb_split = train_xgb[train_mask_xgb]["RUL"].values
X_val_xgb = train_xgb[val_mask_xgb][XGB_FEATURE_COLS].values
y_val_xgb = train_xgb[val_mask_xgb]["RUL"].values

print(f"  XGBoost — train: {X_train_xgb_split.shape}, val: {X_val_xgb.shape}")

# --- Sequence split (LSTM / CNN-LSTM) ---
# Musimy przebudować sekwencje oddzielnie dla train i val units
train_subset = train_df[train_df["unit_id"].isin(train_units)]
val_subset = train_df[train_df["unit_id"].isin(val_units)]

X_train_seq_split, y_train_seq_split = create_sequences(
    train_subset, SENSOR_FEATURES, SEQUENCE_LENGTH
)
X_val_seq, y_val_seq = create_sequences(
    val_subset, SENSOR_FEATURES, SEQUENCE_LENGTH
)

print(f"  Sequences — train: {X_train_seq_split.shape}, val: {X_val_seq.shape}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. ZAPIS PRZETWORZONYCH DANYCH                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 7. Zapis danych ---")

OUTPUT_DIR = "./preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Zapis do .npz (NumPy compressed)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "xgboost_data.npz"),
    X_train=X_train_xgb_split,
    y_train=y_train_xgb_split,
    X_val=X_val_xgb,
    y_val=y_val_xgb,
    X_test=X_test_xgb,
    y_test=y_test_xgb,
    feature_names=np.array(XGB_FEATURE_COLS),
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "sequence_data.npz"),
    X_train=X_train_seq_split,
    y_train=y_train_seq_split,
    X_val=X_val_seq,
    y_val=y_val_seq,
    X_test=X_test_seq,
    y_test=y_test_seq,
    sensor_names=np.array(SENSOR_FEATURES),
)

# Zapis scalera (potrzebny do inference na nowych danych)
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Zapis parametrów preprocessingu
params = {
    "dataset_id": DATASET_ID,
    "rul_clip": RUL_CLIP,
    "sequence_length": SEQUENCE_LENGTH,
    "sensor_features": SENSOR_FEATURES,
    "n_features": len(SENSOR_FEATURES),
    "drop_cols": DROP_COLS + DROP_LOW_VAR,
    "train_units": train_units.tolist(),
    "val_units": val_units.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "params.pkl"), "wb") as f:
    pickle.dump(params, f)

print(f"  [✓] xgboost_data.npz")
print(f"  [✓] sequence_data.npz")
print(f"  [✓] scaler.pkl")
print(f"  [✓] params.pkl")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8. WIZUALIZACJA PRZETWORZONYCH DANYCH                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 8. Wizualizacje preprocessingu ---")

PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# 8.1 Przykładowa sekwencja (input do LSTM)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Losowa sekwencja z train
sample_idx = 500
sample_seq = X_train_seq[sample_idx]  # shape: [30, n_features]

im = axes[0].imshow(sample_seq.T, aspect="auto", cmap="viridis",
                     interpolation="nearest")
axes[0].set_ylabel("Feature index")
axes[0].set_xlabel("Timestep (cykl w oknie)")
axes[0].set_title(f"Przykładowa sekwencja wejściowa (sample #{sample_idx}, "
                   f"RUL={y_train_seq[sample_idx]:.0f})")
axes[0].set_yticks(range(len(SENSOR_FEATURES)))
axes[0].set_yticklabels([s.replace("sensor_", "s") for s in SENSOR_FEATURES],
                         fontsize=8)
plt.colorbar(im, ax=axes[0], label="Wartość (znormalizowana)")

# Sekwencja bliżej awarii
late_idx = np.argmin(np.abs(y_train_seq - 5))  # RUL ≈ 5
late_seq = X_train_seq[late_idx]

im2 = axes[1].imshow(late_seq.T, aspect="auto", cmap="viridis",
                      interpolation="nearest")
axes[1].set_ylabel("Feature index")
axes[1].set_xlabel("Timestep (cykl w oknie)")
axes[1].set_title(f"Sekwencja tuż przed awarią (sample #{late_idx}, "
                   f"RUL={y_train_seq[late_idx]:.0f})")
axes[1].set_yticks(range(len(SENSOR_FEATURES)))
axes[1].set_yticklabels([s.replace("sensor_", "s") for s in SENSOR_FEATURES],
                         fontsize=8)
plt.colorbar(im2, ax=axes[1], label="Wartość (znormalizowana)")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_sequence_examples.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 09_sequence_examples.png")

# 8.2 Rozkład RUL w train/val/test po preprocessingu
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_train_seq_split, bins=40, alpha=0.6, label=f"Train ({len(y_train_seq_split):,})",
        color="#2196F3", edgecolor="white")
ax.hist(y_val_seq, bins=40, alpha=0.6, label=f"Val ({len(y_val_seq):,})",
        color="#FF9800", edgecolor="white")
ax.hist(y_test_seq, bins=20, alpha=0.6, label=f"Test ({len(y_test_seq):,})",
        color="#4CAF50", edgecolor="white")
ax.set_xlabel("RUL (clipped)")
ax.set_ylabel("Częstość")
ax.set_title(f"Rozkład RUL po preprocessingu — train/val/test (clip={RUL_CLIP})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/10_rul_train_val_test.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 10_rul_train_val_test.png")

# 8.3 Znormalizowane przebiegi czujników — przykładowy silnik
fig, axes = plt.subplots(len(SENSOR_FEATURES), 1,
                          figsize=(14, 2.5 * len(SENSOR_FEATURES)), sharex=True)

unit_example = train_df[train_df["unit_id"] == 1].sort_values("cycle")
for i, sensor in enumerate(SENSOR_FEATURES):
    ax = axes[i]
    ax.plot(unit_example["cycle"], unit_example[sensor], lw=1.2, color="#1565C0")
    ax.set_ylabel(sensor.replace("sensor_", "s"), fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    if i == 0:
        ax.set_title("Znormalizowane przebiegi — Unit 1 (po preprocessingu)")
    if i == len(SENSOR_FEATURES) - 1:
        ax.set_xlabel("Cykl operacyjny")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/11_normalized_sensors.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 11_normalized_sensors.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("PODSUMOWANIE PREPROCESSINGU")
print("=" * 70)
print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  Dataset:           {DATASET_ID}                                      ║
║  RUL clip:          {RUL_CLIP}                                       ║
║  Sequence length:   {SEQUENCE_LENGTH}                                        ║
║  Features:          {len(SENSOR_FEATURES)} sensorów                              ║
║                                                                   ║
║  XGBoost:                                                         ║
║    Train:  {X_train_xgb_split.shape[0]:>6,} samples × {X_train_xgb_split.shape[1]:>3} features             ║
║    Val:    {X_val_xgb.shape[0]:>6,} samples × {X_val_xgb.shape[1]:>3} features             ║
║    Test:   {X_test_xgb.shape[0]:>6,} samples × {X_test_xgb.shape[1]:>3} features             ║
║                                                                   ║
║  LSTM / CNN-LSTM:                                                 ║
║    Train:  {X_train_seq_split.shape[0]:>6,} sequences × [{SEQUENCE_LENGTH}, {len(SENSOR_FEATURES)}]           ║
║    Val:    {X_val_seq.shape[0]:>6,} sequences × [{SEQUENCE_LENGTH}, {len(SENSOR_FEATURES)}]           ║
║    Test:   {X_test_seq.shape[0]:>6,} sequences × [{SEQUENCE_LENGTH}, {len(SENSOR_FEATURES)}]           ║
║                                                                   ║
║  Pliki w ./preprocessed/:                                         ║
║    xgboost_data.npz  — dane 2D dla XGBoost                       ║
║    sequence_data.npz — sekwencje 3D dla LSTM/CNN-LSTM             ║
║    scaler.pkl        — MinMaxScaler (do inference)                ║
║    params.pkl        — parametry preprocessingu                   ║
╚═══════════════════════════════════════════════════════════════════╝

Następny krok: python 03_model_xgboost.py
""")