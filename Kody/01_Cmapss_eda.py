"""
=============================================================================
NASA C-MAPSS — Exploratory Data Analysis (EDA)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Dataset: Commercial Modular Aero-Propulsion System Simulation
  - 4 podzbiory: FD001–FD004
  - Kolumny: unit_id, cycle, 3 op_settings, 21 sensorów (26 total)
  - Dane run-to-failure z silników turbowentylatorowych
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. POBRANIE DANYCH                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("1. POBIERANIE DANYCH Z KAGGLE")
print("=" * 70)

# --- Opcja A: Kaggle API (domyślna) ---
path = None
try:
    import kagglehub
    path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    print(f"[✓] Dane pobrane z Kaggle: {path}")
except ImportError:
    print("[!] Brak modułu kagglehub. Instaluję automatycznie...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
    try:
        import kagglehub
        path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
        print(f"[✓] Dane pobrane z Kaggle: {path}")
    except Exception as e2:
        print(f"[!] Kaggle dalej nie działa: {e2}")
except Exception as e:
    print(f"[!] Błąd Kaggle: {e}")


def find_data_dir(root):
    """Szuka folderu zawierającego train_FD001.txt (rekurencyjnie)."""
    if not root or not os.path.exists(root):
        return None
    # Może pliki są bezpośrednio w root
    if os.path.isfile(os.path.join(root, "train_FD001.txt")):
        return root
    # Szukaj w podfolderach (max 3 poziomy)
    for dirpath, dirnames, filenames in os.walk(root):
        if "train_FD001.txt" in filenames:
            return dirpath
    return None


# Sprawdź czy pliki są w pobranej ścieżce (mogą być w podfolderze!)
actual_path = find_data_dir(path)

# Jeśli nie — szukaj lokalnie
if actual_path is None:
    candidates = [
        "./data",
        "./CMAPSSData",
        "./CMaps",
        os.path.join(os.path.dirname(__file__), "data"),
        os.path.join(os.path.dirname(__file__), "CMAPSSData"),
    ]
    for candidate in candidates:
        found = find_data_dir(candidate)
        if found:
            actual_path = found
            print(f"[✓] Znaleziono dane lokalnie: {actual_path}")
            break

if actual_path is None:
    print("\n" + "=" * 60)
    print("BŁĄD: Nie znaleziono danych C-MAPSS!")
    print("=" * 60)
    print("Rozwiązanie — wybierz jedno z poniższych:")
    print()
    print("  1) Zainstaluj kagglehub:")
    print("     pip install kagglehub")
    print("     (przy pierwszym użyciu potrzebujesz API token z kaggle.com)")
    print()
    print("  2) Pobierz ręcznie z:")
    print("     https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    print("     i rozpakuj pliki do folderu 'data/' obok tego skryptu")
    print()
    sys.exit(1)

path = actual_path
print(f"Ścieżka do plików: {path}\n")

# Wylistuj pliki
files = sorted([f for f in os.listdir(path) if not f.startswith('.')])
print("Dostępne pliki:")
for f in files:
    size_kb = os.path.getsize(os.path.join(path, f)) / 1024
    print(f"  {f:30s} ({size_kb:8.1f} KB)")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. ŁADOWANIE DANYCH + DEFINICJA KOLUMN                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("2. ŁADOWANIE DANYCH")
print("=" * 70)

# Nazwy kolumn (brak nagłówków w plikach .txt)
COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Informacje o podzbiorach
DATASET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": 1, "desc": "1 warunek, 1 tryb uszkodzenia (HPC)"},
    "FD002": {"conditions": 6, "fault_modes": 1, "desc": "6 warunków, 1 tryb uszkodzenia (HPC)"},
    "FD003": {"conditions": 1, "fault_modes": 2, "desc": "1 warunek, 2 tryby uszkodzenia (HPC + Fan)"},
    "FD004": {"conditions": 6, "fault_modes": 2, "desc": "6 warunków, 2 tryby uszkodzenia (HPC + Fan)"},
}


def load_dataset(data_path, dataset_id):
    """Ładuje train, test i RUL dla danego podzbioru."""
    train = pd.read_csv(
        os.path.join(data_path, f"train_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=COLUMNS
    )
    test = pd.read_csv(
        os.path.join(data_path, f"test_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=COLUMNS
    )
    rul = pd.read_csv(
        os.path.join(data_path, f"RUL_{dataset_id}.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )
    return train, test, rul


def add_rul_column(df):
    """Oblicza RUL = max_cycle - current_cycle dla zbioru treningowego."""
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)
    return df


# Ładowanie wszystkich podzbiorów
datasets = {}
for ds_id in ["FD001", "FD002", "FD003", "FD004"]:
    train, test, rul = load_dataset(path, ds_id)
    train = add_rul_column(train)
    datasets[ds_id] = {"train": train, "test": test, "rul": rul}
    n_train_units = train["unit_id"].nunique()
    n_test_units = test["unit_id"].nunique()
    print(f"\n{ds_id} — {DATASET_INFO[ds_id]['desc']}")
    print(f"  Train: {len(train):>6,} wierszy, {n_train_units:>3} silników")
    print(f"  Test:  {len(test):>6,} wierszy, {n_test_units:>3} silników")
    print(f"  RUL:   {len(rul):>6,} wartości")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. ANALIZA STRUKTURY DANYCH (focus: FD001)                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("3. STRUKTURA DANYCH — FD001")
print("=" * 70)

train_fd001 = datasets["FD001"]["train"]

print("\n--- Pierwsze 5 wierszy ---")
print(train_fd001.head().to_string())

print("\n--- Info o typach danych ---")
print(train_fd001.dtypes.value_counts().to_string())
print(f"\nLiczba kolumn: {len(train_fd001.columns)}")
print(f"Kształt:       {train_fd001.shape}")

print("\n--- Statystyki opisowe ---")
print(train_fd001.describe().round(3).to_string())

print("\n--- Brakujące wartości ---")
missing = train_fd001.isnull().sum()
if missing.sum() == 0:
    print("Brak brakujących wartości — dataset jest kompletny!")
else:
    print(missing[missing > 0])

print("\n--- Duplikaty ---")
n_dupes = train_fd001.duplicated().sum()
print(f"Liczba zduplikowanych wierszy: {n_dupes}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. IDENTYFIKACJA CZUJNIKÓW O STAŁEJ / NISKIEJ ZMIENNOŚCI               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("4. ANALIZA ZMIENNOŚCI CZUJNIKÓW")
print("=" * 70)

sensor_cols = [c for c in train_fd001.columns if c.startswith("sensor_")]
sensor_stats = train_fd001[sensor_cols].describe().T
sensor_stats["cv"] = sensor_stats["std"] / sensor_stats["mean"].abs()  # coeff. of variation
sensor_stats["range"] = sensor_stats["max"] - sensor_stats["min"]

# Czujniki o (prawie) zerowej zmienności
low_var_sensors = sensor_stats[sensor_stats["std"] < 1e-6].index.tolist()
near_const_sensors = sensor_stats[
    (sensor_stats["std"] >= 1e-6) & (sensor_stats["cv"].abs() < 0.001)
].index.tolist()

print(f"\nCzujniki o zerowej zmienności (std ≈ 0):       {low_var_sensors or 'brak'}")
print(f"Czujniki o bardzo niskiej zmienności (CV < 0.1%): {near_const_sensors or 'brak'}")

useful_sensors = [s for s in sensor_cols if s not in low_var_sensors + near_const_sensors]
print(f"\nPrzydatne czujniki ({len(useful_sensors)}): {useful_sensors}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. WIZUALIZACJE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("5. GENEROWANIE WIZUALIZACJI")
print("=" * 70)

OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# 5.1 Rozkład RUL (Remaining Useful Life) — FD001
# ────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram żywotności silników (max cycle per unit)
lifetimes = train_fd001.groupby("unit_id")["cycle"].max()
axes[0].hist(lifetimes, bins=25, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].axvline(lifetimes.mean(), color="#FF5722", linestyle="--", lw=2,
                label=f"Średnia = {lifetimes.mean():.0f} cykli")
axes[0].axvline(lifetimes.median(), color="#4CAF50", linestyle="--", lw=2,
                label=f"Mediana = {lifetimes.median():.0f} cykli")
axes[0].set_xlabel("Całkowita żywotność (cykle)")
axes[0].set_ylabel("Liczba silników")
axes[0].set_title("FD001 — Rozkład żywotności silników")
axes[0].legend()

# Porównanie rozkładu żywotności FD001–FD004
all_lifetimes = {}
for ds_id in ["FD001", "FD002", "FD003", "FD004"]:
    lt = datasets[ds_id]["train"].groupby("unit_id")["cycle"].max()
    all_lifetimes[ds_id] = lt

bp = axes[1].boxplot(
    [all_lifetimes[ds] for ds in ["FD001", "FD002", "FD003", "FD004"]],
    labels=["FD001", "FD002", "FD003", "FD004"],
    patch_artist=True,
    boxprops=dict(facecolor="#E3F2FD", edgecolor="#1565C0"),
    medianprops=dict(color="#FF5722", lw=2),
)
axes[1].set_ylabel("Żywotność (cykle)")
axes[1].set_title("Porównanie żywotności silników — FD001–FD004")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_lifetime_distribution.png", bbox_inches="tight")
plt.close()
print("  [✓] 01_lifetime_distribution.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.2 Przebiegi czujników — wybrane silniki
# ────────────────────────────────────────────────────────────────────────────
# Wybieramy 6 najbardziej informatywnych czujników
top_sensors = useful_sensors[:6] if len(useful_sensors) >= 6 else useful_sensors
sample_units = [1, 25, 50, 75, 100]  # kilka przykładowych silników

fig, axes = plt.subplots(len(top_sensors), 1, figsize=(14, 3 * len(top_sensors)), sharex=False)
colors = plt.cm.tab10(np.linspace(0, 1, len(sample_units)))

for i, sensor in enumerate(top_sensors):
    ax = axes[i]
    for j, uid in enumerate(sample_units):
        unit_data = train_fd001[train_fd001["unit_id"] == uid]
        ax.plot(unit_data["cycle"], unit_data[sensor],
                alpha=0.7, lw=1.2, color=colors[j], label=f"Unit {uid}")
    ax.set_ylabel(sensor.replace("_", " ").title())
    ax.set_title(f"{sensor} — degradacja w czasie")
    if i == 0:
        ax.legend(loc="upper left", fontsize=8, ncol=len(sample_units))
    if i == len(top_sensors) - 1:
        ax.set_xlabel("Cykl operacyjny")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_sensor_trajectories.png", bbox_inches="tight")
plt.close()
print("  [✓] 02_sensor_trajectories.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.3 Macierz korelacji czujników
# ────────────────────────────────────────────────────────────────────────────
corr_matrix = train_fd001[useful_sensors].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, ax=ax,
    xticklabels=[s.replace("sensor_", "s") for s in useful_sensors],
    yticklabels=[s.replace("sensor_", "s") for s in useful_sensors],
    annot_kws={"size": 7},
)
ax.set_title("Macierz korelacji czujników (FD001) — tylko przydatne sensory")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("  [✓] 03_correlation_matrix.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.4 Korelacja czujników z RUL
# ────────────────────────────────────────────────────────────────────────────
rul_corr = train_fd001[useful_sensors + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["#EF5350" if v < 0 else "#42A5F5" for v in rul_corr.values]
rul_corr.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="white")
ax.set_xlabel("Korelacja Pearsona z RUL")
ax.set_title("Korelacja czujników z Remaining Useful Life (FD001)")
ax.axvline(0, color="black", lw=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_sensor_rul_correlation.png", bbox_inches="tight")
plt.close()
print("  [✓] 04_sensor_rul_correlation.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.5 Rozkład wartości czujników (violin plots)
# ────────────────────────────────────────────────────────────────────────────
# Normalizacja dla lepszej wizualizacji
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_sensors = pd.DataFrame(
    scaler.fit_transform(train_fd001[useful_sensors]),
    columns=useful_sensors
)

fig, ax = plt.subplots(figsize=(14, 6))
scaled_sensors_melted = scaled_sensors.melt(var_name="sensor", value_name="value")
scaled_sensors_melted["sensor"] = scaled_sensors_melted["sensor"].str.replace("sensor_", "s")
sns.boxplot(data=scaled_sensors_melted, x="sensor", y="value", ax=ax,
            fliersize=1, color="#90CAF9")
ax.set_title("Rozkład znormalizowanych wartości czujników (FD001)")
ax.set_xlabel("Czujnik")
ax.set_ylabel("Wartość (z-score)")
ax.set_ylim(-5, 5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_sensor_distributions.png", bbox_inches="tight")
plt.close()
print("  [✓] 05_sensor_distributions.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.6 Ustawienia operacyjne
# ────────────────────────────────────────────────────────────────────────────
op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(op_cols):
    axes[i].hist(train_fd001[col], bins=50, color="#7E57C2", edgecolor="white", alpha=0.8)
    axes[i].set_title(f"{col}")
    axes[i].set_xlabel("Wartość")
    axes[i].set_ylabel("Częstość")
plt.suptitle("Rozkład ustawień operacyjnych — FD001", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_operational_settings.png", bbox_inches="tight")
plt.close()
print("  [✓] 06_operational_settings.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.7 RUL z clippingiem — porównanie
# ────────────────────────────────────────────────────────────────────────────
RUL_CLIP = 125

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Oryginalny RUL
axes[0].hist(train_fd001["RUL"], bins=40, color="#26A69A", edgecolor="white", alpha=0.85)
axes[0].set_title("Oryginalny RUL")
axes[0].set_xlabel("RUL (cykle)")
axes[0].set_ylabel("Częstość")

# Clipped RUL
rul_clipped = train_fd001["RUL"].clip(upper=RUL_CLIP)
axes[1].hist(rul_clipped, bins=40, color="#FF7043", edgecolor="white", alpha=0.85)
axes[1].axvline(RUL_CLIP, color="black", linestyle="--", lw=1.5,
                label=f"Clip @ {RUL_CLIP}")
axes[1].set_title(f"RUL z clippingiem (max={RUL_CLIP})")
axes[1].set_xlabel("RUL (cykle)")
axes[1].set_ylabel("Częstość")
axes[1].legend()

plt.suptitle("Piece-wise linear RUL — motywacja do clippingu", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_rul_clipping.png", bbox_inches="tight")
plt.close()
print("  [✓] 07_rul_clipping.png")

# ────────────────────────────────────────────────────────────────────────────
# 5.8 Heatmapa degradacji — pojedynczy silnik
# ────────────────────────────────────────────────────────────────────────────
unit_example = train_fd001[train_fd001["unit_id"] == 1]
sensor_data_norm = (unit_example[useful_sensors] - unit_example[useful_sensors].mean()) / unit_example[useful_sensors].std()

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    sensor_data_norm.T, cmap="RdYlBu_r", ax=ax,
    xticklabels=10, yticklabels=[s.replace("sensor_", "s") for s in useful_sensors],
    cbar_kws={"label": "Z-score"},
)
ax.set_xlabel("Cykl operacyjny")
ax.set_title("Heatmapa degradacji — Unit 1, FD001 (znormalizowane czujniki)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_degradation_heatmap.png", bbox_inches="tight")
plt.close()
print("  [✓] 08_degradation_heatmap.png")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. PODSUMOWANIE KLUCZOWYCH WNIOSKÓW                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("6. PODSUMOWANIE EDA")
print("=" * 70)

# Top 5 czujników najbardziej skorelowanych z RUL (bezwzględnie)
top5_sensors = rul_corr.abs().sort_values(ascending=False).head(5)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  KLUCZOWE WNIOSKI — FD001                                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Liczba silników (train):  {train_fd001['unit_id'].nunique():>4}                                  ║
║  Liczba wierszy (train):   {len(train_fd001):>6,}                                ║
║  Liczba kolumn:            {len(COLUMNS):>4}                                  ║
║  Brakujące wartości:       {'BRAK':>6}                                ║
║                                                                      ║
║  Średnia żywotność:        {lifetimes.mean():>6.1f} cykli                        ║
║  Min/Max żywotność:        {lifetimes.min():>3}/{lifetimes.max():<3} cykli                         ║
║                                                                      ║
║  Czujniki stałe (do usunięcia): {len(low_var_sensors) + len(near_const_sensors):>2}                             ║
║  Czujniki informatywne:         {len(useful_sensors):>2}                             ║
║                                                                      ║
║  Top 5 czujników (korelacja z RUL):                                  ║""")
for sensor, corr_val in top5_sensors.items():
    real_corr = rul_corr[sensor]
    print(f"║    {sensor:>12s}: r = {real_corr:+.3f}                                  ║")
print(f"""║                                                                      ║
║  Rekomendacja: clip RUL @ {RUL_CLIP} cykli (piece-wise linear)         ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print(f"\nWszystkie wykresy zapisane w: {OUTPUT_DIR}/")
print("Gotowe! Następny krok: preprocessing + budowa modelu.")