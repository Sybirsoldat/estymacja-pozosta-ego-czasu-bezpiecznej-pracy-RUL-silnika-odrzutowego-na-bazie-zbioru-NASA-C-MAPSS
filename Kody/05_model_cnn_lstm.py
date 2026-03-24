"""
=============================================================================
NASA C-MAPSS — Model 3: CNN-LSTM Hybrid (Krok 5)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Model hybrydowy — CNN (ekstrakcja lokalnych wzorców) + LSTM (sekwencja)
Conv1D wyciąga krótkoterminowe wzorce z sąsiednich cykli,
LSTM modeluje ich zmiany w czasie → predykcja RUL
Ewaluacja: RMSE + MAE + NASA Scoring Function + R²
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  0. INSTALACJA PyTorch (jeśli brak)                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[!] Instaluję PyTorch (CPU)...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "torch", "-q"
    ])
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. ŁADOWANIE DANYCH SEKWENCYJNYCH                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("MODEL 3: CNN-LSTM Hybrid — NASA C-MAPSS (FD001)")
print("=" * 70)

PREPROCESSED_DIR = "./preprocessed"
PLOT_DIR = "./plots"
MODELS_DIR = "./models"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Sprawdź czy dane istnieją
if not os.path.exists(os.path.join(PREPROCESSED_DIR, "sequence_data.npz")):
    print("BŁĄD: Brak przetworzonych danych sekwencyjnych!")
    print("Najpierw uruchom: python 02_preprocessing.py")
    sys.exit(1)

# Wczytanie danych 3D: [samples, timesteps, features]
data = np.load(os.path.join(PREPROCESSED_DIR, "sequence_data.npz"),
               allow_pickle=True)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

# Wczytanie parametrów
with open(os.path.join(PREPROCESSED_DIR, "params.pkl"), "rb") as f:
    params = pickle.load(f)

RUL_CLIP = params["rul_clip"]
SEQ_LEN = X_train.shape[1]
N_FEATURES = X_train.shape[2]

print(f"\n[✓] Dane sekwencyjne załadowane:")
print(f"  Train: {X_train.shape}  (samples × timesteps × features)")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Sequence length: {SEQ_LEN}")
print(f"  Features: {N_FEATURES}")
print(f"  RUL clip: {RUL_CLIP}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. PRZYGOTOWANIE DANYCH PyTorch                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 2. Przygotowanie tensorów PyTorch ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
print(f"  PyTorch version: {torch.__version__}")

# Konwersja na tensory
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# DataLoader
BATCH_SIZE = 256

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"  Batch size: {BATCH_SIZE}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. ARCHITEKTURA CNN-LSTM                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM Hybrid do predykcji RUL.

    Architektura:
        Input [batch, 30, 10]

        ── BLOK CNN (ekstrakcja lokalnych wzorców) ──
        → Conv1D (10→32 filtry, kernel=3, padding=1) + BatchNorm + ReLU
        → Conv1D (32→64 filtry, kernel=3, padding=1) + BatchNorm + ReLU
        → MaxPool1D (kernel=2) → zmniejsza seq_len: 30 → 15

        ── BLOK LSTM (modelowanie sekwencji) ──
        → LSTM warstwa 1 (64 → 64 units)
        → Dropout (0.3)
        → LSTM warstwa 2 (64 → 32 units)
        → ostatni timestep

        ── HEAD (regresja) ──
        → Dropout (0.3)
        → Dense (32 → 32) + ReLU
        → Dense (32 → 1) → RUL

    Conv1D w PyTorch oczekuje [batch, channels, seq_len],
    więc transponujemy: [batch, seq_len, features] → [batch, features, seq_len]
    Po konwolucji transponujemy z powrotem dla LSTM.
    """
    def __init__(self, n_features, conv1_filters=32, conv2_filters=64,
                 kernel_size=3, lstm_hidden_1=64, lstm_hidden_2=32,
                 dense_units=32, dropout=0.3):
        super(CNNLSTMModel, self).__init__()

        # ── Blok CNN ──
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv1_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # same padding → zachowuje seq_len
        )
        self.bn1 = nn.BatchNorm1d(conv1_filters)

        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters)

        self.pool = nn.MaxPool1d(kernel_size=2)  # seq_len: 30 → 15
        self.relu = nn.ReLU()

        # ── Blok LSTM ──
        self.lstm1 = nn.LSTM(
            input_size=conv2_filters,    # wejście = wyjście z CNN (64)
            hidden_size=lstm_hidden_1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            batch_first=True,
        )

        # ── Head (regresja) ──
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_2, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):
        # x: [batch, seq_len=30, features=10]

        # ── CNN ──
        # Conv1d wymaga [batch, channels, seq_len]
        out = x.permute(0, 2, 1)         # [batch, 10, 30]
        out = self.relu(self.bn1(self.conv1(out)))  # [batch, 32, 30]
        out = self.relu(self.bn2(self.conv2(out)))  # [batch, 64, 30]
        out = self.pool(out)              # [batch, 64, 15]

        # Transpozycja z powrotem do [batch, seq_len, features] dla LSTM
        out = out.permute(0, 2, 1)        # [batch, 15, 64]

        # ── LSTM ──
        out, _ = self.lstm1(out)          # [batch, 15, 64]
        out = self.dropout(out)
        out, _ = self.lstm2(out)          # [batch, 15, 32]
        out = out[:, -1, :]              # ostatni timestep: [batch, 32]

        # ── Head ──
        out = self.dropout(out)
        out = self.relu(self.fc1(out))    # [batch, 32]
        out = self.fc2(out)               # [batch, 1]
        return out.squeeze(-1)            # [batch]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. METRYKI EWALUACJI                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def nasa_score(y_true, y_pred):
    """
    NASA Scoring Function (asymetryczna):
    d < 0 (za wcześnie): s += exp(-d/13) - 1  (mniejsza kara)
    d ≥ 0 (za późno):    s += exp(d/10) - 1   (większa kara!)
    """
    d = y_pred - y_true
    score = 0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13) - 1
        else:
            score += np.exp(di / 10) - 1
    return score


def evaluate_model(y_true, y_pred, set_name="Test"):
    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "NASA Score": nasa_score(y_true, y_pred),
    }
    print(f"\n  {set_name} Metrics:")
    for name, value in metrics.items():
        if name == "NASA Score":
            print(f"    {name:>12s}: {value:>12,.1f}")
        else:
            print(f"    {name:>12s}: {value:>10.4f}")
    return metrics


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. TRENING CNN-LSTM                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 3. Architektura CNN-LSTM ---")

# Hiperparametry
CONV1_FILTERS = 32    # Conv1D warstwa 1
CONV2_FILTERS = 64    # Conv1D warstwa 2
KERNEL_SIZE = 3       # rozmiar okna konwolucji
LSTM_HIDDEN_1 = 64    # LSTM warstwa 1
LSTM_HIDDEN_2 = 32    # LSTM warstwa 2
DENSE_UNITS = 32      # warstwa gęsta
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15         # early stopping

model = CNNLSTMModel(
    n_features=N_FEATURES,
    conv1_filters=CONV1_FILTERS,
    conv2_filters=CONV2_FILTERS,
    kernel_size=KERNEL_SIZE,
    lstm_hidden_1=LSTM_HIDDEN_1,
    lstm_hidden_2=LSTM_HIDDEN_2,
    dense_units=DENSE_UNITS,
    dropout=DROPOUT,
).to(device)

# Liczba parametrów
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  Architektura:")
print(f"    ── Blok CNN ──")
print(f"    Conv1D 1: {N_FEATURES} → {CONV1_FILTERS} filtry (kernel={KERNEL_SIZE})")
print(f"    Conv1D 2: {CONV1_FILTERS} → {CONV2_FILTERS} filtry (kernel={KERNEL_SIZE})")
print(f"    MaxPool1D: seq_len {SEQ_LEN} → {SEQ_LEN // 2}")
print(f"    ── Blok LSTM ──")
print(f"    LSTM 1: {CONV2_FILTERS} → {LSTM_HIDDEN_1} units")
print(f"    LSTM 2: {LSTM_HIDDEN_1} → {LSTM_HIDDEN_2} units")
print(f"    ── Head ──")
print(f"    Dense:  {LSTM_HIDDEN_2} → {DENSE_UNITS} → 1")
print(f"    Dropout: {DROPOUT}")
print(f"    Parametry: {total_params:,} (trenowalne: {trainable_params:,})")

print(f"\n  Hiperparametry treningu:")
print(f"    Optimizer: Adam (lr={LEARNING_RATE})")
print(f"    Loss: MSE")
print(f"    Epochs: max {EPOCHS} (early stopping patience={PATIENCE})")
print(f"    Batch size: {BATCH_SIZE}")

# Optimizer i loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Scheduler — redukcja LR gdy val loss się nie poprawia
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
)

print(f"\n--- 4. Trening CNN-LSTM ---\n")

# ── Pętla treningowa z early stopping ────────────────────────────────────────
best_val_loss = float("inf")
best_epoch = 0
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "lr": []}

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # --- TRENING ---
    model.train()
    train_losses = []

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # --- WALIDACJA ---
    model.eval()
    val_losses = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)
    current_lr = optimizer.param_groups[0]["lr"]

    # Zapis historii
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["lr"].append(current_lr)

    # Scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    # Log co 5 epok lub przy poprawie
    if epoch % 5 == 0 or epoch == 1 or epochs_no_improve == 0:
        marker = " ★" if epochs_no_improve == 0 else ""
        print(f"  Epoch {epoch:>3}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.1e}{marker}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n  [!] Early stopping — brak poprawy od {PATIENCE} epok")
        break

train_time = time.time() - start_time

# Przywróć najlepszy model
model.load_state_dict(best_model_state)
print(f"\n  [✓] Trening zakończony w {train_time:.1f}s")
print(f"  Najlepsza epoka: {best_epoch}/{epoch}")
print(f"  Najlepszy val loss: {best_val_loss:.4f} "
      f"(RMSE ≈ {np.sqrt(best_val_loss):.2f})")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. EWALUACJA                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 5. Ewaluacja ---")

model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_t).cpu().numpy()
    y_pred_val = model(X_val_t).cpu().numpy()
    y_pred_test = model(X_test_t).cpu().numpy()

# Clip predykcji do [0, RUL_CLIP]
y_pred_train = np.clip(y_pred_train, 0, RUL_CLIP)
y_pred_val = np.clip(y_pred_val, 0, RUL_CLIP)
y_pred_test = np.clip(y_pred_test, 0, RUL_CLIP)

y_train_np = y_train
y_val_np = y_val
y_test_np = y_test

metrics_train = evaluate_model(y_train_np, y_pred_train, "Train")
metrics_val = evaluate_model(y_val_np, y_pred_val, "Validation")
metrics_test = evaluate_model(y_test_np, y_pred_test, "Test")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. WIZUALIZACJE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 6. Wizualizacje ---")

# ── 7.1 Learning Curves ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
epochs_range = range(1, len(history["train_loss"]) + 1)
ax.plot(epochs_range, history["train_loss"],
        label="Train", color="#2196F3", lw=1.5)
ax.plot(epochs_range, history["val_loss"],
        label="Validation", color="#FF5722", lw=1.5)
ax.axvline(best_epoch, color="gray", linestyle="--",
           alpha=0.7, label=f"Best epoch = {best_epoch}")
ax.set_xlabel("Epoka")
ax.set_ylabel("MSE Loss")
ax.set_title("CNN-LSTM — Learning Curves (Loss)")
ax.legend()

ax = axes[1]
train_rmse = [np.sqrt(l) for l in history["train_loss"]]
val_rmse = [np.sqrt(l) for l in history["val_loss"]]
ax.plot(epochs_range, train_rmse,
        label="Train", color="#2196F3", lw=1.5)
ax.plot(epochs_range, val_rmse,
        label="Validation", color="#FF5722", lw=1.5)
ax.axvline(best_epoch, color="gray", linestyle="--",
           alpha=0.7, label=f"Best epoch = {best_epoch}")
ax.set_xlabel("Epoka")
ax.set_ylabel("RMSE")
ax.set_title("CNN-LSTM — Learning Curves (RMSE)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/22_cnn_lstm_learning_curves.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 22_cnn_lstm_learning_curves.png")

# ── 7.2 Predicted vs Actual (Test) ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(y_test_np, y_pred_test, alpha=0.7, s=40,
           c="#E91E63", edgecolors="white")
ax.plot([0, RUL_CLIP], [0, RUL_CLIP],
        "r--", lw=2, label="Idealna predykcja")
ax.fill_between([0, RUL_CLIP], [0 - 15, RUL_CLIP - 15],
                [0 + 15, RUL_CLIP + 15],
                alpha=0.15, color="green", label="±15 cykli")
ax.set_xlabel("Rzeczywisty RUL")
ax.set_ylabel("Przewidywany RUL")
ax.set_title(f"CNN-LSTM — Predicted vs Actual (RMSE={metrics_test['RMSE']:.2f})")
ax.legend()
ax.set_xlim(-5, RUL_CLIP + 5)
ax.set_ylim(-5, RUL_CLIP + 5)

ax = axes[1]
errors = y_pred_test - y_test_np
ax.hist(errors, bins=25, color="#FF9800", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", lw=2, linestyle="--")
ax.axvline(errors.mean(), color="blue", lw=2, linestyle="--",
           label=f"Średni błąd = {errors.mean():.1f}")
ax.set_xlabel("Błąd predykcji (predicted - actual)")
ax.set_ylabel("Częstość")
ax.set_title("Rozkład błędów (Test)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/23_cnn_lstm_predictions.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 23_cnn_lstm_predictions.png")

# ── 7.3 Predykcja per silnik (Test) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

unit_ids = np.arange(1, len(y_test_np) + 1)
width = 0.35

ax.bar(unit_ids - width / 2, y_test_np, width,
       label="Rzeczywisty RUL", color="#2196F3", alpha=0.8)
ax.bar(unit_ids + width / 2, y_pred_test, width,
       label="Predykcja CNN-LSTM", color="#E91E63", alpha=0.8)

ax.set_xlabel("Nr silnika (test)")
ax.set_ylabel("RUL (cykle)")
ax.set_title("CNN-LSTM — RUL per silnik testowy")
ax.legend()
ax.set_xlim(0, len(y_test_np) + 1)

for i in range(len(y_test_np)):
    if abs(y_pred_test[i] - y_test_np[i]) > 25:
        ax.annotate("!", (unit_ids[i], max(y_test_np[i], y_pred_test[i]) + 3),
                    ha="center", fontsize=10, color="red", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/24_cnn_lstm_per_unit.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 24_cnn_lstm_per_unit.png")

# ── 7.4 NASA Score per silnik ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

d_test = y_pred_test - y_test_np
penalties = []
for di in d_test:
    if di < 0:
        penalties.append(np.exp(-di / 13) - 1)
    else:
        penalties.append(np.exp(di / 10) - 1)

colors = ["#F44336" if d >= 0 else "#4CAF50" for d in d_test]
ax.bar(unit_ids, penalties, color=colors, alpha=0.8)
ax.set_xlabel("Nr silnika (test)")
ax.set_ylabel("NASA penalty")
ax.set_title(f"CNN-LSTM — Kara NASA per silnik "
             f"(total = {metrics_test['NASA Score']:,.0f})")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#F44336", alpha=0.8, label="Za późno (d≥0)"),
    Patch(facecolor="#4CAF50", alpha=0.8, label="Za wcześnie (d<0)"),
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/25_cnn_lstm_nasa_score.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 25_cnn_lstm_nasa_score.png")

# ── 7.5 Learning Rate Schedule ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs_range, history["lr"], color="#9C27B0", lw=1.5)
ax.set_xlabel("Epoka")
ax.set_ylabel("Learning Rate")
ax.set_title("CNN-LSTM — Learning Rate Schedule (ReduceLROnPlateau)")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/26_cnn_lstm_lr_schedule.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 26_cnn_lstm_lr_schedule.png")

# ── 7.6 Wizualizacja filtrów Conv1D (co CNN "widzi") ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Filtry Conv1D warstwy 1: [out_channels=32, in_channels=10, kernel=3]
conv1_weights = model.conv1.weight.detach().cpu().numpy()

ax = axes[0]
im = ax.imshow(conv1_weights.mean(axis=2).T, aspect="auto",
               cmap="RdBu_r", interpolation="nearest")
ax.set_xlabel("Nr filtra (32 filtry)")
ax.set_ylabel("Kanał wejściowy (sensor)")
ax.set_title("Conv1D warstwa 1 — średnia waga filtrów per sensor")
ax.set_yticks(range(N_FEATURES))
ax.set_yticklabels([f"s{i}" for i in range(N_FEATURES)], fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)

# Aktywacje Conv1D — średnia odpowiedź na dane testowe
model.eval()
with torch.no_grad():
    # Przepuść test przez sam blok CNN
    x_test_perm = X_test_t.permute(0, 2, 1)            # [100, 10, 30]
    conv1_out = model.relu(model.bn1(model.conv1(x_test_perm)))  # [100, 32, 30]
    conv1_activation = conv1_out.cpu().numpy()

# Średnia aktywacja per filtr per timestep (uśredniona po silnikach)
avg_activation = conv1_activation.mean(axis=0)  # [32, 30]

ax = axes[1]
im = ax.imshow(avg_activation, aspect="auto", cmap="viridis",
               interpolation="nearest")
ax.set_xlabel("Timestep (cykl operacyjny)")
ax.set_ylabel("Nr filtra")
ax.set_title("Conv1D warstwa 1 — średnia aktywacja na danych testowych")
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/27_cnn_lstm_conv_filters.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 27_cnn_lstm_conv_filters.png")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8. ZAPIS MODELU I WYNIKÓW                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 7. Zapis ---")

# Zapis modelu PyTorch
model_path = os.path.join(MODELS_DIR, "cnn_lstm_model.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "architecture": {
        "n_features": N_FEATURES,
        "conv1_filters": CONV1_FILTERS,
        "conv2_filters": CONV2_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "lstm_hidden_1": LSTM_HIDDEN_1,
        "lstm_hidden_2": LSTM_HIDDEN_2,
        "dense_units": DENSE_UNITS,
        "dropout": DROPOUT,
    },
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
}, model_path)
print(f"  [✓] {model_path}")

# Zapis wyników
results_dict = {
    "model": "CNN-LSTM",
    "train_time_s": train_time,
    "best_epoch": best_epoch,
    "total_epochs": epoch,
    "hyperparams": {
        "conv1_filters": CONV1_FILTERS,
        "conv2_filters": CONV2_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "lstm_hidden_1": LSTM_HIDDEN_1,
        "lstm_hidden_2": LSTM_HIDDEN_2,
        "dense_units": DENSE_UNITS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "seq_length": SEQ_LEN,
    },
    "total_params": total_params,
    "history": history,
    "metrics_train": metrics_train,
    "metrics_val": metrics_val,
    "metrics_test": metrics_test,
    "y_test_true": y_test_np,
    "y_test_pred": y_pred_test,
}

with open(os.path.join(MODELS_DIR, "cnn_lstm_results.pkl"), "wb") as f:
    pickle.dump(results_dict, f)
print(f"  [✓] {MODELS_DIR}/cnn_lstm_results.pkl")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("PODSUMOWANIE — CNN-LSTM Hybrid")
print("=" * 70)
print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  Model:             CNN-LSTM Hybrid                             ║
║  Parametry:         {total_params:>6,}                                    ║
║  Czas treningu:     {train_time:>6.1f}s                                  ║
║  Best epoch:        {best_epoch:>4} / {epoch}                                  ║
║  Sequence length:   {SEQ_LEN} timesteps × {N_FEATURES} features               ║
║  CNN:  Conv1D({CONV1_FILTERS}) → Conv1D({CONV2_FILTERS}) → MaxPool            ║
║  LSTM: {LSTM_HIDDEN_1} → {LSTM_HIDDEN_2} units                                ║
║                                                                   ║
║  ┌─────────────┬──────────┬──────────┬──────────┐               ║
║  │   Metryka   │  Train   │   Val    │   Test   │               ║
║  ├─────────────┼──────────┼──────────┼──────────┤               ║
║  │ RMSE        │ {metrics_train['RMSE']:>8.2f} │ {metrics_val['RMSE']:>8.2f} │ {metrics_test['RMSE']:>8.2f} │               ║
║  │ MAE         │ {metrics_train['MAE']:>8.2f} │ {metrics_val['MAE']:>8.2f} │ {metrics_test['MAE']:>8.2f} │               ║
║  │ R²          │ {metrics_train['R²']:>8.4f} │ {metrics_val['R²']:>8.4f} │ {metrics_test['R²']:>8.4f} │               ║
║  │ NASA Score  │ {metrics_train['NASA Score']:>8.0f} │ {metrics_val['NASA Score']:>8.0f} │ {metrics_test['NASA Score']:>8.0f} │               ║
║  └─────────────┴──────────┴──────────┴──────────┘               ║
║                                                                   ║
║  Następny krok: python 06_model_comparison.py                     ║
╚═══════════════════════════════════════════════════════════════════╝
""")