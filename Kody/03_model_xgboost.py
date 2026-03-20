"""
=============================================================================
NASA C-MAPSS — Model 1: XGBoost Baseline (Krok 3)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Baseline model — XGBoost Regressor z ręcznymi feature'ami
Ewaluacja: RMSE + MAE + NASA Scoring Function + R²
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. ŁADOWANIE PRZETWORZONYCH DANYCH                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("MODEL 1: XGBoost Baseline — NASA C-MAPSS (FD001)")
print("=" * 70)

PREPROCESSED_DIR = r"C:\Users\Błażej\Desktop\Optymalizacjaalgo\preprocessed"
PLOT_DIR = r"C:\Users\Błażej\Desktop\Optymalizacjaalgo\plots"
MODELS_DIR = r"C:\Users\Błażej\Desktop\Optymalizacjaalgo\models"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Sprawdź czy dane istnieją
if not os.path.exists(os.path.join(PREPROCESSED_DIR, "xgboost_data.npz")):
    print("BŁĄD: Brak przetworzonych danych!")
    print("Najpierw uruchom: python 02_preprocessing.py")
    sys.exit(1)

# Wczytanie danych
data = np.load(os.path.join(PREPROCESSED_DIR, "xgboost_data.npz"),
               allow_pickle=True)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]
feature_names = data["feature_names"]

# Wczytanie parametrów
with open(os.path.join(PREPROCESSED_DIR, "params.pkl"), "rb") as f:
    params = pickle.load(f)

print(f"\n[✓] Dane załadowane:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  RUL clip: {params['rul_clip']}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. METRYKI EWALUACJI                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def nasa_score(y_true, y_pred):
    """
    NASA Scoring Function (asymetryczna):
    - Kara za spóźnioną prognozę (predicted > actual → silnik się zepsuł
      zanim go naprawiliśmy) jest WIĘKSZA niż za zbyt wczesną.
    - s = Σ (e^(-d/13) - 1)  jeśli d < 0 (za wcześnie — lepiej)
    - s = Σ (e^(d/10) - 1)   jeśli d ≥ 0 (za późno — gorzej!)
    Gdzie d = predicted - actual (błąd predykcji)
    """
    d = y_pred - y_true  # dodatnie = za późno, ujemne = za wcześnie
    score = 0
    for di in d:
        if di < 0:
            score += np.exp(-di / 13) - 1  # za wcześnie — mniejsza kara
        else:
            score += np.exp(di / 10) - 1   # za późno — większa kara
    return score


def evaluate_model(y_true, y_pred, set_name="Test"):
    """Wyświetla wszystkie metryki."""
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
# ║  3. TRENING XGBoost                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 3. Trening XGBoost ---")

# Instalacja xgboost jeśli brak
try:
    import xgboost as xgb
except ImportError:
    print("[!] Instaluję xgboost...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb

print(f"  XGBoost version: {xgb.__version__}")

# --- Hiperparametry ---
xgb_params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,       # L1 regularization
    "reg_lambda": 1.0,      # L2 regularization
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

print(f"\n  Hiperparametry:")
for k, v in xgb_params.items():
    print(f"    {k}: {v}")

# Trening z early stopping
model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=30)

print(f"\n  Trenuję model...")
start_time = time.time()

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50,
)

train_time = time.time() - start_time
print(f"\n  [✓] Trening zakończony w {train_time:.1f}s")
print(f"  Najlepszy wynik na val: iteracja {model.best_iteration}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. EWALUACJA                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 4. Ewaluacja ---")

# Predykcje
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Clip predykcji do [0, RUL_CLIP]
y_pred_train = np.clip(y_pred_train, 0, params["rul_clip"])
y_pred_val = np.clip(y_pred_val, 0, params["rul_clip"])
y_pred_test = np.clip(y_pred_test, 0, params["rul_clip"])

# Metryki
metrics_train = evaluate_model(y_train, y_pred_train, "Train")
metrics_val = evaluate_model(y_val, y_pred_val, "Validation")
metrics_test = evaluate_model(y_test, y_pred_test, "Test")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. WIZUALIZACJE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 5. Wizualizacje ---")

# ── 5.1 Learning Curves ──────────────────────────────────────────────────────
results = model.evals_result()
epochs = len(results["validation_0"]["rmse"])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(epochs), results["validation_0"]["rmse"],
        label="Train", color="#2196F3", lw=1.5)
ax.plot(range(epochs), results["validation_1"]["rmse"],
        label="Validation", color="#FF5722", lw=1.5)
ax.axvline(model.best_iteration, color="gray", linestyle="--",
           alpha=0.7, label=f"Best iter = {model.best_iteration}")
ax.set_xlabel("Boosting Rounds")
ax.set_ylabel("RMSE")
ax.set_title("XGBoost — Learning Curves")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/12_xgb_learning_curves.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 12_xgb_learning_curves.png")

# ── 5.2 Predicted vs Actual (Test) ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax = axes[0]
ax.scatter(y_test, y_pred_test, alpha=0.7, s=40, c="#2196F3", edgecolors="white")
ax.plot([0, params["rul_clip"]], [0, params["rul_clip"]],
        "r--", lw=2, label="Idealna predykcja")
ax.fill_between([0, params["rul_clip"]], [0 - 15, params["rul_clip"] - 15],
                [0 + 15, params["rul_clip"] + 15],
                alpha=0.15, color="green", label="±15 cykli")
ax.set_xlabel("Rzeczywisty RUL")
ax.set_ylabel("Przewidywany RUL")
ax.set_title(f"XGBoost — Predicted vs Actual (RMSE={metrics_test['RMSE']:.2f})")
ax.legend()
ax.set_xlim(-5, params["rul_clip"] + 5)
ax.set_ylim(-5, params["rul_clip"] + 5)

# Histogram błędów
ax = axes[1]
errors = y_pred_test - y_test
ax.hist(errors, bins=25, color="#FF9800", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", lw=2, linestyle="--")
ax.axvline(errors.mean(), color="blue", lw=2, linestyle="--",
           label=f"Średni błąd = {errors.mean():.1f}")
ax.set_xlabel("Błąd predykcji (predicted - actual)")
ax.set_ylabel("Częstość")
ax.set_title("Rozkład błędów (Test)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/13_xgb_predictions.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 13_xgb_predictions.png")

# ── 5.3 Predykcja per silnik (Test) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

unit_ids = np.arange(1, len(y_test) + 1)
width = 0.35

bars1 = ax.bar(unit_ids - width / 2, y_test, width,
               label="Rzeczywisty RUL", color="#2196F3", alpha=0.8)
bars2 = ax.bar(unit_ids + width / 2, y_pred_test, width,
               label="Predykcja XGBoost", color="#FF5722", alpha=0.8)

ax.set_xlabel("Nr silnika (test)")
ax.set_ylabel("RUL (cykle)")
ax.set_title("XGBoost — RUL per silnik testowy")
ax.legend()
ax.set_xlim(0, len(y_test) + 1)

# Podświetl silniki z dużym błędem (>25 cykli)
for i in range(len(y_test)):
    if abs(y_pred_test[i] - y_test[i]) > 25:
        ax.annotate("!", (unit_ids[i], max(y_test[i], y_pred_test[i]) + 3),
                    ha="center", fontsize=10, color="red", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/14_xgb_per_unit.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 14_xgb_per_unit.png")

# ── 5.4 Feature Importance (Top 25) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

importance = model.feature_importances_
indices = np.argsort(importance)[-25:]  # top 25

ax.barh(range(len(indices)), importance[indices], color="#7E57C2", alpha=0.85)
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("XGBoost — Top 25 najważniejszych cech")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/15_xgb_feature_importance.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 15_xgb_feature_importance.png")

# ── 5.5 Asymetria NASA Score — wizualizacja ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wizualizacja funkcji kary NASA
d_range = np.linspace(-50, 50, 200)
penalty_early = np.exp(-d_range[d_range < 0] / 13) - 1
penalty_late = np.exp(d_range[d_range >= 0] / 10) - 1

ax = axes[0]
ax.plot(d_range[d_range < 0], penalty_early,
        color="#4CAF50", lw=2.5, label="Za wcześnie (d<0)")
ax.plot(d_range[d_range >= 0], penalty_late,
        color="#F44336", lw=2.5, label="Za późno (d≥0)")
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("d = predicted - actual")
ax.set_ylabel("Kara")
ax.set_title("NASA Scoring Function — asymetryczna kara")
ax.legend()
ax.set_ylim(-0.5, 50)

# Kara per silnik testowy
ax = axes[1]
d_test = y_pred_test - y_test
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
ax.set_title(f"Kara NASA per silnik (total = {metrics_test['NASA Score']:,.0f})")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/16_xgb_nasa_score.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 16_xgb_nasa_score.png")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. ZAPIS MODELU I WYNIKÓW                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 6. Zapis ---")

# Zapis modelu
model.save_model(os.path.join(MODELS_DIR, "xgboost_baseline.json"))
print(f"  [✓] {MODELS_DIR}/xgboost_baseline.json")

# Zapis wyników
results_dict = {
    "model": "XGBoost",
    "train_time_s": train_time,
    "best_iteration": model.best_iteration,
    "hyperparams": xgb_params,
    "metrics_train": metrics_train,
    "metrics_val": metrics_val,
    "metrics_test": metrics_test,
    "y_test_true": y_test,
    "y_test_pred": y_pred_test,
}

with open(os.path.join(MODELS_DIR, "xgboost_results.pkl"), "wb") as f:
    pickle.dump(results_dict, f)
print(f"  [✓] {MODELS_DIR}/xgboost_results.pkl")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("PODSUMOWANIE — XGBoost Baseline")
print("=" * 70)
print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  Model:             XGBoost Regressor                           ║
║  Czas treningu:     {train_time:>6.1f}s                                  ║
║  Best iteration:    {model.best_iteration:>4} / {xgb_params['n_estimators']}                              ║
║  Features:          {len(feature_names)} (rolling stats z 10 sensorów)      ║
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
║  Następny model: python 04_model_lstm.py                          ║
╚═══════════════════════════════════════════════════════════════════╝
""")