"""
=============================================================================
NASA C-MAPSS — Porównanie modeli (Krok 6)
Kurs: Zastosowania modeli AI w automatyce
=============================================================================
Porównanie 3 modeli predykcji RUL na zbiorze FD001:
  1. XGBoost Baseline (ręczne features)
  2. LSTM (sekwencyjny)
  3. CNN-LSTM Hybrid (konwolucja + sekwencja)
Metryki: RMSE, MAE, R², NASA Score, czas treningu
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. ŁADOWANIE WYNIKÓW WSZYSTKICH MODELI                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("PORÓWNANIE MODELI — NASA C-MAPSS (FD001)")
print("=" * 70)

MODELS_DIR = "./models"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Definicja modeli do porównania
model_files = {
    "XGBoost":  "xgboost_results.pkl",
    "LSTM":     "lstm_results.pkl",
    "CNN-LSTM": "cnn_lstm_results.pkl",
}

# Kolory dla każdego modelu (spójne we wszystkich wykresach)
MODEL_COLORS = {
    "XGBoost":  "#FF5722",   # pomarańczowo-czerwony
    "LSTM":     "#4CAF50",   # zielony
    "CNN-LSTM": "#E91E63",   # różowy
}

results = {}
for name, fname in model_files.items():
    fpath = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [!] Brak wyników: {fpath} — pomiń {name}")
        continue
    with open(fpath, "rb") as f:
        results[name] = pickle.load(f)
    print(f"  [✓] {name}: załadowano {fname}")

if len(results) < 2:
    print("\nBŁĄD: Potrzeba minimum 2 modeli do porównania!")
    print("Uruchom najpierw skrypty 03, 04, 05.")
    sys.exit(1)

model_names = list(results.keys())
n_models = len(model_names)
print(f"\n  Modele do porównania: {n_models} → {', '.join(model_names)}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. TABELA PORÓWNAWCZA                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 2. Tabela porównawcza ---\n")

# Zbierz metryki testowe
comparison = {}
for name in model_names:
    r = results[name]
    comparison[name] = {
        "RMSE":       r["metrics_test"]["RMSE"],
        "MAE":        r["metrics_test"]["MAE"],
        "R²":         r["metrics_test"]["R²"],
        "NASA Score": r["metrics_test"]["NASA Score"],
        "Train Time": r["train_time_s"],
    }

# Wyświetl tabelę
header = f"  {'Metryka':<14}"
for name in model_names:
    header += f"│ {name:>12} "
print(header)
print("  " + "─" * 14 + ("┼" + "─" * 14) * n_models)

for metric in ["RMSE", "MAE", "R²", "NASA Score", "Train Time"]:
    row = f"  {metric:<14}"
    values = [comparison[name][metric] for name in model_names]

    # Znajdź najlepszy model dla danej metryki
    if metric == "R²":
        best_idx = np.argmax(values)
    elif metric == "Train Time":
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmin(values)

    for i, name in enumerate(model_names):
        val = comparison[name][metric]
        if metric == "NASA Score":
            cell = f"{val:>10,.0f}"
        elif metric == "Train Time":
            cell = f"{val:>9.1f}s"
        elif metric == "R²":
            cell = f"{val:>10.4f}"
        else:
            cell = f"{val:>10.2f}"

        marker = " ★" if i == best_idx else "  "
        row += f"│ {cell}{marker}"
    print(row)

print()

# Ranking ogólny (suma rang)
print("  Ranking (suma rang — im mniej, tym lepiej):")
rank_metrics = ["RMSE", "MAE", "NASA Score"]  # R² i czas to dodatkowe info
rank_sums = {name: 0 for name in model_names}

for metric in rank_metrics:
    values = [(comparison[name][metric], name) for name in model_names]
    values.sort()  # mniejsze = lepsze
    for rank, (val, name) in enumerate(values, 1):
        rank_sums[name] += rank

ranking = sorted(rank_sums.items(), key=lambda x: x[1])
for pos, (name, score) in enumerate(ranking, 1):
    medal = ["🥇", "🥈", "🥉"][pos - 1] if pos <= 3 else "  "
    print(f"    {medal} {pos}. {name}: suma rang = {score}")

best_model = ranking[0][0]
print(f"\n  >>> Najlepszy model: {best_model} <<<")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. WIZUALIZACJE                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 3. Wizualizacje ---")

colors = [MODEL_COLORS[name] for name in model_names]

# ── 3.1 Metryki — wykres słupkowy ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = [
    ("RMSE", "RMSE (im mniej, tym lepiej)", False),
    ("MAE", "MAE (im mniej, tym lepiej)", False),
    ("R²", "R² (im więcej, tym lepiej)", True),
    ("NASA Score", "NASA Score (im mniej, tym lepiej)", False),
]

for idx, (metric, title, higher_better) in enumerate(metrics_to_plot):
    ax = axes[idx // 2][idx % 2]
    values = [comparison[name][metric] for name in model_names]

    bars = ax.bar(model_names, values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5)

    # Podświetl najlepszy
    if higher_better:
        best_i = np.argmax(values)
    else:
        best_i = np.argmin(values)
    bars[best_i].set_edgecolor("gold")
    bars[best_i].set_linewidth(3)

    # Etykiety na słupkach
    for bar, val in zip(bars, values):
        if metric == "NASA Score":
            label = f"{val:,.0f}"
        elif metric == "R²":
            label = f"{val:.4f}"
        else:
            label = f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(metric)

plt.suptitle("Porównanie modeli — metryki testowe (FD001)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/28_comparison_metrics.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 28_comparison_metrics.png")

# ── 3.2 Predicted vs Actual — wszystkie modele na jednym wykresie ────────────
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
if n_models == 1:
    axes = [axes]

for idx, name in enumerate(model_names):
    ax = axes[idx]
    r = results[name]
    y_true = r["y_test_true"]
    y_pred = r["y_test_pred"]
    rul_clip = 125  # stałe dla FD001

    ax.scatter(y_true, y_pred, alpha=0.7, s=40,
               c=MODEL_COLORS[name], edgecolors="white")
    ax.plot([0, rul_clip], [0, rul_clip], "k--", lw=1.5, alpha=0.5)
    ax.fill_between([0, rul_clip], [0 - 15, rul_clip - 15],
                    [0 + 15, rul_clip + 15],
                    alpha=0.1, color="green")

    rmse_val = r["metrics_test"]["RMSE"]
    nasa_val = r["metrics_test"]["NASA Score"]
    ax.set_title(f"{name}\nRMSE={rmse_val:.2f}  NASA={nasa_val:,.0f}",
                 fontsize=11)
    ax.set_xlabel("Rzeczywisty RUL")
    if idx == 0:
        ax.set_ylabel("Przewidywany RUL")
    ax.set_xlim(-5, rul_clip + 5)
    ax.set_ylim(-5, rul_clip + 5)
    ax.set_aspect("equal")

plt.suptitle("Predicted vs Actual — porównanie modeli",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/29_comparison_scatter.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 29_comparison_scatter.png")

# ── 3.3 Predykcja per silnik — overlay ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))

y_true = results[model_names[0]]["y_test_true"]
unit_ids = np.arange(1, len(y_true) + 1)

# Actual RUL jako szary bar
ax.bar(unit_ids, y_true, width=0.8, color="#BDBDBD", alpha=0.5,
       label="Rzeczywisty RUL", zorder=1)

# Predykcje jako linie/markery
marker_styles = ["o", "s", "D"]
for idx, name in enumerate(model_names):
    y_pred = results[name]["y_test_pred"]
    ax.scatter(unit_ids, y_pred, s=25, marker=marker_styles[idx],
               color=MODEL_COLORS[name], label=f"{name}", zorder=3,
               alpha=0.8, edgecolors="white", linewidths=0.5)

ax.set_xlabel("Nr silnika (test)", fontsize=11)
ax.set_ylabel("RUL (cykle)", fontsize=11)
ax.set_title("Predykcje RUL per silnik — wszystkie modele", fontsize=13)
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(0, len(y_true) + 1)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/30_comparison_per_unit.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 30_comparison_per_unit.png")

# ── 3.4 Rozkład błędów — histogramy obok siebie ─────────────────────────────
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]

for idx, name in enumerate(model_names):
    ax = axes[idx]
    r = results[name]
    errors = r["y_test_pred"] - r["y_test_true"]

    ax.hist(errors, bins=25, color=MODEL_COLORS[name],
            edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=1.5, linestyle="--", alpha=0.5)
    ax.axvline(errors.mean(), color="blue", lw=2, linestyle="--",
               label=f"Średni = {errors.mean():.1f}")
    ax.axvline(np.median(errors), color="red", lw=2, linestyle=":",
               label=f"Mediana = {np.median(errors):.1f}")

    ax.set_xlabel("Błąd (predicted − actual)")
    ax.set_ylabel("Częstość")
    ax.set_title(f"{name}", fontsize=11)
    ax.legend(fontsize=9)

plt.suptitle("Rozkład błędów predykcji — porównanie modeli",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/31_comparison_errors.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 31_comparison_errors.png")

# ── 3.5 NASA Score — porównanie asymetrii ────────────────────────────────────
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]

for idx, name in enumerate(model_names):
    ax = axes[idx]
    r = results[name]
    d_test = r["y_test_pred"] - r["y_test_true"]

    penalties = []
    for di in d_test:
        if di < 0:
            penalties.append(np.exp(-di / 13) - 1)
        else:
            penalties.append(np.exp(di / 10) - 1)

    bar_colors = ["#F44336" if d >= 0 else "#4CAF50" for d in d_test]
    ax.bar(unit_ids, penalties, color=bar_colors, alpha=0.8)
    ax.set_xlabel("Nr silnika")
    ax.set_ylabel("NASA penalty")

    total = r["metrics_test"]["NASA Score"]
    n_late = sum(1 for d in d_test if d >= 0)
    n_early = sum(1 for d in d_test if d < 0)
    ax.set_title(f"{name}\nNASA={total:,.0f}  "
                 f"({n_early} wczesnych, {n_late} późnych)", fontsize=10)

plt.suptitle("NASA Score per silnik — porównanie modeli",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/32_comparison_nasa.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 32_comparison_nasa.png")

# ── 3.6 Czas treningu ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

times = [results[name]["train_time_s"] for name in model_names]
bars = ax.barh(model_names, times, color=colors, alpha=0.85,
               edgecolor="white", linewidth=1.5)

for bar, t in zip(bars, times):
    ax.text(bar.get_width() + max(times) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s", va="center", fontweight="bold", fontsize=12)

ax.set_xlabel("Czas treningu (sekundy)")
ax.set_title("Porównanie czasu treningu", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/33_comparison_time.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 33_comparison_time.png")

# ── 3.7 Radar chart — profil modeli ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Normalizacja metryk do [0, 1] — wyższy = lepszy
radar_metrics = ["RMSE", "MAE", "NASA Score", "R²"]
raw_values = {}
for name in model_names:
    raw_values[name] = [comparison[name][m] for m in radar_metrics]

# Odwróć RMSE, MAE, NASA (mniejsze = lepsze → zamień na "jakość")
normalized = {}
for name in model_names:
    norm = []
    for i, metric in enumerate(radar_metrics):
        all_vals = [raw_values[n][i] for n in model_names]
        min_v, max_v = min(all_vals), max(all_vals)
        val = raw_values[name][i]

        if max_v == min_v:
            norm.append(1.0)
        elif metric == "R²":
            # Wyższe = lepsze
            norm.append((val - min_v) / (max_v - min_v))
        else:
            # Niższe = lepsze → odwróć
            norm.append(1 - (val - min_v) / (max_v - min_v))
    normalized[name] = norm

# Rysuj
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]  # zamknij wielokąt

for name in model_names:
    values = normalized[name] + normalized[name][:1]
    ax.plot(angles, values, "o-", linewidth=2, label=name,
            color=MODEL_COLORS[name], markersize=6)
    ax.fill(angles, values, alpha=0.1, color=MODEL_COLORS[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_title("Profil modeli (znormalizowane — wyżej = lepiej)",
             fontsize=13, fontweight="bold", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/34_comparison_radar.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 34_comparison_radar.png")

# ── 3.8 Tabela zbiorcza jako obraz ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

# Dane do tabeli
col_labels = ["Metryka"] + model_names
row_data = []

for metric, fmt in [("RMSE", ".2f"), ("MAE", ".2f"), ("R²", ".4f"),
                     ("NASA Score", ",.0f")]:
    row = [metric]
    values = [comparison[name][metric] for name in model_names]
    if metric == "R²":
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)

    for i, name in enumerate(model_names):
        val = format(comparison[name][metric], fmt)
        if i == best_idx:
            val = f"★ {val}"
        row.append(val)
    row_data.append(row)

# Dodaj czas
row = ["Czas treningu"]
times = [results[name]["train_time_s"] for name in model_names]
best_idx = np.argmin(times)
for i, name in enumerate(model_names):
    val = f"{times[i]:.1f}s"
    if i == best_idx:
        val = f"★ {val}"
    row.append(val)
row_data.append(row)

# Dodaj liczbę parametrów
row = ["Parametry"]
for name in model_names:
    r = results[name]
    if "total_params" in r:
        row.append(f"{r['total_params']:,}")
    elif name == "XGBoost":
        row.append("160 feat.")
    else:
        row.append("—")
row_data.append(row)

table = ax.table(cellText=row_data, colLabels=col_labels,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Kolorowanie nagłówków
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#37474F")
    cell.set_text_props(color="white", fontweight="bold")

# Kolorowanie kolumn modeli (lekkie tło)
for i in range(len(row_data)):
    for j, name in enumerate(model_names, 1):
        cell = table[i + 1, j]
        cell.set_facecolor(MODEL_COLORS[name] + "15")  # bardzo lekki kolor

ax.set_title("Podsumowanie wyników — NASA C-MAPSS FD001",
             fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/35_comparison_table.png", bbox_inches="tight")
plt.close()
print(f"  [✓] 35_comparison_table.png")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. ANALIZA SZCZEGÓŁOWA                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n--- 4. Analiza szczegółowa ---")

# Silniki z największym błędem per model
y_true = results[model_names[0]]["y_test_true"]

print("\n  Silniki z największym błędem (>25 cykli) per model:")
for name in model_names:
    y_pred = results[name]["y_test_pred"]
    errors = y_pred - y_true
    bad_engines = np.where(np.abs(errors) > 25)[0] + 1  # 1-indexed
    if len(bad_engines) > 0:
        print(f"    {name}: {len(bad_engines)} silników → {bad_engines.tolist()}")
    else:
        print(f"    {name}: 0 silników z błędem >25 cykli!")

# Silniki trudne dla WSZYSTKICH modeli
print("\n  Silniki trudne dla WSZYSTKICH modeli (błąd >15 cykli):")
difficult = np.ones(len(y_true), dtype=bool)
for name in model_names:
    y_pred = results[name]["y_test_pred"]
    errors = np.abs(y_pred - y_true)
    difficult &= (errors > 15)

difficult_ids = np.where(difficult)[0] + 1
if len(difficult_ids) > 0:
    print(f"    {len(difficult_ids)} silników: {difficult_ids.tolist()}")
    for eid in difficult_ids:
        i = eid - 1
        true_rul = y_true[i]
        preds = {name: results[name]["y_test_pred"][i] for name in model_names}
        pred_str = ", ".join([f"{n}={v:.0f}" for n, v in preds.items()])
        print(f"      Silnik {eid}: RUL={true_rul:.0f} → {pred_str}")
else:
    print("    Brak silników trudnych dla wszystkich modeli jednocześnie.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PODSUMOWANIE KOŃCOWE                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("PODSUMOWANIE KOŃCOWE — Porównanie modeli")
print("=" * 70)

# Najlepsze wyniki per metryka
print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║  NASA C-MAPSS FD001 — Wyniki 3 modeli                           ║
║                                                                   ║""")

for name in model_names:
    r = results[name]
    mt = r["metrics_test"]
    time_s = r["train_time_s"]
    print(f"║  {name:<12s}: RMSE={mt['RMSE']:>6.2f}  MAE={mt['MAE']:>6.2f}  "
          f"R²={mt['R²']:.4f}  NASA={mt['NASA Score']:>6.0f}  "
          f"t={time_s:>5.1f}s ║")

print(f"""║                                                                   ║
║  ★ Najlepszy model: {best_model:<12s}                                ║
║                                                                   ║
║  Wykresy 28–35 zapisane w ./plots/                                ║
╚═══════════════════════════════════════════════════════════════════╝
""")