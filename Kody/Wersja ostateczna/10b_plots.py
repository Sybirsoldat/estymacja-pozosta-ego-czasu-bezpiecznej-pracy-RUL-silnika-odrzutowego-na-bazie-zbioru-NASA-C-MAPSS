"""
10b_plots.py — Wykresy porównawcze 53–60
Ładuje wyniki z pkl, generuje wykresy.
"""
from shared import *

print("=" * 70)
print("WYKRESY PORÓWNAWCZE (53–60)")
print("=" * 70)

all_results = load_results()
if all_results is None:
    print("BŁĄD: Brak wyników! Uruchom najpierw 10a_train.py")
    sys.exit(1)
all_studies = load_studies()

# Wczytaj v1/v2
v1, v2 = None, None
for p in ["./results_all/all_results.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f: v1 = pickle.load(f)
        break
for p in ["./results_optuna_v2/optuna_v2_results.pkl"]:
    if os.path.exists(p):
        with open(p, "rb") as f: v2 = pickle.load(f)
        break

CV = {"XGBoost v1": "#FFAB91", "LSTM v1": "#A5D6A7",
      "XGBoost v2": "#FF8A65", "LSTM v2": "#66BB6A",
      "XGBoost v3": "#E64A19", "LSTM v3": "#2E7D32"}

# 53: v1 vs v2 vs v3
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ai, (metric, title) in enumerate([("RMSE", "RMSE (↓)"), ("NASA Score", "NASA Score (↓)")]):
    ax = axes[ai]
    x = np.arange(len(DATASETS))
    w, off = 0.13, 0
    if v1:
        for m in ["XGBoost", "LSTM"]:
            vals = [v1[ds][m]["metrics"][metric] for ds in DATASETS]
            ax.bar(x + off * w, vals, w, label=f"{m} v1", color=CV[f"{m} v1"], alpha=0.6)
            off += 1
    if v2:
        for m in ["XGBoost", "LSTM"]:
            vals = [v2[ds][m]["metrics"][metric] for ds in DATASETS]
            ax.bar(x + off * w, vals, w, label=f"{m} v2", color=CV[f"{m} v2"], alpha=0.75)
            off += 1
    for m in ["XGBoost", "LSTM"]:
        vals = [all_results[ds][m]["metrics"][metric] for ds in DATASETS]
        bars = ax.bar(x + off * w, vals, w, label=f"{m} v3", color=CV[f"{m} v3"], alpha=0.95)
        for xi, val in zip(x + off * w, vals):
            fmt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            ax.text(xi, val, fmt, ha="center", va="bottom", fontsize=6, rotation=45)
        off += 1
    ax.set_xticks(x + w * (off - 1) / 2); ax.set_xticklabels(DATASETS)
    ax.set_title(title); ax.legend(fontsize=7, ncol=3)
plt.suptitle("v1 vs v2 vs v3", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/53_v1_v2_v3_comparison.png", bbox_inches="tight"); plt.close()
print(f"  [✓] 53")

# 54: Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ai, (metric, title) in enumerate([("RMSE", "RMSE v3 (↓)"), ("NASA Score", "NASA Score v3 (↓)")]):
    ax = axes[ai]
    mat = np.array([[all_results[ds][m]["metrics"][metric] for ds in DATASETS] for m in ["XGBoost", "LSTM"]])
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DATASETS))); ax.set_xticklabels(DATASETS)
    ax.set_yticks(range(2)); ax.set_yticklabels(["XGBoost", "LSTM"]); ax.set_title(title)
    for i in range(2):
        for j in range(len(DATASETS)):
            val = mat[i, j]; txt = f"{val:,.0f}" if metric == "NASA Score" else f"{val:.1f}"
            best = val == mat[:, j].min(); clr = "white" if val > np.median(mat) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    fontweight="bold" if best else "normal", color=clr)
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle("Heatmap v3", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/54_v3_heatmap.png", bbox_inches="tight"); plt.close()
print(f"  [✓] 54")

# 55: Scatter
fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 5 * len(DATASETS)))
for row, ds in enumerate(DATASETS):
    yt = all_results[ds]["y_test"]
    for col, (m, clr) in enumerate([("XGBoost", "#E64A19"), ("LSTM", "#2E7D32")]):
        ax = axes[row][col]; yp = all_results[ds][m]["y_pred"]; met = all_results[ds][m]["metrics"]
        ax.scatter(yt, yp, alpha=0.5, s=20, c=clr, edgecolors="none")
        ax.plot([0, RUL_CLIP], [0, RUL_CLIP], "k--", lw=1, alpha=0.4)
        ax.fill_between([0, RUL_CLIP], [-15, RUL_CLIP-15], [15, RUL_CLIP+15], alpha=0.08, color="green")
        ax.set_xlim(-5, RUL_CLIP+5); ax.set_ylim(-5, RUL_CLIP+5); ax.set_aspect("equal")
        ax.set_title(f"{ds} — {m}\nRMSE={met['RMSE']:.1f} NASA={met['NASA Score']:,.0f}", fontsize=10)
plt.suptitle("Predicted vs Actual — v3", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/55_v3_scatter.png", bbox_inches="tight"); plt.close()
print(f"  [✓] 55")

# 56-57: LSTM params + evolution table
fig, ax = plt.subplots(figsize=(14, 5)); ax.axis("off")
cols = ["Zbiór", "hidden", "layers", "dense", "dropout", "lr", "batch", "seq", "loss", "RMSE", "NASA"]
rows = []
for ds in DATASETS:
    bp = all_results[ds]["LSTM"]["best_params"]; met = all_results[ds]["LSTM"]["metrics"]
    ln = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) if bp.get("use_huber") else "MSE"
    rows.append([ds, bp["hidden"], bp["n_layers"], bp["dense"], f"{bp['dropout']:.2f}",
                 f"{bp['lr']:.5f}", bp["batch_size"], bp["seq_length"], ln,
                 f"{met['RMSE']:.1f}", f"{met['NASA Score']:,.0f}"])
table = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.05, 2.0)
for j in range(len(cols)):
    table[0, j].set_facecolor("#37474F"); table[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Hiperparametry LSTM v3", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/56_v3_lstm_params.png", bbox_inches="tight"); plt.close()
print(f"  [✓] 56")

# 58-60: Parallel Coordinates + Importance + History (jeśli studies dostępne)
if all_studies:
    for ds_id in DATASETS:
        for model_name, params_list in [
            ("XGBoost", ["max_depth", "lr", "subsample", "colsample", "min_child_w", "reg_alpha", "reg_lambda", "gamma"]),
            ("LSTM", ["hidden", "n_layers", "dense", "dropout", "lr", "batch_size", "seq_length", "use_huber"]),
        ]:
            study = all_studies[ds_id].get(model_name)
            if not study: continue
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(trials) < 3: continue
            fig, ax = plt.subplots(figsize=(14, 6))
            pv = {p: [] for p in params_list}; ov = []
            for t in trials:
                ov.append(t.value)
                for p in params_list:
                    v = t.params.get(p, 0); pv[p].append(int(v) if isinstance(v, bool) else v)
            nd = []; 
            for p in params_list:
                vals = np.array(pv[p], dtype=float); mn, mx = vals.min(), vals.max()
                nd.append((vals - mn) / (mx - mn + 1e-10))
            oa = np.array(ov); omn, omx = oa.min(), oa.max()
            no = (oa - omn) / (omx - omn + 1e-10); nd.append(no)
            cmap = plt.cm.RdYlGn_r; xt = range(len(params_list) + 1)
            for i in range(len(trials)):
                c = cmap(no[i]); a = 0.8 if ov[i] <= np.percentile(ov, 20) else 0.15
                lw = 2.0 if a > 0.5 else 0.5
                ax.plot(xt, [n[i] for n in nd], color=c, alpha=a, lw=lw)
            ax.set_xticks(xt); ax.set_xticklabels(params_list + ["objective"], rotation=30, ha="right")
            ax.set_title(f"Parallel Coordinates — {model_name} {ds_id} ({len(trials)} prób)")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(omn, omx)); sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.8); plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/58_parallel_{model_name.lower()}_{ds_id}.png", bbox_inches="tight")
            plt.close()
    print(f"  [✓] 58 (parallel coordinates)")

    # Importance
    for ds_id in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for mi, (model_name, ax) in enumerate(zip(["XGBoost", "LSTM"], axes)):
            study = all_studies[ds_id].get(model_name)
            if not study: continue
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if len(trials) < 5: continue
            common_p = set(trials[0].params.keys())
            for t in trials: common_p &= set(t.params.keys())
            params = sorted(common_p); imp = {}; ov = np.array([t.value for t in trials])
            for p in params:
                pv = np.array([t.params[p] for t in trials], dtype=float)
                if pv.std() > 1e-10:
                    c = abs(np.corrcoef(pv, ov)[0, 1]); imp[p] = c if not np.isnan(c) else 0
                else: imp[p] = 0
            si = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            ax.barh(range(len(si)), [x[1] for x in si], color=COLORS[model_name], alpha=0.85)
            ax.set_yticks(range(len(si))); ax.set_yticklabels([x[0] for x in si], fontsize=9)
            ax.invert_yaxis(); ax.set_xlabel("|Korelacja|"); ax.set_title(f"{model_name} — {ds_id}")
            for bi, (_, v) in enumerate(si):
                ax.text(v + 0.01, bi, f"{v:.3f}", va="center", fontsize=9)
        plt.suptitle(f"Ważność hiperparametrów — {ds_id}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/59_importance_{ds_id}.png", bbox_inches="tight"); plt.close()
    print(f"  [✓] 59 (importance)")

    # History
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, ds_id in enumerate(DATASETS):
        ax = axes[idx // 2][idx % 2]
        for mn, clr in [("XGBoost", "#E64A19"), ("LSTM", "#2E7D32")]:
            study = all_studies[ds_id].get(mn)
            if not study: continue
            trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            vals = [t.value for t in trials]; best = []; cb = float("inf")
            for v in vals: cb = min(cb, v); best.append(cb)
            ls = " (RMSE)" if mn == "XGBoost" else " (MSE)"
            ax.plot(range(1, len(vals)+1), vals, "o", alpha=0.3, color=clr, markersize=3)
            ax.plot(range(1, len(best)+1), best, "-", color=clr, lw=2, label=f"{mn}{ls}")
        ax.set_xlabel("Nr próby"); ax.set_ylabel("Objective (CV)"); ax.set_title(ds_id); ax.legend(fontsize=9)
    plt.suptitle("Optimization History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/60_optuna_history.png", bbox_inches="tight"); plt.close()
    print(f"  [✓] 60 (history)")

print(f"\n  Wykresy w {PLOT_DIR}/")