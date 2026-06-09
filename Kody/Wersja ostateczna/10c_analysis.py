"""
10c_analysis.py — Analiza i walidacja wyników (wykresy 61–73)
"""
from shared import *
from scipy import stats
from sklearn.metrics import roc_curve, auc as sk_auc

print("=" * 70)
print("ANALIZA I WALIDACJA (61–73)")
print("=" * 70)

all_results = load_results()
preprocessed = load_preprocessed()
if all_results is None:
    print("BŁĄD: Brak wyników! Uruchom 10a_train.py"); sys.exit(1)

data_path = get_data_path()

# ── 61: Accuracy Window ─────────────────────────────────────────────────────
print(f"\n  61 Accuracy Window...")
windows = [5, 10, 15, 20]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]; yt = all_results[ds]["y_test"]
    for mi, m in enumerate(["XGBoost", "LSTM"]):
        yp = all_results[ds][m]["y_pred"]
        accs = [100 * np.mean(np.abs(yp - yt) <= w) for w in windows]
        x = np.arange(len(windows))
        bars = ax.bar(x + mi * 0.35, accs, 0.35, label=m, color=COLORS[m], alpha=0.85)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                    f"{acc:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x + 0.175); ax.set_xticklabels([f"±{w}" for w in windows])
    ax.set_ylabel("% w oknie"); ax.set_title(ds); ax.set_ylim(0, 105); ax.legend()
plt.suptitle("Accuracy Window", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/61_accuracy_window.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 62: Rezyduały ────────────────────────────────────────────────────────────
print(f"  62 Rezyduały...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]; yt = all_results[ds]["y_test"]
    for m in ["XGBoost", "LSTM"]:
        yp = all_results[ds][m]["y_pred"]
        ax.scatter(yt, yp - yt, alpha=0.5, s=25, color=COLORS[m], label=m, edgecolors="none")
    ax.axhline(0, color="black", lw=1.5, ls="--", alpha=0.5)
    ax.fill_between([0, RUL_CLIP], -15, 15, alpha=0.05, color="green")
    ax.set_xlabel("Rzeczywisty RUL"); ax.set_ylabel("Błąd"); ax.set_title(ds); ax.legend(fontsize=8)
plt.suptitle("Analiza rezyduów", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/62_residual_analysis.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 63: Wilcoxon ─────────────────────────────────────────────────────────────
print(f"  63 Wilcoxon...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]; yt = all_results[ds]["y_test"]
    ex = np.abs(all_results[ds]["XGBoost"]["y_pred"] - yt)
    el = np.abs(all_results[ds]["LSTM"]["y_pred"] - yt)
    stat, pval = stats.wilcoxon(ex, el)
    sig = "★ TAK" if pval < 0.05 else "NIE"
    better = "LSTM" if np.median(el) < np.median(ex) else "XGBoost"
    parts = ax.violinplot([ex, el], positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]): pc.set_facecolor([COLORS["XGBoost"], COLORS["LSTM"]][i]); pc.set_alpha(0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["XGBoost", "LSTM"])
    ax.set_title(f"{ds}: p={pval:.4f} → {sig}, lepszy: {better}")
    print(f"    {ds}: p={pval:.4f} → {sig}, lepszy: {better}")
plt.suptitle("Test Wilcoxona", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/63_wilcoxon_test.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 64: Koszt ────────────────────────────────────────────────────────────────
print(f"  64 Analiza kosztowa...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]; yt = all_results[ds]["y_test"]
    ml, ev, lv = [], [], []
    for m in ["XGBoost", "LSTM"]:
        d = all_results[ds][m]["y_pred"] - yt
        ec = np.sum(np.abs(d[d < 0])); lc = np.sum(d[d >= 0]) * 10
        ml.append(m); ev.append(ec); lv.append(lc)
    x = np.arange(2)
    ax.bar(x, ev, 0.5, label="Wczesna (×1)", color="#4CAF50", alpha=0.8)
    ax.bar(x, lv, 0.5, bottom=ev, label="Awaria (×10)", color="#F44336", alpha=0.8)
    for i in range(2): ax.text(i, ev[i]+lv[i]+50, f"Σ={ev[i]+lv[i]:,.0f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ml); ax.set_title(ds); ax.legend(fontsize=8)
plt.suptitle("Analiza kosztowa", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/64_cost_analysis.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 65: SHAP ─────────────────────────────────────────────────────────────────
print(f"  65 SHAP...")
try:
    install_if_missing("shap"); import shap
    if preprocessed and data_path:
        for ds_s in DATASETS:
            pp = preprocessed[ds_s]; bp = all_results[ds_s]["XGBoost"].get("best_params", {})
            if not bp: continue
            tr_x = create_enhanced_features(pp["train_df"], pp["xgb_base"])
            te_x = create_enhanced_features(pp["test_df"], pp["xgb_base"])
            fc = [c for c in tr_x.columns if c not in ["unit_id","cycle","RUL"]+pp["op_cols"]+pp["sensor_all"]]
            clean_inf(tr_x, fc); clean_inf(te_x, fc)
            uids = pp["train_df"]["unit_id"].unique(); rng = np.random.RandomState(SEED); rng.shuffle(uids)
            sp = int(len(uids)*0.8); tr_m = tr_x["unit_id"].isin(uids[:sp]); val_m = tr_x["unit_id"].isin(uids[sp:])
            te_l = te_x.groupby("unit_id").last().reset_index()
            mdl = xgb.XGBRegressor(n_estimators=500, max_depth=bp.get("max_depth",4),
                                    learning_rate=bp.get("lr",0.03), base_score=0.5,
                                    random_state=SEED, n_jobs=-1, early_stopping_rounds=30)
            mdl.fit(tr_x[tr_m][fc].values, tr_x[tr_m]["RUL"].values,
                    eval_set=[(tr_x[val_m][fc].values, tr_x[val_m]["RUL"].values)], verbose=0)
            exp = shap.TreeExplainer(mdl.get_booster())
            sv = exp.shap_values(te_l[fc].values.astype(np.float32))
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, te_l[fc].values, feature_names=fc, max_display=15, show=False)
            plt.title(f"SHAP — XGBoost {ds_s}"); plt.tight_layout()
            plt.savefig(f"{ANALYSIS_DIR}/65_shap_{ds_s}.png", bbox_inches="tight"); plt.close()
            print(f"    [✓] 65_shap_{ds_s}")
except Exception as e:
    print(f"    [!] SHAP error: {e}")

# ── 66: Saliency ─────────────────────────────────────────────────────────────
print(f"  66 Gradient Saliency...")
if preprocessed:
    for ds_s in DATASETS:
        bp = all_results[ds_s]["LSTM"].get("best_params", {}); pp = preprocessed[ds_s]
        if not bp: continue
        sl = bp.get("seq_length", 40); n_f = pp["n_features"]
        uids = pp["train_df"]["unit_id"].unique(); rng = np.random.RandomState(SEED); rng.shuffle(uids)
        tr_sub = pp["train_df"][pp["train_df"]["unit_id"].isin(uids[:int(len(uids)*0.8)])]
        Xtr, ytr = build_seqs(tr_sub, pp["seq_feat"], sl)
        Xte, yte = build_test_seqs(pp["test_df"], pp["seq_feat"], sl)
        mdl = train_lstm_quick(Xtr, ytr, Xte[:10], yte[:10], n_f, bp)
        mdl.eval(); X_s = torch.FloatTensor(Xte); X_s.requires_grad_(True)
        mdl(X_s).sum().backward()
        sal = X_s.grad.abs().numpy(); avg = sal.mean(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im = axes[0].imshow(avg.T, aspect="auto", cmap="hot", interpolation="nearest")
        axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Sensor"); axes[0].set_title("Heatmap")
        plt.colorbar(im, ax=axes[0], shrink=0.8)
        si = avg.mean(axis=0); sidx = np.argsort(si)[::-1]
        axes[1].barh(range(n_f), si[sidx], color="#2E7D32", alpha=0.85)
        axes[1].set_yticks(range(n_f)); axes[1].set_yticklabels([pp["seq_feat"][i] for i in sidx], fontsize=7)
        axes[1].invert_yaxis(); axes[1].set_title("Per sensor")
        ti = avg.mean(axis=1); axes[2].bar(range(sl), ti, color="#2E7D32", alpha=0.85)
        axes[2].set_xlabel("Timestep"); axes[2].set_title("Per timestep")
        plt.suptitle(f"Saliency — LSTM {ds_s}", fontsize=13, fontweight="bold")
        plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/66_saliency_{ds_s}.png", bbox_inches="tight"); plt.close()
        print(f"    [✓] 66_saliency_{ds_s}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── 67: Degradation ──────────────────────────────────────────────────────────
print(f"  67 Krzywe degradacji...")
if preprocessed:
    for ds_d in ["FD001", "FD003"]:
        bp = all_results[ds_d]["LSTM"].get("best_params", {}); pp = preprocessed[ds_d]
        if not bp: continue
        sl = bp.get("seq_length", 40)
        uids = pp["train_df"]["unit_id"].unique(); rng = np.random.RandomState(SEED); rng.shuffle(uids)
        tr_sub = pp["train_df"][pp["train_df"]["unit_id"].isin(uids[:int(len(uids)*0.8)])]
        Xtr, ytr = build_seqs(tr_sub, pp["seq_feat"], sl)
        mdl = train_lstm_quick(Xtr, ytr, Xtr[:10], ytr[:10], pp["n_features"], bp)
        mdl.eval()
        test_ruls = pp["test_df"].groupby("unit_id")["RUL"].min().sort_values()
        selected = list(test_ruls.iloc[::max(1, len(test_ruls)//6)].index[:6])
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i, uid in enumerate(selected):
            ax = axes.flatten()[i]
            unit = pp["test_df"][pp["test_df"]["unit_id"]==uid].sort_values("cycle")
            actual = unit["RUL"].values; cycles = unit["cycle"].values
            data = unit[pp["seq_feat"]].values; preds = []
            for t in range(len(data)):
                if t < sl: seq = np.vstack([np.zeros((sl-t-1, pp["n_features"])), data[:t+1]])
                else: seq = data[t-sl+1:t+1]
                with torch.no_grad(): preds.append(np.clip(mdl(torch.FloatTensor(seq).unsqueeze(0)).item(), 0, RUL_CLIP))
            ax.plot(cycles, actual, "b-", lw=2, label="Rzeczywisty")
            ax.plot(cycles, preds, "r-", lw=1.5, alpha=0.8, label="Predykcja")
            ax.set_title(f"Silnik {uid} (RUL={actual[-1]:.0f})"); ax.legend(fontsize=8)
        plt.suptitle(f"Krzywe degradacji — {ds_d}", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/67_degradation_{ds_d}.png", bbox_inches="tight"); plt.close()
        print(f"    [✓] 67_degradation_{ds_d}")

# ── 68–73: CDF, Box, Bland-Altman, Agreement, ROC, Radar ────────────────────
print(f"  68 CDF...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]; yt = all_results[ds]["y_test"]
    for m in ["XGBoost", "LSTM"]:
        yp = all_results[ds][m]["y_pred"]; es = np.sort(np.abs(yp-yt))
        ax.plot(es, np.arange(1,len(es)+1)/len(es)*100, lw=2, color=COLORS[m], label=m)
    for th in [10,15,20]: ax.axvline(th, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("|Błąd|"); ax.set_ylabel("% ≤ X"); ax.set_title(ds); ax.set_xlim(0,60); ax.legend()
plt.suptitle("CDF błędów", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/68_cdf_errors.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"  69 Box ploty per RUL range...")
rul_bins = [(0,25,"0-25"),(25,50,"25-50"),(50,75,"50-75"),(75,100,"75-100"),(100,126,"100-125")]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]; yt = all_results[ds]["y_test"]
    pos, lbl, dx, dl = [], [], [], []
    p = 0
    for lo, hi, lb in rul_bins:
        mask = (yt >= lo) & (yt < hi)
        if mask.sum() < 3: continue
        lbl.append(lb); pos.append(p); p += 1
        for m, d in [("XGBoost",dx),("LSTM",dl)]: d.append(np.abs(all_results[ds][m]["y_pred"][mask]-yt[mask]))
    if dx: ax.boxplot(dx, positions=np.array(pos)-0.2, widths=0.35, patch_artist=True,
                       boxprops=dict(facecolor=COLORS["XGBoost"], alpha=0.6), medianprops=dict(color="black"))
    if dl: ax.boxplot(dl, positions=np.array(pos)+0.2, widths=0.35, patch_artist=True,
                       boxprops=dict(facecolor=COLORS["LSTM"], alpha=0.6), medianprops=dict(color="black"))
    ax.set_xticks(pos); ax.set_xticklabels(lbl); ax.set_title(ds)
plt.suptitle("Błąd per zakres RUL", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/69_error_per_rul.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"  70 Bland-Altman...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]
    px, pl = all_results[ds]["XGBoost"]["y_pred"], all_results[ds]["LSTM"]["y_pred"]
    mp, dp = (px+pl)/2, px-pl; md, sd = np.mean(dp), np.std(dp)
    ax.scatter(mp, dp, alpha=0.6, s=30, c="#9C27B0", edgecolors="none")
    ax.axhline(md, color="blue", lw=2, label=f"Bias={md:.1f}")
    ax.axhline(md+1.96*sd, color="red", lw=1.5, ls="--", label=f"+1.96σ={md+1.96*sd:.1f}")
    ax.axhline(md-1.96*sd, color="red", lw=1.5, ls="--", label=f"−1.96σ={md-1.96*sd:.1f}")
    ax.set_xlabel("Średnia"); ax.set_ylabel("XGB − LSTM"); ax.set_title(ds); ax.legend(fontsize=8)
plt.suptitle("Bland-Altman", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/70_bland_altman.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"  71 Mapa zgodności...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]; yt = all_results[ds]["y_test"]
    ex = np.abs(all_results[ds]["XGBoost"]["y_pred"]-yt) <= 15
    el = np.abs(all_results[ds]["LSTM"]["y_pred"]-yt) <= 15
    colors = ["#4CAF50" if ex[i] and el[i] else "#2196F3" if el[i] else "#FF9800" if ex[i] else "#F44336" for i in range(len(yt))]
    ax.bar(range(1,len(yt)+1), yt, color=colors, alpha=0.7, width=0.8)
    ax.set_xlabel("Nr silnika"); ax.set_title(ds)
    if idx == 0:
        ax.legend(handles=[Patch(facecolor="#4CAF50",alpha=0.7,label="Oba OK"),
                           Patch(facecolor="#2196F3",alpha=0.7,label="Tylko LSTM"),
                           Patch(facecolor="#FF9800",alpha=0.7,label="Tylko XGB"),
                           Patch(facecolor="#F44336",alpha=0.7,label="Oba źle")], fontsize=7)
plt.suptitle("Mapa zgodności (±15)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/71_agreement_map.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"  72 ROC/AUC...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]; yt = all_results[ds]["y_test"]; yb = (yt <= 30).astype(int)
    for m in ["XGBoost", "LSTM"]:
        sc = np.clip(1-all_results[ds][m]["y_pred"]/RUL_CLIP, 0, 1)
        fpr, tpr, _ = roc_curve(yb, sc); a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=COLORS[m], label=f"{m} (AUC={a:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.3); ax.set_title(f"{ds}"); ax.legend(fontsize=9)
plt.suptitle("ROC/AUC — alert ≤30 cykli", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/72_roc_auc.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"  73 Radar...")
fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
rm = ["RMSE", "MAE", "NASA Score", "R²"]
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]
    raw = {m: [all_results[ds][m]["metrics"][r] for r in rm] for m in ["XGBoost","LSTM"]}
    norm = {}
    for m in raw:
        n = []
        for i, r in enumerate(rm):
            vals = [raw[mm][i] for mm in raw]; mn, mx = min(vals), max(vals); v = raw[m][i]
            if mx == mn: n.append(1.0)
            elif r == "R²": n.append((v-mn)/(mx-mn))
            else: n.append(1-(v-mn)/(mx-mn))
        norm[m] = n
    angles = np.linspace(0, 2*np.pi, len(rm), endpoint=False).tolist(); angles += angles[:1]
    for m in norm:
        vals = norm[m] + norm[m][:1]
        ax.plot(angles, vals, "o-", lw=2, label=m, color=COLORS[m]); ax.fill(angles, vals, alpha=0.15, color=COLORS[m])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(rm); ax.set_ylim(0,1.15)
    ax.set_title(ds, fontsize=12, fontweight="bold", y=1.08); ax.legend(fontsize=9, bbox_to_anchor=(1.3,1.1))
plt.suptitle("Radar", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.savefig(f"{ANALYSIS_DIR}/73_radar.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"\n  Wykresy 61–73 w {ANALYSIS_DIR}/")