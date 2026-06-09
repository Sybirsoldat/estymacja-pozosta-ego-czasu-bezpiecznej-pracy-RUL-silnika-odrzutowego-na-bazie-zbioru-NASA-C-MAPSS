"""
10d_robustness.py — Testy odporności (wykresy 74–78)
"""
from shared import *

print("=" * 70)
print("TESTY ODPORNOŚCI (74–78)")
print("=" * 70)

all_results = load_results()
preprocessed = load_preprocessed()
if all_results is None or preprocessed is None:
    print("BŁĄD: Brak wyników! Uruchom 10a_train.py"); sys.exit(1)

# ── 74: Szum sensorów ───────────────────────────────────────────────────────
print(f"\n  74 Szum sensorów...")
noise_levels = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]; pp = preprocessed[ds]; yt = all_results[ds]["y_test"]
    bp_l = all_results[ds]["LSTM"].get("best_params", {}); sl = bp_l.get("seq_length", 40)
    uids = pp["train_df"]["unit_id"].unique().copy()
    rng = np.random.RandomState(SEED); rng.shuffle(uids)
    tr_sub = pp["train_df"][pp["train_df"]["unit_id"].isin(uids[:int(len(uids)*0.8)])]
    Xtr, ytr = build_seqs(tr_sub, pp["seq_feat"], sl)
    Xte, yte = build_test_seqs(pp["test_df"], pp["seq_feat"], sl)
    mdl_l = train_lstm_quick(Xtr, ytr, Xte[:10], yte[:10], pp["n_features"], bp_l)
    mdl_l.eval()
    rmse_l = []
    for sigma in noise_levels:
        Xn = Xte if sigma == 0 else np.clip(Xte + np.random.RandomState(SEED).normal(0, sigma, Xte.shape), 0, 1).astype(np.float32)
        with torch.no_grad(): pred = mdl_l(torch.FloatTensor(Xn)).numpy()
        rmse_l.append(rmse(yte, np.clip(pred, 0, RUL_CLIP)))

    # XGBoost
    bp_x = all_results[ds]["XGBoost"].get("best_params", {})
    tr_xgb = create_enhanced_features(pp["train_df"], pp["xgb_base"])
    te_xgb = create_enhanced_features(pp["test_df"], pp["xgb_base"])
    fc = [c for c in tr_xgb.columns if c not in ["unit_id","cycle","RUL"]+pp["op_cols"]+pp["sensor_all"]]
    clean_inf(tr_xgb, fc); clean_inf(te_xgb, fc)
    tr_m = tr_xgb["unit_id"].isin(uids[:int(len(uids)*0.8)]); val_m = tr_xgb["unit_id"].isin(uids[int(len(uids)*0.8):])
    te_l = te_xgb.groupby("unit_id").last().reset_index()
    mdl_x = xgb.XGBRegressor(n_estimators=500, max_depth=bp_x.get("max_depth",4),
                               learning_rate=bp_x.get("lr",0.03), random_state=SEED,
                               n_jobs=-1, early_stopping_rounds=30)
    mdl_x.fit(tr_xgb[tr_m][fc].values, tr_xgb[tr_m]["RUL"].values,
              eval_set=[(tr_xgb[val_m][fc].values, tr_xgb[val_m]["RUL"].values)], verbose=0)
    Xtc = te_l[fc].values.astype(np.float32)
    rmse_x = []
    for sigma in noise_levels:
        Xn = Xtc if sigma == 0 else Xtc + np.random.RandomState(SEED).normal(0, sigma, Xtc.shape)
        rmse_x.append(rmse(yt, np.clip(mdl_x.predict(Xn.astype(np.float32)), 0, RUL_CLIP)))

    ax.plot(noise_levels, rmse_x, "o-", color=COLORS["XGBoost"], lw=2, label="XGBoost")
    ax.plot(noise_levels, rmse_l, "s-", color=COLORS["LSTM"], lw=2, label="LSTM")
    ax.set_xlabel("σ szumu"); ax.set_ylabel("RMSE"); ax.set_title(ds); ax.legend()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

plt.suptitle("Odporność na szum sensorów", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ROBUST_DIR}/74_noise_robustness.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 75: Dropout sensorów ─────────────────────────────────────────────────────
print(f"  75 Dropout sensorów...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx // 2][idx % 2]; pp = preprocessed[ds]
    bp_l = all_results[ds]["LSTM"].get("best_params", {}); sl = bp_l.get("seq_length", 40)
    uids = pp["train_df"]["unit_id"].unique().copy()
    rng = np.random.RandomState(SEED); rng.shuffle(uids)
    tr_sub = pp["train_df"][pp["train_df"]["unit_id"].isin(uids[:int(len(uids)*0.8)])]
    Xtr, ytr = build_seqs(tr_sub, pp["seq_feat"], sl)
    Xte, yte = build_test_seqs(pp["test_df"], pp["seq_feat"], sl)
    mdl = train_lstm_quick(Xtr, ytr, Xte[:10], yte[:10], pp["n_features"], bp_l)
    mdl.eval()
    with torch.no_grad(): rmse_base = rmse(yte, np.clip(mdl(torch.FloatTensor(Xte)).numpy(), 0, RUL_CLIP))
    deltas = []
    for si in range(pp["n_features"]):
        Xd = Xte.copy(); Xd[:,:,si] = 0
        with torch.no_grad(): r = rmse(yte, np.clip(mdl(torch.FloatTensor(Xd)).numpy(), 0, RUL_CLIP))
        deltas.append(r - rmse_base)
    si_sort = np.argsort(deltas)[::-1]
    colors_b = ["#F44336" if deltas[i]>1 else "#FF9800" if deltas[i]>0.5 else "#4CAF50" for i in si_sort]
    ax.barh(range(pp["n_features"]), [deltas[i] for i in si_sort], color=colors_b, alpha=0.85)
    ax.set_yticks(range(pp["n_features"])); ax.set_yticklabels([pp["seq_feat"][i] for i in si_sort], fontsize=7)
    ax.invert_yaxis(); ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("ΔRMSE"); ax.set_title(f"{ds} (base={rmse_base:.2f})")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

plt.suptitle("Dropout sensorów — który jest krytyczny?", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ROBUST_DIR}/75_sensor_dropout.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 76: Skrócona historia ────────────────────────────────────────────────────
print(f"  76 Skrócona historia...")
test_sls = [5, 10, 15, 20, 25, 30, 35, 40]
fig, ax = plt.subplots(figsize=(12, 6))
for ds in DATASETS:
    pp = preprocessed[ds]; bp_l = all_results[ds]["LSTM"].get("best_params", {})
    full_sl = bp_l.get("seq_length", 40)
    uids = pp["train_df"]["unit_id"].unique().copy()
    rng = np.random.RandomState(SEED); rng.shuffle(uids)
    tr_sub = pp["train_df"][pp["train_df"]["unit_id"].isin(uids[:int(len(uids)*0.8)])]
    Xtr, ytr = build_seqs(tr_sub, pp["seq_feat"], full_sl)
    Xte, yte = build_test_seqs(pp["test_df"], pp["seq_feat"], full_sl)
    mdl = train_lstm_quick(Xtr, ytr, Xte[:10], yte[:10], pp["n_features"], bp_l)
    mdl.eval(); rmses = []
    for tsl in test_sls:
        if tsl > full_sl: rmses.append(np.nan); continue
        Xs = np.zeros_like(Xte); Xs[:, full_sl-tsl:, :] = Xte[:, full_sl-tsl:, :]
        with torch.no_grad(): pred = mdl(torch.FloatTensor(Xs)).numpy()
        rmses.append(rmse(yte, np.clip(pred, 0, RUL_CLIP)))
    ax.plot(test_sls, rmses, "o-", lw=2, markersize=6, label=ds)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
ax.set_xlabel("Timestepów"); ax.set_ylabel("RMSE")
ax.set_title("Ile historii LSTM potrzebuje?"); ax.legend()
plt.tight_layout(); plt.savefig(f"{ROBUST_DIR}/76_history_length.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 77: Bootstrap CI ─────────────────────────────────────────────────────────
print(f"  77 Bootstrap CI...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ds in enumerate(DATASETS):
    ax = axes[idx//2][idx%2]; yt = all_results[ds]["y_test"]; n = len(yt)
    for m in ["XGBoost", "LSTM"]:
        yp = all_results[ds][m]["y_pred"]; rng_b = np.random.RandomState(SEED)
        br = [rmse(yt[ix:=rng_b.choice(n,n,replace=True)], yp[ix]) for _ in range(1000)]
        br = np.array(br); ci_l, ci_h = np.percentile(br, [2.5, 97.5])
        ax.hist(br, bins=40, alpha=0.5, color=COLORS[m],
                label=f"{m}: {np.mean(br):.2f} [{ci_l:.2f}, {ci_h:.2f}]")
        ax.axvline(np.mean(br), color=COLORS[m], lw=2)
        ax.axvline(ci_l, color=COLORS[m], lw=1.5, ls="--", alpha=0.7)
        ax.axvline(ci_h, color=COLORS[m], lw=1.5, ls="--", alpha=0.7)
        print(f"    {ds} {m}: {np.mean(br):.2f} [{ci_l:.2f}, {ci_h:.2f}]")
    ax.set_xlabel("RMSE"); ax.set_title(ds); ax.legend(fontsize=8)
plt.suptitle("Bootstrap 95% CI (1000 prób)", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{ROBUST_DIR}/77_bootstrap_ci.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

# ── 78: Cross-dataset ────────────────────────────────────────────────────────
print(f"  78 Cross-dataset generalizacja...")
cross_pairs = [("FD001","FD003"),("FD003","FD001"),("FD002","FD004"),("FD004","FD002")]
cross_res = []
for tr_ds, te_ds in cross_pairs:
    pp_tr, pp_te = preprocessed[tr_ds], preprocessed[te_ds]
    common = sorted(set(pp_tr["seq_feat"]) & set(pp_te["seq_feat"]))
    if len(common) < 5: cross_res.append({"train":tr_ds,"test":te_ds,"rmse_cross":np.nan}); continue
    bp = all_results[tr_ds]["LSTM"].get("best_params", {}); sl = bp.get("seq_length", 40)
    uids = pp_tr["train_df"]["unit_id"].unique().copy()
    rng = np.random.RandomState(SEED); rng.shuffle(uids)
    tr_sub = pp_tr["train_df"][pp_tr["train_df"]["unit_id"].isin(uids[:int(len(uids)*0.8)])]
    Xtr, ytr = build_seqs(tr_sub, common, sl)
    Xte, yte = build_test_seqs(pp_te["test_df"], common, sl)
    mdl = train_lstm_quick(Xtr, ytr, Xte[:10], yte[:10], len(common), bp)
    mdl.eval()
    with torch.no_grad(): pred = mdl(torch.FloatTensor(Xte)).numpy()
    rc = rmse(yte, np.clip(pred, 0, RUL_CLIP))
    rn = all_results[te_ds]["LSTM"]["metrics"]["RMSE"]
    cross_res.append({"train":tr_ds,"test":te_ds,"rmse_cross":rc,"rmse_native":rn})
    print(f"    {tr_ds}→{te_ds}: RMSE={rc:.2f} (natywny={rn:.2f}, Δ={(rc-rn)/rn*100:+.1f}%)")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

fig, ax = plt.subplots(figsize=(12, 6))
valid = [r for r in cross_res if not np.isnan(r.get("rmse_cross", np.nan))]
if valid:
    labels = [f"{r['train']}→{r['test']}" for r in valid]
    rn = [r["rmse_native"] for r in valid]; rc = [r["rmse_cross"] for r in valid]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x-w/2, rn, w, label="Natywny", color="#4CAF50", alpha=0.85)
    ax.bar(x+w/2, rc, w, label="Cross-dataset", color="#F44336", alpha=0.85)
    for i in range(len(labels)):
        d = (rc[i]-rn[i])/rn[i]*100; c = "#F44336" if d > 0 else "#4CAF50"
        ax.text(x[i]+w/2, rc[i]+0.3, f"{d:+.1f}%", ha="center", fontsize=9, color=c, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15); ax.set_ylabel("RMSE")
    ax.set_title("Cross-dataset generalizacja LSTM"); ax.legend()
plt.tight_layout(); plt.savefig(f"{ROBUST_DIR}/78_cross_dataset.png", bbox_inches="tight"); plt.close()
print(f"    [✓]")

print(f"\n  Wykresy 74–78 w {ROBUST_DIR}/")