"""
10a_train.py — Fazy 1–3: Preprocessing + XGBoost + LSTM
Zapis wyników do ./results_optuna_v3/
"""
from shared import *

print("=" * 70)
print("FAZY 1–3: TRENING (XGBoost + LSTM)")
print("=" * 70)

if USE_GPU:
    print(f"[✓] GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
print(f"  Device: {device}")

data_path = get_data_path()
if data_path is None:
    print("BŁĄD: Nie znaleziono danych C-MAPSS!")
    sys.exit(1)
print(f"  Dane: {data_path}")

# ── FAZA 1: Preprocessing ────────────────────────────────────────────────────
print(f"\n{'='*70}\nFAZA 1: Preprocessing\n{'='*70}")

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

save_preprocessed(preprocessed)

# ── FAZA 2: XGBoost ──────────────────────────────────────────────────────────
print(f"\n{'='*70}\nFAZA 2: XGBoost + Optuna\n{'='*70}")

def optimize_xgboost_cv(train_df, test_df, xgb_base, sensor_all, op_cols,
                        n_trials=50):
    tr_xgb = create_enhanced_features(train_df, xgb_base)
    te_xgb = create_enhanced_features(test_df, xgb_base)
    feat_cols = [c for c in tr_xgb.columns
                 if c not in ["unit_id", "cycle", "RUL"] + op_cols + sensor_all]
    clean_inf(tr_xgb, feat_cols)
    clean_inf(te_xgb, feat_cols)
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
            "random_state": SEED, "n_jobs": -1, "early_stopping_rounds": 30,
        }
        scores = []
        for tr_uids, val_uids in folds:
            tr_m = tr_xgb["unit_id"].isin(tr_uids)
            val_m = tr_xgb["unit_id"].isin(val_uids)
            m = xgb.XGBRegressor(**p)
            m.fit(tr_xgb[tr_m][feat_cols].values, tr_xgb[tr_m]["RUL"].values,
                  eval_set=[(tr_xgb[val_m][feat_cols].values,
                             tr_xgb[val_m]["RUL"].values)], verbose=0)
            pred = np.clip(m.predict(tr_xgb[val_m][feat_cols].values), 0, RUL_CLIP)
            scores.append(rmse(tr_xgb[val_m]["RUL"].values, pred))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    final_p = {
        "n_estimators": 1000, "max_depth": bp["max_depth"],
        "learning_rate": bp["lr"], "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample"], "min_child_weight": bp["min_child_w"],
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
    return metrics, y_pred, train_time, bp, y_test, study

all_results = {}
all_studies = {}

for ds_id in DATASETS:
    print(f"\n  ── {ds_id} ──")
    pp = preprocessed[ds_id]
    print(f"    Feature engineering...")
    print(f"    Optuna XGBoost: {XGB_TRIALS} prób × {K_FOLDS} foldów...")
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

# ── FAZA 3: LSTM ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}\nFAZA 3: LSTM + Optuna + Transfer + Augmentacja\n{'='*70}")

def train_lstm_once_full(X_tr, y_tr, X_val, y_val, n_features,
                         hidden, n_layers, dense, dropout, lr, batch_size,
                         use_huber, huber_delta,
                         epochs=80, patience=15, seed=42, trial=None,
                         use_augmentation=False, pretrained_state=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_augmentation:
        X_tr, y_tr = augment_sequences(X_tr, y_tr, AUG_NOISE_STD, AUG_COPIES)
    model = FlexLSTM(n_features, hidden, n_layers, dense, dropout).to(device)
    if pretrained_state is not None:
        try: model.load_state_dict(pretrained_state, strict=False)
        except RuntimeError: pass
    Xt = torch.FloatTensor(X_tr).to(device)
    yt_t = torch.FloatTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)
    loader = DataLoader(TensorDataset(Xt, yt_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.3, div_factor=10, final_div_factor=100)
    criterion = nn.HuberLoss(delta=huber_delta) if use_huber else nn.MSELoss()
    mse_fn = nn.MSELoss()
    best_val, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        model.eval()
        vms, vc = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vms += mse_fn(model(Xb), yb).item() * len(yb)
                vc += len(yb)
        val_mse = vms / vc
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if trial and epoch % 10 == 0:
            trial.report(val_mse, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
        if no_improve >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, best_val

def optimize_lstm_cv(train_df, test_df, seq_feat, n_features,
                     n_trials=40, pretrained_state=None):
    folds = kfold_unit_splits(train_df, k=K_FOLDS, seed=SEED)
    seq_cache, test_seq_cache = {}, {}
    def get_fold_data(fi, sl):
        key = (fi, sl)
        if key not in seq_cache:
            tr_u, val_u = folds[fi]
            seq_cache[key] = (*build_seqs(train_df[train_df["unit_id"].isin(tr_u)], seq_feat, sl),
                              *build_seqs(train_df[train_df["unit_id"].isin(val_u)], seq_feat, sl))
        return seq_cache[key]
    def get_test_data(sl):
        if sl not in test_seq_cache:
            test_seq_cache[sl] = build_test_seqs(test_df, seq_feat, sl)
        return test_seq_cache[sl]

    def objective(trial):
        h = trial.suggest_categorical("hidden", [64, 128])
        nl = trial.suggest_int("n_layers", 1, 2)
        d = trial.suggest_categorical("dense", [32, 64, 128])
        dr = trial.suggest_float("dropout", 0.15, 0.45)
        lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [64, 128, 256])
        sl = trial.suggest_categorical("seq_length", [30, 40])
        uh = trial.suggest_categorical("use_huber", [True, False])
        hd = trial.suggest_float("huber_delta", 5.0, 20.0) if uh else 1.0
        scores = []
        for fi in range(K_FOLDS):
            Xtr, ytr, Xv, yv = get_fold_data(fi, sl)
            _, vl = train_lstm_once_full(Xtr, ytr, Xv, yv, n_features,
                                         h, nl, d, dr, lr, bs, uh, hd,
                                         epochs=80, patience=12, seed=SEED,
                                         trial=trial, pretrained_state=pretrained_state)
            scores.append(vl)
            if USE_GPU: torch.cuda.empty_cache()
        return np.mean(scores)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(RuntimeError,))

    bp = study.best_params
    sl = bp["seq_length"]
    ln = "Huber(d={:.1f})".format(bp.get("huber_delta", 1)) if bp["use_huber"] else "MSE"
    print(f"    Best: hidden={bp['hidden']}, layers={bp['n_layers']}, "
          f"dense={bp['dense']}, drop={bp['dropout']:.2f}, "
          f"lr={bp['lr']:.5f}, bs={bp['batch_size']}, seq={sl}, loss={ln}")
    n_pr = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"    Pruned: {n_pr}/{n_trials}")

    Xte, yte = get_test_data(sl)
    tr_u, val_u = folds[-1]
    Xtr_f, ytr_f = build_seqs(train_df[train_df["unit_id"].isin(tr_u)], seq_feat, sl)
    Xv_l, yv_l = build_seqs(train_df[train_df["unit_id"].isin(val_u)], seq_feat, sl)

    print(f"    Ensemble: {len(ENSEMBLE_SEEDS)} modeli...")
    t0 = time.time()
    preds = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        mdl, _ = train_lstm_once_full(
            Xtr_f, ytr_f, Xv_l, yv_l, n_features,
            bp["hidden"], bp["n_layers"], bp["dense"], bp["dropout"],
            bp["lr"], bp["batch_size"], bp["use_huber"], bp.get("huber_delta", 1.0),
            epochs=120, patience=18, seed=seed,
            use_augmentation=True, pretrained_state=pretrained_state)
        mdl.eval()
        tl = DataLoader(TensorDataset(torch.FloatTensor(Xte).to(device),
                         torch.zeros(len(Xte)).to(device)), batch_size=256, shuffle=False)
        pp_list = []
        with torch.no_grad():
            for Xb, _ in tl: pp_list.append(mdl(Xb).cpu().numpy())
        pred = np.clip(np.concatenate(pp_list), 0, RUL_CLIP)
        preds.append(pred)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"      Model {i+1}/{len(ENSEMBLE_SEEDS)}: RMSE={rmse(yte, pred):.2f}")
        del mdl
        if USE_GPU: torch.cuda.empty_cache()

    train_time = time.time() - t0
    y_ens = np.clip(np.mean(preds, axis=0), 0, RUL_CLIP)
    metrics = evaluate(yte, y_ens)
    print(f"    Ensemble RMSE={metrics['RMSE']:.2f}")
    n_params = sum(p.numel() for p in FlexLSTM(
        n_features, bp["hidden"], bp["n_layers"], bp["dense"], bp["dropout"]).parameters())
    return metrics, y_ens, train_time, bp, n_params, yte, study

def pretrain_lstm(train_df, seq_feat, n_features, bp, sl):
    folds = kfold_unit_splits(train_df, k=K_FOLDS, seed=SEED)
    tr_u, val_u = folds[-1]
    Xtr, ytr = build_seqs(train_df[train_df["unit_id"].isin(tr_u)], seq_feat, sl)
    Xv, yv = build_seqs(train_df[train_df["unit_id"].isin(val_u)], seq_feat, sl)
    mdl, _ = train_lstm_once_full(
        Xtr, ytr, Xv, yv, n_features,
        bp["hidden"], bp["n_layers"], bp["dense"], bp["dropout"],
        bp["lr"], bp["batch_size"], bp.get("use_huber", False),
        bp.get("huber_delta", 1.0), epochs=80, patience=12, seed=SEED,
        use_augmentation=True)
    return {k: v.clone() for k, v in mdl.state_dict().items()}

process_order = ["FD001", "FD003", "FD002", "FD004"]
pretrained_states = {}

for ds_id in process_order:
    print(f"\n  ── {ds_id} ──")
    pp = preprocessed[ds_id]
    n_feat = pp["n_features"]
    source_id = TRANSFER_MAP[ds_id]
    source_pp = preprocessed[source_id]
    can_tf = (source_id in pretrained_states and source_pp["n_features"] == n_feat)
    pt_state = pretrained_states.get(source_id) if can_tf else None
    if can_tf: print(f"    Transfer: {source_id} → {ds_id}")
    else: print(f"    Bez transfer")

    print(f"    Optuna LSTM: {LSTM_TRIALS} prób × {K_FOLDS} foldów...")
    m_lstm, p_lstm, t_lstm, bp_lstm, n_params, yte, study_lstm = optimize_lstm_cv(
        pp["train_df"], pp["test_df"], pp["seq_feat"], n_feat,
        n_trials=LSTM_TRIALS, pretrained_state=pt_state)
    all_results[ds_id]["LSTM"] = {
        "metrics": m_lstm, "y_pred": p_lstm,
        "time": t_lstm, "best_params": bp_lstm, "n_params": n_params}
    all_studies[ds_id]["LSTM"] = study_lstm
    print(f"    ★ RMSE={m_lstm['RMSE']:.2f}  NASA={m_lstm['NASA Score']:,.0f}")

    print(f"    Pretraining {ds_id}...")
    pretrained_states[ds_id] = pretrain_lstm(
        pp["train_df"], pp["seq_feat"], n_feat, bp_lstm, bp_lstm["seq_length"])

# Zapis
save_results(all_results)
save_studies(all_studies)

# Podsumowanie
print(f"\n{'='*70}\nPODSUMOWANIE TRENINGU\n{'='*70}")
for ds in DATASETS:
    mx = all_results[ds]["XGBoost"]["metrics"]
    ml = all_results[ds]["LSTM"]["metrics"]
    print(f"  {ds}: XGB RMSE={mx['RMSE']:.1f} NASA={mx['NASA Score']:,.0f}  "
          f"| LSTM RMSE={ml['RMSE']:.1f} NASA={ml['NASA Score']:,.0f}")
total_t = sum(all_results[ds][m]["time"] for ds in DATASETS for m in ["XGBoost", "LSTM"])
print(f"\n  Łączny czas: {total_t:.0f}s ({total_t/60:.1f} min)")