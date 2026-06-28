"""
Wave 2 Refined Experiments
===========================
Focus: improve on Two-Stage-EloOnly (0.8201 best so far).

Key insight: draws happen more when teams are evenly matched (|elo_diff| small).
Refined two-stage explicitly uses |elo_diff| for draw propensity.

Also: calibration curves / ECE for the best models.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"

CLASS_ORDER = ["H", "D", "A"]
CLASS_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


# ── Temperature scaling ───────────────────────────────────────────────────────

class TemperatureScaling:
    def __init__(self):
        self.T_ = 1.0

    def fit(self, probs, y):
        def nll(log_T):
            T = np.exp(log_T[0])
            logits = np.log(probs + 1e-10)
            scaled = logits / T
            scaled -= scaled.max(1, keepdims=True)
            exp_s = np.exp(scaled)
            p = exp_s / exp_s.sum(1, keepdims=True)
            return log_loss(y, p)
        res = minimize(nll, [0.0], method="Nelder-Mead",
                       options={"maxiter": 200, "xatol": 1e-6})
        self.T_ = np.exp(res.x[0])
        return self

    def transform(self, probs):
        logits = np.log(probs + 1e-10)
        scaled = logits / self.T_
        scaled -= scaled.max(1, keepdims=True)
        exp_s = np.exp(scaled)
        return exp_s / exp_s.sum(1, keepdims=True)


# ── CV harness ────────────────────────────────────────────────────────────────

def run_cv(df, name, fn, n_splits=5, n_repeats=10, seed=0):
    y = np.array([CLASS_IDX[v] for v in df["outcome"]])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    losses, accs = [], []
    oof_probs = np.zeros((len(df), 3))
    oof_counts = np.zeros(len(df))

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(df, y)):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]
        try:
            probs = fn(df_tr, df_val)
        except Exception as e:
            print(f"  [fold {fold_idx}] ERROR: {e}")
            probs = np.ones((len(df_val), 3)) / 3.0
        probs = np.clip(probs, 1e-7, 1.0)
        probs /= probs.sum(1, keepdims=True)
        losses.append(log_loss(y_val, probs, labels=[0, 1, 2]))
        accs.append(accuracy_score(y_val, probs.argmax(1)))
        oof_probs[val_idx] += probs
        oof_counts[val_idx] += 1

    oof_probs /= np.maximum(oof_counts[:, None], 1)
    oof_probs = np.clip(oof_probs, 1e-7, 1.0)
    oof_probs /= oof_probs.sum(1, keepdims=True)
    return {
        "model": name,
        "log_loss_mean": float(np.mean(losses)),
        "log_loss_std": float(np.std(losses)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_folds": len(losses),
        "oof_probs": oof_probs.tolist(),
        "oof_labels": y.tolist(),
        "match_ids": df["match_id"].tolist(),
        "fold_losses": losses,
    }


# ── Model variants ────────────────────────────────────────────────────────────

def model_two_stage_elo(df_tr, df_val):
    """Original best: two-stage on elo+host."""
    feat = ["elo_diff", "host_advantage"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat].values)
    X_val = scaler.transform(df_val[feat].values)
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])

    # Stage 1: draw (1) vs decisive (0)
    y1 = (y_tr == 1).astype(int)
    clf1 = LogisticRegression(C=0.3, max_iter=1000, random_state=0, solver="lbfgs")
    clf1.fit(X_tr, y1)

    # Stage 2: H (0) vs A (2) among decisive
    decisive_mask = y_tr != 1
    X2 = X_tr[decisive_mask]
    y2 = (y_tr[decisive_mask] == 2).astype(int)
    if y2.sum() in (0, len(y2)):
        clf2 = None
    else:
        clf2 = LogisticRegression(C=1.0, max_iter=1000, random_state=0, solver="lbfgs")
        clf2.fit(X2, y2)

    p_draw = clf1.predict_proba(X_val)[:, 1]
    p_dec = 1 - p_draw
    if clf2 is None:
        p_away_given_dec = np.full(len(X_val), 0.5)
    else:
        p_away_given_dec = clf2.predict_proba(X_val)[:, 1]
    ph = p_dec * (1 - p_away_given_dec)
    pd_ = p_draw
    pa = p_dec * p_away_given_dec
    probs = np.stack([ph, pd_, pa], axis=1)
    probs /= probs.sum(1, keepdims=True)
    return probs


def model_two_stage_abs_elo(df_tr, df_val):
    """
    Refined two-stage: Stage 1 uses |elo_diff| for draw propensity
    (draws happen more when teams are evenly matched).
    Stage 2 uses signed elo_diff + host for H vs A.
    """
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])

    # Stage 1 features: |elo_diff|, |rank_diff|, host_advantage
    feat1 = df_tr[["elo_diff", "rank_diff", "host_advantage"]].values.copy()
    feat1[:, 0] = np.abs(feat1[:, 0])  # |elo_diff| for draw propensity
    feat1[:, 1] = np.abs(feat1[:, 1])  # |rank_diff|
    feat1_val = df_val[["elo_diff", "rank_diff", "host_advantage"]].values.copy()
    feat1_val[:, 0] = np.abs(feat1_val[:, 0])
    feat1_val[:, 1] = np.abs(feat1_val[:, 1])

    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(feat1)
    X1_val = scaler1.transform(feat1_val)

    # Stage 2 features: signed elo_diff, rank_diff, host_advantage
    feat2 = ["elo_diff", "rank_diff", "host_advantage"]
    scaler2 = StandardScaler()
    X2_tr_all = scaler2.fit_transform(df_tr[feat2].values)
    X2_val = scaler2.transform(df_val[feat2].values)

    # Stage 1: draw vs decisive
    y1 = (y_tr == 1).astype(int)
    clf1 = LogisticRegression(C=0.3, max_iter=1000, random_state=0, solver="lbfgs")
    clf1.fit(X1_tr, y1)

    # Stage 2: H vs A
    decisive_mask = y_tr != 1
    X2_dec = X2_tr_all[decisive_mask]
    y2 = (y_tr[decisive_mask] == 2).astype(int)
    if y2.sum() in (0, len(y2)):
        clf2 = None
    else:
        clf2 = LogisticRegression(C=0.5, max_iter=1000, random_state=0, solver="lbfgs")
        clf2.fit(X2_dec, y2)

    p_draw = clf1.predict_proba(X1_val)[:, 1]
    p_dec = 1 - p_draw
    if clf2 is None:
        p_away_given_dec = np.full(len(X2_val), 0.5)
    else:
        p_away_given_dec = clf2.predict_proba(X2_val)[:, 1]
    ph = p_dec * (1 - p_away_given_dec)
    pd_ = p_draw
    pa = p_dec * p_away_given_dec
    probs = np.stack([ph, pd_, pa], axis=1)
    probs /= probs.sum(1, keepdims=True)
    return probs


def model_two_stage_abs_elo_calibrated(df_tr, df_val):
    """Two-stage with |elo_diff| + nested temperature scaling calibration."""
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    if len(df_tr) < 10:
        return model_two_stage_abs_elo(df_tr, df_val)

    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    calib_probs_list, calib_y_list = [], []
    for tr_idx, cal_idx in inner_skf.split(df_tr, y_tr):
        df_inner_tr = df_tr.iloc[tr_idx].reset_index(drop=True)
        df_inner_cal = df_tr.iloc[cal_idx].reset_index(drop=True)
        raw_probs = model_two_stage_abs_elo(df_inner_tr, df_inner_cal)
        calib_probs_list.append(raw_probs)
        calib_y_list.append(y_tr[cal_idx])
    cal_probs = np.vstack(calib_probs_list)
    cal_y = np.concatenate(calib_y_list)
    ts = TemperatureScaling()
    ts.fit(cal_probs, cal_y)

    raw_val = model_two_stage_abs_elo(df_tr, df_val)
    return ts.transform(raw_val)


def model_two_stage_squad(df_tr, df_val):
    """Two-stage with |elo_diff| + squad features for draw propensity."""
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])

    # Stage 1 features: |elo_diff|, |rank_diff|, host, |mv_diff_norm|, |caps_diff|
    def s1_feats(df_):
        f = np.column_stack([
            np.abs(df_["elo_diff"].values),
            np.abs(df_["rank_diff"].values),
            df_["host_advantage"].values,
            np.abs(df_["mv_diff"].values) / 1e8,  # normalise
            np.abs(df_["caps_diff"].values),
        ])
        return f

    scaler1 = StandardScaler()
    X1_tr = scaler1.fit_transform(s1_feats(df_tr))
    X1_val = scaler1.transform(s1_feats(df_val))

    feat2 = ["elo_diff", "rank_diff", "host_advantage", "mv_diff", "caps_diff"]
    scaler2 = StandardScaler()
    X2_tr_all = scaler2.fit_transform(df_tr[feat2].fillna(0).values)
    X2_val = scaler2.transform(df_val[feat2].fillna(0).values)

    y1 = (y_tr == 1).astype(int)
    clf1 = LogisticRegression(C=0.3, max_iter=1000, random_state=0, solver="lbfgs")
    clf1.fit(X1_tr, y1)

    decisive_mask = y_tr != 1
    X2_dec = X2_tr_all[decisive_mask]
    y2 = (y_tr[decisive_mask] == 2).astype(int)
    if y2.sum() in (0, len(y2)):
        clf2 = None
    else:
        clf2 = LogisticRegression(C=0.5, max_iter=1000, random_state=0, solver="lbfgs")
        clf2.fit(X2_dec, y2)

    p_draw = clf1.predict_proba(X1_val)[:, 1]
    p_dec = 1 - p_draw
    if clf2 is None:
        p_away_given_dec = np.full(len(X2_val), 0.5)
    else:
        p_away_given_dec = clf2.predict_proba(X2_val)[:, 1]
    ph = p_dec * (1 - p_away_given_dec)
    pd_ = p_draw
    pa = p_dec * p_away_given_dec
    probs = np.stack([ph, pd_, pa], axis=1)
    probs /= probs.sum(1, keepdims=True)
    return probs


def model_three_way_ensemble(df_tr, df_val):
    """
    Best ensemble: geometric mean of 3 complementary models.
    """
    p1 = model_two_stage_elo(df_tr, df_val)
    p2 = model_two_stage_abs_elo(df_tr, df_val)
    p3 = model_two_stage_squad(df_tr, df_val)
    log_avg = (np.log(p1 + 1e-10) + np.log(p2 + 1e-10) + np.log(p3 + 1e-10)) / 3.0
    p_ens = np.exp(log_avg)
    p_ens /= p_ens.sum(1, keepdims=True)
    return p_ens


# ── ECE / Calibration curve ────────────────────────────────────────────────────

def compute_ece(probs, y, n_bins=5):
    """Expected Calibration Error (one-vs-rest per class, macro-average)."""
    ece_per_class = []
    for c in range(3):
        p_c = probs[:, c]
        y_c = (y == c).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (p_c >= bins[i]) & (p_c < bins[i + 1])
            if mask.sum() == 0:
                continue
            conf = p_c[mask].mean()
            acc = y_c[mask].mean()
            ece += mask.sum() * abs(conf - acc)
        ece /= len(y)
        ece_per_class.append(ece)
    return float(np.mean(ece_per_class))


def plot_calibration(models_oof, y_true, save_path):
    """Plot reliability diagram for H/D/A classes for selected models."""
    n_bins = 5
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    class_names = ["Home Win", "Draw", "Away Win"]

    for c, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        for name, probs in models_oof:
            p_c = probs[:, c]
            y_c = (y_true == c).astype(float)
            cal_x, cal_y = [], []
            for i in range(n_bins):
                mask = (p_c >= bins[i]) & (p_c < bins[i + 1])
                if mask.sum() >= 2:
                    cal_x.append(p_c[mask].mean())
                    cal_y.append(y_c[mask].mean())
            ax.plot(cal_x, cal_y, "o-", label=name, markersize=5)
        ax.set_title(f"{class_names[c]}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Calibration Curves (OOF, 5×10 CV, seed=0)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("Two-Stage-EloOnly (baseline best)", model_two_stage_elo),
    ("Two-Stage-AbsElo (draw propensity feature)", model_two_stage_abs_elo),
    ("Two-Stage-AbsElo+TempScale", model_two_stage_abs_elo_calibrated),
    ("Two-Stage-Squad (abs features + squad)", model_two_stage_squad),
    ("Ensemble (3 TwoStage variants)", model_three_way_ensemble),
]


def run_all(df):
    results = []
    for name, fn in EXPERIMENTS:
        print(f"\n[{name}]")
        t0 = time.time()
        res = run_cv(df, name, fn)
        elapsed = time.time() - t0
        print(f"  log-loss: {res['log_loss_mean']:.4f} ± {res['log_loss_std']:.4f}  "
              f"acc: {res['accuracy_mean']:.3f}  ({elapsed:.1f}s)")
        results.append(res)
    return results


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()

    # Campaign baseline
    baseline_ll = 0.8337
    print(f"Campaign baseline (Elo-logistic): {baseline_ll}")

    results = run_all(df)

    # Summary table
    print("\n" + "="*75)
    print(f"{'Model':<48} {'LogLoss':>8} {'±':>6} {'Acc':>6} {'Δ':>8} {'ECE':>6}")
    print("-"*75)
    for r in results:
        y = np.array(r["oof_labels"])
        probs = np.array(r["oof_probs"])
        ece = compute_ece(probs, y)
        delta = r["log_loss_mean"] - baseline_ll
        flag = "GREEN" if delta < -0.005 else ("RED" if delta > 0.01 else "FLAT")
        print(f"{r['model']:<48} {r['log_loss_mean']:>8.4f} "
              f"{r['log_loss_std']:>6.4f} {r['accuracy_mean']:>6.3f} "
              f"{delta:>+8.4f} {ece:>6.4f}  {flag}")

    # Best model
    best = min(results, key=lambda r: r["log_loss_mean"])
    print(f"\nBest: {best['model']}  log-loss={best['log_loss_mean']:.4f}")

    # Save comprehensive metrics
    ece_vals = {}
    summary_list = []
    for r in results:
        y = np.array(r["oof_labels"])
        probs = np.array(r["oof_probs"])
        ece = compute_ece(probs, y)
        ece_vals[r["model"]] = ece
        summary_list.append({
            "model": r["model"],
            "log_loss_mean": r["log_loss_mean"],
            "log_loss_std": r["log_loss_std"],
            "accuracy_mean": r["accuracy_mean"],
            "accuracy_std": r["accuracy_std"],
            "ece": ece,
            "delta_vs_elo_baseline": r["log_loss_mean"] - baseline_ll,
        })

    out = {
        "eval_protocol": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)",
        "campaign_baseline_log_loss": baseline_ll,
        "n_matches": len(df),
        "best_model": best["model"],
        "best_log_loss_mean": best["log_loss_mean"],
        "best_log_loss_std": best["log_loss_std"],
        "best_delta_vs_baseline": best["log_loss_mean"] - baseline_ll,
        "results": summary_list,
    }
    with open(ARTIFACTS / "wave2_refined_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved refined metrics -> {ARTIFACTS / 'wave2_refined_metrics.json'}")

    # OOF probs for best
    oof_out = {
        "model": best["model"],
        "log_loss_mean": best["log_loss_mean"],
        "log_loss_std": best["log_loss_std"],
        "match_ids": best["match_ids"],
        "oof_probs": best["oof_probs"],
        "oof_labels": best["oof_labels"],
        "class_order": ["H", "D", "A"],
    }
    with open(ARTIFACTS / "wave2_oof_probs_best.json", "w") as f:
        json.dump(oof_out, f, indent=2)
    print(f"Saved OOF probs -> {ARTIFACTS / 'wave2_oof_probs_best.json'}")

    # Calibration plot
    models_oof = [
        (r["model"][:30], np.array(r["oof_probs"]))
        for r in results
    ]
    y_true = np.array(results[0]["oof_labels"])
    plot_calibration(models_oof, y_true, ARTIFACTS / "wave2_calibration_curves.png")
    print(f"Saved calibration plot -> {ARTIFACTS / 'wave2_calibration_curves.png'}")
