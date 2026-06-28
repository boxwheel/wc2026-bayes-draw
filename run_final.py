"""
Final Wave 2 run: produce all artifacts for Flywheel.
- Elo-logistic baseline (verify = 0.8337)
- Two-Stage-EloOnly C1=0.3, C2=1.0 (untuned, a priori, bias-free = 0.8201)
- Two-Stage-EloOnly C1=0.1, C2=5.0 (tuned via same CV — slight selection bias)
- Per-match OOF probabilities for ensemble/significance agent
- Calibration curves + ECE
- run.json with exact command and versions
"""
import json
import time
import platform
import subprocess
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
CLASS_IDX = {"H": 0, "D": 1, "A": 2}


def two_stage_elo(df_tr, df_val, C1=0.3, C2=1.0):
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    feat = ["elo_diff", "host_advantage"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat].values)
    X_val = scaler.transform(df_val[feat].values)

    y1 = (y_tr == 1).astype(int)
    clf1 = LogisticRegression(C=C1, max_iter=1000, random_state=0, solver="lbfgs")
    clf1.fit(X_tr, y1)

    decisive_mask = y_tr != 1
    X2 = X_tr[decisive_mask]
    y2 = (y_tr[decisive_mask] == 2).astype(int)
    if y2.sum() in (0, len(y2)):
        clf2 = None
    else:
        clf2 = LogisticRegression(C=C2, max_iter=1000, random_state=0, solver="lbfgs")
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
    probs = np.clip(np.stack([ph, pd_, pa], axis=1), 1e-7, 1.0)
    probs /= probs.sum(1, keepdims=True)
    return probs


def elo_logistic_baseline(df_tr, df_val):
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    feat = ["elo_diff", "host_advantage"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat].values)
    X_val = scaler.transform(df_val[feat].values)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_val)


def run_cv(df, name, fn, n_splits=5, n_repeats=10, seed=0):
    y = np.array([CLASS_IDX[v] for v in df["outcome"]])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    losses, accs = [], []
    oof_probs = np.zeros((len(df), 3))
    oof_counts = np.zeros(len(df))

    for train_idx, val_idx in rskf.split(df, y):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]
        probs = fn(df_tr, df_val_fold)
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
        "model": name, "log_loss_mean": float(np.mean(losses)),
        "log_loss_std": float(np.std(losses)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "oof_probs": oof_probs.tolist(), "oof_labels": y.tolist(),
        "match_ids": df["match_id"].tolist(), "fold_losses": losses,
    }


def compute_ece(probs, y, n_bins=5):
    ece = 0.0
    for c in range(3):
        p_c = probs[:, c]
        y_c = (y == c).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (p_c >= bins[i]) & (p_c < bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() * abs(p_c[mask].mean() - y_c[mask].mean())
    return float(ece / (3 * len(y)))


def plot_calibration(results, y_true, save_path):
    n_bins = 5
    bins = np.linspace(0, 1, n_bins + 1)
    class_names = ["Home Win", "Draw", "Away Win"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for c, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect", lw=1.5)
        for r in results:
            probs = np.array(r["oof_probs"])
            p_c = probs[:, c]
            y_c = (y_true == c).astype(float)
            cal_x, cal_y, ns = [], [], []
            for i in range(n_bins):
                mask = (p_c >= bins[i]) & (p_c < bins[i + 1])
                if mask.sum() >= 2:
                    cal_x.append(p_c[mask].mean())
                    cal_y.append(y_c[mask].mean())
                    ns.append(mask.sum())
            label = r["model"][:30]
            ax.plot(cal_x, cal_y, "o-", label=label, markersize=6, lw=1.5)
        ax.set_title(f"{class_names[c]}", fontsize=11)
        ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Calibration Reliability Diagrams (OOF, 5×10 CV, seed=0)\n"
                 "Wave 2: Bayesian Shrinkage & Draw Modeling", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration plot -> {save_path}")


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    print(f"Loaded {len(df)} matches. Outcomes: {df['outcome'].value_counts().to_dict()}")

    t0 = time.time()
    r_base = run_cv(df, "Elo-Logistic Baseline", elo_logistic_baseline)
    r_ts_default = run_cv(df, "Two-Stage-EloOnly (C1=0.3, C2=1.0)",
                          lambda a, b: two_stage_elo(a, b, C1=0.3, C2=1.0))
    r_ts_tuned = run_cv(df, "Two-Stage-EloOnly (C1=0.1, C2=5.0, CV-selected)",
                        lambda a, b: two_stage_elo(a, b, C1=0.1, C2=5.0))
    elapsed = time.time() - t0

    results = [r_base, r_ts_default, r_ts_tuned]
    baseline_ll = r_base["log_loss_mean"]

    print("\n" + "="*80)
    print(f"{'Model':<48} {'LogLoss':>8} {'±':>6} {'Acc':>6} {'Δ':>8} {'ECE':>6}")
    print("-"*80)
    for r in results:
        y = np.array(r["oof_labels"])
        probs = np.array(r["oof_probs"])
        ece = compute_ece(probs, y)
        delta = r["log_loss_mean"] - baseline_ll
        flag = "GREEN" if delta < -0.005 else ("RED" if delta > 0.01 else "FLAT")
        print(f"{r['model']:<48} {r['log_loss_mean']:>8.4f} "
              f"{r['log_loss_std']:>6.4f} {r['accuracy_mean']:>6.3f} "
              f"{delta:>+8.4f} {ece:>6.4f}  {flag}")

    # Save calibration plot
    y_true = np.array(r_base["oof_labels"])
    plot_calibration(results, y_true, ARTIFACTS / "wave2_calibration_final.png")

    # Save metrics JSON (for Flywheel artifact)
    metrics = {
        "eval_protocol": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)",
        "campaign_baseline_log_loss": 0.8337,
        "campaign_baseline_log_loss_std": 0.134,
        "n_matches": len(df),
        "elapsed_seconds": elapsed,
        "models": []
    }
    for r in results:
        y = np.array(r["oof_labels"])
        probs = np.array(r["oof_probs"])
        ece = compute_ece(probs, y)
        metrics["models"].append({
            "model": r["model"],
            "log_loss_mean": r["log_loss_mean"],
            "log_loss_std": r["log_loss_std"],
            "accuracy_mean": r["accuracy_mean"],
            "accuracy_std": r["accuracy_std"],
            "ece": ece,
            "delta_vs_elo_baseline": r["log_loss_mean"] - 0.8337,
            "verdict": "GREEN" if r["log_loss_mean"] - 0.8337 < -0.005 else
                       ("RED" if r["log_loss_mean"] - 0.8337 > 0.01 else "FLAT"),
        })

    with open(ARTIFACTS / "wave2_final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {ARTIFACTS / 'wave2_final_metrics.json'}")

    # Save OOF probs for best model (untuned, bias-free)
    best_r = r_ts_default
    oof_out = {
        "model": best_r["model"],
        "log_loss_mean": best_r["log_loss_mean"],
        "log_loss_std": best_r["log_loss_std"],
        "description": "Out-of-fold probabilities from Two-Stage-EloOnly (untuned C1=0.3, C2=1.0). "
                       "Use for ensemble/significance testing. Class order: [H=0, D=1, A=2].",
        "match_ids": best_r["match_ids"],
        "oof_probs": best_r["oof_probs"],
        "oof_labels": best_r["oof_labels"],
        "class_order": ["H", "D", "A"],
    }
    with open(ARTIFACTS / "wave2_oof_probs_for_ensemble.json", "w") as f:
        json.dump(oof_out, f, indent=2)
    print(f"Saved OOF probs -> {ARTIFACTS / 'wave2_oof_probs_for_ensemble.json'}")

    # run.json
    run_info = {
        "command": "python3 run_final.py",
        "working_directory": str(Path(__file__).parent),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "random_seed": 0,
        "n_folds": 5,
        "n_repeats": 10,
        "features_stage1": ["elo_diff", "host_advantage"],
        "features_stage2": ["elo_diff", "host_advantage"],
        "model": "TwoStageLogistic",
        "hyperparams": {
            "C1_untuned": 0.3, "C2_untuned": 1.0,
            "C1_tuned": 0.1, "C2_tuned": 5.0,
        },
        "leakage_prevention": "Scaler fitted inside each CV fold on training rows only. "
                              "No post-match features used.",
    }
    with open(ARTIFACTS / "wave2_run.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run info -> {ARTIFACTS / 'wave2_run.json'}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Best (untuned, bias-free): LL={r_ts_default['log_loss_mean']:.4f} ± "
          f"{r_ts_default['log_loss_std']:.4f}  Δ={r_ts_default['log_loss_mean']-0.8337:+.4f}")
