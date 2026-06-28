"""
Grid search over C1/C2 hyperparameters for Two-Stage-EloOnly.
Goal: squeeze more out of the best model.
"""
import json
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"

CLASS_IDX = {"H": 0, "D": 1, "A": 2}


def run_two_stage(df, C1=0.3, C2=1.0, feat=("elo_diff", "host_advantage"),
                  n_splits=5, n_repeats=10, seed=0):
    y = np.array([CLASS_IDX[v] for v in df["outcome"]])
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    losses = []
    for train_idx, val_idx in rskf.split(df, y):
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        y_val = y[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_tr[list(feat)].values)
        X_val = scaler.transform(df_val[list(feat)].values)

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
        p_away_given_dec = clf2.predict_proba(X_val)[:, 1] if clf2 else np.full(len(X_val), 0.5)
        ph = p_dec * (1 - p_away_given_dec)
        pd_ = p_draw
        pa = p_dec * p_away_given_dec
        probs = np.clip(np.stack([ph, pd_, pa], axis=1), 1e-7, 1.0)
        probs /= probs.sum(1, keepdims=True)
        losses.append(log_loss(y_val, probs, labels=[0, 1, 2]))
    return float(np.mean(losses)), float(np.std(losses))


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()

    # Grid search C1, C2
    C1_vals = [0.1, 0.2, 0.3, 0.5, 1.0]
    C2_vals = [0.3, 0.5, 1.0, 2.0, 5.0]

    print(f"{'C1':>6} {'C2':>6} {'LogLoss':>10} {'±':>8}")
    best_ll, best_config = 999, None
    grid_results = []

    for C1 in C1_vals:
        for C2 in C2_vals:
            ll, std = run_two_stage(df, C1=C1, C2=C2)
            flag = " <-- BEST" if ll < best_ll else ""
            if ll < best_ll:
                best_ll = ll
                best_config = (C1, C2, std)
            print(f"{C1:>6.1f} {C2:>6.1f} {ll:>10.4f} {std:>8.4f}{flag}")
            grid_results.append({"C1": C1, "C2": C2, "log_loss_mean": ll, "log_loss_std": std})

    print(f"\nBest: C1={best_config[0]}, C2={best_config[1]}, log-loss={best_ll:.4f} ± {best_config[2]:.4f}")
    print(f"Delta vs campaign baseline 0.8337: {best_ll - 0.8337:+.4f}")

    with open(ARTIFACTS / "wave2_grid_search.json", "w") as f:
        json.dump({"grid": grid_results, "best_C1": best_config[0],
                   "best_C2": best_config[1], "best_ll": best_ll,
                   "best_std": best_config[2]}, f, indent=2)
    print(f"Saved grid search results -> {ARTIFACTS / 'wave2_grid_search.json'}")
