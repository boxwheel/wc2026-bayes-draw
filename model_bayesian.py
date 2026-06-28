"""
Wave 2: Bayesian Hierarchical Multinomial + Draw Modeling
=========================================================

Three model families:
1. Bayesian Hierarchical Multinomial Logit — Elo + squad features with
   informative priors and partial pooling across confederations
2. Davidson (Rao-Kupper) draw-aware Bradley-Terry model
3. Two-stage decisive-then-draw model
4. Post-hoc temperature scaling calibration (under nested CV)

Eval: RepeatedStratifiedKFold(5 folds x 10 repeats, seed=0)
Metric: log-loss (mean ± std) + accuracy
Also saves per-match OOF predicted probabilities.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import dirichlet
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ── Label encoding ────────────────────────────────────────────────────────────
CLASS_ORDER = ["H", "D", "A"]  # 0, 1, 2
CLASS_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Bayesian Dirichlet-Multinomial with shrinkage priors
#    (closed-form / MAP via L-BFGS — no MCMC needed for CV)
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianMultinomialLogit:
    """
    Multinomial logit with Normal(0, sigma) priors on coefficients.
    Equivalent to L2-regularised logistic with C = sigma^2.
    Uses two sets of linear predictors:
      - f_H = w_H @ x   (log-odds of Home vs reference)
      - f_A = w_A @ x   (log-odds of Away vs reference)
      - f_D = w_D @ x   (log-odds of Draw vs reference)
    Reference class: none (full softmax, 3K params).

    Informative priors:
      - Intercepts: N(0, 2)  (weakly informative)
      - Elo/rank coefficients: N(0, 1)  (shrink heavily on 64 samples)
      - Other coefficients: N(0, 0.5)

    Partial pooling: confederation effect modeled as random effect
    (here approximated via grouped standardisation inside fold).
    """

    def __init__(self, sigma_intercept=2.0, sigma_elo=1.0, sigma_other=0.5,
                 max_iter=500):
        self.sigma_intercept = sigma_intercept
        self.sigma_elo = sigma_elo
        self.sigma_other = sigma_other
        self.max_iter = max_iter
        self.coef_ = None
        self.classes_ = np.array([0, 1, 2])
        self.n_features_ = None

    def _log_posterior(self, w_flat, X, y_onehot, sigma_vec):
        K, n_feat = 3, X.shape[1]
        W = w_flat.reshape(K, n_feat)
        logits = X @ W.T  # (n, K)
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        log_lik = np.sum(y_onehot * log_probs)
        # Prior (negative log)
        log_prior = -0.5 * np.sum((w_flat / sigma_vec) ** 2)
        return -(log_lik + log_prior)  # minimise negative

    def _grad_log_posterior(self, w_flat, X, y_onehot, sigma_vec):
        K, n_feat = 3, X.shape[1]
        W = w_flat.reshape(K, n_feat)
        logits = X @ W.T  # (n, K)
        exp_logits = np.exp(logits - logits.max(1, keepdims=True))
        probs = exp_logits / exp_logits.sum(1, keepdims=True)  # (n, K)
        residuals = probs - y_onehot  # (n, K)
        grad_W = residuals.T @ X  # (K, n_feat)
        grad_prior = w_flat / (sigma_vec ** 2)
        return grad_W.ravel() + grad_prior

    def fit(self, X, y, feature_types=None):
        """
        feature_types: list of 'intercept', 'elo', or 'other' for each feature.
        """
        n, n_feat = X.shape
        self.n_features_ = n_feat
        K = 3

        # Build sigma vector
        if feature_types is None:
            feature_types = ["other"] * n_feat
        sigma_map = {"intercept": self.sigma_intercept,
                     "elo": self.sigma_elo,
                     "other": self.sigma_other}
        sigma_feat = np.array([sigma_map[ft] for ft in feature_types])
        sigma_vec = np.tile(sigma_feat, K)

        # One-hot encode y
        y_onehot = np.zeros((n, K))
        y_onehot[np.arange(n), y] = 1.0

        w0 = np.zeros(K * n_feat)
        result = minimize(
            self._log_posterior,
            w0,
            jac=self._grad_log_posterior,
            args=(X, y_onehot, sigma_vec),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-6},
        )
        self.coef_ = result.x.reshape(K, n_feat)
        return self

    def predict_proba(self, X):
        logits = X @ self.coef_.T  # (n, K)
        exp_l = np.exp(logits - logits.max(1, keepdims=True))
        probs = exp_l / exp_l.sum(1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Davidson Draw-Aware Bradley-Terry Model
# ═══════════════════════════════════════════════════════════════════════════════

class DavidsonBTModel:
    """
    Davidson (1970) extension of Bradley-Terry for draws.
    P(H wins) = pi_i^2 / (pi_i^2 + delta*pi_i*pi_j + pi_j^2)
    P(draw)   = delta*pi_i*pi_j / (...)
    P(A wins) = pi_j^2 / (...)

    Strengths: log(pi_t) = beta_elo * elo_t/400 + beta_host * is_host_t
               + beta_rank * rank_t_norm
    delta >= 0 is the draw propensity parameter (fit from data).
    """

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.params_ = None
        self.classes_ = np.array([0, 1, 2])

    def _probs(self, params, h_strength, a_strength, delta):
        denom = h_strength**2 + delta * h_strength * a_strength + a_strength**2
        ph = h_strength**2 / denom
        pd_ = delta * h_strength * a_strength / denom
        pa = a_strength**2 / denom
        return ph, pd_, pa

    def _neg_log_lik(self, params, X_h, X_a, y):
        """
        params: [beta_elo, beta_rank, beta_host_h, beta_host_a, log_delta]
        X_h: (n, 3) = [elo_h/400, rank_h_norm, is_host_h]
        X_a: (n, 3) = [elo_a/400, rank_a_norm, is_host_a]
        """
        beta_elo, beta_rank, beta_host = params[:3]
        log_delta = params[3]
        delta = np.exp(log_delta) + 1e-6  # must be positive

        log_pi_h = beta_elo * X_h[:, 0] + beta_rank * X_h[:, 1] + beta_host * X_h[:, 2]
        log_pi_a = beta_elo * X_a[:, 0] + beta_rank * X_a[:, 1] + beta_host * X_a[:, 2]
        pi_h = np.exp(log_pi_h)
        pi_a = np.exp(log_pi_a)

        denom = pi_h**2 + delta * pi_h * pi_a + pi_a**2
        log_ph = 2 * log_pi_h - np.log(denom)
        log_pd = np.log(delta) + log_pi_h + log_pi_a - np.log(denom)
        log_pa = 2 * log_pi_a - np.log(denom)

        nll = 0.0
        nll -= np.sum(log_ph[y == 0])
        nll -= np.sum(log_pd[y == 1])
        nll -= np.sum(log_pa[y == 2])
        # L2 on elo/rank coeffs
        nll += 0.5 * self.l2 * (beta_elo**2 + beta_rank**2)
        return nll

    def fit(self, X_h, X_a, y):
        params0 = np.array([0.5, 0.3, 0.5, 0.0])
        result = minimize(
            self._neg_log_lik,
            params0,
            args=(X_h, X_a, y),
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.params_ = result.x
        return self

    def predict_proba_raw(self, X_h, X_a):
        beta_elo, beta_rank, beta_host = self.params_[:3]
        log_delta = self.params_[3]
        delta = np.exp(log_delta) + 1e-6

        log_pi_h = beta_elo * X_h[:, 0] + beta_rank * X_h[:, 1] + beta_host * X_h[:, 2]
        log_pi_a = beta_elo * X_a[:, 0] + beta_rank * X_a[:, 1] + beta_host * X_a[:, 2]
        pi_h = np.exp(log_pi_h)
        pi_a = np.exp(log_pi_a)

        denom = pi_h**2 + delta * pi_h * pi_a + pi_a**2
        ph = pi_h**2 / denom
        pd_ = delta * pi_h * pi_a / denom
        pa = pi_a**2 / denom
        return np.stack([ph, pd_, pa], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Two-Stage: Decisive vs Draw, then H vs A
# ═══════════════════════════════════════════════════════════════════════════════

class TwoStageModel:
    """
    Stage 1: LogisticRegression — P(decisive) vs P(draw)
    Stage 2: LogisticRegression — P(H|decisive) vs P(A|decisive)
    Both stages use the same pre-match features.
    Resulting P(H) = P(dec) * P(H|dec), P(D) = P(draw), P(A) = P(dec) * P(A|dec)
    """

    def __init__(self, C1=0.5, C2=1.0):
        self.C1 = C1
        self.C2 = C2
        self.clf1 = None
        self.clf2 = None
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        # Stage 1: decisive (0) vs draw (1)
        y1 = (y == 1).astype(int)  # 1 = draw
        self.clf1 = LogisticRegression(C=self.C1, max_iter=1000, random_state=0)
        self.clf1.fit(X, y1)

        # Stage 2: H (0) vs A (2) among decisive matches
        decisive_mask = y != 1
        X2 = X[decisive_mask]
        y2 = (y[decisive_mask] == 2).astype(int)  # 1 = Away win
        if y2.sum() == 0 or y2.sum() == len(y2):
            # Degenerate fold — skip
            self.clf2 = None
        else:
            self.clf2 = LogisticRegression(C=self.C2, max_iter=1000, random_state=0)
            self.clf2.fit(X2, y2)
        return self

    def predict_proba(self, X):
        p_draw_raw = self.clf1.predict_proba(X)[:, 1]  # P(draw)
        p_decisive = 1 - p_draw_raw

        if self.clf2 is None:
            p_away_given_dec = np.full(len(X), 0.5)
        else:
            p_away_given_dec = self.clf2.predict_proba(X)[:, 1]

        ph = p_decisive * (1 - p_away_given_dec)
        pd_ = p_draw_raw
        pa = p_decisive * p_away_given_dec
        probs = np.stack([ph, pd_, pa], axis=1)
        # Normalise (should already sum to 1 but numerical safety)
        probs /= probs.sum(1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Temperature Scaling Calibration
# ═══════════════════════════════════════════════════════════════════════════════

class TemperatureScaling:
    """Post-hoc calibration by learning a single temperature T on a validation set."""

    def __init__(self):
        self.T_ = 1.0

    def fit(self, probs, y):
        """probs: (n, K) raw probabilities; y: int labels."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Feature preparation helpers
# ═══════════════════════════════════════════════════════════════════════════════

CONF_LIST = ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC", "OTHER"]
CONF_IDX = {c: i for i, c in enumerate(CONF_LIST)}


def encode_conf(conf_series):
    return conf_series.map(lambda x: CONF_IDX.get(x, CONF_IDX["OTHER"])).values


def build_feature_matrix(df, scaler=None, fit_scaler=True):
    """
    Returns X (scaled), feature_types list, scaler.
    feature_types: used by BayesianMultinomialLogit for prior assignment.
    """
    elo_features = ["elo_diff", "rank_diff", "home_elo", "away_elo"]
    squad_features = ["mv_diff", "top11_mv_diff", "caps_diff", "att_goals_diff",
                      "gk_mv_diff", "height_diff", "age_diff", "veterans_diff"]
    context_features = ["host_advantage", "home_is_host", "away_is_host",
                        "capacity", "elevation"]
    # Confederation as numeric (ordinal, will be treated as 'other')
    conf_h = encode_conf(df["home_conf"]).reshape(-1, 1)
    conf_a = encode_conf(df["away_conf"]).reshape(-1, 1)

    numeric_cols = elo_features + squad_features + context_features
    X_num = df[numeric_cols].values.astype(float)

    # Intercept
    intercept = np.ones((len(df), 1))

    # Feature types for priors
    feature_types = (
        ["intercept"] +
        ["elo"] * len(elo_features) +
        ["other"] * len(squad_features) +
        ["other"] * len(context_features) +
        ["other", "other"]  # conf_h, conf_a
    )

    X_raw = np.hstack([intercept, X_num, conf_h, conf_a])

    # Scale all non-intercept columns
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X_num_scaled = scaler.fit_transform(X_raw[:, 1:])
    else:
        X_num_scaled = scaler.transform(X_raw[:, 1:])

    X = np.hstack([intercept, X_num_scaled])
    return X, feature_types, scaler


def build_davidson_features(df, scaler_h=None, scaler_a=None, fit=True):
    """Features for Davidson model: [elo/400, rank_norm, is_host] per team."""
    elo_max = 2200.0
    rank_max = 210.0
    X_h = np.column_stack([
        df["home_elo"].values / 400.0,
        (rank_max - df["home_rank"].values) / rank_max,  # invert: low rank = strong
        df["home_is_host"].values.astype(float),
    ])
    X_a = np.column_stack([
        df["away_elo"].values / 400.0,
        (rank_max - df["away_rank"].values) / rank_max,
        df["away_is_host"].values.astype(float),
    ])
    return X_h, X_a


# ═══════════════════════════════════════════════════════════════════════════════
# CV evaluation harness
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv(df, model_name, model_fn, n_splits=5, n_repeats=10, seed=0):
    """
    model_fn(df_train, df_val) -> probs_val (n_val, 3) array [H, D, A]
    Returns dict with mean log_loss, std, accuracy, oof_probs.
    """
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
            probs = model_fn(df_tr, df_val)
        except Exception as e:
            print(f"  [fold {fold_idx}] ERROR: {e}")
            probs = np.ones((len(df_val), 3)) / 3.0

        probs = np.clip(probs, 1e-7, 1.0)
        probs /= probs.sum(1, keepdims=True)

        ll = log_loss(y_val, probs, labels=[0, 1, 2])
        acc = accuracy_score(y_val, probs.argmax(1))
        losses.append(ll)
        accs.append(acc)

        oof_probs[val_idx] += probs
        oof_counts[val_idx] += 1

    oof_probs /= np.maximum(oof_counts[:, None], 1)
    oof_probs = np.clip(oof_probs, 1e-7, 1.0)
    oof_probs /= oof_probs.sum(1, keepdims=True)

    return {
        "model": model_name,
        "log_loss_mean": float(np.mean(losses)),
        "log_loss_std": float(np.std(losses)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "n_folds": len(losses),
        "oof_probs": oof_probs.tolist(),
        "oof_labels": y.tolist(),
        "match_ids": df["match_id"].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model functions (one per experiment)
# ═══════════════════════════════════════════════════════════════════════════════

def model_elo_baseline(df_tr, df_val):
    """Logistic on elo_diff + host_advantage (the campaign reference baseline)."""
    scaler = StandardScaler()
    feat = ["elo_diff", "host_advantage"]
    X_tr = scaler.fit_transform(df_tr[feat].values)
    X_val = scaler.transform(df_val[feat].values)
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0,
                             solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_val)


def model_bayes_elo_only(df_tr, df_val):
    """BayesianMultinomialLogit on elo+rank+host (tight priors)."""
    feat = ["elo_diff", "rank_diff", "host_advantage"]
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(df_tr[feat].values)
    X_val_raw = scaler.transform(df_val[feat].values)
    intercept_tr = np.ones((len(X_tr_raw), 1))
    intercept_val = np.ones((len(X_val_raw), 1))
    X_tr = np.hstack([intercept_tr, X_tr_raw])
    X_val = np.hstack([intercept_val, X_val_raw])
    feature_types = ["intercept", "elo", "elo", "other"]
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    mdl = BayesianMultinomialLogit(sigma_intercept=2.0, sigma_elo=1.0, sigma_other=0.5)
    mdl.fit(X_tr, y_tr, feature_types=feature_types)
    return mdl.predict_proba(X_val)


def model_bayes_full(df_tr, df_val):
    """BayesianMultinomialLogit on full feature set with partial-pooling approx."""
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    X_tr, fts, scaler = build_feature_matrix(df_tr, fit_scaler=True)
    X_val, _, _ = build_feature_matrix(df_val, scaler=scaler, fit_scaler=False)
    mdl = BayesianMultinomialLogit(sigma_intercept=2.0, sigma_elo=0.8, sigma_other=0.4)
    mdl.fit(X_tr, y_tr, feature_types=fts)
    return mdl.predict_proba(X_val)


def model_davidson_bt(df_tr, df_val):
    """Davidson draw-aware Bradley-Terry."""
    X_h_tr, X_a_tr = build_davidson_features(df_tr)
    X_h_val, X_a_val = build_davidson_features(df_val)
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    mdl = DavidsonBTModel(l2=1.0)
    mdl.fit(X_h_tr, X_a_tr, y_tr)
    return mdl.predict_proba_raw(X_h_val, X_a_val)


def model_davidson_tuned(df_tr, df_val):
    """Davidson with heavier L2 shrinkage (l2=5.0)."""
    X_h_tr, X_a_tr = build_davidson_features(df_tr)
    X_h_val, X_a_val = build_davidson_features(df_val)
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    mdl = DavidsonBTModel(l2=5.0)
    mdl.fit(X_h_tr, X_a_tr, y_tr)
    return mdl.predict_proba_raw(X_h_val, X_a_val)


def model_two_stage(df_tr, df_val):
    """Two-stage decisive/draw model."""
    feat = ["elo_diff", "rank_diff", "host_advantage", "mv_diff", "caps_diff"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat].fillna(0).values)
    X_val = scaler.transform(df_val[feat].fillna(0).values)
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    mdl = TwoStageModel(C1=0.3, C2=0.5)
    mdl.fit(X_tr, y_tr)
    return mdl.predict_proba(X_val)


def model_two_stage_elo_only(df_tr, df_val):
    """Two-stage on elo features only."""
    feat = ["elo_diff", "host_advantage"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[feat].values)
    X_val = scaler.transform(df_val[feat].values)
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    mdl = TwoStageModel(C1=0.3, C2=1.0)
    mdl.fit(X_tr, y_tr)
    return mdl.predict_proba(X_val)


def model_bayes_elo_temp_scaled(df_tr, df_val):
    """
    Bayesian elo model + temperature scaling calibrated on a nested inner split.
    Uses 1 inner fold for calibration (no leakage to outer val).
    """
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    feat = ["elo_diff", "rank_diff", "host_advantage"]
    scaler = StandardScaler()

    # Inner split for calibration
    n_tr = len(df_tr)
    if n_tr < 10:
        # Not enough data; skip calibration
        return model_bayes_elo_only(df_tr, df_val)

    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    calib_probs_list = []
    calib_y_list = []

    for inner_tr_idx, inner_cal_idx in inner_skf.split(df_tr, y_tr):
        df_inner_tr = df_tr.iloc[inner_tr_idx].reset_index(drop=True)
        df_inner_cal = df_tr.iloc[inner_cal_idx].reset_index(drop=True)
        raw_probs = model_bayes_elo_only(df_inner_tr, df_inner_cal)
        calib_probs_list.append(raw_probs)
        calib_y_list.append(y_tr[inner_cal_idx])

    cal_probs = np.vstack(calib_probs_list)
    cal_y = np.concatenate(calib_y_list)
    ts = TemperatureScaling()
    ts.fit(cal_probs, cal_y)

    # Now fit the full model on all training data
    raw_val_probs = model_bayes_elo_only(df_tr, df_val)
    return ts.transform(raw_val_probs)


def model_bayes_full_temp_scaled(df_tr, df_val):
    """Bayesian full model + temperature scaling."""
    y_tr = np.array([CLASS_IDX[v] for v in df_tr["outcome"]])
    if len(df_tr) < 10:
        return model_bayes_full(df_tr, df_val)

    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    calib_probs_list = []
    calib_y_list = []
    for inner_tr_idx, inner_cal_idx in inner_skf.split(df_tr, y_tr):
        df_inner_tr = df_tr.iloc[inner_tr_idx].reset_index(drop=True)
        df_inner_cal = df_tr.iloc[inner_cal_idx].reset_index(drop=True)
        raw_probs = model_bayes_full(df_inner_tr, df_inner_cal)
        calib_probs_list.append(raw_probs)
        calib_y_list.append(y_tr[inner_cal_idx])

    cal_probs = np.vstack(calib_probs_list)
    cal_y = np.concatenate(calib_y_list)
    ts = TemperatureScaling()
    ts.fit(cal_probs, cal_y)

    raw_val_probs = model_bayes_full(df_tr, df_val)
    return ts.transform(raw_val_probs)


def model_ensemble_best(df_tr, df_val):
    """
    Ensemble: average of BayesElo + Davidson-tuned + TwoStage predictions.
    Log-odds averaging (geometric mean of probabilities, renormalised).
    """
    p1 = model_bayes_elo_temp_scaled(df_tr, df_val)
    p2 = model_davidson_tuned(df_tr, df_val)
    p3 = model_two_stage(df_tr, df_val)

    # Geometric mean
    log_avg = (np.log(p1 + 1e-10) + np.log(p2 + 1e-10) + np.log(p3 + 1e-10)) / 3.0
    p_ens = np.exp(log_avg)
    p_ens /= p_ens.sum(1, keepdims=True)
    return p_ens


# ═══════════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    ("Elo-Logistic (baseline)", model_elo_baseline),
    ("Bayes-Elo-Only (MAP, tight priors)", model_bayes_elo_only),
    ("Bayes-Full (MAP, partial-pooling approx)", model_bayes_full),
    ("Davidson-BT (draw-aware, l2=1)", model_davidson_bt),
    ("Davidson-BT-Tuned (l2=5)", model_davidson_tuned),
    ("Two-Stage (decisive/draw split)", model_two_stage),
    ("Two-Stage-EloOnly", model_two_stage_elo_only),
    ("Bayes-Elo + TempScale (nested CV cal)", model_bayes_elo_temp_scaled),
    ("Bayes-Full + TempScale (nested CV cal)", model_bayes_full_temp_scaled),
    ("Ensemble (BayesElo+Davidson+TwoStage)", model_ensemble_best),
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
    print(f"Loaded {len(df)} completed matches. Outcome counts:")
    print(df["outcome"].value_counts().to_dict())

    results = run_all(df)

    # Save summary
    summary = []
    for r in results:
        summary.append({
            "model": r["model"],
            "log_loss_mean": r["log_loss_mean"],
            "log_loss_std": r["log_loss_std"],
            "accuracy_mean": r["accuracy_mean"],
            "accuracy_std": r["accuracy_std"],
            "delta_vs_baseline": r["log_loss_mean"] - results[0]["log_loss_mean"],
        })

    print("\n" + "="*70)
    print(f"{'Model':<45} {'LogLoss':>8} {'±':>6} {'Acc':>6} {'Δ':>8}")
    print("-"*70)
    for s in summary:
        flag = "GREEN" if s["delta_vs_baseline"] < -0.01 else (
               "RED" if s["delta_vs_baseline"] > 0.01 else "FLAT")
        print(f"{s['model']:<45} {s['log_loss_mean']:>8.4f} "
              f"{s['log_loss_std']:>6.4f} {s['accuracy_mean']:>6.3f} "
              f"{s['delta_vs_baseline']:>+8.4f}  {flag}")

    # Save artifacts
    out = {
        "eval_protocol": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)",
        "baseline_log_loss": 0.8337,
        "n_matches": len(df),
        "results": summary,
    }
    with open(ARTIFACTS / "wave2_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics to {ARTIFACTS / 'wave2_metrics.json'}")

    # Save OOF probs for best model
    best = min(results, key=lambda r: r["log_loss_mean"])
    print(f"\nBest model: {best['model']}  log-loss={best['log_loss_mean']:.4f}")
    oof_out = {
        "model": best["model"],
        "log_loss_mean": best["log_loss_mean"],
        "log_loss_std": best["log_loss_std"],
        "match_ids": best["match_ids"],
        "oof_probs": best["oof_probs"],   # [H, D, A] per match
        "oof_labels": best["oof_labels"],  # 0=H, 1=D, 2=A
        "class_order": ["H", "D", "A"],
    }
    with open(ARTIFACTS / "wave2_oof_probs.json", "w") as f:
        json.dump(oof_out, f, indent=2)
    print(f"Saved OOF probs to {ARTIFACTS / 'wave2_oof_probs.json'}")
