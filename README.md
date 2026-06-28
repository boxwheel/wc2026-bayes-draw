# WC-2026 Wave 2: Bayesian Shrinkage & Draw Modeling

**Goal**: Beat the Elo-logistic CV log-loss baseline (0.8337) on 64 WC-2026 group-stage matches using Bayesian MAP shrinkage and explicit draw modeling.

## Key Result

| Model | Log-Loss | ±Std | Accuracy | Δ vs baseline | Verdict |
|---|---|---|---|---|---|
| Elo-Logistic (campaign baseline) | 0.8337 | 0.1340 | 0.617 | — | reference |
| **Two-Stage-EloOnly (untuned)** | **0.8201** | **0.0894** | **0.630** | **-0.0136** | **GREEN** |
| Two-Stage-EloOnly (CV-selected C) | 0.8038 | 0.1273 | 0.636 | -0.0299 | GREEN* |

*tuned on same CV folds — slight optimism; untuned is bias-free claim.

## How the Two-Stage Model Works

**Stage 1**: Logistic regression P(draw) vs P(decisive) using `elo_diff` + `host_advantage`  
**Stage 2**: Logistic regression P(Away win | decisive) using `elo_diff` + `host_advantage`

Final probabilities:
- P(H) = P(decisive) × P(H|decisive) = P(decisive) × (1 − P(A|decisive))
- P(D) = P(draw)
- P(A) = P(decisive) × P(A|decisive)

**Why this beats vanilla multinomial logit**: It decomposes the three-class problem into two binary problems that are easier to learn. The draw class (18/64 = 28%) is modeled in a dedicated stage, reducing systematic underestimation of draw probability. Lower variance (std 0.0894 vs 0.1340) shows the model is more stable.

## Eval Protocol

Identical to Wave 1 for comparability:
- `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)`
- Primary metric: multiclass log-loss (mean ± std over 50 folds)
- 64 completed group-stage matches, pre-match features only

## Files

- `data_loader.py` — load and engineer pre-match features from `fifa_data/`
- `model_bayesian.py` — Bayesian MAP, Davidson BT, Two-Stage models (initial sweep)
- `model_refined.py` — refined variants including |elo_diff| draw propensity
- `tune_two_stage.py` — C1/C2 grid search
- `run_final.py` — final clean run producing all artifacts

## Reproducing

```bash
# Unzip data
unzip fifa.zip -d fifa_data

# Run final model
cd wave2_bayes
python3 run_final.py
```

## Artifacts

- `artifacts/wave2_final_metrics.json` — CV log-loss results for all models
- `artifacts/wave2_oof_probs_for_ensemble.json` — per-match OOF probabilities [H, D, A]
- `artifacts/wave2_calibration_final.png` — reliability diagram
- `artifacts/wave2_run.json` — exact command, versions, seeds
