# src/eval_metrics.py

"""
Evaluation metrics for synthetic tabular data.

Three families:
    1. Per-column JSD on categoricals - the key metric for showing how
       well a method recovers each feature's marginal distribution.
       Reported per feature so we can break down by cardinality.

    2. Correlation-matrix L2 - Frobenius norm of the difference between
       real and synthetic Pearson correlation matrices (computed over the
       full feature set, categoricals treated as their integer labels).

    3. TSTR (Train on Synthetic, Test on Real) - train XGBoost and
       Logistic Regression on synthetic (x, y) and report accuracy / F1
       on the real held-out test set. Standard utility-style probe used
       by TabDDPM / TabSyn.

All functions accept the dict format produced by `data_utils.load_split`:
    {"x_num": (N, n_num) float32, "x_cat": (N, n_cat) int64, "y": (N,) int64}
or directly the underlying arrays where that's cleaner.

JSD is reported in **bits** (base-2 log) so its theoretical range is [0, 1]
regardless of K - convenient for comparing across features of different
cardinality.
"""

from typing import Dict, Sequence

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import accuracy_score, f1_score
from xgboost              import XGBClassifier


# JSD ---------------------------------------


def _jsd_one_column(real_col: np.ndarray, synth_col: np.ndarray, K: int,
                    smoothing: float = 1e-10) -> float:
    """
    JSD in bits between two categorical columns over K classes.

    JSD(P || Q) = 0.5 · KL(P || M) + 0.5 · KL(Q || M),   M = 0.5(P + Q)

    `bincount(minlength=K)` handles unseen categories cleanly; a tiny
    additive smoothing prevents 0/0 in KL when one side has empty bins.
    """
    # Build discrete distributions by counting occurrences of each category [0, K).
    # minlength=K ensures bins for unseen categories are included as zeros.
    # Additive smoothing avoids division-by-zero in the KL computation.
    p = np.bincount(real_col,  minlength=K).astype(np.float64) + smoothing
    q = np.bincount(synth_col, minlength=K).astype(np.float64) + smoothing
    p /= p.sum()
    q /= q.sum()

    # JSD is defined via the mixture M = (P + Q) / 2 and is symmetric.
    m = 0.5 * (p + q)
    # scipy.stats.entropy(p, m, base=2) computes KL(P || M) in bits.
    return 0.5 * (scipy.stats.entropy(p, m, base=2) + scipy.stats.entropy(q, m, base=2))


def per_column_jsd(
    real_x_cat:  np.ndarray,
    synth_x_cat: np.ndarray,
    cardinalities: Sequence[int],
) -> np.ndarray:
    """
    Per-categorical-column JSD (bits). Output is (n_cat,) aligned with
    `cardinalities` (i.e. the column order in x_cat).

    Args:
        real_x_cat:    (N_real, n_cat)  int - label-encoded reals.
        synth_x_cat:   (N_synth, n_cat) int - label-encoded synthetics.
        cardinalities: list of K_i per column.

    Returns:
        (n_cat,) float - JSD per column, in bits ∈ [0, 1].
    """
    if real_x_cat.shape[1] != len(cardinalities) or synth_x_cat.shape[1] != len(cardinalities):
        raise ValueError(
            f"column-count mismatch: real has {real_x_cat.shape[1]}, "
            f"synth has {synth_x_cat.shape[1]}, cardinalities has {len(cardinalities)}"
        )
    return np.array(
        [_jsd_one_column(real_x_cat[:, i].astype(np.int64),
                         synth_x_cat[:, i].astype(np.int64),
                         K)
         for i, K in enumerate(cardinalities)]
    )


# Correlation matrix ---------------------------------------


def _full_feature_matrix(x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
    """Concatenate numericals + integer-encoded categoricals to a single (N, D) float.

    Categoricals are kept as their integer labels (not one-hot) so Pearson
    correlation treats them as ordinal - a coarse but standard simplification.
    """
    return np.concatenate([x_num.astype(np.float64), x_cat.astype(np.float64)], axis=1)


def correlation_l2(
    real_x_num:  np.ndarray,
    real_x_cat:  np.ndarray,
    synth_x_num: np.ndarray,
    synth_x_cat: np.ndarray,
) -> float:
    """
    Frobenius norm of (corr_real - corr_synth) over the full feature set.

    Categoricals are treated as their integer labels (Pearson correlation on
    label values). This is a coarse but standard summary that captures whether
    cross-feature dependencies are preserved.
    """
    real  = _full_feature_matrix(real_x_num,  real_x_cat)
    synth = _full_feature_matrix(synth_x_num, synth_x_cat)

    # Compute full Pearson correlation matrix for each dataset.
    # rowvar=False tells np.corrcoef that columns (not rows) are variables.
    cr = np.corrcoef(real,  rowvar=False)
    cs = np.corrcoef(synth, rowvar=False)

    # Frobenius norm: sqrt(sum of squared element-wise differences).
    # Lower = better preservation of cross-feature dependencies.
    return float(np.linalg.norm(cr - cs, ord="fro"))


# TSTR ---------------------------------------


def _xgb_matrix(x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
    """XGBoost handles integer cats fine as numeric features - just concat."""
    return _full_feature_matrix(x_num, x_cat)


def _lr_matrix(x_num: np.ndarray, x_cat: np.ndarray,
               cardinalities: Sequence[int]) -> np.ndarray:
    """LR needs one-hot for categoricals; numericals are already in QT-normal space.

    Unlike XGBoost (which handles integer-coded categoricals natively),
    logistic regression requires explicit one-hot encoding so it can learn
    independent coefficients per category.
    """
    n = x_num.shape[0]
    parts = [x_num.astype(np.float32)]
    for i, K in enumerate(cardinalities):
        oh = np.zeros((n, K), dtype=np.float32)
        # Clip to valid range - synthetics from a correctly-sampled model are
        # already in [0, K), but guard against any edge case.
        idx = np.clip(x_cat[:, i].astype(np.int64), 0, K - 1)
        oh[np.arange(n), idx] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1)


def _classify(X_train, y_train, X_test, y_test, seed: int) -> Dict[str, float]:
    """Train XGB + LR; return accuracy and binary F1 on the test set.

    Both classifiers are deterministic given the seed. XGBoost uses
    gradient-boosted trees; LR uses L2-regularised logistic regression
    (scikit-learn default). max_iter=2000 for LR avoids convergence
    warnings on the one-hot-expanded feature space.
    """
    out: Dict[str, float] = {}

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=seed, eval_metric="logloss",
        verbosity=0, n_jobs=1,
    )
    xgb.fit(X_train, y_train)
    yp = xgb.predict(X_test)
    out["xgb_acc"] = float(accuracy_score(y_test, yp))
    out["xgb_f1"]  = float(f1_score(y_test, yp, average="binary"))

    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(X_train, y_train)
    yp = lr.predict(X_test)
    out["lr_acc"]  = float(accuracy_score(y_test, yp))
    out["lr_f1"]   = float(f1_score(y_test, yp, average="binary"))
    return out


def tstr(
    synth:         Dict[str, np.ndarray],
    real_test:     Dict[str, np.ndarray],
    cardinalities: Sequence[int],
    seed:          int = 42,
) -> Dict[str, float]:
    """
    Train classifiers on synthetic (x, y), evaluate on the real test set.

    Args:
        synth:     {'x_num', 'x_cat', 'y'}  - generated samples + their labels.
        real_test: {'x_num', 'x_cat', 'y'}  - held-out real data.
        cardinalities: K_i per categorical column (for LR one-hot).
        seed:      RNG seed for both classifiers.

    Returns:
        dict with `xgb_acc`, `xgb_f1`, `lr_acc`, `lr_f1`.
    """
    # Build feature matrices: XGBoost uses raw integer cats, LR uses one-hot.
    # Training data is synthetic; test data is real (the TSTR protocol).
    X_train_xgb = _xgb_matrix(synth["x_num"],     synth["x_cat"])
    X_test_xgb  = _xgb_matrix(real_test["x_num"], real_test["x_cat"])
    X_train_lr  = _lr_matrix(synth["x_num"],     synth["x_cat"],     cardinalities)
    X_test_lr   = _lr_matrix(real_test["x_num"], real_test["x_cat"], cardinalities)

    # Two classifiers, each with its own input matrix - but the train/test
    # split (synthetic -> real) is the same. We call _classify twice rather than
    # nesting the logic because the feature layouts differ across classifiers.
    out = {}
    xgb_metrics = _classify(X_train_xgb, synth["y"], X_test_xgb, real_test["y"], seed)
    out.update({k: v for k, v in xgb_metrics.items() if k.startswith("xgb_")})
    lr_metrics  = _classify(X_train_lr,  synth["y"], X_test_lr,  real_test["y"], seed)
    out.update({k: v for k, v in lr_metrics.items() if k.startswith("lr_")})
    return out


def trtr(
    real_train:    Dict[str, np.ndarray],
    real_test:     Dict[str, np.ndarray],
    cardinalities: Sequence[int],
    seed:          int = 42,
) -> Dict[str, float]:
    """Train-Real Test-Real baseline - upper bound that TSTR aims to approach.

    If the generative model perfectly captures the data distribution, TSTR
    scores should be close to TRTR scores. The gap between TRTR and TSTR
    quantifies the utility loss from using synthetic data.
    """
    return tstr(real_train, real_test, cardinalities, seed=seed)


# Sanity check ---------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_utils import load_split, load_meta, CAT_COLS

    meta          = load_meta()
    cardinalities = meta["cardinalities"]
    train         = load_split("train")
    test          = load_split("test")

    print(f"[eval] sizes: train={len(train['y']):,}  test={len(test['y']):,}")
    print(f"[eval] cardinalities = {cardinalities}")

    # --- 1. JSD: train vs train (should be 0) ---------------------------------
    jsd_self = per_column_jsd(train["x_cat"], train["x_cat"], cardinalities)
    print(f"\n[eval] JSD(train, train) - should all be 0:")
    print(f"  max = {jsd_self.max():.2e}   mean = {jsd_self.mean():.2e}")
    assert jsd_self.max() < 1e-10, "JSD(P, P) should be 0"

    # --- 2. JSD: train vs test (small but nonzero) ----------------------------
    jsd_tt = per_column_jsd(train["x_cat"], test["x_cat"], cardinalities)
    print(f"\n[eval] JSD(train, test) - small but nonzero (bits, ∈ [0, 1]):")
    print(f"  {'feature':<18s} {'K':>3s} {'JSD':>8s}")
    for col, K, j in zip(CAT_COLS, cardinalities, jsd_tt):
        print(f"  {col:<18s} {K:>3d} {j:>8.5f}")
    print(f"  overall: mean={jsd_tt.mean():.5f}  max={jsd_tt.max():.5f}")
    assert (jsd_tt >= 0).all() and (jsd_tt <= 1).all(), "JSD must be ∈ [0, 1]"
    assert jsd_tt.max() > 0, "train and test should differ at least a bit"

    # --- 3. correlation_l2 on train-vs-test -----------------------------------
    cl2 = correlation_l2(train["x_num"], train["x_cat"],
                          test["x_num"],  test["x_cat"])
    print(f"\n[eval] correlation_l2(train, test) = {cl2:.4f}   "
          f"(small for splits of the same population)")

    # --- 4. TSTR runs end-to-end + TRTR baseline ------------------------------
    print(f"\n[eval] running tstr(train→test) - proxy for TRTR (upper bound):")
    trtr_metrics = trtr(train, test, cardinalities, seed=42)
    for k, v in trtr_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Quick noise-as-synth check: shuffle synthetic y to break the signal
    print(f"\n[eval] sanity: with shuffled-y 'synth' (should DROP):")
    rng = np.random.default_rng(0)
    noisy_synth = {
        "x_num": train["x_num"],
        "x_cat": train["x_cat"],
        "y":     rng.permutation(train["y"]),
    }
    noisy_metrics = tstr(noisy_synth, test, cardinalities, seed=42)
    for k, v in noisy_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n[eval] ✓ Sanity check complete.")
