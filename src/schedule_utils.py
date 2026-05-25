# src/schedule_utils.py

"""
Mutual-information-based schedule adjustment for multinomial diffusion.

TabDDPM applies the same shared ᾱ_t schedule to every categorical feature,
regardless of cardinality. This module computes per-cardinality schedules
by equalising mutual information across features:

    I(ᾱ_t_adjusted, K_i) = I(ᾱ_t_base, K_ref)

MI formula (uniform prior over K classes, in nats):

    I(ᾱ, K) = log(K) + p · log(p) + (K-1) · q · log(q)

where
    p = ᾱ + (1 - ᾱ) / K     [P(x_t = x_0)]
    q = (1 - ᾱ) / K          [P(x_t = j) for each j ≠ x_0]

I is monotonically increasing in ᾱ ∈ [0, 1], so bisection (brentq)
recovers the adjusted ᾱ uniquely. scipy.special.xlogy handles the
0·log(0) = 0 limit at ᾱ = 1, giving MI(0, K)=0 and MI(1, K)=log(K)
exactly - necessary for brentq to always bracket the root.
"""

from typing import Dict, List, Sequence, Union

import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy


ArrayLike = Union[float, np.ndarray]


# Mutual information ---------------------------------------


def mutual_info_multinomial(alpha_bar: ArrayLike, K: int) -> ArrayLike:
    """
    I(x_0; x_t) in nats for the uniform-noise multinomial forward process.

    Args:
        alpha_bar: scalar or array of ᾱ ∈ [0, 1].
        K:         number of categories (K ≥ 2).

    Returns:
        Mutual information in nats, broadcast to the shape of alpha_bar.

    Boundary values are exact (no epsilon hack):
        MI(0, K) = 0,  MI(1, K) = log(K).
    """
    if K < 2:
        raise ValueError(f"K must be ≥ 2, got {K}")
    ab = np.asarray(alpha_bar, dtype=np.float64)

    # p = P(x_t = x_0): probability the noisy sample keeps its original class.
    # q = P(x_t = j != x_0): probability of switching to any single wrong class.
    # These follow from the transition matrix Q_t = (1-beta_t)I + beta_t (1/K) 11^T.
    p = ab + (1.0 - ab) / K
    q = (1.0 - ab) / K

    # MI formula: I = log(K) + p*log(p) + (K-1)*q*log(q)
    # xlogy(a, b) computes a*log(b) and correctly returns 0 when a=0,
    # avoiding the NaN that log(0) would produce at the boundary alpha_bar=1.
    return np.log(K) + xlogy(p, p) + (K - 1) * xlogy(q, q)


# Solver ---------------------------------------


def solve_adjusted_alpha_bar(target_MI: float, K: int, tol: float = 1e-10) -> float:
    """
    Find ᾱ ∈ [0, 1] such that I(ᾱ, K) = target_MI.

    Bisection via scipy.optimize.brentq; I is monotonically increasing
    in ᾱ so the root is unique.

    Args:
        target_MI: desired mutual information in nats.
        K:         number of categories.
        tol:       brentq absolute tolerance on ᾱ (default 1e-10).

    Returns:
        ᾱ_adjusted ∈ [0, 1]. Clamped to endpoints when target_MI is
        outside the achievable range [0, log(K)].
    """
    if K < 2:
        raise ValueError(f"K must be ≥ 2, got {K}")

    # Edge cases: MI=0 at alpha_bar=0 (pure noise), MI=log(K) at alpha_bar=1 (no noise).
    # Clamp to these endpoints if the target falls outside the achievable range.
    if target_MI <= 0.0:
        return 0.0
    max_MI = np.log(K)
    if target_MI >= max_MI:
        return 1.0

    # Root-finding: solve MI(alpha_bar, K) - target = 0 over [0, 1].
    # MI is strictly monotonically increasing in alpha_bar, so there is
    # exactly one root and brentq (Brent's bisection method) converges.
    f = lambda ab: mutual_info_multinomial(ab, K) - target_MI
    return float(brentq(f, 0.0, 1.0, xtol=tol))


# Per-feature schedules ---------------------------------------


def compute_adjusted_schedules(
    base_alpha_bars: Sequence[float],
    cardinalities:   Sequence[int],
    K_ref:           int = 2,
    strength:        float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Per-cardinality MI-equalised ᾱ schedule across all timesteps.

    For each unique K_i in `cardinalities`, solve
        I(ᾱ_t_adjusted, K_i) = I(ᾱ_t_base, K_ref)
    for every t.

    When strength < 1.0, interpolate between base and fully adjusted:
        ᾱ_final = (1 - strength) · ᾱ_base + strength · ᾱ_MI_adjusted

    Args:
        base_alpha_bars: (T,) shared schedule (e.g. cumulative product
                         from the Gaussian linear schedule).
        cardinalities:   list of K_i - one per categorical feature.
        K_ref:           reference cardinality (default 2 = binary).
        strength:        interpolation weight in [0, 1]. 1.0 = full MI
                         equalization, 0.0 = baseline schedule.

    Returns:
        dict {K_i → np.ndarray (T,)} of adjusted ᾱ for each unique K_i.
        K_ref is included as a copy of the base schedule.
    """
    base = np.asarray(base_alpha_bars, dtype=np.float64)
    T = base.shape[0]

    # Step 1: Compute the MI that the *reference* cardinality (K_ref, default=2)
    # achieves at each timestep under the base (shared) schedule.  These become
    # the per-timestep targets that every other cardinality must match.
    target_MI = mutual_info_multinomial(base, K_ref)  # (T,)

    adjusted: Dict[int, np.ndarray] = {}
    for K in sorted(set(int(k) for k in cardinalities)):
        # The reference cardinality keeps the original schedule unchanged.
        if K == K_ref:
            adjusted[K] = base.copy()
            continue

        # Step 2: For each non-reference K_i, invert the MI function at every t
        # to find the alpha_bar that yields the same MI as K_ref does under base.
        # High-K features get *lower* alpha_bar (more noise) because they carry
        # more MI at the same alpha_bar than low-K features.
        ab = np.empty(T, dtype=np.float64)
        for t in range(T):
            ab[t] = solve_adjusted_alpha_bar(float(target_MI[t]), K)

        # Optional partial adjustment: interpolate between base and fully adjusted
        # schedules. strength=1.0 is full MI equalisation; strength=0.0 is baseline.
        if strength < 1.0:
            ab = (1.0 - strength) * base + strength * ab
        adjusted[K] = ab

    return adjusted


# Sanity check ---------------------------------------


if __name__ == "__main__":
    # 1. MI curve values for K ∈ {2, 8, 16, 41} at a few sample ᾱ
    print("=== MI I(ᾱ, K) in nats - at sample ᾱ ===")
    sample_abs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    Ks = [2, 8, 16, 41]
    header = "  ᾱ   " + "".join(f"  K={K:<2d}(max={np.log(K):.3f})" for K in Ks)
    print(header)
    for ab in sample_abs:
        row = f" {ab:>4.2f} "
        for K in Ks:
            mi = float(mutual_info_multinomial(ab, K))
            row += f"   {mi:>10.4f}     "
        print(row)

    # 2. Monotonicity sanity - I should strictly increase with ᾱ for each K
    print("\n=== Monotonicity check (1000-point grid) ===")
    grid = np.linspace(0.0, 1.0, 1000)
    for K in Ks:
        mi_grid = mutual_info_multinomial(grid, K)
        diffs = np.diff(mi_grid)
        mono = (diffs >= -1e-12).all()
        print(f"  K={K:>2d}: min ΔI = {diffs.min():+.3e}   monotone-↑ = {mono}")

    # 3. Round-trip check: solve I → α → recompute I, should match
    print("\n=== Round-trip: solve(target) → MI(solution) ≈ target ===")
    for K in Ks:
        for target in [0.05, 0.1, 0.3, 0.6]:
            if target >= np.log(K):
                continue
            ab = solve_adjusted_alpha_bar(target, K)
            recovered = float(mutual_info_multinomial(ab, K))
            err = abs(recovered - target)
            print(f"  K={K:>2d}  target={target:.3f}  →  ᾱ={ab:.6f}  MI(ᾱ)={recovered:.6f}  err={err:.2e}")

    # 4. Adjusted schedules for Adult cardinalities
    print("\n=== Adjusted schedule for Adult cardinalities (K_ref=2) ===")
    print("    base schedule: β linear ∈ [1e-4, 0.02], T=1000")
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)
    alpha_bars = np.cumprod(1.0 - betas)

    adult_cards = [8, 16, 7, 14, 6, 5, 2, 41]
    schedules = compute_adjusted_schedules(alpha_bars, adult_cards, K_ref=2)

    sample_ts = [0, 100, 250, 500, 750, 900, 999]
    Ks_sorted = sorted(set(adult_cards))
    print(f"\n  {'t':>4s}  {'base':>9s}" + "".join(f"   K={K:<2d}" for K in Ks_sorted))
    for t in sample_ts:
        row = f"  {t:>4d}  {alpha_bars[t]:>9.6f}"
        for K in Ks_sorted:
            row += f"  {schedules[K][t]:>6.4f}"
        print(row)

    # 5. Direction summary at a middle timestep
    t_mid = 500
    base_mid = alpha_bars[t_mid]
    target_mid = float(mutual_info_multinomial(base_mid, 2))
    print(f"\n=== Direction at t={t_mid} (base ᾱ = {base_mid:.6f}, target MI = {target_mid:.4f} nats) ===")
    for K in Ks_sorted:
        ab_adj = schedules[K][t_mid]
        d = ab_adj - base_mid
        if d > 1e-9:
            tag = "HIGHER ᾱ → LESS noise"
        elif d < -1e-9:
            tag = "LOWER  ᾱ → MORE noise"
        else:
            tag = "unchanged"
        print(f"  K={K:>2d}: ᾱ_adj = {ab_adj:.6f}   Δ = {d:+.6f}   ({tag})")

    print("\n[schedule_utils] ✓ Sanity check complete.")
