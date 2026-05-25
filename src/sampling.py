# src/sampling.py

"""
Joint reverse sampling for hybrid TabDDPM.

Starts from pure noise - x_num ~ N(0, I), x_cat[:, i] ~ Uniform({0,…,K_i-1}) -
then iterates t = T-1 down to 0, stepping the Gaussian (numerical) and
multinomial (categorical) reverse processes with a single MLP forward pass
per step. Both the shared-schedule baseline and the per-feature MI-adjusted
variant work through the same loop - the difference lives entirely in the
`MultinomialDiffusion` instance supplied by the caller.
"""

from typing import Optional, Tuple, Union

import torch

from src.gaussian_diffusion    import GaussianDiffusion
from src.multinomial_diffusion import MultinomialDiffusion
from src.mlp                   import TabDDPMMLP, indices_to_onehot


DeviceLike = Union[str, torch.device]


@torch.no_grad()
def sample(
    model:         TabDDPMMLP,
    gauss_diff:    GaussianDiffusion,
    mn_diff:       MultinomialDiffusion,
    n_samples:     int,
    y:             Optional[torch.Tensor] = None,
    device:        Optional[DeviceLike]   = None,
    show_progress: bool                   = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic mixed-type batch.

    Args:
        model:         Trained `TabDDPMMLP`. Set to `.eval()` automatically.
        gauss_diff:    `GaussianDiffusion` instance (numerical schedule).
        mn_diff:       `MultinomialDiffusion` instance (categorical schedule -
                       shared OR per-feature adjusted; the loop is identical).
        n_samples:     batch size to generate.
        y:             (n_samples,) class labels for conditional generation.
                       If None, sampled uniformly from [0, n_classes).
        device:        Device for inputs. Defaults to the model's device.
        show_progress: Wrap the t-loop in tqdm if installed.

    Returns:
        x_num: (n_samples, n_num)  float - quantile-normal space (use
               `meta['qt'].inverse_transform` to recover original units).
        x_cat: (n_samples, n_cat)  long  - class indices in [0, K_i) per column.
    """
    model.eval()
    device = torch.device(device) if device is not None else next(model.parameters()).device

    T = gauss_diff.n_steps
    if mn_diff.T != T:
        raise ValueError(f"Schedule length mismatch: gauss T={T}, mn T={mn_diff.T}")

    cardinalities = model.cardinalities
    n_num         = model.n_num

    # ---- Initialise at t = T (pure noise) ----
    # Numericals: sample from the standard Gaussian prior N(0, I).
    x_num = torch.randn(n_samples, n_num, device=device)
    # Categoricals: sample each feature independently from Uniform({0, ..., K_i - 1}).
    x_cat = torch.stack(
        [torch.randint(0, K, (n_samples,), device=device, dtype=torch.long)
         for K in cardinalities],
        dim=1,
    )

    # If no class labels supplied, draw them uniformly for unconditional generation.
    if y is None:
        n_classes = model.y_embed.num_embeddings
        y = torch.randint(0, n_classes, (n_samples,), device=device, dtype=torch.long)
    else:
        y = y.to(device).long()

    # ---- Reverse loop: iterate from t = T-1 down to t = 0 ----
    iterator = range(T - 1, -1, -1)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="sampling", total=T)
        except ImportError:
            pass

    for t_idx in iterator:
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        # Convert integer cat indices to one-hot so the MLP can process them.
        x_cat_onehot = indices_to_onehot(x_cat, cardinalities)

        # Single forward pass: MLP predicts epsilon (for numericals) and
        # x_0 logits (for each categorical feature) simultaneously.
        eps_pred, cat_logits = model(x_num, x_cat_onehot, t, y)

        # Gaussian reverse step: use predicted epsilon to denoise x_num by one step.
        x_num = gauss_diff.p_sample_step(x_num, t, eps_pred)

        # Multinomial reverse step: softmax the logits to get x_0 prediction,
        # compute the posterior q(x_{t-1} | x_t, x_0_hat), and sample from it.
        # The per-feature adjusted schedule (if any) is handled inside mn_diff.
        x_cat = mn_diff.p_sample_step(x_cat, t, cat_logits)

    return x_num, x_cat


# Sanity check ---------------------------------------

if __name__ == "__main__":
    import sys
    import time
    sys.path.append(".")

    import numpy as np
    from src.data_utils     import load_meta
    from src.schedule_utils import compute_adjusted_schedules

    meta          = load_meta()
    cardinalities = meta["cardinalities"]
    n_num         = meta["n_num"]
    n_classes     = 2
    T             = 1000
    B             = 64

    print(f"[sample] cardinalities = {cardinalities}")
    print(f"[sample] n_num={n_num}  n_classes={n_classes}  T={T}  B={B}")

    torch.manual_seed(0)
    model = TabDDPMMLP(
        n_num=n_num,
        cardinalities=cardinalities,
        n_classes=n_classes,
        hidden_dim=256,
        n_layers=3,
        t_emb_dim=128,
    )
    print(f"[sample] model: {sum(p.numel() for p in model.parameters()):,} params (UNTRAINED)")

    gauss_diff = GaussianDiffusion(n_steps=T)

    # --- 1. Baseline (shared) schedule ----------------------------------------
    mn_base = MultinomialDiffusion(cardinalities=cardinalities, n_steps=T)

    print(f"\n[sample] generating with BASELINE (shared) schedule ...")
    t0 = time.time()
    x_num, x_cat = sample(model, gauss_diff, mn_base, n_samples=B)
    dt = time.time() - t0
    print(f"[sample] done in {dt:.2f}s  ({dt/T*1000:.1f} ms/step)")

    # Shape + dtype checks
    print(f"\n[sample] shape checks:")
    print(f"  x_num: shape={tuple(x_num.shape)}  dtype={x_num.dtype}")
    print(f"  x_cat: shape={tuple(x_cat.shape)}  dtype={x_cat.dtype}")
    assert x_num.shape == (B, n_num) and x_num.dtype == torch.float32
    assert x_cat.shape == (B, len(cardinalities)) and x_cat.dtype == torch.long

    # Numerical finiteness + stats (UNTRAINED model → expect drift, no NaN/Inf)
    finite = torch.isfinite(x_num).all().item()
    print(f"\n[sample] x_num stats (untrained model - N(0,1) target only with training):")
    print(f"  all finite = {finite}")
    print(f"  mean = {x_num.mean().item():+.4f}   std = {x_num.std().item():.4f}")
    print(f"  min  = {x_num.min().item():+.4f}   max = {x_num.max().item():+.4f}")
    assert finite, "non-finite x_num - math is broken, not just untrained"

    # Categorical bounds - must always hold regardless of training
    print(f"\n[sample] x_cat bounds (must satisfy 0 ≤ x_cat[:,i] < K_i for every i):")
    all_ok = True
    for i, K in enumerate(cardinalities):
        col = x_cat[:, i]
        lo, hi = col.min().item(), col.max().item()
        ok = (lo >= 0) and (hi < K)
        all_ok = all_ok and ok
        n_unique = col.unique().numel()
        print(f"  feat {i}  K={K:>2d}  range=[{lo:>2d},{hi:>2d}]  "
              f"unique={n_unique:>2d}/{K}  {'✓' if ok else '✗'}")
    assert all_ok, "categorical bounds violated"

    # --- 2. Adjusted schedule -------------------------------------------------
    betas           = np.linspace(1e-4, 0.02, T)
    base_alpha_bars = np.cumprod(1.0 - betas)
    adj_schedules   = compute_adjusted_schedules(base_alpha_bars, cardinalities, K_ref=2)
    mn_adj = MultinomialDiffusion(
        cardinalities=cardinalities, n_steps=T, adjusted_alpha_bars=adj_schedules
    )

    print(f"\n[sample] generating with ADJUSTED (MI-equalised) schedule ...")
    t0 = time.time()
    x_num_a, x_cat_a = sample(model, gauss_diff, mn_adj, n_samples=B)
    dt = time.time() - t0
    print(f"[sample] done in {dt:.2f}s  ({dt/T*1000:.1f} ms/step)")

    assert torch.isfinite(x_num_a).all().item(), "non-finite x_num under adjusted schedule"
    cat_bounds_ok = all(
        (x_cat_a[:, i].min().item() >= 0) and (x_cat_a[:, i].max().item() < K)
        for i, K in enumerate(cardinalities)
    )
    print(f"  x_num finite: {torch.isfinite(x_num_a).all().item()}   "
          f"x_num stats:  mean={x_num_a.mean().item():+.4f} std={x_num_a.std().item():.4f}")
    print(f"  x_cat in-bounds for every feature: {cat_bounds_ok}")
    assert cat_bounds_ok

    print(f"\n[sample] ✓ Sanity check complete.")
