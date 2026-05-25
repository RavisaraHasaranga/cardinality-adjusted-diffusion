# src/train.py

"""
Hybrid TabDDPM training loop.

Loss (Kotelnikov et al., 2023):
    L_total = L_MSE (Gaussian numericals) + λ · L_KL (multinomial categoricals)

Per-batch:
    1. Sample t ~ Uniform[0, T) per sample.
    2. q_sample noise into numerical and categorical blocks.
    3. MLP forward: returns ε̂ for numericals, x̂_0 logits for categoricals.
    4. Compute Gaussian MSE on ε and per-feature multinomial VLB; sum and step.

The schedule (shared vs MI-adjusted) lives entirely inside the
`MultinomialDiffusion` instance - the training loop is identical for both.
"""

import time
from pathlib import Path
from typing  import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.gaussian_diffusion    import GaussianDiffusion
from src.multinomial_diffusion import MultinomialDiffusion
from src.mlp                   import TabDDPMMLP, indices_to_onehot


DeviceLike = Union[str, torch.device]


# Single training step ---------------------------------------


def train_step(
    model:        TabDDPMMLP,
    gauss_diff:   GaussianDiffusion,
    mn_diff:      MultinomialDiffusion,
    x_num:        torch.Tensor,
    x_cat:        torch.Tensor,
    y:            torch.Tensor,
    t:            torch.Tensor,
    lambda_mn:    float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hybrid loss for a single batch.

    Args:
        model, gauss_diff, mn_diff: components (model is in train mode).
        x_num:      (B, n_num)  float - clean numericals.
        x_cat:      (B, n_cat)  long  - clean categorical indices.
        y:          (B,)        long  - class labels.
        t:          (B,)        long  - sampled timesteps.
        lambda_mn:  multiplier on the multinomial loss (TabDDPM uses 1.0).

    Returns:
        total_loss, gaussian_loss, multinomial_loss  - each a scalar tensor.
    """
    # ---- Forward diffusion (add noise) ----
    # Gaussian: x_num_noisy = sqrt(alpha_bar_t) * x_num + sqrt(1-alpha_bar_t) * epsilon
    x_num_noisy, noise = gauss_diff.q_sample(x_num, t)
    # Multinomial: sample x_cat_noisy from q(x_t | x_0) per the transition matrix.
    # Uses per-feature adjusted alpha_bars when mn_diff was constructed with them.
    x_cat_noisy        = mn_diff.q_sample(x_cat, t)
    # MLP expects one-hot categoricals, not integer indices.
    x_cat_onehot       = indices_to_onehot(x_cat_noisy, mn_diff.cardinalities)

    # ---- Model prediction ----
    # eps_pred: predicted noise for numericals (same shape as x_num).
    # cat_logits: list of (B, K_i) tensors - unnormalised log-probs of x_0 per feature.
    eps_pred, cat_logits = model(x_num_noisy, x_cat_onehot, t, y)

    # ---- Loss computation ----
    # Gaussian: MSE between true noise epsilon and predicted epsilon.
    loss_g = gauss_diff.compute_loss(noise, eps_pred)
    # Multinomial: KL divergence of the variational lower bound (VLB),
    # comparing the true posterior q(x_{t-1}|x_t,x_0) against the model's
    # predicted posterior p_theta(x_{t-1}|x_t) derived from cat_logits.
    loss_m = mn_diff.compute_loss(x_cat, x_cat_noisy, cat_logits, t)
    # TabDDPM combines both losses; lambda_mn weights the categorical term.
    total  = loss_g + lambda_mn * loss_m
    return total, loss_g, loss_m


# DataLoader helper ---------------------------------------


def make_dataloader(
    split:      Dict[str, np.ndarray],
    batch_size: int  = 256,
    shuffle:    bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from a `load_split`-style dict."""
    x_num = torch.from_numpy(split["x_num"]).float()
    x_cat = torch.from_numpy(split["x_cat"]).long()
    y     = torch.from_numpy(split["y"]).long()
    return DataLoader(
        TensorDataset(x_num, x_cat, y),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=False,
    )


# Training loop ---------------------------------------


def train(
    model:            TabDDPMMLP,
    gauss_diff:       GaussianDiffusion,
    mn_diff:          MultinomialDiffusion,
    train_loader:     DataLoader,
    n_steps:          int   = 10_000,
    lr:               float = 1e-3,
    weight_decay:     float = 0.0,
    lambda_mn:        float = 1.0,
    grad_clip:        Optional[float] = 1.0,
    use_scheduler:    bool  = True,
    log_every:        int   = 100,
    checkpoint_path:  Optional[Union[str, Path]] = None,
    checkpoint_every: Optional[int] = None,
    device:           DeviceLike = "cpu",
    seed:             int   = 42,
) -> List[Tuple[int, float, float, float]]:
    """
    Run `n_steps` of AdamW training. Returns the loss log.

    The training loop cycles `train_loader` indefinitely so `n_steps` is
    treated as raw optimizer step count, not epochs.

    Returns:
        list of (step_index, total_loss, gaussian_loss, multinomial_loss).
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model  = model.to(device)

    # AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
    # Weight decay is kept at 0 by default (matching TabDDPM reference).
    optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing smoothly decays LR from `lr` to 0 over `n_steps`.
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=n_steps, eta_min=0)

    T = gauss_diff.n_steps  # total diffusion timesteps (typically 1000)
    losses: List[Tuple[int, float, float, float]] = []

    model.train()

    # Infinite iterator: cycle through the DataLoader so we can train for a
    # fixed number of optimizer steps rather than counting epochs.
    def _cycle(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch
    data_iter = _cycle(train_loader)

    t0 = time.time()
    for step in range(n_steps):
        x_num, x_cat, y = next(data_iter)
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y     = y    .to(device, non_blocking=True)

        # Sample a random timestep t in [0, T) independently for each sample
        # in the batch (uniform timestep sampling as in standard DDPM).
        B = x_num.shape[0]
        t = torch.randint(0, T, (B,), device=device)

        total, lg, lm = train_step(model, gauss_diff, mn_diff,
                                   x_num, x_cat, y, t, lambda_mn)

        # Standard gradient update with optional gradient clipping to stabilise
        # training (max_norm=1.0 by default).
        optim.zero_grad(set_to_none=True)  # faster than zero_grad() - frees memory
        total.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optim.step()
        if scheduler is not None:
            scheduler.step()

        losses.append((step, total.detach().item(), lg.detach().item(), lm.detach().item()))

        # Periodic logging: report running averages of total, Gaussian, and
        # multinomial losses over the last `log_every` steps.
        if log_every and (step + 1) % log_every == 0:
            recent = losses[-log_every:]
            mt = np.mean([r[1] for r in recent])
            mg = np.mean([r[2] for r in recent])
            mm = np.mean([r[3] for r in recent])
            print(f"  step {step+1:>6d}  loss={mt:.4f}  "
                  f"(g={mg:.4f}  m={mm:.4f})  "
                  f"{time.time()-t0:5.1f}s")

        if checkpoint_path and checkpoint_every and (step + 1) % checkpoint_every == 0:
            torch.save(
                {"step": step + 1,
                 "model": model.state_dict(),
                 "optim": optim.state_dict(),
                 "losses": losses},
                str(checkpoint_path),
            )

    return losses


# Sanity check ---------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_utils import load_split, load_meta

    meta          = load_meta()
    cardinalities = meta["cardinalities"]
    n_num         = meta["n_num"]
    n_classes     = 2

    train_split  = load_split("train")
    train_loader = make_dataloader(train_split, batch_size=256)
    print(f"[train] data: {len(train_split['y']):,} train rows, batch=256")

    torch.manual_seed(0)
    model = TabDDPMMLP(
        n_num=n_num,
        cardinalities=cardinalities,
        n_classes=n_classes,
        hidden_dim=256, n_layers=3, t_emb_dim=128,
    )
    print(f"[train] model: {sum(p.numel() for p in model.parameters()):,} params")

    gauss_diff = GaussianDiffusion(n_steps=1000)
    mn_diff    = MultinomialDiffusion(cardinalities=cardinalities, n_steps=1000)

    print(f"\n[train] running 100-step sanity check (baseline schedule) ...")
    losses = train(
        model, gauss_diff, mn_diff, train_loader,
        n_steps=100, lr=1e-3, log_every=10,
        device="cpu", seed=42,
    )

    # Decreasing-loss sanity
    n_window  = 10
    first_avg = float(np.mean([r[1] for r in losses[:n_window]]))
    last_avg  = float(np.mean([r[1] for r in losses[-n_window:]]))
    drop      = first_avg - last_avg
    first_g   = float(np.mean([r[2] for r in losses[:n_window]]))
    last_g    = float(np.mean([r[2] for r in losses[-n_window:]]))
    first_m   = float(np.mean([r[3] for r in losses[:n_window]]))
    last_m    = float(np.mean([r[3] for r in losses[-n_window:]]))

    print(f"\n[train] loss summary (avg of first vs last {n_window} steps):")
    print(f"  total : {first_avg:.4f}  →  {last_avg:.4f}   "
          f"({'↓' if drop > 0 else '↑'} {drop:+.4f})")
    print(f"  gauss : {first_g:.4f}  →  {last_g:.4f}   ({first_g - last_g:+.4f})")
    print(f"  multi : {first_m:.4f}  →  {last_m:.4f}   ({first_m - last_m:+.4f})")
    assert drop > 0, f"loss did not decrease: first={first_avg:.4f} last={last_avg:.4f}"

    # Repeat sanity under adjusted schedule
    from src.schedule_utils import compute_adjusted_schedules
    betas           = np.linspace(1e-4, 0.02, 1000)
    base_alpha_bars = np.cumprod(1.0 - betas)
    adj             = compute_adjusted_schedules(base_alpha_bars, cardinalities, K_ref=2)

    torch.manual_seed(0)
    model_adj = TabDDPMMLP(n_num=n_num, cardinalities=cardinalities, n_classes=n_classes,
                            hidden_dim=256, n_layers=3, t_emb_dim=128)
    mn_diff_adj = MultinomialDiffusion(cardinalities=cardinalities, n_steps=1000,
                                        adjusted_alpha_bars=adj)

    print(f"\n[train] running 100-step sanity check (adjusted schedule) ...")
    losses_adj = train(
        model_adj, gauss_diff, mn_diff_adj, train_loader,
        n_steps=100, lr=1e-3, log_every=10,
        device="cpu", seed=42,
    )
    first_a = float(np.mean([r[1] for r in losses_adj[:n_window]]))
    last_a  = float(np.mean([r[1] for r in losses_adj[-n_window:]]))
    print(f"  adjusted total: {first_a:.4f}  →  {last_a:.4f}   "
          f"({'↓' if first_a > last_a else '↑'} {first_a - last_a:+.4f})")
    assert first_a > last_a, "adjusted-schedule loss did not decrease"

    print(f"\n[train] ✓ Sanity check complete.")
