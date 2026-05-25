# src/gaussian_diffusion.py

"""
Gaussian diffusion for the numerical block of TabDDPM.

This module is identical across all experiments - only the multinomial
schedule changes between baseline and cardinality-adjusted variants.

Forward (training, closed-form jump from x_0 → x_t):
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,    ε ~ N(0, I)

Linear schedule (Ho et al., 2020):
    β_1 = 1e-4,  β_T = 0.02,  T = 1000
    ᾱ_t = ∏_{s=1}^{t} (1 - β_s)

Loss (simplified DDPM - predict ε):
    L = E_{t, x_0, ε} [ ||ε − ε_θ(x_t, t)||² ]

Reverse step (DDPM, x_t → x_{t-1}):
    x_{t-1} = (1/√α_t) · (x_t − (β_t/√(1-ᾱ_t)) · ε_θ) + σ_t · z
    σ_t    = √β_t,   z ~ N(0, I) for t > 0, else 0 (deterministic final step)

Design choice - reverse step takes the model's ε prediction as input rather
than the model itself. This keeps gaussian_diffusion.py free of the MLP's
joint num/cat input signature; the model call is orchestrated in
`src/sampling.py`.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


DeviceLike = Union[str, torch.device]


class GaussianDiffusion:
    """Standard DDPM for continuous features. Schedule tensors live on `device`."""

    def __init__(
        self,
        n_steps:    int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        device:     DeviceLike = "cpu",
    ):
        self.n_steps = n_steps
        self.device  = torch.device(device)

        # Linear noise schedule: beta grows linearly from 1e-4 to 0.02
        # over T=1000 steps (Ho et al., 2020, Section 4).
        self.betas      = torch.linspace(beta_start, beta_end, n_steps, device=self.device)

        # alpha_t = 1 - beta_t (per-step signal retention)
        self.alphas     = 1.0 - self.betas

        # alpha_bar_t = product of all alphas up to t -- cumulative signal fraction.
        # Decays from ~1.0 (pure signal) to ~0.0 (pure noise) over T steps.
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute square roots used repeatedly in forward/reverse formulas.
        # sqrt(alpha_bar_t) -- coefficient of x_0 in the forward process
        self.sqrt_alpha_bars            = torch.sqrt(self.alpha_bars)
        # sqrt(1 - alpha_bar_t) -- coefficient of epsilon in the forward process
        self.sqrt_one_minus_alpha_bars  = torch.sqrt(1.0 - self.alpha_bars)
        # sqrt(1 / alpha_t) -- used in the reverse mean computation
        self.sqrt_recip_alphas          = torch.sqrt(1.0 / self.alphas)

    # Forward (training) ---------------------------------------

    def q_sample(
        self,
        x_0:   torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0) in closed form.

        Args:
            x_0:   (B, D)  clean numerical features.
            t:     (B,)    integer timesteps in [0, n_steps).
            noise: (B, D) or None - if None, fresh standard-normal noise is sampled.

        Returns:
            x_t:   (B, D)  noisy sample.
            noise: (B, D)  the noise used (returned so the loss can target it).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Index into precomputed schedule; unsqueeze for broadcasting over D features.
        sqrt_ab = self.sqrt_alpha_bars[t].unsqueeze(-1)            # (B, 1)
        sqrt_om = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)  # (B, 1)

        # Closed-form reparameterisation: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
        # This jumps directly from x_0 to any timestep t without iterating.
        x_t = sqrt_ab * x_0 + sqrt_om * noise
        return x_t, noise

    def compute_loss(self, noise_true: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """Simplified DDPM loss: L = E[||eps - eps_theta(x_t, t)||^2].
        Equivalent to the variational bound up to a constant (Ho et al., Eq. 14)."""
        return F.mse_loss(noise_pred, noise_true)

    # Reverse (sampling) ---------------------------------------

    def p_sample_step(
        self,
        x_t:      torch.Tensor,
        t:        torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDPM reverse step x_t → x_{t-1}.

        Args:
            x_t:      (B, D)  current sample.
            t:        (B,)    integer timestep - same value across the batch is
                              typical at sampling time, but per-sample t is supported.
            eps_pred: (B, D)  model's ε̂_θ(x_t, t).

        Returns:
            x_{t-1}: (B, D)
        """
        # Gather per-timestep schedule values; unsqueeze for broadcasting over D.
        beta_t  = self.betas[t].unsqueeze(-1)
        sqrt_ra = self.sqrt_recip_alphas[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)

        # Posterior mean: mu_theta = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
        # This is DDPM Eq. 11 -- subtracts the predicted noise, scaled appropriately.
        mean = sqrt_ra * (x_t - (beta_t / sqrt_om) * eps_pred)

        # At t=0, the reverse step is deterministic (no noise added).
        # For t>0, add Gaussian noise scaled by sigma_t = sqrt(beta_t).
        nonzero = (t > 0).float().unsqueeze(-1)
        sigma_t = torch.sqrt(beta_t)
        z       = torch.randn_like(x_t)
        return mean + nonzero * sigma_t * z


# Sanity check ---------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_utils import load_split

    train = load_split("train")
    x_num = torch.from_numpy(train["x_num"]).float()
    D     = x_num.shape[1]
    print(f"[gauss] loaded Adult numericals: {tuple(x_num.shape)}  dtype={x_num.dtype}")

    diff = GaussianDiffusion(n_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu")
    print(f"[gauss] T={diff.n_steps}  β∈[{diff.betas[0]:.4e}, {diff.betas[-1]:.4e}]")
    print(f"[gauss] ᾱ_t: t=0 → {diff.alpha_bars[0].item():.6f}   "
          f"t=T-1 → {diff.alpha_bars[-1].item():.3e}")

    # 1. Shape sanity at a single t
    B = 64
    x0 = x_num[:B]
    t500 = torch.full((B,), 500, dtype=torch.long)
    x_t, eps = diff.q_sample(x0, t500)
    assert x_t.shape == x0.shape and eps.shape == x0.shape, "shape mismatch"
    print(f"[gauss] q_sample @ t=500 : x_t={tuple(x_t.shape)}, eps={tuple(eps.shape)}  ✓")

    # 2. Marginal statistics across timesteps - x_t should drift toward N(0,1)
    print(f"\n[gauss] x_t marginals over a large batch (target: mean≈0, std≈1 at high t):")
    print(f"  {'t':>5s}  {'ᾱ_t':>10s}  {'√ᾱ_t':>8s}  {'mean':>8s}  {'std':>8s}  {'corr(x0,xt)':>12s}")
    xb = x_num[:4096]
    for t_val in [0, 50, 100, 250, 500, 750, 900, 999]:
        t_b      = torch.full((xb.shape[0],), t_val, dtype=torch.long)
        x_t_b, _ = diff.q_sample(xb, t_b)
        # Pearson correlation between flattened x_0 and x_t - should drop to ≈0
        x0_flat = (xb - xb.mean()) / xb.std()
        xt_flat = (x_t_b - x_t_b.mean()) / x_t_b.std()
        corr    = (x0_flat * xt_flat).mean().item()
        print(f"  {t_val:>5d}  {diff.alpha_bars[t_val].item():>10.6f}  "
              f"{diff.sqrt_alpha_bars[t_val].item():>8.4f}  "
              f"{x_t_b.mean().item():>8.4f}  {x_t_b.std().item():>8.4f}  "
              f"{corr:>12.4f}")

    # 3. Loss sanity - perfect ε → 0, garbage ε > 0
    perfect = diff.compute_loss(eps, eps).item()
    garbage = diff.compute_loss(eps, torch.randn_like(eps)).item()
    print(f"\n[gauss] MSE loss : perfect ε → {perfect:.2e}   random ε → {garbage:.4f}  ✓")

    # 4. Reverse-step shape
    x_prev = diff.p_sample_step(x_t, t500, torch.zeros_like(x_t))
    assert x_prev.shape == x_t.shape
    print(f"[gauss] p_sample_step  : x_{{t-1}}={tuple(x_prev.shape)}  ✓")

    # 5. Reverse step at t=0 is deterministic given eps_pred (no σ·z term)
    t0     = torch.zeros(B, dtype=torch.long)
    ep0    = torch.zeros_like(x_t)
    a      = diff.p_sample_step(x_t, t0, ep0)
    b      = diff.p_sample_step(x_t, t0, ep0)
    delta  = (a - b).abs().max().item()
    print(f"[gauss] p_sample @ t=0 deterministic: max|Δ|={delta:.2e}  ✓")

    # 6. Reverse step at t>0 is stochastic
    a2 = diff.p_sample_step(x_t, t500, ep0)
    b2 = diff.p_sample_step(x_t, t500, ep0)
    delta2 = (a2 - b2).abs().max().item()
    print(f"[gauss] p_sample @ t=500 stochastic:  max|Δ|={delta2:.4f}  ✓")

    print("\n[gauss] ✓ Sanity check complete.")
