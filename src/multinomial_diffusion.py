# src/multinomial_diffusion.py

"""
Multinomial diffusion for the categorical block of TabDDPM.

Forward process for one categorical feature with K classes (uniform noise):
    Q_t[v, j] = (1 - β_t) · δ(v, j) + β_t · (1/K)        (one-step transition)
    P(x_t = x_0)     = ᾱ_t + (1 - ᾱ_t) / K
    P(x_t = j ≠ x_0) = (1 - ᾱ_t) / K

Reverse posterior (Bayes' rule):
    q(x_{t-1}=v | x_t, x_0) ∝ q(x_t | x_{t-1}=v) · q(x_{t-1}=v | x_0)
                           = [(1-β_t) · 1{v=x_t} + β_t/K] · [ᾱ_{t-1} · 1{v=x_0} + (1-ᾱ_{t-1})/K]

VLB loss (per-feature, summed across timesteps in expectation):
    t > 0 :  L_t = KL[ q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t) ]
    t = 0 :  L_0 = -log p_θ(x_0 | x_1)   (reconstruction NLL)

Auxiliary CE loss (D3PM-style, Austin et al. 2021):
    L_aux = CE(softmax(logits), x_0)   at every timestep

    The VLB KL is naturally tiny (~0.001 nats) because the posterior is
    dominated by the likelihood q(x_t|x_{t-1}) at most timesteps - the
    model gets almost no gradient for categorical prediction from KL alone.
    The auxiliary CE directly trains x̂_0 prediction and provides ~100x
    stronger gradient signal.  Total: L_m = L_VLB + aux_weight · L_CE.

The MLP predicts x̂_0 logits (one head per feature). We plug F.softmax(logits)
into the same q_posterior formula in place of the true x_0 one-hot - the
formula is linear in x_0, so the predicted distribution flows through cleanly.

Per-feature schedules:
    adjusted_alpha_bars=None  → baseline (shared schedule for every feature)
    adjusted_alpha_bars=dict  → {K_i: np.ndarray (T,)} from `schedule_utils`

The schedule is stored as a (C, T) tensor so per-sample lookup is just
`alpha_bars[i, t]`. Per-feature β is derived from per-feature ᾱ via
β_t = 1 - ᾱ_t / ᾱ_{t-1} so a single q_posterior code path serves both modes.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


DeviceLike = Union[str, torch.device]


class MultinomialDiffusion:
    """Uniform-noise multinomial diffusion with optional per-feature schedules."""

    def __init__(
        self,
        cardinalities:       Sequence[int],
        n_steps:             int   = 1000,
        beta_start:          float = 1e-4,
        beta_end:            float = 0.02,
        adjusted_alpha_bars: Optional[Dict[int, np.ndarray]] = None,
        aux_weight:          float = 1.0,
        device:              DeviceLike = "cpu",
    ):
        self.cardinalities = [int(k) for k in cardinalities]
        self.C             = len(self.cardinalities)       # number of categorical features
        self.T             = int(n_steps)
        self.device        = torch.device(device)
        self.adjusted      = adjusted_alpha_bars is not None
        self.aux_weight    = aux_weight

        # Build the shared linear schedule (same as gaussian_diffusion).
        shared_betas      = torch.linspace(beta_start, beta_end, n_steps,
                                            device=self.device, dtype=torch.float32)
        shared_alpha_bars = torch.cumprod(1.0 - shared_betas, dim=0)

        # Both modes produce alpha_bars of shape (C, T) -- one row per feature.
        # In baseline mode every row is identical; in adjusted mode each feature
        # has a different schedule derived from MI equalisation.
        if not self.adjusted:
            # Baseline: every feature uses the same schedule. `.expand` is a view,
            # so .contiguous() materialises it for safe in-place ops downstream.
            self.alpha_bars = shared_alpha_bars.unsqueeze(0).expand(self.C, -1).contiguous()
            self.betas      = shared_betas.unsqueeze(0).expand(self.C, -1).contiguous()
        else:
            # Adjusted mode: look up the precomputed alpha_bar curve for each
            # feature's cardinality K_i (from schedule_utils.compute_adjusted_schedules).
            self.alpha_bars = torch.empty(self.C, self.T,
                                          device=self.device, dtype=torch.float32)
            for i, K in enumerate(self.cardinalities):
                if K not in adjusted_alpha_bars:
                    raise KeyError(
                        f"adjusted_alpha_bars missing K={K} "
                        f"(provided keys: {sorted(adjusted_alpha_bars.keys())})"
                    )
                self.alpha_bars[i] = torch.as_tensor(
                    adjusted_alpha_bars[K], device=self.device, dtype=torch.float32
                )
            # Recover per-step betas from cumulative alpha_bars:
            # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, with alpha_bar_{-1} = 1.
            # Clamp denominator to avoid division by zero at late timesteps.
            self.betas = torch.empty_like(self.alpha_bars)
            self.betas[:, 0]  = 1.0 - self.alpha_bars[:, 0]
            self.betas[:, 1:] = 1.0 - self.alpha_bars[:, 1:] / self.alpha_bars[:, :-1].clamp(min=1e-12)
            self.betas = self.betas.clamp(0.0, 1.0 - 1e-7)

        # Shifted alpha_bars for the posterior: alpha_bar_{t-1} with the convention
        # alpha_bar_{-1} = 1 (i.e. no noise before the first step).
        ones_col = torch.ones(self.C, 1, device=self.device, dtype=torch.float32)
        self.alpha_bars_prev = torch.cat([ones_col, self.alpha_bars[:, :-1]], dim=1)

    # Forward sampling ---------------------------------------

    def q_sample(self, x_0_indices: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) per categorical feature.

        Args:
            x_0_indices: (B, C) long - class indices in [0, K_i) per column.
            t:           (B,)  long - timestep indices in [0, T).

        Returns:
            x_t_indices: (B, C) long - sampled noisy class indices.
        """
        B   = x_0_indices.shape[0]
        x_t = torch.empty_like(x_0_indices)

        # Process each categorical feature independently (each may have a
        # different cardinality K and, in adjusted mode, a different schedule).
        for i, K in enumerate(self.cardinalities):
            x0_i = x_0_indices[:, i].long()                          # (B,)
            ab_t = self.alpha_bars[i, t]                             # (B,)

            # Multinomial forward marginal (closed-form jump from x_0 to x_t):
            #   P(x_t = x_0) = alpha_bar_t + (1 - alpha_bar_t) / K
            #   P(x_t = j != x_0) = (1 - alpha_bar_t) / K
            # The first term is the "stay" probability; the second spreads
            # the remaining mass uniformly across all K classes (including x_0).
            p_correct   = (ab_t + (1.0 - ab_t) / K).unsqueeze(-1)    # (B, 1)
            p_incorrect = ((1.0 - ab_t) / K).unsqueeze(-1)           # (B, 1)

            # Start with uniform incorrect probability, then overwrite the
            # true-class column with the higher "correct" probability.
            probs = p_incorrect.expand(B, K).clone()                 # (B, K)
            probs.scatter_(1, x0_i.unsqueeze(-1), p_correct)

            # Sample from the per-class categorical distribution.
            x_t[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return x_t

    # Posterior ---------------------------------------

    @staticmethod
    def _q_posterior(
        xt_onehot: torch.Tensor,
        x0_dist:   torch.Tensor,
        ab_t:      torch.Tensor,
        ab_tm1:    torch.Tensor,
        beta_t:    torch.Tensor,
        K:         int,
    ) -> torch.Tensor:
        """
        q(x_{t-1} = v | x_t, x_0) for a single categorical feature.

        Bayes' rule decomposes the posterior into two factors:
            q(x_{t-1}=v | x_t, x_0) proportional to q(x_t | x_{t-1}=v) * q(x_{t-1}=v | x_0)
        Both factors have closed forms under the uniform-noise transition matrix.

        When x0_dist is the model's softmax prediction (instead of a true one-hot),
        this computes p_theta's posterior -- the formula is linear in x_0 so the
        soft distribution substitutes directly.

        Args:
            xt_onehot: (B, K)  one-hot of x_t.
            x0_dist:   (B, K)  one-hot of true x_0  OR  softmax of x_hat_0 logits.
            ab_t:      (B,)    alpha_bar_t.
            ab_tm1:    (B,)    alpha_bar_{t-1}  (= 1 at t=0).
            beta_t:    (B,)    beta_t.
            K:         int     cardinality.

        Returns:
            posterior: (B, K)  normalised distribution over x_{t-1}.
        """
        beta_t = beta_t.unsqueeze(-1)
        ab_tm1 = ab_tm1.unsqueeze(-1)

        # Factor 1 -- likelihood: q(x_t | x_{t-1}=v)
        # From the transition matrix Q_t = (1-beta_t)*I + (beta_t/K)*11^T:
        #   probability is (1-beta_t) if v == x_t, else beta_t/K.
        # Written compactly: (1-beta_t)*one_hot(x_t) + beta_t/K.
        likelihood = (1.0 - beta_t) * xt_onehot + beta_t / K       # (B, K)

        # Factor 2 -- prior: q(x_{t-1}=v | x_0)
        # Marginal from the cumulative schedule up to step t-1:
        #   alpha_bar_{t-1} * one_hot(x_0) + (1-alpha_bar_{t-1})/K.
        prior      = ab_tm1 * x0_dist + (1.0 - ab_tm1) / K          # (B, K)

        # Element-wise product and normalise to get a valid distribution.
        unnorm     = likelihood * prior
        return unnorm / unnorm.sum(dim=-1, keepdim=True).clamp(min=1e-30)

    # Reverse sampling ---------------------------------------

    def p_sample_step(
        self,
        x_t_indices: torch.Tensor,
        t:           torch.Tensor,
        cat_logits:  List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Single reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t) per feature.

        p_θ(x_{t-1} | x_t) = q_posterior(x_t, x̂_0) with x̂_0 = softmax(cat_logits).
        At t=0 the prior collapses to x̂_0 (ᾱ_{-1}=1), so sampling yields the
        model's final categorical prediction.

        Args:
            x_t_indices: (B, C) long.
            t:           (B,)   long.
            cat_logits:  list of C tensors, each (B, K_i).

        Returns:
            x_{t-1}_indices: (B, C) long.
        """
        x_prev = torch.empty_like(x_t_indices)

        for i, K in enumerate(self.cardinalities):
            xt_i          = x_t_indices[:, i].long()
            xt_onehot     = F.one_hot(xt_i, num_classes=K).float()

            # Model predicts x_hat_0 as logits; softmax gives a distribution over classes.
            x0_pred_probs = F.softmax(cat_logits[i], dim=-1)

            # Fetch the per-feature schedule values at timestep t.
            # In adjusted mode these differ across features; in baseline they are identical.
            ab_t   = self.alpha_bars[i,      t]
            ab_tm1 = self.alpha_bars_prev[i, t]
            beta_t = self.betas[i,           t]

            # Compute p_theta's posterior by plugging the predicted x_hat_0
            # distribution into the exact posterior formula, then sample.
            pred_post = self._q_posterior(xt_onehot, x0_pred_probs, ab_t, ab_tm1, beta_t, K)
            x_prev[:, i] = torch.multinomial(pred_post, num_samples=1).squeeze(-1)

        return x_prev

    # VLB loss ---------------------------------------

    def compute_loss(
        self,
        x_0_indices: torch.Tensor,
        x_t_indices: torch.Tensor,
        cat_logits:  List[torch.Tensor],
        t:           torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-batch VLB loss, mean over features.

            t > 0 : KL[ q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t) ]
            t = 0 : -log p_θ(x_0 | x_1)   (reconstruction)

        Args:
            x_0_indices: (B, C) long.
            x_t_indices: (B, C) long  (from `q_sample`).
            cat_logits:  list of C tensors, each (B, K_i) - model's x̂_0 logits.
            t:           (B,)  long - timestep indices.

        Returns:
            scalar loss - mean over features of (batch-mean per-feature loss).
        """
        per_feature: List[torch.Tensor] = []

        for i, K in enumerate(self.cardinalities):
            x0_i   = x_0_indices[:, i].long()
            xt_i   = x_t_indices[:, i].long()
            logits = cat_logits[i]                  # raw x_hat_0 logits from MLP head i

            x0_onehot     = F.one_hot(x0_i, num_classes=K).float()
            xt_onehot     = F.one_hot(xt_i, num_classes=K).float()
            x0_pred_probs = F.softmax(logits, dim=-1)

            ab_t   = self.alpha_bars[i,      t]
            ab_tm1 = self.alpha_bars_prev[i, t]
            beta_t = self.betas[i,           t]

            # True posterior uses the ground-truth x_0 one-hot.
            true_post = self._q_posterior(xt_onehot, x0_onehot,     ab_t, ab_tm1, beta_t, K)
            # Predicted posterior substitutes the model's soft x_hat_0 prediction.
            pred_post = self._q_posterior(xt_onehot, x0_pred_probs, ab_t, ab_tm1, beta_t, K)

            # KL divergence: KL[q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t)]
            # Summed over the K classes for each sample in the batch.
            kl = (true_post * (torch.log(true_post.clamp(min=1e-30))
                              - torch.log(pred_post.clamp(min=1e-30)))).sum(dim=-1)

            # At t=0, use reconstruction NLL instead of KL:
            # L_0 = -log p_theta(x_0 | x_1), the negative log-prob of the true class.
            recon = -torch.log(
                x0_pred_probs.gather(-1, x0_i.unsqueeze(-1)).squeeze(-1).clamp(min=1e-30)
            )

            # Switch between KL (t>0) and reconstruction NLL (t=0).
            vlb_i = torch.where(t == 0, recon, kl)

            # Auxiliary cross-entropy loss (D3PM, Austin et al. 2021):
            # The VLB KL is very small (~0.001 nats) at most timesteps because the
            # posterior is dominated by the likelihood factor q(x_t|x_{t-1}).
            # Adding CE on the x_hat_0 prediction gives ~100x stronger gradients,
            # which is critical for the model to actually learn categorical prediction.
            if self.aux_weight > 0:
                ce_aux = F.cross_entropy(logits, x0_i, reduction='none')
                loss_i = vlb_i + self.aux_weight * ce_aux
            else:
                loss_i = vlb_i
            per_feature.append(loss_i.mean())

        # Average across all C categorical features.
        return torch.stack(per_feature).mean()


# Sanity check ---------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_utils      import load_split, load_meta
    from src.schedule_utils  import compute_adjusted_schedules

    meta          = load_meta()
    cardinalities = meta["cardinalities"]
    T             = 1000

    print(f"[mn-diff] cardinalities (Adult): {cardinalities}")

    # --- 1. Construct both modes ----------------------------------------------
    mn_base = MultinomialDiffusion(cardinalities, n_steps=T, device="cpu")
    print(f"[mn-diff] baseline:  alpha_bars shape = {tuple(mn_base.alpha_bars.shape)}, "
          f"all features identical? "
          f"{torch.allclose(mn_base.alpha_bars[0], mn_base.alpha_bars[-1])}")

    betas_np         = np.linspace(1e-4, 0.02, T)
    base_alpha_bars  = np.cumprod(1.0 - betas_np)
    adj_schedules    = compute_adjusted_schedules(base_alpha_bars, cardinalities, K_ref=2)
    mn_adj = MultinomialDiffusion(cardinalities, n_steps=T,
                                  adjusted_alpha_bars=adj_schedules, device="cpu")

    print(f"\n[mn-diff] ᾱ at t=500 (base vs adjusted):")
    print(f"  {'feat':>4s} {'K':>3s} {'ᾱ_base':>10s} {'ᾱ_adj':>10s}")
    for i, K in enumerate(cardinalities):
        print(f"  {i:>4d} {K:>3d} {mn_base.alpha_bars[i,500].item():>10.4f} "
              f"{mn_adj.alpha_bars[i,500].item():>10.4f}")

    # --- 2. Forward sampling: empirical P(correct) vs analytical -------------
    print(f"\n[mn-diff] q_sample empirical P(x_t=x_0) - 10,000 samples per feature, t=500")

    def empirical_check(mn, label):
        torch.manual_seed(0)
        N    = 10_000
        x0   = torch.zeros((N, mn.C), dtype=torch.long)
        for i, K in enumerate(mn.cardinalities):
            x0[:, i] = 3 % K   # arbitrary fixed class within range
        t500 = torch.full((N,), 500, dtype=torch.long)
        x_t  = mn.q_sample(x0, t500)

        print(f"  {label}")
        print(f"  {'K':>3s} {'analytical':>11s} {'empirical':>10s} {'|Δ|':>7s}")
        for i, K in enumerate(mn.cardinalities):
            ab        = mn.alpha_bars[i, 500].item()
            analytic  = ab + (1.0 - ab) / K
            empiric   = (x_t[:, i] == x0[:, i]).float().mean().item()
            print(f"  {K:>3d} {analytic:>11.4f} {empiric:>10.4f} "
                  f"{abs(analytic - empiric):>7.4f}")

    empirical_check(mn_base, "baseline schedule:")
    empirical_check(mn_adj,  "adjusted schedule (K_ref=2):")

    # --- 3. KL loss finite & positive on a real batch ------------------------
    print(f"\n[mn-diff] KL loss on a real batch (B=64):")
    train      = load_split("train")
    x_cat_real = torch.from_numpy(train["x_cat"][:64]).long()
    B          = 64

    torch.manual_seed(0)
    t_rand    = torch.randint(1, T, (B,))   # KL path
    x_t_rand  = mn_base.q_sample(x_cat_real, t_rand)
    rnd_logits = [torch.randn(B, K) for K in cardinalities]

    loss_rand = mn_base.compute_loss(x_cat_real, x_t_rand, rnd_logits, t_rand)
    print(f"  random logits, KL path  : loss = {loss_rand.item():.4f}   "
          f"finite={torch.isfinite(loss_rand).item()}  >0={(loss_rand.item()>0)}")

    t_zero    = torch.zeros(B, dtype=torch.long)
    x_t_zero  = mn_base.q_sample(x_cat_real, t_zero)
    loss_rec  = mn_base.compute_loss(x_cat_real, x_t_zero, rnd_logits, t_zero)
    print(f"  random logits, recon (t=0): loss = {loss_rec.item():.4f}   "
          f"(expected ≈ mean log K_i = {np.mean(np.log(cardinalities)):.4f})")

    # --- 4. Perfect prediction → ~0 loss ------------------------------------
    print(f"\n[mn-diff] KL loss with PERFECT logits (expect ≈ 0):")
    perfect_logits = []
    for i, K in enumerate(cardinalities):
        l = torch.zeros(B, K)
        l.scatter_(1, x_cat_real[:, i:i+1], 20.0)  # softmax ≈ one-hot at x_0
        perfect_logits.append(l)

    loss_perfect_kl  = mn_base.compute_loss(x_cat_real, x_t_rand,  perfect_logits, t_rand)
    loss_perfect_rec = mn_base.compute_loss(x_cat_real, x_t_zero, perfect_logits, t_zero)
    print(f"  perfect logits, KL path  : loss = {loss_perfect_kl.item():.2e}")
    print(f"  perfect logits, recon    : loss = {loss_perfect_rec.item():.2e}")

    # --- 5. Same checks under adjusted schedule -----------------------------
    print(f"\n[mn-diff] Same KL checks under adjusted schedule:")
    x_t_rand_adj = mn_adj.q_sample(x_cat_real, t_rand)
    loss_rand_adj    = mn_adj.compute_loss(x_cat_real, x_t_rand_adj, rnd_logits,     t_rand)
    loss_perfect_adj = mn_adj.compute_loss(x_cat_real, x_t_rand_adj, perfect_logits, t_rand)
    print(f"  random  logits : {loss_rand_adj.item():.4f}")
    print(f"  perfect logits : {loss_perfect_adj.item():.2e}")

    print(f"\n[mn-diff] ✓ Sanity check complete.")
