# src/mlp.py

"""
Shared MLP backbone for TabDDPM (baseline and cardinality-adjusted).

Architecture (from Kotelnikov et al., 2023):
    Input:  [x_num_noisy, x_cat_onehot_noisy, t_emb, y_emb]
    Body:   ResBlock(Linear → SiLU → Dropout → Linear + skip) × N_layers
    Output: [ε_pred (n_num dims), x̂₀_cat_1 (K_1 dims), ..., x̂₀_cat_C (K_C dims)]

This module is IDENTICAL for the shared-schedule baseline and the
cardinality-adjusted variant.  Only the noise schedule changes —
the backbone never sees or needs the schedule directly.

Design notes:
    - Timestep conditioning via sinusoidal embedding + MLP projection,
      injected additively into each residual block.
    - Class label (y) conditioning via learned embedding, added to t_emb
      before injection.  For unconditional generation, pass y=0 and use
      n_classes=1 (single dummy class).
    - Categorical inputs arrive as concatenated one-hot vectors from the
      diffusion module (not integer indices).  Input dim for categoricals
      is sum(cardinalities), not n_cat.
    - Separate output heads: one linear layer for ε (numericals), one
      linear layer per categorical feature producing K_i logits.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple


# Timestep embedding ---------------------------------------


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for diffusion timesteps.

    Maps integer timestep t → ℝ^dim via:
        emb_{2i}   = sin(t / 10000^{2i/dim})
        emb_{2i+1} = cos(t / 10000^{2i/dim})

    Same formulation as Vaswani et al. (2017), repurposed for
    continuous-valued timestep indices in DDPM (Ho et al., 2020).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timesteps.
        Returns:
            (B, dim) sinusoidal embedding.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# Residual MLP block ---------------------------------------


class ResMLPBlock(nn.Module):
    """
    Pre-activation residual block with additive timestep conditioning.

        h = SiLU(Linear(x))
        h = h + proj(cond)          ← timestep + label conditioning
        h = Dropout(SiLU(Linear(h)))
        out = x + h                 ← skip connection

    Using SiLU (Swish) rather than ReLU — smoother gradients, standard
    in modern diffusion MLPs.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.linear1   = nn.Linear(hidden_dim, hidden_dim)
        self.linear2   = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.act       = nn.SiLU()
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, hidden_dim)
            cond: (B, cond_dim)  — combined timestep + label embedding
        Returns:
            (B, hidden_dim)
        """
        h = self.act(self.linear1(x))
        h = h + self.cond_proj(cond)
        h = self.dropout(self.act(self.linear2(h)))
        return x + h


# Full model ---------------------------------------


class TabDDPMMLP(nn.Module):
    """
    TabDDPM denoising network for mixed-type tabular data.

    Numericals:    predicts noise ε  (ε-parameterisation)
    Categoricals:  predicts clean x₀ logits (x₀-parameterisation)

    The model is class-conditional: it receives label y and can
    generate data conditioned on the target variable.

    Args:
        n_num:         number of numerical features  (6 for Adult)
        cardinalities: list of K_i per categorical feature
                       e.g. [8, 16, 7, 14, 6, 5, 2, 41] for Adult
        n_classes:     number of target classes (2 for Adult: <=50K / >50K)
        hidden_dim:    MLP hidden dimension (default 256)
        n_layers:      number of residual blocks (default 3)
        t_emb_dim:     timestep embedding dimension (default 128)
        dropout:       dropout rate (default 0.0)
    """

    def __init__(
        self,
        n_num: int,
        cardinalities: List[int],
        n_classes: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        t_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_num = n_num
        self.cardinalities = cardinalities
        self.n_cat = len(cardinalities)
        self.cat_input_dim = sum(cardinalities)

        # Conditioning
        self.t_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.y_embed = nn.Embedding(n_classes, t_emb_dim)

        # Input projection
        input_dim = n_num + self.cat_input_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual backbone
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden_dim, t_emb_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.num_head = nn.Linear(hidden_dim, n_num) if n_num > 0 else None
        self.cat_heads = nn.ModuleList([
            nn.Linear(hidden_dim, K_i) for K_i in cardinalities
        ])

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linears, zeros for output head biases only."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # Zero-init output head biases only — zeroing weights would collapse
        # gradients at step 0 since the Jacobian would be zero.
        if self.num_head is not None:
            nn.init.zeros_(self.num_head.bias)
        for head in self.cat_heads:
            nn.init.zeros_(head.bias)

    def forward(
        self,
        x_num_noisy: torch.Tensor,
        x_cat_onehot_noisy: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x_num_noisy:        (B, n_num)            noisy numerical features
            x_cat_onehot_noisy: (B, sum(K_i))         concatenated one-hot noisy categoricals
            t:                  (B,)                   integer timestep indices
            y:                  (B,)                   integer class labels

        Returns:
            eps_pred:   (B, n_num)          predicted noise for numericals
            cat_logits: list of C tensors   [(B, K_1), (B, K_2), ..., (B, K_C)]
                        predicted x₀ logits for each categorical feature
        """
        t_emb = self.t_embed(t)
        y_emb = self.y_embed(y)
        cond  = t_emb + y_emb

        if self.n_num > 0:
            x = torch.cat([x_num_noisy, x_cat_onehot_noisy], dim=1)
        else:
            x = x_cat_onehot_noisy

        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h, cond)

        eps_pred   = self.num_head(h) if self.num_head is not None else None
        cat_logits = [head(h) for head in self.cat_heads]

        return eps_pred, cat_logits


# Utilities ---------------------------------------


def indices_to_onehot(x_cat: torch.Tensor, cardinalities: List[int]) -> torch.Tensor:
    """
    Convert integer-encoded categoricals to concatenated one-hot vectors.

    Args:
        x_cat:         (B, C)  integer indices, column i ∈ [0, K_i)
        cardinalities: list of K_i per feature

    Returns:
        (B, sum(K_i))  concatenated one-hot vectors
    """
    parts = []
    for i, K_i in enumerate(cardinalities):
        one_hot = torch.zeros(x_cat.size(0), K_i, device=x_cat.device)
        one_hot.scatter_(1, x_cat[:, i:i+1].long(), 1.0)
        parts.append(one_hot)
    return torch.cat(parts, dim=1)


# Sanity check ---------------------------------------


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_utils import load_meta

    meta          = load_meta()
    n_num         = meta['n_num']
    cardinalities = meta['cardinalities']
    n_classes     = 2
    B             = 32

    model = TabDDPMMLP(
        n_num=n_num,
        cardinalities=cardinalities,
        n_classes=n_classes,
        hidden_dim=256,
        n_layers=3,
        t_emb_dim=128,
        dropout=0.0,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mlp] Total parameters: {n_params:,}")

    x_num    = torch.randn(B, n_num)
    x_cat_oh = torch.zeros(B, sum(cardinalities))
    offset   = 0
    for K_i in cardinalities:
        idx = torch.randint(0, K_i, (B,))
        x_cat_oh[torch.arange(B), offset + idx] = 1.0
        offset += K_i

    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, n_classes, (B,))

    eps_pred, cat_logits = model(x_num, x_cat_oh, t, y)

    print(f"[mlp] eps_pred shape:    {eps_pred.shape}")
    print(f"[mlp] cat_logits shapes: {[l.shape for l in cat_logits]}")

    x_cat_int = torch.stack([torch.randint(0, K, (B,)) for K in cardinalities], dim=1)
    x_cat_oh2 = indices_to_onehot(x_cat_int, cardinalities)
    print(f"[mlp] indices_to_onehot: {x_cat_int.shape} → {x_cat_oh2.shape}")

    print("[mlp] ✓ Sanity check passed.")
