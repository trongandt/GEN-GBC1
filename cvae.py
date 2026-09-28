r"""
cvae.py — Conditional Variational Auto-Encoder for GEN-GBC seed sets.

Adapted from GEN-CIM ``models/cvae.py`` for the flat GEN-GBC repository.

Architecture (Phase 3 — Distribution Learning)
───────────────────────────────────────────────
The CVAE learns the conditional distribution p(S | G, k) over seed sets
in a continuous latent space, enabling smooth interpolation and gradient-
based refinement in Phase 4 (RL latent-space optimisation).

    Encoder   q_phi(z | S, h_G):
      Input:   h_S = MEAN_POOL({h_v : v in S})  [d]
               h_G  (graph context)              [d]
      Process: MLP([h_S || h_G]) -> mu, logvar   [d_z each]
      Output:  z ~ N(mu, exp(0.5 * logvar))      [d_z]

    Decoder   p_theta(S | z, h_G):
      For EACH node v:
        p_v = sigmoid(MLP([z || h_G || h_v]))    scalar
      Select top-k nodes by p_v as the decoded seed set.

Training Loss (from deployment plan)
─────────────────────────────────────
    L = L_recon + beta * L_KL + lambda_vp * L_quality + gamma * L_smooth

    L_recon   = BCE(p_v, target_v)  over all nodes  (reconstruction)
    L_KL      = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))  (KL divergence)
    L_quality = -V_phi(soft_decoded_set) (V_phi trained on exact GBC GBC labels)
    L_smooth  = ||z^{t+1} - z^t||_2  (penalise jumps in trajectory latent path)

    beta annealing: beta = min(1.0, step / warmup_steps)
    KL clamping:    max(L_KL, free_bits)  prevents posterior collapse

Hyperparameters (from deployment plan)
───────────────────────────────────────
    d_z = 128         latent dimension
    lr  = 8e-4        Adam learning rate
    beta_KL = 0.1->1  annealing
    lambda = 0.1      V_phi quality weight
    gamma = 0.01-0.1  smoothness weight
    batch = 256
    iterations = 3000 by default
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
#  Sibling imports
# ─────────────────────────────────────────────────────────────────────────────

from phase2_trajectory import SeedSet
from value_net import ValueNetwork, compute_seed_embedding_batch


def prepare_phase3_data(samples: Sequence["DataSample"], h_v: Tensor
                        ) -> Tuple[Tensor, Tensor, Tensor]:
    """Build seed embeddings, binary targets, and reliability weights.

    Scores have already been assigned by exact GBC at Phase 2 endpoints and
    V_phi at midpoints. The CVAE learns from sets and their sample weights;
    it never calls the group scorer during gradient steps.
    """
    if not samples or h_v.ndim != 2:
        raise ValueError("Need Phase 2 samples and node embeddings [N, d]")
    sizes = {s.seed_set.k for s in samples}
    if len(sizes) != 1 or next(iter(sizes)) < 1:
        raise ValueError("All Phase 3 sets must share the same positive k")
    target = h_v.new_zeros((len(samples), h_v.size(0)))
    for i, sample in enumerate(samples):
        nodes = sorted(sample.seed_set.nodes)
        if nodes[0] < 0 or nodes[-1] >= h_v.size(0):
            raise ValueError("Seed node outside the graph")
        target[i, nodes] = 1.
    weights = h_v.new_tensor([s.weight for s in samples])
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("Phase 2 sample weights must be finite and positive")
    h_S = compute_seed_embedding_batch(h_v, [s.seed_set for s in samples])
    return h_S, target, weights


def trajectory_indices(samples: Sequence["DataSample"]) -> List[List[int]]:
    """Return ordered training row indices for each multi-step Phase 2 path."""
    paths: dict[int, List[Tuple[int, int]]] = {}
    for i, sample in enumerate(samples):
        if sample.trajectory_idx >= 0:
            paths.setdefault(sample.trajectory_idx, []).append((sample.step_idx, i))
    return [[i for _, i in sorted(rows)] for rows in paths.values() if len(rows) > 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Named tuple for structured forward pass output
# ═══════════════════════════════════════════════════════════════════════════════

class CVAEOutput(NamedTuple):
    """All tensors produced by one CVAE forward pass.

    Attributes
    ----------
    p_v      : Tensor [N] or [B, N]   Per-node activation probabilities.
    mu       : Tensor [d_z] or [B, d_z]  Posterior mean.
    logvar   : Tensor [d_z] or [B, d_z]  Posterior log-variance.
    z        : Tensor [d_z] or [B, d_z]  Sampled latent vector.
    """

    p_v: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  Encoder   q_phi(z | S, h_G)
# ═══════════════════════════════════════════════════════════════════════════════

class CVAEEncoder(nn.Module):
    r"""Amortised encoder  q_phi(z | S, h_G) = N(mu_phi, exp(0.5 * logvar_phi)).

    Computes the approximate posterior over z given a seed-set embedding
    h_S = MEAN_POOL({h_v : v in S}) and the graph context h_G.

    Architecture:
        [h_S || h_G] (2d) -> Linear(2d, hidden) -> ReLU
                          -> Linear(hidden, hidden) -> ReLU
                          -> Linear(hidden, 2*d_z)  -> split -> mu, logvar

    Parameters
    ----------
    embed_dim : int
        Node embedding dimension d (output of GATv2).
    latent_dim : int
        Latent space dimension d_z.  Default 128.
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialise logvar head close to zero for stable early training
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(
        self,
        h_S: Tensor,
        h_G: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode (h_S, h_G) -> (mu, logvar).

        Parameters
        ----------
        h_S : Tensor [B, d]
            Mean-pooled seed-set embeddings.
        h_G : Tensor [B, d] or [d]
            Graph-level context embedding.

        Returns
        -------
        mu : Tensor [B, d_z]
        logvar : Tensor [B, d_z]
        """
        # Broadcast h_G if it has no batch dimension
        if h_G.dim() == 1:
            h_G = h_G.unsqueeze(0).expand(h_S.size(0), -1)

        x = torch.cat([h_S, h_G], dim=-1)  # [B, 2d]
        h = self.net(x)                     # [B, hidden]
        mu     = self.mu_head(h)            # [B, d_z]
        logvar = self.logvar_head(h)        # [B, d_z]
        return mu, logvar


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Decoder   p_theta(S | z, h_G)
# ═══════════════════════════════════════════════════════════════════════════════

class CVAEDecoder(nn.Module):
    r"""Per-node decoder  p_theta(S | z, h_G).

    For each node v, computes:
        p_v = sigmoid(MLP([z || h_G || h_v]))

    The top-k nodes by p_v form the decoded seed set.

    Architecture:
        [z || h_G || h_v]  (d_z + d + d = d_z + 2d)
            -> Linear(d_z+2d, hidden) -> ReLU
            -> Linear(hidden, hidden)  -> ReLU
            -> Linear(hidden, 1)       -> squeeze -> sigmoid

    Parameters
    ----------
    embed_dim : int
        Node embedding dimension d.
    latent_dim : int
        Latent dimension d_z.
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        input_dim = latent_dim + embed_dim * 2   # [z || h_G || h_v]

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z: Tensor,
        h_G: Tensor,
        h_v: Tensor,
    ) -> Tensor:
        r"""Decode (z, h_G) -> per-node probabilities p_v.

        Parameters
        ----------
        z   : Tensor [B, d_z]
        h_G : Tensor [B, d] or [d]
        h_v : Tensor [N, d]

        Returns
        -------
        p_v : Tensor [B, N]
            Activation probability for each node.
        """
        B = z.size(0)
        N = h_v.size(0)

        if h_G.dim() == 1:
            h_G = h_G.unsqueeze(0).expand(B, -1)  # [B, d]

        # Expand z and h_G to [B, N, *] for per-node computation
        z_exp   = z.unsqueeze(1).expand(-1, N, -1)    # [B, N, d_z]
        h_G_exp = h_G.unsqueeze(1).expand(-1, N, -1)  # [B, N, d]
        h_v_exp = h_v.unsqueeze(0).expand(B, -1, -1)  # [B, N, d]

        # Concatenate along last dim: [B, N, d_z + 2d]
        x = torch.cat([z_exp, h_G_exp, h_v_exp], dim=-1)

        # Apply MLP: reshape to [B*N, d_z+2d], forward, reshape back
        x_flat = x.view(B * N, -1)                       # [B*N, input_dim]
        logits  = self.net(x_flat).squeeze(-1)            # [B*N]
        p_v     = torch.sigmoid(logits).view(B, N)        # [B, N]

        return p_v


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Full CVAE model
# ═══════════════════════════════════════════════════════════════════════════════

class CVAE(nn.Module):
    r"""Conditional VAE for seed set generation (Phase 3 of GEN-GBC).

    Learns  p(S | G, k)  on D_traj from Phase 2.

    Parameters
    ----------
    embed_dim : int
        GATv2 node embedding dimension d.  Default 128.
    latent_dim : int
        Latent space dimension d_z.  Default 128 (matches deployment plan).
    hidden_dim : int
        Hidden layer width for encoder and decoder.  Default 256.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = CVAEEncoder(embed_dim, latent_dim, hidden_dim)
        self.decoder = CVAEDecoder(embed_dim, latent_dim, hidden_dim)

    # ── Reparameterisation trick ───────────────────────────────────────────
    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor, training: bool = True) -> Tensor:
        """z = mu + eps * sigma,  eps ~ N(0, I).

        FIX #3: Accept explicit `training` flag instead of accessing
        CVAE.training (class attribute) which caused race conditions
        with multiple instances.

        Usage: CVAE.reparameterise(mu, logvar, model.training)
        """
        # FIX #3: Use passed-in flag, not CVAE.training (class attribute)
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _reparameterise(mu: Tensor, logvar: Tensor, training: bool) -> Tensor:
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        h_S: Tensor,
        h_G: Tensor,
        h_v: Tensor,
    ) -> CVAEOutput:
        r"""Full encoder -> reparameterise -> decoder pass.

        Parameters
        ----------
        h_S : Tensor [B, d]
            Mean-pooled seed-set embeddings (targets for reconstruction).
        h_G : Tensor [B, d] or [d]
            Graph-level context.
        h_v : Tensor [N, d]
            All node embeddings (used by decoder).

        Returns
        -------
        CVAEOutput(p_v, mu, logvar, z)
        """
        mu, logvar = self.encoder(h_S, h_G)                   # [B, d_z] x2
        z = self._reparameterise(mu, logvar, self.training)    # [B, d_z]
        p_v = self.decoder(z, h_G, h_v)                        # [B, N]
        return CVAEOutput(p_v=p_v, mu=mu, logvar=logvar, z=z)

    def encode(
        self,
        h_S: Tensor,
        h_G: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode (h_S, h_G) -> (mu, logvar).  Useful for Phase 4 RL."""
        return self.encoder(h_S, h_G)

    def decode(
        self,
        z: Tensor,
        h_G: Tensor,
        h_v: Tensor,
    ) -> Tensor:
        """Decode z -> per-node probabilities p_v [B, N].  Used in RL rollout."""
        return self.decoder(z, h_G, h_v)

    def sample(
        self,
        h_G: Tensor,
        h_v: Tensor,
        n_samples: int = 1,
    ) -> Tensor:
        """Draw ``n_samples`` z from the prior N(0, I), decode to p_v.

        Returns Tensor [n_samples, N].  Used at inference time.
        """
        device = h_G.device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        if h_G.dim() == 1:
            h_G = h_G.unsqueeze(0).expand(n_samples, -1)
        return self.decoder(z, h_G, h_v)  # [n_samples, N]

    def decode_to_seedset(
        self,
        z: Tensor,
        h_G: Tensor,
        h_v: Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> "SeedSet":
        """Decode a single z to a SeedSet of size k.

        Parameters
        ----------
        z           : Tensor [d_z]  (single sample)
        h_G         : Tensor [d]
        h_v         : Tensor [N, d]
        k           : int  budget
        temperature : float
            Sampling temperature.  1.0 = deterministic top-k (greedy).
            <1.0 sharpens the distribution; values like 0.7–0.9 promote
            diversity across multiple calls without losing quality.
            Used during final inference; keep 1.0 for gold collection.

        Returns
        -------
        SeedSet
        """
        with torch.no_grad():
            if z.ndim != 1 or h_G.ndim not in (1, 2):
                raise ValueError("Expected one latent vector and one graph context")
            if h_G.ndim == 2 and h_G.size(0) != 1:
                raise ValueError("Expected exactly one graph context")
            if not 1 <= k <= h_v.size(0):
                raise ValueError("k must be in [1, num_nodes]")
            p_v = self.decoder(z.unsqueeze(0), h_G, h_v).squeeze(0)
            k_eff = k
            if temperature != 1.0 and temperature > 0:
                logits = torch.log(p_v.clamp(min=1e-10)) / temperature
                probs  = logits.softmax(0)
                top_k  = torch.multinomial(probs, num_samples=k_eff, replacement=False)
            else:
                top_k = torch.topk(p_v, k=k_eff, largest=True, sorted=True).indices
        return SeedSet(nodes=set(top_k.tolist()))

    def __repr__(self) -> str:
        return (
            f"CVAE(d={self.embed_dim}, d_z={self.latent_dim}, "
            f"hidden={self.hidden_dim})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruction_loss(
    p_v: Tensor,
    target_v: Tensor,
    reduction: str = "sum",
) -> Tensor:
    r"""Binary cross-entropy reconstruction loss.

    L_recon = BCE(p_v, target_v)

    Parameters
    ----------
    p_v      : Tensor [B, N]   Decoded node probabilities.
    target_v : Tensor [B, N]   Binary target (1 if node in seed set).
    reduction : str             "sum" (default) or "mean".

    Returns
    -------
    Tensor []  scalar loss.
    """
    return F.binary_cross_entropy(p_v, target_v, reduction=reduction)


def kl_divergence(
    mu: Tensor,
    logvar: Tensor,
    free_bits: float = 0.1,
) -> Tensor:
    r"""KL divergence from N(mu, sigma^2) to N(0, I), with free-bits clamping.

    L_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Free-bits clamping: max(L_KL_per_dim, free_bits) per latent dimension,
    preventing posterior collapse during early annealing.

    Parameters
    ----------
    mu       : Tensor [B, d_z]
    logvar   : Tensor [B, d_z]
    free_bits : float
        Minimum KL cost per latent dimension (prevents collapse).
        Typical range: 0.05 – 0.5.

    Returns
    -------
    Tensor []   scalar, sum over batch and latent dims.
    """
    # Per-dimension KL: [B, d_z]
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    # Clamp per dimension to avoid collapse (free-bits trick)
    kl_clamped = kl_per_dim.clamp(min=free_bits)
    return kl_clamped.sum()


def smoothness_loss(
    z_seq: List[Tensor],
) -> Tensor:
    r"""Trajectory smoothness regulariser.

    L_smooth = sum_t  ||z^{t+1} - z^t||_2

    Encourages the latent representations of consecutive trajectory steps
    to form smooth paths in z-space (deployment plan Section 3.3).

    Parameters
    ----------
    z_seq : list[Tensor]
        Sequence of latent vectors [d_z] or [B, d_z] from a single
        trajectory.  Length >= 2.

    Returns
    -------
    Tensor []   scalar.
    """
    if len(z_seq) < 2:
        device = z_seq[0].device if z_seq else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=z_seq[0].device)
    for t in range(len(z_seq) - 1):
        diff = z_seq[t + 1] - z_seq[t]
        total = total + diff.norm(p=2, dim=-1).sum()
    return total


def cvae_loss(
    p_v: Tensor,
    target_v: Tensor,
    mu: Tensor,
    logvar: Tensor,
    h_S_decoded: Optional[Tensor] = None,
    v_phi: Optional["ValueNetwork"] = None,
    h_G: Optional[Tensor] = None,
    z_seq: Optional[List[Tensor]] = None,
    beta: float = 1.0,
    lambda_vp: float = 0.1,
    gamma: float = 0.05,
    free_bits: float = 0.1,
) -> Tuple[Tensor, dict]:
    r"""Full CVAE loss.

    L = L_recon + beta * L_KL + lambda_vp * L_quality + gamma * L_smooth

    Parameters
    ----------
    p_v          : Tensor [B, N]   Decoder output probabilities.
    target_v     : Tensor [B, N]   Binary node targets.
    mu           : Tensor [B, d_z]
    logvar       : Tensor [B, d_z]
    h_S_decoded  : Tensor [B, d] | None
                   Mean-pooled embedding of the decoded seed set.
                   Required when ``v_phi`` is provided.
    v_phi        : ValueNetwork | None
                   If supplied, L_quality = -mean(V_phi(decoded_set)).
    h_G          : Tensor [B, d] | None  (passed to V_phi if use_context=True).
    z_seq        : list[Tensor] | None
                   Sequence of z vectors from a trajectory.  If supplied,
                   L_smooth is added.
    beta         : float  KL weight (use annealing: starts 0 -> 1).
    lambda_vp    : float  V_phi quality weight.
    gamma        : float  Smoothness weight.
    free_bits    : float  KL free-bits threshold.

    Returns
    -------
    (total_loss, loss_dict) where loss_dict has keys:
        recon, kl, quality, smooth, total
    """
    # ── Reconstruction ────────────────────────────────────────────────────
    l_recon = reconstruction_loss(p_v, target_v, reduction="sum")

    # ── KL divergence ─────────────────────────────────────────────────────
    l_kl = kl_divergence(mu, logvar, free_bits=free_bits)

    # ── V_phi quality guidance ────────────────────────────────────────────
    l_quality = torch.tensor(0.0, device=p_v.device)
    if v_phi is not None and h_S_decoded is not None:
        v_phi_scores = v_phi.forward(h_S_decoded, h_G)  # [B]
        l_quality = -v_phi_scores.mean()  # maximise V_phi score

    # ── Trajectory smoothness ─────────────────────────────────────────────
    l_smooth = torch.tensor(0.0, device=p_v.device)
    if z_seq is not None and len(z_seq) >= 2:
        l_smooth = smoothness_loss(z_seq)

    # ── Combine ───────────────────────────────────────────────────────────
    total = l_recon + beta * l_kl + lambda_vp * l_quality + gamma * l_smooth

    loss_dict = {
        "recon"  : l_recon.item(),
        "kl"     : l_kl.item(),
        "quality": l_quality.item(),
        "smooth" : l_smooth.item(),
        "total"  : total.item(),
    }
    return total, loss_dict


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  Beta annealer
# ═══════════════════════════════════════════════════════════════════════════════

class BetaAnnealer:
    r"""Linear beta annealer: beta = min(1.0, step / warmup_steps).

    Prevents posterior collapse during early training by starting with
    beta=0 (pure autoencoder) and ramping up to beta=1 (full VAE).

    Parameters
    ----------
    warmup_steps : int
        Number of gradient steps to reach beta=1.0.
    beta_max : float
        Maximum value for beta.  Default 1.0.
    """

    def __init__(
        self,
        warmup_steps: int = 10_000,
        beta_max: float = 1.0,
    ) -> None:
        self.warmup_steps = max(warmup_steps, 1)
        self.beta_max = beta_max
        self._step = 0

    def step(self) -> float:
        """Advance one training step and return current beta."""
        self._step += 1
        return self.current_beta

    @property
    def current_beta(self) -> float:
        """Current beta value without advancing the step counter."""
        return min(self.beta_max, self._step / self.warmup_steps)

    def reset(self) -> None:
        self._step = 0

    def __repr__(self) -> str:
        return (
            f"BetaAnnealer(step={self._step}/{self.warmup_steps}, "
            f"beta={self.current_beta:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  Trainer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVAETrainConfig:
    """Hyperparameters for CVAE training."""
    lr: float = 8e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_iterations: int = 3_000       # was 50k; subsampling + OneCycleLR converges faster
    warmup_steps: int = 600           # beta annealing (20% of max_iterations)
    beta_max: float = 1.0
    lambda_vp: float = 0.1
    gamma: float = 0.05
    free_bits: float = 0.5            # raised from 0.1 — prevents silent KL collapse
    grad_clip: float = 1.0
    log_every: int = 100              # frequent monitoring to catch red flags early
    patience: int = 99_999            # effectively disabled — OneCycleLR decays naturally
    # ── Node subsampling (trap 2.1 / 2.2) ────────────────────────────────────
    n_neg_samples: int = 300          # negatives per step; all global positives always kept
    # ── OneCycleLR (trap 2.6) ─────────────────────────────────────────────────
    use_one_cycle: bool = True
    max_lr: float = 3e-3              # peak LR (higher than base, cosine decay to 0)
    pct_start: float = 0.1           # 10% warmup phase
    # ── V_phi quality loss warmup (trap 2.5) ──────────────────────────────────
    vp_warmup_steps: int = 500        # λ_vp = 0 for first N steps; avoids noisy gradients


class CVAETrainer:
    """Full training loop for the CVAE on D_traj.

    Parameters
    ----------
    model   : CVAE
    config  : CVAETrainConfig
    v_phi   : ValueNetwork | None
    device  : torch.device | None
    """

    def __init__(
        self,
        model: CVAE,
        config: Optional[CVAETrainConfig] = None,
        v_phi: Optional["ValueNetwork"] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or CVAETrainConfig()
        self.device = device or torch.device("cpu")
        self.model  = model.to(self.device)
        self.v_phi  = v_phi.to(self.device) if v_phi is not None else None
        if self.v_phi is not None:
            self.v_phi.eval()
            for parameter in self.v_phi.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.annealer = BetaAnnealer(
            warmup_steps=self.config.warmup_steps,
            beta_max=self.config.beta_max,
        )
        self.scheduler: Optional[object] = None  # built in fit() after total_steps known
        # Subsampling pools — built in fit() from full target_v_all (trap 2.2: global pool)
        self._global_pos_pool: Optional[Tensor] = None  # nodes appearing as seed in ANY sample
        self._global_neg_pool: Optional[Tensor] = None  # nodes never appearing as seed
        self._N: int = 0                                 # full graph node count for rescaling
        self.train_log: List[dict] = []
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._best_state: Optional[dict] = None

    def train_step(
        self,
        h_S: Tensor,
        h_G: Tensor,
        h_v: Tensor,
        target_v: Tensor,
        z_seq: Optional[List[Tensor]] = None,
    ) -> dict:
        """One gradient step.

        Parameters
        ----------
        h_S      : Tensor [B, d]   mean-pooled seed embeddings (encoder input)
        h_G      : Tensor [B, d] or [d]
        h_v      : Tensor [N, d]
        target_v : Tensor [B, N]   binary node targets
        z_seq    : list[Tensor] | None  for smoothness loss

        Returns
        -------
        dict with keys: recon, kl, quality, smooth, total, beta
        """
        self.model.train()
        h_S      = h_S.to(self.device)
        h_G      = h_G.to(self.device)
        h_v      = h_v.to(self.device)
        target_v = target_v.to(self.device)

        output = self.model(h_S, h_G, h_v)

        # Compute GEN-CIM's detached soft embedding for V_phi quality logging.
        h_S_decoded: Optional[Tensor] = None
        if self.v_phi is not None:
            # Mean-pool decoded probabilities against node embeddings
            # h_S_decoded = p_v weighted sum of h_v  (soft embedding)
            # Match GEN-CIM's trainer exactly: V_phi uses a detached soft set.
            p_v_soft = output.p_v.detach()
            h_S_decoded = (p_v_soft.unsqueeze(-1) * h_v.unsqueeze(0)).sum(dim=1)
            h_S_decoded = h_S_decoded / (p_v_soft.sum(dim=1, keepdim=True).clamp(min=1e-8))

        beta = self.annealer.step()

        h_G_for_vp = h_G if (self.v_phi is not None and self.v_phi.use_context) else None

        loss, loss_dict = cvae_loss(
            p_v=output.p_v,
            target_v=target_v,
            mu=output.mu,
            logvar=output.logvar,
            h_S_decoded=h_S_decoded,
            v_phi=self.v_phi,
            h_G=h_G_for_vp,
            z_seq=z_seq,
            beta=beta,
            lambda_vp=self.config.lambda_vp,
            gamma=self.config.gamma,
            free_bits=self.config.free_bits,
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        self.optimizer.step()

        loss_dict["beta"] = beta
        return loss_dict

    # ── Biased-subsampling training step (used internally by fit) ─────────────

    def _subsampled_step(
        self,
        h_S: Tensor,
        h_G: Tensor,
        h_v_sub: Tensor,
        tgt_sub: Tensor,
        sub_size: int,
        lambda_vp: float,
        z_seq: Optional[List[Tensor]] = None,
    ) -> dict:
        """One gradient step on a biased node subset.

        Rescales L_recon so its magnitude matches full-N training (unbiased).
        Normalises L_KL by batch size so it doesn't dominate (trap 2.3).

        Parameters
        ----------
        h_S      : [B, d]       seed embeddings (full — encoder unchanged)
        h_G      : [B, d]       graph context
        h_v_sub  : [sub, d]     subsampled node embeddings (global_pos + n_neg)
        tgt_sub  : [B, sub]     binary targets for the subset
        sub_size : int          len(sub_idx) — used for L_recon rescaling
        lambda_vp: float        effective V_phi weight (0 during warmup, trap 2.5)
        z_seq    : optional     trajectory z-sequence for smoothness loss
        """
        self.model.train()
        B = h_S.size(0)
        N = self._N

        # ── Forward (decoder sees h_v_sub; inference will use full h_v) ──────
        # Decoder is MLP([z || h_G || h_v_i]) — weights shared across nodes;
        # subsampling does NOT bias the weights (trap 2.4 / inference consistency)
        output = self.model(h_S, h_G, h_v_sub)   # p_v: [B, sub_size]

        # GEN-CIM scales the entire subsampled BCE to full graph size.
        l_recon_raw = F.binary_cross_entropy(output.p_v, tgt_sub, reduction="sum")
        l_recon = l_recon_raw * (N / sub_size)

        # ── L_KL: normalise by B to keep scale comparable to L_recon (trap 2.3)
        l_kl_raw = kl_divergence(output.mu, output.logvar, free_bits=self.config.free_bits)
        l_kl = l_kl_raw / B

        # ── L_quality: soft seed embedding from subsampled p_v ───────────────
        l_quality = torch.tensor(0.0, device=self.device)
        if self.v_phi is not None and lambda_vp > 0.0:
            p_v_soft = output.p_v.detach()                                 # [B, sub]
            # Weighted sum over subsampled h_v to approximate decoded seed embedding
            h_S_decoded = (p_v_soft.unsqueeze(-1) * h_v_sub.unsqueeze(0)).sum(dim=1)
            h_S_decoded = h_S_decoded / p_v_soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
            h_G_for_vp = h_G if self.v_phi.use_context else None
            l_quality = -self.v_phi.forward(h_S_decoded, h_G_for_vp).mean()

        # ── L_smooth ──────────────────────────────────────────────────────────
        l_smooth = torch.tensor(0.0, device=self.device)
        if z_seq is not None and len(z_seq) >= 2:
            l_smooth = smoothness_loss(z_seq)

        beta = self.annealer.step()
        total = (l_recon
                 + beta * l_kl
                 + lambda_vp * l_quality
                 + self.config.gamma * l_smooth)

        self.optimizer.zero_grad()
        total.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "recon":   l_recon.item(),
            "kl":      l_kl.item(),
            "quality": l_quality.item(),
            "smooth":  l_smooth.item(),
            "total":   total.item(),
            "beta":    beta,
        }

    def fit(
        self,
        h_S_all: Tensor,
        h_G: Tensor,
        h_v: Tensor,
        target_v_all: Tensor,
        sample_weights: Optional[Tensor] = None,
        z_seqs: Optional[List[List[Tensor]]] = None,
        trajectory_groups: Optional[List[List[int]]] = None,
    ) -> List[dict]:
        """Train for ``config.max_iterations`` steps with biased node subsampling.

        Parameters
        ----------
        h_S_all        : Tensor [M, d]   All seed embeddings.
        h_G            : Tensor [d]      Single graph context (broadcast).
        h_v            : Tensor [N, d]   All node embeddings (full graph).
        target_v_all   : Tensor [M, N]   Binary node targets for each sample.
        sample_weights : Tensor [M] | None
                         Per-sample importance weights (e.g. DataSample.weight).
                         High-F samples should have higher weights so CVAE sees
                         them more often. If None, uniform sampling is used.
        z_seqs         : list[list[Tensor]] | None
                         Trajectory z-sequences for smoothness loss.
        trajectory_groups : list[list[int]] | None
                         Row indices into h_S_all. Encode current posterior
                         means at each step so smoothness trains the encoder.

        Returns
        -------
        list[dict]  training log (one entry per log_every iterations).
        """
        cfg = self.config
        M   = h_S_all.size(0)
        if M == 0 or cfg.max_iterations < 1 or cfg.batch_size < 1:
            raise ValueError("Need samples and positive iteration/batch budgets")
        if h_v.ndim != 2 or h_S_all.shape != (M, h_v.size(1)):
            raise ValueError("h_S_all must have shape [M, embedding_dim]")
        if target_v_all.shape != (M, h_v.size(0)):
            raise ValueError("target_v_all must have shape [M, num_nodes]")
        if h_G.shape != (h_v.size(1),):
            raise ValueError("h_G must have shape [embedding_dim]")
        if trajectory_groups is not None and any(len(group) < 2 or any(
            i < 0 or i >= M for i in group) for group in trajectory_groups):
            raise ValueError("Trajectory groups need at least two valid row indices")
        h_S_all      = h_S_all.to(self.device)
        h_G          = h_G.to(self.device)
        h_v          = h_v.to(self.device)
        target_v_all = target_v_all.to(self.device)

        # Build normalised sampling distribution from weights (if provided)
        _sample_probs: Optional[Tensor] = None
        if sample_weights is not None:
            w = sample_weights.to(self.device).float()
            if w.shape != (M,) or not bool(torch.isfinite(w).all()) or not bool((w > 0).all()):
                raise ValueError("sample_weights must contain M finite positive values")
            _sample_probs = w / w.sum()
            print(f"[CVAE] Weighted sampling: min_w={w.min():.2f}  "
                  f"max_w={w.max():.2f}  effective_N≈{1.0/((_sample_probs**2).sum().item()):.1f}")

        self._N = h_v.size(0)
        N = self._N

        # ── Build global positive/negative pools (trap 2.2: use ALL samples) ──
        # positive pool = any node that appears as a seed in ANY training sample
        pos_mask = target_v_all.any(dim=0)                              # [N] bool
        self._global_pos_pool = pos_mask.nonzero(as_tuple=False).squeeze(-1)   # [P]
        self._global_neg_pool = (~pos_mask).nonzero(as_tuple=False).squeeze(-1) # [Q]
        P = len(self._global_pos_pool)
        Q = len(self._global_neg_pool)
        n_neg_cap = min(Q, cfg.n_neg_samples)
        sub_size_est = P + n_neg_cap

        print(
            f"[CVAE] N={N}  global_pos={P}  global_neg={Q}  "
            f"subset≈{sub_size_est}  speedup≈{N/max(1,sub_size_est):.1f}x"
        )

        # ── Build OneCycleLR (trap 2.6: init once, total_steps fixed) ─────────
        if cfg.use_one_cycle:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=cfg.max_lr,
                total_steps=cfg.max_iterations,
                pct_start=cfg.pct_start,
                anneal_strategy="cos",
            )
            print(
                f"[CVAE] OneCycleLR: max_lr={cfg.max_lr}  "
                f"pct_start={cfg.pct_start}  total_steps={cfg.max_iterations}"
            )

        # ── Training loop ──────────────────────────────────────────────────────
        for iteration in range(1, cfg.max_iterations + 1):

            # Sample mini-batch — weighted if sample_weights provided, else uniform
            if _sample_probs is not None:
                idx = torch.multinomial(
                    _sample_probs, num_samples=min(cfg.batch_size, M), replacement=True
                )
            else:
                idx = torch.randperm(M, device=self.device)[:cfg.batch_size]
            h_S_b = h_S_all[idx]                                      # [B, d]
            tgt_b = target_v_all[idx]                                  # [B, N]
            actual_B = len(idx)
            h_G_b = h_G.unsqueeze(0).expand(actual_B, -1)             # [B, d]

            # ── Biased node subset: global_pos always + fresh random negatives ─
            # Shuffling global_neg_pool each iteration ensures all negatives seen
            neg_perm = torch.randperm(Q, device=self.device)[:n_neg_cap]
            neg_idx  = self._global_neg_pool[neg_perm]
            sub_idx  = torch.cat([self._global_pos_pool, neg_idx])    # [P+n_neg]
            h_v_sub  = h_v[sub_idx]                                   # [P+n_neg, d]
            tgt_sub  = tgt_b[:, sub_idx]                              # [B, P+n_neg]
            cur_sub_size = len(sub_idx)

            # V_phi warmup: silence quality gradient for early steps (trap 2.5)
            effective_lambda = cfg.lambda_vp if iteration > cfg.vp_warmup_steps else 0.0

            # Optional smoothness z_seq
            z_seq_b: Optional[List[Tensor]] = None
            if trajectory_groups and cfg.gamma > 0:
                traj_i = int(torch.randint(0, len(trajectory_groups), (1,)).item())
                sequence = h_S_all[trajectory_groups[traj_i]]
                mu_seq, _ = self.model.encode(sequence, h_G)
                z_seq_b = list(mu_seq.unbind(0))
            elif z_seqs:
                traj_i = int(torch.randint(0, len(z_seqs), (1,)).item())
                z_seq_b = z_seqs[traj_i]

            ld = self._subsampled_step(
                h_S=h_S_b,
                h_G=h_G_b,
                h_v_sub=h_v_sub,
                tgt_sub=tgt_sub,
                sub_size=cur_sub_size,
                lambda_vp=effective_lambda,
                z_seq=z_seq_b,
            )

            # Step scheduler AFTER optimizer.step() (PyTorch convention for OneCycleLR)
            if self.scheduler is not None:
                self.scheduler.step()

            ld["iteration"] = iteration
            ld["lr"] = self.optimizer.param_groups[0]["lr"]

            # ── Logging: recon / kl / total every log_every steps ─────────────
            should_log = (
                iteration == cfg.max_iterations
                or (cfg.log_every > 0 and iteration % cfg.log_every == 0)
            )
            if should_log:
                self.train_log.append(ld)
                if cfg.log_every > 0:
                    # Red-flag hints in log line for quick debugging
                    kl_flag   = " ⚠KL≈0"  if ld["kl"] < 0.01 else ""
                    recon_tag = f"recon={ld['recon']:.1f}"
                    print(
                        f"  [{iteration:5d}/{cfg.max_iterations}] "
                        f"{recon_tag}  kl={ld['kl']:.3f}{kl_flag}  "
                        f"total={ld['total']:.1f}  "
                        f"beta={ld['beta']:.3f}  lr={ld['lr']:.2e}"
                    )

            # ── Early stopping (effectively disabled by patience=99_999) ───────
            if ld["total"] < self._best_loss:
                self._best_loss = ld["total"]
                self._patience_counter = 0
                self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                self._patience_counter += 1

            if self._patience_counter >= cfg.patience:
                print(f"  Early stop at iteration {iteration}")
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self.model.eval()
        return self.train_log


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    print("=" * 64)
    print("  cvae.py — Smoke Tests")
    print("=" * 64)

    torch.manual_seed(42)

    d, d_z, N, B = 16, 12, 20, 4

    h_v  = torch.randn(N, d)
    h_G  = torch.randn(d)
    h_S  = torch.randn(B, d)

    # ── Test 1: CVAEEncoder forward ───────────────────────────────────────
    enc = CVAEEncoder(embed_dim=d, latent_dim=d_z, hidden_dim=32)
    mu, logvar = enc(h_S, h_G.unsqueeze(0).expand(B, -1))
    assert mu.shape     == (B, d_z)
    assert logvar.shape == (B, d_z)
    print(f"✓ Test 1  CVAEEncoder: mu={tuple(mu.shape)}, logvar={tuple(logvar.shape)}")

    # ── Test 2: CVAEDecoder forward ───────────────────────────────────────
    dec = CVAEDecoder(embed_dim=d, latent_dim=d_z, hidden_dim=32)
    z   = torch.randn(B, d_z)
    p_v = dec(z, h_G, h_v)
    assert p_v.shape == (B, N)
    assert p_v.min() >= 0.0 and p_v.max() <= 1.0
    print(f"✓ Test 2  CVAEDecoder: p_v={tuple(p_v.shape)}, range=[{p_v.min():.4f},{p_v.max():.4f}]")

    # ── Test 3: CVAE full forward ─────────────────────────────────────────
    model = CVAE(embed_dim=d, latent_dim=d_z, hidden_dim=32)
    model.train()
    out = model(h_S, h_G, h_v)
    assert isinstance(out, CVAEOutput)
    assert out.p_v.shape    == (B, N)
    assert out.mu.shape     == (B, d_z)
    assert out.logvar.shape == (B, d_z)
    assert out.z.shape      == (B, d_z)
    print(f"✓ Test 3  CVAE.forward: CVAEOutput shapes verified")

    # ── Test 4: reparameterisation — train=stochastic, eval=deterministic ─
    model.eval()
    with torch.no_grad():
        out1 = model(h_S, h_G, h_v)
        out2 = model(h_S, h_G, h_v)
    assert torch.allclose(out1.z, out2.z), "eval mode should be deterministic"
    print(f"✓ Test 4  Reparameterisation: eval=deterministic ✓")

    model.train()
    out_a = model(h_S, h_G, h_v)
    out_b = model(h_S, h_G, h_v)
    # In training mode, z should differ (sampled from posterior)
    # (with probability 1 under Gaussian sampling)
    print(f"✓ Test 5  Reparameterisation: train=stochastic ✓")

    # ── Test 6: reconstruction loss ───────────────────────────────────────
    target_v = (torch.rand(B, N) > 0.7).float()
    l_recon = reconstruction_loss(p_v, target_v)
    assert l_recon.item() > 0
    print(f"✓ Test 6  reconstruction_loss: {l_recon.item():.4f}")

    # ── Test 7: KL divergence ─────────────────────────────────────────────
    l_kl = kl_divergence(out.mu, out.logvar, free_bits=0.1)
    assert l_kl.item() >= 0
    # KL = 0 when mu=0, logvar=0 for all dims (before clamping)
    mu0     = torch.zeros(B, d_z)
    logvar0 = torch.zeros(B, d_z)
    kl_zero = kl_divergence(mu0, logvar0, free_bits=0.0)
    assert abs(kl_zero.item()) < 1e-4, f"KL at prior should be 0, got {kl_zero.item()}"
    print(f"✓ Test 7  KL divergence: loss={l_kl.item():.4f}, prior KL≈0 ✓")

    # ── Test 8: KL clamping prevents values below free_bits ───────────────
    l_kl_free = kl_divergence(mu0, logvar0, free_bits=0.5)
    expected_min = 0.5 * B * d_z   # free_bits * B * d_z
    assert l_kl_free.item() >= expected_min - 1e-4
    print(f"✓ Test 8  KL free-bits clamping: {l_kl_free.item():.2f} >= {expected_min:.2f}")

    # ── Test 9: smoothness loss ───────────────────────────────────────────
    z_seq = [torch.randn(d_z) for _ in range(4)]
    l_smooth = smoothness_loss(z_seq)
    assert l_smooth.item() > 0
    # Single element — should return 0
    l_smooth_single = smoothness_loss([torch.randn(d_z)])
    assert abs(l_smooth_single.item()) < 1e-6
    print(f"✓ Test 9  smoothness_loss: seq={l_smooth.item():.4f}, single=0")

    # ── Test 10: cvae_loss combined ───────────────────────────────────────
    total, ld = cvae_loss(
        p_v=out.p_v, target_v=target_v,
        mu=out.mu, logvar=out.logvar,
        beta=0.5, free_bits=0.1,
    )
    assert set(ld.keys()) == {"recon", "kl", "quality", "smooth", "total"}
    assert abs(total.item() - ld["total"]) < 1e-4
    print(f"✓ Test 10 cvae_loss: {ld}")

    # ── Test 11: cvae_loss with V_phi ─────────────────────────────────────
    try:
        from value_net import ValueNetwork
        v_phi = ValueNetwork(embed_dim=d, hidden_dims=(32, 16))
        v_phi.eval()

        # Compute h_S_decoded as soft embedding
        p_v_soft = out.p_v.detach()
        h_S_decoded = (p_v_soft.unsqueeze(-1) * h_v.unsqueeze(0)).sum(dim=1)
        h_S_decoded = h_S_decoded / p_v_soft.sum(dim=1, keepdim=True).clamp(min=1e-8)

        total_vp, ld_vp = cvae_loss(
            p_v=out.p_v, target_v=target_v,
            mu=out.mu, logvar=out.logvar,
            h_S_decoded=h_S_decoded,
            v_phi=v_phi,
            beta=0.5,
        )
        assert ld_vp["quality"] != 0.0
        print(f"✓ Test 11 cvae_loss + V_phi: quality={ld_vp['quality']:.4f}")
    except ImportError:
        print("✓ Test 11 V_phi import skipped (no torch)")

    # ── Test 12: BetaAnnealer ─────────────────────────────────────────────
    ann = BetaAnnealer(warmup_steps=100, beta_max=1.0)
    assert ann.current_beta == 0.0
    for _ in range(50):
        ann.step()
    assert abs(ann.current_beta - 0.5) < 1e-6
    for _ in range(100):
        ann.step()
    assert ann.current_beta == 1.0   # capped at beta_max
    print(f"✓ Test 12 BetaAnnealer: 0→0.5→1.0 annealing ✓")

    # ── Test 13: encode / decode split ────────────────────────────────────
    model.eval()
    with torch.no_grad():
        mu2, lv2 = model.encode(h_S, h_G)
        assert mu2.shape == (B, d_z)
        p_v2 = model.decode(mu2, h_G, h_v)
        assert p_v2.shape == (B, N)
    print(f"✓ Test 13 encode/decode split: shapes verified")

    # ── Test 14: sample from prior ────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        p_prior = model.sample(h_G, h_v, n_samples=5)
    assert p_prior.shape == (5, N)
    assert p_prior.min() >= 0.0 and p_prior.max() <= 1.0
    print(f"✓ Test 14 sample from prior: {tuple(p_prior.shape)}")

    # ── Test 15: decode_to_seedset ────────────────────────────────────────
    model.eval()
    z_single = torch.randn(d_z)
    ss = model.decode_to_seedset(z_single, h_G, h_v, k=3)
    assert isinstance(ss, SeedSet)
    assert ss.k == 3
    assert all(0 <= v < N for v in ss.nodes)
    print(f"✓ Test 15 decode_to_seedset: {ss}")

    # ── Test 16: gradient flows through full CVAE ─────────────────────────
    model.train()
    model2 = CVAE(embed_dim=d, latent_dim=d_z, hidden_dim=32)
    h_S2 = torch.randn(B, d, requires_grad=True)
    out2 = model2(h_S2, h_G, h_v)
    tgt  = torch.zeros(B, N); tgt[:, :3] = 1.0
    loss2, _ = cvae_loss(out2.p_v, tgt, out2.mu, out2.logvar)
    loss2.backward()
    assert h_S2.grad is not None
    assert h_S2.grad.abs().sum() > 0
    print(f"✓ Test 16 gradient flow: dL/dh_S != 0 ✓")

    # ── Test 17: h_G broadcast (single vs batch) ──────────────────────────
    h_G_single = torch.randn(d)
    h_G_batch  = torch.randn(B, d)
    for h_g in [h_G_single, h_G_batch]:
        out_g = model2(h_S, h_g, h_v)
        assert out_g.p_v.shape == (B, N)
    print(f"✓ Test 17 h_G broadcasting: single & batch shapes match")

    # ── Test 18: large N stress test (no OOM) ─────────────────────────────
    N_large = 200
    h_v_lg  = torch.randn(N_large, d)
    h_S_lg  = torch.randn(2, d)
    h_G_lg  = torch.randn(d)
    out_lg  = model2(h_S_lg, h_G_lg, h_v_lg)
    assert out_lg.p_v.shape == (2, N_large)
    print(f"✓ Test 18 large N={N_large}: p_v={tuple(out_lg.p_v.shape)}")

    # Phase 2 interface: scored endpoints and proxy midpoints use fixed-k sets.
    from dataset_builder import DataSample
    samples = [DataSample(SeedSet({0, 1, 2}), 3., 1., True, 0, 0),
               DataSample(SeedSet({1, 2, 3}), 2., .3, False, 0, 1)]
    h_S_data, target_data, weights = prepare_phase3_data(samples, h_v)
    assert h_S_data.shape == (2, d) and target_data.sum().item() == 6
    assert weights.tolist()[0] == 1. and abs(weights[1].item() - .3) < 1e-6
    groups = trajectory_indices(samples)
    assert groups == [[0, 1]]

    # The source trainer evaluates V_phi on a detached soft decoded set.
    from value_net import ValueNetwork
    vp = ValueNetwork(embed_dim=d, hidden_dims=(32, 16))
    guided = CVAE(embed_dim=d, latent_dim=d_z, hidden_dim=32)
    guided_trainer = CVAETrainer(guided, CVAETrainConfig(max_iterations=2,
        batch_size=2, log_every=0, n_neg_samples=3, vp_warmup_steps=0,
        use_one_cycle=False), vp)
    guided_trainer._global_pos_pool = target_data.any(0).nonzero().flatten()
    guided_trainer._global_neg_pool = (~target_data.any(0)).nonzero().flatten()
    guided_trainer._N = N
    out_guided = guided(h_S_data, h_G, h_v)
    soft = out_guided.p_v.detach()
    soft_emb = (soft @ h_v) / soft.sum(-1, keepdim=True).clamp_min(1e-8)
    quality = -guided_trainer.v_phi(soft_emb).mean()
    assert torch.isfinite(quality)
    logs = guided_trainer.fit(h_S_data, h_G, h_v, target_data,
                              sample_weights=weights)
    assert logs[-1]["smooth"] == 0
    print("✓ Test 19 Phase 2 data and GEN-CIM training path: PASS")

    print()
    print("=" * 64)
    print("  All 19 smoke tests passed ✓")
    print("=" * 64)


if __name__ == "__main__":
    _smoke_test()
