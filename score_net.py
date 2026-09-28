r"""
score_net.py — Score-based latent guidance for GEN-GBC Phase 3/4.

Role in GEN-GBC
───────────────
CVAE encodes Phase 2 seed sets. Endpoint scores come from exact_gbc.cpp
and midpoint scores from V_phi. ScoreNet learns a latent density
weighted by those scores and the reliability of each sample.

Core idea
─────────
Alongside the value-network direction, learn a latent density gradient:

    s_θ(z) ≈ ∇_z log p_good(z)

where  p_good(z) is the score-weighted empirical distribution of encoded
Phase 2 seed sets, smoothed by Gaussian noise.

This is the score of the *distribution* of latent codes that decode to
good seed sets.  Unlike ∇F, the score field is:
It can be trained offline and evaluated without another exact GBC call.

Hybrid reward helper retained from GEN-CIM's REINFORCE path
────────────────────────────────────────────────────────────
    Δz_hybrid = α · ∇F(z)         (value direction, from V_phi / PropagationNet)
              + (1-α) · s_θ(z)    (distribution direction, from ScoreNet)

    α ∈ [0.2, 0.65] — available if REINFORCE is added later. The active
    CEM path uses ScoreNet for Langevin initialization instead.

When ∇F is unreliable (early training, non-submodular region), α is small
and the score field provides navigation.  As training matures, α increases
to let the value gradient dominate.

Training (Importance-Weighted Denoising Score Matching)
────────────────────────────────────────────────────────
Objective: learn s_θ(z, σ) = ∇_z log p_σ(z) at multiple noise levels

    L_DSM = E_{z ~ w(z)} E_{σ,ε} [ σ² · ||s_θ(z+σε, σ) + ε/σ||² ]

where   w(z) = softmax(β · F_normalized(z))
        ε ~ N(0, I)  (perturbation noise)
        σ ~ U(0.01, 1.0)

The σ² factor balances denoising loss across noise levels.

Langevin Dynamics for CEM warm-start
──────────────────────────────────────
Instead of initialising CEM from z ~ N(0, I) (random), we use
Annealed Langevin Dynamics guided by s_θ across the trained noise range:

    z_{t+1} = z_t + ε_step · s_θ(z_t, σ_t) + √(2ε_step) · ξ_t

This biases the initial population toward the learned latent distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from phase2_trajectory import SeedSet
from value_net import compute_seed_embedding_batch


def encode_phase2_samples(samples: Sequence["DataSample"], cvae: "CVAE",
                          h_v: Tensor, h_G: Tensor) -> List[dict]:
    """Encode Phase 2 sets; retain exact GBC endpoint and proxy midpoint weights.

    ScoreNet never evaluates a group itself. ``sample.score`` is the exact GBC
    score at verified endpoints and V_phi's estimate at midpoints.
    """
    if not samples:
        raise ValueError("Phase 2 supplied no training samples")
    if any(not math.isfinite(float(s.score)) or s.weight <= 0 for s in samples):
        raise ValueError("Phase 2 scores and weights must be finite and positive")
    if any(s.seed_set.k < 1 or any(v < 0 or v >= h_v.size(0)
                                  for v in s.seed_set.nodes) for s in samples):
        raise ValueError("Phase 2 samples contain invalid seed sets")
    device = next(cvae.parameters()).device
    cvae.eval()
    with torch.no_grad():
        h_S = compute_seed_embedding_batch(h_v.to(device),
                                           [s.seed_set for s in samples])
        mu, _ = cvae.encode(h_S, h_G.to(device))
    return [{"z": z.detach(), "F": float(s.score)}
            for z, s in zip(mu, samples)]


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  Score Network
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreNet(nn.Module):
    r"""Noise-conditional score network  s_θ(z, σ) ≈ ∇_z log p_σ(z).

    Architecture (NCSN-style):
        [z || embed(σ)]  →  Linear(d_z + d_σ, hidden)  →  SiLU
                         →  Linear(hidden, hidden)       →  SiLU  (× n_layers-2)
                         →  Linear(hidden, d_z)

    σ is encoded as a scalar appended to z, as in GEN-CIM.

    Parameters
    ----------
    latent_dim : int    d_z — must match CVAE latent_dim (default 128).
    hidden_dim : int    Hidden layer width (default 256).
    n_layers   : int    Total depth (default 4, min 2).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        n_layers:   int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input: [z (d_z) || σ (1)]
        in_dim = latent_dim + 1

        layers: list[nn.Module] = []
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else latent_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small output scale for stability
        last = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, z: Tensor, sigma: Tensor) -> Tensor:
        r"""Compute score s_θ(z, σ).

        Parameters
        ----------
        z     : Tensor [B, d_z]  or  [d_z]   Latent vectors.
        sigma : Tensor [B]       or  scalar   Noise level.

        Returns
        -------
        score : Tensor, same shape as z
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        B = z.shape[0]

        # Broadcast sigma to [B, 1]
        if sigma.dim() == 0:
            sigma = sigma.expand(B).unsqueeze(-1)
        elif sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        # sigma: [B, 1]

        inp = torch.cat([z, sigma], dim=-1)   # [B, d_z+1]
        out = self.net(inp)                    # [B, d_z]

        return out.squeeze(0) if squeeze else out

    def score_at_default_sigma(
        self, z: Tensor, sigma: float = 0.1
    ) -> Tensor:
        """Convenience: score at a single fixed sigma (for hybrid gradient)."""
        sig = torch.tensor(sigma, dtype=z.dtype, device=z.device)
        return self.forward(z, sig)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  ScoreNet Training Config + Trainer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoreNetConfig:
    """Hyperparameters for ScoreNet training.

    Attributes
    ----------
    beta          : float   Temperature for importance weights w ∝ exp(β·F_norm).
                            Higher β → focuses more on top-F trajectories.
    n_epochs      : int     Training epochs.
    batch_size    : int     Mini-batch size (sampled with importance weights).
    lr            : float   Adam learning rate.
    sigma_min/max : float   Range of noise levels used for DSM.
    n_sigma_levels: int     GEN-CIM configuration (10 by default).
    grad_clip     : float   Gradient clipping norm.
    log_every     : int     Print interval (0 = silent).
    """
    beta:           float = 3.0
    n_epochs:       int   = 300
    batch_size:     int   = 256
    lr:             float = 3e-4
    sigma_min:      float = 0.01
    sigma_max:      float = 1.0
    n_sigma_levels: int   = 10
    grad_clip:      float = 1.0
    log_every:      int   = 50


class ScoreNetTrainer:
    r"""Trains ScoreNet offline from D_traj using importance-weighted DSM.

    Usage
    -----
    After Phase 2 produces D_traj = [{'z': Tensor, 'F': float}, ...]:

        trainer = ScoreNetTrainer(
            score_net = ScoreNet(latent_dim=128),
            d_traj    = d_traj,
            config    = ScoreNetConfig(),
            device    = device,
        )
        trainer.fit()
        score_net = trainer.score_net   # trained, ready for Phase 4

    Parameters
    ----------
    score_net : ScoreNet
    d_traj    : list of dicts with keys 'z' (Tensor [d_z]) and 'F' (float).
    config    : ScoreNetConfig
    device    : torch.device | None
    """

    def __init__(
        self,
        score_net: ScoreNet,
        d_traj:    List[dict],
        config:    Optional[ScoreNetConfig] = None,
        device:    Optional[torch.device]   = None,
    ) -> None:
        self.config    = config or ScoreNetConfig()
        self.device    = device or torch.device("cpu")
        self.score_net = score_net.to(self.device)
        if not d_traj or self.config.n_epochs < 1 or self.config.batch_size < 1:
            raise ValueError("Need latent samples and positive epoch/batch budgets")
        if (self.config.sigma_min <= 0 or self.config.sigma_max < self.config.sigma_min
                or self.config.n_sigma_levels < 1):
            raise ValueError("Invalid ScoreNet noise range")
        if any(not math.isfinite(float(item["F"])) for item in d_traj):
            raise ValueError("Scores must be finite")

        # ── Build tensors from D_traj ──────────────────────────────────────
        self.zs = torch.stack(
            [item["z"].detach() for item in d_traj]
        ).to(self.device)                                  # [N, d_z]
        if self.zs.shape != (len(d_traj), self.score_net.latent_dim) or not bool(
            torch.isfinite(self.zs).all()
        ):
            raise ValueError("Latent codes must be finite vectors of model.latent_dim")

        Fs = torch.tensor(
            [float(item["F"]) for item in d_traj],
            dtype=torch.float32, device=self.device
        )                                                  # [N]

        # Importance weights: softmax(β · F_normalized)
        # GEN-CIM uses the sample standard deviation; guard the one-sample
        # case so a tiny D_traj still has a finite uniform weight.
        F_std           = Fs.std() if len(d_traj) > 1 else Fs.new_tensor(0.)
        F_norm          = (Fs - Fs.mean()) / (F_std + 1e-8)
        self.log_w      = self.config.beta * F_norm
        self.weights    = torch.softmax(self.log_w, dim=0) # [N] normalised
        self.N          = len(d_traj)

        self.optimizer  = torch.optim.Adam(
            self.score_net.parameters(), lr=self.config.lr
        )
        self.train_log: List[dict] = []

    def _dsm_loss_batch(self, z_batch: Tensor) -> Tensor:
        r"""Importance-weighted denoising score matching loss.

        L = mean_over_batch [ σ² · ||s_θ(z+σε, σ) − (−ε/σ)||² ]

        The term (−ε/σ) is the target score:
            ∇_{z_noisy} log p_σ(z_noisy | z) = −(z_noisy − z) / σ² = −ε/σ

        Multiplying by σ² (NCSN weighting) equalises gradient magnitudes
        across noise levels.
        """
        cfg = self.config
        B   = z_batch.shape[0]

        # GEN-CIM draws an independent uniform noise level for each latent.
        sigma = torch.empty(B, dtype=z_batch.dtype, device=self.device).uniform_(
            cfg.sigma_min, cfg.sigma_max)                  # [B]

        # Perturb z
        noise   = torch.randn_like(z_batch)                # [B, d_z]
        z_noisy = z_batch + sigma.unsqueeze(-1) * noise    # [B, d_z]

        # Target score: ∇ log p(z_noisy | z) = −noise / sigma
        target = -noise / sigma.unsqueeze(-1)              # [B, d_z]

        # Predict
        s_pred = self.score_net(z_noisy, sigma)            # [B, d_z]

        # NCSN-weighted MSE: σ² · ||pred - target||²
        loss = (sigma.unsqueeze(-1) ** 2 * (s_pred - target) ** 2).mean()
        return loss

    def fit(self) -> List[dict]:
        """Train ScoreNet for config.n_epochs epochs.

        Returns
        -------
        list[dict]  training log.
        """
        cfg = self.config
        self.score_net.train()

        for epoch in range(1, cfg.n_epochs + 1):
            # Importance-weighted sampling
            idx = torch.multinomial(
                self.weights,
                num_samples=min(cfg.batch_size, self.N),
                replacement=True,
            )
            z_batch = self.zs[idx]                         # [B, d_z]

            loss = self._dsm_loss_batch(z_batch)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.score_net.parameters(), cfg.grad_clip
            )
            self.optimizer.step()

            if epoch == cfg.n_epochs or (cfg.log_every > 0 and epoch % cfg.log_every == 0):
                entry = {"epoch": epoch, "dsm_loss": loss.item()}
                self.train_log.append(entry)
                print(
                    f"  [ScoreNet epoch {epoch:4d}/{cfg.n_epochs}] "
                    f"dsm_loss={loss.item():.5f}"
                )

        self.score_net.eval()
        return self.train_log


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Langevin Dynamics — CEM warm-start population
# ═══════════════════════════════════════════════════════════════════════════════

def langevin_sample(
    score_net:   ScoreNet,
    n_samples:   int,
    d_z:         int,
    n_steps:     int   = 60,
    step_size:   float = 0.01,
    sigma_start: float = 1.0,
    sigma_end:   float = 0.05,
    noise_scale: float = 1.0,
    device:      Optional[torch.device] = None,
    seed:        Optional[int]          = None,
) -> Tensor:
    r"""Annealed Langevin dynamics used for GEN-CIM's CEM warm-start.

    Algorithm (Langevin):
        Linearly anneal sigma from 1.0 to 0.05 by default.
        At each step t:
            z_{t+1} = z_t + step_size · s_θ(z_t, σ_t)
                            + √(2 · step_size) · noise_scale · ξ_t

    Parameters
    ----------
    score_net   : ScoreNet   Trained score model.
    n_samples   : int        Number of parallel chains.
    d_z         : int        Latent dimension.
    n_steps     : int        Number of Langevin steps (default 60).
    step_size   : float      Step size ε (default 0.01).
    sigma_start : float      Initial noise level (default 1.0).
    sigma_end   : float      Final noise level (default 0.05).
    noise_scale : float      Multiplier on Langevin noise (default 1.0).
    device      : torch.device | None
    seed        : int | None

    Returns
    -------
    z : Tensor [n_samples, d_z]
    """
    if device is None:
        device = next(score_net.parameters()).device

    if n_samples < 1 or d_z != score_net.latent_dim or n_steps < 1 or step_size <= 0:
        raise ValueError("Invalid Langevin population, latent dimension or step")
    if sigma_start <= 0 or sigma_end <= 0 or sigma_start < sigma_end:
        raise ValueError("Need sigma_start >= sigma_end > 0")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    score_net.eval()

    z = torch.randn(n_samples, d_z, device=device, generator=generator)
    sigmas = torch.linspace(sigma_start, sigma_end, n_steps, device=device)

    with torch.no_grad():
        for sigma in sigmas:
            sigma_batch = sigma.expand(n_samples)
            score = score_net(z, sigma_batch)
            noise = torch.randn(z.shape, dtype=z.dtype, device=device,
                                generator=generator)
            z = (
                z
                + step_size * score
                + math.sqrt(2 * step_size) * noise_scale * noise
            )

    return z.detach()


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Hybrid Reward Helper
# ═══════════════════════════════════════════════════════════════════════════════

def score_alignment_reward(
    z:         Tensor,
    z_prime:   Tensor,
    score_net: ScoreNet,
    sigma:     float = 0.1,
) -> float:
    r"""Dense reward from score alignment: r_score = cosine(s_θ(z,σ), Δz).

    R_hybrid = α · R_vphi  +  (1-α) · r_score
    """
    score_net.eval()
    with torch.no_grad():
        sig   = torch.tensor(sigma, dtype=z.dtype, device=z.device)
        score = score_net(z.unsqueeze(0), sig.unsqueeze(0)).squeeze(0)
        delta = z_prime - z
        align = F.cosine_similarity(
            score.unsqueeze(0), delta.unsqueeze(0), dim=-1
        ).item()
    return align


def adaptive_alpha(
    epsilon:    float,
    f_current:  float,
    f_prev:     float,
    alpha_min:  float = 0.2,
    alpha_max:  float = 0.65,
) -> float:
    r"""Compute hybrid weight α for V_phi vs score guidance.

    α = weight on V_phi reward;  (1-α) = weight on score-alignment reward.
    """
    improving = (f_current - f_prev) > 1e-3

    if epsilon > 0.4:
        return alpha_min

    elif epsilon > 0.1:
        return 0.40 if improving else 0.30

    else:
        return alpha_max if improving else 0.50


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    print("=" * 64)
    print("  score_net.py — Smoke Tests")
    print("=" * 64)

    torch.manual_seed(0)
    d_z = 16
    B   = 8
    N   = 50

    net   = ScoreNet(latent_dim=d_z, hidden_dim=32, n_layers=3)
    z     = torch.randn(B, d_z)
    sigma = torch.rand(B) * 0.9 + 0.1
    out   = net(z, sigma)
    assert out.shape == (B, d_z)
    print(f"✓ Test 1  forward (batch):     shape={tuple(out.shape)}")

    z1 = torch.randn(d_z)
    s1 = torch.tensor(0.1)
    o1 = net(z1, s1)
    assert o1.shape == (d_z,)
    print(f"✓ Test 2  forward (single):    shape={tuple(o1.shape)}")

    d_traj = [
        {"z": torch.randn(d_z), "F": float(torch.rand(1).item() * 0.5 + 0.5)}
        for _ in range(N)
    ]
    trainer = ScoreNetTrainer(
        score_net = ScoreNet(latent_dim=d_z, hidden_dim=32),
        d_traj    = d_traj,
        config    = ScoreNetConfig(n_epochs=20, batch_size=16, log_every=10),
    )
    log = trainer.fit()
    assert len(log) > 0
    print(f"✓ Test 3  training:            final_loss={log[-1]['dsm_loss']:.5f}")

    net2    = trainer.score_net
    samples = langevin_sample(net2, n_samples=10, d_z=d_z, n_steps=5, seed=42)
    assert samples.shape == (10, d_z)
    print(f"✓ Test 4  langevin_sample:     shape={tuple(samples.shape)}")

    z_a = torch.randn(d_z)
    z_b = torch.randn(d_z)
    r   = score_alignment_reward(z_a, z_b, net2)
    assert -1.0 <= r <= 1.0
    print(f"✓ Test 5  score_alignment_reward: r={r:.4f}")

    a1 = adaptive_alpha(0.5, 0.6, 0.5)
    a2 = adaptive_alpha(0.2, 0.7, 0.65)
    a3 = adaptive_alpha(0.05, 0.8, 0.79)
    assert a1 < a2 < a3
    print(f"✓ Test 6  adaptive_alpha:      early={a1:.2f} mid={a2:.2f} late={a3:.2f}")

    # GEN-GBC integration: carry Phase 2 exact GBC/proxy labels.
    from cvae import CVAE
    from dataset_builder import DataSample
    h_v = torch.randn(6, 8)
    cvae = CVAE(embed_dim=8, latent_dim=d_z, hidden_dim=32)
    samples = [DataSample(SeedSet({0, 1}), 100., 1., True, 0, 0),
               DataSample(SeedSet({2, 3}), 90., .3, False, 0, 1)]
    encoded = encode_phase2_samples(samples, cvae, h_v, h_v.mean(0))
    assert len(encoded) == 2 and encoded[0]["z"].shape == (d_z,)
    assert encoded[0]["F"] == 100. and encoded[1]["F"] == 90.
    single = ScoreNetTrainer(ScoreNet(d_z, 32), [encoded[0]],
                             ScoreNetConfig(n_epochs=1, log_every=0))
    assert torch.isfinite(single.weights).all() and single.weights.item() == 1.
    print("✓ Test 7  Phase 2 encoding and one-sample DSM: PASS")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
