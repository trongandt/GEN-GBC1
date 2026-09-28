"""Phase 4 latent optimization for GEN-GBC (CEM, without REINFORCE).

The Gaussian elite update follows GEN-CIM. Every decoded k-set is evaluated
by the exact C++ GBC evaluator used in Phase 2; no IC simulator
or community labels are involved. ``train_gim.py`` orchestrates one initial
CEM and, for the Full variant, two refinement loops with conditional gold
injection, matching GEN-CIM's training flow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from cvae import CVAE
from exact_gbc_scorer import ExactGBCScorer
from phase2_trajectory import SeedSet
from score_net import ScoreNet, langevin_sample


@dataclass
class CEMConfig:
    pop_size: int = 60
    elite_frac: float = 0.2
    n_iter: int = 25
    sigma_init: float = 1.0
    sigma_min: float = 0.05
    sigma_decay: float = 0.95
    sigma_scale: float = 1.0
    langevin_steps: int = 60
    langevin_sigma_start: float = 1.0
    langevin_sigma_end: float = 0.05
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.pop_size < 2 or self.n_iter < 1 or not 0 < self.elite_frac <= 1:
            raise ValueError("Need pop_size >= 2, n_iter >= 1, elite_frac in (0,1]")
        if min(self.sigma_init, self.sigma_min, self.sigma_decay,
               self.sigma_scale, self.langevin_sigma_start,
               self.langevin_sigma_end) <= 0 or self.langevin_steps < 1:
            raise ValueError("Sigma, scale, decay and Langevin steps must be positive")
        if self.langevin_sigma_start < self.langevin_sigma_end:
            raise ValueError("Langevin sigma must anneal from high to low")


class CEMTrainer:
    """Search a diagonal Gaussian over CVAE latents and retain the best k-set.

    A scorer cache deduplicates repeated decoded sets. The best score is a
    exact raw ordered-pair GBC in the same scale as Phase 2.
    """

    def __init__(self, cvae: CVAE, h_v: Tensor, h_G: Tensor, k: int,
                 scorer: ExactGBCScorer, config: Optional[CEMConfig] = None,
                 score_net: Optional[ScoreNet] = None) -> None:
        self.config = config or CEMConfig()
        if not 1 <= k <= h_v.size(0) or h_v.ndim != 2 or h_G.shape != (h_v.size(1),):
            raise ValueError("Invalid k, node embeddings or graph context")
        if scorer.k != k or scorer.graph.num_nodes != h_v.size(0):
            raise ValueError("Exact GBC scorer must match graph and k")
        if cvae.embed_dim != h_v.size(1):
            raise ValueError("CVAE embedding width differs from Phase 1")
        if score_net is not None and score_net.latent_dim != cvae.latent_dim:
            raise ValueError("ScoreNet and CVAE latent widths differ")
        self.cvae, self.h_v, self.h_G = cvae, h_v, h_G
        self.k, self.scorer, self.score_net = k, scorer, score_net
        self.device = h_v.device
        self.best_F = -math.inf
        self.best_z: Optional[Tensor] = None
        self.best_S: Optional[SeedSet] = None
        self.history: List[dict] = []

    def _init_population(self, z_init: Optional[Tensor],
                         gaussian_proposal: Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        cfg, d_z = self.config, self.cvae.latent_dim
        if gaussian_proposal is not None:
            center, spread = (v.detach().to(self.device) for v in gaussian_proposal)
            if center.shape != (d_z,) or spread.shape != (d_z,) or not bool(
                    torch.isfinite(center).all() and torch.isfinite(spread).all()
                    and (spread > 0).all()):
                raise ValueError("Gaussian proposal needs finite center and positive spread [d_z]")
        elif z_init is not None:
            center = z_init.detach().to(self.device)
            spread = torch.full((d_z,), cfg.sigma_init, device=self.device)
            if center.shape != (d_z,) or not bool(torch.isfinite(center).all()):
                raise ValueError("z_init must be a finite [d_z] vector")
        else:
            center = torch.zeros(d_z, device=self.device)
            spread = torch.full((d_z,), cfg.sigma_init, device=self.device)
        if self.score_net is not None:
            warm = langevin_sample(
                self.score_net, cfg.pop_size, d_z, n_steps=cfg.langevin_steps,
                sigma_start=cfg.langevin_sigma_start,
                sigma_end=cfg.langevin_sigma_end,
                device=self.device,
            )
            # GEN-CIM: Langevin supplies the spread; an elite D_traj proposal,
            # if present, supplies only the centre. z_init is lower priority.
            center = (gaussian_proposal[0].detach().to(self.device)
                      if gaussian_proposal is not None else warm.mean(dim=0))
            spread = warm.std(dim=0, unbiased=False).clamp_min(cfg.sigma_min)
            return center, spread * cfg.sigma_scale
        # The sigma scale is a GEN-CIM Langevin-only option.
        return center, spread.clamp_min(cfg.sigma_min)

    def fit(self, z_init: Optional[Tensor] = None,
            gaussian_proposal: Optional[Tuple[Tensor, Tensor]] = None
            ) -> Tuple[Tensor, SeedSet, float]:
        cfg = self.config
        self.cvae.eval()
        mu, sigma = self._init_population(z_init, gaussian_proposal)
        n_elite = max(1, int(cfg.pop_size * cfg.elite_frac))
        for iteration in range(cfg.n_iter):
            zs = mu[None] + torch.randn(cfg.pop_size, self.cvae.latent_dim,
                                        device=self.device) * sigma[None]
            with torch.no_grad():
                sets = [self.cvae.decode_to_seedset(z, self.h_G, self.h_v, self.k)
                        for z in zs]
            scores = self.scorer.score_many(sets)
            if len(scores) != cfg.pop_size or any(not math.isfinite(x) for x in scores):
                raise ValueError("exact GBC must return one finite score per candidate")
            values = torch.tensor(scores, dtype=zs.dtype, device=self.device)
            elite_idx = values.topk(n_elite).indices
            elite_z = zs[elite_idx]
            mu = elite_z.mean(0)
            sigma = elite_z.std(0, unbiased=False).clamp_min(cfg.sigma_min)
            sigma = sigma * cfg.sigma_decay
            winner = int(values.argmax().item())
            if scores[winner] > self.best_F:
                self.best_F = scores[winner]
                self.best_S = sets[winner]
                self.best_z = zs[winner].detach().clone()
            self.history.append({"iteration": iteration + 1,
                                 "mean_GBC": float(values.mean().item()),
                                 "elite_GBC": float(values[elite_idx].mean().item()),
                                 "best_GBC": self.best_F,
                                 "sigma": float(sigma.mean().item())})
            if cfg.verbose:
                print(f"  [CEM] {iteration+1}/{cfg.n_iter} "
                      f"elite={self.history[-1]['elite_GBC']:.4f} "
                      f"best={self.best_F:.4f}")
        assert self.best_z is not None and self.best_S is not None
        return self.best_z, self.best_S, self.best_F


def _smoke_test() -> None:
    from unittest.mock import Mock
    torch.manual_seed(7)
    h_v = torch.randn(6, 8)
    cvae = CVAE(embed_dim=8, latent_dim=4, hidden_dim=16)
    scorer = Mock(k=2, graph=Mock(num_nodes=6))
    scorer.score_many.side_effect = lambda sets: [float(sum(s.nodes)) for s in sets]
    cfg = CEMConfig(pop_size=5, elite_frac=0.4, n_iter=2, verbose=False)
    trainer = CEMTrainer(cvae, h_v, h_v.mean(0), 2, scorer, cfg)
    z, seed_set, score = trainer.fit()
    assert z.shape == (4,) and seed_set.k == 2 and score == sum(seed_set.nodes)
    assert len(trainer.history) == 2 and scorer.score_many.call_count == 2
    try:
        CEMTrainer(cvae, h_v, h_v.mean(0), 3, scorer)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched checkpoint k was accepted")
    print("rl_policy.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
