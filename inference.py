"""GEN-GBC inference: CVAE candidates, V_phi shortlist, exact GBC selection.

The best Phase 2 endpoint and all CEM winners are protected candidates.
V_phi can screen additional samples, but a final choice is always compared
by the exact C++ evaluator in raw ordered-pair internal-node GBC units.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from cvae import CVAE
from exact_gbc_scorer import ExactGBCScorer
from phase2_trajectory import SeedSet
from value_net import ValueNetwork


@dataclass
class InferenceSample:
    seed_set: SeedSet
    v_phi_score: float
    exact_score: float
    source: str


@dataclass
class InferenceResult:
    best_seed_set: SeedSet
    best_exact: float
    best_v_phi: float
    source: str
    all_samples: List[InferenceSample]
    elapsed_s: float


def run_direct_inference(v_phi: ValueNetwork, scorer: ExactGBCScorer,
                         h_v: Tensor, h_G: Tensor,
                         candidates: Sequence[SeedSet],
                         names: Sequence[str]) -> InferenceResult:
    """GEN-CIM's default ``inf_skip_swap`` path with exact GBC verification.

    The CEM winner and top D_traj starting sets are compared directly. With
    an exact evaluator, CEM's cached score is already a ground-truth score.
    """
    start = time.perf_counter()
    if not candidates or len(candidates) != len(names):
        raise ValueError("Supply named CEM or D_traj candidates")
    unique: dict[frozenset[int], tuple[SeedSet, str]] = {}
    for candidate, name in zip(candidates, names):
        if (candidate.k != scorer.k or min(candidate.nodes) < 0
                or max(candidate.nodes) >= scorer.graph.num_nodes):
            raise ValueError("Candidate disagrees with exact scorer graph or k")
        unique.setdefault(frozenset(candidate.nodes), (candidate, name))
    sets, sources = zip(*unique.values())
    exact = scorer.score_many(sets)
    v_phi.eval()
    proxy = v_phi.predict_batch(h_v, sets, h_G)
    if any(not math.isfinite(float(f)) for f in exact + proxy):
        raise ValueError("Non-finite candidate score")
    rows = [InferenceSample(S, float(v), float(f), name)
            for S, v, f, name in zip(sets, proxy, exact, sources)]
    rows.sort(key=lambda row: row.exact_score, reverse=True)
    best = rows[0]
    return InferenceResult(best.seed_set, best.exact_score,
                           best.v_phi_score, best.source, rows,
                           time.perf_counter() - start)


def greedy_swap_polish(seed_set: SeedSet, scorer: ExactGBCScorer,
                       n_rounds: int = 2, two_hop: bool = False) -> SeedSet:
    """GEN-CIM's first-improvement 1/2-hop swap with MC replaced by exact GBC.

    The MC-based two-hop pre-screen becomes an exact ordering; cached exact
    scores are reused when evaluating the ordered candidates again.
    """
    if n_rounds < 0 or seed_set.k != scorer.k:
        raise ValueError("Invalid polish rounds or group size")
    n = scorer.graph.num_nodes
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for src, dst in scorer.graph.edge_index.cpu().t().tolist():
        neighbors[src].append(dst)
    current = set(seed_set.nodes)
    current_score = scorer.score(seed_set)

    def candidates(v: int, hop: int) -> list[int]:
        if hop == 1:
            return [u for u in neighbors[v] if u not in current]
        seen = set(neighbors[v]) | {v}
        result = []
        for w in neighbors[v]:
            for u in neighbors[w]:
                if u not in current and u not in seen:
                    seen.add(u)
                    result.append(u)
        return result

    for hop in ([1, 2] if two_hop else [1]):
        for _ in range(n_rounds):
            improved = False
            for v in list(current):
                options = candidates(v, hop)
                if hop == 2 and options:
                    trials = [SeedSet((current - {v}) | {u}) for u in options]
                    estimates = scorer.score_many(trials)
                    options = [u for _, u in sorted(zip(estimates, options),
                                                    reverse=True)]
                for u in options:
                    trial = SeedSet((current - {v}) | {u})
                    value = scorer.score(trial)
                    if value > current_score + 1e-4:
                        current, current_score = set(trial.nodes), value
                        improved = True
                        break
            if not improved:
                break
    return SeedSet(current)


def polish_with_restarts(seed_set: SeedSet, scorer: ExactGBCScorer, *,
                         extra_starts: Sequence[SeedSet] = (),
                         n_rounds: int = 2, restarts: int = 3,
                         kick: int = 1, seed: int = 42) -> SeedSet:
    """Polish CVAE/CEM/D_traj starts and perturb the current winner."""
    if restarts < 0 or kick < 1:
        raise ValueError("Invalid perturbation restart budget")
    best = greedy_swap_polish(seed_set, scorer, n_rounds)
    best_score = scorer.score(best)
    for candidate in extra_starts:
        refined = greedy_swap_polish(candidate, scorer, n_rounds)
        value = scorer.score(refined)
        if value > best_score:
            best, best_score = refined, value
    rng = random.Random(seed + 7777)
    for _ in range(restarts):
        available = [v for v in range(scorer.graph.num_nodes) if v not in best.nodes]
        radius = min(kick, len(available), best.k)
        if radius == 0:
            break
        kicked = SeedSet((best.nodes - set(rng.sample(sorted(best.nodes), radius))) |
                         set(rng.sample(available, radius)))
        refined = greedy_swap_polish(kicked, scorer, n_rounds)
        value = scorer.score(refined)
        if value > best_score:
            best, best_score = refined, value
    return best


def sample_candidates(cvae: CVAE, h_v: Tensor, h_G: Tensor, k: int,
                      n_samples: int, z_init: Optional[Tensor] = None,
                      noise_scale: float = 0.1) -> List[SeedSet]:
    """Decode prior samples, or perturb a supplied high-quality latent."""
    if n_samples < 0 or noise_scale < 0:
        raise ValueError("Sample count and noise scale must be nonnegative")
    if n_samples == 0:
        return []
    if z_init is not None and (z_init.shape != (cvae.latent_dim,) or not bool(
            torch.isfinite(z_init).all())):
        raise ValueError("z_init needs one finite latent vector")
    cvae.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, cvae.latent_dim, device=h_v.device)
        zs = noise if z_init is None else z_init.to(h_v.device)[None] + noise_scale * noise
        return [cvae.decode_to_seedset(z, h_G, h_v, k) for z in zs]


def run_inference(cvae: CVAE, v_phi: ValueNetwork, scorer: ExactGBCScorer,
                  h_v: Tensor, h_G: Tensor, k: int, *, n_samples: int = 200,
                  shortlist: int = 20, z_init: Optional[Tensor] = None,
                  protected: Sequence[SeedSet] = (),
                  protected_names: Optional[Sequence[str]] = None) -> InferenceResult:
    """Select the best exact GBC from protected and generated candidates.

    exact GBC scores every unique finalist; this includes CEM winners even when
    V_phi ranks them poorly. Pass ``n_samples=0`` for checkpoint-only inference.
    """
    start = time.perf_counter()
    if not 1 <= k <= h_v.size(0) or scorer.k != k or scorer.graph.num_nodes != h_v.size(0):
        raise ValueError("Invalid graph or k for exact GBC scorer")
    if shortlist < 0 or n_samples < 0:
        raise ValueError("Candidate counts must be nonnegative")
    if protected_names is not None and len(protected_names) != len(protected):
        raise ValueError("Need one label per protected candidate")
    if any(s.k != k or min(s.nodes) < 0 or max(s.nodes) >= h_v.size(0)
           for s in protected):
        raise ValueError("Protected seed set does not match graph and k")
    generated = sample_candidates(cvae, h_v, h_G, k, n_samples, z_init)
    v_phi.eval()
    # Deduplicate first; a deterministic top-k CVAE often emits the same set.
    unique: dict[frozenset[int], tuple[SeedSet, str]] = {}
    for i, candidate in enumerate(protected):
        unique.setdefault(frozenset(candidate.nodes),
                          (candidate, protected_names[i] if protected_names else "protected"))
    for candidate in generated:
        unique.setdefault(frozenset(candidate.nodes), (candidate, "CVAE"))
    if not unique:
        raise ValueError("No candidates: supply protected sets or sample latents")
    candidates = list(unique.values())
    proxy = v_phi.predict_batch(h_v, [s for s, _ in candidates], h_G)
    if len(proxy) != len(candidates) or any(not math.isfinite(float(p)) for p in proxy):
        raise ValueError("V_phi returned invalid candidate scores")
    protected_count = len({frozenset(s.nodes) for s in protected})
    # Protected candidates are always checked; additional CVAE sets use V_phi.
    selected = [i for i in range(len(candidates)) if i < protected_count]
    selected += sorted(range(protected_count, len(candidates)),
                       key=lambda i: (-proxy[i], i))[:shortlist]
    finalists = [candidates[i][0] for i in selected]
    scores = scorer.score_many(finalists)
    if len(scores) != len(finalists) or any(not math.isfinite(float(f)) for f in scores):
        raise ValueError("exact GBC returned invalid candidate scores")
    rows = [InferenceSample(candidates[i][0], float(proxy[i]), float(score),
                            candidates[i][1]) for i, score in zip(selected, scores)]
    rows.sort(key=lambda row: row.exact_score, reverse=True)
    best = rows[0]
    return InferenceResult(best.seed_set, best.exact_score,
                           best.v_phi_score, best.source, rows,
                           time.perf_counter() - start)


def _smoke_test() -> None:
    from unittest.mock import Mock
    torch.manual_seed(4)
    h_v = torch.randn(6, 8)
    cvae = CVAE(embed_dim=8, latent_dim=4, hidden_dim=16)
    value = Mock()
    value.predict_batch.side_effect = lambda hv, sets, hg: [float(-sum(s.nodes)) for s in sets]
    scorer = Mock(k=2, graph=Mock(num_nodes=6))
    scorer.score_many.side_effect = lambda sets: [float(sum(s.nodes)) for s in sets]
    protected = [SeedSet({4, 5}), SeedSet({0, 1})]
    result = run_inference(cvae, value, scorer, h_v, h_v.mean(0), 2,
                           n_samples=4, shortlist=0, protected=protected,
                           protected_names=["CEM", "Phase2"])
    assert result.best_seed_set.nodes == {4, 5} and result.source == "CEM"
    assert result.best_exact == 9.0 and len(result.all_samples) == 2
    assert scorer.score_many.call_count == 1
    direct = run_direct_inference(value, scorer, h_v, h_v.mean(0), protected,
                                  ["CEM", "D_traj"])
    assert direct.best_seed_set.nodes == {4, 5}
    path_graph = Mock(num_nodes=4, edge_index=torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long))
    path_scorer = Mock(k=1, graph=path_graph)
    path_scorer.score.side_effect = lambda S: float(sum(S.nodes))
    path_scorer.score_many.side_effect = lambda sets: [float(sum(S.nodes)) for S in sets]
    assert greedy_swap_polish(SeedSet({1}), path_scorer, n_rounds=2).nodes == {3}
    print("inference.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
