r"""
neighbor.py — Locality-Biased 1-opt Neighbour Generation for GEN-GBC.

Maps to ``src/phases/neighbor.py`` in the deployment plan.

Role in the pipeline
────────────────────
This is the **trajectory expansion engine** of Phase 2 Stage B.
For each step in a trajectory the neighbour generator produces K candidate
successor seed sets by swapping one seed node for one semantically similar
non-seed node, then V_phi (Tier 2) scores all candidates in ~1 ms each:

    S_current  ──generate_neighbors──►  [S₁, S₂, …, Sₖ]   (1-opt swaps)
                                              ↓
                                   score with V_phi (Tier 2, ~1 ms each)
                                              ↓
                               best(Sᵢ) → next step in trajectory

"Locality bias" = exploit GATv2 embedding space:
  nodes close in h_v-space share structural/influence roles, so replacing
  v_out with its nearest neighbour in embedding space is more likely to
  preserve or improve GBC(S) than a random swap.

Algorithm (from deployment plan pseudocode, verbatim)
──────────────────────────────────────────────────────
def generate_1opt_neighbors(S_current, h_v, K=5):
    candidates = []
    for v_in in S_current:
        similar = knn_by_embedding(h_v[v_in], V - S_current,
                                   k = K // len(S_current) + 1)
        for v_out in similar:
            candidates.append((S_current - {v_in}) | {v_out})
    return candidates[:K]

Implementation strategy
───────────────────────
1. Cosine similarity via batched matrix multiply → O(N*d), fully vectorised.
2. Per-seed quota  ceil(K / |S|)  so every seed node gets a fair shot
   at contributing candidates, preventing a single well-placed seed from
   monopolising all K slots.
3. Duplicate pruning via frozenset hashing eliminates identical seed sets
   that can arise when two seeds share the same top-similar non-seed nodes.
4. Ranked output: candidates are ordered by cosine similarity of the
   incoming node (best swap first) before the hard cap at K.
5. Graceful degradation: handles |S|=0, |V-S| < K, zero-norm embeddings,
   single-node graphs, and k > all possible distinct swaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# SeedSet is obtained from the incoming object to avoid a circular import.


# ═══════════════════════════════════════════════════════════════════════════════
#  Result container — carries provenance for downstream analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NeighborCandidate:
    """A single 1-opt neighbour with full swap provenance.

    Attributes
    ----------
    seed_set : SeedSet
        Candidate S′ = (S \\ {removed}) ∪ {added}.
    removed : int
        The seed node that was swapped out.
    added : int
        The non-seed node that replaced it.
    similarity : float
        Cosine similarity(h_v[removed], h_v[added]).
        Higher = replacement stays in the same structural neighbourhood.
    rank : int
        0-based position in the output list (0 = best candidate).
    """

    seed_set: SeedSet
    removed: int
    added: int
    similarity: float
    rank: int = 0

    def __repr__(self) -> str:
        return (
            f"NeighborCandidate(rank={self.rank}, "
            f"{self.removed}→{self.added}, "
            f"sim={self.similarity:.4f}, {self.seed_set})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Core vectorised KNN primitive
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_topk(
    query: Tensor,
    pool_embeddings: Tensor,
    pool_ids: Tensor,
    k: int,
) -> Tuple[Tensor, Tensor]:
    """Top-k cosine similarity lookup, fully vectorised.

    Parameters
    ----------
    query : Tensor [d]
        Embedding of the seed node being considered for removal.
    pool_embeddings : Tensor [M, d]
        L2-normalised embeddings of all candidate replacement nodes.
    pool_ids : Tensor [M]  (int64)
        Global node ids corresponding to rows of ``pool_embeddings``.
    k : int
        Number of top similar nodes to return.

    Returns
    -------
    sims : Tensor [k′]
        Cosine similarities, descending.  k′ = min(k, M).
    ids  : Tensor [k′]
        Corresponding global node ids.
    """
    M = pool_embeddings.size(0)
    if M == 0:
        return torch.empty(0, device=query.device), torch.empty(
            0, dtype=torch.long, device=query.device
        )

    k_eff = min(k, M)

    # L2-normalise the query (handles zero-norm gracefully via eps)
    q_norm = F.normalize(query.unsqueeze(0), p=2, dim=1, eps=1e-8)  # [1, d]

    # pool_embeddings are already normalised by the caller
    sims = (q_norm @ pool_embeddings.T).squeeze(0)  # [M]

    # torch.topk is O(M) — faster than full sort for large M
    topk_sims, topk_local_idx = torch.topk(sims, k=k_eff, largest=True, sorted=True)

    return topk_sims, pool_ids[topk_local_idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def generate_neighbors(
    S: "SeedSet",
    h_v: Tensor,
    k: int = 5,
) -> List["SeedSet"]:
    """Generate up to *k* 1-opt neighbour seed sets using cosine similarity.

    This is the main entry point called by Phase 2 Stage B at every
    trajectory step.  For each seed node v_out ∈ S, it finds the most
    embedding-similar non-seed nodes and proposes the swap
    S′ = (S \\ {v_out}) ∪ {v_in}.

    Parameters
    ----------
    S : SeedSet
        Current seed set.
    h_v : Tensor [N, d]
        Node embedding matrix from GATv2 (Phase 1 output).
        Must have at least ``max(S.nodes) + 1`` rows.
    k : int
        Maximum number of distinct neighbour seed sets to return.
        Deployment plan default: K = 5.

    Returns
    -------
    list[SeedSet]
        Deduplicated list of candidate seed sets, length ≤ k.
        Ordered by cosine similarity of the incoming node (best first).
        Returns [] if no valid swap exists.

    Notes
    -----
    The returned list only contains *SeedSet* objects, not provenance.
    Use :func:`generate_neighbors_with_meta` when swap metadata
    (removed node, added node, similarity) is needed for logging or
    ablation studies.
    """
    candidates = generate_neighbors_with_meta(S, h_v, k=k)
    return [c.seed_set for c in candidates]


def generate_neighbors_with_meta(
    S: "SeedSet",
    h_v: Tensor,
    k: int = 5,
) -> List[NeighborCandidate]:
    """Like :func:`generate_neighbors` but returns full provenance.

    Parameters
    ----------
    S : SeedSet
    h_v : Tensor [N, d]
    k : int

    Returns
    -------
    list[NeighborCandidate]
        Up to *k* candidates ordered by cosine similarity (best first).
        Each carries .removed, .added, .similarity, .rank, .seed_set.
    """
    # ── edge case: empty seed set ────────────────────────────────────────
    if S.k == 0:
        return []

    N, d = h_v.shape
    device = h_v.device

    # ── edge case: nothing to swap into (all nodes are already seeds) ────
    non_seed_mask = _build_non_seed_mask(S.nodes, N, device)
    n_available = int(non_seed_mask.sum().item())
    if n_available == 0:
        return []

    # ── pre-normalise ALL embeddings once (reused per-seed query) ────────
    h_norm = F.normalize(h_v, p=2, dim=1, eps=1e-8)  # [N, d]

    # ── build pool: embeddings + ids of non-seed nodes ───────────────────
    pool_ids = non_seed_mask.nonzero(as_tuple=False).squeeze(1)  # [M]
    pool_emb = h_norm[pool_ids]                                   # [M, d]

    # ── per-seed quota: each seed contributes ceil(k / |S|) candidates ──
    #   matches the deployment plan:  k = K // len(S_current) + 1
    quota = k // S.k + 1

    # ── collect raw (similarity, removed, added) triples ─────────────────
    raw: List[Tuple[float, int, int]] = []

    for v_out in sorted(S.nodes):  # deterministic iteration order
        sims, top_ids = _cosine_topk(h_norm[v_out], pool_emb, pool_ids, quota)

        for sim_t, v_in_t in zip(sims.tolist(), top_ids.tolist()):
            v_in = int(v_in_t)
            raw.append((float(sim_t), v_out, v_in))

    # ── sort all triples globally by similarity (desc) ────────────────────
    raw.sort(key=lambda t: t[0], reverse=True)

    # ── deduplicate: frozenset(new_nodes) as the identity key ─────────────
    seen: Set[FrozenSet[int]] = set()
    output: List[NeighborCandidate] = []

    for sim, v_out, v_in in raw:
        if len(output) >= k:
            break

        new_nodes: FrozenSet[int] = frozenset((S.nodes - {v_out}) | {v_in})

        if new_nodes in seen:
            continue  # duplicate seed set — skip
        if new_nodes == frozenset(S.nodes):
            continue  # degenerate: swap produced the same set

        seen.add(new_nodes)
        output.append(
            NeighborCandidate(
                seed_set=type(S)(nodes=set(new_nodes)),
                removed=v_out,
                added=v_in,
                similarity=sim,
                rank=len(output),  # sequential rank after dedup
            )
        )

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_non_seed_mask(
    seed_nodes: Set[int],
    N: int,
    device: torch.device,
) -> Tensor:
    """Boolean mask [N] — True for nodes NOT in seed_nodes."""
    mask = torch.ones(N, dtype=torch.bool, device=device)
    if seed_nodes:
        idx = torch.tensor(sorted(seed_nodes), dtype=torch.long, device=device)
        mask[idx] = False
    return mask


def top_candidate(
    S: "SeedSet",
    h_v: Tensor,
    k: int = 5,
) -> Optional["SeedSet"]:
    """Return the single best 1-opt swap (highest cosine similarity).

    Convenience wrapper for Phase 2 greedy step:
    pick the top-ranked candidate from ``generate_neighbors``.

    Returns ``None`` if no swap is possible.
    """
    candidates = generate_neighbors(S, h_v, k=k)
    return candidates[0] if candidates else None


def generate_neighbors_ranked_by_score(
    S: "SeedSet",
    h_v: Tensor,
    score_fn,
    k: int = 5,
) -> List[Tuple["SeedSet", float]]:
    """Generate candidates, then re-rank by an external score function.

    Intended for Phase 2 Stage B where V_φ is the scoring function:

        neighbors = generate_neighbors_ranked_by_score(
            S, h_v, score_fn=lambda s: v_phi.predict(h_v, s), k=5
        )

    Parameters
    ----------
    S : SeedSet
    h_v : Tensor [N, d]
    score_fn : callable(SeedSet) -> float
        E.g. ``ValueNetwork.predict(h_v, seed_set)``.
    k : int
        Number of candidate neighbours to generate *and* score.

    Returns
    -------
    list[(SeedSet, float)]
        Pairs (seed_set, score), sorted descending by score.
    """
    candidates = generate_neighbors(S, h_v, k=k)
    if not candidates:
        return []
    scored = [(c, score_fn(c)) for c in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Check cardinality, uniqueness and 1-opt swap metadata."""
    from phase2_trajectory import SeedSet
    torch.manual_seed(42)
    h_v = torch.randn(12, 8)
    current = SeedSet({0, 1, 2})
    found = generate_neighbors_with_meta(current, h_v, k=5)
    assert 0 < len(found) <= 5
    assert len({frozenset(x.seed_set.nodes) for x in found}) == len(found)
    assert all(x.seed_set.k == 3 and x.removed in current.nodes
               and x.added not in current.nodes for x in found)
    assert generate_neighbors(SeedSet(set(range(12))), h_v) == []
    print("neighbor.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
