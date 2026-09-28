"""Phase 2 initial experts and 1-opt trajectories for raw group betweenness.

Adapted from GEN-CIM's ``phases/phase2_trajectory.py``.  The 42 default
initializers are degree, greedy_embedding, 28 good_random, 8 semi_random,
and four path-aware GBC strategies. All 42 use graph structure, Phase 1
singleton GBC labels, or Phase 1 embeddings, without community labels.
Stage B uses ValueNetwork alone; exact GBC scores enter in Stages A and C+D.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import random
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gbc_types import GraphData
from neighbor import generate_neighbors
from value_net import ValueNetwork


@dataclass
class SeedSet:
    """A set of internal contiguous node IDs, in GEN-CIM's seed-set form."""
    nodes: Set[int]

    def __post_init__(self) -> None:
        self.nodes = set(self.nodes)

    @property
    def k(self) -> int:
        return len(self.nodes)

    def __hash__(self) -> int:
        return hash(frozenset(self.nodes))


@dataclass
class TrajectoryStep:
    S: SeedSet
    score: float


@dataclass
class Trajectory:
    steps: List[TrajectoryStep] = field(default_factory=list)

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def terminal(self) -> TrajectoryStep:
        return self.steps[-1]

    @property
    def best(self) -> TrajectoryStep:
        return max(self.steps, key=lambda step: step.score)

    def __iter__(self) -> Iterator[TrajectoryStep]:
        return iter(self.steps)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Structural and random initializers (GEN-CIM)
# ═══════════════════════════════════════════════════════════════════════════════

def _degrees(graph: GraphData) -> List[int]:
    return torch.bincount(graph.edge_index[0].cpu(), minlength=graph.num_nodes).tolist()


def _top_k(scores: Sequence[float], k: int) -> SeedSet:
    return SeedSet(set(sorted(range(len(scores)), key=lambda v: (-scores[v], v))[:k]))


def _bridge_scores(graph: GraphData, traffic: Sequence[float]) -> List[float]:
    """Favor sampled path hubs whose neighbors are not tightly linked.

    Local clustering acts as a partition-free bridge proxy: a path hub with
    low clustering is more likely to connect otherwise separate regions.
    """
    adjacent = [set() for _ in range(graph.num_nodes)]
    for u, v in graph.edge_index.cpu().t().tolist():
        if u != v:
            adjacent[u].add(v)
            adjacent[v].add(u)
    scores = []
    for v, neighbors in enumerate(adjacent):
        degree = len(neighbors)
        if degree < 2:
            scores.append(0.)
            continue
        links = sum(len(adjacent[u] & neighbors) for u in neighbors) / 2
        clustering = 2 * links / (degree * (degree - 1))
        scores.append(float(traffic[v]) * (1 - clustering))
    return scores


def init_degree(graph: GraphData, k: int) -> SeedSet:
    return _top_k(_degrees(graph), k)


def init_random(graph: GraphData, k: int, seed: int = 42) -> SeedSet:
    return SeedSet(set(random.Random(seed).sample(range(graph.num_nodes), k)))


def init_greedy_embedding(graph: GraphData, k: int, h_v: Tensor,
                          seed: int = 42) -> SeedSet:
    """GEN-CIM greedy furthest-point sampling in cosine embedding space."""
    h_norm = F.normalize(h_v, p=2, dim=1, eps=1e-8)
    first = random.Random(seed).randrange(graph.num_nodes)
    selected = [first]
    max_sim = h_norm @ h_norm[first]
    for _ in range(k - 1):
        max_sim[selected] = 2.0
        next_node = int(torch.argmin(max_sim))
        selected.append(next_node)
        max_sim = torch.maximum(max_sim, h_norm @ h_norm[next_node])
    return SeedSet(set(selected))


def init_random_filtered(graph: GraphData, k: int, num_good: int = 28,
                         oversample_factor: int = 2, seed: int = 42) -> Dict[str, SeedSet]:
    """Keep the top degree-sum random draws, as in GEN-CIM."""
    rng = random.Random(seed)
    degree = _degrees(graph)
    drawn = [SeedSet(set(rng.sample(range(graph.num_nodes), k)))
             for _ in range(num_good * oversample_factor)]
    drawn.sort(key=lambda s: sum(degree[v] for v in s.nodes), reverse=True)
    return {f"good_random_{i}": s for i, s in enumerate(drawn[:num_good])}


def init_random_local(graph: GraphData, k: int, num_sets: int = 8,
                      anchor_top_frac: float = 0.25, seed: int = 42) -> Dict[str, SeedSet]:
    """Anchor each semi-random group at a high-degree node and its neighbors."""
    rng = random.Random(seed)
    degree = _degrees(graph)
    pool = sorted(range(graph.num_nodes), key=lambda v: (-degree[v], v))
    anchors = pool[:max(1, int(anchor_top_frac * graph.num_nodes))]
    adj = [set() for _ in pool]
    for u, v in graph.edge_index.cpu().t().tolist():
        adj[u].add(v)
    result = {}
    for i in range(num_sets):
        anchor = rng.choice(anchors)
        nodes = {anchor}
        for v in sorted(adj[anchor], key=lambda v: (-degree[v], v)):
            if len(nodes) == k:
                break
            nodes.add(v)
        if len(nodes) < k:
            remaining = [v for v in pool if v not in nodes]
            nodes.update(rng.sample(remaining, k - len(nodes)))
        result[f"semi_random_{i}"] = SeedSet(nodes)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  2. GBC experts: singleton scores and sampled shortest-path traffic
# ═══════════════════════════════════════════════════════════════════════════════

def _shortest_path_load(graph: GraphData, sources: int,
                        targets_per_source: int, seed: int) -> List[float]:
    """Sample shortest paths for expert construction, never for GBC scoring.

    The path from each sampled source to a sampled reachable target is drawn
    from the exact shortest-path distribution using predecessor counts.
    Endpoints are excluded. No partition or community labels are needed.
    These path statistics are heuristic expert features, not GBC estimates.
    """
    n = graph.num_nodes
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    weights = (graph.edge_weight.cpu().tolist() if graph.is_weighted else None)
    for i, (u, v) in enumerate(graph.edge_index.cpu().t().tolist()):
        if u != v:
            adjacency[u].append((v, weights[i] if weights is not None else 1.0))
    traffic = [0.] * n
    rng = random.Random(seed)
    source_ids = rng.sample(range(n), min(sources, n))
    for source in source_ids:
        predecessors: List[List[int]] = [[] for _ in range(n)]
        counts = [0.] * n
        counts[source] = 1.
        if weights is None:
            distance = [-1] * n
            distance[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v, _ in adjacency[u]:
                    if distance[v] < 0:
                        distance[v] = distance[u] + 1
                        queue.append(v)
                    if distance[v] == distance[u] + 1:
                        predecessors[v].append(u)
                        counts[v] += counts[u]
        else:
            distance = [float('inf')] * n
            distance[source] = 0.
            queue = [(0., source)]
            while queue:
                d, u = heapq.heappop(queue)
                if d > distance[u]:
                    continue
                for v, cost in adjacency[u]:
                    candidate = d + cost
                    if candidate < distance[v] - 1e-10:
                        distance[v], predecessors[v], counts[v] = candidate, [u], counts[u]
                        heapq.heappush(queue, (candidate, v))
                    elif abs(candidate - distance[v]) < 1e-10:
                        predecessors[v].append(u)
                        counts[v] += counts[u]
        reachable = [v for v in range(n) if v != source and counts[v] > 0]
        for target in rng.sample(reachable, min(targets_per_source, len(reachable))):
            current = target
            while current != source:
                parents = predecessors[current]
                if not parents:
                    break
                current = rng.choices(parents, weights=[counts[p] for p in parents])[0]
                if current != source:
                    traffic[current] += 1.
    return traffic


def _spread(k: int, ranks: Sequence[int], h_v: Tensor) -> SeedSet:
    """Select candidates while penalizing embedding similarity to picks."""
    if not ranks:
        return SeedSet(set())
    h = F.normalize(h_v, dim=1, eps=1e-8)
    selected = [ranks[0]]
    pool = list(ranks[1:])
    priority = {v: i for i, v in enumerate(ranks)}
    while len(selected) < k and pool:
        similarities = (h[pool] @ h[selected].T).amax(dim=1).tolist()
        picked = min(zip(pool, similarities),
                     key=lambda item: (item[1], priority[item[0]]))[0]
        selected.append(picked)
        pool.remove(picked)
    return SeedSet(set(selected))


def init_seed_sets(graph: GraphData, k: int, h_v: Tensor,
                   random_seed: int = 42,
                   num_good_random: int = 28, num_semi_random: int = 8,
                   oversample_factor: int = 2, anchor_top_frac: float = 0.25,
                   node_gbc: Optional[Tensor] = None, path_sources: int = 64,
                   targets_per_source: int = 32) -> Dict[str, SeedSet]:
    """Return 42 experts by default using no community labels.

    ``node_gbc`` should be Phase 1 exact singleton labels. When omitted, this
    function computes them via Phase 1's Brandes routine for consistency.
    """
    if not 0 < k <= graph.num_nodes or h_v.ndim != 2 or h_v.size(0) != graph.num_nodes:
        raise ValueError("Expected 1 <= k <= N and h_v of shape [N, d]")
    if num_good_random < 0 or num_semi_random < 0 or oversample_factor < 1:
        raise ValueError("Random strategy counts and oversampling must be valid")
    if not 0 < anchor_top_frac <= 1:
        raise ValueError("anchor_top_frac must be in (0, 1]")
    if node_gbc is None:
        from phase1_representation import compute_node_betweenness_labels
        node_gbc = compute_node_betweenness_labels(graph)
    if node_gbc.numel() != graph.num_nodes:
        raise ValueError("node_gbc needs one singleton score per vertex")
    singleton = node_gbc.detach().cpu().tolist()
    traffic = _shortest_path_load(
        graph, path_sources, targets_per_source, random_seed + 2000)
    result = {
        "degree": init_degree(graph, k),
        "greedy_embedding": init_greedy_embedding(graph, k, h_v, random_seed),
    }
    result.update(init_random_filtered(graph, k, num_good_random,
                                       oversample_factor, random_seed))
    result.update(init_random_local(graph, k, num_semi_random,
                                    anchor_top_frac, random_seed + 1000))
    order = sorted(range(graph.num_nodes), key=lambda v: (-singleton[v], v))
    result["gbc_diverse"] = _spread(k, order, h_v)
    result["gbc_bridge"] = _top_k(_bridge_scores(graph, traffic), k)
    # Choose high-traffic nodes while discouraging duplicate local coverage.
    hot_order = sorted(range(graph.num_nodes), key=lambda v: (-traffic[v], -singleton[v], v))
    result["gbc_hotspot_spread"] = _spread(k, hot_order, h_v)
    maximum_singleton = max(max(singleton), 1.)
    result["gbc_local_path"] = _top_k(
        [traffic[v] + singleton[v] / maximum_singleton
         for v in range(graph.num_nodes)], k)
    assert all(s.k == k for s in result.values())
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Trajectory construction (GEN-CIM Stage B)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trajectory(S_init: SeedSet, h_v: Tensor, V_phi: ValueNetwork,
                     H: int = 5, k_neighbors: int = 5, h_G: Optional[Tensor] = None,
                     early_stop: bool = True) -> Trajectory:
    """Iteratively choose the highest V_phi-scored 1-opt swap, never exact GBC."""
    if H < 0 or k_neighbors < 1:
        raise ValueError("H must be nonnegative and k_neighbors positive")
    trajectory = Trajectory([TrajectoryStep(S_init, V_phi.predict(h_v, S_init, h_G))])
    for _ in range(H):
        neighbors = generate_neighbors(trajectory.terminal.S, h_v, k_neighbors)
        if not neighbors:
            break
        scores = V_phi.predict_batch(h_v, neighbors, h_G)
        index = max(range(len(scores)), key=lambda i: scores[i])
        if early_stop and scores[index] <= trajectory.terminal.score:
            break
        trajectory.append(TrajectoryStep(neighbors[index], scores[index]))
    return trajectory


@dataclass
class TrajectoryDataset:
    trajectories: List[Trajectory] = field(default_factory=list)
    strategy_names: List[str] = field(default_factory=list)
    endpoint_weight: float = 1.0
    midpoint_weight: float = 0.3

    def add(self, trajectory: Trajectory, name: str) -> None:
        self.trajectories.append(trajectory)
        self.strategy_names.append(name)

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    def total_steps(self) -> int:
        return sum(t.length for t in self.trajectories)

    def all_steps_with_weights(self):
        for trajectory in self.trajectories:
            for i, step in enumerate(trajectory):
                yield step, (self.endpoint_weight if i in (0, trajectory.length - 1)
                             else self.midpoint_weight)

    def best_seed_set(self) -> Optional[SeedSet]:
        best = [trajectory.best for trajectory in self.trajectories if trajectory.steps]
        return max(best, key=lambda s: s.score).S if best else None


def build_all_trajectories(graph: GraphData, k: int, h_v: Tensor,
                           V_phi: ValueNetwork,
                           H: int = 5, k_neighbors: int = 5, h_G: Optional[Tensor] = None,
                           early_stop: bool = True, random_seed: int = 42,
                           num_good_random: int = 28, num_semi_random: int = 8,
                           oversample_factor: int = 2, anchor_top_frac: float = 0.25,
                           endpoint_weight: float = 1.0, midpoint_weight: float = 0.3,
                           seed_sets_override: Optional[Dict[str, SeedSet]] = None,
                           node_gbc: Optional[Tensor] = None) -> TrajectoryDataset:
    """Run one improvement chain per quality-gated initializer."""
    seeds = seed_sets_override if seed_sets_override is not None else init_seed_sets(
        graph, k, h_v, random_seed, num_good_random, num_semi_random,
        oversample_factor, anchor_top_frac, node_gbc)
    dataset = TrajectoryDataset(endpoint_weight=endpoint_weight,
                                midpoint_weight=midpoint_weight)
    for name, S in seeds.items():
        dataset.add(build_trajectory(S, h_v, V_phi, H, k_neighbors, h_G, early_stop), name)
    return dataset


def _smoke_test() -> None:
    """Verify exactly 42 experts and monotone greedy trajectories."""
    torch.manual_seed(42)
    graph = GraphData(torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]]), 6)
    h_v = torch.randn(6, 8)
    labels = torch.tensor([1., 2., 3., 4., 2., 1.])
    base = init_seed_sets(graph, 2, h_v, node_gbc=labels,
                          path_sources=4, targets_per_source=3)
    assert len(base) == 42 and all(s.k == 2 for s in base.values())
    assert set(base) == ({"degree", "greedy_embedding", "gbc_diverse",
                          "gbc_bridge", "gbc_hotspot_spread", "gbc_local_path"}
                         | {f"good_random_{i}" for i in range(28)}
                         | {f"semi_random_{i}" for i in range(8)})
    value = ValueNetwork(embed_dim=8)
    dataset = build_all_trajectories(graph, 2, h_v, value, H=2,
                                     seed_sets_override={"degree": base["degree"]})
    assert dataset.num_trajectories == 1 and 1 <= dataset.total_steps() <= 3
    assert all(b.score > a.score for a, b in zip(dataset.trajectories[0].steps,
                                                  dataset.trajectories[0].steps[1:]))
    print("phase2_trajectory.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
