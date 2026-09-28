"""phase1_representation.py — Phase 1 runner for GEN-GBC.

Adapted from GEN-CIM's ``phases/phase1_representation.py``:
    1. Build three structural input features when none are supplied.
    2. Compute one exact Brandes singleton GBC label per node.
    3. Train GATv2 with the pairwise ranking loss.
    4. Save the trained node embeddings and encoder checkpoint.

The label of node v is the raw GBC score of {v}: all reachable ordered
source-target pairs contribute the fraction of their shortest paths whose
*internal* nodes contain v. This matches exact_gbc.cpp's score convention.
Group scores after Phase 1 are computed by exact_gbc.cpp.
"""

from __future__ import annotations

from collections import deque
import heapq
from pathlib import Path
import time
from typing import Optional, Union

import torch
from torch import Tensor

from gbc_types import GraphData
from gatv2 import Phase1Config, Phase1Trainer


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Exact singleton GBC labels (replaces GEN-CIM's MC-IC labels)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_node_betweenness_labels(graph: GraphData) -> Tensor:
    r"""Compute exact raw GBC({v}) for every v using Brandes accumulation.

    For each source s, BFS (unit costs) or Dijkstra (positive edge costs)
    provides shortest-path counts sigma and predecessor lists. Accumulating
    dependencies in reverse distance order gives all singleton scores in one
    pass over the source. Unreachable pairs contribute zero.

    An undirected graph is stored as two directed COO arcs per edge. Every
    source runs once, so both endpoint orientations are counted: do *not*
    divide by two. Endpoints themselves never receive dependency credit.

    Parameters
    ----------
    graph : GraphData
        Internal contiguous IDs and correctly oriented COO arcs.

    Returns
    -------
    Tensor [N], CPU float64
        Exact, unnormalized ordered-pair singleton GBC labels. The trainer
        moves them to its device while preserving float64 ranking order.
    """
    n = graph.num_nodes
    edges = graph.edge_index.detach().cpu().t().tolist()
    costs = (graph.edge_weight.detach().cpu().tolist()
             if graph.edge_weight is not None else None)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        if u != v:
            adjacency[u].append((v, float(costs[i]) if costs is not None else 1.0))

    labels = [0.0] * n
    for source in range(n):
        predecessors: list[list[int]] = [[] for _ in range(n)]
        sigma = [0.0] * n
        sigma[source] = 1.0
        stack: list[int] = []

        if costs is None:
            # Unweighted single-source shortest paths (Brandes BFS).
            distance = [-1] * n
            distance[source] = 0
            queue = deque([source])
            while queue:
                v = queue.popleft()
                stack.append(v)
                for w, _ in adjacency[v]:
                    if distance[w] < 0:
                        distance[w] = distance[v] + 1
                        queue.append(w)
                    if distance[w] == distance[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)
        else:
            # Positive costs ensure predecessors are settled before targets.
            distance = [float("inf")] * n
            distance[source] = 0.0
            heap = [(0.0, source)]
            while heap:
                distance_v, v = heapq.heappop(heap)
                if distance_v != distance[v]:
                    continue
                stack.append(v)
                for w, cost in adjacency[v]:
                    candidate = distance_v + cost
                    if candidate < distance[w]:
                        distance[w] = candidate
                        heapq.heappush(heap, (candidate, w))
                        sigma[w] = sigma[v]
                        predecessors[w] = [v]
                    elif candidate == distance[w]:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

        dependency = [0.0] * n
        while stack:
            w = stack.pop()
            if sigma[w]:
                scale = (1.0 + dependency[w]) / sigma[w]
                for v in predecessors[w]:
                    dependency[v] += sigma[v] * scale
            if w != source:
                labels[w] += dependency[w]

    return torch.tensor(labels, dtype=torch.float64)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Node features when the dataset has no raw node attributes
# ═══════════════════════════════════════════════════════════════════════════════

def build_node_features(graph: GraphData) -> Tensor:
    r"""Reproduce GEN-CIM's three structural input features for Phase 1.

        [log(1+d_out), d_out/max(d_out), 1/sqrt(d_out+1)]

    The SAME three formulas apply to both graph types. For undirected COO,
    which stores both arc orientations, d_out is the usual undirected degree.
    For directed COO, d_out is the out-degree. The C++ group evaluator takes
    the edge list directly and uses no learned node features.

    Returns
    -------
    Tensor [N, 3] on graph.device
    """
    deg = torch.bincount(graph.edge_index[0].cpu(),
                         minlength=graph.num_nodes).to(torch.float32)
    maximum = deg.max().clamp(min=1.0)
    return torch.stack((torch.log1p(deg), deg / maximum,
                        torch.rsqrt(deg + 1.0)), dim=1).to(graph.device)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Main Phase 1 runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(
    graph: GraphData,
    node_features: Optional[Tensor] = None,
    config: Optional[Phase1Config] = None,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_dir: Union[str, Path] = "experiments/checkpoints",
    dataset_name: str = "graph",
    return_labels: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Run exact labels -> GATv2 ranking training -> node embeddings.

    Parameters
    ----------
    graph : GraphData
        Directed arcs, or both orientations for every undirected edge.
    node_features : Tensor [N, d_in] | None
        When None, use GEN-CIM's three structural features for either graph
        type. The exact C++ evaluator does not use these features.
    config : Phase1Config | None
        GATv2 and optimizer configuration.
    device : torch.device | str | None
        Default: CUDA if available, otherwise CPU.
    checkpoint_dir : str | Path
        Where to save the encoder checkpoint and h_v tensor.
    dataset_name : str
        Suffix used in output file names.
    return_labels : bool
        When True, also return exact singleton labels for Phase 2 reuse.

    Returns
    -------
    Tensor [N, hidden_channels] | (Tensor [N, hidden_channels], Tensor [N])
        Learned node embeddings, optionally with the exact singleton labels
        so Phase 2 can reuse them when constructing its initial experts.
    """
    if graph.num_nodes < 1:
        raise ValueError("Graph is empty")
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if node_features is None:
        node_features = build_node_features(graph)
    if (node_features.ndim != 2 or node_features.size(0) != graph.num_nodes
            or node_features.size(1) < 1):
        raise ValueError("node_features must have shape [num_nodes, feature_dim]")
    print(f"[Phase1] Exact Brandes singleton labels: N={graph.num_nodes}")
    start = time.perf_counter()
    labels = compute_node_betweenness_labels(graph)
    print(f"[Phase1] Brandes done in {time.perf_counter() - start:.1f}s")
    trainer = Phase1Trainer(node_features.size(1), config=config, device=dev)
    print(f"[Phase1] Training GATv2 on {dev}")
    losses = trainer.fit(node_features, graph.edge_index, labels)
    h_v = trainer.encode(node_features, graph.edge_index)
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trainer.save(directory / f"phase1_{dataset_name}.pt")
    torch.save(h_v.detach().cpu(), directory / f"h_v_{dataset_name}.pt")
    print(f"[Phase1] h_v={tuple(h_v.shape)}, epochs={len(losses)}, "
          f"best_loss={min(losses):.6f}")
    return (h_v, labels) if return_labels else h_v


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Check score convention, graph direction, features, training, and files."""
    print("=" * 64)
    print("  phase1_representation.py — GEN-GBC Smoke Tests")
    print("=" * 64)
    torch.manual_seed(0)

    # One undirected path, stored with both arc directions. Only its center
    # lies internally on a shortest path, for the pairs (0,2) and (2,0).
    graph = GraphData(torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), 3)
    assert compute_node_betweenness_labels(graph).tolist() == [0., 2., 0.]
    print("✓ Test 1  undirected ordered-pair raw GBC: [0, 2, 0]")

    directed = GraphData(torch.tensor([[0, 1], [1, 2]]), 3, directed=True)
    assert compute_node_betweenness_labels(directed).tolist() == [0., 1., 0.]
    print("✓ Test 2  directed path raw GBC: [0, 1, 0]")

    # Two equal shortest paths 0->1->3 and 0->2->3 each contribute one half.
    diamond = GraphData(torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]]),
                        4, directed=True)
    assert compute_node_betweenness_labels(diamond).tolist() == [0., .5, .5, 0.]
    print("✓ Test 3  multiple shortest paths split fractional credit")

    weighted = GraphData(torch.tensor([[0, 0, 1], [1, 2, 2]]), 3,
                         edge_weight=torch.tensor([1., 3., 1.]), directed=True)
    assert compute_node_betweenness_labels(weighted).tolist() == [0., 1., 0.]
    print("✓ Test 4  weighted Dijkstra follows the cheaper two-hop path")

    disconnected = GraphData(torch.tensor([[0], [1]]), 3, directed=True)
    assert compute_node_betweenness_labels(disconnected).tolist() == [0., 0., 0.]
    print("✓ Test 5  unreachable pairs and isolated node contribute zero")

    undirected_features = build_node_features(graph)
    directed_features = build_node_features(directed)
    assert undirected_features.shape == directed_features.shape == (3, 3)
    assert torch.allclose(undirected_features[:, 0],
                          torch.log1p(torch.tensor([1., 2., 1.])))
    assert torch.allclose(directed_features[:, 0],
                          torch.log1p(torch.tensor([1., 1., 0.])))
    print("✓ Test 6  same three GEN-CIM feature formulas for both graph types")

    if not torch_geometric_available():
        print("[SKIP] Tests 7–8 require torch_geometric")
        return

    from tempfile import TemporaryDirectory
    config = Phase1Config(hidden_channels=16, heads=4, n_epochs=3,
                          patience=5, log_every=0, n_pairs=32)
    with TemporaryDirectory() as temp:
        h_v = run_phase1(graph, config=config, device="cpu",
                         checkpoint_dir=temp, dataset_name="undirected")
        assert h_v.shape == (3, 16)
        assert (Path(temp) / "phase1_undirected.pt").is_file()
        assert torch.load(Path(temp) / "h_v_undirected.pt",
                          weights_only=True).shape == (3, 16)
        print("✓ Test 7  undirected end-to-end training and saved embeddings")

        directed_hv = run_phase1(directed, config=config, device="cpu",
                                 checkpoint_dir=temp, dataset_name="directed")
        assert directed_hv.shape == (3, 16)
        assert (Path(temp) / "phase1_directed.pt").is_file()
        print("✓ Test 8  directed end-to-end training and checkpoint")

    print("  All 8 phase1_representation.py smoke tests passed ✓")


def torch_geometric_available() -> bool:
    from gatv2 import GATv2Conv
    return GATv2Conv is not None


if __name__ == "__main__":
    _smoke_test()
