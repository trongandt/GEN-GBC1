"""Load and preprocess directed or undirected edge lists for GEN-GBC.

Follow GEN-CIM's sorted external-ID mapping.  Keep edge orientation only when
requested; for undirected graphs store both directions in COO.
The resulting node map also matches ``exact_gbc.cpp --group-ids internal``
when both programs read the same edge-list file.  Model-specific node features
and attention matrices belong to their respective model modules.

Pipeline position
-----------------
    SNAP edge list -> GraphData + node map -> model-specific preprocessing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from gbc_types import GraphData


# ═══════════════════════════════════════════════════════════════════════════════
#  1. COO normalization
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_edge_index(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, int, Optional[Tensor]]:
    """Reindex sorted node labels; remove self-loops and repeated directed arcs.

    The first weight wins when duplicate arcs occur, as in GEN-CIM's utility.
    The number of vertices is established *before* removing self-loops, so a
    vertex appearing only in a self-loop retains its internal ID.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long IDs")
    if edge_weight is not None and (
        edge_weight.ndim != 1 or edge_weight.numel() != edge_index.size(1)
    ):
        raise ValueError("edge_weight must have shape [E]")
    if edge_index.size(1) == 0:
        return edge_index.clone(), 0, edge_weight.clone() if edge_weight is not None else None

    labels, inverse = torch.unique(edge_index.reshape(-1), sorted=True, return_inverse=True)
    n = labels.numel()
    remapped = inverse.reshape(2, -1)
    keep = remapped[0] != remapped[1]
    arcs = remapped[:, keep]
    weights = edge_weight[keep] if edge_weight is not None else None
    if arcs.size(1) == 0:
        return edge_index.new_empty((2, 0)), n, weights

    # Stable sorting keeps the first occurrence of every repeated arc.
    code = arcs[0] * n + arcs[1]
    order = torch.argsort(code, stable=True)
    sorted_code = code[order]
    first = torch.ones(order.numel(), dtype=torch.bool, device=order.device)
    first[1:] = sorted_code[1:] != sorted_code[:-1]
    chosen = order[first]
    return arcs[:, chosen].contiguous(), n, weights[chosen] if weights is not None else None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. SNAP edge-list loader and cache
# ═══════════════════════════════════════════════════════════════════════════════

def load_edge_list(
    file_path: Union[str, Path], directed: bool = False,
) -> Tuple[GraphData, Dict[int, int]]:
    """Read an unweighted SNAP edge list and its external-ID map.

    All labels are sorted before remapping, just as in GEN-CIM and the
    supplied exact evaluator.  Directed input keeps only its given arcs;
    undirected input adds the reverse of each edge.
    """
    if not isinstance(directed, bool):
        raise TypeError("directed must be a bool")
    path = Path(file_path)
    edges: List[Tuple[int, int]] = []
    labels = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content = line.split("#", 1)[0].strip()
            if not content or content.startswith("%"):
                continue
            parts = content.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number}: expected two node IDs")
            try:
                source, target = int(parts[0]), int(parts[1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: node IDs must be integers") from error
            labels.update((source, target))
            edges.append((source, target))

    if not labels:
        raise ValueError(f"Graph has no nodes: {path}")
    node_map = {external: internal for internal, external in enumerate(sorted(labels))}

    # Normalize after adding reverse arcs for undirected input only.
    arcs = [(node_map[u], node_map[v]) for u, v in edges]
    if not directed:
        arcs += [(v, u) for u, v in arcs]
    edge_index = torch.tensor(arcs, dtype=torch.long).t().contiguous()
    edge_index, num_nodes, _ = normalize_edge_index(edge_index)
    if num_nodes != len(node_map):
        raise AssertionError("Preprocessing changed the external-to-internal node map")
    return GraphData(edge_index=edge_index, num_nodes=num_nodes, directed=directed), node_map


def save_node_map(node_map: Dict[int, int], path: Union[str, Path]) -> None:
    """Save the mapping used to convert GBC candidate sets to original IDs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({str(k): v for k, v in sorted(node_map.items())}, indent=2) + "\n",
        encoding="utf-8",
    )


def load_node_map(path: Union[str, Path]) -> Dict[int, int]:
    """Restore original integer labels and their internal contiguous IDs."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(external): int(internal) for external, internal in raw.items()}


def load_graph(
    name: str, root: Union[str, Path] = "data/", use_cache: bool = True,
    directed: bool = False,
) -> GraphData:
    """Load ``root/name.txt`` and save ``name_node_map.json``.

    A cache stores tensors and the mapping together.  Its source fingerprint
    prevents a changed edge list or direction mode from using an older graph.
    """
    dataset = name.lower()
    directory = Path(root)
    source = directory / f"{dataset}.txt"
    source_stat = source.stat()
    fingerprint = (source_stat.st_size, source_stat.st_mtime_ns)
    cache_path = directory / f"{dataset}.pt"
    map_path = directory / f"{dataset}_node_map.json"

    if use_cache and cache_path.is_file():
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (cache.get("source_fingerprint") == fingerprint
                    and cache.get("directed", False) == directed):
                graph = GraphData(
                    edge_index=cache["edge_index"],
                    num_nodes=cache["num_nodes"],
                    edge_weight=cache.get("edge_weight"),
                    directed=directed,
                )
                node_map = cache["node_map"]
                if len(node_map) != graph.num_nodes:
                    raise ValueError("Cached node map and graph disagree")
                save_node_map(node_map, map_path)
                print(f"[INFO] Loaded cached graph: {cache_path}")
                return graph
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
            print(f"[WARN] Cache unavailable ({error}); reading edge list")

    print(f"[INFO] Loading SNAP dataset: {source}")
    graph, node_map = load_edge_list(source, directed=directed)
    save_node_map(node_map, map_path)
    print(f"[INFO] Saved final node map: {map_path} ({len(node_map)} nodes)")
    if use_cache:
        torch.save(
            {
                "source_fingerprint": fingerprint,
                "directed": directed,
                "edge_index": graph.edge_index,
                "num_nodes": graph.num_nodes,
                "edge_weight": graph.edge_weight,
                "node_map": node_map,
            },
            cache_path,
        )
        print(f"[INFO] Saved cache: {cache_path}")
    return graph


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Representations required by graph models
# ═══════════════════════════════════════════════════════════════════════════════

def build_adj_list(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    """Build outgoing adjacency lists, including lists for isolated vertices."""
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
    return adjacency


def to_pyg_data(graph: GraphData, node_features: Optional[Tensor] = None):
    """Build PyG Data; leave ``x`` unset when the caller supplies no features."""
    from torch_geometric.data import Data

    if node_features is not None and (
        node_features.ndim != 2 or node_features.size(0) != graph.num_nodes
    ):
        raise ValueError("node_features must have shape [num_nodes, feature_dim]")
    edge_attr = graph.edge_weight
    if edge_attr is None:
        edge_attr = torch.ones(graph.num_edges, dtype=torch.float32, device=graph.device)
    return Data(
        x=node_features.to(graph.device) if node_features is not None else None,
        edge_index=graph.edge_index,
        edge_attr=edge_attr.unsqueeze(-1),
        num_nodes=graph.num_nodes,
    )


def print_graph_stats(graph: GraphData) -> None:
    """Print counts with an explicit distinction between edges and COO arcs."""
    average_degree = graph.num_edges / graph.num_nodes
    print("===== GRAPH STATS =====")
    print(f"Nodes: {graph.num_nodes}")
    print(f"Directed: {graph.directed}")
    print(f"Edges: {graph.num_edges if graph.directed else graph.num_edges // 2}")
    print(f"COO arcs: {graph.num_edges}")
    print(f"Avg out-degree: {average_degree:.2f}")
    print(f"Weighted: {graph.is_weighted}")
    print("=======================")


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Check sorted IDs, duplicate removal, self-loops, and cache."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp:
        source = Path(temp) / "toy.txt"
        source.write_text("# undirected toy\n30 10\n10 30\n10 20\n20 20\n", encoding="utf-8")
        graph = load_graph("toy", temp)
        assert graph.num_nodes == 3 and graph.num_edges == 4
        assert load_node_map(Path(temp) / "toy_node_map.json") == {10: 0, 20: 1, 30: 2}
        assert graph.edge_index.tolist() == [[0, 0, 1, 2], [1, 2, 0, 0]]
        assert build_adj_list(graph.edge_index, 3) == [[1, 2], [0], [0]]
        assert load_graph("toy", temp).edge_index.equal(graph.edge_index)
        directed = load_graph("toy", temp, directed=True)
        assert directed.directed and directed.num_edges == 3
        assert directed.edge_index.tolist() == [[0, 0, 2], [1, 2, 0]]
        assert build_adj_list(directed.edge_index, 3) == [[1, 2], [], [0]]
        assert load_graph("toy", temp, directed=False).num_edges == 4
        assert load_graph("toy", temp, directed=True).num_edges == 3
        source.write_text("30 10\n10 20\n10 40\n", encoding="utf-8")
        assert load_graph("toy", temp).num_nodes == 4  # stale cache rejected
    print("graph_utils.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
