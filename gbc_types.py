"""Core graph data structure shared by GEN-GBC graph models.

Follow GEN-CIM's single GraphData container: the loader owns external-ID
mapping, while the container validates and transfers its sparse COO tensors.

This module defines graph storage only; each model chooses its own node
features and representation learning procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
#  1. GraphData
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class GraphData:
    """Directed graph or bidirectional COO representation of an undirected graph.

    Parameters
    ----------
    edge_index : Tensor [2, E]
        Directed COO entries with internal IDs in ``[0, num_nodes)``.
    num_nodes : int
        Number of vertices, including vertices incident only to self-loops
        removed during preprocessing.  Original IDs are saved separately.
    edge_weight : Tensor [E], optional
        Positive path costs aligned with ``edge_index``.  ``None`` means unit
        costs.  The ca-GrQc loader presently returns an unweighted graph.
    directed : bool
        Whether source-target orientation is part of the input graph.  For an
        undirected graph, the loader stores both orientations of each edge.
    """

    edge_index: Tensor
    num_nodes: int
    edge_weight: Optional[Tensor] = None
    directed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.edge_index, Tensor):
            raise TypeError("edge_index must be a torch.Tensor")
        if self.edge_index.ndim != 2 or self.edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if self.edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long IDs")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not isinstance(self.directed, bool):
            raise TypeError("directed must be a bool")
        if self.edge_index.numel() and (
            bool((self.edge_index < 0).any())
            or bool((self.edge_index >= self.num_nodes).any())
        ):
            raise ValueError("edge_index contains a node outside [0, num_nodes)")
        if self.edge_weight is not None:
            if self.edge_weight.ndim != 1 or self.edge_weight.numel() != self.num_edges:
                raise ValueError("edge_weight must have shape [E]")
            if self.edge_weight.device != self.edge_index.device:
                raise ValueError("edge_weight and edge_index must share a device")
            if not bool(torch.isfinite(self.edge_weight).all()) or not bool(
                (self.edge_weight > 0).all()
            ):
                raise ValueError("edge_weight must contain finite positive costs")

    @property
    def num_edges(self) -> int:
        """Number of COO arcs (twice the edge count if undirected)."""
        return self.edge_index.size(1)

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    @property
    def is_weighted(self) -> bool:
        return self.edge_weight is not None

    def to(self, device: Union[str, torch.device]) -> GraphData:
        """Return a new instance with all graph tensors on ``device``."""
        return GraphData(
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            edge_weight=(
                self.edge_weight.to(device) if self.edge_weight is not None else None
            ),
            directed=self.directed,
        )

    def __repr__(self) -> str:
        kind = "weighted" if self.is_weighted else "unweighted"
        return (
            f"GraphData(nodes={self.num_nodes}, arcs={self.num_edges}, "
            f"directed={self.directed}, {kind}, device={self.device})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Check graph validation and device transfer on a two-node graph."""
    graph = GraphData(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=2,
    )
    assert graph.num_edges == 2 and not graph.is_weighted
    assert graph.to("cpu").edge_index.equal(graph.edge_index)
    one_way = GraphData(torch.tensor([[0], [1]], dtype=torch.long), 2, directed=True)
    assert one_way.directed and one_way.to("cpu").directed
    try:
        GraphData(torch.tensor([[0], [2]], dtype=torch.long), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range node was accepted")
    print("gbc_types.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
