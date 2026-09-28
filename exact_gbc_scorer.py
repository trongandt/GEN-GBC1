"""Batch exact group betweenness scores using the supplied C++ evaluator.

The raw score counts ordered reachable pairs whose shortest path has a group
node strictly inside the path. Candidate IDs use graph_utils' sorted mapping.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, FrozenSet, List, Sequence, Union

import torch

from gbc_types import GraphData
from phase2_trajectory import SeedSet


def compile_exact_gbc(source: Union[str, Path], binary: Union[str, Path]) -> Path:
    """Compile the unchanged exact_gbc.cpp once per source revision."""
    source, binary = Path(source), Path(binary)
    if not source.is_file():
        raise FileNotFoundError(f"Exact GBC source missing: {source}")
    if not binary.is_file() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["g++", "-O2", "-std=c++17", "-fopenmp",
                        str(source.resolve()), "-o", str(binary.resolve())], check=True)
    return binary.resolve()


class ExactGBCScorer:
    """Score fixed-k groups exactly, batching uncached groups per C++ call."""

    def __init__(self, graph: GraphData, graph_path: Union[str, Path],
                 binary_path: Union[str, Path], k: int, threads: int = 1) -> None:
        from graph_utils import load_edge_list
        self.graph_path = Path(graph_path).resolve()
        self.binary = Path(binary_path).resolve()
        if not self.binary.is_file():
            raise FileNotFoundError(f"Compile exact_gbc.cpp first: {self.binary}")
        if graph.is_weighted:
            raise ValueError("The current edge-list loader supports only unweighted graphs")
        if not 1 <= k <= graph.num_nodes or threads < 1:
            raise ValueError("Invalid group size or thread count")
        expected, _ = load_edge_list(self.graph_path, directed=graph.directed)
        if expected.num_nodes != graph.num_nodes or not torch.equal(
                expected.edge_index.cpu(), graph.edge_index.cpu()):
            raise ValueError("Exact evaluator and model have different graphs or ID maps")
        self.graph = graph
        self.k = k
        self.threads = threads
        self.cache: Dict[FrozenSet[int], float] = {}
        self.calls = 0  # Number of distinct exact group evaluations.

    def score_many(self, sets: Sequence[SeedSet]) -> List[float]:
        if not sets:
            return []
        keys = [frozenset(s.nodes) for s in sets]
        if any(len(key) != self.k or min(key) < 0 or max(key) >= self.graph.num_nodes
               for key in keys):
            raise ValueError("Exact scorer requires distinct valid fixed-k groups")
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            # A single invocation loads the graph once and shares the SSSP
            # traversals across all groups. Keep a temporary file for large
            # batches to avoid platform command-line argument limits.
            with tempfile.TemporaryDirectory() as tmp:
                groups = Path(tmp) / "groups.txt"
                groups.write_text("".join(
                    f"g{i}: {' '.join(map(str, sorted(key)))}\n"
                    for i, key in enumerate(missing)), encoding="utf-8")
                cmd = [str(self.binary), "--graph", str(self.graph_path),
                       "--groups-file", str(groups), "--group-ids", "internal",
                       "--threads", str(self.threads), "--quiet"]
                if self.graph.directed:
                    cmd.append("--directed")
                payload = json.loads(subprocess.run(
                    cmd, check=True, capture_output=True, text=True).stdout)
            if (payload.get("pair_domain") != "all_distinct_ordered_pairs"
                    or payload.get("coverage") != "at_least_one_internal_group_node"
                    or payload.get("input_group_id_space") != "internal"
                    or payload.get("num_nodes") != self.graph.num_nodes
                    or payload.get("directed") != self.graph.directed
                    or payload.get("weighted") is not False):
                raise ValueError("Exact evaluator returned an incompatible GBC convention")
            rows = payload.get("results", [])
            if len(rows) != len(missing):
                raise ValueError("Exact evaluator returned an unexpected number of scores")
            calculated = {}
            for i, (key, row) in enumerate(zip(missing, rows)):
                score = float(row["raw_gbc"])
                if (row["method"] != f"g{i}" or row["k"] != self.k
                        or not math.isfinite(score) or score < -1e-8):
                    raise ValueError("Exact evaluator returned an invalid group score")
                calculated[key] = max(0., score)
            self.cache.update(calculated)
            self.calls += len(missing)
        return [self.cache[key] for key in keys]

    def score(self, seed_set: SeedSet) -> float:
        return self.score_many([seed_set])[0]


def _smoke_test() -> None:
    from graph_utils import load_edge_list
    from phase1_representation import compute_node_betweenness_labels
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.txt"
        path.write_text("10 20\n20 30\n30 40\n", encoding="utf-8")
        graph, _ = load_edge_list(path)
        binary = compile_exact_gbc(Path(__file__).with_name("exact_gbc.cpp"),
                                   Path(tmp) / "exact_gbc")
        scorer = ExactGBCScorer(graph, path, binary, 1)
        a, b = SeedSet({1}), SeedSet({2})
        assert scorer.score_many([a, b, a]) == [4., 4., 4.]
        assert scorer.calls == 2
        assert scorer.score(a) == 4. and scorer.calls == 2
        labels = compute_node_betweenness_labels(graph)
        assert scorer.score_many([SeedSet({v}) for v in range(4)]) == labels.tolist()
        try:
            scorer.score(SeedSet({0, 1}))
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted a wrong-size group")
        directed, _ = load_edge_list(path, directed=True)
        directed_scorer = ExactGBCScorer(directed, path, binary, 1)
        assert directed_scorer.score_many([a, b]) == [2., 2.]
    print("exact_gbc_scorer.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
