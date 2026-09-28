"""gatv2.py — Phase 1: GATv2 node representation for GEN-GBC.

Adapted from GEN-CIM's ``models/gatv2.py``.  The encoder and ranking trainer
retain its architecture and training hyperparameters.  The supervision is
exact, raw singleton group betweenness rather than MC influence spread.

Architecture
------------
    Input  : graph COO edge_index, structural features x_v [N, d_in]
    GATv2  : 2-3 layers, 8 attention heads, ELU
    Output : h_v [N, d], h_G = mean(h_v) [d]

No community representation is built: GEN-GBC does not use community quotas.
Exact group scoring is handled separately by exact_gbc.cpp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch import Tensor, nn

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:
    GATv2Conv = None


# ═══════════════════════════════════════════════════════════════════════════════
#  1. GATv2Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class GATv2Encoder(nn.Module):
    r"""Multi-head GATv2 backbone, following GEN-CIM Phase 1.

    Parameters
    ----------
    in_channels : int
        Number of input features per node (three for structural defaults).
    hidden_channels : int
        Output embedding dimension d; default 128.
    num_layers : int
        Number of GATv2 layers; default 2.
    heads : int
        Heads per layer; each outputs d / heads channels.
    dropout : float
        Attention dropout and dropout between GATv2 layers.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 128,
        num_layers: int = 2, heads: int = 8, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if GATv2Conv is None:
            raise ImportError("GATv2Encoder requires torch_geometric")
        if in_channels < 1 or hidden_channels < 1 or num_layers < 1 or heads < 1:
            raise ValueError("Feature width, embedding width, layers, and heads must be positive")
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GATv2Conv(
                in_channels if layer == 0 else hidden_channels,
                hidden_channels // heads, heads=heads, concat=True,
                dropout=dropout, add_self_loops=True, share_weights=False,
            ))
        self.dropout_layer = nn.Dropout(dropout)
        self.act = nn.ELU()
        for conv in self.convs:
            for name, param in conv.named_parameters():
                if param.ndim >= 2 and "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode all nodes: (x [N, d_in], COO [2, E]) -> h_v [N, d]."""
        if x.ndim != 2 or x.size(1) != self.in_channels:
            raise ValueError("x must have shape [N, in_channels]")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        for layer, conv in enumerate(self.convs):
            x = self.act(conv(x, edge_index))
            if layer < len(self.convs) - 1:
                x = self.dropout_layer(x)
        return x

    @staticmethod
    def get_graph_embedding(h_v: Tensor) -> Tensor:
        """Mean pool to h_G [d]; community outputs are absent in GEN-GBC."""
        return h_v.mean(dim=0)

    def encode_all(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Return the node and graph embeddings (h_v, h_G)."""
        h_v = self(x, edge_index)
        return h_v, self.get_graph_embedding(h_v)

    def __repr__(self) -> str:
        return (
            f"GATv2Encoder(in={self.in_channels}, d={self.hidden_channels}, "
            f"layers={self.num_layers}, heads={self.heads})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Pairwise ranking loss (exact singleton GBC supervision)
# ═══════════════════════════════════════════════════════════════════════════════

class BetweennessRankingLoss(nn.Module):
    r"""Encourage nodes with greater exact GBC({v}) to get greater scores.

    For a pair (u, v) whose labels differ, minimize
        max(0, margin - sign(label_u - label_v) * (score_u - score_v)).
    Tied labels do not give a ranking target.
    """

    def __init__(self, embed_dim: int = 128, margin: float = 1.0,
                 n_pairs: int = 1024) -> None:
        super().__init__()
        if embed_dim < 2 or n_pairs < 1 or margin < 0:
            raise ValueError("Invalid embedding dimension, pair count, or margin")
        self.n_pairs = n_pairs
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        for module in self.score_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, h_v: Tensor, labels: Tensor) -> Tensor:
        """Return the mean margin loss for random, distinct node pairs."""
        if labels.ndim != 1 or labels.numel() != h_v.size(0):
            raise ValueError("labels must have shape [N]")
        if not bool(torch.isfinite(labels).all()):
            raise ValueError("labels must be finite")
        n = h_v.size(0)
        scores = self.score_head(h_v).squeeze(-1)
        if n < 2:
            return scores.sum() * 0.0
        # Draw distinct pairs without wasting samples on i == j.
        idx1 = torch.randint(n, (self.n_pairs,), device=h_v.device)
        idx2 = torch.randint(n - 1, (self.n_pairs,), device=h_v.device)
        idx2 += (idx2 >= idx1).long()
        target = torch.sign(labels[idx1] - labels[idx2])
        valid = target != 0
        if not bool(valid.any()):
            return scores.sum() * 0.0
        return self.ranking_loss(scores[idx1[valid]], scores[idx2[valid]], target[valid])


# The GEN-CIM name remains available for readers adapting downstream code.
InfluenceRankingLoss = BetweennessRankingLoss


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Phase1Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Phase1Config:
    """GEN-CIM Phase 1 hyperparameters, with exact GBC ranking labels."""
    hidden_channels: int = 128
    num_layers: int = 2
    heads: int = 8
    dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 1e-5
    n_epochs: int = 1000
    patience: int = 50
    log_every: int = 100
    ranking_margin: float = 1.0
    n_pairs: int = 1024


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Phase1Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class Phase1Trainer:
    """Train GATv2 and its score head; restore the best training checkpoint.

    The exact labels are produced by ``phase1_representation.py``. The score
    head is optimized jointly with the encoder; later phases use h_v.
    """

    def __init__(self, in_channels: int, config: Optional[Phase1Config] = None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        self.config = config or Phase1Config()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        cfg = self.config
        if cfg.n_epochs < 1 or cfg.patience < 1 or cfg.lr <= 0:
            raise ValueError("n_epochs, patience, and lr must be positive")
        self.model = GATv2Encoder(in_channels, cfg.hidden_channels, cfg.num_layers,
                                  cfg.heads, cfg.dropout).to(self.device)
        self.loss_fn = BetweennessRankingLoss(
            cfg.hidden_channels, cfg.ranking_margin, cfg.n_pairs).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.train_log: List[float] = []

    def fit(self, x: Tensor, edge_index: Tensor, labels: Tensor) -> List[float]:
        """Fit on node features, graph arcs, and raw GBC({v}) labels.

        Returns one ranking-loss value per epoch; early stopping follows
        GEN-CIM's training-loss criterion (there is no validation split).
        """
        x = x.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device)
        # Preserve exact-label ordering beyond float32's 24-bit integer range.
        labels = labels.to(self.device, dtype=torch.float64)
        if x.size(0) != labels.numel():
            raise ValueError("Node features and labels must cover the same nodes")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(labels).all()):
            raise ValueError("Features and labels must be finite")
        best = float("inf")
        bad_epochs = 0
        best_state = None
        self.train_log = []
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        for epoch in range(1, self.config.n_epochs + 1):
            self.model.train()
            self.loss_fn.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(x, edge_index), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            value = float(loss.detach())
            self.train_log.append(value)
            if value < best:
                best, bad_epochs = value, 0
                best_state = (
                    {k: v.detach().clone() for k, v in self.model.state_dict().items()},
                    {k: v.detach().clone() for k, v in self.loss_fn.state_dict().items()},
                )
            else:
                bad_epochs += 1
            # The loss above was measured before the update. Snapshot that
            # same parameter state, then take the optimizer step.
            self.optimizer.step()
            if self.config.log_every > 0 and epoch % self.config.log_every == 0:
                print(f"[Phase1] epoch={epoch} loss={value:.6f} best={best:.6f}")
            if bad_epochs >= self.config.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state[0])
            self.loss_fn.load_state_dict(best_state[1])
        self.model.eval()
        self.loss_fn.eval()
        return self.train_log

    @torch.no_grad()
    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Extract h_v after training [N, hidden_channels]."""
        self.model.eval()
        return self.model(x.to(self.device, dtype=torch.float32),
                          edge_index.to(self.device))

    @torch.no_grad()
    def encode_all(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Extract (h_v [N, d], h_G [d]) after training."""
        self.model.eval()
        return self.model.encode_all(x.to(self.device, dtype=torch.float32),
                                     edge_index.to(self.device))

    def save(self, path: Union[str, Path]) -> None:
        """Save model, score head, config, and label semantics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "loss_fn_state": self.loss_fn.state_dict(),
            "in_channels": self.model.in_channels,
            "config": asdict(self.config),
            "label_semantics": "exact_raw_singleton_bc_all_ordered_pairs_internal_nodes",
        }, path)

    def load(self, path: Union[str, Path]) -> None:
        """Restore a compatible Phase 1 checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if (checkpoint["in_channels"] != self.model.in_channels
                or checkpoint["config"] != asdict(self.config)):
            raise ValueError("Checkpoint and trainer architecture/configuration disagree")
        self.model.load_state_dict(checkpoint["model_state"])
        self.loss_fn.load_state_dict(checkpoint["loss_fn_state"])
        self.model.eval()
        self.loss_fn.eval()


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke tests (same component-by-component style as GEN-CIM)
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Test encoder, pooling, gradients, ranking, training and checkpointing."""
    print("=" * 64)
    print("  gatv2.py — GEN-GBC Phase 1 Smoke Tests")
    print("=" * 64)
    torch.manual_seed(42)

    # The loss head can be tested even when PyG is not installed.
    loss_fn = BetweennessRankingLoss(embed_dim=16, n_pairs=64)
    probe = torch.randn(12, 16, requires_grad=True)
    labels = torch.arange(12, dtype=torch.float64)
    loss = loss_fn(probe, labels)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert probe.grad is not None and bool(probe.grad.abs().sum() > 0)
    print("✓ Test 1  exact-label pairwise ranking: finite loss and gradients")

    tie_probe = torch.randn(12, 16, requires_grad=True)
    tie_loss = loss_fn(tie_probe, torch.ones(12, dtype=torch.float64))
    assert tie_loss.item() == 0
    tie_loss.backward()
    assert tie_probe.grad is not None
    print("✓ Test 2  tied GBC labels: differentiable zero loss")

    if GATv2Conv is None:
        print("[SKIP] Tests 3–10 require torch_geometric")
        return

    from tempfile import TemporaryDirectory

    n, d_in, d = 12, 3, 16
    # One-way arcs deliberately exercise PyG's directed edge_index handling.
    src = torch.arange(n - 1, dtype=torch.long)
    dst = src + 1
    edge_index = torch.stack((src, dst))
    features = torch.randn(n, d_in)
    model = GATv2Encoder(d_in, hidden_channels=d, heads=4, num_layers=2)

    h_v = model(features, edge_index)
    assert h_v.shape == (n, d)
    print(f"✓ Test 3  GATv2 forward: h_v={tuple(h_v.shape)}")

    h_G = model.get_graph_embedding(h_v)
    assert h_G.shape == (d,)
    print(f"✓ Test 4  graph mean pooling: h_G={tuple(h_G.shape)}")

    model.eval()
    hv2, hg2 = model.encode_all(features, edge_index)
    assert hv2.shape == (n, d) and hg2.shape == (d,)
    assert torch.allclose(hg2, hv2.mean(dim=0))
    print("✓ Test 5  encode_all returns h_v and h_G")

    model.zero_grad(set_to_none=True)
    model(features, edge_index).sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in model.parameters())
    print("✓ Test 6  encoder gradient flow")

    cfg = Phase1Config(hidden_channels=d, heads=4, n_epochs=3,
                       patience=5, log_every=0, n_pairs=64)
    trainer = Phase1Trainer(in_channels=d_in, config=cfg, device="cpu")
    losses = trainer.fit(features, edge_index, labels)
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    print(f"✓ Test 7  trainer fit: {len(losses)} epochs")

    trained_hv, trained_hg = trainer.encode_all(features, edge_index)
    assert trained_hv.shape == (n, d) and trained_hg.shape == (d,)
    print("✓ Test 8  trained node and graph embeddings")

    with TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "phase1_toy.pt"
        trainer.save(checkpoint)
        restored = Phase1Trainer(in_channels=d_in, config=cfg, device="cpu")
        restored.load(checkpoint)
        assert torch.allclose(trainer.encode(features, edge_index),
                              restored.encode(features, edge_index))
    print("✓ Test 9  checkpoint save/load reproduces embeddings")

    assert isinstance(model.act, nn.ELU)
    print("✓ Test 10 ELU activation")
    print("  All 10 gatv2.py smoke tests passed ✓")


if __name__ == "__main__":
    _smoke_test()
