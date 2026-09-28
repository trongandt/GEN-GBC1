"""Value network V_phi for GEN-GBC Phase 2.

Adapted from GEN-CIM's models/value_net.py: mean-pooled Phase 1 embeddings,
256/128 MLP, weighted MSE plus pairwise ranking, online update and checkpoint.
Stage A and C+D labels come from exact_gbc.cpp. ValueNet predictions
returned by ``predict`` and ``predict_batch`` are denormalized to raw GBC
units, so trajectory scores and dataset labels share the same scale.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

# Seed sets are supplied by phase2_trajectory.py; avoid a circular import.


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  compute_seed_embedding  — h_S from node embeddings + seed set
# ═══════════════════════════════════════════════════════════════════════════════

def compute_seed_embedding(
    h_v: Tensor,
    seed_set: "SeedSet",
) -> Tensor:
    """Mean-pool node embeddings of seed nodes to produce h_S.

    This is the canonical feature extraction for V_φ, matching the
    deployment plan:  h_S = MEAN_POOL({h_v : v ∈ S}).

    Parameters
    ----------
    h_v : Tensor [N, d]
        Node embedding matrix from GATv2 (Phase 1 output).
    seed_set : SeedSet
        Seed node ids.

    Returns
    -------
    Tensor [d]
        Mean-pooled embedding.  Returns a zero vector if the seed set
        is empty (safe for downstream MLP — no NaN).
    """
    if seed_set.k == 0:
        return torch.zeros(h_v.size(1), dtype=h_v.dtype, device=h_v.device)

    idx = torch.tensor(sorted(seed_set.nodes), dtype=torch.long, device=h_v.device)
    return h_v[idx].mean(dim=0)  # [d]


def compute_seed_embedding_batch(
    h_v: Tensor,
    seed_sets: Sequence["SeedSet"],
) -> Tensor:
    """Batched version: returns [B, d] for a list of seed sets.

    Uses zero-padded index gather + mask to avoid a Python for-loop
    over the batch.

    Parameters
    ----------
    h_v : Tensor [N, d]
    seed_sets : sequence of SeedSet  (length B)

    Returns
    -------
    Tensor [B, d]
    """
    B = len(seed_sets)
    d = h_v.size(1)
    device = h_v.device

    if B == 0:
        return torch.empty(0, d, dtype=h_v.dtype, device=device)

    # Find max seed set size for padding
    max_k = max(ss.k for ss in seed_sets) if seed_sets else 0
    if max_k == 0:
        return torch.zeros(B, d, dtype=h_v.dtype, device=device)

    # Build padded index matrix [B, max_k] and mask [B, max_k]
    idx_pad = torch.zeros(B, max_k, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_k, dtype=torch.bool, device=device)

    for i, ss in enumerate(seed_sets):
        nodes = sorted(ss.nodes)
        k = len(nodes)
        if k > 0:
            idx_pad[i, :k] = torch.tensor(nodes, dtype=torch.long, device=device)
            mask[i, :k] = True

    # Gather: [B, max_k, d]
    gathered = h_v[idx_pad]  # safe: padded indices are 0, masked out below

    # Masked mean pooling
    mask_f = mask.unsqueeze(-1).to(h_v.dtype)  # [B, max_k, 1]
    summed = (gathered * mask_f).sum(dim=1)  # [B, d]
    counts = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]  avoid /0

    return summed / counts  # [B, d]


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  ValueNetwork  — the MLP  V_φ
# ═══════════════════════════════════════════════════════════════════════════════

class ValueNetwork(nn.Module):
    """Lightweight MLP proxy for raw GBC(S).

    Parameters
    ----------
    embed_dim : int
        Dimension of seed embedding h_S (= d from GATv2, default 128).
    hidden_dims : tuple[int, ...]
        Hidden layer widths.  Default (256, 128) matches the deployment plan.
    use_context : bool
        If True, input is [h_S ‖ h_G] of dimension 2·embed_dim.
        Used when graph-level context improves prediction accuracy.
    dropout : float
        Dropout rate between hidden layers (0.0 = off).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (256, 128),
        use_context: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.use_context = use_context
        self._normalizer: Optional[RunningNormalizer] = None

        input_dim = embed_dim * 2 if use_context else embed_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.mlp = nn.Sequential(*layers)

        # Xavier initialisation for stable early training
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h_S: Tensor,
        h_G: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict GBC(S) from seed embedding.

        Parameters
        ----------
        h_S : Tensor [d] or [B, d]
            Mean-pooled seed embedding.
        h_G : Tensor [d] or [B, d] | None
            Graph-level context (required if ``use_context=True``).

        Returns
        -------
        Tensor []  or  [B]
            Scalar score(s).
        """
        if self.use_context:
            if h_G is None:
                raise ValueError("h_G is required when use_context=True")
            x = torch.cat([h_S, h_G], dim=-1)
        else:
            x = h_S

        out = self.mlp(x).squeeze(-1)  # remove last dim [B,1]->[B] or [1]->[]
        return out

    def predict(
        self,
        h_v: Tensor,
        seed_set: "SeedSet",
        h_G: Optional[Tensor] = None,
    ) -> float:
        """Convenience: embedding + forward in one call (single seed set).

        Returns a Python float (no gradient).
        """
        with torch.no_grad():
            h_S = compute_seed_embedding(h_v, seed_set)
            score = self.forward(h_S.unsqueeze(0), h_G.unsqueeze(0) if h_G is not None else None)
            if self._normalizer is not None:
                score = self._normalizer.denormalize(score)
        return score.item()

    def predict_batch(
        self,
        h_v: Tensor,
        seed_sets: Sequence["SeedSet"],
        h_G: Optional[Tensor] = None,
    ) -> List[float]:
        """Convenience: batch embedding + forward (no gradient)."""
        with torch.no_grad():
            h_S_batch = compute_seed_embedding_batch(h_v, seed_sets)  # [B, d]
            if h_G is not None:
                h_G_batch = h_G.unsqueeze(0).expand(len(seed_sets), -1)
            else:
                h_G_batch = None
            scores = self.forward(h_S_batch, h_G_batch)
            if self._normalizer is not None:
                scores = self._normalizer.denormalize(scores)
        return scores.tolist()

    def __repr__(self) -> str:
        ctx = "+ctx" if self.use_context else ""
        return f"ValueNetwork(d={self.embed_dim}{ctx}, hidden={self.hidden_dims})"


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Target normaliser for stable training
# ═══════════════════════════════════════════════════════════════════════════════

class RunningNormalizer:
    """Online Welford normaliser for GBC(S) targets.

    The deployment plan trains V_φ on small initial batches
    (Stage 2A) and later on trajectory points (Stage 2D).  A running mean/std
    normaliser avoids numerical instability when target magnitudes vary
    across datasets (e.g. Email vs Skitter).

    Usage
    -----
    >>> norm = RunningNormalizer()
    >>> targets_n = norm.normalize(raw_targets)    # for loss computation
    >>> predictions_raw = norm.denormalize(pred_n) # for interpretation
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0  # sum of (x - mean)^2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return math.sqrt(self._M2 / self._count) + self.eps

    def update(self, values: Tensor) -> None:
        """Incorporate a batch of raw GBC(S) values into running stats."""
        for x in values.detach().cpu().tolist():
            self._count += 1
            delta = x - self._mean
            self._mean += delta / self._count
            delta2 = x - self._mean
            self._M2 += delta * delta2

    def normalize(self, values: Tensor) -> Tensor:
        """z-score normalise: (x - μ) / σ."""
        return (values - self._mean) / self.std

    def denormalize(self, values: Tensor) -> Tensor:
        """Inverse z-score: x * σ + μ."""
        return values * self.std + self._mean

    def state_dict(self) -> Dict[str, float]:
        return {"count": float(self._count), "mean": self._mean, "M2": self._M2}

    def load_state_dict(self, d: Dict[str, float]) -> None:
        self._count = int(d["count"])
        self._mean = d["mean"]
        self._M2 = d["M2"]

    def __repr__(self) -> str:
        return f"RunningNormalizer(n={self._count}, μ={self._mean:.4f}, σ={self.std:.4f})"


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  ValueNetTrainer — full training loop
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Hyperparameters for V_φ training."""
    lr: float = 5e-4
    weight_decay: float = 1e-5
    epochs: int = 200
    batch_size: int = 32
    patience: int = 30          # early stopping
    normalize_targets: bool = True
    log_every: int = 50
    lambda_rank: float = 0.5    # weight for pairwise RankNet loss (0 = disabled)
    rank_pairs_per_batch: int = 32  # random pairs sampled per batch for ranking


class ValueNetTrainer:
    """End-to-end trainer for V_φ with weighted MSE and target normalisation.

    The deployment plan specifies:
      - Stage 2A: bootstrap on 4 points (small batch, many epochs)
      - Stage 2D: retrain on ~24 points with sample weights
                  (endpoints w=1.0, midpoints w=0.3)
      - Phase 3 : CVAE loss includes λ·V_φ(S_decoded); V_φ frozen or fine-tuned
      - Online update in Stage 2C when exact GBC scoring deviates

    Parameters
    ----------
    model : ValueNetwork
    config : TrainConfig
    device : torch.device
    """

    def __init__(
        self,
        model: ValueNetwork,
        config: Optional[TrainConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.normalizer = RunningNormalizer()
        self.model._normalizer = self.normalizer
        self.best_state: Optional[Dict] = None
        self.train_losses: List[float] = []

    def fit(
        self,
        h_S_batch: Tensor,
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        h_G_batch: Optional[Tensor] = None,
    ) -> List[float]:
        """Train V_φ on a dataset of (h_S, GBC(S)) pairs.

        Parameters
        ----------
        h_S_batch : Tensor [N_samples, d]
            Pre-computed seed embeddings.
        targets : Tensor [N_samples]
            Ground-truth GBC(S) values from the C++ exact evaluator.
        sample_weights : Tensor [N_samples] | None
            Per-sample importance weights.  Deployment plan Stage 2D:
            endpoints w=1.0, midpoints w=0.3.
        h_G_batch : Tensor [N_samples, d] | None
            Graph context (if model.use_context=True).

        Returns
        -------
        list[float]
            Loss curve (one value per epoch).
        """
        cfg = self.config
        N = h_S_batch.size(0)
        if N == 0:
            raise ValueError("Need at least one training sample")

        h_S_batch = h_S_batch.to(self.device)
        targets = targets.to(self.device).float()
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device).float()
        if h_G_batch is not None:
            h_G_batch = h_G_batch.to(self.device)

        # ── normalise targets ────────────────────────────────────────────
        if cfg.normalize_targets:
            self.normalizer = RunningNormalizer()
            self.model._normalizer = self.normalizer
            self.normalizer.update(targets)
            targets_n = self.normalizer.normalize(targets)
        else:
            self.model._normalizer = None
            targets_n = targets

        # ── training loop ────────────────────────────────────────────────
        self.model.train()
        best_loss = float("inf")
        patience_counter = 0
        self.train_losses = []

        for epoch in range(1, cfg.epochs + 1):
            # Mini-batch iteration (or full-batch if N ≤ batch_size)
            perm = torch.randperm(N, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, cfg.batch_size):
                end = min(start + cfg.batch_size, N)
                idx = perm[start:end]

                h_S_b = h_S_batch[idx]
                t_b = targets_n[idx]
                h_G_b = h_G_batch[idx] if h_G_batch is not None else None
                w_b = sample_weights[idx] if sample_weights is not None else None

                pred = self.model(h_S_b, h_G_b)  # [B]
                residual = (pred - t_b) ** 2      # [B]

                if w_b is not None:
                    mse_loss = (residual * w_b).mean()
                else:
                    mse_loss = residual.mean()

                # Pairwise RankNet loss — directly optimises ranking quality.
                # Samples K random pairs (i, j) where target_i > target_j and
                # penalises pred_i ≤ pred_j.
                rank_loss = torch.tensor(0.0, device=self.device)
                B = pred.size(0)
                if cfg.lambda_rank > 0 and B >= 2:
                    n_pairs = min(cfg.rank_pairs_per_batch, B * (B - 1) // 2)
                    ri = torch.randint(0, B, (n_pairs,), device=self.device)
                    rj = torch.randint(0, B, (n_pairs,), device=self.device)
                    valid = ri != rj
                    ri, rj = ri[valid], rj[valid]
                    if ri.numel() > 0:
                        diff_t = t_b[ri] - t_b[rj]          # positive = i better
                        diff_p = pred[ri] - pred[rj]
                        rank_loss = torch.nn.functional.softplus(-diff_t.sign() * diff_p).mean()

                loss = mse_loss + cfg.lambda_rank * rank_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            # ── early stopping ───────────────────────────────────────────
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                if cfg.log_every > 0:
                    print(f"  Early stop at epoch {epoch} (best loss={best_loss:.6f})")
                break

            if cfg.log_every > 0 and epoch % cfg.log_every == 0:
                print(f"  Epoch {epoch:4d}  loss={avg_loss:.6f}  best={best_loss:.6f}")

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        self.model.eval()
        return self.train_losses

    def online_update(
        self,
        h_S: Tensor,
        target: float,
        h_G: Optional[Tensor] = None,
        lr_multiplier: float = 0.1,
        n_steps: int = 5,
    ) -> float:
        """Stage 2C: single-sample online fine-tune when exact GBC deviates.

        Performs a few gradient steps on a single (h_S, F_mc) pair using
        a reduced learning rate.  Returns the final per-sample loss.
        """
        self.model.train()
        h_S = h_S.unsqueeze(0).to(self.device)
        h_G_b = h_G.unsqueeze(0).to(self.device) if h_G is not None else None
        t = torch.tensor([target], dtype=torch.float32, device=self.device)

        # FIX #8: Always update normalizer first, then normalize consistently.
        # If not enough data yet, skip update to avoid destabilizing model
        # (model was trained on normalized targets; mixing scales destroys it).
        self.normalizer.update(t)
        if self.normalizer._count >= 2:
            t_n = self.normalizer.normalize(t)
        else:
            # Not enough statistics yet — skip this online update safely
            self.model.eval()
            return 0.0

        lr_orig = self.optimizer.param_groups[0]["lr"]
        self.optimizer.param_groups[0]["lr"] = lr_orig * lr_multiplier

        final_loss = 0.0
        for _ in range(n_steps):
            pred = self.model(h_S, h_G_b)
            loss = ((pred - t_n) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            final_loss = loss.item()

        self.optimizer.param_groups[0]["lr"] = lr_orig
        self.model.eval()
        return final_loss

    def evaluate(
        self,
        h_S_batch: Tensor,
        targets: Tensor,
        h_G_batch: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics (MSE, MAE, R²) in original scale."""
        self.model.eval()
        h_S_batch = h_S_batch.to(self.device)
        targets = targets.to(self.device).float()
        h_G_batch = h_G_batch.to(self.device) if h_G_batch is not None else None

        with torch.no_grad():
            pred_n = self.model(h_S_batch, h_G_batch)

            # Denormalise to original scale
            if self.config.normalize_targets and self.normalizer._count >= 2:
                pred = self.normalizer.denormalize(pred_n)
            else:
                pred = pred_n

            mse = ((pred - targets) ** 2).mean().item()
            mae = (pred - targets).abs().mean().item()

            ss_res = ((targets - pred) ** 2).sum().item()
            ss_tot = ((targets - targets.mean()) ** 2).sum().item()
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"mse": mse, "mae": mae, "r2": r2}

    def save(self, path: str) -> None:
        """Save model + normaliser state."""
        torch.save({
            "model_state": self.model.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "config": {
                "embed_dim": self.model.embed_dim,
                "hidden_dims": self.model.hidden_dims,
                "use_context": self.model.use_context,
            },
        }, path)

    def load(self, path: str) -> None:
        """Load model + normaliser state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.normalizer.load_state_dict(ckpt["normalizer"])
        self.model._normalizer = self.normalizer if self.config.normalize_targets else None
        self.model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Exercise raw-scale predictions, minibatches and checkpoint round-trip."""
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from phase2_trajectory import SeedSet

    torch.manual_seed(42)
    h_v = torch.randn(8, 4)
    sets = [SeedSet({0, 1}), SeedSet({2, 3}), SeedSet({4, 5})]
    pooled = compute_seed_embedding_batch(h_v, sets)
    assert pooled.shape == (3, 4)
    assert torch.allclose(pooled[0], compute_seed_embedding(h_v, sets[0]))
    model = ValueNetwork(embed_dim=4, hidden_dims=(8, 4))
    trainer = ValueNetTrainer(model, TrainConfig(epochs=12, patience=12, log_every=0))
    trainer.fit(pooled, torch.tensor([100., 200., 300.]))
    batch = model.predict_batch(h_v, sets)
    assert len(batch) == 3 and abs(batch[0]) > 10
    assert abs(model.predict(h_v, sets[0]) - batch[0]) < 1e-4
    with TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "value.pt")
        trainer.save(path)
        other = ValueNetTrainer(ValueNetwork(embed_dim=4, hidden_dims=(8, 4)))
        other.load(path)
        assert abs(other.model.predict(h_v, sets[0]) - batch[0]) < 1e-4
    print("value_net.py smoke test: PASS")


if __name__ == "__main__":
    _smoke_test()
