"""Run GEN-GBC Phases 1–4 and inference on a SNAP edge list.

Usage: python train_gim.py --dataset ca-grqc --data-root /content/data \\
           --k 10 --exact-source exact_gbc.cpp --exact-threads 4
Run ``python train_gim.py --smoke-test`` for a small end-to-end integration.
The C++ evaluator is compiled automatically. No group regressor is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from cvae import CVAE, CVAETrainConfig, CVAETrainer, prepare_phase3_data
from dataset_builder import DataSample, run_phase2
from exact_gbc_scorer import ExactGBCScorer, compile_exact_gbc
from gatv2 import Phase1Config
from graph_utils import load_graph, print_graph_stats
from gbc_types import GraphData
from inference import (InferenceResult, InferenceSample,
                       polish_with_restarts, run_direct_inference, run_inference)
from phase1_representation import run_phase1
from phase2_trajectory import SeedSet
from rl_policy import CEMConfig, CEMTrainer
from score_net import ScoreNet, ScoreNetConfig, ScoreNetTrainer, encode_phase2_samples
from value_net import RunningNormalizer, ValueNetwork


@dataclass
class GIMConfig:
    dataset: str = "ca-grqc"
    data_root: str = "data"
    k: int = 10
    exact_source: str = str(Path(__file__).with_name("exact_gbc.cpp"))
    exact_threads: int = 1
    checkpoint_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results"
    device: str = "auto"
    seed: int = 42
    resume: bool = True
    inference_only: bool = False
    p1_hidden: int = 128
    p1_layers: int = 2
    p1_heads: int = 8
    p1_epochs: int = 1000
    p1_lr: float = 5e-4
    p2_H: int = 5
    p2_neighbors: int = 5
    p2_value_epochs: int = 500
    p2_top_trajectories: int = 50
    p2_quality_floor: float = 1.0
    p2_quality_ratio: float = 0.5
    p2_min_random_keep: int = 10
    p2_perturb_sampling: bool = False
    p2_crossover: bool = False
    p2_elite_top_k: int = 5
    p2_elite_weight: float = 4.0
    p3_latent_dim: int = 256
    p3_hidden: int = 512
    p3_iters: int = 3000
    p3_lr: float = 8e-4
    p3_beta_max: float = 1.0
    p3_lambda_vp: float = 0.1
    p3_gamma: float = 0.05
    score_enabled: bool = True
    score_epochs: int = 300
    score_beta: float = 3.0
    score_sigma_min: float = 0.01
    score_sigma_max: float = 1.0
    score_n_sigma_levels: int = 10
    score_lr: float = 3e-4
    # One initial CEM plus two CEM-Refine loops, as in GEN-CIM Table 5.
    cem_runs: int = 3
    cem_pop: int = 60
    cem_iter: int = 25
    cem_elite: float = 0.2
    cem_sigma_init: float = 1.0
    cem_sigma_min: float = 0.05
    cem_sigma_decay: float = 0.95
    cem_sigma_scale: float = 1.0
    langevin_steps: int = 60
    gaussian_topk: int = 20
    inference_samples: int = 200
    inference_shortlist: int = 20
    inf_skip_swap: bool = True
    inf_extra_starts: int = 3
    inf_swap_rounds: int = 2
    inf_perturb_restarts: int = 3
    inf_perturb_kick: int = 1
    variant: str = "full"  # phase2_only | latent_opt | full
    gold_weight: float = 3.0
    gold_quantile: float = 0.9
    gold_threshold_gap: float = 4.0  # GEN-CIM: max(p90, best - gap).
    refine_p3_iters: Optional[int] = None  # GEN-CIM: max(500, p3_iters // 3).
    refine_score_epochs: Optional[int] = None  # Re-train for score_epochs.

    def get_device(self) -> torch.device:
        return torch.device("cuda" if self.device == "auto" and torch.cuda.is_available()
                            else "cpu" if self.device == "auto" else self.device)


def _save(path: Path, payload: dict, fingerprint: str, k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "k": k, "payload": payload}, path)


def _load(path: Path, fingerprint: str, k: int, enabled: bool) -> Optional[dict]:
    if not enabled or not path.is_file():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("fingerprint") != fingerprint or saved.get("k") != k:
        raise ValueError(f"Checkpoint does not match graph, exact GBC or phase settings: {path}")
    return saved["payload"]


def _sample_records(samples: list[DataSample]) -> list[dict]:
    return [dict(nodes=sorted(s.seed_set.nodes), score=s.score, weight=s.weight,
                 is_exact_scored=s.is_exact_scored, trajectory_idx=s.trajectory_idx,
                 step_idx=s.step_idx, strategy=s.strategy) for s in samples]


def _from_records(records: list[dict]) -> list[DataSample]:
    return [DataSample(SeedSet(set(s["nodes"])), s["score"], s["weight"],
                       s["is_exact_scored"], s["trajectory_idx"],
                       s["step_idx"], s["strategy"]) for s in records]


def build_gaussian_proposal(samples: list[DataSample], cvae: CVAE,
                            h_v: Tensor, h_G: Tensor, top_k: int
                            ) -> Optional[Tuple[Tensor, Tensor]]:
    """Fit a diagonal latent Gaussian to top exact GBC-scored Phase 2 sets."""
    verified = sorted((s for s in samples if s.is_exact_scored),
                      key=lambda s: s.score, reverse=True)
    unique = list({frozenset(s.seed_set.nodes): s for s in verified}.values())[:top_k]
    if len(unique) < 2:
        return None
    from value_net import compute_seed_embedding_batch
    with torch.no_grad():
        h_S = compute_seed_embedding_batch(h_v, [s.seed_set for s in unique])
        mu, _ = cvae.encode(h_S, h_G)
    return mu.mean(0), mu.std(0, unbiased=False).clamp_min(0.05)


def _write_result(cfg: GIMConfig, result: InferenceResult,
                  winners: list[dict], loop_log: list[dict], start: float,
                  phase_times: dict[str, float], exact_calls: int,
                  training_time_s: float) -> None:
    output = Path(cfg.results_dir) / (
        f"gbc_{cfg.dataset.lower()}_k{cfg.k}_seed{cfg.seed}_{cfg.variant}_exact.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": cfg.dataset, "k": cfg.k,
        "variant": cfg.variant,
        "score_kind": "exact raw ordered-pair internal-node GBC",
        "best_nodes_internal": sorted(result.best_seed_set.nodes),
        "exact_raw_GBC": result.best_exact, "source": result.source,
        "cem_scores": [w["score"] for w in winners], "loops": loop_log,
        "phase_times_s": phase_times,
        "training_time_s": training_time_s,
        "inference_time_s": phase_times["inference_s"],
        "inference_only": cfg.inference_only,
        "exact_unique_calls": exact_calls,
        "seed": cfg.seed,
        "exact_source": str(cfg.exact_source),
        "elapsed_s": time.perf_counter() - start}, indent=2), encoding="utf-8")
    print(f"[Result/{cfg.variant}] exact raw GBC={result.best_exact:.4f}, "
          f"seeds={sorted(result.best_seed_set.nodes)}; saved {output}")


def _refine_models(cfg: GIMConfig, samples: list[DataSample], cvae: CVAE,
                   value: ValueNetwork, h_v: Tensor, h_G: Tensor,
                   device: torch.device) -> ScoreNet | None:
    """Warm-start CVAE on augmented D_traj and refit ScoreNet on its new latents."""
    data, target, weights = prepare_phase3_data(samples, h_v)
    refine_iters = (cfg.refine_p3_iters if cfg.refine_p3_iters is not None
                    else max(500, cfg.p3_iters // 3))
    # GEN-CIM fine-tunes from the preceding CVAE with 0.3 times the LR.
    CVAETrainer(cvae, CVAETrainConfig(
        lr=cfg.p3_lr * 0.3, max_iterations=refine_iters,
        beta_max=cfg.p3_beta_max, lambda_vp=cfg.p3_lambda_vp,
        gamma=cfg.p3_gamma,
        log_every=max(1, refine_iters // 5),
        warmup_steps=0),
        v_phi=value, device=device).fit(
        data, h_G, h_v, target, sample_weights=weights)
    cvae.eval()
    if not cfg.score_enabled:
        return None
    score_epochs = (cfg.refine_score_epochs if cfg.refine_score_epochs is not None
                    else cfg.score_epochs)
    score_net = ScoreNet(latent_dim=cvae.latent_dim, n_layers=3).to(device)
    ScoreNetTrainer(score_net, encode_phase2_samples(samples, cvae, h_v, h_G),
                    ScoreNetConfig(n_epochs=score_epochs, beta=cfg.score_beta,
                        batch_size=min(256, len(samples) * 4), lr=cfg.score_lr,
                        sigma_min=cfg.score_sigma_min,
                        sigma_max=cfg.score_sigma_max,
                        n_sigma_levels=cfg.score_n_sigma_levels,
                        log_every=max(1, score_epochs // 5)),
                    device=device).fit()
    score_net.eval()
    return score_net


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase-specific state, checkpoints, and training runners
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunContext:
    """Immutable identity of a graph/k/exact GBC experiment and its paths."""
    cfg: GIMConfig
    graph: GraphData
    scorer: ExactGBCScorer
    device: torch.device
    graph_fingerprint: str
    exact_fingerprint: str
    directory: Path

    def checkpoint(self, phase: str) -> Path:
        # Keep corrected multi-noise CEM checkpoints separate from older runs.
        name = ('p1' if phase == 'p1' else 'exact_' + phase)
        if phase == 'p3':
            name += '_cim_source_v3'
        if phase == 'score_net':
            name += '_cim_multiscale_v3'
        if phase.startswith('p4_'):
            name += '_cim_multiscale_v3'
        return self.directory / (
            f"{name}_{self.cfg.dataset.lower()}_k{self.cfg.k}_seed{self.cfg.seed}.pt")

    def fingerprint_for(self, phase: str) -> str:
        """Bind cached outputs to their upstream model and training settings."""
        cfg = self.cfg
        if phase == "p1":
            identity = {"graph": self.graph_fingerprint, "seed": cfg.seed,
                "config": [cfg.p1_hidden, cfg.p1_layers, cfg.p1_heads,
                           cfg.p1_epochs, cfg.p1_lr]}
        elif phase == "p2":
            identity = {"upstream": self.fingerprint_for("p1"),
                "exact": self.exact_fingerprint,
                "config": [cfg.p2_H, cfg.p2_neighbors, cfg.p2_value_epochs,
                    cfg.p2_top_trajectories, cfg.p2_quality_floor,
                    cfg.p2_quality_ratio, cfg.p2_min_random_keep,
                    cfg.p2_perturb_sampling, cfg.p2_crossover,
                    cfg.p2_elite_top_k, cfg.p2_elite_weight]}
        elif phase == "p3":
            identity = {"upstream": self.fingerprint_for("p2"),
                "algorithm": "cim_source_subsampled_cvae_v3",
                "config": [cfg.p3_latent_dim, cfg.p3_hidden,
                           cfg.p3_iters, cfg.p3_lr, cfg.p3_beta_max,
                           cfg.p3_lambda_vp, cfg.p3_gamma]}
        elif phase == "score_net":
            identity = {"upstream": self.fingerprint_for("p3"),
                "config": [cfg.score_enabled, cfg.score_epochs,
                           cfg.score_beta, cfg.score_sigma_min,
                           cfg.score_sigma_max, cfg.score_n_sigma_levels,
                           cfg.score_lr]}
        elif phase.startswith("p4_"):
            upstream = self.fingerprint_for(
                "score_net" if cfg.score_enabled else "p3")
            identity = {"upstream": upstream, "variant": cfg.variant,
                "algorithm": "cim_multiscale_cem_refine_v3",
                "config": [cfg.cem_runs, cfg.cem_pop, cfg.cem_iter, cfg.cem_elite,
                    cfg.cem_sigma_init, cfg.cem_sigma_min, cfg.cem_sigma_decay,
                    cfg.cem_sigma_scale,
                    cfg.langevin_steps, cfg.gaussian_topk, cfg.gold_weight,
                    cfg.gold_quantile, cfg.gold_threshold_gap,
                    cfg.refine_p3_iters, cfg.refine_score_epochs]}
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def load(self, phase: str) -> Optional[dict]:
        path = self.checkpoint(phase)
        fingerprint = self.fingerprint_for(phase)
        result = _load(path, fingerprint, self.cfg.k,
                       self.cfg.resume or self.cfg.inference_only)
        if result is None and self.cfg.inference_only:
            raise FileNotFoundError(f"Inference-only requires {path}")
        return result

    def save(self, phase: str, payload: dict) -> None:
        fingerprint = self.fingerprint_for(phase)
        path = self.checkpoint(phase)
        _save(path, payload, fingerprint, self.cfg.k)
        print(f"[Checkpoint/{phase}] Saved {path}")


@dataclass
class Phase2State:
    """Data flowing from Phase 2 to the latent models and inference."""
    samples: list[DataSample]
    h_G: Tensor
    value: ValueNetwork


def _banner(title: str) -> None:
    print("\n" + "=" * 64)
    print("  " + title)
    print("=" * 64)


def phase1(ctx: RunContext) -> tuple[Tensor, Optional[Tensor]]:
    """Learn embeddings and retain singleton GBC labels for Phase 2 experts."""
    _banner("PHASE 1: Exact-label GATv2 representation")
    cfg = ctx.cfg
    saved = ctx.load("p1")
    if saved is None:
        features = Phase1Config(hidden_channels=cfg.p1_hidden,
            num_layers=cfg.p1_layers, heads=cfg.p1_heads, n_epochs=cfg.p1_epochs,
            lr=cfg.p1_lr, log_every=max(1, cfg.p1_epochs // 5))
        h_v, node_gbc = run_phase1(ctx.graph, config=features, device=ctx.device,
            checkpoint_dir=ctx.directory,
            dataset_name=f"{cfg.dataset.lower()}_k{cfg.k}_seed{cfg.seed}",
            return_labels=True)
        ctx.save("p1", {"h_v": h_v.detach().cpu(),
            "node_gbc": node_gbc.detach().cpu(),
            "architecture": {"hidden": cfg.p1_hidden, "layers": cfg.p1_layers,
                             "heads": cfg.p1_heads}})
    else:
        architecture = saved.get("architecture", {})
        if architecture and architecture["hidden"] != cfg.p1_hidden:
            raise ValueError("Phase 1 checkpoint has a different embedding width")
        h_v = saved["h_v"].to(ctx.device)
        # Older Phase 1 checkpoints contain only h_v; Phase 2 will calculate
        # singleton labels once if it must build a new trajectory checkpoint.
        node_gbc = saved.get("node_gbc")
        print(f"[Phase1] Resumed embeddings {tuple(h_v.shape)}")
    if h_v.shape != (ctx.graph.num_nodes, cfg.p1_hidden):
        raise ValueError("Phase 1 checkpoint shape does not match graph/config")
    if node_gbc is not None and node_gbc.shape != (ctx.graph.num_nodes,):
        raise ValueError("Phase 1 checkpoint singleton labels do not match graph")
    return h_v, node_gbc


def phase2(ctx: RunContext, h_v: Tensor,
           node_gbc: Optional[Tensor] = None) -> Phase2State:
    """Build Stage A–D trajectories, label endpoints by exact GBC."""
    _banner("PHASE 2: Expert trajectories and exact GBC labels")
    cfg = ctx.cfg
    saved = ctx.load("p2")
    if saved is None:
        output = run_phase2(ctx.graph, h_v, ctx.scorer, cfg.k,
            node_gbc=node_gbc,
            H=cfg.p2_H, k_neighbors=cfg.p2_neighbors,
            random_seed=cfg.seed, value_epochs=cfg.p2_value_epochs,
            quality_floor=cfg.p2_quality_floor,
            quality_ratio=cfg.p2_quality_ratio,
            min_random_keep=cfg.p2_min_random_keep,
            top_trajectories=cfg.p2_top_trajectories,
            perturb_sampling=cfg.p2_perturb_sampling,
            crossover=cfg.p2_crossover,
            elite_top_k=cfg.p2_elite_top_k,
            elite_weight=cfg.p2_elite_weight)
        state = Phase2State(output.samples, output.h_G, output.value_net)
        value = state.value
        ctx.save("p2", {"samples": _sample_records(state.samples),
            "h_G": state.h_G.detach().cpu(),
            "value_state": value.state_dict(),
            "value_config": {"embed_dim": value.embed_dim,
                             "hidden_dims": value.hidden_dims,
                             "use_context": value.use_context},
            "normalizer": value._normalizer.state_dict() if value._normalizer else None})
    else:
        value = ValueNetwork(**saved["value_config"]).to(ctx.device)
        value.load_state_dict(saved["value_state"])
        if saved["normalizer"] is not None:
            value._normalizer = RunningNormalizer()
            value._normalizer.load_state_dict(saved["normalizer"])
        state = Phase2State(_from_records(saved["samples"]),
                            saved["h_G"].to(ctx.device), value)
        print(f"[Phase2] Resumed {len(state.samples)} trajectory samples")
    state.value.eval()
    verified = [sample for sample in state.samples if sample.is_exact_scored]
    if not verified or state.h_G.shape != (h_v.size(1),):
        raise ValueError("Phase 2 lacks exact GBC endpoints or has invalid h_G")
    print(f"[Phase2] samples={len(state.samples)}, exact GBC endpoints={len(verified)}, "
          f"best exact GBC={max(item.score for item in verified):.6f}")
    return state


def phase3(ctx: RunContext, state: Phase2State, h_v: Tensor) -> CVAE:
    """Fit quality-weighted CVAE on D_traj and restore on resume."""
    _banner("PHASE 3: CVAE latent distribution")
    cfg = ctx.cfg
    saved = ctx.load("p3")
    if saved is None:
        cvae = CVAE(embed_dim=h_v.size(1), latent_dim=cfg.p3_latent_dim,
                    hidden_dim=cfg.p3_hidden).to(ctx.device)
        h_S, targets, weights = prepare_phase3_data(state.samples, h_v)
        trainer = CVAETrainer(cvae, CVAETrainConfig(lr=cfg.p3_lr,
            max_iterations=cfg.p3_iters, warmup_steps=max(1, cfg.p3_iters // 5),
            beta_max=cfg.p3_beta_max, lambda_vp=cfg.p3_lambda_vp,
            gamma=cfg.p3_gamma,
            use_one_cycle=cfg.p3_iters >= 10,
            log_every=max(1, cfg.p3_iters // 5)),
            v_phi=state.value, device=ctx.device)
        trainer.fit(h_S, state.h_G, h_v, targets, sample_weights=weights)
        ctx.save("p3", {"state": cvae.state_dict(),
            "embed_dim": cvae.embed_dim, "latent_dim": cvae.latent_dim,
            "hidden_dim": cvae.hidden_dim})
    else:
        cvae = CVAE(saved["embed_dim"], saved["latent_dim"],
                    saved["hidden_dim"]).to(ctx.device)
        cvae.load_state_dict(saved["state"])
        print(f"[Phase3] Resumed CVAE latent_dim={cvae.latent_dim}")
    if cvae.embed_dim != h_v.size(1):
        raise ValueError("CVAE checkpoint embedding width disagrees with Phase 1")
    cvae.eval()
    return cvae


def phase3_5(ctx: RunContext, state: Phase2State, h_v: Tensor,
             cvae: CVAE) -> Optional[ScoreNet]:
    """Train GEN-CIM's multi-noise ScoreNet on CVAE latents."""
    if not ctx.cfg.score_enabled:
        print("[Phase3.5] Disabled by configuration")
        return None
    _banner("PHASE 3.5: Importance-weighted ScoreNet")
    cfg = ctx.cfg
    saved = ctx.load("score_net")
    if saved is None:
        net = ScoreNet(latent_dim=cvae.latent_dim, n_layers=3).to(ctx.device)
        encoded = encode_phase2_samples(state.samples, cvae, h_v, state.h_G)
        print(f"[Phase3.5] encoded={len(encoded)}, "
              f"sigma=[{cfg.score_sigma_min}, {cfg.score_sigma_max}], "
              f"epochs={cfg.score_epochs}")
        ScoreNetTrainer(net, encoded, ScoreNetConfig(
            n_epochs=cfg.score_epochs, lr=cfg.score_lr, beta=cfg.score_beta,
            batch_size=min(256, len(encoded) * 4),
            sigma_min=cfg.score_sigma_min, sigma_max=cfg.score_sigma_max,
            n_sigma_levels=cfg.score_n_sigma_levels,
            log_every=max(1, cfg.score_epochs // 5)), device=ctx.device).fit()
        ctx.save("score_net", {"state": net.state_dict(),
            "latent_dim": net.latent_dim, "hidden_dim": net.hidden_dim,
            "n_layers": sum(isinstance(m, torch.nn.Linear) for m in net.net),
            "sigma_min": cfg.score_sigma_min, "sigma_max": cfg.score_sigma_max,
            "n_sigma_levels": cfg.score_n_sigma_levels})
    else:
        if (saved["sigma_min"], saved["sigma_max"], saved["n_sigma_levels"]) != (
                cfg.score_sigma_min, cfg.score_sigma_max, cfg.score_n_sigma_levels):
            raise ValueError("ScoreNet checkpoint was trained at another noise range")
        net = ScoreNet(saved["latent_dim"], saved["hidden_dim"],
                       saved["n_layers"]).to(ctx.device)
        net.load_state_dict(saved["state"])
        print("[Phase3.5] Resumed multi-noise ScoreNet")
    if net.latent_dim != cvae.latent_dim:
        raise ValueError("ScoreNet latent width disagrees with CVAE")
    net.eval()
    return net


def phase4(ctx: RunContext, state: Phase2State, h_v: Tensor,
           cvae: CVAE, score_net: Optional[ScoreNet]
           ) -> tuple[list[dict], list[dict], list[DataSample]]:
    """Run initial CEM and optionally two CEM-Refine loops.

    Phase-2 Only bypasses this function. Latent Opt runs loop 0 only; Full
    uses cfg.cem_runs total fits (three by default), retaining the global
    best even when a later fit regresses. Between fits, CVAE is fine-tuned
    and ScoreNet is retrained whether or not a new gold set qualifies, as
    in GEN-CIM's training loop. All scores come from the *same* exact C++
    evaluator, including gold verification.
    """
    _banner("PHASE 4: Score-guided CEM and CEM-Refine")
    cfg = ctx.cfg
    n_runs = 1 if cfg.variant == "latent_opt" else cfg.cem_runs
    settings = {"runs": n_runs, "population": cfg.cem_pop,
        "iterations": cfg.cem_iter, "elite": cfg.cem_elite,
        "sigma_init": cfg.cem_sigma_init, "sigma_min": cfg.cem_sigma_min,
        "sigma_decay": cfg.cem_sigma_decay, "sigma_scale": cfg.cem_sigma_scale,
        "langevin_steps": cfg.langevin_steps,
        "score_enabled": cfg.score_enabled,
        "score_sigma_min": cfg.score_sigma_min,
        "score_sigma_max": cfg.score_sigma_max,
        "gold_weight": cfg.gold_weight, "gold_quantile": cfg.gold_quantile,
        "gold_threshold_gap": cfg.gold_threshold_gap,
        "refine_p3_iters": cfg.refine_p3_iters,
        "refine_score_epochs": cfg.refine_score_epochs}
    saved = ctx.load(f"p4_{cfg.variant}")
    if saved is not None:
        if saved.get("settings") != settings:
            raise ValueError("Phase 4 checkpoint settings changed; use --no-resume")
        cvae.load_state_dict(saved["final_cvae_state"])
        cvae.eval()
        if score_net is not None and saved["final_score_state"] is not None:
            score_net.load_state_dict(saved["final_score_state"])
            score_net.eval()
        print(f"[Phase4] Resumed {len(saved['winners'])} CEM fits")
        return saved["winners"], saved["loops"], _from_records(saved["samples"])

    samples = list(state.samples)
    proposal = build_gaussian_proposal(samples, cvae, h_v, state.h_G,
                                       cfg.gaussian_topk)
    winners: list[dict] = []
    loop_log: list[dict] = []
    for loop in range(n_runs):
        print(f"[Phase4] Fit {loop + 1}/{n_runs}, "
              f"D_traj={len(samples)}, ScoreNet={'on' if score_net else 'off'}")
        cem = CEMTrainer(cvae, h_v, state.h_G, cfg.k, ctx.scorer,
            CEMConfig(pop_size=cfg.cem_pop, elite_frac=cfg.cem_elite,
                n_iter=cfg.cem_iter, sigma_init=cfg.cem_sigma_init,
                sigma_min=cfg.cem_sigma_min, sigma_decay=cfg.cem_sigma_decay,
                sigma_scale=cfg.cem_sigma_scale,
                langevin_steps=cfg.langevin_steps,
                verbose=True), score_net=score_net)
        # Identical loop-0 search for Latent Opt and Full, independent of
        # whether earlier phases were trained here or resumed.
        with torch.random.fork_rng(devices=[ctx.device.index or 0]
                                   if ctx.device.type == "cuda" else []):
            torch.manual_seed(cfg.seed + 13_000 + loop)
            z, S, score = cem.fit(gaussian_proposal=proposal)
        winners.append({"z": z.cpu(), "nodes": sorted(S.nodes),
                        "score": score, "history": cem.history})
        global_best = max(winners, key=lambda item: item["score"])
        log = {"loop": loop, "cem_GBC": score,
               "global_best_GBC": global_best["score"],
               "injected": False, "refined": False,
               "candidate_count": len(samples)}
        if loop < n_runs - 1:
            measured = torch.tensor([s.score for s in samples if s.is_exact_scored])
            best_existing = float(measured.max())
            threshold = max(float(torch.quantile(measured, cfg.gold_quantile)),
                            best_existing - cfg.gold_threshold_gap)
            gold = SeedSet(set(global_best["nodes"]))
            # CEM already used the exact scorer: unlike MC, re-verification
            # returns precisely the same value. Keep its threshold and weight.
            verified = global_best["score"]
            log.update(threshold=threshold, verified_GBC=verified)
            if verified >= threshold:
                samples.append(DataSample(gold, verified, cfg.gold_weight,
                    True, len(samples), 0, f"gold_cem_loop{loop + 1}"))
                log.update(injected=True, candidate_count=len(samples))
                print(f"[CEM-Refine] Injected gold GBC={verified:.6f}")
            else:
                print(f"[CEM-Refine] No qualified seed set "
                      f"(score={verified:.6f}, threshold={threshold:.6f})")
            score_net = _refine_models(cfg, samples, cvae, state.value,
                                       h_v, state.h_G, ctx.device)
            proposal = (build_gaussian_proposal(samples, cvae, h_v,
                                                state.h_G, cfg.gaussian_topk)
                        if score_net is None else None)
            log["refined"] = True
            print("[CEM-Refine] Fine-tuned CVAE and retrained ScoreNet "
                  "before the next CEM fit")
        loop_log.append(log)
    ctx.save(f"p4_{cfg.variant}", {"settings": settings, "winners": winners,
        "loops": loop_log, "final_cvae_state": cvae.state_dict(),
        "final_score_state": score_net.state_dict() if score_net else None,
        "samples": _sample_records(samples)})
    return winners, loop_log, samples


def run_inference_and_evaluate(ctx: RunContext, state: Phase2State,
                               h_v: Tensor, cvae: CVAE, winners: list[dict],
                               samples: list[DataSample]) -> InferenceResult:
    """Follow GEN-CIM's default direct CEM/D_traj selection with exact GBC."""
    _banner("INFERENCE: CEM and D_traj direct selection")
    cfg = ctx.cfg
    protected = [SeedSet(set(w["nodes"])) for w in winners]
    names = [f"CEM_{i + 1}" for i in range(len(winners))]
    verified = sorted((s for s in samples if s.is_exact_scored),
                      key=lambda s: s.score, reverse=True)
    seen = {frozenset(s.nodes) for s in protected}
    for sample in verified:
        if len(names) >= len(winners) + cfg.inf_extra_starts:
            break
        key = frozenset(sample.seed_set.nodes)
        if key in seen:
            continue
        seen.add(key)
        protected.append(sample.seed_set)
        names.append("D_traj")
    best_cem = max(winners, key=lambda w: w["score"])
    if cfg.inf_skip_swap:
        result = run_direct_inference(state.value, ctx.scorer, h_v,
                                      state.h_G, protected, names)
        print(f"[Inference] source={result.source}, "
              f"finalists={len(result.all_samples)}, "
              f"exact_raw_GBC={result.best_exact:.6f}")
        return result
    with torch.random.fork_rng(devices=[ctx.device.index or 0]
                                   if ctx.device.type == "cuda" else []):
        torch.manual_seed(cfg.seed + 1_000_003)
        result = run_inference(cvae, state.value, ctx.scorer, h_v,
            state.h_G, cfg.k, n_samples=cfg.inference_samples,
            shortlist=cfg.inference_shortlist,
            z_init=best_cem["z"].to(ctx.device),
            protected=protected, protected_names=names)
    # GEN-CIM's optional full inference adds greedy swap and perturbation
    # restarts after the proxy shortlist. All trial scores use exact GBC.
    t0 = time.perf_counter()
    polished = polish_with_restarts(result.best_seed_set, ctx.scorer,
        extra_starts=protected, n_rounds=cfg.inf_swap_rounds,
        restarts=cfg.inf_perturb_restarts, kick=cfg.inf_perturb_kick,
        seed=cfg.seed)
    polished_score = ctx.scorer.score(polished)
    if polished_score > result.best_exact + 1e-4:
        proxy = state.value.predict(h_v, polished, state.h_G)
        row = InferenceSample(polished, proxy, polished_score, "swap_polish")
        result = InferenceResult(polished, polished_score, proxy, row.source,
                                 [row] + result.all_samples,
                                 result.elapsed_s + time.perf_counter() - t0)
    print(f"[Inference] source={result.source}, finalists={len(result.all_samples)}, "
          f"exact_raw_GBC={result.best_exact:.6f}")
    return result


def main(cfg: GIMConfig) -> InferenceResult:
    """Train/resume GEN-GBC, then return the best exact GBC-ranked k-set.

    ``inference_only`` requires all checkpoints needed by the chosen variant.
    For paper-style ablations, run_ablation.py calls this for each variant and
    evaluates all resulting seed sets with one exact GBC evaluator call.
    """
    if cfg.variant not in ("phase2_only", "latent_opt", "full"):
        raise ValueError("variant must be phase2_only, latent_opt, or full")
    if (cfg.k < 1 or cfg.cem_runs < 1 or cfg.cem_pop < 2 or cfg.cem_iter < 1
            or cfg.p1_hidden < 1 or cfg.p1_hidden % cfg.p1_heads
            or cfg.p1_epochs < 1 or cfg.p2_value_epochs < 1
            or cfg.p3_iters < 1 or cfg.score_epochs < 1
            or cfg.p2_elite_top_k < 0 or cfg.p2_elite_weight <= 0
            or cfg.p3_beta_max < 0 or cfg.p3_lambda_vp < 0 or cfg.p3_gamma < 0
            or cfg.cem_sigma_scale <= 0):
        raise ValueError("Invalid graph budget or training iterations")
    if (not 0 < cfg.gold_quantile < 1 or cfg.gold_threshold_gap < 0
            or cfg.gold_weight <= 0 or (cfg.refine_p3_iters is not None
            and cfg.refine_p3_iters < 1)
            or (cfg.refine_score_epochs is not None and cfg.refine_score_epochs < 1)):
        raise ValueError("Invalid CEM-Refine threshold, weight, or budget")
    if (cfg.score_sigma_min <= 0 or cfg.score_sigma_max < cfg.score_sigma_min
            or cfg.score_n_sigma_levels < 1 or cfg.inf_extra_starts < 0
            or cfg.inf_swap_rounds < 0 or cfg.inf_perturb_restarts < 0
            or cfg.inf_perturb_kick < 1):
        raise ValueError("Invalid ScoreNet noise range or inference starts")
    start = time.perf_counter()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = cfg.get_device()
    graph_path = Path(cfg.data_root) / f"{cfg.dataset.lower()}.txt"
    fingerprint = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if cfg.exact_threads < 1:
        raise ValueError("--exact-threads must be positive")
    _banner(f"GEN-GBC dataset={cfg.dataset}, k={cfg.k}, variant={cfg.variant}")
    print(f"[Run] device={device}, seed={cfg.seed}, resume={cfg.resume}, "
          f"inference_only={cfg.inference_only}")
    graph = load_graph(cfg.dataset, root=cfg.data_root, directed=False)
    print_graph_stats(graph)
    if cfg.k > graph.num_nodes:
        raise ValueError("k exceeds the graph's node count")
    directory = Path(cfg.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(cfg.exact_source)
    binary = compile_exact_gbc(source, directory / "exact_gbc_binary")
    exact_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    scorer = ExactGBCScorer(graph, graph_path, binary, cfg.k, cfg.exact_threads)
    ctx = RunContext(cfg, graph, scorer, device, fingerprint,
                     f"exact-cpp-v1:{fingerprint}:{exact_hash}", directory)
    print(f"[Exact GBC] binary={binary}, threads={cfg.exact_threads}, "
          f"graph_sha256={fingerprint[:12]}, source_sha256={exact_hash[:12]}")
    times: dict[str, float] = {}

    def timed(name: str, action: Callable):
        t0 = time.perf_counter()
        result = action()
        times[name] = time.perf_counter() - t0
        return result

    h_v, node_gbc = timed("phase1_s", lambda: phase1(ctx))
    state = timed("phase2_s", lambda: phase2(ctx, h_v, node_gbc))
    verified = [s for s in state.samples if s.is_exact_scored]

    if cfg.variant == "phase2_only":
        training_time_s = time.perf_counter() - start
        _banner("INFERENCE: Direct D_traj top-k")
        t0 = time.perf_counter()
        winner = max(verified, key=lambda s: s.score).seed_set
        exact_score = scorer.score(winner)
        result = InferenceResult(winner, exact_score,
            state.value.predict(h_v, winner, state.h_G), "Phase2", [],
            time.perf_counter() - t0)
        times.update(phase3_s=0.0, score_net_s=0.0, phase4_s=0.0,
                     inference_s=time.perf_counter() - t0)
        _write_result(cfg, result, [], [], start, times, scorer.calls,
                      training_time_s)
        print(f"[Timing] training={training_time_s:.2f}s, "
              f"inference={times['inference_s']:.2f}s")
        return result

    cvae = timed("phase3_s", lambda: phase3(ctx, state, h_v))
    score_net = timed("score_net_s", lambda: phase3_5(ctx, state, h_v, cvae))
    winners, loop_log, samples = timed("phase4_s", lambda:
        phase4(ctx, state, h_v, cvae, score_net))
    training_time_s = time.perf_counter() - start
    result = timed("inference_s", lambda:
        run_inference_and_evaluate(ctx, state, h_v, cvae, winners, samples))
    _write_result(cfg, result, winners, loop_log, start, times, scorer.calls,
                  training_time_s)
    print(f"[Timing] training={training_time_s:.2f}s, "
          f"inference={times['inference_s']:.2f}s; " +
          ", ".join(f"{name}={seconds:.2f}s" for name, seconds in times.items()))
    return result


def _smoke_test() -> None:
    """Exercise actual Phase 1–4 modules, checkpoint resume, and inference."""
    from unittest.mock import patch
    from phase1_representation import compute_node_betweenness_labels
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "tiny.txt"
        data.write_text("0 1\n1 2\n2 3\n3 4\n4 5\n5 0\n0 3\n", encoding="utf-8")
        graph = load_graph("tiny", root=root)
        cfg = GIMConfig(dataset="tiny", data_root=str(root), k=2,
            checkpoint_dir=str(root / "ckpt"),
            results_dir=str(root / "results"), device="cpu", p1_hidden=8,
            p1_heads=2, p1_epochs=1, p2_H=1, p2_neighbors=2,
            p2_value_epochs=1, p3_latent_dim=4, p3_hidden=16, p3_iters=2,
            score_epochs=1, cem_runs=3, cem_pop=4, cem_iter=2,
            langevin_steps=1, inference_samples=3, inference_shortlist=2,
            refine_p3_iters=2, refine_score_epochs=1)
        with patch("phase1_representation.compute_node_betweenness_labels",
                   wraps=compute_node_betweenness_labels) as brandes:
            phase2 = main(replace(cfg, variant="phase2_only"))
            assert brandes.call_count == 1  # Phase 2 reuses Phase 1 labels.
        latent = main(replace(cfg, variant="latent_opt"))
        with patch(__name__ + "._refine_models", wraps=_refine_models) as refinements:
            first = main(cfg)
            assert refinements.call_count == cfg.cem_runs - 1
        second = main(cfg)
        cached_only = main(replace(cfg, inference_only=True))
        assert all(result.best_seed_set.k == 2 for result in (phase2, latent, first, second))
        assert abs(first.best_exact - second.best_exact) < 1e-5
        assert abs(first.best_exact - cached_only.best_exact) < 1e-5
        assert (root / "results" / "gbc_tiny_k2_seed42_full_exact.json").exists()
        full_result = json.loads((root / "results" /
            "gbc_tiny_k2_seed42_full_exact.json").read_text())
        phase2_result = json.loads((root / "results" /
            "gbc_tiny_k2_seed42_phase2_only_exact.json").read_text())
        assert len(full_result["loops"]) == 3
        assert all(item["verified_GBC"] >= 0 for item in full_result["loops"][:-1])
        assert [item["refined"] for item in full_result["loops"]] == [True, True, False]
        for record in (full_result, phase2_result):
            assert record["training_time_s"] >= 0
            assert record["inference_time_s"] >= 0
            assert record["inference_time_s"] == record["phase_times_s"]["inference_s"]
        assert phase2_result["phase_times_s"]["phase3_s"] == 0.0
        try:
            main(replace(cfg, inference_only=True, p3_hidden=32))
        except ValueError as error:
            assert "Checkpoint does not match" in str(error)
        else:
            raise AssertionError("Incompatible CVAE checkpoint was accepted")
        try:
            main(replace(cfg, inference_only=True,
                         checkpoint_dir=str(root / "missing_ckpt")))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Inference-only trained without checkpoints")
        # Force the model-refit path even when a random toy CEM finds no gold.
        h_v = torch.randn(6, 8)
        cvae = CVAE(8, 4, 16)
        before = {key: tensor.clone() for key, tensor in cvae.state_dict().items()}
        gold = [DataSample(SeedSet({0, 1}), 2., 1., True, 0, 0),
                DataSample(SeedSet({0, 2}), 3., 1., True, 0, 1),
                DataSample(SeedSet({2, 3}), 4., 3., True, 1, 0)]
        guided = _refine_models(cfg, gold, cvae, ValueNetwork(8), h_v, h_v.mean(0),
                                torch.device("cpu"))
        assert guided is not None and any(not torch.equal(before[key], value)
                                           for key, value in cvae.state_dict().items())
    print("train_gim.py smoke test: PASS")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI and smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN-GBC: Phase 1–4 with exact GBC")
    data = p.add_argument_group("Graph and experiment")
    data.add_argument("--dataset", default="ca-grqc", help="Load DATA_ROOT/DATASET.txt")
    data.add_argument("--data-root", default="data")
    data.add_argument("--k", type=int, default=10, help="Fixed group size")
    data.add_argument("--exact-source", default=str(Path(__file__).with_name("exact_gbc.cpp")),
                      help="Path to the supplied exact_gbc.cpp")
    data.add_argument("--exact-threads", type=int, default=1)
    data.add_argument("--checkpoint-dir", default="experiments/checkpoints")
    data.add_argument("--results-dir", default="experiments/results")
    data.add_argument("--device", default="auto", help="auto, cpu or cuda")
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--variant", choices=["phase2_only", "latent_opt", "full"],
                      default="full")
    data.add_argument("--no-resume", action="store_true",
                      help="Train from scratch and overwrite this experiment's checkpoints")
    data.add_argument("--inference-only", action="store_true",
                      help="Require existing phase checkpoints; never train")

    p1 = p.add_argument_group("Phase 1: singleton GBC labels and GATv2")
    p1.add_argument("--p1-hidden", type=int, default=128)
    p1.add_argument("--p1-layers", type=int, default=2)
    p1.add_argument("--p1-heads", type=int, default=8)
    p1.add_argument("--p1-epochs", type=int, default=1000)
    p1.add_argument("--p1-lr", type=float, default=5e-4)

    p2 = p.add_argument_group("Phase 2: expert trajectories")
    p2.add_argument("--p2-h", type=int, default=5)
    p2.add_argument("--p2-neighbors", type=int, default=5)
    p2.add_argument("--p2-value-epochs", type=int, default=500)
    p2.add_argument("--p2-top-trajectories", type=int, default=50)
    p2.add_argument("--p2-quality-floor", type=float, default=1.0)
    p2.add_argument("--p2-quality-ratio", type=float, default=0.5)
    p2.add_argument("--p2-min-random-keep", type=int, default=10)
    p2.add_argument("--p2-perturb-sampling", action="store_true")
    p2.add_argument("--p2-crossover", action="store_true")
    p2.add_argument("--p2-elite-top-k", type=int, default=5)
    p2.add_argument("--p2-elite-weight", type=float, default=4.0)

    p3 = p.add_argument_group("Phase 3: CVAE and ScoreNet")
    p3.add_argument("--p3-latent-dim", type=int, default=256)
    p3.add_argument("--p3-hidden", type=int, default=512)
    p3.add_argument("--p3-iters", type=int, default=3000)
    p3.add_argument("--p3-lr", type=float, default=8e-4)
    p3.add_argument("--p3-beta-max", type=float, default=1.0)
    p3.add_argument("--p3-lambda-vp", type=float, default=0.1)
    p3.add_argument("--p3-gamma", type=float, default=0.05)
    p3.add_argument("--no-score-net", action="store_true")
    p3.add_argument("--score-epochs", type=int, default=300)
    p3.add_argument("--score-beta", type=float, default=3.0)
    p3.add_argument("--score-sigma-min", type=float, default=0.01)
    p3.add_argument("--score-sigma-max", type=float, default=1.0)
    p3.add_argument("--score-n-sigma-levels", type=int, default=10)
    p3.add_argument("--score-lr", type=float, default=3e-4)

    p4 = p.add_argument_group("Phase 4: CEM and CEM-Refine")
    p4.add_argument("--n-loops", type=int, default=2,
                    help="Full variant's extra CEM-Refine loops (default: 2)")
    p4.add_argument("--cem-runs", type=int,
                    help="Legacy total CEM fits; overrides --n-loops")
    p4.add_argument("--cem-pop", type=int, default=60)
    p4.add_argument("--cem-iter", type=int, default=25)
    p4.add_argument("--cem-elite", type=float, default=0.2)
    p4.add_argument("--cem-sigma-init", type=float, default=1.0)
    p4.add_argument("--cem-sigma-min", type=float, default=0.05)
    p4.add_argument("--cem-sigma-decay", type=float, default=0.95)
    p4.add_argument("--cem-sigma-scale", type=float, default=1.0)
    p4.add_argument("--langevin-steps", type=int, default=60)
    p4.add_argument("--gaussian-topk", type=int, default=20)
    p4.add_argument("--gold-weight", type=float, default=3.0)
    p4.add_argument("--gold-quantile", type=float, default=0.9)
    p4.add_argument("--gold-threshold-gap", type=float, default=4.0)
    p4.add_argument("--refine-p3-iters", type=int,
                    help="Default: max(500, p3_iters // 3), as in GEN-CIM")
    p4.add_argument("--refine-score-epochs", type=int,
                    help="Default: retrain for score_epochs, as in GEN-CIM")

    inference = p.add_argument_group("Inference")
    inference.add_argument("--inference-samples", type=int, default=200)
    inference.add_argument("--inference-shortlist", type=int, default=20)
    inference.add_argument("--inf-full-search", action="store_true",
                           help="Use CVAE sampling instead of GEN-CIM's default direct CEM/D_traj selection")
    inference.add_argument("--inf-extra-starts", type=int, default=3)
    inference.add_argument("--inf-swap-rounds", type=int, default=2)
    inference.add_argument("--inf-perturb-restarts", type=int, default=3)
    inference.add_argument("--inf-perturb-kick", type=int, default=1)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> GIMConfig:
    if args.inference_only and args.no_resume:
        raise ValueError("--inference-only and --no-resume cannot be combined")
    if args.n_loops < 0:
        raise ValueError("--n-loops must be nonnegative")
    return GIMConfig(
        dataset=args.dataset, data_root=args.data_root, k=args.k,
        exact_source=args.exact_source, exact_threads=args.exact_threads,
        checkpoint_dir=args.checkpoint_dir, results_dir=args.results_dir,
        device=args.device, seed=args.seed, resume=not args.no_resume,
        inference_only=args.inference_only, variant=args.variant,
        p1_hidden=args.p1_hidden, p1_layers=args.p1_layers,
        p1_heads=args.p1_heads, p1_epochs=args.p1_epochs, p1_lr=args.p1_lr,
        p2_H=args.p2_h, p2_neighbors=args.p2_neighbors,
        p2_value_epochs=args.p2_value_epochs,
        p2_top_trajectories=args.p2_top_trajectories,
        p2_quality_floor=args.p2_quality_floor,
        p2_quality_ratio=args.p2_quality_ratio,
        p2_min_random_keep=args.p2_min_random_keep,
        p2_perturb_sampling=args.p2_perturb_sampling,
        p2_crossover=args.p2_crossover,
        p2_elite_top_k=args.p2_elite_top_k,
        p2_elite_weight=args.p2_elite_weight,
        p3_latent_dim=args.p3_latent_dim, p3_hidden=args.p3_hidden,
        p3_iters=args.p3_iters, p3_lr=args.p3_lr,
        p3_beta_max=args.p3_beta_max, p3_lambda_vp=args.p3_lambda_vp,
        p3_gamma=args.p3_gamma,
        score_enabled=not args.no_score_net, score_epochs=args.score_epochs,
        score_beta=args.score_beta, score_sigma_min=args.score_sigma_min,
        score_sigma_max=args.score_sigma_max,
        score_n_sigma_levels=args.score_n_sigma_levels,
        score_lr=args.score_lr,
        cem_runs=args.cem_runs if args.cem_runs is not None else args.n_loops + 1,
        cem_pop=args.cem_pop, cem_iter=args.cem_iter,
        cem_elite=args.cem_elite, cem_sigma_init=args.cem_sigma_init,
        cem_sigma_min=args.cem_sigma_min, cem_sigma_decay=args.cem_sigma_decay,
        cem_sigma_scale=args.cem_sigma_scale,
        langevin_steps=args.langevin_steps, gaussian_topk=args.gaussian_topk,
        gold_weight=args.gold_weight, gold_quantile=args.gold_quantile,
        gold_threshold_gap=args.gold_threshold_gap,
        refine_p3_iters=args.refine_p3_iters,
        refine_score_epochs=args.refine_score_epochs,
        inference_samples=args.inference_samples,
        inference_shortlist=args.inference_shortlist,
        inf_skip_swap=not args.inf_full_search,
        inf_extra_starts=args.inf_extra_starts,
        inf_swap_rounds=args.inf_swap_rounds,
        inf_perturb_restarts=args.inf_perturb_restarts,
        inf_perturb_kick=args.inf_perturb_kick)


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        main(config_from_args(args))
