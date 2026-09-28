"""Reproduce GEN-GBC's Phase 3–4 ablation, adapted from GEN-CIM Table 5.

Rows: Phase-2 Only, +Latent Opt (one CEM), Full (initial CEM + two
CEM-Refine loops with optional gold injection). All three returned k-sets
are scored together with the same exact internal-node ordered-pair GBC
C++ evaluator throughout.

Run once per dataset/k/seed; the CSV is updated by that key:
    python run_ablation.py --dataset ca-grqc --data-root /content/data \\
        --k 10 --exact-source exact_gbc.cpp --threads 4
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

from train_gim import GIMConfig, main
from exact_gbc_scorer import compile_exact_gbc

VARIANTS = ("phase2_only", "latent_opt", "full")
FIELDS = ("dataset", "k", "seed", "phase2_only_raw_gbc", "latent_opt_raw_gbc",
          "full_raw_gbc", "latent_opt_delta_pct", "full_delta_pct",
          "phase2_only_online_exact", "latent_opt_online_exact", "full_online_exact",
          "phase2_only_training_time_s", "phase2_only_inference_time_s",
          "latent_opt_training_time_s", "latent_opt_inference_time_s",
          "full_training_time_s", "full_inference_time_s",
          "loop0_exact", "loop1_exact", "loop2_exact",
          "refine_injections", "evaluation")


def exact_evaluate(binary: Path, graph_path: Path,
                   seed_sets: Dict[str, list[int]], threads: int = 1) -> Dict[str, float]:
    """Use one graph load/one evaluator invocation for all three methods."""
    cmd = [str(binary), "--graph", str(graph_path), "--group-ids", "internal",
           "--threads", str(threads), "--quiet"]
    for name, nodes in seed_sets.items():
        if not nodes:
            raise ValueError("Cannot evaluate an empty seed group")
        cmd += ["--group", name + ":" + ",".join(map(str, sorted(nodes)))]
    payload = json.loads(subprocess.run(cmd, check=True, capture_output=True,
                                        text=True).stdout)
    if payload.get("pair_domain") != "all_distinct_ordered_pairs" or payload.get(
            "coverage") != "at_least_one_internal_group_node":
        raise ValueError("Exact evaluator uses a different GBC convention")
    scores = {item["method"]: float(item["raw_gbc"]) for item in payload["results"]}
    if set(scores) != set(seed_sets):
        raise ValueError("Exact evaluator returned the wrong set of methods")
    return scores


def _percent_gain(after: float, baseline: float) -> Optional[float]:
    return 100.0 * (after - baseline) / baseline if baseline > 0 else None


def run_ablation(cfg: GIMConfig, *, csv_path: Optional[Path] = None,
                 exact_source: Optional[Path] = None,
                 threads: int = 1) -> dict:
    """Run all variants with the same Phase 1/2 and exact GBC evaluator."""
    if threads < 1:
        raise ValueError("threads must be positive")
    # --no-resume retrains Phase 1/2 once; the subsequent variants must
    # reuse that same starting state for a controlled comparison.
    # Table 5 isolates the training phases. Extra CVAE samples at inference
    # would add a separate search method to the two CEM variants.
    cfg = replace(cfg, exact_threads=threads,
                  exact_source=str(exact_source or cfg.exact_source))
    results = {name: main(replace(cfg, variant=name,
                                  inference_samples=0, inference_shortlist=0,
                                  resume=cfg.resume or i > 0))
               for i, name in enumerate(VARIANTS)}
    binary = compile_exact_gbc(cfg.exact_source, Path(cfg.checkpoint_dir) / "exact_gbc_binary")
    scores = exact_evaluate(binary, Path(cfg.data_root) / f"{cfg.dataset.lower()}.txt",
        {name: sorted(result.best_seed_set.nodes) for name, result in results.items()},
        threads=threads)
    # Each call to main() writes one result JSON with the wall times measured
    # immediately before and during inference. Read all three after training.
    run_logs = {}
    for name in VARIANTS:
        path = Path(cfg.results_dir) / (
            f"gbc_{cfg.dataset.lower()}_k{cfg.k}_seed{cfg.seed}_{name}_exact.json")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["dataset"] != cfg.dataset or record["k"] != cfg.k
                or record["variant"] != name or record["seed"] != cfg.seed):
            raise ValueError(f"Ablation result belongs to another experiment: {path}")
        run_logs[name] = record
    loops = run_logs["full"]["loops"]
    row = {"dataset": cfg.dataset, "k": cfg.k, "seed": cfg.seed,
           "phase2_only_raw_gbc": scores["phase2_only"],
           "latent_opt_raw_gbc": scores["latent_opt"],
           "full_raw_gbc": scores["full"],
           "latent_opt_delta_pct": _percent_gain(scores["latent_opt"],
                                                   scores["phase2_only"]),
           "full_delta_pct": _percent_gain(scores["full"], scores["phase2_only"]),
           "phase2_only_online_exact": results["phase2_only"].best_exact,
           "latent_opt_online_exact": results["latent_opt"].best_exact,
           "full_online_exact": results["full"].best_exact,
           "refine_injections": sum(bool(item["injected"]) for item in loops),
           "evaluation": "exact_raw_ordered_internal_node_gbc"}
    for i in range(3):
        row[f"loop{i}_exact"] = loops[i]["cem_GBC"] if i < len(loops) else None
    for name, record in run_logs.items():
        row[f"{name}_training_time_s"] = float(record["training_time_s"])
        row[f"{name}_inference_time_s"] = float(record["inference_time_s"])
    output = csv_path or Path(cfg.results_dir) / "ablation_phase3_phase4_exact.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = []
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Existing CSV has a different schema: {output}")
            prior = [item for item in reader if not (
                item["dataset"] == cfg.dataset and int(item["k"]) == cfg.k
                and int(item.get("seed", 42)) == cfg.seed)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(prior + [row])
    print(f"[Ablation] Phase2={scores['phase2_only']:.6f}, "
          f"+Latent={scores['latent_opt']:.6f}, Full={scores['full']:.6f} "
          f"(exact raw GBC); saved {output}")
    return row


def _smoke_test() -> None:
    import torch
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "tiny.txt"
        path.write_text("0 1\n1 2\n2 3\n3 4\n4 5\n5 0\n0 3\n", encoding="utf-8")
        cfg = GIMConfig(dataset="tiny", data_root=str(root), k=2,
            checkpoint_dir=str(root / "ckpt"),
            results_dir=str(root / "results"), device="cpu", p1_hidden=8,
            p1_heads=2, p1_epochs=1, p2_H=1, p2_neighbors=2,
            p2_value_epochs=1, p3_latent_dim=4, p3_hidden=16, p3_iters=2,
            score_epochs=1, cem_runs=3, cem_pop=4, cem_iter=2,
            refine_p3_iters=2, refine_score_epochs=1,
            langevin_steps=1, inference_samples=3, inference_shortlist=2)
        row = run_ablation(cfg, threads=1)
        assert row["phase2_only_raw_gbc"] >= 0 and row["full_raw_gbc"] >= 0
        assert row["evaluation"] == "exact_raw_ordered_internal_node_gbc"
        for name in VARIANTS:
            assert row[f"{name}_training_time_s"] >= 0
            assert row[f"{name}_inference_time_s"] >= 0
        again = run_ablation(cfg, threads=1)
        assert row["full_raw_gbc"] == again["full_raw_gbc"]
        with (root / "results" / "ablation_phase3_phase4_exact.csv").open(
                newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 1
    print("run_ablation.py smoke test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEN-GBC Table 5 ablation")
    parser.add_argument("--dataset", default="ca-grqc")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--exact-source", default=str(Path(__file__).with_name("exact_gbc.cpp")))
    parser.add_argument("--checkpoint-dir", default="experiments/checkpoints")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-score-net", action="store_true")
    parser.add_argument("--p1-epochs", type=int, default=1000)
    parser.add_argument("--p2-value-epochs", type=int, default=500)
    parser.add_argument("--p3-iters", type=int, default=3000)
    parser.add_argument("--p3-beta-max", type=float, default=1.0)
    parser.add_argument("--p3-lambda-vp", type=float, default=0.1)
    parser.add_argument("--p3-gamma", type=float, default=0.05)
    parser.add_argument("--p2-elite-top-k", type=int, default=5)
    parser.add_argument("--p2-elite-weight", type=float, default=4.0)
    parser.add_argument("--score-epochs", type=int, default=300)
    parser.add_argument("--cem-pop", type=int, default=60)
    parser.add_argument("--cem-iter", type=int, default=25)
    parser.add_argument("--cem-sigma-scale", type=float, default=1.0)
    parser.add_argument("--refine-p3-iters", type=int,
                        help="Default: max(500, p3_iters // 3) as in GEN-CIM")
    parser.add_argument("--refine-score-epochs", type=int,
                        help="Default: use score_epochs as in GEN-CIM")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        run_ablation(GIMConfig(dataset=args.dataset, data_root=args.data_root,
            k=args.k, exact_source=args.exact_source,
            checkpoint_dir=args.checkpoint_dir, results_dir=args.results_dir,
            device=args.device, seed=args.seed, resume=not args.no_resume,
            score_enabled=not args.no_score_net, p1_epochs=args.p1_epochs,
            p2_value_epochs=args.p2_value_epochs,
            p2_elite_top_k=args.p2_elite_top_k,
            p2_elite_weight=args.p2_elite_weight,
            p3_iters=args.p3_iters, p3_beta_max=args.p3_beta_max,
            p3_lambda_vp=args.p3_lambda_vp, p3_gamma=args.p3_gamma,
            score_epochs=args.score_epochs, cem_pop=args.cem_pop,
            cem_iter=args.cem_iter, cem_sigma_scale=args.cem_sigma_scale,
            refine_p3_iters=args.refine_p3_iters,
            refine_score_epochs=args.refine_score_epochs),
            csv_path=args.output_csv, threads=args.threads)
