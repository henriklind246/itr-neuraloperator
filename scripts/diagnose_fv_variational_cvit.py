#!/usr/bin/env python3
"""Single-problem causal variational-CN reachability gate for InterfaceCViT."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import problem_from_config
from scripts.diagnose_fv_direct_state import (
    active_system,
    build_homogeneous_gate_case,
    normalized_rhs,
    optimize_dense_table,
    preregistered_max_lead,
    sequential_exact_trajectory,
    trajectory_gate_metrics,
)
from scripts.diagnose_supervised_overfit import (
    _forcing_supervised_metrics,
    _predict_forcing_probe,
)
from src.operators.train import (
    _advance_scheduler,
    build_optimizer,
    build_scheduler,
    set_seed,
)
from src.operators.train_pino import (
    _decode_in_chunks,
    _forcing_fixed_probe_case,
    _interface_spatial_channels_from_normalized,
    _validate_forcing_layered_dataset,
    build_cvit,
    build_forcing_image,
    build_forcing_transition_image,
    forcing_transition_physics_loss,
    load_diffusion_data,
    load_verified_transition_gate,
    normalize_forcing_interface_scalars,
    sample_transition_physics_intervals,
    transition_consecutive_interval_trigger,
)
from src.operators.cvit import ForcingTransitionCViT
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import default_ramp_seconds
from src.physics.one_step_objective import (
    build_cn_tensors,
    causal_variational_terms,
    explicit_cn_rhs,
    implicit_cn_action,
)


def rollout_residual_rms(
    prediction_K: np.ndarray,
    solver,
    *,
    sigma: float,
) -> float:
    deviation = (np.asarray(prediction_K, dtype=np.float64) - 300.0) / float(sigma)
    matrix, _ = active_system(solver)
    residuals = []
    for n in range(len(solver.t) - 1):
        z_np1 = deviation[n + 1, : solver.Nx - 1].ravel(order="F")
        rhs = normalized_rhs(
            solver, deviation[n], float(solver.t[n]), sigma=float(sigma)
        )
        residuals.append(matrix @ z_np1 - rhs)
    return float(np.sqrt(np.mean(np.square(np.concatenate(residuals)))))


def build_fixed_solver(problem, params, physics, x_grid, y_grid, *, dt, t_final, t_ramp):
    base_kwargs = {
        "a": float(physics["a"]),
        "b": float(physics["b"]),
        "c": float(physics["c"]),
        "d": float(physics["d"]),
        "Nx": int(len(x_grid)),
        "Ny": int(len(y_grid)),
        "lam_target": 0.8,
        "layers": physics["layers"],
        "t_final": float(round(float(t_final), 6)),
        "flux_f": 0.0,
        "flux_A": 0.0,
        "t_on": 0.0,
        "t_off": 0.2,
        "phase": 0.0,
        "tukey_alpha": 0.5,
        "dt": float(dt),
        "y_grid": y_grid,
        "ramp_seconds": float(t_ramp),
    }
    return problem.configure_solver(params, base_kwargs)


def _transition_probe_model(seed: int, device: torch.device) -> ForcingTransitionCViT:
    torch.manual_seed(int(seed))
    return ForcingTransitionCViT(
        forcing_in_ch=3,
        source_in_ch=1,
        out_dim=1,
        emb_dim=32,
        dec_emb_dim=32,
        source_patch_size=5,
        source_grid_size=(20, 20),
        forcing_patch_size=4,
        forcing_grid_size=(20, 32),
        depth_enc=2,
        depth_dec=2,
        num_heads=4,
        mlp_ratio=2.0,
        query_time_conditioning=True,
        film_time_conditioning=True,
        film_init_std=1.0e-3,
        t_final=0.3,
    ).to(device)


def _transition_probe_params(metadata: dict) -> dict:
    forcing = metadata["forcing"]
    return {
        "temporal_family": forcing["temporal_family"],
        "temporal_params": forcing["temporal_params"],
        "spatial_family": forcing["spatial_family"],
        "spatial_params": forcing["spatial_params"],
        "ic_family": metadata["family"],
        "ic_params": metadata["ic_params"],
    }


def _transition_probe_mesh(
    sim, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    x_grid = torch.as_tensor(sim.grid_x, dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(sim.grid_y, dtype=torch.float32, device=device)
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack(
        (gx.reshape(-1), gy.reshape(-1)), dim=-1,
    ).unsqueeze(0)
    return x_grid, y_grid, mesh, np.asarray(sim.grid_y, dtype=np.float64)


def _predict_transition_probe_steps(
    model: ForcingTransitionCViT,
    source: torch.Tensor,
    params: dict,
    steps: torch.Tensor,
    mesh: torch.Tensor,
    y_img: np.ndarray,
    *,
    dt: float,
    t_ramp: float,
    query_chunk: int,
    source_time_value: float = 0.0,
) -> torch.Tensor:
    if steps.ndim != 1 or bool((steps <= 0).any().item()):
        raise ValueError("probe endpoint steps must be a positive vector")
    count = int(steps.numel())
    lead = steps.to(device=source.device, dtype=source.dtype) * float(dt)
    source_time = torch.full_like(lead, float(source_time_value))
    forcing = build_forcing_transition_image(
        [params] * count,
        y_img,
        source_time,
        lead,
        32,
        300.0,
        source.device,
        float(t_ramp),
        0.3,
    ).to(dtype=source.dtype)
    source_batch = source.expand(count, -1, -1, -1)
    source_tokens = model.encode_source(source)
    forcing_tokens = model.encode_forcing(forcing)
    encoding = model.fuse(
        source_tokens.expand(count, -1, -1), forcing_tokens,
    )
    coords = mesh.expand(count, -1, -1)
    if query_chunk <= 0 or coords.shape[1] <= query_chunk:
        decoded = model.decode(
            encoding, source_batch, coords, source_time, lead,
        )
    else:
        decoded = torch.cat([
            model.decode(
                encoding,
                source_batch,
                coords[:, start:start + query_chunk],
                source_time,
                lead,
            )
            for start in range(0, coords.shape[1], query_chunk)
        ], dim=1)
    return decoded[..., 0].reshape(count, 20, 20)


@torch.no_grad()
def evaluate_transition_probe(
    model: ForcingTransitionCViT,
    source: torch.Tensor,
    params: dict,
    truth: np.ndarray,
    sim,
    mesh: torch.Tensor,
    y_img: np.ndarray,
    *,
    mu: float,
    sigma: float,
    query_chunk: int,
    source_time_value: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    steps = torch.arange(
        1, len(sim.t), device=source.device, dtype=torch.long,
    )
    normalized = _predict_transition_probe_steps(
        model,
        source,
        params,
        steps,
        mesh,
        y_img,
        dt=float(sim.dt),
        t_ramp=2.0 * float(sim.dt),
        query_chunk=int(query_chunk),
        source_time_value=float(source_time_value),
    )
    prediction = np.empty_like(truth)
    prediction[0] = source[0, 0].detach().cpu().numpy()
    prediction[1:] = normalized.detach().cpu().numpy()
    basic = trajectory_gate_metrics(
        prediction, truth, sim, sigma=float(sigma),
    )
    copy_prediction = np.broadcast_to(truth[0], truth.shape).copy()
    copy_metrics = trajectory_gate_metrics(
        copy_prediction, truth, sim, sigma=float(sigma),
    )
    skills = {}
    low_fractions = []
    edges = (0.0, 0.05, 0.10, 0.20, 0.30)
    error_K = float(sigma) * (prediction - truth)
    copy_error_K = float(sigma) * (copy_prediction - truth)
    for lead_bin, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (sim.t > lo + 1.0e-12) & (sim.t <= hi + 1.0e-12)
        if not bool(mask.any()):
            continue
        error = float(np.sqrt(np.mean(error_K[mask] ** 2)))
        copy_error = float(np.sqrt(np.mean(copy_error_K[mask] ** 2)))
        signal = float(np.sqrt(np.mean(
            (float(sigma) * (truth[mask] - truth[0])) ** 2
        )))
        if signal >= 0.01 * float(sigma) and copy_error > 0.0:
            skills[str(lead_bin)] = 1.0 - (error / copy_error) ** 2
        spectrum = np.fft.fft2(error_K[mask], axes=(-2, -1), norm="ortho")
        total = float(np.sum(np.abs(spectrum) ** 2))
        low = float(np.sum(np.abs(spectrum[..., :4, :4]) ** 2))
        low_fractions.append(0.0 if total == 0.0 else low / total)
    denominator = float(np.sqrt(np.mean(
        (float(sigma) * (truth[1:] - truth[0])) ** 2
    )))
    departure = float(np.sqrt(np.mean(
        (float(sigma) * (prediction[1:] - prediction[0])) ** 2
    )))
    zero_params = copy.deepcopy(params)
    zero_params["temporal_params"]["A"] = 0.0
    zero_normalized = _predict_transition_probe_steps(
        model,
        source,
        zero_params,
        steps,
        mesh,
        y_img,
        dt=float(sim.dt),
        t_ramp=2.0 * float(sim.dt),
        query_chunk=int(query_chunk),
        source_time_value=float(source_time_value),
    ).detach().cpu().numpy()
    forcing_response = float(np.sqrt(np.mean(
        (float(sigma) * (prediction[1:] - zero_normalized)) ** 2
    )))
    metrics = {
        **basic,
        "copy_overall_field": copy_metrics["overall_field"],
        "copy_longest_lead": copy_metrics["longest_lead"],
        "copy_physical_defect": copy_metrics["physical_defect"],
        "copy_skills_by_lead": skills,
        "low_frequency_fractions": low_fractions,
        "source_departure_ratio": (
            0.0 if denominator == 0.0 else departure / denominator
        ),
        "forcing_response_ratio": (
            0.0 if denominator == 0.0 else forcing_response / denominator
        ),
        "finite": bool(np.isfinite(prediction).all()),
    }
    return metrics, prediction


def _train_transition_supervised_probe(
    model: ForcingTransitionCViT,
    source: torch.Tensor,
    params: dict,
    truth: np.ndarray,
    sim,
    mesh: torch.Tensor,
    y_img: np.ndarray,
    *,
    updates: int,
    learning_rate: float,
    seed: int,
    query_chunk: int,
    optimizer: torch.optim.Optimizer | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.optim.Optimizer, np.random.Generator]:
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(learning_rate),
        )
    if rng is None:
        rng = np.random.default_rng(int(seed) + 30_001)
    truth_t = torch.as_tensor(
        truth, device=source.device, dtype=source.dtype,
    )
    source_steps = np.asarray([0], dtype=np.int64)
    for update in range(int(updates)):
        intervals = sample_transition_physics_intervals(
            rng,
            source_steps=source_steps,
            dt=float(sim.dt),
            t_final=0.3,
            lead_edges=[0.0, 0.05, 0.10, 0.20, 0.30],
            include_anchor_interval=True,
            intervals_per_cell=1,
            consecutive_intervals={"enabled": False},
        )
        max_lead = preregistered_max_lead(update, updates, 0.3)
        steps = torch.unique(intervals["start_step"] + 1)
        steps = steps[
            steps.to(dtype=torch.float64) * float(sim.dt)
            <= max_lead + 1.0e-12
        ].to(device=source.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict_transition_probe_steps(
            model,
            source,
            params,
            steps,
            mesh,
            y_img,
            dt=float(sim.dt),
            t_ramp=2.0 * float(sim.dt),
            query_chunk=int(query_chunk),
        )
        loss = (
            prediction - truth_t.index_select(0, steps)
        ).square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite supervised transition probe")
        loss.backward()
        optimizer.step()
    return optimizer, rng


def _train_transition_physics_probe(
    model: ForcingTransitionCViT,
    source: torch.Tensor,
    params: dict,
    sim,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    y_img: np.ndarray,
    *,
    sigma: float,
    objective: str,
    updates: int,
    learning_rate: float,
    seed: int,
    query_chunk: int,
    consecutive: dict[str, Any],
) -> dict[str, float | bool]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(seed) + 40_001)
    last = {}
    for update in range(int(updates)):
        intervals = sample_transition_physics_intervals(
            rng,
            source_steps=np.asarray([0], dtype=np.int64),
            dt=float(sim.dt),
            t_final=0.3,
            lead_edges=[0.0, 0.05, 0.10, 0.20, 0.30],
            include_anchor_interval=True,
            intervals_per_cell=1,
            consecutive_intervals=consecutive,
        )
        max_lead = preregistered_max_lead(update, updates, 0.3)
        mask = (
            (intervals["start_step"] + 1).to(dtype=torch.float64)
            * float(sim.dt)
            <= max_lead + 1.0e-12
        )
        intervals = {key: value[mask] for key, value in intervals.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = forcing_transition_physics_loss(
            model=model,
            source_fields=source,
            params=[params],
            source_steps=np.asarray([0], dtype=np.int64),
            source_bins=np.asarray([0], dtype=np.int64),
            intervals=intervals,
            x_grid=x_grid,
            y_grid=y_grid,
            y_img=y_img,
            nt_img=32,
            a_ref=300.0,
            t_ramp=2.0 * float(sim.dt),
            t_final=0.3,
            dt=float(sim.dt),
            sigma_global=float(sigma),
            right_value=0.0,
            objective=objective,
            causal_epsilon=0.01,
            query_chunk=int(query_chunk),
        )
        loss = result["loss"]
        if not bool(result["finite"]) or not bool(torch.isfinite(loss).item()):
            return {"finite": False, "last_loss": float("nan")}
        loss.backward()
        if not all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            return {"finite": False, "last_loss": float(loss.detach().cpu())}
        optimizer.step()
        last = {
            "finite": True,
            "last_loss": float(loss.detach().cpu()),
            "physical_defect_mse": float(
                result["physical_defect_mse"].detach().cpu()
            ),
        }
    return last


def run_transition_gate(args: argparse.Namespace) -> dict:
    direct_gate, direct_hash = load_verified_transition_gate(
        args.gate_summary,
        expected_stage="dense_direct_state_gate",
        require_passed=not args.smoke,
    )
    gate_sigma = float(
        direct_gate.get("configuration", {}).get(
            "resolved_sigma_global",
            direct_gate.get("configuration", {}).get("sigma", 10.0),
        )
    )
    if args.sigma is not None and not math.isclose(
        float(args.sigma), gate_sigma, rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError("network gate sigma differs from the direct-state gate")
    args.sigma = gate_sigma
    objectives = tuple(
        args.objectives
        or direct_gate.get("qualified_objectives")
        or ()
    )
    if not objectives:
        raise ValueError("the direct-state gate qualified no objective")
    unknown = sorted(set(objectives) - {"raw_ls", "variational", "defect"})
    if unknown:
        raise ValueError(f"unknown transition objectives {unknown}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    sim, truth, metadata = build_homogeneous_gate_case(
        family="grf_2d",
        source_time=0.0,
        grid_size=20,
        dt=0.005,
        sigma=float(args.sigma),
        seed=int(args.case_seed),
    )
    params = _transition_probe_params(metadata)
    x_grid, y_grid, mesh, y_img = _transition_probe_mesh(sim, device)
    source = torch.as_tensor(
        truth[0], dtype=torch.float32, device=device,
    ).unsqueeze(0).unsqueeze(0)
    floor = sequential_exact_trajectory(
        sim, truth[0], sigma=float(args.sigma),
    )
    floor_metrics = trajectory_gate_metrics(
        floor, truth, sim, sigma=float(args.sigma),
    )

    initial_seeds = [41, 42, 43]
    supervised_budget = int(args.supervised_budget)
    capacity_ceiling = int(args.capacity_ceiling)
    if not args.smoke and (
        supervised_budget != 4_000 or capacity_ceiling != 8_000
    ):
        raise ValueError("registered network budgets are 4000 and 8000")
    supervised: dict[str, dict[str, Any]] = {}
    comparator_budget = supervised_budget
    for seed in initial_seeds:
        model = _transition_probe_model(seed, device)
        supervised_optimizer, supervised_rng = _train_transition_supervised_probe(
            model,
            source,
            params,
            truth,
            sim,
            mesh,
            y_img,
            updates=supervised_budget,
            learning_rate=float(args.learning_rate),
            seed=int(args.case_seed),
            query_chunk=int(args.query_chunk),
        )
        matched, _ = evaluate_transition_probe(
            model,
            source,
            params,
            truth,
            sim,
            mesh,
            y_img,
            mu=300.0,
            sigma=float(args.sigma),
            query_chunk=int(args.query_chunk),
        )
        copy_gap = (
            matched["copy_overall_field"] - floor_metrics["overall_field"]
        )
        closed = (
            matched["copy_overall_field"] - matched["overall_field"]
        ) / max(copy_gap, np.finfo(float).tiny)
        capacity = copy.deepcopy(matched)
        capacity_closed = float(closed)
        if closed < 0.80 and capacity_ceiling > supervised_budget:
            supervised_optimizer, supervised_rng = (
                _train_transition_supervised_probe(
                model,
                source,
                params,
                truth,
                sim,
                mesh,
                y_img,
                updates=capacity_ceiling - supervised_budget,
                learning_rate=float(args.learning_rate),
                seed=int(args.case_seed),
                query_chunk=int(args.query_chunk),
                optimizer=supervised_optimizer,
                rng=supervised_rng,
            ))
            capacity, _ = evaluate_transition_probe(
                model,
                source,
                params,
                truth,
                sim,
                mesh,
                y_img,
                mu=300.0,
                sigma=float(args.sigma),
                query_chunk=int(args.query_chunk),
            )
            capacity_closed = (
                capacity["copy_overall_field"] - capacity["overall_field"]
            ) / max(copy_gap, np.finfo(float).tiny)
        supervised[str(seed)] = {
            "matched_budget": supervised_budget,
            "matched_metrics": matched,
            "matched_gap_closed": float(closed),
            "capacity_ceiling": capacity_ceiling,
            "capacity_metrics": capacity,
            "capacity_gap_closed": float(capacity_closed),
            "capacity_valid": bool(capacity_closed >= 0.80),
        }
    if not all(row["capacity_valid"] for row in supervised.values()):
        summary = {
            "schema_version": 1,
            "stage": "single_instance_network_gate",
            "passed": False,
            "invalid_reason": "reduced_probe_supervised_capacity",
            "direct_state_gate_sha256": direct_hash,
            "supervised": supervised,
        }
        canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        summary["gate_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        with (output_dir / "network_gate_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary
    if any(row["matched_gap_closed"] < 0.80 for row in supervised.values()):
        comparator_budget = capacity_ceiling

    objective_results = {}
    for objective_index, objective in enumerate(objectives):
        dense, _ = optimize_dense_table(
            sim,
            truth[0],
            sigma=float(args.sigma),
            objective=objective,
            optimizer_name="adam",
            updates=comparator_budget,
            learning_rate=0.01,
            lbfgs_iterations=0,
            defect_sweeps=1,
            defect_omega=2.0 / 3.0,
            seed=int(args.case_seed) + objective_index,
        )
        dense_metrics = trajectory_gate_metrics(
            dense, truth, sim, sigma=float(args.sigma),
        )
        seed_rows = []
        for seed in initial_seeds:
            model = _transition_probe_model(seed, device)
            train_info = _train_transition_physics_probe(
                model,
                source,
                params,
                sim,
                x_grid,
                y_grid,
                y_img,
                sigma=float(args.sigma),
                objective=objective,
                updates=comparator_budget,
                learning_rate=float(args.learning_rate),
                seed=int(args.case_seed),
                query_chunk=int(args.query_chunk),
                consecutive={"enabled": False},
            )
            metrics, prediction = evaluate_transition_probe(
                model,
                source,
                params,
                truth,
                sim,
                mesh,
                y_img,
                mu=300.0,
                sigma=float(args.sigma),
                query_chunk=int(args.query_chunk),
            )
            comparator = (
                supervised[str(seed)]["matched_metrics"]
                if comparator_budget == supervised_budget
                else supervised[str(seed)]["capacity_metrics"]
            )
            overall_threshold = comparator["overall_field"] + 0.20 * (
                metrics["copy_overall_field"] - comparator["overall_field"]
            )
            longest_threshold = comparator["longest_lead"] + 0.20 * (
                metrics["copy_longest_lead"] - comparator["longest_lead"]
            )
            defect_threshold = max(
                5.0 * dense_metrics["physical_defect"],
                0.05 * metrics["copy_physical_defect"],
            )
            finite = bool(metrics["finite"] and train_info.get("finite", False))
            collapsed = bool(
                metrics["forcing_response_ratio"] < 0.01
                or metrics["source_departure_ratio"] < 0.01
            )
            criteria = {
                "overall": metrics["overall_field"] <= overall_threshold,
                "longest": metrics["longest_lead"] <= longest_threshold,
                "copy_skill": bool(
                    metrics["copy_skills_by_lead"]
                    and all(
                        value > 0.0
                        for value in metrics["copy_skills_by_lead"].values()
                    )
                ),
                "finite": finite,
                "defect": (
                    metrics["physical_defect"] <= defect_threshold
                ),
                "not_collapsed": not collapsed,
            }
            contingency = transition_consecutive_interval_trigger(
                overall_passed=bool(criteria["overall"]),
                longest_passed=bool(criteria["longest"]),
                copy_skills=list(metrics["copy_skills_by_lead"].values()),
                forcing_response_ratio=metrics["forcing_response_ratio"],
                source_departure_ratio=metrics["source_departure_ratio"],
                finite_and_defect_passed=bool(
                    criteria["finite"] and criteria["defect"]
                ),
                low_frequency_fractions=metrics["low_frequency_fractions"],
            )
            if contingency:
                model = _transition_probe_model(seed, device)
                train_info = _train_transition_physics_probe(
                    model,
                    source,
                    params,
                    sim,
                    x_grid,
                    y_grid,
                    y_img,
                    sigma=float(args.sigma),
                    objective=objective,
                    updates=comparator_budget,
                    learning_rate=float(args.learning_rate),
                    seed=int(args.case_seed),
                    query_chunk=int(args.query_chunk),
                    consecutive={
                        "enabled": True,
                        "probability": 0.25,
                        "min_length": 3,
                        "max_length": 3,
                    },
                )
                metrics, prediction = evaluate_transition_probe(
                    model,
                    source,
                    params,
                    truth,
                    sim,
                    mesh,
                    y_img,
                    mu=300.0,
                    sigma=float(args.sigma),
                    query_chunk=int(args.query_chunk),
                )
                criteria["overall"] = (
                    metrics["overall_field"] <= overall_threshold
                )
                criteria["longest"] = (
                    metrics["longest_lead"] <= longest_threshold
                )
                criteria["copy_skill"] = bool(
                    metrics["copy_skills_by_lead"]
                    and all(
                        value > 0.0
                        for value in metrics["copy_skills_by_lead"].values()
                    )
                )
                criteria["finite"] = bool(
                    metrics["finite"] and train_info.get("finite", False)
                )
                criteria["defect"] = (
                    metrics["physical_defect"] <= defect_threshold
                )
                criteria["not_collapsed"] = bool(
                    metrics["forcing_response_ratio"] >= 0.01
                    and metrics["source_departure_ratio"] >= 0.01
                )
            passed = bool(all(criteria.values()))
            score = 0.5 * (
                metrics["overall_field"] / max(
                    metrics["copy_overall_field"], np.finfo(float).tiny,
                )
                + metrics["longest_lead"] / max(
                    metrics["copy_longest_lead"], np.finfo(float).tiny,
                )
            )
            seed_rows.append({
                "seed": seed,
                "training": train_info,
                "metrics": metrics,
                "criteria": criteria,
                "contingency_used": contingency,
                "passed": passed,
                "calibrated_score": float(score),
            })
            np.savez_compressed(
                output_dir / f"{objective}_seed{seed}_prediction.npz",
                prediction=prediction,
                truth=truth,
                t_grid=sim.t,
            )
        passed_count = sum(bool(row["passed"]) for row in seed_rows)
        no_bad_seed = all(
            row["criteria"]["finite"] and row["criteria"]["not_collapsed"]
            for row in seed_rows
        )
        objective_results[objective] = {
            "dense_table_metrics": dense_metrics,
            "seeds": seed_rows,
            "median_calibrated_score": float(np.median(
                [row["calibrated_score"] for row in seed_rows]
            )),
            "worst_calibrated_score": float(max(
                row["calibrated_score"] for row in seed_rows
            )),
            "passed": bool(passed_count >= 2 and no_bad_seed),
        }
    qualified = [
        name for name, row in objective_results.items() if row["passed"]
    ]
    selected = (
        min(
            qualified,
            key=lambda name: objective_results[name][
                "median_calibrated_score"
            ],
        )
        if qualified else None
    )
    selected_policy = (
        {"enabled": False, "probability": 0.25, "min_length": 2, "max_length": 4}
        if selected is None
        or not any(
            row["contingency_used"]
            for row in objective_results[selected]["seeds"]
        )
        else {
            "enabled": True,
            "probability": 0.25,
            "min_length": 3,
            "max_length": 3,
        }
    )
    summary = {
        "schema_version": 1,
        "stage": "single_instance_network_gate",
        "passed": bool(selected is not None),
        "direct_state_gate_sha256": direct_hash,
        "selected_objective": selected,
        "matched_budget": comparator_budget,
        "initialization_seeds": initial_seeds,
        "decision_rule": "at_least_two_of_three_and_no_collapse_or_nonfinite_seed",
        "consecutive_intervals": selected_policy,
        "supervised": supervised,
        "objectives": objective_results,
        "optimization_uses_solution_fields": False,
        "normalization_uses_training_trajectories": bool(
            direct_gate.get("normalization_uses_training_trajectories", False)
        ),
        "checkpoint_selection_uses_validation_targets": True,
        "direct_state_gate_uses_fv_reference_solutions": True,
        "network_gate_uses_supervised_capacity_baseline": True,
        "physics_only_claim_scope": "optimization_objective",
    }
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    summary["gate_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with (output_dir / "network_gate_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run_transition_mini_operator(args: argparse.Namespace) -> dict:
    direct_gate, direct_hash = load_verified_transition_gate(
        args.gate_summary,
        expected_stage="dense_direct_state_gate",
        require_passed=not args.smoke,
    )
    gate_sigma = float(
        direct_gate.get("configuration", {}).get(
            "resolved_sigma_global",
            direct_gate.get("configuration", {}).get("sigma", 10.0),
        )
    )
    if args.sigma is not None and not math.isclose(
        float(args.sigma), gate_sigma, rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError("mini-operator sigma differs from the direct-state gate")
    args.sigma = gate_sigma
    network_gate, network_hash = load_verified_transition_gate(
        args.network_gate_summary,
        expected_stage="single_instance_network_gate",
        require_passed=not args.smoke,
    )
    if str(network_gate.get("direct_state_gate_sha256")) != direct_hash:
        raise ValueError("mini-operator gates do not form an authorized chain")
    objective = str(network_gate.get("selected_objective"))
    if objective not in {"raw_ls", "variational", "defect"}:
        if not args.smoke:
            raise ValueError("network gate selected no production objective")
        objective = str(args.objectives[0] if args.objectives else "variational")
    consecutive = dict(
        network_gate.get("consecutive_intervals")
        or {
            "enabled": False,
            "probability": 0.25,
            "min_length": 2,
            "max_length": 4,
        }
    )
    updates = int(args.mini_updates)
    if not args.smoke and updates != 4_000:
        raise ValueError("the mini-operator budget is preregistered at 4000")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    families = (
        "uniform_2d",
        "random_sinusoid_2d",
        "grf_2d",
        "hot_spot_2d",
    )
    source_times = (0.0, 0.075, 0.150, 0.225)
    cases = []
    for family_index, family in enumerate(families):
        for time_index, source_time in enumerate(source_times):
            sim, truth, metadata = build_homogeneous_gate_case(
                family=family,
                source_time=source_time,
                grid_size=20,
                dt=0.005,
                sigma=float(args.sigma),
                seed=int(args.case_seed) + 10 * family_index + time_index,
            )
            cases.append({
                "family": family,
                "source_time": source_time,
                "source_bin": time_index,
                "sim": sim,
                "truth": truth,
                "metadata": metadata,
                "params": _transition_probe_params(metadata),
            })
    model = _transition_probe_model(42, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args.learning_rate),
    )
    x_grid, y_grid, mesh, y_img = _transition_probe_mesh(
        cases[0]["sim"], device,
    )
    source = torch.as_tensor(
        np.stack([case["truth"][0] for case in cases]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    source_steps = np.asarray([
        int(round(case["source_time"] / 0.005)) for case in cases
    ], dtype=np.int64)
    source_bins = np.asarray([
        int(case["source_bin"]) for case in cases
    ], dtype=np.int64)
    params = [case["params"] for case in cases]
    rng = np.random.default_rng(int(args.case_seed) + 50_001)
    last_loss = float("nan")
    last_defect = float("nan")
    for update in range(updates):
        intervals = sample_transition_physics_intervals(
            rng,
            source_steps=source_steps,
            dt=0.005,
            t_final=0.3,
            lead_edges=[0.0, 0.05, 0.10, 0.20, 0.30],
            include_anchor_interval=True,
            intervals_per_cell=1,
            consecutive_intervals=consecutive,
        )
        max_lead = preregistered_max_lead(update, updates, 0.3)
        mask = (
            (intervals["start_step"] + 1).to(dtype=torch.float64) * 0.005
            <= max_lead + 1.0e-12
        )
        intervals = {key: value[mask] for key, value in intervals.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = forcing_transition_physics_loss(
            model=model,
            source_fields=source,
            params=params,
            source_steps=source_steps,
            source_bins=source_bins,
            intervals=intervals,
            x_grid=x_grid,
            y_grid=y_grid,
            y_img=y_img,
            nt_img=32,
            a_ref=300.0,
            t_ramp=0.01,
            t_final=0.3,
            dt=0.005,
            sigma_global=float(args.sigma),
            right_value=0.0,
            objective=objective,
            causal_epsilon=0.01,
            query_chunk=int(args.query_chunk),
        )
        if not bool(result["finite"]):
            raise FloatingPointError("non-finite 16-source mini-operator loss")
        loss = result["loss"]
        loss.backward()
        if not all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(
                "non-finite 16-source mini-operator gradient"
            )
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        last_defect = float(result["physical_defect_mse"].detach().cpu())

    evaluations = []
    for index, case in enumerate(cases):
        metrics, prediction = evaluate_transition_probe(
            model,
            source[index:index + 1],
            case["params"],
            case["truth"],
            case["sim"],
            mesh,
            y_img,
            mu=300.0,
            sigma=float(args.sigma),
            query_chunk=int(args.query_chunk),
            source_time_value=float(case["source_time"]),
        )
        evaluations.append({
            "family": case["family"],
            "source_time": case["source_time"],
            "metrics": metrics,
        })
        np.savez_compressed(
            output_dir / f"case_{index:02d}_prediction.npz",
            prediction=prediction,
            truth=case["truth"],
            t_grid=case["sim"].t,
        )
    overall_skills = [
        1.0 - (
            row["metrics"]["overall_field"]
            / max(
                row["metrics"]["copy_overall_field"],
                np.finfo(float).tiny,
            )
        ) ** 2
        for row in evaluations
    ]
    summary = {
        "schema_version": 1,
        "stage": "multi_source_mini_operator",
        "passed": bool(
            all(row["metrics"]["finite"] for row in evaluations)
            and float(np.median(overall_skills)) > 0.0
        ),
        "objective": objective,
        "updates": updates,
        "sources": 16,
        "sources_per_ic_family": 4,
        "grid_size": 20,
        "balanced_source_time_bins": True,
        "varied_forcing_amplitudes_and_frequencies": True,
        "consecutive_intervals": consecutive,
        "last_loss": last_loss,
        "last_physical_defect_mse": last_defect,
        "median_overall_copy_skill": float(np.median(overall_skills)),
        "worst_overall_copy_skill": float(np.min(overall_skills)),
        "cases": evaluations,
        "direct_state_gate_sha256": direct_hash,
        "network_gate_sha256": network_hash,
        "optimization_uses_solution_fields": False,
        "normalization_uses_training_trajectories": bool(
            direct_gate.get("normalization_uses_training_trajectories", False)
        ),
        "checkpoint_selection_uses_validation_targets": False,
        "direct_state_gate_uses_fv_reference_solutions": True,
        "network_gate_uses_supervised_capacity_baseline": True,
        "physics_only_claim_scope": "optimization_objective",
    }
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    summary["gate_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    torch.save(
        {
            "model_state": model.state_dict(),
            "summary": summary,
        },
        output_dir / "mini_operator.pt",
    )
    with (output_dir / "mini_operator_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run(args: argparse.Namespace) -> dict:
    source_path = Path(args.source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    config["training"]["epochs"] = int(args.updates)
    if args.lr is not None:
        config["training"]["learning_rate"] = float(args.lr)
    set_seed(args.seed)
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = float(data["mu_global"]), float(data["sigma_global"])
    x_grid = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid = np.asarray(data["y_grid"], dtype=np.float64)
    saved_t = np.asarray(data["t_grid"], dtype=np.float64)
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    problem = problem_from_config(config)

    image_meta = dict(
        checkpoint.get("interface_forcing") or checkpoint.get("forcing_image") or {}
    )
    dt = float(image_meta["fv_dt"])
    t_ramp = float((image_meta.get("ramp", {}) or {}).get(
        "duration", default_ramp_seconds(dt)
    ))
    q_ref = float(image_meta.get("a_ref", 300.0))
    physics, interface_face, _ = _validate_forcing_layered_dataset(
        problem, data, sim_params, dt
    )
    params = _forcing_fixed_probe_case(args.probe_case)
    solver = build_fixed_solver(
        problem,
        params,
        physics,
        x_grid,
        y_grid,
        dt=dt,
        t_final=float(saved_t[-1]),
        t_ramp=t_ramp,
    )
    _, _, _, truth_K = solver.solve(
        np.full((solver.Nx, solver.Ny), float(physics["T_right"])),
        store_trajectory=True,
    )
    t_full = np.asarray(solver.t, dtype=np.float64)

    ny_img = int(image_meta.get("ny_img", 96))
    nt_img = int(image_meta.get("nt_img", 256))
    y_img = np.linspace(float(y_grid[0]), float(y_grid[-1]), ny_img)
    t_img = np.linspace(0.0, float(saved_t[-1]), nt_img)
    right_value = (float(physics["T_right"]) - mu) / sigma
    fixed_ic = np.full((solver.Nx, solver.Ny), right_value, dtype=np.float32)
    spatial = torch.from_numpy(_interface_spatial_channels_from_normalized(
        fixed_ic, float(physics["interface_x"]), x_grid,
    )).unsqueeze(0).to(device)
    forcing_image = build_forcing_image(
        [params], y_img, t_img, q_ref, device, t_ramp
    )
    scalars = torch.from_numpy(normalize_forcing_interface_scalars(
        float(physics["interface_x"]), float(params["R_c"])
    )).unsqueeze(0).to(device)

    model = build_cvit(
        config,
        mu,
        sigma,
        grid_size=(solver.Nx, solver.Ny),
        t_final=float(saved_t[-1]),
        variant="interfaces",
    ).to(device)
    if not model.hard_right_dirichlet:
        raise ValueError("the variational gate requires the hard right Dirichlet ansatz")
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    cn = build_cn_tensors(solver, sigma=sigma, device=device, dtype=torch.float32)

    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    coords = torch.from_numpy(np.stack(
        (gx.reshape(-1), gy.reshape(-1)), axis=-1
    ).astype(np.float32)).unsqueeze(0).to(device)
    n_points = solver.Nx * solver.Ny
    ic_count = min(int(args.ic_points), n_points)
    ic_indices = torch.linspace(
        0, n_points - 1, steps=ic_count, device=device
    ).round().to(torch.long)
    ic_coords = coords.index_select(1, ic_indices)
    ic_target = torch.full(
        (1, ic_count, 1), float(right_value), device=device
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)
    init_prediction = _predict_forcing_probe(
        model, spatial, forcing_image, scalars, x_grid, y_grid, t_full,
        mu, sigma, args.query_chunk, device,
    )
    init_metrics = _forcing_supervised_metrics(
        init_prediction, truth_K, init_prediction, interface_face
    )
    init_metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
        init_prediction, solver, sigma=sigma
    )
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "mu_global": mu,
        "sigma_global": sigma,
        "probe_params": params,
        "source_checkpoint": str(source_path),
        "objective": "causal_capacity_weighted_variational_cn",
    }, output_dir / "cvit_init.pt")
    np.savez_compressed(
        output_dir / "initial_predictions.npz",
        prediction_K=init_prediction,
        truth_K=truth_K,
        t_grid=t_full,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    metric_names = list(init_metrics)
    fields = [
        "update", "causal_stage", "causal_fraction", "sampled_step",
        "variational_objective", "cn_residual_rms_normalized", "ic_mse",
        "total_objective", "gradient_norm", "lr", "elapsed_seconds",
        *metric_names,
    ]
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "update": 0,
            "causal_stage": 0,
            "causal_fraction": args.stage_fractions[0],
            "sampled_step": "",
            "variational_objective": "",
            "cn_residual_rms_normalized": "",
            "ic_mse": "",
            "total_objective": "",
            "gradient_norm": "",
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": 0.0,
            **init_metrics,
        })

    print(
        f"[variational-cvit] device={device} updates={args.updates} "
        f"grid={solver.Nx}x{solver.Ny} steps={len(t_full) - 1} "
        f"case={args.probe_case} Rc={params['R_c']} lr={optimizer.param_groups[0]['lr']} "
        f"out={output_dir}",
        flush=True,
    )
    print(
        f"Update 0: field={init_metrics['field_rmse_K']:.4f}K "
        f"response={init_metrics['temporal_response_ratio']:.3e} "
        f"jump={init_metrics['pred_jump_rms_K']:.4f}/"
        f"{init_metrics['true_jump_rms_K']:.4f}K",
        flush=True,
    )

    stage_updates = [int(v) for v in args.stage_updates]
    stage_fractions = [float(v) for v in args.stage_fractions]
    if len(stage_updates) != len(stage_fractions) or stage_updates[0] != 0:
        raise ValueError("stage updates/fractions must align and begin at zero")
    if stage_updates != sorted(stage_updates) or any(
        not 0.0 < value <= 1.0 for value in stage_fractions
    ):
        raise ValueError("invalid causal stage schedule")
    rng = np.random.default_rng(args.seed + 4081)
    start = time.perf_counter()
    best_score = float("inf")
    best_update = 0
    latest_metrics = init_metrics
    last_training = {}

    for update in range(1, int(args.updates) + 1):
        stage = max(i for i, value in enumerate(stage_updates) if update - 1 >= value)
        fraction = stage_fractions[stage]
        active_steps = max(1, int(math.ceil((len(t_full) - 1) * fraction)))
        step = int(rng.integers(0, active_steps))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        latent = model.encode(spatial, forcing_image, scalars)
        time_np1 = torch.full(
            (1, n_points, 1), float(t_full[step + 1]), device=device
        )
        prediction_np1 = _decode_in_chunks(
            model, latent, coords, time_np1, int(args.query_chunk)
        )[..., 0].view(1, solver.Nx, solver.Ny)
        if step == 0:
            prediction_n = torch.full_like(prediction_np1, float(right_value))
        else:
            with torch.no_grad():
                time_n = torch.full(
                    (1, n_points, 1), float(t_full[step]), device=device
                )
                prediction_n = _decode_in_chunks(
                    model, latent, coords, time_n, int(args.query_chunk)
                )[..., 0].view(1, solver.Nx, solver.Ny)
        step_tensor = torch.tensor([step], device=device, dtype=torch.long)
        variational, residual_rms = causal_variational_terms(
            prediction_np1,
            prediction_n,
            step_tensor,
            cn,
            right_value=right_value,
        )
        time_zero = torch.zeros((1, ic_count, 1), device=device)
        prediction_ic = model.decode(latent, ic_coords, time_zero)
        ic_mse = (prediction_ic - ic_target).square().mean()
        loss = variational + float(args.ic_weight) * ic_mse
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite variational CViT objective")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.grad_clip)
        )
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        last_training = {
            "variational_objective": float(variational.detach().cpu()),
            "cn_residual_rms_normalized": float(residual_rms.detach().cpu()),
            "ic_mse": float(ic_mse.detach().cpu()),
            "total_objective": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

        do_eval = update % int(args.validate_every) == 0 or update == int(args.updates)
        if not do_eval:
            continue
        prediction = _predict_forcing_probe(
            model, spatial, forcing_image, scalars, x_grid, y_grid, t_full,
            mu, sigma, args.query_chunk, device,
        )
        latest_metrics = _forcing_supervised_metrics(
            prediction, truth_K, init_prediction, interface_face
        )
        latest_metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
            prediction, solver, sigma=sigma
        )
        elapsed = time.perf_counter() - start
        row = {
            "update": update,
            "causal_stage": stage,
            "causal_fraction": fraction,
            "sampled_step": step,
            **last_training,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed,
            **latest_metrics,
        }
        with metrics_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        score = latest_metrics["field_rmse_K"] + latest_metrics["node_jump_rmse_K"]
        if score < best_score:
            best_score = score
            best_update = update
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "completed_updates": update,
                "metrics": latest_metrics,
                "probe_params": params,
                "source_checkpoint": str(source_path),
            }, output_dir / "cvit_best.pt")
        print(
            f"Update {update}: stage={stage}/{fraction:.2f} step={step} "
            f"E={last_training['variational_objective']:.3e} "
            f"r={last_training['cn_residual_rms_normalized']:.3e} "
            f"field={latest_metrics['field_rmse_K']:.4f}K "
            f"response={latest_metrics['temporal_response_ratio']:.3e} "
            f"jump={latest_metrics['pred_jump_rms_K']:.4f}/"
            f"{latest_metrics['true_jump_rms_K']:.4f}K "
            f"rollout_r={latest_metrics['rollout_residual_rms_normalized']:.3e} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    final_prediction = _predict_forcing_probe(
        model, spatial, forcing_image, scalars, x_grid, y_grid, t_full,
        mu, sigma, args.query_chunk, device,
    )
    final_metrics = _forcing_supervised_metrics(
        final_prediction, truth_K, init_prediction, interface_face
    )
    final_metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
        final_prediction, solver, sigma=sigma
    )
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "completed_updates": int(args.updates),
        "metrics": final_metrics,
        "probe_params": params,
        "source_checkpoint": str(source_path),
    }, output_dir / "cvit_last.pt")
    np.savez_compressed(
        output_dir / "final_predictions.npz",
        prediction_K=final_prediction,
        truth_K=truth_K,
        t_grid=t_full,
        x_grid=x_grid,
        y_grid=y_grid,
    )
    summary = {
        "hypothesis": (
            "Capacity-weighted variational CN training with detached causal "
            "states can escape the constant-field basin using the unchanged CViT."
        ),
        "source_checkpoint": str(source_path),
        "probe_case": args.probe_case,
        "seed": int(args.seed),
        "completed_updates": int(args.updates),
        "best_update": int(best_update),
        "wall_seconds": elapsed,
        "objective": {
            "kind": "capacity_weighted_variational_cn",
            "previous_state_detached": True,
            "temperature_labels": 0,
            "ic_weight": float(args.ic_weight),
            "stage_updates": stage_updates,
            "stage_fractions": stage_fractions,
        },
        "initial_metrics": init_metrics,
        "final_metrics": final_metrics,
        "last_training_values": last_training,
    }
    with (output_dir / "final_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[variational-cvit] done -> {output_dir / 'final_metrics.json'}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=(
            "interface", "forcing_transition", "forcing_transition_mini",
        ),
        default="interface",
    )
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--gate-summary", default=None)
    parser.add_argument("--network-gate-summary", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-case", default="pulse_low")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--ic-points", type=int, default=512)
    parser.add_argument("--ic-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-seed", type=int, default=1701)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument(
        "--objectives", nargs="+",
        choices=("raw_ls", "variational", "defect"),
        default=None,
    )
    parser.add_argument("--supervised-budget", type=int, default=4000)
    parser.add_argument("--capacity-ceiling", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--mini-updates", type=int, default=4000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage-updates", type=int, nargs="+", default=[0, 250, 500, 750]
    )
    parser.add_argument(
        "--stage-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    args = parser.parse_args()
    if args.mode == "interface" and args.source_checkpoint is None:
        parser.error("--source-checkpoint is required for --mode interface")
    if args.mode == "forcing_transition" and args.gate_summary is None:
        parser.error("--gate-summary is required for --mode forcing_transition")
    if args.mode == "forcing_transition_mini" and (
        args.gate_summary is None or args.network_gate_summary is None
    ):
        parser.error(
            "--gate-summary and --network-gate-summary are required for "
            "--mode forcing_transition_mini"
        )
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "forcing_transition":
        run_transition_gate(parsed)
    elif parsed.mode == "forcing_transition_mini":
        run_transition_mini_operator(parsed)
    else:
        run(parsed)
