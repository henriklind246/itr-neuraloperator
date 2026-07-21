#!/usr/bin/env python3
"""State-conditioned one-step InterfaceCViT trained by variational CN physics."""

from __future__ import annotations

import argparse
import copy
import csv
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
from scripts.diagnose_fv_variational_cvit import (
    build_cn_tensors,
    build_fixed_solver,
    causal_variational_terms,
    rollout_residual_rms,
)
from scripts.diagnose_supervised_overfit import _forcing_supervised_metrics
from src.operators.train import (
    _advance_scheduler,
    build_optimizer,
    build_scheduler,
    set_seed,
)
from src.operators.cvit import moving_interface_jump_enrichment
from src.operators.train_pino import (
    _decode_in_chunks,
    _forcing_fixed_probe_case,
    _validate_forcing_layered_dataset,
    build_cvit,
    load_diffusion_data,
    normalize_forcing_interface_scalars,
    sample_forcing_params,
)
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    TEMPORAL_FAMILIES,
    default_ramp_seconds,
)
from src.physics.fv_residual import (
    build_cn_geom_per_interface,
    interface_trace_constraint_residuals,
)
from src.operators.one_step import (
    build_interval_forcing_images,
    material_channels,
    predict_one_step_field as predict_one_step,
    rollout_field as causal_rollout,
)
from src.physics.one_step_objective import (
    conservative_energy_storage_projection,
    interface_flux_head_physics_terms,
    left_energy_interface_flux_closure,
    one_step_interface_trace_terms,
    state_conditioned_spatial,
    two_sided_energy_interface_flux_closure,
)


def loss_gradient_geometry(
    losses: dict[str, torch.Tensor],
    parameters,
) -> dict[str, float]:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = {}
    for name, loss in losses.items():
        gradients[name] = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

    norms = {}
    for name, grads in gradients.items():
        norm_sq = sum(
            (grad.square().sum() for grad in grads if grad is not None),
            start=losses[name].new_zeros(()),
        )
        norms[name] = norm_sq.sqrt()

    result = {
        f"grad_norm_{name}": float(value.detach().cpu())
        for name, value in norms.items()
    }
    first = gradients["variational"]
    for name in (key for key in gradients if key != "variational"):
        dot = sum(
            (
                left.mul(right).sum()
                for left, right in zip(first, gradients[name])
                if left is not None and right is not None
            ),
            start=losses[name].new_zeros(()),
        )
        denominator = norms["variational"] * norms[name]
        cosine = torch.where(
            denominator > 0.0,
            dot / denominator,
            denominator.new_zeros(()),
        )
        result[f"grad_cos_variational_{name}"] = float(cosine.detach().cpu())
    return result


def rollout_interface_trace_metrics(
    rollout: torch.Tensor,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    interface_x: float,
    resistance: float,
    dt: float,
    k_left: float,
    k_right: float,
    rho: float,
    cp: float,
    sigma: float,
    q_ref: float,
) -> dict[str, float]:
    count = rollout.shape[0] - 1
    geom = build_cn_geom_per_interface(
        x_grid,
        y_grid,
        float(k_left),
        float(k_right),
        np.full(count, float(interface_x)),
        np.full(count, float(resistance)),
        float(dt),
        sigma_global=float(sigma),
        rho=float(rho),
        cp=float(cp),
        device=rollout.device,
        dtype=rollout.dtype,
    )
    traces = interface_trace_constraint_residuals(
        rollout[:-1],
        rollout[1:],
        geom,
        x_grid,
        np.full(count, float(resistance)),
        k_left=float(k_left),
        k_right=float(k_right),
        sigma_global=float(sigma),
        q_ref=float(q_ref),
    )
    flux = traces["flux"][:, -1]
    contact = traces["contact"][:, -1]
    return {
        "trace_flux_rms_normalized": float(flux.square().mean().sqrt().cpu()),
        "trace_contact_rms_normalized": float(contact.square().mean().sqrt().cpu()),
        "trace_q_series_rms": float(
            traces["q_series"][:, -1].square().mean().sqrt().cpu()
        ),
    }


def interface_flux_agreement_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    prediction = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if float(np.std(prediction)) > 0.0 and float(np.std(target)) > 0.0
        else 0.0
    )
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        f"{prefix}_correlation": correlation,
    }


def _interface_image_metadata(checkpoint: dict, checkpoint_path: Path) -> dict:
    current = checkpoint
    current_path = checkpoint_path
    for _ in range(4):
        metadata = dict(
            current.get("interface_forcing") or current.get("forcing_image") or {}
        )
        if metadata:
            return metadata
        parent = current.get("source_checkpoint")
        if not parent:
            break
        current_path = Path(parent).expanduser().resolve()
        current = torch.load(current_path, map_location="cpu", weights_only=False)
    raise ValueError(
        f"could not resolve interface forcing metadata from {checkpoint_path}"
    )


def prepare_one_step_probe(
    case: str,
    problem,
    physics: dict,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    saved_t: np.ndarray,
    *,
    dt: float,
    t_ramp: float,
    q_ref: float,
    sigma: float,
    ny_image: int,
    image_time_points: int,
    device: torch.device,
    params: dict | None = None,
) -> dict:
    params = copy.deepcopy(
        _forcing_fixed_probe_case(case) if params is None else params
    )
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
    time_grid = np.asarray(solver.t, dtype=np.float64)
    y_image = np.linspace(float(y_grid[0]), float(y_grid[-1]), int(ny_image))
    interval_images = torch.from_numpy(build_interval_forcing_images(
        params,
        y_image,
        time_grid,
        t_ramp=t_ramp,
        q_ref=q_ref,
        image_time_points=int(image_time_points),
    )).to(device)
    scalars = torch.from_numpy(normalize_forcing_interface_scalars(
        float(physics["interface_x"]), float(params["R_c"])
    )).unsqueeze(0).to(device)
    interface_x = torch.tensor(
        [float(physics["interface_x"])], device=device, dtype=torch.float32
    )
    jump_scale = torch.tensor(
        [float(params["R_c"]) * q_ref / sigma],
        device=device,
        dtype=torch.float32,
    )
    cn = build_cn_tensors(solver, sigma=sigma, device=device, dtype=torch.float32)
    geom = build_cn_geom_per_interface(
        x_grid,
        y_grid,
        float(physics["k_left"]),
        float(physics["k_right"]),
        [float(physics["interface_x"])],
        [float(params["R_c"])],
        dt,
        sigma_global=sigma,
        rho=float(physics["rho_left"]),
        cp=float(physics["cp_left"]),
        device=device,
        dtype=torch.float32,
    )
    q_left_integrals = []
    for step in range(len(time_grid) - 1):
        integral = np.asarray(
            solver.q_left_integral(time_grid[step], time_grid[step + 1]),
            dtype=np.float64,
        )
        if integral.ndim == 0:
            integral = np.full(solver.Ny, float(integral))
        q_left_integrals.append(integral)
    q_left_integrals = torch.from_numpy(
        np.stack(q_left_integrals).astype(np.float32)
    ).to(device)
    interface_face = int(geom.face_idx.reshape(-1)[0].item())
    true_interface_flux = np.asarray(solver.G_x[interface_face])[None] * (
        truth_K[1:, interface_face] - truth_K[1:, interface_face + 1]
    )
    return {
        "case": str(case),
        "params": params,
        "solver": solver,
        "truth_K": np.asarray(truth_K),
        "time_grid": time_grid,
        "interval_images": interval_images,
        "scalars": scalars,
        "interface_x": interface_x,
        "jump_scale": jump_scale,
        "cn": cn,
        "geom": geom,
        "q_left_integrals": q_left_integrals,
        "true_interface_flux": true_interface_flux,
        "interface_face": interface_face,
    }


def sample_stratified_forcing_probes(
    seed: int,
    *,
    prefix: str,
    per_combination: int,
    dt: float,
    t_final: float,
    y_bounds: tuple[float, float],
) -> dict[str, dict]:
    if int(per_combination) <= 0:
        raise ValueError("per_combination must be positive")
    rng = np.random.default_rng(int(seed))
    records = {}
    temporal_window = {
        "t_on": 0.0,
        "t_off": 0.2,
        "phase": 0.0,
        "tukey_alpha": 0.5,
    }
    for temporal_family in TEMPORAL_FAMILIES:
        for spatial_family in SPATIAL_FAMILIES:
            for sample_index in range(int(per_combination)):
                params = sample_forcing_params(
                    rng,
                    1,
                    float(dt),
                    float(t_final),
                    c=float(y_bounds[0]),
                    d=float(y_bounds[1]),
                    temporal_window=temporal_window,
                    temporal_family=str(temporal_family),
                    spatial_family=str(spatial_family),
                )[0]
                params["R_c"] = float(rng.uniform(0.05, 1.0))
                params["interface_x"] = 0.5
                name = (
                    f"{prefix}_{temporal_family}_{spatial_family}_"
                    f"{sample_index:02d}"
                )
                records[name] = params
    return records


def evaluate_one_step_probe(
    model,
    probe: dict,
    initial_state: torch.Tensor,
    fixed_channels: torch.Tensor,
    coords: torch.Tensor,
    baseline_prediction_K: np.ndarray | None,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    physics: dict,
    *,
    mu: float,
    sigma: float,
    q_ref: float,
    query_chunk: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rollout, flux, correction, implied_flux_gap = causal_rollout(
        model,
        initial_state,
        probe["interval_images"],
        probe["scalars"],
        fixed_channels,
        coords,
        dt=float(probe["solver"].dt),
        query_chunk=int(query_chunk),
        steps=len(probe["time_grid"]) - 1,
        interface_x=probe["interface_x"],
        jump_scale=probe["jump_scale"],
        closure_geom=probe["geom"],
        q_left_integrals=probe["q_left_integrals"],
        resistance=float(probe["params"]["R_c"]),
        sigma=sigma,
        q_ref=q_ref,
        return_interface_flux=True,
        return_storage_projection=True,
    )
    prediction_K = rollout.cpu().numpy() * sigma + mu
    baseline = prediction_K if baseline_prediction_K is None else baseline_prediction_K
    metrics = _forcing_supervised_metrics(
        prediction_K,
        probe["truth_K"],
        baseline,
        int(probe["interface_face"]),
    )
    metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
        prediction_K, probe["solver"], sigma=sigma
    )
    metrics.update(rollout_interface_trace_metrics(
        rollout,
        x_grid,
        y_grid,
        interface_x=float(physics["interface_x"]),
        resistance=float(probe["params"]["R_c"]),
        dt=float(probe["solver"].dt),
        k_left=float(physics["k_left"]),
        k_right=float(physics["k_right"]),
        rho=float(physics["rho_left"]),
        cp=float(physics["cp_left"]),
        sigma=sigma,
        q_ref=q_ref,
    ))
    physical_flux = q_ref * flux.cpu().numpy()
    series_flux = np.asarray(
        probe["solver"].G_x[int(probe["interface_face"])]
    )[None] * (
        prediction_K[1:, int(probe["interface_face"])]
        - prediction_K[1:, int(probe["interface_face"]) + 1]
    )
    metrics["flux_head_q_rms"] = float(
        (q_ref * flux).square().mean().sqrt().cpu()
    )
    metrics.update(interface_flux_agreement_metrics(
        physical_flux, probe["true_interface_flux"], prefix="flux_head_truth"
    ))
    metrics.update(interface_flux_agreement_metrics(
        series_flux, probe["true_interface_flux"], prefix="flux_series_truth"
    ))
    metrics["storage_projection_rms_K"] = float(
        (sigma * correction).square().mean().sqrt().cpu()
    )
    metrics["implied_flux_gap_rms"] = float(
        implied_flux_gap.square().mean().sqrt().cpu()
    )
    return metrics, rollout, flux, correction, implied_flux_gap


def run_mini_operator(args: argparse.Namespace) -> dict:
    sampled_family_grid = bool(args.sampled_family_grid)
    train_cases = list(dict.fromkeys(args.train_probe_cases or ()))
    holdout_cases = list(dict.fromkeys(args.holdout_probe_cases or ()))
    if sampled_family_grid and (train_cases or holdout_cases):
        raise ValueError(
            "sampled family-grid probes cannot be combined with fixed probes"
        )
    if not sampled_family_grid and (not train_cases or not holdout_cases):
        raise ValueError("mini-operator training requires train and holdout probes")
    if not sampled_family_grid and set(train_cases) & set(holdout_cases):
        raise ValueError("mini-operator train and holdout probes must be disjoint")
    if not args.jump_enrichment or (
        args.jump_flux_mode != "conservative_storage_projection"
    ):
        raise ValueError(
            "mini-operator training requires conservative storage projection"
        )
    if int(args.updates) <= 0:
        raise ValueError("mini-operator training requires positive updates")
    if not sampled_family_grid:
        for case in [*train_cases, *holdout_cases]:
            _forcing_fixed_probe_case(case)

    source_path = Path(args.source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    config["training"]["epochs"] = int(args.updates)
    config["training"]["pino"]["forcing"]["nt_img"] = int(
        args.image_time_points
    )
    interface_config = config["model"].setdefault("interface_cvit", {})
    interface_config["jump_enrichment"] = True
    interface_config["jump_flux_depth"] = int(args.jump_flux_depth)
    interface_config["jump_flux_conditioning"] = str(
        args.jump_flux_conditioning
    )
    interface_config["jump_flux_mode"] = "conservative_storage_projection"
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
    image_meta = _interface_image_metadata(checkpoint, source_path)
    dt = float(image_meta["fv_dt"])
    t_ramp = float((image_meta.get("ramp", {}) or {}).get(
        "duration", default_ramp_seconds(dt)
    ))
    q_ref = float(image_meta.get("a_ref", 300.0))
    physics, _, _ = _validate_forcing_layered_dataset(
        problem, data, sim_params, dt
    )
    ny_image = int(config["training"]["pino"]["forcing"].get("ny_img") or 96)
    probe_parameters = {}
    if sampled_family_grid:
        if int(args.sampled_train_seed) == int(args.sampled_holdout_seed):
            raise ValueError("sampled train and holdout seeds must be distinct")
        train_parameters = sample_stratified_forcing_probes(
            int(args.sampled_train_seed),
            prefix="train",
            per_combination=int(args.sampled_train_per_combination),
            dt=dt,
            t_final=float(saved_t[-1]),
            y_bounds=(float(physics["c"]), float(physics["d"])),
        )
        holdout_parameters = sample_stratified_forcing_probes(
            int(args.sampled_holdout_seed),
            prefix="holdout",
            per_combination=int(args.sampled_holdout_per_combination),
            dt=dt,
            t_final=float(saved_t[-1]),
            y_bounds=(float(physics["c"]), float(physics["d"])),
        )
        train_cases = list(train_parameters)
        holdout_cases = list(holdout_parameters)
        probe_parameters = {**train_parameters, **holdout_parameters}
    case_names = [*train_cases, *holdout_cases]
    probes = {
        case: prepare_one_step_probe(
            case,
            problem,
            physics,
            x_grid,
            y_grid,
            saved_t,
            dt=dt,
            t_ramp=t_ramp,
            q_ref=q_ref,
            sigma=sigma,
            ny_image=ny_image,
            image_time_points=int(args.image_time_points),
            device=device,
            params=probe_parameters.get(case),
        )
        for case in case_names
    }
    probe_parameters = {
        case: copy.deepcopy(probe["params"]) for case, probe in probes.items()
    }
    first_solver = probes[case_names[0]]["solver"]
    right_value = (float(physics["T_right"]) - mu) / sigma
    initial_state = torch.full(
        (1, first_solver.Nx, first_solver.Ny), float(right_value), device=device
    )
    fixed_channels = material_channels(
        right_value,
        float(physics["interface_x"]),
        x_grid,
        first_solver.Ny,
        device=device,
    )
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    coords = torch.from_numpy(np.stack(
        (gx.reshape(-1), gy.reshape(-1)), axis=-1
    ).astype(np.float32)).unsqueeze(0).to(device)
    model = build_cvit(
        config,
        mu,
        sigma,
        grid_size=(first_solver.Nx, first_solver.Ny),
        t_final=float(saved_t[-1]),
        variant="interfaces",
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(
            "invalid mini-operator warm start: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)

    baseline_predictions = {}
    state_buffers = {}
    initial_metrics = {}
    initial_outputs = {}
    for case, probe in probes.items():
        metrics, rollout, flux, correction, gap = evaluate_one_step_probe(
            model,
            probe,
            initial_state,
            fixed_channels,
            coords,
            None,
            x_grid,
            y_grid,
            physics,
            mu=mu,
            sigma=sigma,
            q_ref=q_ref,
            query_chunk=args.query_chunk,
        )
        baseline_predictions[case] = rollout.cpu().numpy() * sigma + mu
        initial_metrics[case] = metrics
        initial_outputs[case] = (rollout, flux, correction, gap)
        if case in train_cases:
            state_buffers[case] = rollout.detach().clone()

    metric_names = list(initial_metrics[case_names[0]])
    training_names = [
        "variational_objective",
        "cn_residual_rms_normalized",
        "flux_head_series_loss",
        "sample_trace_flux_rms_normalized",
        "sample_trace_contact_rms_normalized",
        "sample_storage_projection_rms_K",
        "gradient_norm",
    ]
    fields = [
        "update",
        "split",
        "probe_case",
        "temporal_family",
        "spatial_family",
        "R_c",
        "selected_training_case",
        *training_names,
        "lr",
        "elapsed_seconds",
        *metric_names,
    ]
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in case_names:
            writer.writerow({
                "update": 0,
                "split": "train" if case in train_cases else "holdout",
                "probe_case": case,
                "temporal_family": probes[case]["params"]["temporal_family"],
                "spatial_family": probes[case]["params"]["spatial_family"],
                "R_c": float(probes[case]["params"]["R_c"]),
                "selected_training_case": "",
                **{name: "" for name in training_names},
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": 0.0,
                **initial_metrics[case],
            })
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "train_probe_cases": train_cases,
        "holdout_probe_cases": holdout_cases,
        "probe_parameters": probe_parameters,
        "source_checkpoint": str(source_path),
    }, output_dir / "cvit_init.pt")

    def aggregate(metrics_by_case: dict, names: list[str]) -> dict[str, float]:
        keys = (
            "field_rmse_K",
            "temporal_response_ratio",
            "node_jump_rmse_K",
            "flux_head_truth_rmse",
            "flux_head_truth_correlation",
            "rollout_residual_rms_normalized",
            "storage_projection_rms_K",
            "implied_flux_gap_rms",
        )
        return {
            key: float(np.mean([metrics_by_case[name][key] for name in names]))
            for key in keys
        }

    initial_holdout = aggregate(initial_metrics, holdout_cases)
    best_score = (
        initial_holdout["field_rmse_K"] + initial_holdout["node_jump_rmse_K"]
    )
    best_update = 0
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "completed_updates": 0,
        "metrics": initial_metrics,
        "train_probe_cases": train_cases,
        "holdout_probe_cases": holdout_cases,
        "probe_parameters": probe_parameters,
        "source_checkpoint": str(source_path),
    }, output_dir / "cvit_best_holdout.pt")
    print(
        f"[mini-operator] device={device} updates={args.updates} "
        f"train_cases={len(train_cases)} holdout_cases={len(holdout_cases)} "
        f"sampled_family_grid={sampled_family_grid} "
        f"out={output_dir}",
        flush=True,
    )
    print(
        "Update 0: "
        f"holdout_field={initial_holdout['field_rmse_K']:.4f}K "
        f"holdout_response={initial_holdout['temporal_response_ratio']:.3f} "
        f"holdout_jump_node={initial_holdout['node_jump_rmse_K']:.4f}K",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 8093)
    visits = {case: 0 for case in train_cases}
    start = time.perf_counter()
    last_training = {}
    final_metrics = initial_metrics
    final_outputs = initial_outputs
    for update in range(1, int(args.updates) + 1):
        selected = train_cases[(update - 1) % len(train_cases)]
        probe = probes[selected]
        visits[selected] += 1
        refresh = (
            visits[selected] > 1
            and (visits[selected] - 1) % int(args.buffer_refresh_every) == 0
        )
        if refresh:
            state_buffers[selected] = causal_rollout(
                model,
                initial_state,
                probe["interval_images"],
                probe["scalars"],
                fixed_channels,
                coords,
                dt=dt,
                query_chunk=args.query_chunk,
                steps=len(probe["time_grid"]) - 1,
                interface_x=probe["interface_x"],
                jump_scale=probe["jump_scale"],
                closure_geom=probe["geom"],
                q_left_integrals=probe["q_left_integrals"],
                resistance=float(probe["params"]["R_c"]),
                sigma=sigma,
                q_ref=q_ref,
            )
        step = int(rng.integers(0, len(probe["time_grid"]) - 1))
        state_n = state_buffers[selected][step : step + 1].detach()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, flux, correction, _ = predict_one_step(
            model,
            state_n,
            probe["interval_images"][step : step + 1],
            probe["scalars"],
            fixed_channels,
            coords,
            dt=dt,
            query_chunk=args.query_chunk,
            interface_x=probe["interface_x"],
            jump_scale=probe["jump_scale"],
            closure_geom=probe["geom"],
            q_left_integral=probe["q_left_integrals"][step : step + 1],
            resistance=float(probe["params"]["R_c"]),
            sigma=sigma,
            q_ref=q_ref,
            return_interface_flux=True,
            return_storage_projection=True,
        )
        flux_terms = interface_flux_head_physics_terms(
            state_n,
            prediction,
            flux,
            probe["geom"],
            probe["q_left_integrals"][step : step + 1],
            right_value=right_value,
            q_ref=q_ref,
            sigma=sigma,
        )
        variational, residual_rms = causal_variational_terms(
            prediction,
            state_n,
            torch.tensor([step], device=device, dtype=torch.long),
            probe["cn"],
            right_value=right_value,
        )
        trace = one_step_interface_trace_terms(
            prediction,
            probe["geom"],
            x_grid,
            float(probe["params"]["R_c"]),
            k_left=float(physics["k_left"]),
            k_right=float(physics["k_right"]),
            sigma=sigma,
            q_ref=q_ref,
        )
        total_loss = (
            variational
            + float(args.trace_flux_weight) * trace["flux_loss"]
            + float(args.trace_contact_weight) * trace["contact_loss"]
            + float(args.flux_series_weight) * flux_terms["series_loss"]
            + float(args.flux_left_energy_weight) * flux_terms["left_energy_loss"]
            + float(args.flux_right_energy_weight) * flux_terms["right_energy_loss"]
            + float(args.storage_projection_weight) * correction.square().mean()
        )
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("non-finite mini-operator objective")
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.grad_clip)
        )
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        last_training = {
            "variational_objective": float(variational.detach().cpu()),
            "cn_residual_rms_normalized": float(residual_rms.detach().cpu()),
            "flux_head_series_loss": float(
                flux_terms["series_loss"].detach().cpu()
            ),
            "sample_trace_flux_rms_normalized": float(
                trace["flux_rms"].detach().cpu()
            ),
            "sample_trace_contact_rms_normalized": float(
                trace["contact_rms"].detach().cpu()
            ),
            "sample_storage_projection_rms_K": float(
                (sigma * correction).square().mean().sqrt().detach().cpu()
            ),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        do_eval = (
            update % int(args.validate_every) == 0
            or update == int(args.updates)
        )
        if not do_eval:
            continue
        metrics_by_case = {}
        outputs_by_case = {}
        elapsed = time.perf_counter() - start
        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            for case, eval_probe in probes.items():
                result = evaluate_one_step_probe(
                    model,
                    eval_probe,
                    initial_state,
                    fixed_channels,
                    coords,
                    baseline_predictions[case],
                    x_grid,
                    y_grid,
                    physics,
                    mu=mu,
                    sigma=sigma,
                    q_ref=q_ref,
                    query_chunk=args.query_chunk,
                )
                metrics, rollout, flux_eval, correction_eval, gap_eval = result
                metrics_by_case[case] = metrics
                outputs_by_case[case] = (
                    rollout, flux_eval, correction_eval, gap_eval
                )
                writer.writerow({
                    "update": update,
                    "split": "train" if case in train_cases else "holdout",
                    "probe_case": case,
                    "temporal_family": eval_probe["params"]["temporal_family"],
                    "spatial_family": eval_probe["params"]["spatial_family"],
                    "R_c": float(eval_probe["params"]["R_c"]),
                    "selected_training_case": selected,
                    **last_training,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": elapsed,
                    **metrics,
                })
        final_metrics = metrics_by_case
        final_outputs = outputs_by_case
        train_mean = aggregate(metrics_by_case, train_cases)
        holdout_mean = aggregate(metrics_by_case, holdout_cases)
        score = holdout_mean["field_rmse_K"] + holdout_mean["node_jump_rmse_K"]
        if score < best_score:
            best_score = score
            best_update = update
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "completed_updates": update,
                "metrics": metrics_by_case,
                "train_probe_cases": train_cases,
                "holdout_probe_cases": holdout_cases,
                "probe_parameters": probe_parameters,
                "source_checkpoint": str(source_path),
            }, output_dir / "cvit_best_holdout.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config": config,
            "completed_updates": update,
            "metrics": metrics_by_case,
            "train_probe_cases": train_cases,
            "holdout_probe_cases": holdout_cases,
            "probe_parameters": probe_parameters,
            "source_checkpoint": str(source_path),
        }, output_dir / f"cvit_update_{update:04d}.pt")
        print(
            f"Update {update}: selected={selected} "
            f"train_field={train_mean['field_rmse_K']:.4f}K "
            f"holdout_field={holdout_mean['field_rmse_K']:.4f}K "
            f"holdout_response={holdout_mean['temporal_response_ratio']:.3f} "
            f"holdout_jump_node={holdout_mean['node_jump_rmse_K']:.4f}K "
            f"holdout_proj={holdout_mean['storage_projection_rms_K']:.4f}K "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "completed_updates": int(args.updates),
        "metrics": final_metrics,
        "train_probe_cases": train_cases,
        "holdout_probe_cases": holdout_cases,
        "probe_parameters": probe_parameters,
        "source_checkpoint": str(source_path),
    }, output_dir / "cvit_last.pt")
    for case, (rollout, flux, correction, gap) in final_outputs.items():
        np.savez_compressed(
            output_dir / f"final_rollout_{case}.npz",
            prediction_K=rollout.cpu().numpy() * sigma + mu,
            truth_K=probes[case]["truth_K"],
            t_grid=probes[case]["time_grid"],
            x_grid=x_grid,
            y_grid=y_grid,
            flux_head_q=q_ref * flux.cpu().numpy(),
            storage_projection_K=sigma * correction.cpu().numpy(),
            implied_flux_gap=gap.cpu().numpy(),
        )
    final_train = aggregate(final_metrics, train_cases)
    final_holdout = aggregate(final_metrics, holdout_cases)
    summary = {
        "hypothesis": (
            "A shared physics-only CViT with conservative storage projection can "
            "learn a robust mapping across every temporal and spatial forcing "
            "family and continuous contact resistance, including parameter-held-out "
            "cases."
        ),
        "source_checkpoint": str(source_path),
        "seed": int(args.seed),
        "completed_updates": int(args.updates),
        "best_holdout_update": int(best_update),
        "wall_seconds": elapsed,
        "temperature_labels": 0,
        "sampled_family_grid": sampled_family_grid,
        "sampled_train_seed": (
            int(args.sampled_train_seed) if sampled_family_grid else None
        ),
        "sampled_holdout_seed": (
            int(args.sampled_holdout_seed) if sampled_family_grid else None
        ),
        "train_probe_cases": train_cases,
        "holdout_probe_cases": holdout_cases,
        "probe_parameters": probe_parameters,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "initial_holdout_mean": initial_holdout,
        "final_train_mean": final_train,
        "final_holdout_mean": final_holdout,
        "last_training_values": last_training,
    }
    with (output_dir / "final_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[mini-operator] done -> {output_dir / 'final_metrics.json'}", flush=True)
    return summary


def run(args: argparse.Namespace) -> dict:
    source_path = Path(args.source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    if int(args.updates) < 0:
        raise ValueError("updates must be non-negative")
    if int(args.updates) > 0:
        config["training"]["epochs"] = int(args.updates)
    config["training"]["pino"]["forcing"]["nt_img"] = int(args.image_time_points)
    config["model"].setdefault("interface_cvit", {})["jump_enrichment"] = bool(
        args.jump_enrichment
    )
    config["model"]["interface_cvit"]["jump_flux_depth"] = int(
        args.jump_flux_depth
    )
    config["model"]["interface_cvit"]["jump_flux_conditioning"] = str(
        args.jump_flux_conditioning
    )
    config["model"]["interface_cvit"]["jump_flux_mode"] = str(
        args.jump_flux_mode
    )
    uses_storage_projection = (
        args.jump_flux_mode == "conservative_storage_projection"
    )
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
    image_meta = _interface_image_metadata(checkpoint, source_path)
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
    time_grid = np.asarray(solver.t, dtype=np.float64)
    true_interface_flux = np.asarray(solver.G_x[interface_face])[None] * (
        truth_K[1:, interface_face] - truth_K[1:, interface_face + 1]
    )

    ny_image = int(config["training"]["pino"]["forcing"].get("ny_img") or 96)
    y_image = np.linspace(float(y_grid[0]), float(y_grid[-1]), ny_image)
    interval_images_np = build_interval_forcing_images(
        params,
        y_image,
        time_grid,
        t_ramp=t_ramp,
        q_ref=q_ref,
        image_time_points=int(args.image_time_points),
    )
    interval_images = torch.from_numpy(interval_images_np).to(device)
    right_value = (float(physics["T_right"]) - mu) / sigma
    initial_state = torch.full(
        (1, solver.Nx, solver.Ny), float(right_value), device=device
    )
    fixed_channels = material_channels(
        right_value,
        float(physics["interface_x"]),
        x_grid,
        solver.Ny,
        device=device,
    )
    scalars = torch.from_numpy(normalize_forcing_interface_scalars(
        float(physics["interface_x"]), float(params["R_c"])
    )).unsqueeze(0).to(device)
    interface_x_tensor = torch.tensor(
        [float(physics["interface_x"])], device=device, dtype=torch.float32
    )
    jump_scale = torch.tensor(
        [float(params["R_c"]) * q_ref / sigma], device=device, dtype=torch.float32
    )
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    coords = torch.from_numpy(np.stack(
        (gx.reshape(-1), gy.reshape(-1)), axis=-1
    ).astype(np.float32)).unsqueeze(0).to(device)

    model = build_cvit(
        config,
        mu,
        sigma,
        grid_size=(solver.Nx, solver.Ny),
        t_final=float(saved_t[-1]),
        variant="interfaces",
    ).to(device)
    if args.warm_start:
        incompatible = model.load_state_dict(
            checkpoint["model_state"], strict=not bool(args.jump_enrichment)
        )
        if args.jump_enrichment:
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            if unexpected or any(
                not key.startswith("interface_flux_decoder.") for key in missing
            ):
                raise RuntimeError(
                    f"invalid enrichment warm start: missing={missing}, "
                    f"unexpected={unexpected}"
                )
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    cn = build_cn_tensors(solver, sigma=sigma, device=device, dtype=torch.float32)
    trace_geom = build_cn_geom_per_interface(
        x_grid,
        y_grid,
        float(physics["k_left"]),
        float(physics["k_right"]),
        [float(physics["interface_x"])],
        [float(params["R_c"])],
        dt,
        sigma_global=sigma,
        rho=float(physics["rho_left"]),
        cp=float(physics["cp_left"]),
        device=device,
        dtype=torch.float32,
    )
    q_left_integrals = []
    for step in range(len(time_grid) - 1):
        integral = np.asarray(
            solver.q_left_integral(time_grid[step], time_grid[step + 1]),
            dtype=np.float64,
        )
        if integral.ndim == 0:
            integral = np.full(solver.Ny, float(integral))
        q_left_integrals.append(integral)
    q_left_integrals = torch.from_numpy(
        np.stack(q_left_integrals).astype(np.float32)
    ).to(device)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=False)

    initial_result = causal_rollout(
        model,
        initial_state,
        interval_images,
        scalars,
        fixed_channels,
        coords,
        dt=dt,
        query_chunk=args.query_chunk,
        steps=len(time_grid) - 1,
        interface_x=interface_x_tensor,
        jump_scale=jump_scale,
        return_interface_flux=bool(args.jump_enrichment),
        closure_geom=trace_geom,
        q_left_integrals=q_left_integrals,
        resistance=float(params["R_c"]),
        sigma=sigma,
        q_ref=q_ref,
        return_storage_projection=uses_storage_projection,
    )
    if uses_storage_projection:
        (
            initial_rollout,
            initial_head_flux,
            initial_storage_correction,
            initial_implied_flux_gap,
        ) = initial_result
    elif args.jump_enrichment:
        initial_rollout, initial_head_flux = initial_result
        initial_storage_correction = initial_rollout.new_zeros(
            initial_rollout.shape[0] - 1, *initial_rollout.shape[1:]
        )
        initial_implied_flux_gap = initial_rollout.new_zeros(
            initial_rollout.shape[0] - 1
        )
    else:
        initial_rollout = initial_result
        initial_head_flux = None
        initial_storage_correction = initial_rollout.new_zeros(
            initial_rollout.shape[0] - 1, *initial_rollout.shape[1:]
        )
        initial_implied_flux_gap = initial_rollout.new_zeros(
            initial_rollout.shape[0] - 1
        )
    state_buffer = initial_rollout.detach()
    initial_prediction_K = initial_rollout.cpu().numpy() * sigma + mu
    initial_metrics = _forcing_supervised_metrics(
        initial_prediction_K, truth_K, initial_prediction_K, interface_face
    )
    initial_metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
        initial_prediction_K, solver, sigma=sigma
    )
    initial_metrics.update(rollout_interface_trace_metrics(
        initial_rollout,
        x_grid,
        y_grid,
        interface_x=float(physics["interface_x"]),
        resistance=float(params["R_c"]),
        dt=dt,
        k_left=float(physics["k_left"]),
        k_right=float(physics["k_right"]),
        rho=float(physics["rho_left"]),
        cp=float(physics["cp_left"]),
        sigma=sigma,
        q_ref=q_ref,
    ))
    initial_metrics["flux_head_q_rms"] = (
        float((q_ref * initial_head_flux).square().mean().sqrt().cpu())
        if initial_head_flux is not None else 0.0
    )
    initial_metrics["storage_projection_rms_K"] = float(
        (sigma * initial_storage_correction).square().mean().sqrt().cpu()
    )
    initial_metrics["implied_flux_gap_rms"] = float(
        initial_implied_flux_gap.square().mean().sqrt().cpu()
    )
    initial_head_physical = (
        q_ref * initial_head_flux.cpu().numpy()
        if initial_head_flux is not None
        else np.zeros_like(true_interface_flux)
    )
    initial_series_flux = np.asarray(solver.G_x[interface_face])[None] * (
        initial_prediction_K[1:, interface_face]
        - initial_prediction_K[1:, interface_face + 1]
    )
    initial_metrics.update(interface_flux_agreement_metrics(
        initial_head_physical, true_interface_flux, prefix="flux_head_truth"
    ))
    initial_metrics.update(interface_flux_agreement_metrics(
        initial_series_flux, true_interface_flux, prefix="flux_series_truth"
    ))
    truth_normalized = torch.from_numpy(
        ((truth_K - mu) / sigma).astype(np.float32)
    ).to(device)
    truth_trace_metrics = rollout_interface_trace_metrics(
        truth_normalized,
        x_grid,
        y_grid,
        interface_x=float(physics["interface_x"]),
        resistance=float(params["R_c"]),
        dt=dt,
        k_left=float(physics["k_left"]),
        k_right=float(physics["k_right"]),
        rho=float(physics["rho_left"]),
        cp=float(physics["cp_left"]),
        sigma=sigma,
        q_ref=q_ref,
    )
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "mu_global": mu,
        "sigma_global": sigma,
        "probe_params": params,
        "source_checkpoint": str(source_path),
        "objective": (
            "state_conditioned_one_step_variational_cn_jump_enrichment"
            if args.jump_enrichment
            else "state_conditioned_one_step_variational_cn_interface_trace"
        ),
        "warm_start": bool(args.warm_start),
    }, output_dir / "cvit_init.pt")
    np.savez_compressed(
        output_dir / "initial_rollout.npz",
        prediction_K=initial_prediction_K,
        truth_K=truth_K,
        t_grid=time_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        flux_head_q=(
            q_ref * initial_head_flux.cpu().numpy()
            if initial_head_flux is not None else np.empty((0, solver.Ny))
        ),
        storage_projection_K=(sigma * initial_storage_correction).cpu().numpy(),
        implied_flux_gap=initial_implied_flux_gap.cpu().numpy(),
    )

    metric_names = list(initial_metrics)
    fields = [
        "update", "causal_stage", "causal_fraction", "sampled_step",
        "buffer_refresh", "variational_objective", "cn_residual_rms_normalized",
        "interface_flux_loss", "interface_contact_loss",
        "sample_trace_flux_rms_normalized", "sample_trace_contact_rms_normalized",
        "sample_trace_q_series_rms", "sample_trace_jump_rms_K",
        "flux_head_series_loss", "flux_head_left_energy_loss",
        "flux_head_right_energy_loss", "sample_flux_head_q_rms",
        "sample_flux_head_series_q_rms", "sample_flux_head_series_rms",
        "sample_flux_head_left_energy_rms", "sample_flux_head_right_energy_rms",
        "sample_storage_projection_rms_K", "sample_implied_flux_gap",
        "storage_projection_loss",
        "grad_norm_variational", "grad_norm_interface_flux",
        "grad_norm_interface_contact", "grad_cos_variational_interface_flux",
        "grad_cos_variational_interface_contact",
        "grad_norm_flux_head_constraints",
        "grad_cos_variational_flux_head_constraints",
        "grad_norm_storage_projection",
        "grad_cos_variational_storage_projection",
        "gradient_norm", "lr", "elapsed_seconds", *metric_names,
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
            "buffer_refresh": True,
            "variational_objective": "",
            "cn_residual_rms_normalized": "",
            "interface_flux_loss": "",
            "interface_contact_loss": "",
            "sample_trace_flux_rms_normalized": "",
            "sample_trace_contact_rms_normalized": "",
            "sample_trace_q_series_rms": "",
            "sample_trace_jump_rms_K": "",
            "flux_head_series_loss": "",
            "flux_head_left_energy_loss": "",
            "flux_head_right_energy_loss": "",
            "sample_flux_head_q_rms": "",
            "sample_flux_head_series_q_rms": "",
            "sample_flux_head_series_rms": "",
            "sample_flux_head_left_energy_rms": "",
            "sample_flux_head_right_energy_rms": "",
            "sample_storage_projection_rms_K": "",
            "sample_implied_flux_gap": "",
            "storage_projection_loss": "",
            "grad_norm_variational": "",
            "grad_norm_interface_flux": "",
            "grad_norm_interface_contact": "",
            "grad_cos_variational_interface_flux": "",
            "grad_cos_variational_interface_contact": "",
            "grad_norm_flux_head_constraints": "",
            "grad_cos_variational_flux_head_constraints": "",
            "grad_norm_storage_projection": "",
            "grad_cos_variational_storage_projection": "",
            "gradient_norm": "",
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": 0.0,
            **initial_metrics,
        })
    print(
        f"[one-step-cvit] device={device} updates={args.updates} "
        f"grid={solver.Nx}x{solver.Ny} steps={len(time_grid) - 1} "
        f"forcing_image={ny_image}x{args.image_time_points} "
        f"case={args.probe_case} warm_start={args.warm_start} "
        f"trace_weights=({args.trace_flux_weight:g},{args.trace_contact_weight:g}) "
        f"jump_enrichment={args.jump_enrichment} "
        f"flux_conditioning={args.jump_flux_conditioning} "
        f"flux_mode={args.jump_flux_mode} "
        f"flux_weights=({args.flux_series_weight:g},"
        f"{args.flux_left_energy_weight:g},{args.flux_right_energy_weight:g}) "
        f"storage_projection_weight={args.storage_projection_weight:g} "
        f"out={output_dir}",
        flush=True,
    )
    print(
        f"Update 0: field={initial_metrics['field_rmse_K']:.4f}K "
        f"response={initial_metrics['temporal_response_ratio']:.3e} "
        f"jump={initial_metrics['pred_jump_rms_K']:.4f}/"
        f"{initial_metrics['true_jump_rms_K']:.4f}K",
        flush=True,
    )

    stage_updates = [int(value) for value in args.stage_updates]
    stage_fractions = [float(value) for value in args.stage_fractions]
    if len(stage_updates) != len(stage_fractions) or stage_updates[0] != 0:
        raise ValueError("stage updates/fractions must align and begin at zero")
    rng = np.random.default_rng(args.seed + 8093)
    start = time.perf_counter()
    previous_stage = -1
    best_score = float("inf")
    best_update = 0
    last_training = {}

    for update in range(1, int(args.updates) + 1):
        stage = max(i for i, value in enumerate(stage_updates) if update - 1 >= value)
        fraction = stage_fractions[stage]
        active_steps = max(1, int(math.ceil((len(time_grid) - 1) * fraction)))
        refresh = (
            stage != previous_stage
            or (update - 1) % int(args.buffer_refresh_every) == 0
        )
        if refresh:
            refreshed = causal_rollout(
                model,
                initial_state,
                interval_images,
                scalars,
                fixed_channels,
                coords,
                dt=dt,
                query_chunk=args.query_chunk,
                steps=active_steps,
                interface_x=interface_x_tensor,
                jump_scale=jump_scale,
                closure_geom=trace_geom,
                q_left_integrals=q_left_integrals,
                resistance=float(params["R_c"]),
                sigma=sigma,
                q_ref=q_ref,
            )
            if state_buffer.shape[0] < len(time_grid):
                raise RuntimeError("state buffer has an invalid shape")
            state_buffer[: active_steps + 1] = refreshed
        previous_stage = stage
        step = int(rng.integers(0, active_steps))
        state_n = state_buffer[step : step + 1].detach()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction_result = predict_one_step(
            model,
            state_n,
            interval_images[step : step + 1],
            scalars,
            fixed_channels,
            coords,
            dt=dt,
            query_chunk=args.query_chunk,
            interface_x=interface_x_tensor,
            jump_scale=jump_scale,
            return_interface_flux=bool(args.jump_enrichment),
            closure_geom=trace_geom,
            q_left_integral=q_left_integrals[step : step + 1],
            resistance=float(params["R_c"]),
            sigma=sigma,
            q_ref=q_ref,
            return_storage_projection=uses_storage_projection,
        )
        if uses_storage_projection:
            (
                prediction_np1,
                flux_head_np1,
                storage_correction,
                implied_flux_gap,
            ) = prediction_result
        elif args.jump_enrichment:
            prediction_np1, flux_head_np1 = prediction_result
            storage_correction = prediction_np1.new_zeros(prediction_np1.shape)
            implied_flux_gap = prediction_np1.new_zeros((prediction_np1.shape[0],))
        else:
            prediction_np1 = prediction_result
            storage_correction = prediction_np1.new_zeros(prediction_np1.shape)
            implied_flux_gap = prediction_np1.new_zeros((prediction_np1.shape[0],))
        if args.jump_enrichment:
            flux_head = interface_flux_head_physics_terms(
                state_n,
                prediction_np1,
                flux_head_np1,
                trace_geom,
                q_left_integrals[step : step + 1],
                right_value=right_value,
                q_ref=q_ref,
                sigma=sigma,
            )
        else:
            prediction_np1 = prediction_result
            flux_head = {
                key: prediction_np1.new_zeros(()) for key in (
                    "series_loss", "left_energy_loss", "right_energy_loss",
                    "series_rms", "left_energy_rms", "right_energy_rms",
                    "q_head_rms", "q_series_rms",
                )
            }
        variational, residual_rms = causal_variational_terms(
            prediction_np1,
            state_n,
            torch.tensor([step], device=device, dtype=torch.long),
            cn,
            right_value=right_value,
        )
        trace = one_step_interface_trace_terms(
            prediction_np1,
            trace_geom,
            x_grid,
            float(params["R_c"]),
            k_left=float(physics["k_left"]),
            k_right=float(physics["k_right"]),
            sigma=sigma,
            q_ref=q_ref,
        )
        storage_projection_loss = storage_correction.square().mean()
        total_loss = (
            variational
            + float(args.trace_flux_weight) * trace["flux_loss"]
            + float(args.trace_contact_weight) * trace["contact_loss"]
            + float(args.flux_series_weight) * flux_head["series_loss"]
            + float(args.flux_left_energy_weight) * flux_head["left_energy_loss"]
            + float(args.flux_right_energy_weight) * flux_head["right_energy_loss"]
            + float(args.storage_projection_weight) * storage_projection_loss
        )
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("non-finite one-step physics objective")
        do_eval = update % int(args.validate_every) == 0 or update == int(args.updates)
        gradient_geometry = {}
        if do_eval:
            gradient_losses = {
                    "variational": variational,
                    "interface_flux": trace["flux_loss"],
                    "interface_contact": trace["contact_loss"],
                }
            if args.jump_enrichment:
                gradient_losses["flux_head_constraints"] = (
                    float(args.flux_series_weight) * flux_head["series_loss"]
                    + float(args.flux_left_energy_weight) * flux_head["left_energy_loss"]
                    + float(args.flux_right_energy_weight) * flux_head["right_energy_loss"]
                )
            if float(args.storage_projection_weight) > 0.0:
                gradient_losses["storage_projection"] = (
                    float(args.storage_projection_weight)
                    * storage_projection_loss
                )
            gradient_geometry = loss_gradient_geometry(
                gradient_losses,
                model.parameters(),
            )
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.grad_clip)
        )
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        last_training = {
            "variational_objective": float(variational.detach().cpu()),
            "cn_residual_rms_normalized": float(residual_rms.detach().cpu()),
            "interface_flux_loss": float(trace["flux_loss"].detach().cpu()),
            "interface_contact_loss": float(trace["contact_loss"].detach().cpu()),
            "sample_trace_flux_rms_normalized": float(trace["flux_rms"].detach().cpu()),
            "sample_trace_contact_rms_normalized": float(
                trace["contact_rms"].detach().cpu()
            ),
            "sample_trace_q_series_rms": float(trace["q_series_rms"].detach().cpu()),
            "sample_trace_jump_rms_K": float(trace["jump_rms_K"].detach().cpu()),
            "flux_head_series_loss": float(flux_head["series_loss"].detach().cpu()),
            "flux_head_left_energy_loss": float(
                flux_head["left_energy_loss"].detach().cpu()
            ),
            "flux_head_right_energy_loss": float(
                flux_head["right_energy_loss"].detach().cpu()
            ),
            "sample_flux_head_q_rms": float(flux_head["q_head_rms"].detach().cpu()),
            "sample_flux_head_series_q_rms": float(
                flux_head["q_series_rms"].detach().cpu()
            ),
            "sample_flux_head_series_rms": float(
                flux_head["series_rms"].detach().cpu()
            ),
            "sample_flux_head_left_energy_rms": float(
                flux_head["left_energy_rms"].detach().cpu()
            ),
            "sample_flux_head_right_energy_rms": float(
                flux_head["right_energy_rms"].detach().cpu()
            ),
            "sample_storage_projection_rms_K": float(
                (sigma * storage_correction).square().mean().sqrt().detach().cpu()
            ),
            "sample_implied_flux_gap": float(
                implied_flux_gap.square().mean().sqrt().detach().cpu()
            ),
            "storage_projection_loss": float(
                storage_projection_loss.detach().cpu()
            ),
            **gradient_geometry,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

        if not do_eval:
            continue
        rollout_result = causal_rollout(
            model,
            initial_state,
            interval_images,
            scalars,
            fixed_channels,
            coords,
            dt=dt,
            query_chunk=args.query_chunk,
            steps=len(time_grid) - 1,
            interface_x=interface_x_tensor,
            jump_scale=jump_scale,
            return_interface_flux=bool(args.jump_enrichment),
            closure_geom=trace_geom,
            q_left_integrals=q_left_integrals,
            resistance=float(params["R_c"]),
            sigma=sigma,
            q_ref=q_ref,
            return_storage_projection=uses_storage_projection,
        )
        if uses_storage_projection:
            (
                rollout,
                rollout_head_flux,
                rollout_storage_correction,
                rollout_implied_flux_gap,
            ) = rollout_result
        elif args.jump_enrichment:
            rollout, rollout_head_flux = rollout_result
            rollout_storage_correction = rollout.new_zeros(
                rollout.shape[0] - 1, *rollout.shape[1:]
            )
            rollout_implied_flux_gap = rollout.new_zeros(rollout.shape[0] - 1)
        else:
            rollout = rollout_result
            rollout_head_flux = None
            rollout_storage_correction = rollout.new_zeros(
                rollout.shape[0] - 1, *rollout.shape[1:]
            )
            rollout_implied_flux_gap = rollout.new_zeros(rollout.shape[0] - 1)
        prediction_K = rollout.cpu().numpy() * sigma + mu
        metrics = _forcing_supervised_metrics(
            prediction_K, truth_K, initial_prediction_K, interface_face
        )
        metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
            prediction_K, solver, sigma=sigma
        )
        metrics.update(rollout_interface_trace_metrics(
            rollout,
            x_grid,
            y_grid,
            interface_x=float(physics["interface_x"]),
            resistance=float(params["R_c"]),
            dt=dt,
            k_left=float(physics["k_left"]),
            k_right=float(physics["k_right"]),
            rho=float(physics["rho_left"]),
            cp=float(physics["cp_left"]),
            sigma=sigma,
            q_ref=q_ref,
        ))
        metrics["flux_head_q_rms"] = (
            float((q_ref * rollout_head_flux).square().mean().sqrt().cpu())
            if rollout_head_flux is not None else 0.0
        )
        metrics["storage_projection_rms_K"] = float(
            (sigma * rollout_storage_correction).square().mean().sqrt().cpu()
        )
        metrics["implied_flux_gap_rms"] = float(
            rollout_implied_flux_gap.square().mean().sqrt().cpu()
        )
        rollout_head_physical = (
            q_ref * rollout_head_flux.cpu().numpy()
            if rollout_head_flux is not None else np.zeros_like(true_interface_flux)
        )
        rollout_series_flux = np.asarray(solver.G_x[interface_face])[None] * (
            prediction_K[1:, interface_face]
            - prediction_K[1:, interface_face + 1]
        )
        metrics.update(interface_flux_agreement_metrics(
            rollout_head_physical, true_interface_flux, prefix="flux_head_truth"
        ))
        metrics.update(interface_flux_agreement_metrics(
            rollout_series_flux, true_interface_flux, prefix="flux_series_truth"
        ))
        elapsed = time.perf_counter() - start
        row = {
            "update": update,
            "causal_stage": stage,
            "causal_fraction": fraction,
            "sampled_step": step,
            "buffer_refresh": refresh,
            **last_training,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": elapsed,
            **metrics,
        }
        with metrics_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        response_ok = metrics["temporal_response_ratio"] >= 0.1
        score = metrics["field_rmse_K"] + metrics["node_jump_rmse_K"]
        if response_ok and score < best_score:
            best_score = score
            best_update = update
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "completed_updates": update,
                "metrics": metrics,
                "probe_params": params,
                "source_checkpoint": str(source_path),
            }, output_dir / "cvit_best_noncollapsed.pt")
        torch.save({
            "model_state": model.state_dict(),
            "config": config,
            "completed_updates": update,
            "metrics": metrics,
            "probe_params": params,
            "source_checkpoint": str(source_path),
        }, output_dir / f"cvit_update_{update:04d}.pt")
        print(
            f"Update {update}: stage={stage}/{fraction:.2f} step={step} "
            f"E={last_training['variational_objective']:.3e} "
            f"r={last_training['cn_residual_rms_normalized']:.3e} "
            f"trace=({last_training['sample_trace_flux_rms_normalized']:.2e},"
            f"{last_training['sample_trace_contact_rms_normalized']:.2e}) "
            f"qhead={last_training['sample_flux_head_q_rms']:.2f} "
            f"proj={last_training['sample_storage_projection_rms_K']:.2f}K "
            f"field={metrics['field_rmse_K']:.4f}K "
            f"response={metrics['temporal_response_ratio']:.3e} "
            f"jump={metrics['pred_jump_rms_K']:.4f}/"
            f"{metrics['true_jump_rms_K']:.4f}K "
            f"rollout_r={metrics['rollout_residual_rms_normalized']:.3e} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    final_result = causal_rollout(
        model,
        initial_state,
        interval_images,
        scalars,
        fixed_channels,
        coords,
        dt=dt,
        query_chunk=args.query_chunk,
        steps=len(time_grid) - 1,
        interface_x=interface_x_tensor,
        jump_scale=jump_scale,
        return_interface_flux=bool(args.jump_enrichment),
        closure_geom=trace_geom,
        q_left_integrals=q_left_integrals,
        resistance=float(params["R_c"]),
        sigma=sigma,
        q_ref=q_ref,
        return_storage_projection=uses_storage_projection,
    )
    if uses_storage_projection:
        (
            final_rollout,
            final_head_flux,
            final_storage_correction,
            final_implied_flux_gap,
        ) = final_result
    elif args.jump_enrichment:
        final_rollout, final_head_flux = final_result
        final_storage_correction = final_rollout.new_zeros(
            final_rollout.shape[0] - 1, *final_rollout.shape[1:]
        )
        final_implied_flux_gap = final_rollout.new_zeros(final_rollout.shape[0] - 1)
    else:
        final_rollout = final_result
        final_head_flux = None
        final_storage_correction = final_rollout.new_zeros(
            final_rollout.shape[0] - 1, *final_rollout.shape[1:]
        )
        final_implied_flux_gap = final_rollout.new_zeros(final_rollout.shape[0] - 1)
    final_prediction_K = final_rollout.cpu().numpy() * sigma + mu
    final_metrics = _forcing_supervised_metrics(
        final_prediction_K, truth_K, initial_prediction_K, interface_face
    )
    final_metrics["rollout_residual_rms_normalized"] = rollout_residual_rms(
        final_prediction_K, solver, sigma=sigma
    )
    final_metrics.update(rollout_interface_trace_metrics(
        final_rollout,
        x_grid,
        y_grid,
        interface_x=float(physics["interface_x"]),
        resistance=float(params["R_c"]),
        dt=dt,
        k_left=float(physics["k_left"]),
        k_right=float(physics["k_right"]),
        rho=float(physics["rho_left"]),
        cp=float(physics["cp_left"]),
        sigma=sigma,
        q_ref=q_ref,
    ))
    final_metrics["flux_head_q_rms"] = (
        float((q_ref * final_head_flux).square().mean().sqrt().cpu())
        if final_head_flux is not None else 0.0
    )
    final_metrics["storage_projection_rms_K"] = float(
        (sigma * final_storage_correction).square().mean().sqrt().cpu()
    )
    final_metrics["implied_flux_gap_rms"] = float(
        final_implied_flux_gap.square().mean().sqrt().cpu()
    )
    final_head_physical = (
        q_ref * final_head_flux.cpu().numpy()
        if final_head_flux is not None else np.zeros_like(true_interface_flux)
    )
    final_series_flux = np.asarray(solver.G_x[interface_face])[None] * (
        final_prediction_K[1:, interface_face]
        - final_prediction_K[1:, interface_face + 1]
    )
    final_metrics.update(interface_flux_agreement_metrics(
        final_head_physical, true_interface_flux, prefix="flux_head_truth"
    ))
    final_metrics.update(interface_flux_agreement_metrics(
        final_series_flux, true_interface_flux, prefix="flux_series_truth"
    ))
    elapsed = time.perf_counter() - start
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "completed_updates": int(args.updates),
        "metrics": final_metrics,
        "probe_params": params,
        "source_checkpoint": str(source_path),
        "warm_start": bool(args.warm_start),
    }, output_dir / "cvit_last.pt")
    np.savez_compressed(
        output_dir / "final_rollout.npz",
        prediction_K=final_prediction_K,
        truth_K=truth_K,
        t_grid=time_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        flux_head_q=(
            q_ref * final_head_flux.cpu().numpy()
            if final_head_flux is not None else np.empty((0, solver.Ny))
        ),
        storage_projection_K=(sigma * final_storage_correction).cpu().numpy(),
        implied_flux_gap=final_implied_flux_gap.cpu().numpy(),
    )
    summary = {
        "hypothesis": (
            "A modest penalty on the conservative storage correction can make "
            "the smooth decoder internalize the right-layer energy balance, "
            "reducing projection dependence without sacrificing the improved "
            "field, forcing-response, and jump amplitudes."
            if (
                args.jump_flux_mode == "conservative_storage_projection"
                and float(args.storage_projection_weight) > 0.0
            ) else
            "An exact two-variable conservative projection can retain the "
            "left-energy closure's stable flux timing while a trace-neutral "
            "right-layer storage mode removes the incompatible second-layer "
            "balance error that previously inflated interface-flux amplitude."
            if args.jump_flux_mode == "conservative_storage_projection" else
            "A normalized least-squares flux closure using both layer-energy "
            "identities can retain the stable diffusive timing of the left-only "
            "closure while the right layer corrects its excess transmitted-flux "
            "amplitude."
            if args.jump_flux_mode == "two_sided_energy_closure" else
            "An analytic interface flux solved from the exact left-layer energy "
            "identity can drive the moving jump enrichment without either direct "
            "forcing-copy shortcuts or an unstable freely learned recurrent flux."
            if args.jump_flux_mode == "left_energy_closure" else
            "A moving-interface Heaviside enrichment driven by an explicit "
            "physics-constrained interface-flux head lets one global CViT express "
            "the contact jump without subdomain decomposition or loss-mediated "
            "formation of an unresolved steep layer."
            if args.jump_enrichment else
            "After variational one-step training has established nonzero heat "
            "transport, explicit one-sided interface flux and contact-law "
            "constraints can convert that transport into the missing temperature "
            "jump without collapsing the forcing response."
        ),
        "source_checkpoint": str(source_path),
        "probe_case": args.probe_case,
        "seed": int(args.seed),
        "completed_updates": int(args.updates),
        "best_noncollapsed_update": int(best_update),
        "wall_seconds": elapsed,
        "objective": {
            "kind": "state_conditioned_one_step_variational_cn",
            "temperature_labels": 0,
            "warm_start": bool(args.warm_start),
            "trace_flux_weight": float(args.trace_flux_weight),
            "trace_contact_weight": float(args.trace_contact_weight),
            "jump_enrichment": bool(args.jump_enrichment),
            "jump_flux_depth": int(args.jump_flux_depth),
            "jump_flux_conditioning": str(args.jump_flux_conditioning),
            "jump_flux_mode": str(args.jump_flux_mode),
            "flux_series_weight": float(args.flux_series_weight),
            "flux_left_energy_weight": float(args.flux_left_energy_weight),
            "flux_right_energy_weight": float(args.flux_right_energy_weight),
            "storage_projection_weight": float(args.storage_projection_weight),
            "buffer_refresh_every": int(args.buffer_refresh_every),
            "forcing_image_shape": [ny_image, int(args.image_time_points)],
            "stage_updates": stage_updates,
            "stage_fractions": stage_fractions,
        },
        "initial_metrics": initial_metrics,
        "truth_trace_metrics": truth_trace_metrics,
        "final_metrics": final_metrics,
        "last_training_values": last_training,
    }
    with (output_dir / "final_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[one-step-cvit] done -> {output_dir / 'final_metrics.json'}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-case", default="pulse_low")
    parser.add_argument("--train-probe-cases", nargs="+", default=None)
    parser.add_argument("--holdout-probe-cases", nargs="+", default=None)
    parser.add_argument("--sampled-family-grid", action="store_true")
    parser.add_argument("--sampled-train-seed", type=int, default=1729)
    parser.add_argument("--sampled-holdout-seed", type=int, default=2718)
    parser.add_argument(
        "--sampled-train-per-combination", type=int, default=1
    )
    parser.add_argument(
        "--sampled-holdout-per-combination", type=int, default=1
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--buffer-refresh-every", type=int, default=50)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--image-time-points", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--trace-flux-weight", type=float, default=0.0)
    parser.add_argument("--trace-contact-weight", type=float, default=0.0)
    parser.add_argument("--jump-enrichment", action="store_true")
    parser.add_argument("--jump-flux-depth", type=int, default=1)
    parser.add_argument(
        "--jump-flux-conditioning", choices=("all", "state_param"), default="all"
    )
    parser.add_argument(
        "--jump-flux-mode",
        choices=(
            "learned",
            "left_energy_closure",
            "two_sided_energy_closure",
            "conservative_storage_projection",
        ),
        default="learned",
    )
    parser.add_argument("--flux-series-weight", type=float, default=0.0)
    parser.add_argument("--flux-left-energy-weight", type=float, default=0.0)
    parser.add_argument("--flux-right-energy-weight", type=float, default=0.0)
    parser.add_argument("--storage-projection-weight", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage-updates", type=int, nargs="+", default=[0, 250, 500, 750]
    )
    parser.add_argument(
        "--stage-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.train_probe_cases is not None or arguments.sampled_family_grid:
        run_mini_operator(arguments)
    else:
        run(arguments)
