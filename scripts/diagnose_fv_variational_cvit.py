#!/usr/bin/env python3
"""Single-problem causal variational-CN reachability gate for InterfaceCViT."""

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
from scripts.diagnose_fv_direct_state import active_system, normalized_rhs
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
    load_diffusion_data,
    normalize_forcing_interface_scalars,
)
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
    parser.add_argument("--source-checkpoint", required=True)
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
    parser.add_argument(
        "--stage-updates", type=int, nargs="+", default=[0, 250, 500, 750]
    )
    parser.add_argument(
        "--stage-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
