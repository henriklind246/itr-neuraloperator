"""Supervised single-sim overfit diagnostic (data MSE, NO physics).

The PINO path floors at val rel-L2 ~= 1.0 (the flat-mean predictor) even when
overfitting a single simulation with a dense IC anchor. That rules out loss
weighting and data volume and points at the model's ability to REPRESENT a
decaying trajectory at all. This script isolates that question: it trains the
CViT on one sim with a pure data objective -- query the model on grid nodes at
saved times and regress the (normalized) stored temperature -- with the residual,
IC, and Neumann terms all removed. If the model cannot drive rel-L2 well below
1.0 here, the failure is architectural (the leading suspect is a near time-blind
decoder: on t in [0, t_final=0.3] the shared fourier_freq=1.0 makes fourier_t
nearly constant across the trajectory).

The decisive comparison is freq_t=1 vs freq_t=10, holding everything else fixed:

  .venv/bin/python scripts/diagnose_supervised_overfit.py \
      model.cvit.fourier_freq_t=1  --epochs 400 --tag freqt1
  .venv/bin/python scripts/diagnose_supervised_overfit.py \
      model.cvit.fourier_freq_t=10 --epochs 400 --tag freqt10

If freq_t=1 stays pinned near 1.0 and freq_t=10 fits (rel-L2 -> small), the
time-blind-decoder diagnosis is causally confirmed and cheap. This is a
diagnostic only; it does not touch the training/eval paths.

For the layered forcing InterfaceCViT, pass ``--forcing-source-checkpoint``.
That mode rebuilds the architecture and normalization from the failed physics
checkpoint, generates the deterministic FV probe, and fits the pulse transient
with field-value and interface-trace supervision only.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operators.train import (
    _advance_scheduler,
    build_optimizer,
    build_scheduler,
    load_config,
    set_seed,
)
from src.operators.train_pino import (
    _decode_in_chunks,
    _forcing_fixed_probe_case,
    _interface_spatial_channels_from_normalized,
    _solve_forcing_diagnostic_truth,
    _validate_forcing_layered_dataset,
    build_cvit,
    build_forcing_image,
    load_diffusion_data,
    normalize_forcing_interface_scalars,
    validate_rel_l2,
)
from data.dataset import problem_from_config
from src.operators.utils import resolve_device

GROUP_ENV = {"benchmark": "BENCHMARK", "representation": "REPRESENTATION"}


def _parse_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _apply_override(config: dict, key_path: str, value: object) -> None:
    parts = key_path.split(".")
    current = config
    for idx, part in enumerate(parts[:-1]):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown config path: {'.'.join(parts[: idx + 1])}")
        current = current[part]
    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise KeyError(f"Unknown config path: {key_path}")
    current[leaf] = value


def _forcing_supervised_metrics(
    prediction_K: np.ndarray,
    truth_K: np.ndarray,
    init_prediction_K: np.ndarray,
    interface_face: int,
) -> dict[str, float]:
    prediction = np.asarray(prediction_K, dtype=np.float64)
    truth = np.asarray(truth_K, dtype=np.float64)
    initialization = np.asarray(init_prediction_K, dtype=np.float64)
    error = prediction - truth
    signal = truth - 300.0
    predicted_jump = (
        prediction[:, interface_face, :] - prediction[:, interface_face + 1, :]
    )
    true_jump = truth[:, interface_face, :] - truth[:, interface_face + 1, :]
    pred_temporal = prediction - prediction[0:1]
    true_temporal = truth - truth[0:1]
    signal_norm = float(np.sqrt(np.mean(signal ** 2)))
    temporal_norm = float(np.sqrt(np.mean(true_temporal ** 2)))
    true_jump_norm = float(np.sqrt(np.mean(true_jump ** 2)))
    pred_flat = (prediction - 300.0).reshape(-1)
    truth_flat = signal.reshape(-1)
    correlation = (
        float(np.corrcoef(pred_flat, truth_flat)[0, 1])
        if float(np.std(pred_flat)) > 0.0 and float(np.std(truth_flat)) > 0.0
        else float("nan")
    )
    return {
        "field_rmse_K": float(np.sqrt(np.mean(error ** 2))),
        "field_signal_rel_l2": float(
            np.sqrt(np.mean(error ** 2)) / max(signal_norm, 1e-12)
        ),
        "field_max_abs_error_K": float(np.max(np.abs(error))),
        "field_signal_correlation": correlation,
        "departure_from_300_K": float(np.sqrt(np.mean((prediction - 300.0) ** 2))),
        "drift_from_init_K": float(
            np.sqrt(np.mean((prediction - initialization) ** 2))
        ),
        "temporal_response_ratio": float(
            np.sqrt(np.mean(pred_temporal ** 2)) / max(temporal_norm, 1e-12)
        ),
        "pred_jump_rms_K": float(np.sqrt(np.mean(predicted_jump ** 2))),
        "true_jump_rms_K": true_jump_norm,
        "node_jump_rmse_K": float(
            np.sqrt(np.mean((predicted_jump - true_jump) ** 2))
        ),
        "jump_response_ratio": float(
            np.sqrt(np.mean(predicted_jump ** 2)) / max(true_jump_norm, 1e-12)
        ),
        "left_interface_trace_rmse_K": float(np.sqrt(np.mean(
            error[:, interface_face, :] ** 2
        ))),
        "right_interface_trace_rmse_K": float(np.sqrt(np.mean(
            error[:, interface_face + 1, :] ** 2
        ))),
    }


def _predict_forcing_probe(
    model: torch.nn.Module,
    spatial: torch.Tensor,
    forcing_image: torch.Tensor,
    scalars: torch.Tensor,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_values: np.ndarray,
    mu: float,
    sigma: float,
    query_chunk: int,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    coords = torch.from_numpy(np.stack(
        (gx.reshape(-1), gy.reshape(-1)), axis=-1,
    ).astype(np.float32)).unsqueeze(0).to(device)
    outputs = []
    with torch.no_grad():
        latent = model.encode(spatial, forcing_image, scalars)
        for value in t_values:
            time_query = torch.full(
                (1, coords.shape[1], 1), float(value), device=device,
            )
            prediction = _decode_in_chunks(
                model, latent, coords, time_query, int(query_chunk),
            )
            outputs.append(
                prediction[0, :, 0].view(len(x_grid), len(y_grid)).cpu().numpy()
            )
    if was_training:
        model.train()
    return np.stack(outputs).astype(np.float64) * float(sigma) + float(mu)


def _run_forcing_supervised(
    args: argparse.Namespace,
    dotted: list[str],
) -> int:
    source_path = Path(args.forcing_source_checkpoint).expanduser().resolve()
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    for raw in dotted:
        key_path, raw_value = raw.split("=", 1)
        _apply_override(config, key_path, _parse_value(raw_value))
    config["training"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["training"]["learning_rate"] = float(args.lr)

    if config.get("benchmark", {}).get("name") != "forcing":
        raise ValueError("--forcing-source-checkpoint requires benchmark=forcing")
    set_seed(args.seed)
    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = float(data["mu_global"]), float(data["sigma_global"])
    x_grid = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    Nx, Ny = len(x_grid), len(y_grid)

    image_meta = dict(
        checkpoint.get("interface_forcing")
        or checkpoint.get("forcing_image")
        or {}
    )
    fv_dt = float(image_meta["fv_dt"])
    ramp_meta = image_meta.get("ramp", {}) or {}
    t_ramp = float(ramp_meta.get("duration", 0.0))
    q_ref = float(image_meta.get("a_ref", 300.0))
    ny_img = int(image_meta.get("ny_img", 96))
    nt_img = int(image_meta.get("nt_img", 256))
    y_img = np.linspace(float(y_grid[0]), float(y_grid[-1]), ny_img)
    t_img = np.linspace(0.0, float(t_grid[-1]), nt_img)

    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    problem = problem_from_config(config)
    physics, interface_face, _ = _validate_forcing_layered_dataset(
        problem, data, sim_params, fv_dt,
    )
    params = _forcing_fixed_probe_case(str(args.probe_case))
    truth_K = _solve_forcing_diagnostic_truth(
        problem, params, physics, x_grid, y_grid, t_grid, fv_dt, t_ramp,
    )
    truth_norm = torch.from_numpy(
        ((truth_K - mu) / sigma).astype(np.float32)
    ).to(device)

    fixed_ic = np.full(
        (Nx, Ny), np.float32((float(physics["T_right"]) - mu) / sigma),
    )
    spatial = torch.from_numpy(_interface_spatial_channels_from_normalized(
        fixed_ic, float(physics["interface_x"]), x_grid,
    )).unsqueeze(0).to(device)
    forcing_image = build_forcing_image(
        [params], y_img, t_img, q_ref, device, t_ramp,
    )
    scalars = torch.from_numpy(normalize_forcing_interface_scalars(
        float(physics["interface_x"]), float(params["R_c"]),
    )).unsqueeze(0).to(device)

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=float(t_grid[-1]),
        variant="interfaces",
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    out_dir = Path(args.out).expanduser() if args.out else (
        Path(str(config["paths"]["runs_root"]))
        / "supervised_overfit" / args.tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "supervised_metrics.csv"
    eval_indices = np.unique(np.linspace(
        0, len(t_grid) - 1, max(int(args.eval_times), 2), dtype=int,
    ))
    eval_times = t_grid[eval_indices]
    truth_eval = truth_K[eval_indices]
    init_prediction_full = _predict_forcing_probe(
        model, spatial, forcing_image, scalars, x_grid, y_grid, t_grid,
        mu, sigma, args.query_chunk, device,
    )
    init_prediction = init_prediction_full[eval_indices]
    init_metrics = _forcing_supervised_metrics(
        init_prediction, truth_eval, init_prediction, interface_face,
    )
    fields = [
        "update", "data_mse", "field_data_mse", "interface_trace_mse", "lr",
        *init_metrics.keys(), "elapsed_seconds",
    ]
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "update": 0, "data_mse": "", "field_data_mse": "",
            "interface_trace_mse": "", "lr": optimizer.param_groups[0]["lr"],
            **init_metrics, "elapsed_seconds": 0.0,
        })
    torch.save({
        "model_state": model.state_dict(), "config": config,
        "mu_global": mu, "sigma_global": sigma, "probe_params": params,
        "source_checkpoint": str(source_path),
    }, out_dir / "cvit_supervised_init.pt")
    np.savez_compressed(
        out_dir / "initial_predictions.npz",
        prediction_K=init_prediction_full,
        truth_K=truth_K,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
    )

    print(
        f"[forcing-supervised] case={args.probe_case} device={device} "
        f"updates={args.epochs} field_q={args.n_q} interface_q={args.interface_q} "
        f"grid={Nx}x{Ny} Nt={len(t_grid)} sigma={sigma:.6f}K out={out_dir}",
        flush=True,
    )
    print(
        f"Update 0: field_rmse={init_metrics['field_rmse_K']:.4f}K "
        f"jump_rmse={init_metrics['node_jump_rmse_K']:.4f}K "
        f"jump={init_metrics['pred_jump_rms_K']:.4f}/"
        f"{init_metrics['true_jump_rms_K']:.4f}K",
        flush=True,
    )

    rng = np.random.default_rng(args.seed + 1729)
    x_tensor = torch.as_tensor(x_grid, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_grid, dtype=torch.float32, device=device)
    t_tensor = torch.as_tensor(t_grid, dtype=torch.float32, device=device)
    start = time.perf_counter()
    latest_metrics = init_metrics
    best_score = float("inf")
    best_update = 0
    last_losses = {"data_mse": float("nan"), "field_data_mse": float("nan"),
                   "interface_trace_mse": float("nan")}

    for update in range(1, int(args.epochs) + 1):
        model.train()
        field_t = torch.from_numpy(rng.integers(
            0, len(t_grid), size=int(args.n_q), dtype=np.int64,
        )).to(device)
        field_x = torch.from_numpy(rng.integers(
            0, Nx, size=int(args.n_q), dtype=np.int64,
        )).to(device)
        field_y = torch.from_numpy(rng.integers(
            0, Ny, size=int(args.n_q), dtype=np.int64,
        )).to(device)
        trace_t = torch.from_numpy(rng.integers(
            0, len(t_grid), size=int(args.interface_q), dtype=np.int64,
        )).to(device)
        trace_side = torch.from_numpy(rng.integers(
            0, 2, size=int(args.interface_q), dtype=np.int64,
        )).to(device)
        trace_x = trace_side + int(interface_face)
        trace_y = torch.from_numpy(rng.integers(
            0, Ny, size=int(args.interface_q), dtype=np.int64,
        )).to(device)

        coords_field = torch.stack((x_tensor[field_x], y_tensor[field_y]), dim=-1)
        coords_trace = torch.stack((x_tensor[trace_x], y_tensor[trace_y]), dim=-1)
        coords = torch.cat((coords_field, coords_trace), dim=0).unsqueeze(0)
        query_time = torch.cat((t_tensor[field_t], t_tensor[trace_t]), dim=0)
        query_time = query_time.view(1, -1, 1)
        target_field = truth_norm[field_t, field_x, field_y].view(1, -1, 1)
        target_trace = truth_norm[trace_t, trace_x, trace_y].view(1, -1, 1)

        optimizer.zero_grad(set_to_none=True)
        latent = model.encode(spatial, forcing_image, scalars)
        prediction = _decode_in_chunks(
            model, latent, coords, query_time, int(args.query_chunk),
        )
        pred_field = prediction[:, : int(args.n_q)]
        pred_trace = prediction[:, int(args.n_q):]
        loss_field = (pred_field - target_field).square().mean()
        loss_trace = (pred_trace - target_trace).square().mean()
        loss = loss_field + float(args.interface_weight) * loss_trace
        loss.backward()
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        last_losses = {
            "data_mse": float(loss.detach().cpu()),
            "field_data_mse": float(loss_field.detach().cpu()),
            "interface_trace_mse": float(loss_trace.detach().cpu()),
        }

        do_eval = update % int(args.validate_every) == 0 or update == int(args.epochs)
        if not do_eval:
            continue
        prediction_eval = _predict_forcing_probe(
            model, spatial, forcing_image, scalars, x_grid, y_grid, eval_times,
            mu, sigma, args.query_chunk, device,
        )
        latest_metrics = _forcing_supervised_metrics(
            prediction_eval, truth_eval, init_prediction, interface_face,
        )
        elapsed = time.perf_counter() - start
        lr = float(optimizer.param_groups[0]["lr"])
        with open(metrics_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow({
                "update": update, **last_losses, "lr": lr,
                **latest_metrics, "elapsed_seconds": elapsed,
            })
        score = latest_metrics["field_rmse_K"] + latest_metrics["node_jump_rmse_K"]
        if score < best_score:
            best_score = score
            best_update = update
            torch.save({
                "model_state": model.state_dict(), "config": config,
                "mu_global": mu, "sigma_global": sigma,
                "probe_params": params, "completed_updates": update,
                "metrics": latest_metrics, "source_checkpoint": str(source_path),
            }, out_dir / "cvit_supervised_best.pt")
        print(
            f"Update {update}: loss={last_losses['data_mse']:.3e} "
            f"field_rmse={latest_metrics['field_rmse_K']:.4f}K "
            f"signal_rel={latest_metrics['field_signal_rel_l2'] * 100:.3f}% "
            f"jump_rmse={latest_metrics['node_jump_rmse_K']:.4f}K "
            f"jump={latest_metrics['pred_jump_rms_K']:.3f}/"
            f"{latest_metrics['true_jump_rms_K']:.3f}K "
            f"time_ratio={latest_metrics['temporal_response_ratio']:.3f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    full_prediction = _predict_forcing_probe(
        model, spatial, forcing_image, scalars, x_grid, y_grid, t_grid,
        mu, sigma, args.query_chunk, device,
    )
    full_metrics = _forcing_supervised_metrics(
        full_prediction, truth_K, init_prediction_full, interface_face,
    )
    elapsed = time.perf_counter() - start
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "config": config,
        "mu_global": mu, "sigma_global": sigma, "probe_params": params,
        "completed_updates": int(args.epochs), "metrics": full_metrics,
        "source_checkpoint": str(source_path),
    }, out_dir / "cvit_supervised_last.pt")
    summary = {
        "source_checkpoint": str(source_path),
        "probe_case": str(args.probe_case),
        "seed": int(args.seed),
        "completed_updates": int(args.epochs),
        "best_update": int(best_update),
        "wall_seconds": float(elapsed),
        "training_objective": {
            "field_points_per_update": int(args.n_q),
            "interface_trace_points_per_update": int(args.interface_q),
            "interface_trace_weight": float(args.interface_weight),
            "physics_terms": 0,
        },
        "full_trajectory_metrics": full_metrics,
        "last_training_losses": last_losses,
    }
    with open(out_dir / "final_metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[forcing-supervised] done -> {out_dir / 'final_metrics.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Dotted key=value config overrides.")
    parser.add_argument("--sim-id", type=int, default=None,
                        help="Sim to overfit (default: first train id).")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--n-q", type=int, default=4096,
                        help="Random (space-node, time-node) points sampled per step.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training.learning_rate for this diagnostic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument("--tag", type=str, default="overfit1sim")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: runs_root/supervised_overfit/<tag>).")
    parser.add_argument("--forcing-source-checkpoint", type=str, default=None,
                        help="Run the layered forcing InterfaceCViT control using this checkpoint's exact config.")
    parser.add_argument("--probe-case", choices=("pulse_low", "pulse_high", "smooth_mid"),
                        default="pulse_low")
    parser.add_argument("--interface-q", type=int, default=512,
                        help="Supervised interface-trace samples per update.")
    parser.add_argument("--interface-weight", type=float, default=1.0)
    parser.add_argument("--eval-times", type=int, default=7)
    parser.add_argument("--query-chunk", type=int, default=512)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())

    dotted: list[str] = []
    for raw in args.overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
        key, val = raw.split("=", 1)
        if key in GROUP_ENV:
            os.environ[GROUP_ENV[key]] = val
        elif key == "model.type":
            continue
        else:
            dotted.append(raw)

    if args.forcing_source_checkpoint is not None:
        return _run_forcing_supervised(args, dotted)

    os.environ.setdefault("BENCHMARK", "diffusion")
    os.environ.setdefault("REPRESENTATION", "temporal_encoder")

    config = copy.deepcopy(load_config())
    for raw in dotted:
        key_path, raw_value = raw.split("=", 1)
        _apply_override(config, key_path, _parse_value(raw_value))
    if args.lr is not None:
        config["training"]["learning_rate"] = float(args.lr)

    set_seed(args.seed)
    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])

    sim_id = int(args.sim_id) if args.sim_id is not None else int(data["train_ids"][0])
    ids = np.asarray([sim_id])

    # Encoder input: this sim's normalized IC field.  Target: the full normalized
    # trajectory on grid nodes at saved times -- read exactly, no interpolation.
    ic = np.asarray(data["trajectories"][ids, 0, :, :], dtype=np.float32)
    u = torch.from_numpy((ic - mu) / (sigma + 1e-8)).unsqueeze(1).to(device)  # (1,1,Nx,Ny)

    traj = np.asarray(data["trajectories"][sim_id], dtype=np.float32)          # (Nt,Nx,Ny)
    traj_norm = torch.from_numpy((traj - mu) / (sigma + 1e-8)).to(device)
    Nt = traj_norm.shape[0]
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = torch.as_tensor(data["t_grid"], dtype=torch.float32, device=device)

    t_final = float(data["t_grid"][-1])
    fq_t = config["model"]["cvit"].get("fourier_freq_t", None)
    model = build_cvit(config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    out_dir = Path(args.out) if args.out else (
        Path(str(config["paths"]["runs_root"])) / "supervised_overfit" / args.tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "overfit_metrics.csv"
    fieldnames = ["epoch", "data_mse", "train_rel_l2", "train_rmse_K"]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(
        f"[overfit] sim_id={sim_id} device={device} epochs={args.epochs} n_q={args.n_q} "
        f"grid={Nx}x{Ny} Nt={Nt} | fourier_freq={config['model']['cvit'].get('fourier_freq')} "
        f"fourier_freq_t={fq_t} | lr={config['training'].get('learning_rate')} | out={out_dir}",
        flush=True,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        # Sample random grid-node/time-node points for this single sim.
        it = torch.randint(0, Nt, (args.n_q,), device=device, generator=gen)
        ix = torch.randint(0, Nx, (args.n_q,), device=device, generator=gen)
        iy = torch.randint(0, Ny, (args.n_q,), device=device, generator=gen)
        coords = torch.stack([x_grid[ix], y_grid[iy]], dim=-1).view(1, args.n_q, 2)
        tq = t_grid[it].view(1, args.n_q, 1)
        target = traj_norm[it, ix, iy].view(1, args.n_q, 1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(u, coords, tq)                 # (1, n_q, 1)
        loss = torch.mean((pred - target) ** 2)     # pure data MSE, no physics
        loss.backward()
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)

        row = {"epoch": epoch, "data_mse": float(loss.detach().cpu()),
               "train_rel_l2": "", "train_rmse_K": ""}
        do_val = (epoch % args.validate_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            model.eval()
            # rel-L2 of the model against THIS sim's full trajectory (train==sim).
            val = validate_rel_l2(model, data, ids, device)
            row["train_rel_l2"] = val["val_rel_l2"]
            row["train_rmse_K"] = val["val_rmse_K"]
            print(
                f"Epoch {epoch}: data_mse={row['data_mse']:.6e} "
                f"train_rel_l2={val['val_rel_l2'] * 100:.4f}% "
                f"train_rmse_K={val['val_rmse_K']:.4f}K lr={lr:.2e}",
                flush=True,
            )
        else:
            print(f"Epoch {epoch}: data_mse={row['data_mse']:.6e} lr={lr:.2e}", flush=True)

        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    print(f"[overfit] done -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
