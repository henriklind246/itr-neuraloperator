"""Development operator CN-FV screens with matched raw and exact objectives."""

from dataclasses import replace
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import shutil
import signal
import time

import numpy as np
import torch
import yaml

from data.dataset import assert_dataset_problem_version, compute_global_stats, split_sim_ids
from data.generate_dataset import build_base_setup
from problems.registry import get_problem
from src.operators.cvit import ForcingICCViT
from src.operators.fno2d import FNO2d
from src.operators.train import build_optimizer, build_scheduler, set_seed
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import build_interface_forcing, reconstruct_qL
from src.physics.fv_preconditioner import ExactCNInverse, MGOneCycleInverse
from src.physics.fv_residual import FullBCData, build_homogeneous_cn_geom, cn_step_residual
from src.physics.init_conditions import IC_FAMILIES


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_gate(directory, stage):
    path = Path(directory) / "final_metrics.json"
    result = json.loads(path.read_text())
    if result.get("stage") != stage or result.get("passed") is not True:
        raise ValueError(f"Required {stage} gate did not pass: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "kind": "gate"}


def load_pino_inputs(data_dir, normalization_path):
    """Load prescribed inputs and frozen stats without accessing future labels."""
    root = Path(data_dir)
    spec = get_problem("diffusion_forcing_single")
    meta = assert_dataset_problem_version(spec, root / "t_grid.npy")
    params = np.load(root / "sim_params.npy", allow_pickle=True)
    spec.validate_schema(params, np.arange(len(params)))
    x = np.load(root / "x_grid.npy").astype(np.float64)
    y = np.load(root / "y_grid.npy").astype(np.float64)
    times = np.load(root / "t_grid.npy").astype(np.float64)
    dt = float(np.load(root / "dt.npy"))
    ramp = float(np.load(root / "ramp_seconds.npy"))
    stats = json.loads(Path(normalization_path).read_text())
    mu, sigma = float(stats["mu_global"]), float(stats["sigma_global"])
    if not np.isfinite([mu, sigma, dt, ramp]).all() or sigma <= 0 or dt <= 0 or ramp < 0:
        raise ValueError("Invalid frozen normalization or time metadata")
    if len(x) != len(y) or not np.allclose(x, np.linspace(0, 1, len(x)), atol=1e-7) or not np.allclose(y, x, atol=1e-7):
        raise ValueError("The homogeneous screen requires a uniform isotropic unit-square grid")
    # Persisted float32 nodes are rounded; reconstruct the solver's uniform grid.
    x = np.linspace(0, 1, len(x))
    y = np.linspace(0, 1, len(y))
    for p in params:
        if np.shape(p["T0"]) != (len(x), len(y)) or not np.isfinite(p["T0"]).all():
            raise ValueError("Invalid prescribed IC field")
        if not np.allclose(p["T0"][-1], 300, atol=1e-5):
            raise ValueError("The prescribed IC must satisfy the right Dirichlet wall")
    horizon = float(round(times[-1], 7))
    if not np.isclose(horizon / dt, round(horizon / dt), atol=1e-5):
        raise ValueError("The horizon must contain an integer number of solver steps")
    return dict(spec=spec, params=params, x=x, y=y, dt=dt, ramp_seconds=ramp,
                t_final=horizon, mu_global=mu, sigma_global=sigma,
                normalization=stats, meta=meta)


def freeze_development_normalization(data_dir, destination):
    """Training-simulation-only solver statistics; development setup cost."""
    root = Path(data_dir)
    labels = np.load(root / "trajectories.npy", mmap_mode="r")
    train, val, test = split_sim_ids(labels.shape[0], seed=0)
    mu, sigma = compute_global_stats(labels, train)
    record = dict(mu_global=mu, sigma_global=sigma, source="training_reference_trajectories",
                  source_path=str((root / "trajectories.npy").resolve()),
                  source_sha256=file_sha256(root / "trajectories.npy"),
                  train_ids=train.tolist(), val_ids=val.tolist(), test_ids=test.tolist(),
                  split_seed=0, split_fractions=[0.7, 0.15, 0.15],
                  reference_access="development_normalization_only")
    Path(destination).write_text(json.dumps(record, indent=2) + "\n")
    return record


def build_forcing_image(params, y_img, t_img, a_ref, device, t_ramp):
    if a_ref <= 0:
        raise ValueError("forcing_a_ref must be positive")
    image = np.stack([
        reconstruct_qL(p["temporal_family"], p["temporal_params"],
                       p["spatial_family"], p["spatial_params"], t_ramp)
        .evaluate_grid(y_img, t_img) for p in params
    ]).astype(np.float32)[:, None] / a_ref
    return torch.from_numpy(image).to(device)


def build_cvit(config, inputs, device):
    options = copy.deepcopy(config["physics_test"]["cvit"])
    options.update(ic_grid_size=(len(inputs["x"]), len(inputs["y"])),
                   t_final=inputs["t_final"],
                   t_right_tilde=(300 - inputs["mu_global"]) / inputs["sigma_global"],
                   use_time_query=config["physics_test"]["pino"].get("residual_mode") != "autoregressive")
    options["forcing_grid_size"] = tuple(options["forcing_grid_size"])
    return ForcingICCViT(**options).to(device)


def build_screen_model(config, inputs, device):
    kind = config["physics_test"].get("model_type", "cvit")
    if kind == "cvit":
        return build_cvit(config, inputs, device)
    if kind != "fno":
        raise ValueError(f"Unknown physics-screen model {kind!r}")
    options = {key: value for key, value in config["model"]["parameters"].items()
               if key not in {"activation", "temporal_samples", "hard_right_dirichlet_t_right"}}
    dims = inputs["spec"].dims
    options.update(in_channels=dims.in_channels, cond_static_dim=dims.cond_static_dim,
                   temporal_token_dim=dims.temporal_token_dim, s_y_channel=dims.s_y_channel,
                   use_forcing_time_aug=dims.use_forcing_time_aug,
                   forcing_extender_rc_cond_index=dims.forcing_extender_rc_cond_index,
                   t_right_norm=(300 - inputs["mu_global"]) / inputs["sigma_global"])
    if config["physics_test"]["pino"].get("residual_mode") == "autoregressive" and not options.get("hard_right_dirichlet"):
        raise ValueError("Autoregressive FNO requires model.parameters.hard_right_dirichlet=true: the CN loss uses free nodes only")
    return FNO2d(**options).to(device)


def decode_grid(model, latent, coords, times, nx, ny, chunk_size):
    if isinstance(model, FNO2d):
        spatial, forcing_seq, horizon = latent
        query_times = torch.as_tensor(times, dtype=spatial.dtype, device=spatial.device)
        cond = query_times.reshape(-1, 1).expand(spatial.shape[0], 1) / horizon
        return model(spatial, cond, forcing_seq)[..., 0]
    batches = latent.shape[0]
    if times is not None:
        times = torch.as_tensor(times, dtype=coords.dtype, device=coords.device).reshape(-1, 1, 1)
        if times.shape[0] == 1:
            times = times.expand(batches, -1, -1)
    pieces = []
    for lo in range(0, coords.shape[1], chunk_size):
        query = coords[:, lo:lo + chunk_size].expand(batches, -1, -1)
        pieces.append(model.decode(latent, query, times.expand(-1, query.shape[1], -1) if times is not None else None))
    return torch.cat(pieces, dim=1).reshape(batches, nx, ny)


def anchored_previous(predicted, prescribed_ic, intervals):
    first = torch.as_tensor(np.asarray(intervals) == 0, device=predicted.device)[:, None, None]
    return torch.where(first, prescribed_ic, predicted).detach()


def cn_objective(previous, following, geom, bc, inverse, objective):
    residual = cn_step_residual(previous, following, geom, bc)
    weighted = inverse(residual) if objective in {"exact", "mg"} else residual
    if objective not in {"raw", "exact", "mg"}:
        raise ValueError(f"Unsupported development objective {objective!r}")
    return weighted.square().mean(), residual, weighted


def prefix_cn_objective(states, geom, boundaries, inverse, objective):
    """CN loss from an exact IC, with full temporal gradients for the block inverse."""
    if objective not in {"raw", "spatial_exact", "exact"}:
        raise ValueError(f"Unsupported prefix objective {objective!r}")
    residuals = torch.stack([
        cn_step_residual(states[:, n], states[:, n + 1], geom, bc,
                         detach_previous=objective == "spatial_exact")
        for n, bc in enumerate(boundaries)
    ], dim=1)
    transformed = (inverse.space_time(residuals) if objective == "exact"
                   else inverse(residuals) if objective == "spatial_exact" else residuals)
    return transformed.square().mean(), residuals, transformed


def evaluation_query_times(config, inputs):
    times = config["physics_test"]["pino"]["evaluation_times"]
    mode = config["physics_test"]["pino"].get("residual_mode", "one_step")
    if mode in {"space_time", "autoregressive"}:
        prefix = np.arange(config["physics_test"]["pino"]["prefix_steps"] + 1) * inputs["dt"]
        if mode == "autoregressive":
            pino = config["physics_test"]["pino"]
            if config["physics_test"].get("stage") == "online" and pino.get("online_sampling") == "per_update":
                prefix = np.arange(pino["validation_rollout_steps"] + 1) * inputs["dt"]
            return [round(float(t), 10) for t in prefix]
        return sorted(set(round(float(t), 10) for t in (*times, *prefix)))
    return times


def make_boundary(params, intervals, inputs, device, dtype):
    columns = [build_interface_forcing(p["temporal_family"], p["temporal_params"],
               p["spatial_family"], p["spatial_params"], inputs["y"],
               int(idx) * inputs["dt"], (int(idx) + 1) * inputs["dt"],
               inputs["ramp_seconds"]) for p, idx in zip(params, intervals)]
    qn, qnext, integral = (torch.as_tensor(np.stack(v), device=device, dtype=dtype)
                          for v in zip(*columns))
    return FullBCData(T_right_tilde=torch.tensor(
        (300 - inputs["mu_global"]) / inputs["sigma_global"], device=device, dtype=dtype),
        qL_n=qn, qL_np1=qnext, qL_int=integral)


def step_forcing_image(params, intervals, config, inputs, device, dtype):
    """Exact interval-average flux; fixed-dt CN needs no other time descriptor."""
    height, width = config["physics_test"]["cvit"]["forcing_grid_size"]
    boundary = make_boundary(params, intervals, dict(inputs, y=np.linspace(0, 1, height)), device, dtype)
    average = boundary.qL_int / (inputs["dt"] * config["physics_test"]["pino"]["forcing_a_ref"])
    return average[:, None, :, None].expand(-1, 1, -1, width).contiguous()


def autoregressive_step(model, state, params, intervals, config, inputs, coords):
    if isinstance(model, FNO2d):
        boundary = make_boundary(params, intervals, inputs, state.device, state.dtype)
        average = boundary.qL_int / (inputs["dt"] * config["physics_test"]["pino"]["forcing_a_ref"])
        fields = inputs["spec"].build_step_inputs(state, coords, average, inputs["dt"], inputs["t_final"])
        return model(**fields)[..., 0]
    forcing = step_forcing_image(params, intervals, config, inputs, state.device, state.dtype)
    latent = model.encode(forcing, state[:, None])
    return decode_grid(model, latent, coords, None, *state.shape[-2:],
                       config["physics_test"]["pino"]["query_chunk_size"])


def model_inputs(params, config, inputs, device):
    ic_array = np.stack([p["T0"] for p in params])
    ic = torch.as_tensor(ic_array, device=device, dtype=torch.float32)
    ic = (ic - inputs["mu_global"]) / inputs["sigma_global"]
    if config["physics_test"]["pino"].get("residual_mode") == "autoregressive":
        return None, ic
    if config["physics_test"].get("model_type", "cvit") == "fno":
        X, Y = np.meshgrid(inputs["x"], inputs["y"], indexing="ij")
        context = SimpleNamespace(sim_params=params, sim_ids=range(len(params)),
                    y_grid=inputs["y"], Nx=len(inputs["x"]), Ny=len(inputs["y"]),
                    X_norm=X, Y_norm=Y, time_norm_horizon=inputs["t_final"],
                    ramp_seconds=inputs["ramp_seconds"])
        inputs["spec"].setup_dataset(context)
        fields = [inputs["spec"].build_inputs(context, sid,
                  (ic_array[sid] - inputs["mu_global"]) / inputs["sigma_global"],
                  0.0, inputs["t_final"]) for sid in range(len(params))]
        spatial, sequence = (torch.as_tensor(np.stack([f[key] for f in fields]),
                         dtype=torch.float32, device=device) for key in ("spatial", "forcing_seq"))
        return (spatial, sequence, inputs["t_final"]), ic
    height, width = config["physics_test"]["cvit"]["forcing_grid_size"]
    forcing = build_forcing_image(params, np.linspace(0, 1, height),
               np.linspace(0, inputs["t_final"], width),
               config["physics_test"]["pino"]["forcing_a_ref"], device,
               inputs["ramp_seconds"])
    return forcing, ic


def matched_stream(config, inputs, stage):
    pino = config["physics_test"]["pino"]
    rng = np.random.default_rng(pino["seed"] + 100)
    interval_rng = np.random.default_rng(pino["seed"] + 200)
    train, val, test = split_sim_ids(len(inputs["params"]), seed=0)
    if not len(train) or not len(val):
        raise ValueError("Development data must have nonempty train and validation splits")
    updates = pino[f"{stage}_updates"]
    batch = pino["batch_size"]
    if stage == "online" and pino.get("online_sampling") == "per_update":
        count = pino["validation_cases"]
        if not isinstance(count, int) or count < 4 or count % len(IC_FAMILIES):
            raise ValueError("validation_cases must be a positive multiple of the IC family count")
        generators = online_generators(pino["validation_seed"])
        families = list(np.repeat(list(IC_FAMILIES), count // len(IC_FAMILIES)))
        generators["main"].shuffle(families)
        validation = draw_online_batch(inputs, generators, count, families)
        return dict(problems=[], ids=np.empty((0, batch), dtype=np.int64),
                    intervals=np.empty((0, batch), dtype=np.int64),
                    validation=validation, validation_ids=list(range(count)),
                    train_ids=train.tolist(), reserved_test_ids=test.tolist())
    if stage == "fixed":
        structured = [int(i) for i in train if inputs["params"][i]["ic_family"] == "random_sinusoid_2d"]
        sid = structured[0] if structured else int(train[0])
        problems = [inputs["params"][sid]]
        ids = np.zeros((updates, batch), dtype=np.int64)
        validation = [inputs["params"][sid]]
        validation_ids = [sid]
    else:
        grids = dict(X=np.broadcast_to(inputs["x"][:, None], (len(inputs["x"]), len(inputs["y"]))),
                     Y=np.broadcast_to(inputs["y"][None], (len(inputs["x"]), len(inputs["y"]))),
                     y_grid=inputs["y"])
        time_cfg = dict(dt=inputs["dt"], t_final=inputs["t_final"])
        streams = {k: np.random.default_rng(pino["seed"] + offset)
                   for k, offset in (("ic_family", 300), ("ic_params", 400), ("forcing_params", 500))}
        problems = inputs["spec"].sample_online_params(rng, updates * batch, grids, time_cfg,
                                                       rng_streams=streams)
        ids = np.arange(updates * batch).reshape(updates, batch)
        validation_ids = val.tolist()
        validation = [inputs["params"][sid] for sid in val]
    steps = int(round(inputs["t_final"] / inputs["dt"]))
    if pino.get("residual_mode", "one_step") in {"space_time", "autoregressive"}:
        prefix = pino["prefix_steps"]
        if not isinstance(prefix, int) or not 1 <= prefix <= steps:
            raise ValueError(f"prefix_steps must be an integer in [1,{steps}]")
        intervals = np.full_like(ids, prefix - 1)
        return dict(problems=problems, ids=ids, intervals=intervals,
                    validation=validation, validation_ids=validation_ids,
                    train_ids=train.tolist(), reserved_test_ids=test.tolist())
    intervals = np.empty_like(ids)
    for update in range(updates):
        warmup = pino["causal_warmup_updates"]
        limit = steps if not warmup else max(1, int(np.ceil(steps * min(1, (update + 1) / warmup))))
        intervals[update] = interval_rng.integers(0, limit, size=batch)
    return dict(problems=problems, ids=ids, intervals=intervals,
                validation=validation, validation_ids=validation_ids,
                train_ids=train.tolist(), reserved_test_ids=test.tolist())


def online_generators(seed):
    return {name: np.random.default_rng(seed + offset) for name, offset in
            (("main", 100), ("ic_family", 300), ("ic_params", 400), ("forcing_params", 500))}


def draw_online_batch(inputs, generators, count, families=None):
    nx, ny = len(inputs["x"]), len(inputs["y"])
    grids = dict(X=np.broadcast_to(inputs["x"][:, None], (nx, ny)),
                 Y=np.broadcast_to(inputs["y"][None, :], (nx, ny)), y_grid=inputs["y"])
    return inputs["spec"].sample_online_params(generators["main"], count, grids,
                dict(dt=inputs["dt"], t_final=inputs["t_final"]),
                rng_streams=generators, ic_family_assignment=families)


def autoregressive_backward(model, params, config, inputs, coords, geom, inverse, objective):
    pino = config["physics_test"]["pino"]
    _, state = model_inputs(params, config, inputs, coords.device)
    losses, residuals, corrections = [], [], []
    for n in range(pino["prefix_steps"]):
        previous = state.detach()
        intervals = np.full(len(params), n)
        state = autoregressive_step(model, previous, params, intervals, config, inputs, coords)
        bc = make_boundary(params, intervals, inputs, coords.device, state.dtype)
        step_loss, residual, correction = cn_objective(previous, state, geom, bc, inverse, objective)
        # Step graphs are independent because both uses of the current state detach.
        (step_loss / pino["prefix_steps"]).backward()
        losses.append(step_loss.detach())
        residuals.append(residual.detach())
        corrections.append(correction.detach())
    return torch.stack(losses).mean(), torch.stack(residuals), torch.stack(corrections)


def development_references(inputs, params, evaluation_times):
    """Only the development caller uses solver reference temperatures."""
    setup = build_base_setup(len(params), 1, nx=len(inputs["x"]), ny=len(inputs["y"]),
                            dt=inputs["dt"], t_final=inputs["t_final"],
                            ramp_seconds=inputs["ramp_seconds"])
    indices = [int(round(t / inputs["dt"])) for t in evaluation_times]
    if any(not np.isclose(t, i * inputs["dt"], atol=1e-8) for t, i in zip(evaluation_times, indices)):
        raise ValueError("Development evaluation times must lie on the solver time grid")
    refs = []
    for p in params:
        solver = inputs["spec"].configure_solver(p, setup["base_kwargs"])
        state = np.asarray(p["T0"], dtype=np.float64).copy()
        saved = {0: state.copy()}
        for idx in range(max(indices)):
            state = solver.cn_step(state, idx * inputs["dt"])
            if idx + 1 in indices:
                saved[idx + 1] = state.copy()
        refs.append(np.stack([saved[i] for i in indices]))
    return np.stack(refs)


@torch.no_grad()
def evaluate_development(model, config, inputs, params, references, coords, geom, inverse):
    pino = config["physics_test"]["pino"]
    times = evaluation_query_times(config, inputs)
    device = coords.device
    nx, ny = len(inputs["x"]), len(inputs["y"])
    model.eval()
    rows = []
    autoregressive = pino.get("residual_mode") == "autoregressive"
    for sid, p in enumerate(params):
        forcing, ic = model_inputs([p], config, inputs, device)
        if autoregressive:
            states = [ic]
            for n in range(pino["prefix_steps"]):
                states.append(autoregressive_step(model, states[-1], [p], [n], config, inputs, coords))
        else:
            latent = forcing if isinstance(model, FNO2d) else model.encode(forcing, ic[:, None])
        for tid, t in enumerate(times):
            u = states[tid] if autoregressive else decode_grid(model, latent, coords, [t], nx, ny, pino["query_chunk_size"])
            predicted = u[0].cpu().numpy() * inputs["sigma_global"] + inputs["mu_global"]
            target = references[sid, tid]
            error = predicted - target
            row = dict(case=sid, ic_family=p["ic_family"], time=t,
                       amplitude=p["temporal_params"]["A"], frequency=p["temporal_params"]["f"],
                       rmse_K=float(np.sqrt(np.mean(error ** 2))),
                       sigma_error_percent=float(100 * np.sqrt(np.mean(error ** 2)) / inputs["sigma_global"]),
                       normalized_rel_l2_percent=float(100 * np.linalg.norm(error) / max(np.linalg.norm(target - inputs["mu_global"]), 1e-12)),
                       physical_rel_l2_percent=float(100 * np.linalg.norm(error) / np.linalg.norm(target)),
                       copy_rmse_K=float(np.sqrt(np.mean((p["T0"] - target) ** 2))),
                       equilibrium_rmse_K=float(np.sqrt(np.mean((300 - target) ** 2))),
                       right_wall_rmse_K=float(np.sqrt(np.mean(error[-1] ** 2))))
            if t < times[-1] - inputs["dt"] / 2:
                following = states[tid + 1] if autoregressive else decode_grid(model, latent, coords, [t + inputs["dt"]], nx, ny, pino["query_chunk_size"])
                previous = anchored_previous(u, ic, [round(t / inputs["dt"])])
                bc = make_boundary([p], [round(t / inputs["dt"])], inputs, device, u.dtype)
                residual = cn_step_residual(previous, following, geom, bc)
                correction = inverse(residual)
                row.update(consistency_rmse_K=float(correction.square().mean().sqrt() * inputs["sigma_global"]),
                           raw_residual_rms=float(residual.square().mean().sqrt()),
                           left_residual_rms=float(residual[:, 0].square().mean().sqrt()),
                           adiabatic_residual_rms=float(residual[..., [0, -1]].square().mean().sqrt()))
            rows.append(row)
    def aggregate(key, selected):
        return float(np.sqrt(np.mean([r[key] ** 2 for r in selected])))
    future = [r for r in rows if r["time"] > 0 and
              (autoregressive or any(np.isclose(r["time"], t) for t in pino["evaluation_times"]))]
    first = [r for r in rows if np.isclose(r["time"], inputs["dt"])]
    if not first:
        raise ValueError("Evaluation must include the first solver interval")
    summary = dict(trajectory_rmse_K=aggregate("rmse_K", future),
                   trajectory_sigma_error_percent=aggregate("sigma_error_percent", future),
                   first_step_rmse_K=aggregate("rmse_K", first),
                   first_step_sigma_error_percent=aggregate("sigma_error_percent", first),
                   copy_trajectory_rmse_K=aggregate("copy_rmse_K", future),
                   equilibrium_trajectory_rmse_K=aggregate("equilibrium_rmse_K", future),
                   copy_first_step_rmse_K=aggregate("copy_rmse_K", first),
                   equilibrium_first_step_rmse_K=aggregate("equilibrium_rmse_K", first),
                   consistency_rmse_K=aggregate("consistency_rmse_K", [r for r in rows if "consistency_rmse_K" in r]),
                   ic_rmse_K=aggregate("rmse_K", [r for r in rows if r["time"] == 0]))
    if pino.get("residual_mode", "one_step") in {"space_time", "autoregressive"}:
        prefix = [r for r in rows if 0 < r["time"] <= pino["prefix_steps"] * inputs["dt"] + 1e-9]
        summary.update(prefix_trajectory_rmse_K=aggregate("rmse_K", prefix),
                       prefix_trajectory_sigma_error_percent=aggregate("sigma_error_percent", prefix),
                       prefix_copy_trajectory_rmse_K=aggregate("copy_rmse_K", prefix),
                       prefix_equilibrium_trajectory_rmse_K=aggregate("equilibrium_rmse_K", prefix))
    model.train()
    return summary, rows


def cvit_gate(raw, exact, pino, stage):
    ratio = pino["gate_error_ratio"]
    improved = None if raw is None else all(exact[key] * ratio < raw[key] for key in
                   ("trajectory_rmse_K", "first_step_rmse_K"))
    beats_floors = all(exact[f"{kind}_rmse_K"] < exact[f"{baseline}_{kind}_rmse_K"]
                      for kind in ("trajectory", "first_step") for baseline in ("copy", "equilibrium"))
    fits = stage != "fixed" or exact["trajectory_sigma_error_percent"] <= pino["fixed_max_sigma_error_percent"]
    first_fits = raw is not None or stage != "fixed" or exact["first_step_sigma_error_percent"] <= pino["fixed_max_first_step_sigma_error_percent"]
    return dict(passed=bool((improved if raw is not None else first_fits) and beats_floors and fits), improves_both=improved,
                beats_both_baselines=beats_floors, fixed_fit_tolerance_passed=fits,
                first_step_fit_tolerance_passed=first_fits,
                raw_comparison_performed=raw is not None,
                trajectory_raw_over_exact=None if raw is None else raw["trajectory_rmse_K"] / max(exact["trajectory_rmse_K"], 1e-15),
                first_step_raw_over_exact=None if raw is None else raw["first_step_rmse_K"] / max(exact["first_step_rmse_K"], 1e-15))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def validation_steps(update, config):
    pino = config["physics_test"]["pino"]
    full = update == 0 or update == pino["online_updates"] or update % pino["full_validate_every"] == 0
    return pino["validation_rollout_steps"] if full else pino["prefix_steps"]


@torch.no_grad()
def evaluate_online_mg(model, config, inputs, params, references, coords, steps, stop, progress=None):
    pino = config["physics_test"]["pino"]
    progress = progress or dict(case_start=0, next_step=0, state=None, rows=[],
                                elapsed_seconds=0.0, prefix_seconds=0.0)
    rows = list(progress["rows"])
    prefix_seconds = progress["prefix_seconds"]
    start = time.perf_counter()
    model.eval()
    try:
        for lo in range(progress["case_start"], len(params), pino["validation_batch_size"]):
            batch = params[lo:lo + pino["validation_batch_size"]]
            first_step = progress["next_step"] if lo == progress["case_start"] else 0
            state = (progress["state"].to(coords.device) if first_step else
                     model_inputs(batch, config, inputs, coords.device)[1])
            for n in range(first_step, steps + 1):
                synchronize(coords.device)
                step_start = time.perf_counter()
                if n:
                    state = autoregressive_step(model, state, batch, np.full(len(batch), n - 1),
                                                config, inputs, coords)
                    physical = state.cpu().double().numpy() * inputs["sigma_global"] + inputs["mu_global"]
                else:
                    physical = np.stack([p["T0"] for p in batch]).astype(np.float64)
                for offset, p in enumerate(batch):
                    target = references[lo + offset, n]
                    error = physical[offset] - target
                    rmse = float(np.sqrt(np.mean(error ** 2)))
                    rows.append(dict(case=lo + offset, ic_family=p["ic_family"],
                        time=round(n * inputs["dt"], 10), amplitude=p["temporal_params"]["A"],
                        frequency=p["temporal_params"]["f"], rmse_K=rmse,
                        sigma_error_percent=100 * rmse / inputs["sigma_global"],
                        normalized_rel_l2_percent=float(100 * np.linalg.norm(error) / max(np.linalg.norm(target - inputs["mu_global"]), 1e-12)),
                        physical_rel_l2_percent=float(100 * np.linalg.norm(error) / np.linalg.norm(target)),
                        copy_rmse_K=float(np.sqrt(np.mean((p["T0"] - target) ** 2))),
                        equilibrium_rmse_K=float(np.sqrt(np.mean((300 - target) ** 2))),
                        right_wall_rmse_K=float(np.sqrt(np.mean(error[-1] ** 2)))))
                synchronize(coords.device)
                if n <= pino["prefix_steps"]:
                    prefix_seconds += time.perf_counter() - step_start
                if stop["requested"]:
                    done_batch = n == steps
                    return None, rows, dict(
                        case_start=lo + len(batch) if done_batch else lo,
                        next_step=0 if done_batch else n + 1,
                        state=None if done_batch else state.cpu(), rows=rows,
                        elapsed_seconds=progress["elapsed_seconds"] + time.perf_counter() - start,
                        prefix_seconds=prefix_seconds)
        def aggregate(selected):
            first = [r for r in selected if np.isclose(r["time"], inputs["dt"])]
            prefix = [r for r in selected if 0 < r["time"] <= pino["prefix_steps"] * inputs["dt"] + 1e-9]
            rms = lambda records: float(np.sqrt(np.mean([r["rmse_K"] ** 2 for r in records])))
            result = dict(first_step_rmse_K=rms(first), prefix_trajectory_rmse_K=rms(prefix))
            if steps == pino["validation_rollout_steps"]:
                result["full_trajectory_rmse_K"] = rms([r for r in selected if r["time"] > 0])
            for key, value in list(result.items()):
                result[key.replace("rmse_K", "sigma_error_percent")] = 100 * value / inputs["sigma_global"]
            return result
        summary = aggregate(rows)
        summary.update(validation_steps=steps, validation_cases=len(params),
                       validation_wall_seconds=progress["elapsed_seconds"] + time.perf_counter() - start,
                       prefix_wall_seconds=prefix_seconds,
                       ic_families={family: aggregate([r for r in rows if r["ic_family"] == family])
                                    for family in IC_FAMILIES})
        return summary, rows, None
    finally:
        model.train()


def runtime_report(update, durations, validation_times, pino, peak_memory, remaining_allocation):
    lower = 10 if update == 50 else 50
    seconds = float(np.mean([duration for index, duration in durations if lower < index <= update]))
    remaining = pino["online_updates"] - update
    scheduled = [u for u in range(update + 1, pino["online_updates"] + 1)
                 if u % pino["online_validate_every"] == 0 or u == pino["online_updates"]]
    full_passes = sum(u % pino["full_validate_every"] == 0 or u == pino["online_updates"] for u in scheduled)
    validation_remaining = (full_passes * validation_times["full"] +
                            (len(scheduled) - full_passes) * validation_times["prefix"])
    estimate = remaining * seconds + validation_remaining
    return dict(update=update, updates_per_second=1 / seconds, mean_seconds_per_update=seconds,
                peak_GPU_memory_GiB=peak_memory / 2 ** 30 if peak_memory is not None else None,
                estimated_remaining_training_seconds=remaining * seconds,
                validation_wall_seconds=validation_times.copy(),
                estimated_remaining_total_seconds=estimate,
                remaining_allocation_seconds=max(remaining_allocation, 0),
                estimated_to_fit_allocation=estimate <= remaining_allocation)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_online_mg(initial_state, config, inputs, stream, references, output, device, resume_checkpoint=None):
    pino = config["physics_test"]["pino"]
    model = build_screen_model(config, inputs, device)
    model.load_state_dict(initial_state)
    set_seed(pino["seed"])
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    X, Y = np.meshgrid(inputs["x"], inputs["y"], indexing="ij")
    coords = torch.as_tensor(np.stack([X, Y], -1).reshape(1, -1, 2), device=device, dtype=torch.float32)
    precise = build_homogeneous_cn_geom(inputs["x"], inputs["y"], 1.0, inputs["dt"], sigma_global=inputs["sigma_global"])
    inverse = MGOneCycleInverse(precise, **config["physics_test"]["mg"]["solver"]).to(device=device, dtype=torch.float32)
    geom = replace(precise, **{name: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                    for name, value in vars(precise).items() if torch.is_tensor(value)})
    generators = online_generators(pino["seed"])
    state = dict(successful_updates=0, elapsed_seconds=0.0, window=[], profile_durations=[],
                 runtime_reports=[], validation_times=dict(prefix=0.0, full=0.0),
                 pending_validation=True, validation_progress=None, evaluations=[],
                 manifest_position=0, metrics_position=0)
    fields = ("update", "physics_loss", "raw_residual_rms", "gradient_norm", "learning_rate", "seconds_per_update")
    output.mkdir(exist_ok=resume_checkpoint is not None)
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires the exact frozen experiment configuration")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state = checkpoint["online_state"]
        if scheduler.next_update != state["successful_updates"]:
            raise ValueError("Checkpoint scheduler and successful update counts disagree")
        for name, rng in generators.items():
            rng.bit_generator.state = checkpoint["sampler_rng_states"][name]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    else:
        with (output / "train_metrics.csv").open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()
        (output / "training_manifest.jsonl").touch()
        state["metrics_position"] = (output / "train_metrics.csv").stat().st_size
    for name, position in (("training_manifest.jsonl", state["manifest_position"]), ("train_metrics.csv", state["metrics_position"])):
        with (output / name).open("r+b") as file:
            file.truncate(position)
    start = time.perf_counter()
    prior_elapsed = state["elapsed_seconds"]
    job_start = float(os.environ.get("PCVIT_JOB_START_EPOCH", time.time() - config["physics_test"]["protocol"].get("setup_seconds", 0)))
    stop = dict(requested=False)
    def request_stop(signum, frame):
        stop["requested"] = True
    previous_handler = signal.signal(signal.SIGUSR1, request_stop)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def checkpoint(path):
        state["elapsed_seconds"] = prior_elapsed + time.perf_counter() - start
        state["manifest_position"] = (output / "training_manifest.jsonl").stat().st_size
        state["metrics_position"] = (output / "train_metrics.csv").stat().st_size
        temporary = path.with_suffix(".pt.tmp")
        torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                   scheduler_state_dict=scheduler.state_dict(), successful_updates=state["successful_updates"],
                   online_state=state, config=config, mu_global=inputs["mu_global"], sigma_global=inputs["sigma_global"],
                   torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state(),
                   cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                   sampler_rng_states={name: rng.bit_generator.state for name, rng in generators.items()}), temporary)
        temporary.replace(path)
    try:
        if resume_checkpoint is None:
            checkpoint(output / "checkpoint_000000.pt")
        with (output / "train_metrics.csv").open("a", newline="") as metrics, (output / "training_manifest.jsonl").open("a") as manifest:
            writer = csv.DictWriter(metrics, fieldnames=fields)
            while True:
                update = state["successful_updates"]
                if stop["requested"]:
                    checkpoint(output / "checkpoint_latest.pt")
                    break
                if state["pending_validation"]:
                    checkpoint(output / "checkpoint_latest.pt")
                    if update == pino["online_updates"]:
                        checkpoint(output / "checkpoint_final.pt")
                    summary, rows, progress = evaluate_online_mg(model, config, inputs, stream["validation"],
                               references, coords, validation_steps(update, config), stop, state["validation_progress"])
                    state["validation_progress"] = progress
                    if summary is None:
                        continue
                    summary.update(update=update)
                    state["evaluations"].append(summary)
                    state["pending_validation"] = False
                    state["validation_times"]["prefix"] = summary["prefix_wall_seconds"]
                    if "full_trajectory_rmse_K" in summary:
                        state["validation_times"]["full"] = summary["validation_wall_seconds"]
                    atomic_json(output / f"val_pairs_{update:06d}.json", rows)
                    atomic_json(output / "diagnostics.json", state["evaluations"])
                    atomic_json(output / f"validation_{update:06d}.json", summary)
                    write_development_pairs(output.parent, config, {"mg": state["evaluations"]})
                    text = f"validation update={update} first_step_rmse_K={summary['first_step_rmse_K']:.6g} first_step_sigma_percent={summary['first_step_sigma_error_percent']:.6g} prefix_rmse_K={summary['prefix_trajectory_rmse_K']:.6g} prefix_sigma_percent={summary['prefix_trajectory_sigma_error_percent']:.6g}"
                    if "full_trajectory_rmse_K" in summary:
                        text += f" full_rmse_K={summary['full_trajectory_rmse_K']:.6g} full_sigma_percent={summary['full_trajectory_sigma_error_percent']:.6g}"
                    print(text, flush=True)
                    checkpoint(output / "checkpoint_latest.pt")
                    if device.type == "cuda" and update == 0:
                        torch.cuda.reset_peak_memory_stats(device)
                if update == pino["online_updates"]:
                    checkpoint(output / "checkpoint_final.pt")
                    atomic_json(output / "final_metrics.json", state["evaluations"][-1])
                    break
                synchronize(device)
                update_start = time.perf_counter()
                params = draw_online_batch(inputs, generators, pino["batch_size"])
                manifest.write(json.dumps(dict(update=update + 1,
                               problems=[{k: v for k, v in p.items() if k != "T0"} for p in params]),
                               default=lambda value: value.tolist()) + "\n")
                manifest.flush()
                optimizer.zero_grad(set_to_none=True)
                physics, residual, _ = autoregressive_backward(model, params, config, inputs, coords, geom, inverse, "mg")
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                if not torch.isfinite(physics) or not torch.isfinite(grad):
                    raise FloatingPointError(f"Nonfinite MG loss/gradient at update {update + 1}")
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                synchronize(device)
                duration = time.perf_counter() - update_start
                state["successful_updates"] = update + 1
                state["window"].append(dict(physics_loss=float(physics), residual_square=float(residual.square().mean()),
                                  gradient_norm=float(grad), learning_rate=lr, seconds_per_update=duration))
                if update + 1 <= 100:
                    state["profile_durations"].append((update + 1, duration))
                if (update + 1) % pino["log_every"] == 0:
                    window = state["window"]
                    record = dict(update=update + 1, physics_loss=float(np.mean([r["physics_loss"] for r in window])),
                                  raw_residual_rms=float(np.sqrt(np.mean([r["residual_square"] for r in window]))),
                                  gradient_norm=float(np.mean([r["gradient_norm"] for r in window])),
                                  learning_rate=lr, seconds_per_update=float(np.mean([r["seconds_per_update"] for r in window])))
                    writer.writerow(record)
                    metrics.flush()
                    print("train " + " ".join(f"{k}={v:.6g}" for k, v in record.items()), flush=True)
                    state["window"] = []
                if update + 1 in pino["startup_profile_updates"]:
                    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    report = runtime_report(update + 1, state["profile_durations"], state["validation_times"], pino,
                                            peak, pino["allocation_seconds"] - (time.time() - job_start))
                    state["runtime_reports"].append(report)
                    atomic_json(output / "runtime_diagnostics.json", state["runtime_reports"])
                    print("runtime " + json.dumps(report), flush=True)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                state["pending_validation"] = ((update + 1) % pino["online_validate_every"] == 0 or update + 1 == pino["online_updates"])
        return state, state["evaluations"]
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)


def run_control(objective, initial_state, config, inputs, stream, references, output, device, resume_checkpoint=None):
    pino = config["physics_test"]["pino"]
    stage = config["physics_test"]["stage"]
    if stage == "online" and pino.get("online_sampling") == "per_update":
        return run_online_mg(initial_state, config, inputs, stream, references, output, device, resume_checkpoint)
    model = build_screen_model(config, inputs, device)
    model.load_state_dict(initial_state)
    set_seed(pino["seed"])
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    nx, ny = len(inputs["x"]), len(inputs["y"])
    X, Y = np.meshgrid(inputs["x"], inputs["y"], indexing="ij")
    coords = torch.as_tensor(np.stack([X, Y], axis=-1).reshape(1, -1, 2), device=device, dtype=torch.float32)
    precise = build_homogeneous_cn_geom(inputs["x"], inputs["y"], 1.0, inputs["dt"],
                                       sigma_global=inputs["sigma_global"])
    inverse = ExactCNInverse(precise)
    training_inverse = (MGOneCycleInverse(precise, **config["physics_test"]["mg"]["solver"])
                        .to(device=device, dtype=torch.float32) if objective == "mg" else inverse)
    geom = replace(precise, **{name: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                    for name, value in vars(precise).items() if torch.is_tensor(value)})
    first_update = 0
    prior_elapsed = 0.0
    prior_training = 0.0
    evaluations = []
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume requires the exact frozen experiment configuration")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        first_update = checkpoint["successful_updates"]
        if scheduler.next_update != first_update:
            raise ValueError("Checkpoint optimizer-update and scheduler counters disagree")
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        evaluations = [r for r in json.loads((output / "diagnostics.json").read_text()) if r["update"] <= first_update]
        prior_elapsed = evaluations[-1]["elapsed_seconds"]
        prior_training = evaluations[-1]["training_seconds"]
    output.mkdir(exist_ok=resume_checkpoint is not None)
    fields = ("update", "elapsed_seconds", "learning_rate", "physics_loss", "ic_loss",
              "raw_residual_rms", "transformed_residual_rms", "gradient_norm", "update_seconds")
    validate_every = pino[f"{stage}_validate_every"]
    start = time.perf_counter()
    eval_total = prior_elapsed - prior_training
    metrics_path = output / "train_metrics.csv"
    if resume_checkpoint is not None:
        shutil.copyfile(metrics_path, output / f"train_metrics_before_resume_{time.time_ns()}.csv")
        with metrics_path.open() as file:
            previous_rows = [r for r in csv.DictReader(file) if int(r["update"]) <= first_update]
        with metrics_path.open("w") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(previous_rows)
    with metrics_path.open("a" if resume_checkpoint is not None else "w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if resume_checkpoint is None:
            writer.writeheader()
        for update in range(first_update, len(stream["ids"]) + 1):
            if (resume_checkpoint is None or update > first_update) and (update % validate_every == 0 or update == len(stream["ids"])):
                synchronize(device)
                eval_start = time.perf_counter()
                summary, rows = evaluate_development(model, config, inputs, stream["validation"], references,
                                                       coords, geom, inverse)
                synchronize(device)
                eval_total += time.perf_counter() - eval_start
                elapsed = prior_elapsed + time.perf_counter() - start
                summary.update(update=update, elapsed_seconds=elapsed, training_seconds=elapsed - eval_total)
                evaluations.append(summary)
                (output / "diagnostics.json").write_text(json.dumps(evaluations, indent=2) + "\n")
                (output / f"val_pairs_{update:06d}.json").write_text(json.dumps(rows, indent=2) + "\n")
                checkpoint_path = output / f"checkpoint_{update:06d}.pt"
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(), successful_updates=update,
                                config=config, mu_global=inputs["mu_global"], sigma_global=inputs["sigma_global"],
                                torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state()),
                           temporary)
                temporary.replace(checkpoint_path)
                if not pino.get("checkpoint_keep_all", True):
                    for older in output.glob("checkpoint_*.pt"):
                        if older.name != "checkpoint_000000.pt" and older != checkpoint_path:
                            older.unlink()
                print(f"{objective} update={update} sigma_error={summary['trajectory_sigma_error_percent']:.4g}% "
                      f"first_step={summary['first_step_rmse_K']:.4g}K elapsed={elapsed:.1f}s", flush=True)
            if update == len(stream["ids"]):
                break
            synchronize(device)
            update_start = time.perf_counter()
            params = [stream["problems"][int(sid)] for sid in stream["ids"][update]]
            intervals = stream["intervals"][update]
            forcing, ic = model_inputs(params, config, inputs, device)
            autoregressive = pino.get("residual_mode") == "autoregressive"
            if not autoregressive:
                latent = forcing if isinstance(model, FNO2d) else model.encode(forcing, ic[:, None])
            if autoregressive:
                optimizer.zero_grad(set_to_none=True)
                physics, residual, transformed = autoregressive_backward(model, params, config, inputs,
                                            coords, geom, training_inverse, objective)
                ic_loss = torch.zeros_like(physics)
                loss = physics
            elif pino.get("residual_mode", "one_step") == "space_time":
                predictions = [decode_grid(model, latent, coords, [n * inputs["dt"]], nx, ny,
                               pino["query_chunk_size"]) for n in range(1, pino["prefix_steps"] + 1)]
                states = torch.stack([ic, *predictions], dim=1)
                boundaries = [make_boundary(params, np.full(len(params), n), inputs, device, ic.dtype)
                              for n in range(pino["prefix_steps"])]
                physics, residual, transformed = prefix_cn_objective(states, geom, boundaries, inverse, objective)
            else:
                with torch.no_grad():
                    previous = decode_grid(model, latent, coords, intervals * inputs["dt"], nx, ny,
                                           pino["query_chunk_size"])
                    previous = anchored_previous(previous, ic, intervals)
                following = decode_grid(model, latent, coords, (intervals + 1) * inputs["dt"], nx, ny,
                                        pino["query_chunk_size"])
                bc = make_boundary(params, intervals, inputs, device, following.dtype)
                physics, residual, transformed = cn_objective(previous, following, geom, bc, inverse, objective)
            if not autoregressive:
                predicted_ic = decode_grid(model, latent, coords, [0.0], nx, ny, pino["query_chunk_size"])
                ic_loss = (predicted_ic - ic).square().mean()
                loss = physics + pino["ic_weight"] * ic_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise FloatingPointError(f"Nonfinite {objective} loss/gradient at update {update}")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            synchronize(device)
            duration = time.perf_counter() - update_start
            writer.writerow(dict(update=update + 1, elapsed_seconds=prior_elapsed + time.perf_counter() - start,
                                 learning_rate=lr, physics_loss=float(physics.detach()), ic_loss=float(ic_loss.detach()),
                                 raw_residual_rms=float(residual.detach().square().mean().sqrt()),
                                 transformed_residual_rms=float(transformed.detach().square().mean().sqrt()),
                                 gradient_norm=float(grad), update_seconds=duration))
            file.flush()
            if (update + 1) % pino["log_every"] == 0:
                print(f"{objective} update={update+1} physics={float(physics.detach()):.4g} "
                      f"ic={float(ic_loss.detach()):.4g} seconds/update={duration:.2f}", flush=True)
    (output / "final_metrics.json").write_text(json.dumps(evaluations[-1], indent=2) + "\n")
    return evaluations[-1], evaluations


def plot_development(output, histories, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for objective, rows in histories.items():
        axes[0].semilogy([r["update"] for r in rows], [r["trajectory_sigma_error_percent"] for r in rows], label=objective)
        axes[1].semilogy([r["update"] for r in rows], [r["first_step_rmse_K"] for r in rows], label=objective)
    axes[0].set(xlabel="Optimizer updates", ylabel="Trajectory training-sigma error (%)")
    axes[1].set(xlabel="Optimizer updates", ylabel="First-step error from true IC (K)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    autoregressive = config["physics_test"]["pino"].get("residual_mode") == "autoregressive"
    if autoregressive:
        axes[0].set_ylabel("Prefix training-sigma error (%)")
    fig.savefig(output / ("prefix_accuracy.png" if autoregressive else "absolute_accuracy.png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4))
    for objective, rows in histories.items():
        ax.semilogy([r["update"] for r in rows], [r["consistency_rmse_K"] for r in rows], label=objective)
    ax.set(xlabel="Optimizer updates", ylabel="One-step consistency RMS (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "one_step_consistency.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for objective, rows in histories.items():
        pairs = json.loads((output / objective / f"val_pairs_{rows[-1]['update']:06d}.json").read_text())
        times = sorted({r["time"] for r in pairs})
        for ax, key in zip(axes, ("rmse_K", "consistency_rmse_K")):
            selected = [t for t in times if any(key in r and r["time"] == t for r in pairs)]
            values = [np.sqrt(np.mean([r[key] ** 2 for r in pairs if r["time"] == t and key in r])) for t in selected]
            ax.plot(selected, values, label=objective)
    axes[0].set(xlabel="Absolute query time (s)", ylabel="Trajectory error from true IC (K)")
    axes[1].set(xlabel="Previous query time (s)", ylabel="One-step consistency RMS (K)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / ("prefix_errors.png" if autoregressive else "full_horizon_errors.png"), dpi=160)
    plt.close(fig)


def run_screen(config, data_dir, output, phase0_dir, stage="fixed", fixed_screen_dir=None,
               normalization_path=None, smoke=False, exact_screen_dir=None):
    setup_start = time.perf_counter()
    if stage not in {"fixed", "online"}:
        raise ValueError("Only gated development fixed/online screens are implemented")
    model_type = config["physics_test"].get("model_type", "cvit")
    if model_type not in {"cvit", "fno"}:
        raise ValueError(f"Unknown physics-screen model {model_type!r}")
    pino = config["physics_test"]["pino"]
    mode = pino.get("residual_mode", "one_step")
    controls = pino.get("controls", ["raw", "exact"])
    online_mg = stage == "online" and mode == "autoregressive" and controls == ["mg"]
    allowed = {"raw", "exact", "spatial_exact"} if mode == "space_time" else {"raw", "exact"}
    single_autoregressive = mode == "autoregressive" and controls in (["exact"], ["mg"])
    if mode not in {"one_step", "space_time", "autoregressive"} or (not single_autoregressive and not {"raw", "exact"} <= set(controls) <= allowed) or len(set(controls)) != len(controls):
        raise ValueError("Use unique raw/exact controls, or an exact-only/MG-only autoregressive screen")
    if mode == "autoregressive" and stage != "fixed" and not online_mg:
        raise ValueError("Autoregressive screens require fixed cases or online MG")
    if pino.get("online_sampling", "precomputed") == "per_update" and not online_mg:
        raise ValueError("Per-update sampling requires online autoregressive MG")
    if pino["forcing_a_ref"] <= 0:
        raise ValueError("forcing_a_ref must be positive")
    if online_mg:
        if pino.get("online_sampling") != "per_update" or not pino.get("compact_metrics"):
            raise ValueError("Online MG requires per_update sampling and compact_metrics")
        if normalization_path is None:
            raise ValueError("Online MG requires the already-frozen normalization file")
        for key in ("online_updates", "online_validate_every", "full_validate_every", "batch_size", "validation_batch_size", "log_every"):
            if not isinstance(pino[key], int) or pino[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if pino["full_validate_every"] % pino["online_validate_every"]:
            raise ValueError("full_validate_every must be a multiple of online_validate_every")
        if pino["validation_seed"] == pino["seed"]:
            raise ValueError("Validation must use an independent seed")
        if not set(pino["startup_profile_updates"]) <= {50, 100}:
            raise ValueError("Startup profiling supports updates 50 and 100")
    mg_only = controls == ["mg"]
    prerequisites = [require_gate(phase0_dir, "phase4_direct_state_mg" if online_mg or (mg_only and not smoke) else "phase0_direct_state")]
    historical = None
    if mg_only and not smoke and not online_mg and (model_type == "cvit" or exact_screen_dir is not None):
        if exact_screen_dir is None:
            raise ValueError("The MG preservation screen requires the completed exact prefix screen")
        prerequisites.append(require_gate(exact_screen_dir, f"phase2_fixed_{model_type}_autoregressive_prefix"))
        historical = json.loads((Path(exact_screen_dir) / "final_metrics.json").read_text())["controls"]["exact"]
    if stage == "online":
        prerequisites.append(require_gate(fixed_screen_dir, f"phase4_fixed_{model_type}_autoregressive_prefix_mg" if online_mg else f"phase2_fixed_{model_type}"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    config["benchmark"].update(name="diffusion_forcing_single", representation="temporal_encoder")
    config["physics_test"]["stage"] = stage
    if normalization_path is None:
        normalization_path = output / "normalization.json"
        freeze_development_normalization(data_dir, normalization_path)
    inputs = load_pino_inputs(data_dir, normalization_path)
    if online_mg and not smoke:
        baseline_path = Path(fixed_screen_dir) / "config_used.yaml"
        baseline = yaml.safe_load(baseline_path.read_text())
        if model_type == "cvit":
            # The time Fourier scale is unused by the spatial-only decoder.
            current_options = {k: v for k, v in config["physics_test"]["cvit"].items() if k != "fourier_freq_t"}
            baseline_options = {k: v for k, v in baseline["physics_test"]["cvit"].items() if k != "fourier_freq_t"}
        else:
            current_options = config["model"]["parameters"]
            baseline_options = baseline["model"]["parameters"]
        if current_options != baseline_options or config["physics_test"]["mg"]["solver"] != baseline["physics_test"]["mg"]["solver"]:
            raise ValueError("Online MG must reuse the passing fixed-case architecture and MG settings")
        baseline_protocol = baseline["physics_test"]["protocol"]
        if any(inputs[key] != baseline_protocol["normalization"][key] for key in ("mu_global", "sigma_global")):
            raise ValueError("Online MG must reuse the passing fixed-case frozen normalization")
        if [len(inputs["x"]), len(inputs["y"])] != baseline_protocol["grid"] or any(
                inputs[key] != baseline_protocol[name] for key, name in (("dt", "dt"), ("t_final", "horizon"), ("ramp_seconds", "ramp_seconds"))):
            raise ValueError("Online MG must reuse the passing fixed-case grid and time discretization")
        prerequisites.append(dict(path=str(baseline_path.resolve()), sha256=file_sha256(baseline_path), kind="file"))
    if online_mg and not (isinstance(pino["prefix_steps"], int) and isinstance(pino["validation_rollout_steps"], int)
              and 1 <= pino["prefix_steps"] <= pino["validation_rollout_steps"] <= round(inputs["t_final"] / inputs["dt"])):
        raise ValueError("Require 1 <= training prefix <= validation rollout <= solver horizon")
    if not smoke and (len(inputs["x"]) != 100 or config["physics_test"]["pino"][f"{stage}_updates"] < (1000 if stage == "fixed" else 4000)):
        raise ValueError("A development gate needs the registered 100x100 grid and full update budget; use --smoke for wiring checks")
    config["physics_test"]["smoke"] = smoke
    if model_type == "cvit":
        config["model"] = {"type": "ForcingICCViT", "parameters": copy.deepcopy(config["physics_test"]["cvit"])}
    else:
        config["model"]["type"] = "FNO2d"
    config["paths"]["data_dir"] = str(Path(data_dir).resolve())
    config["data"]["num_sims"] = len(inputs["params"])
    for key, name in (("t_grid_path", "t_grid.npy"), ("x_grid_path", "x_grid.npy"),
                      ("y_grid_path", "y_grid.npy"), ("sim_params_path", "sim_params.npy"),
                      ("trajectories.npy", "trajectories.npy")):
        config["data"][key] = str((Path(data_dir) / name).resolve())
    device = resolve_device(config["training"]["device"])
    config["training"]["device"] = str(device)
    (output / "normalization.json").write_text(json.dumps(inputs["normalization"], indent=2) + "\n")
    stream = matched_stream(config, inputs, stage)
    if not online_mg:
        np.save(output / "problem_stream.npy", np.array(stream["problems"], dtype=object), allow_pickle=True)
        np.savez(output / "interval_stream.npz", ids=stream["ids"], intervals=stream["intervals"])
    np.save(output / "validation_inputs.npy", np.array(stream["validation"], dtype=object), allow_pickle=True)
    if online_mg:
        references = development_references(inputs, stream["validation"], evaluation_query_times(config, inputs))
        np.save(output / "validation_references.npy", references)
        references = np.load(output / "validation_references.npy", mmap_mode="r")
    repo = Path(__file__).resolve().parents[2]
    source_paths = ["src/operators/cvit.py", "src/operators/fno2d.py", "src/operators/train_pino.py", "src/operators/train.py",
                    "problems/diffusion_forcing_single.py", "src/physics/fv_residual.py",
                    "src/physics/fv_preconditioner.py", "src/physics/fv_solver_2d.py",
                    "src/physics/boundary_forcing.py", "src/physics/init_conditions.py",
                    "data/generate_dataset.py", "data/dataset.py", "scripts/run_train_pino.py"]
    if online_mg:
        source_paths.append("slurm/train_pcvit_mg_msi.sbatch")
    config["physics_test"]["protocol"] = dict(
        role="development", selection="final_registered_update", references_accessed=True,
        normalization=inputs["normalization"], data_dir=str(Path(data_dir).resolve()),
        dataset_metadata=inputs["meta"], sampler_version=inputs["spec"].online_sampler_version,
        ic_builder_version=inputs["spec"].ic_builder_version,
        prescribed_input_sha256={name: file_sha256(Path(data_dir) / name) for name in
                                ("sim_params.npy", "x_grid.npy", "y_grid.npy", "dt.npy", "ramp_seconds.npy", "meta.npy")},
        grid=[len(inputs["x"]), len(inputs["y"])], dt=inputs["dt"], horizon=inputs["t_final"],
        ramp_seconds=inputs["ramp_seconds"], validation_ids=stream["validation_ids"],
        reserved_test_ids=stream["reserved_test_ids"], prerequisites=prerequisites,
        source_sha256={name: file_sha256(repo / name) for name in source_paths},
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        model_type=model_type,
        residual_mode=mode, controls=controls,
        prefix_steps=pino.get("prefix_steps") if mode in {"space_time", "autoregressive"} else None,
        prefix_horizon=pino["prefix_steps"] * inputs["dt"] if mode in {"space_time", "autoregressive"} else None,
        total_solver_steps=int(round(inputs["t_final"] / inputs["dt"])),
        temporal_gradient_policy="detached_current_state_input_and_residual; mean_step_loss_gradients" if mode == "autoregressive" else "full_prefix_graph; spatial_exact alone detaches previous states" if mode == "space_time" else "detached_previous_state",
        input_encoding=("current_state_and_exact_interval_average_flux_FNO_tokens" if model_type == "fno" else "current_state_and_exact_interval_average_flux_xy_decoder") if mode == "autoregressive" else "full_horizon_forcing_tokens_and_IC_query_time_cond" if model_type == "fno" else "full_horizon_forcing_image_and_IC_coordinate_time_decoder",
        preconditioner="one_Torch_geometric_MG_V_cycle" if mg_only else "cached_exact_spatial_CN_inverse" if mode == "autoregressive" else "exact_CN_inverse",
        mg_settings=config["physics_test"].get("mg") if mg_only else None,
        historical_exact_metrics=historical,
        gate_policy="registered_final_update_evidence_no_accuracy_gate" if online_mg else "absolute_fit_and_within_registered_factor_of_historical_exact" if mg_only and historical is not None else "absolute_prefix_and_first_step_fit_only_no_raw_comparison" if single_autoregressive else "matched_raw_exact_improvement",
        online_sampling=pino.get("online_sampling", "precomputed"),
        validation_namespace="independent_seed_cases" if online_mg else "dataset_sim_ids",
        reference_role="held_out_validation_only" if online_mg else "development_diagnostics_only",
        setup_seconds=time.perf_counter() - setup_start if online_mg else None,
        learning_rate_schedule_note="Retained smooth exponential; differs from user-described cosine-to-LR/10 paper schedule" if online_mg else None,
        scheduled_final_learning_rate=max(config["training"]["scheduler"]["min_lr"], config["training"]["learning_rate"] *
                  config["training"]["scheduler"]["decay_rate"] ** (pino["online_updates"] / config["training"]["scheduler"]["decay_every"])) if online_mg else None,
        resolved_cvit_kwargs=dict(config["physics_test"]["cvit"], forcing_in_ch=1, ic_in_ch=1, out_dim=1,
                                  ic_grid_size=[len(inputs["x"]), len(inputs["y"])], t_final=inputs["t_final"],
                                  use_time_query=mode != "autoregressive",
                                  t_right_tilde=(300 - inputs["mu_global"]) / inputs["sigma_global"]) if model_type == "cvit" else None,
        resolved_fno_dims=vars(inputs["spec"].dims) if model_type == "fno" else None,
        model_origin="repository_FNO_defaults" if model_type == "fno" else "partial_archive_source_defaults_not_historical_run", objective="mean_free_node_square",
        metric_aggregation="equal_case_equal_nonzero_query_time_RMS", dtype="float32",
        stream_sha256={name: file_sha256(output / name) for name in
                       (("validation_inputs.npy", "validation_references.npy") if online_mg else
                        ("problem_stream.npy", "interval_stream.npz", "validation_inputs.npy"))})
    if historical is not None:
        baseline_config = yaml.safe_load((Path(exact_screen_dir) / "config_used.yaml").read_text())
        baseline_protocol = baseline_config["physics_test"]["protocol"]
        protocol = config["physics_test"]["protocol"]
        for key in ("normalization", "prescribed_input_sha256", "grid", "dt", "horizon", "ramp_seconds",
                    "validation_ids", "resolved_cvit_kwargs", "resolved_fno_dims", "prefix_steps", "stream_sha256"):
            if protocol[key] != baseline_protocol[key]:
                raise ValueError(f"MG/exact comparison requires identical {key}")
        for name in source_paths:
            if name not in {"src/operators/train_pino.py", "src/physics/fv_preconditioner.py", "scripts/run_train_pino.py"} and protocol["source_sha256"][name] != baseline_protocol["source_sha256"][name]:
                raise ValueError(f"MG/exact comparison requires unchanged model/physics source: {name}")
        for key in ("optimizer", "learning_rate", "weight_decay", "grad_clip", "scheduler", "device"):
            if config["training"][key] != baseline_config["training"][key]:
                raise ValueError(f"MG/exact comparison requires identical training.{key}")
        for key in ("seed", "fixed_updates", "fixed_validate_every", "batch_size", "query_chunk_size", "ic_weight"):
            if pino[key] != baseline_config["physics_test"]["pino"][key]:
                raise ValueError(f"MG/exact comparison requires identical pino.{key}")
        for name in ("config_used.yaml", "initial_state.pt"):
            path = Path(exact_screen_dir) / name
            prerequisites.append(dict(path=str(path.resolve()), sha256=file_sha256(path), kind="file"))
    for name in source_paths:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
    frozen = yaml.safe_dump(config, sort_keys=False)
    (output / "config_used.yaml").write_text(frozen)
    protocol_sha = hashlib.sha256(frozen.encode()).hexdigest()
    (output / "status.json").write_text(json.dumps(dict(status="running", protocol_sha256=protocol_sha)))
    if not online_mg:
        references = development_references(inputs, stream["validation"], evaluation_query_times(config, inputs))
    set_seed(config["physics_test"]["pino"]["seed"])
    initial_model = build_screen_model(config, inputs, torch.device("cpu"))
    initial = copy.deepcopy(initial_model.state_dict())
    del initial_model
    if historical is not None:
        baseline_initial = torch.load(Path(exact_screen_dir) / "initial_state.pt", weights_only=True)
        if initial.keys() != baseline_initial.keys() or any(not torch.equal(v, baseline_initial[k]) for k, v in initial.items()):
            raise ValueError("MG/exact comparison requires identical initial model weights")
    torch.save(initial, output / "initial_state.pt")
    metrics, histories = {}, {}
    for objective in controls:
        metrics[objective], histories[objective] = run_control(objective, initial, config, inputs, stream,
                                                            references, output / objective, device)
    return finish_screen(config, metrics, histories, output, protocol_sha)


def write_development_pairs(output, config, histories):
    sim_ids = config["physics_test"]["protocol"]["validation_ids"]
    for objective, evaluations in histories.items():
        rows = []
        for evaluation in evaluations:
            update = evaluation["update"]
            path = output / objective / f"val_pairs_{update:06d}.json"
            for record in json.loads(path.read_text()):
                row = dict(record, epoch=update, update=update,
                           benchmark="diffusion_forcing_single", sim_id=sim_ids[record["case"]],
                           t_s=0.0, t_bar=record["time"], absolute_time=record["time"],
                           rel_l2=record["normalized_rel_l2_percent"],
                           sigma_nrmse_pct=record["sigma_error_percent"])
                rows.append(row)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / objective / "val_pairs.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def finish_screen(config, metrics, histories, output, protocol_sha):
    pino = config["physics_test"]["pino"]
    model_type = config["physics_test"].get("model_type", "cvit")
    if config["physics_test"]["stage"] == "online" and pino.get("online_sampling") == "per_update":
        progress = metrics["mg"]
        completed = progress["successful_updates"] == pino["online_updates"] and not progress["pending_validation"]
        summary = dict(stage="phase1_smoke" if config["physics_test"]["smoke"] else f"phase3_online_{model_type}_autoregressive_mg",
                       model_type=model_type,
                       status="complete" if completed else "interrupted", protocol_sha256=protocol_sha,
                       successful_updates=progress["successful_updates"], registered_updates=pino["online_updates"],
                       accuracy_gate_applied=False, scientific_evidence=not config["physics_test"]["smoke"],
                       controls={"mg": histories["mg"][-1] if histories["mg"] else None},
                       elapsed_seconds=progress["elapsed_seconds"], runtime_reports=progress["runtime_reports"])
        atomic_json(output / "status.json", summary)
        atomic_json(output / ("final_metrics.json" if completed else "interrupted_metrics.json"), summary)
        return summary
    autoregressive = pino.get("residual_mode") == "autoregressive"
    raw = metrics.get("raw")
    candidate = "mg" if "mg" in metrics else "exact"
    full_gate = None if autoregressive else cvit_gate(raw, metrics["exact"], pino, config["physics_test"]["stage"])
    prefix_gate = None
    if pino.get("residual_mode", "one_step") in {"space_time", "autoregressive"}:
        def prefix_metrics(control):
            result = copy.deepcopy(metrics[control])
            for key in ("trajectory_rmse_K", "trajectory_sigma_error_percent",
                        "copy_trajectory_rmse_K", "equilibrium_trajectory_rmse_K"):
                result[key] = result[f"prefix_{key}"]
            return result
        prefix_gate = cvit_gate(prefix_metrics("raw") if raw is not None else None, prefix_metrics(candidate), pino, config["physics_test"]["stage"])
        historical = config["physics_test"]["protocol"].get("historical_exact_metrics")
        if candidate == "mg" and historical is not None:
            ratios = {key: metrics["mg"][key] / max(historical[key], 1e-15)
                      for key in ("prefix_trajectory_rmse_K", "first_step_rmse_K")}
            preserves = max(ratios.values()) <= config["physics_test"]["mg"]["fixed_max_exact_error_ratio"]
            prefix_gate.update(preserves_historical_exact_fit=preserves, mg_over_historical_exact=ratios)
            prefix_gate["passed"] &= preserves
    prefix_only = prefix_gate is not None and (autoregressive or pino["prefix_steps"] < config["physics_test"]["protocol"]["total_solver_steps"])
    gate = prefix_gate if prefix_only else full_gate
    common_training_seconds = min(rows[-1]["training_seconds"] for rows in histories.values())
    wall_clock = {name: max((r for r in rows if r["training_seconds"] <= common_training_seconds),
                           key=lambda r: r["update"]) for name, rows in histories.items()}
    model_type = config["physics_test"].get("model_type", "cvit")
    stage_name = f"phase2_fixed_{model_type}" if config["physics_test"]["stage"] == "fixed" else f"phase3_online_{model_type}"
    if prefix_only:
        stage_name += "_autoregressive_prefix" if autoregressive else "_prefix"
    if candidate == "mg":
        stage_name = f"phase4_fixed_{model_type}_autoregressive_prefix_mg"
    summary = dict(stage="phase1_smoke" if config["physics_test"]["smoke"] else stage_name,
                   model_type=model_type,
                   residual_mode=pino.get("residual_mode", "one_step"),
                   prefix_gate=prefix_gate, full_horizon_gate=full_gate,
                   protocol_sha256=protocol_sha, **gate, controls=metrics,
                   development_common_training_seconds=common_training_seconds,
                   equal_training_wall_clock_scheduled_checkpoints=wall_clock)
    if config["physics_test"]["smoke"]:
        summary["passed"] = False
    (output / "final_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(dict(status="complete", passed=summary["passed"])))
    write_development_pairs(output, config, histories)
    plot_development(output, histories, config)
    return summary


def resume_screen(output):
    output = Path(output)
    config_path = output / "config_used.yaml"
    config = yaml.safe_load(config_path.read_text())
    protocol_sha = file_sha256(config_path)
    protocol = config["physics_test"]["protocol"]
    status = json.loads((output / "status.json").read_text())
    if status.get("status") == "complete":
        raise ValueError("Completed screens cannot be resumed or overwritten")
    repo = Path(__file__).resolve().parents[2]
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(repo / name) != expected:
            raise ValueError(f"Resume source changed: {name}; start a separate declared experiment")
    for name, expected in protocol["stream_sha256"].items():
        if file_sha256(output / name) != expected:
            raise ValueError(f"Frozen replay stream changed: {name}")
    for prerequisite in protocol["prerequisites"]:
        path = Path(prerequisite["path"])
        is_gate = prerequisite.get("kind", "gate" if path.name == "final_metrics.json" else "file") == "gate"
        if file_sha256(path) != prerequisite["sha256"] or (is_gate and not json.loads(path.read_text())["passed"]):
            raise ValueError(f"Prerequisite gate changed: {path}")
    for name, expected in protocol["prescribed_input_sha256"].items():
        if file_sha256(Path(protocol["data_dir"]) / name) != expected:
            raise ValueError(f"Prescribed input changed: {name}")
    inputs = load_pino_inputs(protocol["data_dir"], output / "normalization.json")
    if inputs["normalization"] != protocol["normalization"]:
        raise ValueError("Frozen normalization changed")
    if config["physics_test"]["stage"] == "online" and config["physics_test"]["pino"].get("online_sampling") == "per_update":
        stream = dict(validation=np.load(output / "validation_inputs.npy", allow_pickle=True).tolist())
        references = np.load(output / "validation_references.npy", mmap_mode="r")
        initial = torch.load(output / "initial_state.pt", map_location="cpu", weights_only=True)
        checkpoint = output / "mg/checkpoint_latest.pt"
        if not checkpoint.exists():
            checkpoint = output / "mg/checkpoint_000000.pt"
        progress, histories = run_online_mg(initial, config, inputs, stream, references, output / "mg",
                                           resolve_device(config["training"]["device"]), checkpoint)
        return finish_screen(config, {"mg": progress}, {"mg": histories}, output, protocol_sha)
    replay = np.load(output / "interval_stream.npz")
    stream = dict(problems=np.load(output / "problem_stream.npy", allow_pickle=True).tolist(),
                  ids=replay["ids"], intervals=replay["intervals"],
                  validation=np.load(output / "validation_inputs.npy", allow_pickle=True).tolist())
    references = development_references(inputs, stream["validation"], evaluation_query_times(config, inputs))
    initial = torch.load(output / "initial_state.pt", map_location="cpu", weights_only=True)
    device = resolve_device(config["training"]["device"])
    metrics, histories = {}, {}
    for objective in config["physics_test"]["pino"].get("controls", ["raw", "exact"]):
        control_dir = output / objective
        if (control_dir / "final_metrics.json").exists():
            metrics[objective] = json.loads((control_dir / "final_metrics.json").read_text())
            histories[objective] = json.loads((control_dir / "diagnostics.json").read_text())
        else:
            checkpoints = sorted(control_dir.glob("checkpoint_*.pt"))
            if control_dir.exists() and not checkpoints:
                raise ValueError(f"No recoverable checkpoint in {control_dir}")
            metrics[objective], histories[objective] = run_control(objective, initial, config, inputs, stream,
                references, control_dir, device, checkpoints[-1] if checkpoints else None)
    return finish_screen(config, metrics, histories, output, protocol_sha)
