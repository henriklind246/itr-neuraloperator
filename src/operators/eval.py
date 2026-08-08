import torch
import numpy as np
from data.dataset import (
    TEMPORAL_SAMPLES,
    T_EPS,
    apply_protocol_pairs,
    assert_dataset_problem_version,
    build_protocol_pairs,
    compute_global_stats,
    create_dataloaders,
    load_ramp_seconds,
    load_sim_data,
    load_solver_dt,
    long_lead_pairs,
    problem_from_config,
    split_sim_ids,
)
from src.operators.fno2d import FNO2d
from src.operators.losses import (
    EPS_JUMP,
    EPS_STD,
    build_boundary_mask,
    build_interface_band,
    build_interface_mask,
    compute_interface_rel_l2,
    get_batch_interface_x,
    interface_flanking_nodes,
    interface_flanking_nodes_per_sample,
    per_sample_node_jump_errors,
    per_sample_nrmse,
    per_sample_sq_rms,
    tail_stats,
)
from src.operators.rollout import (
    RolloutOptions,
    predict_autoregressive,
    rollout_is_active,
    rollout_options_from_config,
)
from src.operators.utils import resolve_device
from pathlib import Path
import csv
import hashlib
import math
import json
from datetime import datetime, timezone

"""File loads checkpoint, runs test metrics in normalized AND physical space, outputs seed_report.json.

Per-seed result keys:
    test_rel_l2_norm        normalized-space relative L2 (%) — comparable to val_rel_l2
    test_rel_l2             physical-space (Kelvin) relative L2 (%)
    test_iface_rel_l2_norm  normalized-space interface-weighted relative L2 (%)
    test_iface_rel_l2       physical-space interface-weighted relative L2 (%)
"""

# -------- LOAD TEST SET ---------

class _RamTestTrajectories:
    """Serves only the test simulations from RAM, indexed by *absolute* sim_id.

    The full trajectories file is memory-mapped (9.9 GB) and the test loader
    accesses sims in a near-random order (pairs are sorted by lead time), which
    causes heavy page-fault I/O — and, on Apple unified-memory machines, swap
    thrashing once MPS GPU buffers compete for the same RAM. Caching just the
    test sims (~15% of the data) as a contiguous in-RAM array removes that I/O
    entirely while preserving the absolute-id indexing that ``build_item`` uses.
    """

    def __init__(self, memmap, test_ids):
        self.shape = tuple(memmap.shape)  # (num_sims, Nt, Nx, Ny) — keep full semantics
        self.dtype = memmap.dtype
        # one sequential read per test sim into a compact contiguous cache
        self._cache = {int(sid): __import__("numpy").ascontiguousarray(memmap[int(sid)])
                       for sid in test_ids}

    def __getitem__(self, key):
        # build_item indexes as trajectories[sid, t, :, :]
        if isinstance(key, tuple):
            sid = int(key[0])
            rest = key[1:]
            return self._cache[sid][rest] if rest else self._cache[sid]
        return self._cache[int(key)]


def build_test_loader(
    config,
    mu_global=None,
    sigma_global=None,
    long_lead_only=False,
    eval_all_sims=False,
    time_norm_horizon=None,
    target_times=None,
    protocols=None,
    n_snapshots_test=None,
):
    import numpy as np

    problem = problem_from_config(config)
    assert_dataset_problem_version(problem, config["data"]["t_grid_path"])
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    solver_dt = load_solver_dt(config["data"]["t_grid_path"])
    ramp_seconds = load_ramp_seconds(config["data"]["t_grid_path"])

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

    # OOD: evaluate every simulation in the (redirected) dataset, not the 15%
    # held-out slice. mu/sigma still come from the trained checkpoint's stats.
    if eval_all_sims:
        test_ids = np.arange(int(trajectories.shape[0]))

    # Use provided global stats or recompute from training set
    if mu_global is None or sigma_global is None:
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    _, _, testing_set = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=config["training"]["batch_size"],
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=10,
        n_snapshots_test=(
            n_snapshots_test
            if n_snapshots_test is not None
            else config.get("training", {}).get("n_snapshots_test", 40)
        ),
        dt=solver_dt,
        time_norm_horizon=time_norm_horizon,
        ramp_seconds=ramp_seconds,
        num_workers=0,
        temporal_samples=config["model"]["parameters"].get("temporal_samples", TEMPORAL_SAMPLES),
        problem=problem,
    )

    # OOD time protocols take precedence over long_lead_only: install the tagged
    # fixed_initial / anchored_from_horizon / ood_local_fixed_lead pairs.
    if protocols:
        test_ds = testing_set.dataset
        if target_times is None:
            raise ValueError("protocols set but target_times is None.")
        tagged = build_protocol_pairs(
            test_ds, list(protocols), list(target_times),
        )
        apply_protocol_pairs(test_ds, tagged)
    # Restrict to full-span pairs (source t=0 -> target t_final) so eval measures
    # only the hardest, maximum-lead prediction (one pair per test sim).
    elif long_lead_only:
        test_ds = testing_set.dataset
        pairs = long_lead_pairs(test_ds)
        if not pairs:
            raise ValueError(
                "long_lead_only: no full-span (source t=0 -> target t_final) pairs "
                "found in the test set."
            )
        test_ds._pairs = pairs
        test_ds._lead_times = np.array(
            [float(test_ds.t_grid[j] - test_ds.t_grid[s]) for _, s, j in pairs],
            dtype=np.float32,
        )
        test_ds._active_len = len(pairs)

    # Cache only the test sims in RAM (removes random-access memmap I/O; avoids
    # MPS unified-memory swap thrash). Disable via data.cache_test_in_ram=False.
    if config.get("data", {}).get("cache_test_in_ram", True):
        test_ds = testing_set.dataset
        test_ds.trajectories = _RamTestTrajectories(trajectories, test_ds.sim_ids)

    num_sims = int(trajectories.shape[0])
    return testing_set, x_grid, y_grid, num_sims

# -------- EVAL MODEL ON TEST SET  ---------

def _band_rel_l2(y_pred, y_true, band_float):
    """Per-sample interface-band rel L2 (%), masked-sum form.

    ``band_float`` is a (B, Nx, 1, 1) float tensor selecting each sample's
    interface columns. The voxel count cancels between numerator and
    denominator, so this matches ``compute_interface_rel_l2``'s mean/mean ratio
    while allowing the band to differ per sample.
    """
    num = torch.sum(band_float * (y_pred - y_true) ** 2)
    den = torch.sum(band_float * y_true ** 2)
    return (torch.sqrt(num / den) * 100).item()


def evaluate(
    model,
    test_loader,
    device,
    iface_mask=None,
    boundary_mask=None,
    rollout_options: RolloutOptions | None = None,
    x_grid=None,
    interface_half_width: float = 0.05,
    use_per_sample_interface: bool = False,
    interface_x: float = 0.5,
    temperature_reference_K: float = 300.0,
):
    """Return a dict of test metrics in both normalized and physical space.

    Keys:
        rel_l2_norm:        relative L2 (%) on normalized outputs — directly
                            comparable to train/val_rel_l2 in train_metrics.csv.
        rel_l2_phys:        relative L2 (%) after denormalizing to Kelvin —
                            small because the ~300 K baseline inflates the
                            denominator. Useful for "% of absolute T" intuition.
        iface_rel_l2_norm:  interface-weighted relative L2 (%) on normalized outputs.
        iface_rel_l2_phys:  interface-weighted relative L2 (%) in Kelvin.
        boundary_rel_l2_norm/phys:  edge-band relative L2 (%) — diagnostic for the
                            absolute-pad confound in cross-resolution eval.
    """
    if rollout_is_active(rollout_options):
        return _evaluate_rollout(
            model=model,
            test_loader=test_loader,
            device=device,
            rollout_options=rollout_options,
            iface_mask=iface_mask,
            boundary_mask=boundary_mask,
            x_grid=x_grid,
            interface_half_width=interface_half_width,
            use_per_sample_interface=use_per_sample_interface,
        )

    x_grid_t = None
    if x_grid is not None:
        x_grid_t = torch.as_tensor(x_grid, dtype=torch.float32, device=device)

    fixed_left = fixed_right = None
    if x_grid is not None and not use_per_sample_interface:
        fixed_left, fixed_right = interface_flanking_nodes(x_grid, interface_x)

    with torch.no_grad():
        model.eval()
        # Global sum-of-squares accumulators so rel_l2 matches the train/val
        # convention sqrt(sum_sq_err / sum_sq_true); averaging per-batch ratios
        # mis-weights unequal batch sizes and is a different statistic.
        sse_norm = 0.0
        sst_norm = 0.0
        sse_phys = 0.0
        sst_phys = 0.0
        num_error_cells = 0
        training_mu_values: list[torch.Tensor] = []
        training_sigma_values: list[torch.Tensor] = []
        iface_rel_l2_norm = 0.0
        iface_rel_l2_phys = 0.0
        boundary_rel_l2_norm = 0.0
        boundary_rel_l2_phys = 0.0

        # Unified per-sample metric accumulators (mean-over-pairs convention).
        nrmse_all: list[torch.Tensor] = []
        rmse_K_all: list[torch.Tensor] = []
        node_jump_rmse_K_all: list[torch.Tensor] = []
        node_jump_nrmse_all: list[torch.Tensor] = []
        node_jump_gnrmse_all: list[torch.Tensor] = []
        max_err_K = 0.0

        for batch in test_loader:
            x_spatial = batch["spatial"].to(device)
            cond_static = batch["cond_static"].to(device)
            forcing_seq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
            y_batch = batch["Y"].to(device)
            T_stats = batch["T_stats"].to(device)

            y_pred = model(x_spatial, cond_static, forcing_seq)

            # Normalized-space metric (same convention as train/val_rel_l2)
            sse_norm += torch.sum((y_pred - y_batch) ** 2).item()
            sst_norm += torch.sum(y_batch ** 2).item()

            # Physical-space metric (denormalized to Kelvin)
            mu_s = T_stats[:, 0]
            sigma_s = T_stats[:, 1]
            y_pred_phys = y_pred * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            y_true_phys = y_batch * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            sse_phys += torch.sum((y_pred_phys - y_true_phys) ** 2).item()
            sst_phys += torch.sum(y_true_phys ** 2).item()
            num_error_cells += int(y_true_phys.numel())
            training_mu_values.append(mu_s.detach().cpu())
            training_sigma_values.append(sigma_s.detach().cpu())

            # --- Unified per-sample metrics (nRMSE is normalization-invariant;
            # Kelvin metrics scale by the per-sample sigma). ---
            sig = sigma_s.reshape(-1)
            rms_i = per_sample_sq_rms(y_pred, y_batch)
            nrmse_all.append(per_sample_nrmse(y_pred, y_batch).cpu())
            rmse_K_all.append((rms_i * sig).cpu())
            max_err_K = max(
                max_err_K,
                torch.max(torch.abs(y_pred_phys - y_true_phys)).item(),
            )

            iface_x = get_batch_interface_x(
                batch, device, use_per_sample_interface=use_per_sample_interface
            )

            left, right = fixed_left, fixed_right
            if use_per_sample_interface and iface_x is not None and x_grid_t is not None:
                left, right = interface_flanking_nodes_per_sample(x_grid_t, iface_x)
            if left is not None and right is not None:
                err_rms_i, true_jump_rms_i = per_sample_node_jump_errors(
                    y_pred, y_batch, left, right
                )
                node_jump_rmse_K_all.append((err_rms_i * sig).cpu())
                node_jump_gnrmse_all.append(err_rms_i.cpu())
                node_jump_nrmse_all.append(
                    (err_rms_i / true_jump_rms_i.clamp_min(EPS_JUMP)).cpu()
                )

            if iface_x is not None and x_grid_t is not None:
                band = build_interface_band(x_grid_t, iface_x, interface_half_width)
                bf = band.to(y_batch.dtype).reshape(band.shape[0], band.shape[1], 1, 1)
                iface_rel_l2_norm += _band_rel_l2(y_pred, y_batch, bf)
                iface_rel_l2_phys += _band_rel_l2(y_pred_phys, y_true_phys, bf)
            elif iface_mask is not None:
                iface_rel_l2_norm += compute_interface_rel_l2(y_pred, y_batch, iface_mask)
                iface_rel_l2_phys += compute_interface_rel_l2(y_pred_phys, y_true_phys, iface_mask)

            if boundary_mask is not None:
                boundary_rel_l2_norm += compute_interface_rel_l2(y_pred, y_batch, boundary_mask)
                boundary_rel_l2_phys += compute_interface_rel_l2(y_pred_phys, y_true_phys, boundary_mask)

        n_batches = len(test_loader)
        rel_l2_norm = (sse_norm / sst_norm) ** 0.5 * 100 if sst_norm > 0 else 0.0
        rel_l2_phys = (sse_phys / sst_phys) ** 0.5 * 100 if sst_phys > 0 else 0.0
        iface_rel_l2_norm /= n_batches
        iface_rel_l2_phys /= n_batches
        boundary_rel_l2_norm /= n_batches
        boundary_rel_l2_phys /= n_batches

    nrmse_stats = tail_stats(torch.cat(nrmse_all)) if nrmse_all else tail_stats(torch.empty(0))
    rmse_K_stats = tail_stats(torch.cat(rmse_K_all)) if rmse_K_all else tail_stats(torch.empty(0))
    if training_mu_values and num_error_cells > 0:
        mu_values = torch.cat(training_mu_values).to(torch.float64)
        sigma_values = torch.cat(training_sigma_values).to(torch.float64)
        mu_train = float(mu_values[0])
        sigma_train = float(sigma_values[0])
        if not torch.allclose(mu_values, mu_values[:1], rtol=1e-6, atol=1e-8):
            raise ValueError("evaluate requires one training mean across the test loader")
        if not torch.allclose(sigma_values, sigma_values[:1], rtol=1e-6, atol=1e-8):
            raise ValueError("evaluate requires one training scale across the test loader")
        temperature_rise_scale_K = math.sqrt(
            sigma_train ** 2 + (mu_train - float(temperature_reference_K)) ** 2
        )
        pooled_rmse_K = math.sqrt(sse_phys / num_error_cells)
        field_gnrmse_pct = (
            100.0 * pooled_rmse_K / temperature_rise_scale_K
            if temperature_rise_scale_K > 0.0 else float("nan")
        )
    else:
        temperature_rise_scale_K = float("nan")
        field_gnrmse_pct = float("nan")
    if node_jump_nrmse_all:
        jump_nrmse_stats = tail_stats(torch.cat(node_jump_nrmse_all))
        node_jump_rmse_K_stats = tail_stats(torch.cat(node_jump_rmse_K_all))
        node_jump_gnrmse_stats = tail_stats(torch.cat(node_jump_gnrmse_all))
    else:
        jump_nrmse_stats = tail_stats(torch.empty(0))
        node_jump_rmse_K_stats = tail_stats(torch.empty(0))
        node_jump_gnrmse_stats = tail_stats(torch.empty(0))

    return {
        "rel_l2_norm": rel_l2_norm,
        "rel_l2_phys": rel_l2_phys,
        "iface_rel_l2_norm": iface_rel_l2_norm,
        "iface_rel_l2_phys": iface_rel_l2_phys,
        "boundary_rel_l2_norm": boundary_rel_l2_norm,
        "boundary_rel_l2_phys": boundary_rel_l2_phys,
        "nrmse": nrmse_stats["mean"] * 100.0,
        "nrmse_p50": nrmse_stats["p50"] * 100.0,
        "nrmse_iqr": nrmse_stats["iqr"] * 100.0,
        "nrmse_p90": nrmse_stats["p90"] * 100.0,
        "nrmse_p99": nrmse_stats["p99"] * 100.0,
        "nrmse_max": nrmse_stats["max"] * 100.0,
        "rmse_K": rmse_K_stats["mean"],
        "rmse_K_p90": rmse_K_stats["p90"],
        "rmse_K_p95": rmse_K_stats["p95"],
        "rmse_K_p99": rmse_K_stats["p99"],
        "rmse_K_max": rmse_K_stats["max"],
        "gnrmse_pct": field_gnrmse_pct,
        "gnrmse_pct_p99": (
            100.0 * rmse_K_stats["p99"] / temperature_rise_scale_K
            if temperature_rise_scale_K > 0.0 else float("nan")
        ),
        "temperature_rise_scale_K": temperature_rise_scale_K,
        "max_err_K": max_err_K,
        "node_jump_rmse_K": node_jump_rmse_K_stats["mean"],
        "node_jump_rmse_K_p95": node_jump_rmse_K_stats["p95"],
        "node_jump_nrmse": jump_nrmse_stats["mean"] * 100.0,
        "node_jump_nrmse_p90": jump_nrmse_stats["p90"] * 100.0,
        "node_jump_nrmse_p99": jump_nrmse_stats["p99"] * 100.0,
        "node_jump_nrmse_max": jump_nrmse_stats["max"] * 100.0,
        "node_jump_gnrmse_pct": node_jump_gnrmse_stats["mean"] * 100.0,
        "node_jump_gnrmse_pct_p99": node_jump_gnrmse_stats["p99"] * 100.0,
    }


def _evaluate_rollout(
    model,
    test_loader,
    device,
    rollout_options: RolloutOptions,
    iface_mask=None,
    boundary_mask=None,
    x_grid=None,
    interface_half_width: float = 0.05,
    use_per_sample_interface: bool = False,
):
    dataset = test_loader.dataset
    if not hasattr(dataset, "_pairs"):
        raise ValueError("rollout evaluation requires a SnapshotPairDataset with _pairs")

    x_grid_t = None
    if use_per_sample_interface and x_grid is not None:
        x_grid_t = torch.as_tensor(x_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        model.eval()
        # Global sum-of-squares accumulators (rel_l2 convention matches train/val
        # and the single-step eval path); per-item ratio averaging is a different
        # statistic that over-weights small-signal items.
        sse_norm = 0.0
        sst_norm = 0.0
        sse_phys = 0.0
        sst_phys = 0.0
        iface_rel_l2_norm = 0.0
        iface_rel_l2_phys = 0.0
        boundary_rel_l2_norm = 0.0
        boundary_rel_l2_phys = 0.0

        for idx in range(len(dataset)):
            sim_id, s, j = dataset._pairs[idx]
            item = dataset[idx]
            y_batch = item["Y"].unsqueeze(0).to(device)
            T_stats = item["T_stats"].unsqueeze(0).to(device)
            item_batch = {"T_stats": T_stats}

            y_pred = predict_autoregressive(
                model=model,
                dataset=dataset,
                sim_id=int(sim_id),
                s=int(s),
                j=int(j),
                num_substeps=rollout_options.num_substeps,
                device=device,
            )

            sse_norm += torch.sum((y_pred - y_batch) ** 2).item()
            sst_norm += torch.sum(y_batch ** 2).item()

            mu_s = T_stats[:, 0]
            sigma_s = T_stats[:, 1]
            y_pred_phys = y_pred * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            y_true_phys = y_batch * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            sse_phys += torch.sum((y_pred_phys - y_true_phys) ** 2).item()
            sst_phys += torch.sum(y_true_phys ** 2).item()

            iface_x = get_batch_interface_x(
                item_batch, device, use_per_sample_interface=use_per_sample_interface
            )
            if iface_x is not None and x_grid_t is not None:
                band = build_interface_band(x_grid_t, iface_x, interface_half_width)
                bf = band.to(y_batch.dtype).reshape(band.shape[0], band.shape[1], 1, 1)
                iface_rel_l2_norm += _band_rel_l2(y_pred, y_batch, bf)
                iface_rel_l2_phys += _band_rel_l2(y_pred_phys, y_true_phys, bf)
            elif iface_mask is not None:
                iface_rel_l2_norm += compute_interface_rel_l2(y_pred, y_batch, iface_mask)
                iface_rel_l2_phys += compute_interface_rel_l2(y_pred_phys, y_true_phys, iface_mask)

            if boundary_mask is not None:
                boundary_rel_l2_norm += compute_interface_rel_l2(y_pred, y_batch, boundary_mask)
                boundary_rel_l2_phys += compute_interface_rel_l2(y_pred_phys, y_true_phys, boundary_mask)

        n_items = len(dataset)
        rel_l2_norm = (sse_norm / sst_norm) ** 0.5 * 100 if sst_norm > 0 else 0.0
        rel_l2_phys = (sse_phys / sst_phys) ** 0.5 * 100 if sst_phys > 0 else 0.0
        iface_rel_l2_norm /= n_items
        iface_rel_l2_phys /= n_items
        boundary_rel_l2_norm /= n_items
        boundary_rel_l2_phys /= n_items

    return {
        "rel_l2_norm": rel_l2_norm,
        "rel_l2_phys": rel_l2_phys,
        "iface_rel_l2_norm": iface_rel_l2_norm,
        "iface_rel_l2_phys": iface_rel_l2_phys,
        "boundary_rel_l2_norm": boundary_rel_l2_norm,
        "boundary_rel_l2_phys": boundary_rel_l2_phys,
    }


# --------- EVAL ALL SEEDS IN RUNS ---------

def eval_all_seeds(
    run_root: str,
    data_dir: str | None = None,
    report_name: str = "seed_report.json",
    padding_reference_resolution: int | None = None,
    rollout_enabled: bool | None = None,
    rollout_num_substeps: int | None = None,
    rollout_partition: str | None = None,
    long_lead_only: bool = False,
    eval_all_sims: bool = False,
    time_norm_horizon: float | None = None,
    target_times: list[float] | None = None,
    protocols: list[str] | None = None,
):
    run_root = Path(run_root)
    results = []

    print("Testing started.")
    if data_dir is not None:
        print(f"Overriding dataset paths with --data-dir: {data_dir}")

    for seed_dir in sorted(run_root.glob("seed*")):
        ckpt_path = seed_dir / "fno2d_best.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt['conf']
        device = resolve_device(config.get("training", {}).get("device", "auto"))
        rollout_options = rollout_options_from_config(
            config,
            enabled=rollout_enabled,
            num_substeps=rollout_num_substeps,
            partition=rollout_partition,
        )
        if padding_reference_resolution is not None:
            config["model"]["parameters"]["padding_reference_resolution"] = int(padding_reference_resolution)

        # Cross-resolution eval: point the trained checkpoint at a different
        # dataset (e.g. a finer grid). Only the test split is consumed, and the
        # training-distribution normalization (mu_global/sigma_global from the
        # checkpoint) is reused downstream, so this stays apples-to-apples.
        if data_dir is not None:
            data_dir_path = Path(data_dir)
            config["data"]["trajectories.npy"] = str(data_dir_path / "trajectories.npy")
            config["data"]["x_grid_path"] = str(data_dir_path / "x_grid.npy")
            config["data"]["y_grid_path"] = str(data_dir_path / "y_grid.npy")
            config["data"]["t_grid_path"] = str(data_dir_path / "t_grid.npy")
            config["data"]["sim_params_path"] = str(data_dir_path / "sim_params.npy")

        test_loader, x_grid, y_grid, num_sims = build_test_loader(
            config,
            mu_global=ckpt.get("mu_global"),
            sigma_global=ckpt.get("sigma_global"),
            long_lead_only=long_lead_only,
            eval_all_sims=eval_all_sims,
            time_norm_horizon=time_norm_horizon,
            target_times=target_times,
            protocols=protocols,
        )

        loss_cfg = config.get("training", {}).get("loss", {})
        iface_mask = build_interface_mask(
            x_grid, y_grid,
            loss_cfg.get("interface_x", 0.5), loss_cfg.get("interface_half_width", 0.05),
        ).to(device)
        boundary_mask = build_boundary_mask(x_grid, y_grid, width=0.05).to(device)
        use_per_sample_interface = bool(loss_cfg.get("per_sample_interface_x", False))

        model_cfg = config['model']['parameters']
        dims = problem_from_config(config).dims
        fno = FNO2d(
            modes1=model_cfg["modes1"],
            modes2=model_cfg["modes2"],
            width=model_cfg["width"],
            in_channels=dims.in_channels,
            out_channels=model_cfg.get("out_channels", 1),
            n_layers=model_cfg.get("n_layers", 4),
            cond_static_dim=dims.cond_static_dim,
            cond_hidden=model_cfg.get("cond_hidden", 256),
            temporal_token_dim=dims.temporal_token_dim,
            temporal_hidden=model_cfg.get("temporal_hidden", 128),
            forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
            forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
            use_temporal_encoder=dims.use_temporal_encoder,
            use_forcing_time_aug=dims.use_forcing_time_aug,
            forcing_cond_mode=model_cfg.get("forcing_cond_mode", "both"),
            s_y_channel=dims.s_y_channel,
            padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
            padding_mode=model_cfg.get("padding_mode", "zeros"),
            cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
            hard_right_dirichlet=model_cfg.get("hard_right_dirichlet", False),
        )
        fno.load_state_dict(ckpt['model_state'])
        fno.to(device)

        metrics = evaluate(
            model=fno, test_loader=test_loader, device=device,
            iface_mask=iface_mask, boundary_mask=boundary_mask,
            rollout_options=rollout_options,
            x_grid=x_grid,
            interface_half_width=loss_cfg.get("interface_half_width", 0.05),
            use_per_sample_interface=use_per_sample_interface,
            interface_x=loss_cfg.get("interface_x", 0.5),
        )

        results.append(
            {
                "seed": ckpt.get("seed", seed_dir.name),
                "best_epoch": ckpt["epoch"],
                "best_val": float(ckpt["best_val"]),
                "num_sims": int(num_sims),
                "test_rel_l2_norm": float(metrics["rel_l2_norm"]),
                "test_rel_l2": float(metrics["rel_l2_phys"]),
                "test_iface_rel_l2_norm": float(metrics["iface_rel_l2_norm"]),
                "test_iface_rel_l2": float(metrics["iface_rel_l2_phys"]),
                "test_boundary_rel_l2_norm": float(metrics["boundary_rel_l2_norm"]),
                "test_boundary_rel_l2": float(metrics["boundary_rel_l2_phys"]),
                # NaN (not 0.0) for metrics absent from the rollout path: a
                # silent 0.0 reads as a perfect surrogate in seed_report.json.
                "test_nrmse": float(metrics.get("nrmse", float("nan"))),
                "test_nrmse_p50": float(metrics.get("nrmse_p50", float("nan"))),
                "test_nrmse_iqr": float(metrics.get("nrmse_iqr", float("nan"))),
                "test_nrmse_p90": float(metrics.get("nrmse_p90", float("nan"))),
                "test_nrmse_p99": float(metrics.get("nrmse_p99", float("nan"))),
                "test_nrmse_max": float(metrics.get("nrmse_max", float("nan"))),
                "test_rmse_K": float(metrics.get("rmse_K", float("nan"))),
                "test_rmse_K_p90": float(metrics.get("rmse_K_p90", float("nan"))),
                "test_rmse_K_p95": float(metrics.get("rmse_K_p95", float("nan"))),
                "test_rmse_K_p99": float(metrics.get("rmse_K_p99", float("nan"))),
                "test_rmse_K_max": float(metrics.get("rmse_K_max", float("nan"))),
                "test_gnrmse_pct": float(metrics.get("gnrmse_pct", float("nan"))),
                "test_gnrmse_pct_p99": float(metrics.get("gnrmse_pct_p99", float("nan"))),
                "temperature_rise_scale_K": float(metrics.get("temperature_rise_scale_K", float("nan"))),
                "test_max_err_K": float(metrics.get("max_err_K", float("nan"))),
                "test_node_jump_rmse_K": float(metrics.get("node_jump_rmse_K", float("nan"))),
                "test_node_jump_rmse_K_p95": float(metrics.get("node_jump_rmse_K_p95", float("nan"))),
                "test_node_jump_nrmse": float(metrics.get("node_jump_nrmse", float("nan"))),
                "test_node_jump_nrmse_p90": float(metrics.get("node_jump_nrmse_p90", float("nan"))),
                "test_node_jump_nrmse_p99": float(metrics.get("node_jump_nrmse_p99", float("nan"))),
                "test_node_jump_nrmse_max": float(metrics.get("node_jump_nrmse_max", float("nan"))),
                "test_node_jump_gnrmse_pct": float(metrics.get("node_jump_gnrmse_pct", float("nan"))),
                "test_node_jump_gnrmse_pct_p99": float(metrics.get("node_jump_gnrmse_pct_p99", float("nan"))),
                "rollout_enabled": bool(rollout_options.enabled),
                "rollout_num_substeps": int(rollout_options.num_substeps),
                "rollout_partition": rollout_options.partition,
                "long_lead_only": bool(long_lead_only),
                "eval_all_sims": bool(eval_all_sims),
                "time_norm_horizon": (
                    float(time_norm_horizon) if time_norm_horizon is not None else None
                ),
                "protocols": list(protocols) if protocols else None,
                "ckpt": str(ckpt_path),
            }
        )

    # sort by normalized test performance (apples-to-apples with val_rel_l2)
    results.sort(key=lambda r: r["test_rel_l2_norm"])
    return results


# ---------- COMPUTE MEAN + STD FOR TEST ERROR ACROSS SEEDS ----------

def mean_std(values: list[float]) -> tuple[float, float]:
    """Sample std for small-n reporting; returns (mean, std)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(values) / n
    if n == 1:
        return mu, 0.0
    # variance formula for a sample
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)

def print_seed_report(results: list[dict]) -> dict:
    best_vals = [r["best_val"] for r in results]
    test_rel_l2_norm = [r["test_rel_l2_norm"] for r in results]
    test_rel_l2 = [r["test_rel_l2"] for r in results]
    test_iface_norm = [r["test_iface_rel_l2_norm"] for r in results]
    test_iface = [r["test_iface_rel_l2"] for r in results]
    test_bnd_norm = [r["test_boundary_rel_l2_norm"] for r in results]
    test_bnd = [r["test_boundary_rel_l2"] for r in results]

    val_mu, val_std = mean_std(best_vals)
    norm_mu, norm_std = mean_std(test_rel_l2_norm)
    test_mu, test_std = mean_std(test_rel_l2)
    iface_norm_mu, iface_norm_std = mean_std(test_iface_norm)
    iface_mu, iface_std = mean_std(test_iface)
    bnd_norm_mu, bnd_norm_std = mean_std(test_bnd_norm)
    bnd_mu, bnd_std = mean_std(test_bnd)

    def _seed_stat(key: str) -> tuple[float, float]:
        return mean_std([r.get(key, 0.0) for r in results])

    nrmse_mu, nrmse_std = _seed_stat("test_nrmse")
    nrmse_p50_mu, _ = _seed_stat("test_nrmse_p50")
    nrmse_iqr_mu, _ = _seed_stat("test_nrmse_iqr")
    nrmse_p90_mu, _ = _seed_stat("test_nrmse_p90")
    nrmse_p99_mu, _ = _seed_stat("test_nrmse_p99")
    nrmse_max_mu, _ = _seed_stat("test_nrmse_max")
    rmse_K_mu, rmse_K_std = _seed_stat("test_rmse_K")
    rmse_K_p90_mu, _ = _seed_stat("test_rmse_K_p90")
    rmse_K_p95_mu, _ = _seed_stat("test_rmse_K_p95")
    rmse_K_p99_mu, _ = _seed_stat("test_rmse_K_p99")
    rmse_K_max_mu, _ = _seed_stat("test_rmse_K_max")
    gnrmse_pct_mu, gnrmse_pct_std = _seed_stat("test_gnrmse_pct")
    gnrmse_pct_p99_mu, _ = _seed_stat("test_gnrmse_pct_p99")
    max_err_K_mu, _ = _seed_stat("test_max_err_K")
    jump_rmse_K_mu, jump_rmse_K_std = _seed_stat("test_node_jump_rmse_K")
    jump_rmse_K_p95_mu, _ = _seed_stat("test_node_jump_rmse_K_p95")
    jump_nrmse_mu, jump_nrmse_std = _seed_stat("test_node_jump_nrmse")
    jump_gnrmse_pct_mu, jump_gnrmse_pct_std = _seed_stat("test_node_jump_gnrmse_pct")
    jump_gnrmse_pct_p99_mu, _ = _seed_stat("test_node_jump_gnrmse_pct_p99")

    print("\n===== Seed Report =====")
    print(f"Number of seeds: {len(results)}")
    print(f"best_val_loss            mean, std: ({val_mu}, {val_std})")
    print(f"test_rel_l2_norm         mean, std: ({norm_mu}, {norm_std})   <- comparable to val_rel_l2")
    print(f"test_rel_l2 (physical)   mean, std: ({test_mu}, {test_std})")
    print(f"test_iface_rel_l2_norm   mean, std: ({iface_norm_mu}, {iface_norm_std})")
    print(f"test_iface_rel_l2 (phys) mean, std: ({iface_mu}, {iface_std})")
    print(f"test_boundary_rel_l2_norm mean, std: ({bnd_norm_mu}, {bnd_norm_std})")
    print(f"test_boundary_rel_l2(phys) mean, std: ({bnd_mu}, {bnd_std})")
    print(f"test_rmse_K (Kelvin)     mean, std: ({rmse_K_mu}, {rmse_K_std})   <- physical headline")
    print(f"test_rmse_K tails (K)    p90/p95/p99/max_sample: ({rmse_K_p90_mu}, {rmse_K_p95_mu}, {rmse_K_p99_mu}, {rmse_K_max_mu})")
    print(f"test_max_err_K           mean: {max_err_K_mu}   (Kelvin pointwise worst-case)")
    print(f"test_gnrmse (%)          pooled, pair-p99: ({gnrmse_pct_mu}, {gnrmse_pct_p99_mu})   (training temperature-rise scale)")
    print(f"test_nrmse (%)           mean, std: ({nrmse_mu}, {nrmse_std})   <- diagnostic, small-signal divergence")
    print(f"test_nrmse dist (%)      p50/IQR/p90/p99/max: ({nrmse_p50_mu}, {nrmse_iqr_mu}, {nrmse_p90_mu}, {nrmse_p99_mu}, {nrmse_max_mu})")
    print(f"test_node_jump_rmse_K    mean, std: ({jump_rmse_K_mu}, {jump_rmse_K_std})   (Kelvin)")
    print(f"test_node_jump_rmse_K    p95: {jump_rmse_K_p95_mu}   (Kelvin)")
    print(f"test_node_jump_gnrmse(%) mean, p99: ({jump_gnrmse_pct_mu}, {jump_gnrmse_pct_p99_mu})")
    print(f"test_node_jump_nrmse (%) mean, std: ({jump_nrmse_mu}, {jump_nrmse_std})   <- offset-free interface")

    # Seed selection must never look at the test set: picking the seed that
    # minimizes test error and then quoting that seed's test error reports a
    # minimum over seeds as if it were a draw, which is optimistically biased.
    # best_val is the same quantity _select_seed_checkpoint uses per seed.
    best = min(results, key=lambda r: r["best_val"])
    print(
        f"Best by lowest validation loss: seed={best['seed']} "
        f"best_val={best['best_val']} "
        f"test_rel_l2_norm={best['test_rel_l2_norm']} "
        f"test_rel_l2={best['test_rel_l2']} "
        f"test_iface_rel_l2_norm={best['test_iface_rel_l2_norm']} "
        f"test_iface_rel_l2={best['test_iface_rel_l2']}"
    )

    return {
        "num_seeds": len(results),
        "best_val_loss_mean": val_mu,
        "best_val_loss_std": val_std,
        "test_rel_l2_norm_mean": norm_mu,
        "test_rel_l2_norm_std": norm_std,
        "test_rel_l2_mean": test_mu,
        "test_rel_l2_std": test_std,
        "test_iface_rel_l2_norm_mean": iface_norm_mu,
        "test_iface_rel_l2_norm_std": iface_norm_std,
        "test_iface_rel_l2_mean": iface_mu,
        "test_iface_rel_l2_std": iface_std,
        "test_boundary_rel_l2_norm_mean": bnd_norm_mu,
        "test_boundary_rel_l2_norm_std": bnd_norm_std,
        "test_boundary_rel_l2_mean": bnd_mu,
        "test_boundary_rel_l2_std": bnd_std,
        "test_nrmse_mean": nrmse_mu,
        "test_nrmse_std": nrmse_std,
        "test_nrmse_p50_mean": nrmse_p50_mu,
        "test_nrmse_iqr_mean": nrmse_iqr_mu,
        "test_nrmse_p90_mean": nrmse_p90_mu,
        "test_nrmse_p99_mean": nrmse_p99_mu,
        "test_nrmse_max_mean": nrmse_max_mu,
        "test_rmse_K_mean": rmse_K_mu,
        "test_rmse_K_std": rmse_K_std,
        "test_rmse_K_p90_mean": rmse_K_p90_mu,
        "test_rmse_K_p95_mean": rmse_K_p95_mu,
        "test_rmse_K_p99_mean": rmse_K_p99_mu,
        "test_rmse_K_max_mean": rmse_K_max_mu,
        "test_gnrmse_pct_mean": gnrmse_pct_mu,
        "test_gnrmse_pct_std": gnrmse_pct_std,
        "test_gnrmse_pct_p99_mean": gnrmse_pct_p99_mu,
        "test_max_err_K_mean": max_err_K_mu,
        "test_node_jump_rmse_K_mean": jump_rmse_K_mu,
        "test_node_jump_rmse_K_std": jump_rmse_K_std,
        "test_node_jump_rmse_K_p95_mean": jump_rmse_K_p95_mu,
        "test_node_jump_nrmse_mean": jump_nrmse_mu,
        "test_node_jump_nrmse_std": jump_nrmse_std,
        "test_node_jump_gnrmse_pct_mean": jump_gnrmse_pct_mu,
        "test_node_jump_gnrmse_pct_std": jump_gnrmse_pct_std,
        "test_node_jump_gnrmse_pct_p99_mean": jump_gnrmse_pct_p99_mu,
    }

def save_report(run_root: str, results: list[dict], summary: dict,
                report_name: str = "seed_report.json") -> None:
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "summary": summary,
        "per_seed": results,
    }
    out_path = Path(run_root) / report_name
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved report -> {out_path}")


# ---------- PER-SAMPLE TEST RECORDS (paper figures data source) ----------

TEST_RECORD_FIELDS = [
    "provenance_id", "sim_id", "s", "j", "t_s", "t_bar", "R_c", "benchmark",
    "temporal_family", "spatial_family",
    "x_h", "y_h", "A", "freq", "regime",
    "R_c_amp", "R_c_y0", "R_c_sigma", "R_c_A",
    "x_I", "rel_l2_pct", "iface_rel_l2_pct",
    "nrmse_pct", "rmse_K", "gnrmse_pct",
    "node_jump_rmse_K", "node_jump_nrmse_pct", "node_jump_gnrmse_pct",
    "node_jump_abs_max_pred_K", "node_jump_abs_max_true_K",
    # OOD identity (joined from ood_metadata.jsonl by sim_id; in-distribution
    # defaults when the sidecar is absent).
    "ood_axis", "ood_value", "ood_repeat", "latents_hash", "distribution_class",
    # Time-pair protocol tagging (from dataset._pair_tags; empty without protocols).
    "protocol",
    "source_time_requested", "source_time_actual", "lead_time_actual",
    "target_time_requested", "target_time_actual",
    "dataset_t_final", "time_norm_horizon",
    # Pooled sufficient statistics (physical K^2) so the aggregator reconstructs
    # RMSE_sim and pooled rel-L2 without re-reading trajectories.
    "sse_K2", "num_error_cells",
    "interface_sse_K2", "num_interface_cells",
    "target_sse_K2", "interface_target_sse_K2",
]

# In-distribution defaults for the OOD identity columns when no sidecar exists.
_OOD_RECORD_DEFAULT = {
    "ood_axis": "",
    "ood_value": "",
    "ood_repeat": "",
    "latents_hash": "",
    "distribution_class": "in_distribution",
}


def _load_ood_sidecar(data_dir: Path | None) -> dict[int, dict]:
    """Return {sim_id: ood record} from ``ood_metadata.jsonl`` under `data_dir`.

    Absent file (legacy datasets) yields an empty map, so the caller falls back
    to the in-distribution defaults and existing eval behavior is unchanged.
    """
    if data_dir is None:
        return {}
    path = Path(data_dir) / "ood_metadata.jsonl"
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["sim_id"])] = rec
    return out


def _classify_time_target(target_actual: float, horizon: float) -> str:
    """4-level class for an evaluation-time target vs the trained horizon."""
    atol = 1e-9
    if abs(target_actual - horizon) <= atol:
        return "boundary"
    if target_actual < horizon:
        return "in_distribution"
    return "out_of_distribution"


def _select_seed_checkpoint(run_root: Path, seed=None):
    """Return (seed_dir, ckpt) for the requested seed or the lowest best_val.

    With no `seed`, scans every ``seed*/fno2d_best.pt`` and picks the checkpoint
    with the smallest ``best_val``. With an explicit `seed`, requires
    ``seed{seed}/fno2d_best.pt`` to exist.
    """
    run_root = Path(run_root)
    best = None  # (best_val, seed_dir, ckpt)
    for seed_dir in sorted(run_root.glob("seed*")):
        if seed is not None and seed_dir.name != f"seed{seed}":
            continue
        ckpt_path = seed_dir / "fno2d_best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        best_val = float(ckpt.get("best_val", float("inf")))
        if best is None or best_val < best[0]:
            best = (best_val, seed_dir, ckpt)
    if best is None:
        suffix = f" for seed={seed}" if seed is not None else ""
        raise FileNotFoundError(f"No fno2d_best.pt checkpoint found under {run_root}{suffix}")
    return best[1], best[2]


def _amp_freq_from_params(params: dict) -> tuple[float | None, float | None]:
    """Return (amplitude, frequency) for a sim, or (None, None) when undefined.

    Source carries a top-level patch ``A`` and no frequency; sin-forced
    benchmarks (interfaces, and the sin family of forcing) expose ``A``/``f``
    inside ``temporal_params``. Non-sin forcing families have no scalar
    amplitude/frequency, so both come back None.
    """
    tp = params.get("temporal_params")
    if isinstance(tp, dict) and params.get("temporal_family") == "sin":
        A = tp.get("A")
        f = tp.get("f")
        return (None if A is None else float(A), None if f is None else float(f))
    if "A" in params and tp is None:
        return float(params["A"]), None
    return None, None


TEST_RECORD_PROVENANCE_SCHEMA = "test-records-provenance/v1"
TEST_RECORD_SCHEMA_VERSION = 4
TEMPERATURE_REFERENCE_K = 300.0
NORMALIZATION_PROVENANCE_SIDECAR = "normalization_provenance.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_dataset_file_hashes(config: dict) -> dict[str, str]:
    data = config["data"]
    paths = {
        "trajectories": Path(data["trajectories.npy"]),
        "sim_params": Path(data["sim_params_path"]),
        "x_grid": Path(data["x_grid_path"]),
        "y_grid": Path(data["y_grid_path"]),
        "t_grid": Path(data["t_grid_path"]),
    }
    data_dir = paths["t_grid"].parent
    for name in ("meta.npy", "dt.npy", "ramp_seconds.npy"):
        path = data_dir / name
        if path.is_file():
            paths[name] = path
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _evaluation_population_identity(dataset, x_grid, y_grid, config: dict) -> dict:
    return {
        "sim_ids": np.asarray(dataset.sim_ids, dtype=np.int64).tolist(),
        "pairs": [list(map(int, pair)) for pair in dataset._pairs],
        "t_grid": np.asarray(dataset.t_grid, dtype=np.float64).tolist(),
        "x_grid_sha256": hashlib.sha256(
            np.ascontiguousarray(x_grid).view(np.uint8)
        ).hexdigest(),
        "y_grid_sha256": hashlib.sha256(
            np.ascontiguousarray(y_grid).view(np.uint8)
        ).hexdigest(),
        "dataset_file_hashes": _evaluation_dataset_file_hashes(config),
        "benchmark": config.get("benchmark", {}).get("name"),
        "representation": config.get("benchmark", {}).get("representation"),
    }


def _normalization_provenance_for_records(
    seed_dir: Path, checkpoint: dict
) -> tuple[dict, str, str | None]:
    norm = checkpoint.get("normalization_provenance")
    if norm:
        return norm, "checkpoint", None

    path = seed_dir / NORMALIZATION_PROVENANCE_SIDECAR
    if not path.is_file():
        return {}, "unavailable", None
    payload = json.loads(path.read_text())
    required = {
        "normalization_definition",
        "normalization_definition_hash",
        "training_population_hash",
        "training_population",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"{path} is missing normalization provenance fields: {missing}"
        )
    return payload, str(path), _sha256_file(path)


def _write_test_record_provenance(
    *,
    seed_dir: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    config: dict,
    dataset,
    x_grid,
    y_grid,
    benchmark: str,
    rollout_options: RolloutOptions,
    output_csv_name: str,
) -> tuple[str, Path]:
    mu_train = float(checkpoint["mu_global"])
    sigma_train = float(checkpoint["sigma_global"])
    rise_scale = math.sqrt(
        sigma_train ** 2 + (mu_train - TEMPERATURE_REFERENCE_K) ** 2
    )
    norm, norm_source, norm_source_sha256 = _normalization_provenance_for_records(
        seed_dir, checkpoint
    )
    prediction_mode = (
        f"autoregressive_{int(rollout_options.num_substeps)}_substeps"
        if rollout_is_active(rollout_options) else "direct_pair"
    )
    evaluation_population = _evaluation_population_identity(
        dataset, x_grid, y_grid, config
    )
    payload = {
        "schema": TEST_RECORD_PROVENANCE_SCHEMA,
        "test_record_schema_version": TEST_RECORD_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_selection_metric": config.get("training", {}).get(
            "checkpoint_metric", "val_rel_l2"
        ),
        "validation_selected_epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_value": float(checkpoint.get("best_val", float("nan"))),
        "training_mean_K": mu_train,
        "training_population_std_K": sigma_train,
        "normalization_standard_deviation_convention": "population (ddof=0)",
        "training_temperature_rise_rms_K": rise_scale,
        "temperature_reference_K": TEMPERATURE_REFERENCE_K,
        "normalization_definition": norm.get("normalization_definition"),
        "normalization_definition_hash": norm.get("normalization_definition_hash"),
        "training_population_hash": norm.get("training_population_hash"),
        "normalization_provenance_source": norm_source,
        "normalization_provenance_source_sha256": norm_source_sha256,
        "evaluation_population": evaluation_population,
        "evaluation_population_hash": _canonical_hash(evaluation_population),
        "prediction_mode": prediction_mode,
        "rollout_enabled": bool(rollout_options.enabled),
        "rollout_num_substeps": int(rollout_options.num_substeps),
        "rollout_partition": rollout_options.partition,
        "benchmark": benchmark,
        "representation": config.get("benchmark", {}).get(
            "representation", "temporal_encoder"
        ),
        "seed": str(checkpoint.get("seed", seed_dir.name.removeprefix("seed"))),
        "code_commit": (
            (seed_dir / "git_commit.txt").read_text().strip()
            if (seed_dir / "git_commit.txt").exists() else None
        ),
        "code_dirty": bool(
            (seed_dir / "git_status.txt").exists()
            and (seed_dir / "git_status.txt").read_text().strip()
        ),
        "test_records_file": output_csv_name,
        "n_simulations": int(len(dataset.sim_ids)),
        # Effective snapshot count actually used, not the requested one: it
        # governs the pair count, so a short study CSV can never be mistaken
        # for a full publication CSV.
        "n_snapshots_test": int(len(dataset.t_indices)),
        "n_pairs": int(len(dataset)),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    provenance_id = _canonical_hash(payload)
    payload["provenance_id"] = provenance_id
    provenance_path = seed_dir / f"{Path(output_csv_name).stem}.provenance.json"
    provenance_path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    return provenance_id, provenance_path


def write_test_records(
    run_root,
    seed=None,
    out_name: str = "test_records.csv",
    data_dir: str | None = None,
    rollout_enabled: bool | None = None,
    rollout_num_substeps: int | None = None,
    rollout_partition: str | None = None,
    eval_all_sims: bool = False,
    time_norm_horizon: float | None = None,
    target_times: list[float] | None = None,
    protocols: list[str] | None = None,
    device: str | None = None,
    inference_batch_size: int = 1,
    n_snapshots_test: int | None = None,
) -> Path:
    """Write one per-(sim_id, s, j) test-pair record row for paper figures.

    Loads the best (or requested) seed checkpoint under `run_root`, rebuilds the
    test loader, and forwards every test pair through the model. Metrics are in
    normalized space (same convention as ``train/val_rel_l2`` and
    ``evaluate``'s ``rel_l2_norm``): ``rel_l2_pct`` is the global relative L2 and
    ``iface_rel_l2_pct`` is the interface-band relative L2 computed around THAT
    sample's own ``interface_x`` (dynamic for the interfaces benchmark). Returns
    the written CSV path under the seed directory.
    """
    seed_dir, ckpt = _select_seed_checkpoint(Path(run_root), seed)
    config = ckpt["conf"]
    # The checkpoint bakes in the device it trained on, and "auto" resolves to
    # CPU off CUDA, so re-scoring a checkpoint on a workstation needs an
    # override to reach an available accelerator.
    device_str = device if device is not None else config.get("training", {}).get("device", "auto")
    device = resolve_device(device_str)
    rollout_options = rollout_options_from_config(
        config,
        enabled=rollout_enabled,
        num_substeps=rollout_num_substeps,
        partition=rollout_partition,
    )
    inference_batch_size = int(inference_batch_size)
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")

    # Point the trained checkpoint at a different dataset (e.g. a freshly
    # generated benchmark set). Only the test split is consumed, and the
    # checkpoint's mu_global/sigma_global are reused, mirroring eval_all_seeds.
    if data_dir is not None:
        data_dir_path = Path(data_dir)
        config["data"]["trajectories.npy"] = str(data_dir_path / "trajectories.npy")
        config["data"]["x_grid_path"] = str(data_dir_path / "x_grid.npy")
        config["data"]["y_grid_path"] = str(data_dir_path / "y_grid.npy")
        config["data"]["t_grid_path"] = str(data_dir_path / "t_grid.npy")
        config["data"]["sim_params_path"] = str(data_dir_path / "sim_params.npy")

    # OOD identity sidecar lives beside the trajectories (data_dir override or the
    # checkpoint-baked dataset). Absent file -> in-distribution defaults below.
    sidecar_dir = Path(data_dir) if data_dir is not None else Path(
        config["data"]["trajectories.npy"]
    ).parent
    ood_sidecar = _load_ood_sidecar(sidecar_dir)

    test_loader, x_grid, y_grid, _num_sims = build_test_loader(
        config,
        mu_global=ckpt.get("mu_global"),
        sigma_global=ckpt.get("sigma_global"),
        eval_all_sims=eval_all_sims,
        time_norm_horizon=time_norm_horizon,
        target_times=target_times,
        protocols=protocols,
        n_snapshots_test=n_snapshots_test,
    )
    dataset = test_loader.dataset
    pair_tags = getattr(dataset, "_pair_tags", None)
    ds_t_final = float(dataset.t_final)
    ds_norm_horizon = float(getattr(dataset, "time_norm_horizon", dataset.t_final))
    problem = problem_from_config(config)
    benchmark = problem.name
    dims = problem.dims

    model_cfg = config["model"]["parameters"]
    fno = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg["width"],
        in_channels=dims.in_channels,
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_static_dim=dims.cond_static_dim,
        cond_hidden=model_cfg.get("cond_hidden", 256),
        temporal_token_dim=dims.temporal_token_dim,
        temporal_hidden=model_cfg.get("temporal_hidden", 128),
        forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
        forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        use_temporal_encoder=dims.use_temporal_encoder,
        use_forcing_time_aug=dims.use_forcing_time_aug,
        forcing_cond_mode=model_cfg.get("forcing_cond_mode", "both"),
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
        hard_right_dirichlet=model_cfg.get("hard_right_dirichlet", False),
    )
    fno.load_state_dict(ckpt["model_state"])
    fno.to(device)
    fno.eval()
    uses_forcing = bool(getattr(fno, "use_temporal_encoder", True))

    hw = float(config.get("training", {}).get("loss", {}).get("interface_half_width", 0.05))

    mask_cache: dict[float, torch.Tensor] = {}

    def _mask_for(interface_x: float) -> torch.Tensor:
        key = round(float(interface_x), 4)
        if key not in mask_cache:
            mask_cache[key] = build_interface_mask(x_grid, y_grid, key, hw).to(device)
        return mask_cache[key]

    if ckpt.get("mu_global") is None or ckpt.get("sigma_global") is None:
        raise ValueError(
            "checkpoint must contain mu_global and sigma_global for physical metrics"
        )
    sigma_global = float(ckpt["sigma_global"])
    if sigma_global < 0.0 or not math.isfinite(sigma_global):
        raise ValueError("checkpoint sigma_global must be finite and non-negative")
    physical_temperature_scale = sigma_global + T_EPS
    mu_global = float(ckpt["mu_global"])
    if not math.isfinite(mu_global):
        raise ValueError("checkpoint mu_global must be finite")
    temperature_rise_scale_K = math.sqrt(
        sigma_global ** 2 + (mu_global - TEMPERATURE_REFERENCE_K) ** 2
    )
    flank_cache: dict[float, tuple[int, int]] = {}

    def _flank_for(interface_x: float) -> tuple[int, int]:
        key = round(float(interface_x), 4)
        if key not in flank_cache:
            flank_cache[key] = interface_flanking_nodes(x_grid, key)
        return flank_cache[key]

    checkpoint_path = seed_dir / "fno2d_best.pt"
    provenance_id, provenance_path = _write_test_record_provenance(
        seed_dir=seed_dir,
        checkpoint_path=checkpoint_path,
        checkpoint=ckpt,
        config=config,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        benchmark=benchmark,
        rollout_options=rollout_options,
        output_csv_name=out_name,
    )

    def _batch_metrics(start, items, predictions):
        truths = torch.stack([item["Y"] for item in items]).to(device)
        interfaces = [
            float(dataset.sim_params[int(dataset._pairs[idx][0])].get("interface_x", 0.5))
            for idx in range(start, start + len(items))
        ]
        masks = torch.stack([_mask_for(interface_x) for interface_x in interfaces])
        mask_values = masks.unsqueeze(-1).to(predictions.dtype)

        diff_sq = (predictions - truths) ** 2
        reduce_dims = tuple(range(1, predictions.ndim))
        sse = torch.sum(diff_sq, dim=reduce_dims)
        target_sse = torch.sum(truths ** 2, dim=reduce_dims)
        rms = torch.sqrt(torch.mean(diff_sq, dim=reduce_dims))
        true_std = torch.std(truths, dim=reduce_dims, unbiased=False)

        interface_sse = torch.sum(diff_sq * mask_values, dim=reduce_dims)
        interface_target_sse = torch.sum(
            truths ** 2 * mask_values, dim=reduce_dims
        )
        interface_counts = torch.sum(mask_values, dim=reduce_dims).to(torch.int64)

        flanks = [_flank_for(interface_x) for interface_x in interfaces]
        left = torch.tensor([pair[0] for pair in flanks], device=device)
        right = torch.tensor([pair[1] for pair in flanks], device=device)
        batch = torch.arange(len(items), device=device)
        pred_jump = predictions[batch, right, :, :] - predictions[batch, left, :, :]
        true_jump = truths[batch, right, :, :] - truths[batch, left, :, :]
        jump_reduce_dims = tuple(range(1, pred_jump.ndim))
        jump_error = torch.sqrt(
            torch.mean((pred_jump - true_jump) ** 2, dim=jump_reduce_dims)
        )
        jump_true = torch.sqrt(torch.mean(true_jump ** 2, dim=jump_reduce_dims))
        pred_jump_max = torch.amax(torch.abs(pred_jump), dim=jump_reduce_dims)
        true_jump_max = torch.amax(torch.abs(true_jump), dim=jump_reduce_dims)

        sigma_sq = physical_temperature_scale * physical_temperature_scale
        values = torch.stack(
            [
                100.0 * torch.sqrt(sse / target_sse),
                100.0 * torch.sqrt(interface_sse / interface_target_sse),
                100.0 * rms / true_std.clamp_min(EPS_STD),
                rms * physical_temperature_scale,
                sse * sigma_sq,
                target_sse * sigma_sq,
                interface_sse * sigma_sq,
                interface_target_sse * sigma_sq,
                jump_error * physical_temperature_scale,
                100.0 * jump_error / jump_true.clamp_min(EPS_JUMP),
                100.0 * jump_error,
                pred_jump_max * physical_temperature_scale,
                true_jump_max * physical_temperature_scale,
            ],
            dim=1,
        ).cpu().numpy()
        interface_counts = interface_counts.cpu().numpy()
        num_error_cells = int(np.prod(truths.shape[1:]))
        return [
            {
                "rel_l2": float(row[0]),
                "iface_rel_l2": float(row[1]),
                "nrmse_pct": float(row[2]),
                "rmse_K": float(row[3]),
                "sse_K2": float(row[4]),
                "target_sse_K2": float(row[5]),
                "interface_sse_K2": float(row[6]),
                "interface_target_sse_K2": float(row[7]),
                "node_jump_rmse_K": float(row[8]),
                "node_jump_nrmse_pct": float(row[9]),
                "node_jump_gnrmse_pct": float(row[10]),
                "node_jump_abs_max_pred_K": float(row[11]),
                "node_jump_abs_max_true_K": float(row[12]),
                "num_error_cells": num_error_cells,
                "num_interface_cells": int(interface_counts[offset]),
            }
            for offset, row in enumerate(values)
        ]

    def _prediction_items():
        if rollout_is_active(rollout_options):
            for idx in range(len(dataset)):
                sim_id, s, j = map(int, dataset._pairs[idx])
                item = dataset[idx]
                prediction = predict_autoregressive(
                    model=fno,
                    dataset=dataset,
                    sim_id=sim_id,
                    s=s,
                    j=j,
                    num_substeps=rollout_options.num_substeps,
                    device=device,
                )
                yield idx, item, _batch_metrics(idx, [item], prediction)[0]
            return

        total = len(dataset)
        for start in range(0, total, inference_batch_size):
            stop = min(start + inference_batch_size, total)
            items = [dataset[idx] for idx in range(start, stop)]
            spatial = torch.stack([item["spatial"] for item in items]).to(device)
            cond = torch.stack([item["cond_static"] for item in items]).to(device)
            if uses_forcing:
                forcing = torch.stack([item["forcing_seq"] for item in items]).to(device)
                predictions = fno(spatial, cond, forcing)
            else:
                predictions = fno(spatial, cond)
            batch_metrics = _batch_metrics(start, items, predictions)
            for offset, (item, metrics) in enumerate(zip(items, batch_metrics, strict=True)):
                yield start + offset, item, metrics
            if start == 0 or stop == total or stop % 5000 < inference_batch_size:
                print(f"Scored {stop}/{total} test pairs", flush=True)

    rows = []
    with torch.no_grad():
        for idx, item, metrics in _prediction_items():
            sim_id, s, j = dataset._pairs[idx]
            sim_id, s, j = int(sim_id), int(s), int(j)

            params = dataset.sim_params[sim_id]
            interface_x = float(params.get("interface_x", 0.5))
            rel_l2 = metrics["rel_l2"]
            iface_rel_l2 = metrics["iface_rel_l2"]
            nrmse_pct = metrics["nrmse_pct"]
            rmse_K = metrics["rmse_K"]
            gnrmse_pct = 100.0 * rmse_K / temperature_rise_scale_K
            node_jump_rmse_K = metrics["node_jump_rmse_K"]
            node_jump_gnrmse_pct = metrics["node_jump_gnrmse_pct"]
            node_jump_nrmse_pct = metrics["node_jump_nrmse_pct"]
            node_jump_abs_max_pred_K = metrics["node_jump_abs_max_pred_K"]
            node_jump_abs_max_true_K = metrics["node_jump_abs_max_true_K"]

            t_s_val = float(dataset.t_grid[s])
            t_j_val = float(dataset.t_grid[j])
            A, freq = _amp_freq_from_params(params)

            # OOD identity: sidecar record joined by sim_id, in-distribution
            # defaults otherwise.
            rec = ood_sidecar.get(sim_id)
            if rec is None:
                ood_axis = _OOD_RECORD_DEFAULT["ood_axis"]
                ood_value = _OOD_RECORD_DEFAULT["ood_value"]
                ood_repeat = _OOD_RECORD_DEFAULT["ood_repeat"]
                latents_hash = _OOD_RECORD_DEFAULT["latents_hash"]
                distribution_class = _OOD_RECORD_DEFAULT["distribution_class"]
                rec_t_final = ds_t_final
                rec_norm_horizon = ds_norm_horizon
            else:
                ood_axis = rec.get("ood_axis", "")
                ood_value = rec.get("ood_value")
                ood_value = "" if ood_value is None else ood_value
                ood_repeat = rec.get("ood_repeat", "")
                latents_hash = rec.get("latents_hash", "")
                distribution_class = rec.get("distribution_class")
                rec_t_final = float(rec.get("dataset_t_final", ds_t_final))
                rec_norm_horizon = float(rec.get("time_norm_horizon", ds_norm_horizon))

            # Protocol tag (set by apply_protocol_pairs); empty without protocols.
            tag = pair_tags[idx] if pair_tags is not None else None
            if tag is None:
                protocol = ""
                src_time_req = ""
                src_time_act = ""
                lead_time_act = ""
                tgt_time_req = ""
                tgt_time_act = ""
            else:
                protocol = tag["protocol"]
                src_time_req = float(tag["source_time_requested"])
                src_time_act = float(tag["source_time_actual"])
                lead_time_act = float(tag["lead_time_actual"])
                tgt_time_req = float(tag["target_time_requested"])
                tgt_time_act = float(tag["target_time_actual"])
                # Evaluation-time axis: the sidecar leaves ood_value null per sim;
                # the realized target time is the OOD value, classified vs horizon.
                if ood_value == "" and rec is not None:
                    ood_value = tgt_time_act
                if distribution_class is None:
                    distribution_class = _classify_time_target(
                        tgt_time_act, rec_norm_horizon
                    )
            if distribution_class is None:
                distribution_class = _OOD_RECORD_DEFAULT["distribution_class"]

            rows.append({
                "provenance_id": provenance_id,
                "sim_id": sim_id,
                "s": s,
                "j": j,
                "t_s": t_s_val,
                "t_bar": t_j_val - t_s_val,
                "R_c": float(params.get("R_c", "")) if "R_c" in params else "",
                "benchmark": benchmark,
                "temporal_family": params.get("temporal_family", ""),
                "spatial_family": params.get("spatial_family", ""),
                "x_h": float(params["x_h"]) if "x_h" in params else "",
                "y_h": float(params["y_h"]) if "y_h" in params else "",
                "A": "" if A is None else A,
                "freq": "" if freq is None else freq,
                "regime": params.get("regime", ""),
                "R_c_amp": float(params["R_c_amp"]) if "R_c_amp" in params else "",
                "R_c_y0": float(params["R_c_y0"]) if "R_c_y0" in params else "",
                "R_c_sigma": float(params["R_c_sigma"]) if "R_c_sigma" in params else "",
                "R_c_A": float(params["R_c_A"]) if "R_c_A" in params else "",
                "x_I": interface_x,
                "rel_l2_pct": rel_l2,
                "iface_rel_l2_pct": iface_rel_l2,
                "nrmse_pct": nrmse_pct,
                "rmse_K": rmse_K,
                "gnrmse_pct": gnrmse_pct,
                "node_jump_rmse_K": node_jump_rmse_K,
                "node_jump_nrmse_pct": node_jump_nrmse_pct,
                "node_jump_gnrmse_pct": node_jump_gnrmse_pct,
                "node_jump_abs_max_pred_K": node_jump_abs_max_pred_K,
                "node_jump_abs_max_true_K": node_jump_abs_max_true_K,
                "ood_axis": ood_axis,
                "ood_value": ood_value,
                "ood_repeat": ood_repeat,
                "latents_hash": latents_hash,
                "distribution_class": distribution_class,
                "protocol": protocol,
                "source_time_requested": src_time_req,
                "source_time_actual": src_time_act,
                "lead_time_actual": lead_time_act,
                "target_time_requested": tgt_time_req,
                "target_time_actual": tgt_time_act,
                "dataset_t_final": rec_t_final,
                "time_norm_horizon": rec_norm_horizon,
                "sse_K2": metrics["sse_K2"],
                "num_error_cells": metrics["num_error_cells"],
                "interface_sse_K2": metrics["interface_sse_K2"],
                "num_interface_cells": metrics["num_interface_cells"],
                "target_sse_K2": metrics["target_sse_K2"],
                "interface_target_sse_K2": metrics["interface_target_sse_K2"],
            })

    rows.sort(key=lambda r: (r["sim_id"], r["s"], r["j"]))

    out_path = seed_dir / out_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} test records ({benchmark}) -> {out_path}")
    print(f"Saved test-record provenance -> {provenance_path}")
    return out_path
