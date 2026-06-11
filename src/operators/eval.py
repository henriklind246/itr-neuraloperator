import torch
from data.dataset import (
    TEMPORAL_SAMPLES,
    T_EPS,
    compute_global_stats,
    create_dataloaders,
    load_sim_data,
    load_solver_dt,
    problem_from_config,
    split_sim_ids,
)
from src.operators.fno2d import FNO2d
from src.operators.losses import (
    build_boundary_mask,
    build_interface_band,
    build_interface_mask,
    compute_interface_rel_l2,
    get_batch_interface_x,
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
import math
import json
from datetime import datetime

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


def build_test_loader(config, mu_global=None, sigma_global=None):
    import numpy as np

    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    sim_params = np.load(config["data"]["sim_params_path"], allow_pickle=True)
    solver_dt = load_solver_dt(config["data"]["t_grid_path"])

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)

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
        n_snapshots_test=config.get("training", {}).get("n_snapshots_test", 40),
        dt=solver_dt,
        num_workers=0,
        temporal_samples=config["model"]["parameters"].get("temporal_samples", TEMPORAL_SAMPLES),
        problem=problem_from_config(config),
    )

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
    if use_per_sample_interface and x_grid is not None:
        x_grid_t = torch.as_tensor(x_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        model.eval()
        rel_l2_norm = 0.0
        rel_l2_phys = 0.0
        iface_rel_l2_norm = 0.0
        iface_rel_l2_phys = 0.0
        boundary_rel_l2_norm = 0.0
        boundary_rel_l2_phys = 0.0

        for batch in test_loader:
            x_spatial = batch["spatial"].to(device)
            cond_static = batch["cond_static"].to(device)
            forcing_seq = batch["forcing_seq"].to(device) if "forcing_seq" in batch else None
            y_batch = batch["Y"].to(device)
            T_stats = batch["T_stats"].to(device)

            y_pred = model(x_spatial, cond_static, forcing_seq)

            # Normalized-space metric (same convention as train/val_rel_l2)
            batch_rel_l2_norm = (torch.mean((y_pred - y_batch) ** 2) / torch.mean(y_batch ** 2)) ** 0.5 * 100
            rel_l2_norm += batch_rel_l2_norm.item()

            # Physical-space metric (denormalized to Kelvin)
            mu_s = T_stats[:, 0]
            sigma_s = T_stats[:, 1]
            y_pred_phys = y_pred * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            y_true_phys = y_batch * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            batch_rel_l2_phys = (torch.mean((y_pred_phys - y_true_phys) ** 2) / torch.mean(y_true_phys ** 2)) ** 0.5 * 100
            rel_l2_phys += batch_rel_l2_phys.item()

            iface_x = get_batch_interface_x(
                batch, device, use_per_sample_interface=use_per_sample_interface
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
        rel_l2_norm /= n_batches
        rel_l2_phys /= n_batches
        iface_rel_l2_norm /= n_batches
        iface_rel_l2_phys /= n_batches
        boundary_rel_l2_norm /= n_batches
        boundary_rel_l2_phys /= n_batches

    return {
        "rel_l2_norm": rel_l2_norm,
        "rel_l2_phys": rel_l2_phys,
        "iface_rel_l2_norm": iface_rel_l2_norm,
        "iface_rel_l2_phys": iface_rel_l2_phys,
        "boundary_rel_l2_norm": boundary_rel_l2_norm,
        "boundary_rel_l2_phys": boundary_rel_l2_phys,
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
        rel_l2_norm = 0.0
        rel_l2_phys = 0.0
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

            batch_rel_l2_norm = (torch.mean((y_pred - y_batch) ** 2) / torch.mean(y_batch ** 2)) ** 0.5 * 100
            rel_l2_norm += batch_rel_l2_norm.item()

            mu_s = T_stats[:, 0]
            sigma_s = T_stats[:, 1]
            y_pred_phys = y_pred * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            y_true_phys = y_batch * (sigma_s[:, None, None, None] + T_EPS) + mu_s[:, None, None, None]
            batch_rel_l2_phys = (torch.mean((y_pred_phys - y_true_phys) ** 2) / torch.mean(y_true_phys ** 2)) ** 0.5 * 100
            rel_l2_phys += batch_rel_l2_phys.item()

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
        rel_l2_norm /= n_items
        rel_l2_phys /= n_items
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
            s_y_channel=dims.s_y_channel,
            padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
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
                "rollout_enabled": bool(rollout_options.enabled),
                "rollout_num_substeps": int(rollout_options.num_substeps),
                "rollout_partition": rollout_options.partition,
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

    print("\n===== Seed Report =====")
    print(f"Number of seeds: {len(results)}")
    print(f"best_val_loss            mean, std: ({val_mu}, {val_std})")
    print(f"test_rel_l2_norm         mean, std: ({norm_mu}, {norm_std})   <- comparable to val_rel_l2")
    print(f"test_rel_l2 (physical)   mean, std: ({test_mu}, {test_std})")
    print(f"test_iface_rel_l2_norm   mean, std: ({iface_norm_mu}, {iface_norm_std})")
    print(f"test_iface_rel_l2 (phys) mean, std: ({iface_mu}, {iface_std})")
    print(f"test_boundary_rel_l2_norm mean, std: ({bnd_norm_mu}, {bnd_norm_std})")
    print(f"test_boundary_rel_l2(phys) mean, std: ({bnd_mu}, {bnd_std})")

    best = min(results, key=lambda r: r["test_rel_l2_norm"])
    print(
        f"Best by lowest normalized test error: seed={best['seed']} "
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
    "sim_id", "s", "j", "t_s", "t_bar", "R_c", "benchmark",
    "temporal_family", "spatial_family",
    "x_h", "y_h", "A", "freq", "regime",
    "x_I", "rel_l2_pct", "iface_rel_l2_pct",
]


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
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def write_test_records(
    run_root,
    seed=None,
    out_name: str = "test_records.csv",
    rollout_enabled: bool | None = None,
    rollout_num_substeps: int | None = None,
    rollout_partition: str | None = None,
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
    device = resolve_device(config.get("training", {}).get("device", "auto"))
    rollout_options = rollout_options_from_config(
        config,
        enabled=rollout_enabled,
        num_substeps=rollout_num_substeps,
        partition=rollout_partition,
    )

    test_loader, x_grid, y_grid, _num_sims = build_test_loader(
        config,
        mu_global=ckpt.get("mu_global"),
        sigma_global=ckpt.get("sigma_global"),
    )
    dataset = test_loader.dataset
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
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
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

    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sim_id, s, j = dataset._pairs[idx]
            sim_id, s, j = int(sim_id), int(s), int(j)
            item = dataset[idx]

            spatial = item["spatial"].unsqueeze(0).to(device)
            cond = item["cond_static"].unsqueeze(0).to(device)
            y_true = item["Y"].unsqueeze(0).to(device)
            forcing_seq = item.get("forcing_seq")
            if rollout_is_active(rollout_options):
                y_pred = predict_autoregressive(
                    model=fno,
                    dataset=dataset,
                    sim_id=sim_id,
                    s=s,
                    j=j,
                    num_substeps=rollout_options.num_substeps,
                    device=device,
                )
            elif uses_forcing and forcing_seq is not None:
                y_pred = fno(spatial, cond, forcing_seq.unsqueeze(0).to(device))
            else:
                y_pred = fno(spatial, cond)

            rel_l2 = ((torch.mean((y_pred - y_true) ** 2) / torch.mean(y_true ** 2)) ** 0.5 * 100).item()

            params = dataset.sim_params[sim_id]
            interface_x = float(params.get("interface_x", 0.5))
            iface_rel_l2 = compute_interface_rel_l2(y_pred, y_true, _mask_for(interface_x))

            t_s_val = float(dataset.t_grid[s])
            t_j_val = float(dataset.t_grid[j])
            A, freq = _amp_freq_from_params(params)

            rows.append({
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
                "x_I": interface_x,
                "rel_l2_pct": rel_l2,
                "iface_rel_l2_pct": iface_rel_l2,
            })

    rows.sort(key=lambda r: (r["sim_id"], r["s"], r["j"]))

    out_path = seed_dir / out_name
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEST_RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} test records ({benchmark}) -> {out_path}")
    return out_path
