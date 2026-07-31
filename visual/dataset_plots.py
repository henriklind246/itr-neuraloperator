"""Dataset exploration and model-evaluation plots.

Covers raw trajectory/parameter visualisation (uniformly under registry group
``data``) plus held-out model diagnostics (predictions vs truth, error
breakdowns, interface jump analysis).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from data.dataset import (
    A_AMP_REF,
    RC_RANGE,
    TEMPORAL_SAMPLES,
    T_EPS,
    SnapshotPairDataset,
    build_cond_vector,
    build_forcing_seq,
    build_forcing_summary,
    compute_global_stats,
    load_sim_data,
    problem_from_config,
    split_sim_ids,
    _sample_a,
)
from src.physics.boundary_forcing import (
    SPATIAL_BUILDERS,
    TEMPORAL_BUILDERS,
    FORCING_BINS,
    SIN_AMP_RANGE,
    SIN_FREQ_RANGE,
    build_qL,
    integrate_temporal_bins,
)
from problems.source import (
    INTERFACE_X,
    SOURCE_BINS,
    SPATIAL_CHANNELS_BINS as SOURCE_SPATIAL_IN_CHANNELS,
)
from problems.source_itr import rc_log_norm
from problems.interfaces import INTERFACE_X_RANGE
from src.physics.internal_source import (
    PATCH_A_RANGE,
    PATCH_H,
    PATCH_W,
    RC_VOID_RANGES,
    build_patch_source,
    interface_control_volume_weights,
    integrated_excess_resistance,
    integrate_sin2_pulse,
    make_rc_void_profile,
    make_patch_indicator,
    make_sin2_pulse,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D


def _sin_amp_freq(params: dict) -> tuple[float, float]:
    """Pull (amp, freq) from a sim's temporal params; NaN if not the sin family."""
    if params.get("temporal_family") == "sin":
        tp = params["temporal_params"]
        return float(tp["A"]), float(tp["f"])
    return float("nan"), float("nan")


def _safe_param_float(params: dict, key: str) -> float:
    """Read a top-level scalar sim param as float; NaN if missing/non-numeric.

    Lets benchmark-agnostic record builders collect keys that exist only for
    some benchmarks (e.g. source ``A``/``x_h`` vs forcing ``temporal_params``).
    """
    try:
        return float(params.get(key))
    except (TypeError, ValueError):
        return float("nan")


def _model_predict_item(model, item):
    """Run a single dataset item through the model, honoring its forcing branch.

    Adds a batch dim, forwards ``forcing_seq`` only when the model uses its
    temporal encoder and the item carries one (source items have no forcing).
    """
    import torch

    spatial = item["spatial"].unsqueeze(0)
    cond = item["cond_static"].unsqueeze(0)
    fseq = item.get("forcing_seq")
    if fseq is not None:
        fseq = fseq.unsqueeze(0)
    with torch.no_grad():
        if getattr(model, "use_temporal_encoder", True) and fseq is not None:
            return model(spatial, cond, fseq)
        return model(spatial, cond)


def _forcing_label(params: dict) -> str:
    """Short human-readable label for a sim's temporal forcing."""
    fam = params.get("temporal_family", "?")
    tp = params.get("temporal_params", {})
    if fam == "sin":
        return f"rectified sin A={tp['A']:.0f} f={tp['f']:.2f}"
    if fam == "exp":
        return f"exp A={tp['A']:.0f} t0={tp['t0']:.2f} tau={tp['tau']:.3f}"
    if fam in ("pulse_train", "exp_train"):
        return f"{fam} Np={tp['Np']}"
    return fam
from src.operators.train import load_config
from src.operators.fno2d import FNO2d
from src.physics.init_conditions import (
    GRF_ELL_RANGE,
    HOT_AMP_RANGE,
    HOT_SIGMA_RANGE,
    SINU_AMP_RANGE,
    SINU_KMAX,
    UNIFORM_OFFSET_RANGE,
    build_ic,
)

from visual._common import (
    PLOT_STYLE,
    _add_interface_lines,
    _evaluate_q_left_field,
    _resolve_interface_metadata,
    _save_figure,
)


# ============================================================
# HELPERS — Index sampling and binning
# ============================================================

def _future_target_indices(start_idx: int, n_requested: int, n_total: int) -> np.ndarray:
    """Return unique, increasing future indices after start_idx."""
    if start_idx >= n_total - 1:
        raise ValueError("Source index must leave at least one future target.")

    available = np.arange(start_idx + 1, n_total, dtype=int)
    if n_requested >= len(available):
        return available

    positions = np.linspace(0, len(available) - 1, n_requested)
    indices = np.unique(np.round(positions).astype(int))
    while len(indices) < n_requested:
        indices = np.unique(np.concatenate([indices, np.arange(len(available), dtype=int)]))
        indices = indices[:n_requested]
    return available[indices]


def _build_interface_mask(x_grid: np.ndarray, interface_x: float, interface_half_width: float) -> np.ndarray:
    """Return a boolean mask for the configured interface region."""
    return np.abs(x_grid - interface_x) <= interface_half_width


def _interface_flanking_nodes_from_grid(x_grid: np.ndarray, interface_x: float) -> tuple[int, int]:
    """Return node indices immediately flanking an interface location."""
    iface_idx = int(np.argmin(np.abs(x_grid - interface_x)))
    left_node = iface_idx - 1 if x_grid[iface_idx] >= interface_x else iface_idx
    right_node = left_node + 1
    if left_node < 0 or right_node >= len(x_grid):
        raise ValueError("Interface location is outside the interior of the provided x_grid.")
    return left_node, right_node


def _build_interface_mask_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    interface_x: float,
    interface_half_width: float,
) -> np.ndarray:
    """Return a 2D boolean mask for the configured interface region."""
    mask_1d = _build_interface_mask(x_grid, interface_x, interface_half_width)
    return np.broadcast_to(mask_1d[:, None], (len(x_grid), len(y_grid))).copy()


def _relative_l2_percent(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Return relative L2 error in percent."""
    denom = max(float(np.mean(y_true ** 2)), 1e-12)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2) / denom) * 100.0)


def _interface_jump_profile(field_2d: np.ndarray, left_node: int, right_node: int) -> np.ndarray:
    """Return the discrete interface jump profile ΔT(y) from adjacent cell centers."""
    # Discrete analog of the interface jump, evaluated from the two cell centers
    # immediately adjacent to the interface face.
    return np.asarray(field_2d[right_node, :] - field_2d[left_node, :], dtype=np.float32)


def _interface_jump_map(fields: np.ndarray, left_node: int, right_node: int) -> np.ndarray:
    """Return the discrete interface jump history ΔT(y, t) from a stack of fields."""
    return np.asarray(fields[:, right_node, :] - fields[:, left_node, :], dtype=np.float32)


def _jump_rms(jump_profile: np.ndarray) -> float:
    """Return RMS jump magnitude over y."""
    return float(np.sqrt(np.mean(np.asarray(jump_profile, dtype=np.float64) ** 2)))


# ------------------------------------------------------------
# Imperfect-interface CONTACT jump (physical Kelvin)
# ------------------------------------------------------------
# The node-to-node jump above mixes three effects: the left half-cell bulk
# conduction drop, the contact discontinuity, and the right half-cell bulk drop.
# The *contact* jump isolates the discontinuity at the imperfect interface:
#
#     dT_contact(t, y) = R_c * q_I = R_c * G * (T_left - T_right),
#     G = 1 / (h_L/k_L + R_c + h_R/k_R),
#
# where G is the per-area interface conductance used by the FV solver
# (src/physics/fv_solver_2d.py:376) and (h_L, h_R) are the node->interface
# distances (h_L + h_R = hx). The FNO predicts nodal temperatures; we postprocess
# them with the known interface law to obtain the implied contact jump.

_DEFAULT_K_LEFT = 2.0
_DEFAULT_K_RIGHT = 1.0
_SOURCE_K_LEFT = 3.0
_SOURCE_K_RIGHT = 35.0


def _resolve_layer_conductivities(ds, sid: int, config: dict | None = None) -> tuple[float, float]:
    """Resolve (k_left, k_right) for the imperfect interface, sample-first.

    Order: (1) per-sample material values if a dataset ever serializes them,
    (2) config-declared layer conductivities, (3) benchmark constants.
    """
    try:
        params = ds.sim_params[int(sid)]
        kl, kr = params.get("k_left"), params.get("k_right")
        if kl is not None and kr is not None:
            return float(kl), float(kr)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    if config is not None:
        layers = config.get("physics", {}).get("layers")
        if layers and len(layers) >= 2:
            try:
                return float(layers[0]["k"]), float(layers[1]["k"])
            except (KeyError, TypeError, IndexError):
                pass
        benchmark = str(config.get("benchmark", {}).get("name", ""))
        if benchmark in ("source", "source_itr"):
            return _SOURCE_K_LEFT, _SOURCE_K_RIGHT
    try:
        problem_name = str(getattr(ds.problem, "name", ""))
        if problem_name in ("source", "source_itr"):
            return _SOURCE_K_LEFT, _SOURCE_K_RIGHT
    except AttributeError:
        pass
    return _DEFAULT_K_LEFT, _DEFAULT_K_RIGHT


def _interface_conductance_G(x_grid: np.ndarray, interface_x: float, R_c,
                             k_left: float, k_right: float):
    """Per-area interface conductance G = 1/(h_L/k_L + R_c + h_R/k_R)."""
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_x)
    h_L = float(interface_x) - float(x_grid[left_node])
    h_R = float(x_grid[right_node]) - float(interface_x)
    Rc = np.asarray(R_c, dtype=np.float64)
    G = 1.0 / (h_L / float(k_left) + Rc + h_R / float(k_right))
    return float(G) if G.ndim == 0 else G


def _interface_contact_jump_map(fields: np.ndarray, x_grid: np.ndarray, interface_x: float,
                                R_c, k_left: float, k_right: float) -> np.ndarray:
    """Return the physical contact-jump history dT_contact(t, y) in Kelvin.

    ``fields`` is a stack of shape (Nt, Nx, Ny). Sign convention: positive means a
    temperature drop left->right (T_L^face - T_R^face = R_c*q_I).
    """
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_x)
    G = _interface_conductance_G(x_grid, interface_x, R_c, k_left, k_right)
    Rc = np.asarray(R_c, dtype=np.float64)
    jump = Rc * G * (fields[:, left_node, :] - fields[:, right_node, :])
    return np.asarray(jump, dtype=np.float64)


def _contact_jump_reductions(jump_map: np.ndarray) -> dict[str, np.ndarray]:
    """Per-time y-reductions of a contact-jump map (Nt, Ny) -> dict of (Nt,) arrays.

    ``mean_abs``/``rms`` are robust magnitude headlines; ``signed_mean`` (with
    ``signed_std`` for a band) is secondary and meaningful only where the
    through-interface flux keeps one sign.
    """
    j = np.asarray(jump_map, dtype=np.float64)
    return {
        "mean_abs": np.mean(np.abs(j), axis=1),
        "rms": np.sqrt(np.mean(j ** 2, axis=1)),
        "signed_mean": np.mean(j, axis=1),
        "signed_std": np.std(j, axis=1),
    }


def _jump_error_metrics(jump_pred: np.ndarray, jump_true: np.ndarray,
                        eps: float = 1e-8) -> dict[str, float]:
    """Absolute-Kelvin contact-jump error metrics (primary) + a guarded rel-L2.

    The relative L2 is flagged ``rel_l2_pct_redundant`` because for a scalar R_c /
    fixed-interface case it equals the node-to-node jump relative error (the
    R_c*G scale factor cancels per case); use it only as a secondary annotation.
    """
    jp = np.asarray(jump_pred, dtype=np.float64)
    jt = np.asarray(jump_true, dtype=np.float64)
    diff = jp - jt
    denom = float(np.sqrt(np.sum(jt ** 2)))
    return {
        "abs_rmse_K": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
        "peak_abs_err_K": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "truth_peak_jump_K": float(np.max(np.abs(jt))) if jt.size else 0.0,
        "rel_l2_pct_redundant": float(np.sqrt(np.sum(diff ** 2)) / max(denom, eps) * 100.0),
    }


def _shared_thumbnail_grid(n_panels: int) -> tuple[int, int]:
    """Return a near-square (nrows, ncols) layout for thumbnail-style figures."""
    if n_panels <= 0:
        return 1, 1
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def _plot_field_2d(
    ax,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    field: np.ndarray,
    *,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    interface_positions: list[float] | None = None,
):
    """Render one 2D scalar field with the project plotting conventions."""
    pcm = ax.pcolormesh(x_grid, y_grid, np.asarray(field).T, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax, norm=norm)
    if interface_positions:
        _add_interface_lines(ax, interface_positions)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return pcm


def _compute_binned_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin x and return centers, median, q25, q75, counts for non-empty bins.

    Drops (x, y) pairs where either is NaN — needed because mixed-family
    datasets have NaN amplitude/frequency for non-sin sims.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    x_min = float(x.min())
    x_max = float(x.max())
    if np.isclose(x_min, x_max):
        q25, median, q75 = np.percentile(y, [25, 50, 75])
        return (
            np.array([x_min]),
            np.array([median]),
            np.array([q25]),
            np.array([q75]),
            np.array([len(x)]),
        )

    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_ids = np.clip(np.digitize(x, bins[1:-1], right=False), 0, n_bins - 1)

    centers = []
    medians = []
    q25s = []
    q75s = []
    counts = []
    for idx in range(n_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        y_bin = y[mask]
        centers.append(0.5 * (bins[idx] + bins[idx + 1]))
        q25, median, q75 = np.percentile(y_bin, [25, 50, 75])
        q25s.append(float(q25))
        medians.append(float(median))
        q75s.append(float(q75))
        counts.append(int(mask.sum()))

    return (
        np.array(centers),
        np.array(medians),
        np.array(q25s),
        np.array(q75s),
        np.array(counts),
    )


def _parameter_arrays(sim_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return amplitude, frequency, and contact resistance arrays.

    Amplitude / frequency are sin-specific; non-sin sims contribute NaN.
    """
    pairs = [_sin_amp_freq(p) for p in sim_params]
    amplitudes = np.array([a for a, _ in pairs], dtype=np.float32)
    frequencies = np.array([f for _, f in pairs], dtype=np.float32)
    contact_resistance = np.array([float(p["R_c"]) for p in sim_params], dtype=np.float32)
    return amplitudes, frequencies, contact_resistance


def _select_representative_sim_id(sim_params: np.ndarray) -> int:
    """Choose a deterministic representative simulation near the median condition point.

    Restricted to sin-family sims because the median is computed over (A, f, R_c).
    Falls back to the first sim if no sin sims exist.
    """
    sin_idx = np.array(
        [i for i, p in enumerate(sim_params) if p.get("temporal_family") == "sin"],
        dtype=np.int64,
    )
    if sin_idx.size == 0:
        return 0
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)
    A = amplitudes[sin_idx]
    F = frequencies[sin_idx]
    R = contact_resistance[sin_idx]
    stacked = np.column_stack(
        [
            (A - A.min()) / max(A.max() - A.min(), 1e-12),
            (np.log10(F) - np.log10(F).min())
            / max(np.log10(F).max() - np.log10(F).min(), 1e-12),
            (R - R.min()) / max(R.max() - R.min(), 1e-12),
        ]
    )
    target = np.median(stacked, axis=0)
    return int(sin_idx[int(np.argmin(np.sum((stacked - target) ** 2, axis=1)))])


def _spatial_localness(p: dict) -> float:
    """Smaller is more localized along y. Uniform returns +inf so it sorts last."""
    sf = p["spatial_family"]
    sp = p.get("spatial_params", {})
    if sf == "gaussian":
        return float(sp["sigma_y"])
    if sf == "triangle":
        return float(sp["ell"])
    if sf == "patch":
        return 0.5 * float(sp["w"])
    return float("inf")


def _select_localized_sim_id(sim_params: np.ndarray, family: str | None = None) -> int:
    """Return the sim index with the most localized spatial profile.

    If ``family`` is given, restrict the search to that family. Falls back to
    the first sim when no candidate matches (e.g. uniform-only).
    """
    candidates = [
        i for i, p in enumerate(sim_params)
        if (family is None and p["spatial_family"] != "uniform") or p["spatial_family"] == family
    ]
    if not candidates:
        return 0
    return min(candidates, key=lambda i: _spatial_localness(sim_params[i]))


def _format_spatial_family(p: dict) -> str:
    """Compact one-line description of a sim's spatial profile."""
    sf = p["spatial_family"]
    sp = p.get("spatial_params", {})
    if sf == "uniform":
        return "uniform"
    if sf == "patch":
        return f"patch(y_c={sp['y_c']:.2f}, w={sp['w']:.2f})"
    if sf == "gaussian":
        return f"gaussian(y_c={sp['y_c']:.2f}, σ={sp['sigma_y']:.3f})"
    if sf == "triangle":
        return f"triangle(y_c={sp['y_c']:.2f}, ell={sp['ell']:.2f})"
    return sf


def _snap_indices_in_window(t_grid: np.ndarray, t_lo: float, t_hi: float, n: int) -> np.ndarray:
    """Return ``n`` evenly spaced snapshot indices inside [t_lo, t_hi]."""
    in_window = np.where((t_grid >= t_lo) & (t_grid <= t_hi))[0]
    if len(in_window) == 0:
        return np.linspace(0, len(t_grid) - 1, n, dtype=int)
    if len(in_window) <= n:
        return in_window
    pick = np.linspace(0, len(in_window) - 1, n).round().astype(int)
    return in_window[pick]


# ============================================================
# HELPERS — Data loading and dataset splits
# ============================================================

def _load_plot_data(
    data_path: str | Path,
    x_grid_path: str | Path,
    y_grid_path: str | Path,
    t_grid_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectories and saved grids for plot generation."""
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=str(Path(data_path)),
        x_grid_path=str(Path(x_grid_path)),
        y_grid_path=str(Path(y_grid_path)),
        t_grid_path=str(Path(t_grid_path)),
    )
    return trajectories, x_grid, y_grid, t_grid


def _resolve_plot_config(checkpoint_conf: dict | None = None) -> dict:
    """Prefer checkpoint config when available, otherwise load the default project config."""
    return checkpoint_conf if checkpoint_conf is not None else load_config()


def _load_checkpoint_model(checkpoint_path: str | Path) -> tuple[FNO2d, dict]:
    """Load a checkpoint and reconstruct a 2D FNO model on CPU."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    conf = ckpt["conf"]
    model_cfg = conf.get("model", {}).get("parameters", {})
    if "modes1" not in model_cfg or "modes2" not in model_cfg:
        raise ValueError("Checkpoint is not a 2D FNO checkpoint: missing modes1/modes2")

    dims = problem_from_config(conf).dims
    model = FNO2d(
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        width=model_cfg.get("width", 64),
        in_channels=dims.in_channels,
        out_channels=model_cfg.get("out_channels", 1),
        n_layers=model_cfg.get("n_layers", 4),
        cond_static_dim=dims.cond_static_dim,
        cond_hidden=model_cfg.get("cond_hidden", 256),
        temporal_token_dim=dims.temporal_token_dim,
        temporal_hidden=model_cfg.get("temporal_hidden", 128),
        forcing_embed_dim=model_cfg.get("forcing_embed_dim", 64),
        forcing_spatial_dim=model_cfg.get("forcing_spatial_dim", 16),
        dropout=model_cfg.get("dropout", 0.0),
        spectral_dropout=model_cfg.get("spectral_dropout", 0.0),
        use_temporal_encoder=dims.use_temporal_encoder,
        use_forcing_time_aug=dims.use_forcing_time_aug,
        s_y_channel=dims.s_y_channel,
        padding_reference_resolution=model_cfg.get("padding_reference_resolution"),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        cin_exclude_padding=model_cfg.get("cin_exclude_padding", False),
        hard_right_dirichlet=model_cfg.get("hard_right_dirichlet", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._mu_global = ckpt.get("mu_global")
    model._sigma_global = ckpt.get("sigma_global")
    return model, conf


def _build_split_datasets(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    dt: float | None = None,
) -> dict[str, SnapshotPairDataset]:
    """Build train/val/test snapshot-pair datasets that mirror training-time sampling.

    `dt` is the solver dt (saved alongside trajectories). Required for cond-vec
    consistency when save_stride > 1; falls back to t_grid spacing if None.
    """
    num_sims = trajectories.shape[0]
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    train_ids, val_ids, test_ids = split_sim_ids(
        num_sims=num_sims,
        train_frac=data_cfg.get("train_split", 0.7),
        val_frac=data_cfg.get("val_split", 0.15),
        seed=0,
    )
    n_snapshots = training_cfg.get("n_snapshots", 15)
    n_snapshots_test = training_cfg.get("n_snapshots_test", None)
    test_snapshots = n_snapshots_test if n_snapshots_test is not None else n_snapshots
    mu_global, sigma_global = compute_global_stats(trajectories, train_ids)
    problem = problem_from_config(config)
    return {
        "train": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=train_ids,
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=n_snapshots,
            dt=dt,
            problem=problem,
        ),
        "val": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=val_ids,
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=n_snapshots,
            dt=dt,
            problem=problem,
        ),
        "test": SnapshotPairDataset(
            trajectories=trajectories,
            t_grid=t_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            sim_ids=test_ids,
            sim_params=sim_params,
            mu_global=mu_global,
            sigma_global=sigma_global,
            n_snapshots=test_snapshots,
            dt=dt,
            problem=problem,
        ),
    }


def _prepare_prediction_case(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int,
    s: int,
    target_indices: np.ndarray,
    config: dict | None = None,
    dt: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Prepare truth/prediction arrays for one source snapshot and multiple target times."""
    import torch

    config = _resolve_plot_config(config)
    problem = problem_from_config(config)
    interface_meta = _resolve_interface_metadata(
        config=config, sim_params=sim_params, sim_id=sim_id,
    )
    t_targets = t_grid[target_indices]

    # Use global normalization stats from model (attached during checkpoint loading)
    mu_global = getattr(model, "_mu_global", None)
    sigma_global = getattr(model, "_sigma_global", None)
    if mu_global is None or sigma_global is None:
        # Fallback: compute from training split
        from data.dataset import split_sim_ids as _split
        train_ids, _, _ = _split(trajectories.shape[0], 0.7, 0.15, seed=0)
        mu_global, sigma_global = compute_global_stats(trajectories, train_ids)

    plot_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        t_grid=t_grid,
        x_grid=x_grid,
        y_grid=y_grid,
        sim_ids=np.array([sim_id], dtype=np.int64),
        sim_params=sim_params,
        mu_global=float(mu_global),
        sigma_global=float(sigma_global),
        n_snapshots=None,
        noise_std=0.0,
        dt=dt,
        temporal_samples=config.get("model", {}).get("parameters", {}).get(
            "temporal_samples", TEMPORAL_SAMPLES,
        ),
        problem=problem,
    )
    items = [
        problem.build_item(plot_dataset, int(sim_id), int(s), int(target_idx))
        for target_idx in target_indices
    ]
    spatial = torch.from_numpy(np.stack([item["spatial"] for item in items], axis=0))
    cond_static = torch.from_numpy(np.stack([item["cond_static"] for item in items], axis=0))
    forcing_seq = None
    if all("forcing_seq" in item for item in items):
        forcing_seq = torch.from_numpy(np.stack([item["forcing_seq"] for item in items], axis=0))

    device = next(model.parameters()).device
    use_temporal = bool(getattr(model, "use_temporal_encoder", True))
    with torch.no_grad():
        if use_temporal and forcing_seq is not None:
            Y_pred_norm = model(
                spatial.to(device),
                cond_static.to(device),
                forcing_seq.to(device),
            ).cpu().numpy()[..., 0]
        else:
            Y_pred_norm = model(
                spatial.to(device),
                cond_static.to(device),
            ).cpu().numpy()[..., 0]

    Y_pred_rows = []
    Y_true_rows = []
    for row, item in enumerate(items):
        stats = np.asarray(item["T_stats"], dtype=np.float32)
        mu_s = float(stats[0])
        sigma_s = float(stats[1])
        Y_pred_rows.append((Y_pred_norm[row] * (sigma_s + T_EPS) + mu_s).astype(np.float32))
        Y_true_rows.append((item["Y"][..., 0] * (sigma_s + T_EPS) + mu_s).astype(np.float32))
        if stats.shape[0] > 2:
            interface_meta = {
                **interface_meta,
                "interface_x": float(stats[2]),
                "positions": [float(stats[2])],
            }

    params = sim_params[int(sim_id)]
    amp, freq = _sin_amp_freq(params)
    return {
        "interface_x": interface_meta["interface_x"],
        "interface_positions": interface_meta["positions"],
        "interface_half_width": interface_meta["interface_half_width"],
        "t_targets": t_targets,
        "t_bars": t_grid[target_indices] - t_grid[s],
        "source_time": float(t_grid[s]),
        "Y_pred": np.stack(Y_pred_rows, axis=0),
        "Y_true": np.stack(Y_true_rows, axis=0),
        "amp": amp,
        "freq": freq,
        "R_c": float(params["R_c"]),
    }


def _compute_pair_error_records(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict | None = None,
    max_samples: int = 256,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Evaluate sampled snapshot pairs and return per-pair error records."""
    import torch

    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot compute error records.")

    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, len(dataset))
    sample_indices = np.arange(len(dataset))
    if sample_size < len(dataset):
        sample_indices = np.sort(rng.choice(sample_indices, size=sample_size, replace=False))

    x_batch = []
    cond_batch = []
    forcing_batch = []
    y_batch = []
    stats_batch = []
    has_forcing = True
    for idx in sample_indices:
        item = dataset[idx]
        x_batch.append(item["spatial"])
        cond_batch.append(item["cond_static"])
        fseq = item.get("forcing_seq")
        if fseq is None:
            has_forcing = False
        else:
            forcing_batch.append(fseq)
        y_batch.append(item["Y"])
        stats_batch.append(item["T_stats"])

    x_tensor = torch.stack(x_batch, dim=0)
    cond_tensor = torch.stack(cond_batch, dim=0)
    forcing_tensor = torch.stack(forcing_batch, dim=0) if has_forcing else None
    y_tensor = torch.stack(y_batch, dim=0)
    stats_tensor = torch.stack(stats_batch, dim=0)

    interface_meta = _resolve_interface_metadata(config=_resolve_plot_config(config))
    device = next(model.parameters()).device
    use_temporal = bool(getattr(model, "use_temporal_encoder", True))
    model.eval()

    preds = []
    for start in range(0, sample_size, batch_size):
        stop = min(start + batch_size, sample_size)
        with torch.no_grad():
            if use_temporal and forcing_tensor is not None:
                pred = model(
                    x_tensor[start:stop].to(device),
                    cond_tensor[start:stop].to(device),
                    forcing_tensor[start:stop].to(device),
                ).cpu()
            else:
                pred = model(
                    x_tensor[start:stop].to(device),
                    cond_tensor[start:stop].to(device),
                ).cpu()
        preds.append(pred)
    y_pred = torch.cat(preds, dim=0)

    mu_s = stats_tensor[:, 0][:, None, None, None]
    sigma_s = stats_tensor[:, 1][:, None, None, None]
    y_pred_phys = y_pred * (sigma_s + T_EPS) + mu_s
    y_true_phys = y_tensor * (sigma_s + T_EPS) + mu_s

    global_rel = (
        torch.mean((y_pred_phys - y_true_phys) ** 2, dim=(1, 2, 3))
        / torch.clamp(torch.mean(y_true_phys ** 2, dim=(1, 2, 3)), min=1e-12)
    ).sqrt() * 100.0
    y_pred_phys_field = y_pred_phys[..., 0]
    y_true_phys_field = y_true_phys[..., 0]
    if stats_tensor.shape[1] > 2:
        interface_x_values = stats_tensor[:, 2].numpy().astype(np.float32)
        iface_rel_rows = []
        jump_rel_rows = []
        true_jump_rms_rows = []
        pred_jump_rms_rows = []
        for row, interface_x in enumerate(interface_x_values):
            interface_metric_mask = torch.from_numpy(
                _build_interface_mask_2d(
                    x_grid, y_grid, float(interface_x),
                    interface_meta["interface_half_width"],
                )
            )
            left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, float(interface_x))
            pred_iface = y_pred_phys_field[row, interface_metric_mask]
            true_iface = y_true_phys_field[row, interface_metric_mask]
            iface_rel_rows.append(
                (
                    torch.mean((pred_iface - true_iface) ** 2)
                    / torch.clamp(torch.mean(true_iface ** 2), min=1e-12)
                ).sqrt() * 100.0
            )
            pred_jump_row = y_pred_phys_field[row, right_node, :] - y_pred_phys_field[row, left_node, :]
            true_jump_row = y_true_phys_field[row, right_node, :] - y_true_phys_field[row, left_node, :]
            jump_rel_rows.append(
                (
                    torch.mean((pred_jump_row - true_jump_row) ** 2)
                    / torch.clamp(torch.mean(true_jump_row ** 2), min=1e-12)
                ).sqrt() * 100.0
            )
            true_jump_rms_rows.append(torch.sqrt(torch.mean(true_jump_row ** 2)))
            pred_jump_rms_rows.append(torch.sqrt(torch.mean(pred_jump_row ** 2)))

        iface_rel = torch.stack(iface_rel_rows)
        jump_rel = torch.stack(jump_rel_rows)
        true_jump_rms = torch.stack(true_jump_rms_rows)
        pred_jump_rms = torch.stack(pred_jump_rms_rows)
    else:
        interface_x_values = np.full(sample_size, float(interface_meta["interface_x"]), dtype=np.float32)
        interface_metric_mask = torch.from_numpy(
            _build_interface_mask_2d(
                x_grid, y_grid,
                interface_meta["interface_x"],
                interface_meta["interface_half_width"],
            )
        )
        left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_meta["interface_x"])
        iface_rel = (
            torch.mean((y_pred_phys_field[:, interface_metric_mask] - y_true_phys_field[:, interface_metric_mask]) ** 2, dim=1)
            / torch.clamp(torch.mean(y_true_phys_field[:, interface_metric_mask] ** 2, dim=1), min=1e-12)
        ).sqrt() * 100.0
        pred_jump = y_pred_phys_field[:, right_node, :] - y_pred_phys_field[:, left_node, :]
        true_jump = y_true_phys_field[:, right_node, :] - y_true_phys_field[:, left_node, :]
        jump_rel = (
            torch.mean((pred_jump - true_jump) ** 2, dim=1)
            / torch.clamp(torch.mean(true_jump ** 2, dim=1), min=1e-12)
        ).sqrt() * 100.0
        true_jump_rms = torch.sqrt(torch.mean(true_jump ** 2, dim=1))
        pred_jump_rms = torch.sqrt(torch.mean(pred_jump ** 2, dim=1))

    sim_ids = np.array([dataset._pairs[idx][0] for idx in sample_indices], dtype=np.int64)
    params = dataset.sim_params[sim_ids]
    source_indices = np.array([dataset._pairs[idx][1] for idx in sample_indices], dtype=np.int64)
    target_indices = np.array([dataset._pairs[idx][2] for idx in sample_indices], dtype=np.int64)

    return {
        "sim_id": sim_ids,
        "lead_time": np.array([dataset._lead_times[idx] for idx in sample_indices], dtype=np.float32),
        "global_rel_l2": global_rel.numpy(),
        "iface_rel_l2": iface_rel.numpy(),
        "jump_rel_l2": jump_rel.numpy(),
        "amplitude": np.array([_sin_amp_freq(p)[0] for p in params], dtype=np.float32),
        "frequency": np.array([_sin_amp_freq(p)[1] for p in params], dtype=np.float32),
        "R_c": np.array([float(p["R_c"]) for p in params], dtype=np.float32),
        # source-benchmark fields (NaN/empty for forcing/interface sims)
        "A": np.array([_safe_param_float(p, "A") for p in params], dtype=np.float32),
        "x_h": np.array([_safe_param_float(p, "x_h") for p in params], dtype=np.float32),
        "y_h": np.array([_safe_param_float(p, "y_h") for p in params], dtype=np.float32),
        "interface_x": interface_x_values,
        "regime": np.array([str(p.get("regime", "")) for p in params], dtype=object),
        "true_jump_rms": true_jump_rms.numpy(),
        "pred_jump_rms": pred_jump_rms.numpy(),
        "source_index": source_indices,
        "target_index": target_indices,
    }


def _compute_binned_means(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin x and return (centers, mean_y, counts) for non-empty bins."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    x_min = float(x.min())
    x_max = float(x.max())
    if np.isclose(x_min, x_max):
        return np.array([x_min]), np.array([float(y.mean())]), np.array([len(x)])

    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_ids = np.clip(np.digitize(x, bins[1:-1], right=False), 0, n_bins - 1)

    centers = []
    means = []
    counts = []
    for idx in range(n_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        centers.append(0.5 * (bins[idx] + bins[idx + 1]))
        means.append(float(y[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(means), np.array(counts)


def _lead_time_coverage_counts(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
) -> dict[str, np.ndarray]:
    """Return lead-time arrays and pair counts for split-aware coverage plots/tests."""
    datasets = _build_split_datasets(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    return {
        "train": datasets["train"]._lead_times.copy(),
        "val": datasets["val"]._lead_times.copy(),
        "test": datasets["test"]._lead_times.copy(),
        "train_count": np.array([len(datasets["train"])]),
        "val_count": np.array([len(datasets["val"])]),
        "test_count": np.array([len(datasets["test"])]),
    }


# ============================================================
# PLOTS — Model evaluation
# ============================================================

def plot_prediction_vs_truth(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int,
    s: int,
    n_steps: int = 40,
    config: dict | None = None,
    save_path: str | Path | None = None,
    dt: float | None = None,
):
    """Plot full 2D truth, prediction, residual, and jump profile diagnostics."""
    config = _resolve_plot_config(config)
    candidate_indices = _future_target_indices(s, n_steps, len(t_grid))
    if len(candidate_indices) > 3:
        pick = np.unique(np.round(np.linspace(0, len(candidate_indices) - 1, 3)).astype(int))
        target_indices = candidate_indices[pick]
    else:
        target_indices = candidate_indices
    case = _prepare_prediction_case(
        model,
        trajectories,
        x_grid,
        y_grid,
        t_grid,
        sim_params,
        sim_id,
        s,
        target_indices,
        config=config,
        dt=dt,
    )
    t_targets = case["t_targets"]
    Y_pred = case["Y_pred"]
    Y_true = case["Y_true"]
    vmin = min(Y_true.min(), Y_pred.min())
    vmax = max(Y_true.max(), Y_pred.max())
    interface_metric_mask = _build_interface_mask_2d(x_grid, y_grid, case["interface_x"], case["interface_half_width"])
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, case["interface_x"])
    residual = Y_pred - Y_true
    true_jump = _interface_jump_map(Y_true, left_node, right_node)
    pred_jump = _interface_jump_map(Y_pred, left_node, right_node)
    residual_limit = max(float(np.max(np.abs(residual))), 1e-8)
    residual_norm = TwoSlopeNorm(vcenter=0.0, vmin=-residual_limit, vmax=residual_limit)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(target_indices), 4, figsize=(18, 5.0 * len(target_indices)), squeeze=False, constrained_layout=True)
        truth_pred_axes = []
        resid_axes = []
        pcm_truth = None
        pcm_resid = None
        for row, t_target in enumerate(t_targets):
            true_field = Y_true[row]
            pred_field = Y_pred[row]
            resid_field = residual[row]
            true_jump_row = true_jump[row]
            pred_jump_row = pred_jump[row]
            global_rel = _relative_l2_percent(pred_field, true_field)
            iface_rel = _relative_l2_percent(pred_field[interface_metric_mask], true_field[interface_metric_mask])
            jump_rel = _relative_l2_percent(pred_jump_row, true_jump_row)
            jump_rmse = float(np.sqrt(np.mean((pred_jump_row - true_jump_row) ** 2)))

            ax_truth, ax_pred, ax_resid, ax_jump = axes[row]
            pcm_truth = _plot_field_2d(
                ax_truth,
                x_grid,
                y_grid,
                true_field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=case["interface_positions"],
            )
            pcm_truth = _plot_field_2d(
                ax_pred,
                x_grid,
                y_grid,
                pred_field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=case["interface_positions"],
            )
            pcm_resid = _plot_field_2d(
                ax_resid,
                x_grid,
                y_grid,
                resid_field,
                cmap="coolwarm",
                norm=residual_norm,
                interface_positions=case["interface_positions"],
            )
            ax_truth.set_title(f"Truth — t={t_target:.3f}")
            ax_pred.set_title("Prediction")
            ax_resid.set_title("Signed Residual")
            truth_pred_axes.extend([ax_truth, ax_pred])
            resid_axes.append(ax_resid)

            ax_resid.text(
                0.03,
                0.03,
                f"Global rel. L2: {global_rel:.2f}%\nIface rel. L2: {iface_rel:.2f}%\nJump rel. err.: {jump_rel:.2f}%",
                transform=ax_resid.transAxes,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )

            ax_jump.plot(y_grid, true_jump_row, color="black", label="Truth")
            ax_jump.plot(y_grid, pred_jump_row, color="C3", linestyle="--", label="Prediction")
            ax_jump.axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
            ax_jump.set_xlabel("y")
            ax_jump.set_ylabel("Node-to-node ΔT")
            ax_jump.set_title(f"Interface Jump Profile — jump rel. err. {jump_rel:.2f}%")
            ax_jump.text(
                0.03,
                0.04,
                f"Field rel. L2: {global_rel:.2f}%\nJump RMSE: {jump_rmse:.3f} K",
                transform=ax_jump.transAxes,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )
            ax_jump.grid(True)
            if row == 0:
                ax_jump.legend(loc="best")

        fig.colorbar(pcm_truth, ax=truth_pred_axes, label="Temperature", shrink=0.92)
        fig.colorbar(pcm_resid, ax=resid_axes, label="Pred - Truth", shrink=0.92)

        fig.suptitle(
            f"Prediction vs Truth — Sim {sim_id}, t_s={case['source_time']:.3f}, "
            f"A={case['amp']:.0f}, f={case['freq']:.2f}, R_c={case['R_c']:.2f}"
        )
        _save_figure(fig, save_path, "data", "prediction_vs_truth", layout="constrained")


def plot_trajectory_heatmap(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray | None = None,
    t_window: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
):
    """Plot six full-field temperature snapshots for one simulation.

    When ``t_window`` is given, five snapshots are drawn from that window plus
    one trailing post-window frame (useful for visualizing forcing during the
    flux-active window).
    """
    Nt = trajectories.shape[1]
    if t_window is not None:
        snap_main = _snap_indices_in_window(t_grid, t_window[0], t_window[1], 5)
        post = np.array([Nt - 1], dtype=int)
        snap_indices = np.unique(np.concatenate([snap_main, post]))
    else:
        snap_indices = np.linspace(0, Nt - 1, 6, dtype=int)
    snapshots = trajectories[sim_id, snap_indices]
    interface_meta = _resolve_interface_metadata()
    vmin = float(np.min(snapshots))
    vmax = float(np.max(snapshots))

    family_str = ""
    if sim_params is not None:
        family_str = f" — {_format_spatial_family(sim_params[sim_id])}"

    with plt.rc_context(PLOT_STYLE):
        nrows = 2
        ncols = int(np.ceil(len(snap_indices) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        pcm = None
        for ax, t_idx, field in zip(axes_flat, snap_indices, snapshots):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"t = {t_grid[t_idx]:.3f}")
        for ax in axes_flat[len(snap_indices):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes_flat.tolist(), label="Temperature", shrink=0.9)
        fig.suptitle(f"Temperature Snapshots — Simulation {sim_id}{family_str}")
        _save_figure(fig, save_path, "data", "trajectory_heatmap", layout="constrained")


def plot_trajectory_deviation_heatmap(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray | None = None,
    t_window: tuple[float, float] | None = None,
    T_right: float = 300.0,
    save_path: str | Path | None = None,
):
    """Plot the *deviation* T(x, y, t) - T_right for one simulation.

    Subtracting the right Dirichlet exposes the flux-induced 2D pattern, which
    is otherwise crushed by the dominant uniform background. Uses a diverging
    colormap so heating (>0) and cooling (<0) regions are visually distinct.
    """
    Nt = trajectories.shape[1]
    if t_window is not None:
        snap_main = _snap_indices_in_window(t_grid, t_window[0], t_window[1], 5)
        post = np.array([Nt - 1], dtype=int)
        snap_indices = np.unique(np.concatenate([snap_main, post]))
    else:
        snap_indices = np.linspace(0, Nt - 1, 6, dtype=int)
    snapshots = trajectories[sim_id, snap_indices].astype(np.float32) - float(T_right)
    interface_meta = _resolve_interface_metadata()

    # Symmetric range so 0 (no deviation) sits at the colormap midpoint.
    vmax = float(np.max(np.abs(snapshots)))
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    family_str = ""
    if sim_params is not None:
        family_str = f" — {_format_spatial_family(sim_params[sim_id])}"

    with plt.rc_context(PLOT_STYLE):
        nrows = 2
        ncols = int(np.ceil(len(snap_indices) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows),
                                 constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        pcm = None
        for ax, t_idx, field in zip(axes_flat, snap_indices, snapshots):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"t = {t_grid[t_idx]:.3f}")
        for ax in axes_flat[len(snap_indices):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes_flat.tolist(),
                     label=f"T - T_right  (T_right = {T_right:.0f} K)", shrink=0.9)
        fig.suptitle(f"Deviation from Right Dirichlet — Simulation {sim_id}{family_str}")
        _save_figure(fig, save_path, "data", "trajectory_deviation_heatmap",
                     layout="constrained")


def plot_spatial_family_breakdown(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    t_window: tuple[float, float] = (0.0, 0.20),
    n_snaps: int = 4,
    save_path: str | Path | None = None,
):
    """Tile y-perturbation snapshots for one representative sim per spatial family.

    Rows: spatial families (uniform, patch, gaussian, triangle).
    Cols: ``n_snaps`` snapshots in the flux-active window.
    Each panel shows T(x, y, t) - <T>_y(x, t) so the y-structure is visible.
    Each row uses its own symmetric colorbar.
    """
    families = ["uniform", "patch", "gaussian", "triangle"]
    snap_indices = _snap_indices_in_window(t_grid, t_window[0], t_window[1], n_snaps)

    # Floor on y-perturbation magnitude; below this the autoscaled colormap
    # would saturate float64 round-off from the per-column mean subtraction.
    signal_floor = 1e-3

    selected_ids = []
    for fam in families:
        candidates = [
            i for i, p in enumerate(sim_params) if p["spatial_family"] == fam
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda i: _spatial_localness(sim_params[i]))
        chosen = candidates[0]
        for sid in candidates:
            window_fields = trajectories[sid, snap_indices].astype(np.float32)
            window_pert = window_fields - window_fields.mean(axis=2, keepdims=True)
            if float(np.max(np.abs(window_pert))) >= signal_floor:
                chosen = sid
                break
        selected_ids.append((fam, chosen))

    if not selected_ids:
        raise ValueError("No simulations available for any spatial family.")
    interface_meta = _resolve_interface_metadata()
    nrows = len(selected_ids)
    ncols = len(snap_indices)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.8 * ncols + 1.5, 3.6 * nrows),
            constrained_layout=True, squeeze=False,
        )

        for r, (fam, sid) in enumerate(selected_ids):
            fields = trajectories[sid, snap_indices].astype(np.float32)
            perturbations = fields - fields.mean(axis=2, keepdims=True)
            abs_max = max(float(np.max(np.abs(perturbations))), 1e-12)
            row_pcm = None
            for c, (t_idx, field) in enumerate(zip(snap_indices, perturbations)):
                ax = axes[r, c]
                row_pcm = _plot_field_2d(
                    ax,
                    x_grid,
                    y_grid,
                    field,
                    cmap="coolwarm",
                    vmin=-abs_max,
                    vmax=abs_max,
                    interface_positions=interface_meta["positions"],
                )
                if r == 0:
                    ax.set_title(f"t = {t_grid[t_idx]:.3f}")
                if c != 0:
                    ax.set_ylabel("")
            row_label = f"{fam}\nsim {sid}\n{_format_spatial_family(sim_params[sid])}"
            axes[r, 0].annotate(
                row_label,
                xy=(-0.30, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=10,
            )
            fig.colorbar(row_pcm, ax=axes[r, :].tolist(), label=r"$T - \langle T \rangle_y$", shrink=0.85)

        fig.suptitle("Spatial-Family Breakdown — y-perturbation per family", fontsize=14)
        _save_figure(fig, save_path, "data", "spatial_family_breakdown", layout="constrained")


def plot_initial_conditions(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_samples: int = 20,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot sampled 2D initial-condition fields."""
    rng = np.random.default_rng(seed)
    num_sims = trajectories.shape[0]
    selected = rng.choice(num_sims, size=min(n_samples, num_sims), replace=False)
    nrows, ncols = _shared_thumbnail_grid(len(selected))
    fields = trajectories[selected, 0]
    interface_meta = _resolve_interface_metadata()
    vmin = float(np.min(fields))
    vmax = float(np.max(fields))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False, constrained_layout=True)
        axes_flat = axes.ravel()
        pcm = None
        for ax, sim_id, field in zip(axes_flat, sorted(selected), trajectories[sorted(selected), 0]):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                vmin=vmin,
                vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"Sim {sim_id}")

        for ax in axes_flat[len(selected):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.9)
        fig.suptitle(f"Initial Conditions T(x, y, t=0) — {len(selected)} simulations")
        _save_figure(fig, save_path, "data", "initial_conditions", layout="constrained")


# ============================================================
# IC FAMILY PROGRESSION PLOTS
# ============================================================
# One figure per IC family with three panels showing the IC field at
# low / median / high values of one driving parameter. List-valued params
# (sinusoid wavenumbers, hot-spot positions) are drawn once with a fixed
# seed so the panels differ only along the swept axis.

_IC_PROG_NX = 100
_IC_PROG_NY = 100
_IC_PROG_T_RIGHT = 300.0
_IC_PROG_FIXED_SEED = 0


def _ic_progression_grid() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, _IC_PROG_NX)
    y = np.linspace(0.0, 1.0, _IC_PROG_NY)
    return np.meshgrid(x, y, indexing="ij")


def _render_ic_progression(
    family: str,
    panels: list[tuple[str, dict]],
    name: str,
    suptitle: str,
    save_path: str | Path | None,
) -> None:
    """Render a 1x3 progression of IC fields for one family.

    panels: list of (panel_title, ic_params_kwargs).
    """
    X, Y = _ic_progression_grid()
    interface_meta = _resolve_interface_metadata()

    fields = [
        build_ic(family, params, X, Y, T_right=_IC_PROG_T_RIGHT, b=1.0)
        for _, params in panels
    ]
    deviations = [field.astype(np.float64) - _IC_PROG_T_RIGHT for field in fields]
    abs_max = max(float(np.max(np.abs(d))) for d in deviations)
    abs_max = max(abs_max, 1e-12)
    vmin = _IC_PROG_T_RIGHT - abs_max
    vmax = _IC_PROG_T_RIGHT + abs_max

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(13, 4.4), constrained_layout=True, squeeze=False,
        )
        pcm = None
        for ax, (title, _), field in zip(axes[0], panels, fields):
            pcm = _plot_field_2d(
                ax, X[:, 0], Y[0, :], field,
                cmap="coolwarm", vmin=vmin, vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(title)
        for ax in axes[0, 1:]:
            ax.set_ylabel("")
        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Temperature", shrink=0.85)
        fig.suptitle(suptitle)
        _save_figure(fig, save_path, "data", name, layout="constrained")


def plot_ic_uniform_progression(save_path: str | Path | None = None) -> None:
    """Uniform IC across low / mean / high T0_offset."""
    lo, hi = UNIFORM_OFFSET_RANGE
    panels = [
        (f"T0_offset = {lo:+.1f}", {"T0_offset": float(lo)}),
        (f"T0_offset = {0.5 * (lo + hi):+.1f}", {"T0_offset": 0.5 * (lo + hi)}),
        (f"T0_offset = {hi:+.1f}", {"T0_offset": float(hi)}),
    ]
    _render_ic_progression(
        "uniform_2d", panels,
        name="ic_uniform_progression",
        suptitle="Uniform IC progression — T0_offset sweep",
        save_path=save_path,
    )


def plot_ic_random_sinusoid_progression(save_path: str | Path | None = None) -> None:
    """Random-sinusoid IC at low / mean / high amplitude (fixed wavenumbers)."""
    rng = np.random.default_rng(_IC_PROG_FIXED_SEED)
    N = 4
    nx_list = [float(v) for v in rng.uniform(1.0, SINU_KMAX, size=N)]
    ny_list = [int(v) for v in rng.integers(1, SINU_KMAX + 1, size=N)]

    lo, hi = SINU_AMP_RANGE
    amplitudes = [float(lo), 0.5 * (lo + hi), float(hi)]

    def make_params(scale: float) -> dict:
        return {
            "A_list": [scale] * N,
            "nx_list": nx_list,
            "ny_list": ny_list,
        }

    panels = [(f"A = {amp:.1f}", make_params(amp)) for amp in amplitudes]
    _render_ic_progression(
        "random_sinusoid_2d", panels,
        name="ic_random_sinusoid_progression",
        suptitle=f"Random-sinusoid IC progression — amplitude sweep (N={N})",
        save_path=save_path,
    )


def plot_ic_grf_progression(save_path: str | Path | None = None) -> None:
    """Random-field IC at low / log-mean / high correlation length (fixed sigma, wn_seed)."""
    lo, hi = GRF_ELL_RANGE
    sigma_fixed = 10.0
    wn_seed = _IC_PROG_FIXED_SEED
    ells = [float(lo), float(np.sqrt(lo * hi)), float(hi)]

    panels = [
        (f"ell = {ell:.3f}", {"ell": ell, "sigma": sigma_fixed, "wn_seed": wn_seed})
        for ell in ells
    ]
    _render_ic_progression(
        "grf_2d", panels,
        name="ic_grf_progression",
        suptitle=(
            "Random-field IC progression — correlation length sweep "
            f"(sigma={sigma_fixed:.1f})"
        ),
        save_path=save_path,
    )


def plot_ic_hot_spot_progression(save_path: str | Path | None = None) -> None:
    """Hot-spot IC at low / log-mean / high sigma (fixed bump centers and amplitudes)."""
    rng = np.random.default_rng(_IC_PROG_FIXED_SEED)
    N = 3
    amp_lo, amp_hi = HOT_AMP_RANGE
    A_list = [float(v) for v in rng.uniform(amp_lo, amp_hi, size=N)]
    s_lo, s_hi = HOT_SIGMA_RANGE
    sigmas = [float(s_lo), float(np.sqrt(s_lo * s_hi)), float(s_hi)]

    margin_max = 2.0 * s_hi  # keep centers in-bounds for the widest sigma panel
    margin = min(margin_max, 0.45)
    mu_x_list = [float(v) for v in rng.uniform(margin, 1.0 - margin, size=N)]
    mu_y_list = [float(v) for v in rng.uniform(margin, 1.0 - margin, size=N)]

    def make_params(sigma: float) -> dict:
        return {
            "A_list": A_list,
            "mu_x_list": mu_x_list,
            "mu_y_list": mu_y_list,
            "sigma_list": [sigma] * N,
        }

    panels = [(f"sigma = {s:.3f}", make_params(s)) for s in sigmas]
    _render_ic_progression(
        "hot_spot_2d", panels,
        name="ic_hot_spot_progression",
        suptitle=f"Hot-spot IC progression — sigma sweep (N={N})",
        save_path=save_path,
    )


def plot_snapshot_pair_samples(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    n_samples: int = 3,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot sampled 2D source/target snapshot pairs from train/val/test splits."""
    config = _resolve_plot_config(config)
    interface_meta = _resolve_interface_metadata(config=config)
    datasets = _build_split_datasets(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    rng = np.random.default_rng(seed)
    splits = [("Train", datasets["train"]), ("Val", datasets["val"]), ("Test", datasets["test"])]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 2 * n_samples, figsize=(5.2 * n_samples, 11), squeeze=False, constrained_layout=True)

        for row, (split_name, dataset) in enumerate(splits):
            row_axes = axes[row]
            if len(dataset) == 0:
                for ax in row_axes:
                    ax.set_axis_off()
                continue

            sample_size = min(n_samples, len(dataset))
            chosen_indices = rng.choice(len(dataset), size=sample_size, replace=False)

            for col in range(n_samples):
                if col >= sample_size:
                    row_axes[2 * col].set_axis_off()
                    row_axes[2 * col + 1].set_axis_off()
                    continue
                pair_idx = int(chosen_indices[col])
                sim_id, s, j = dataset._pairs[pair_idx]
                lead_time = dataset._lead_times[pair_idx]
                source = trajectories[sim_id, s]
                target = trajectories[sim_id, j]
                params = sim_params[sim_id]
                R_c = float(params["R_c"])
                forcing_label = _forcing_label(params)
                vmin = float(min(np.min(source), np.min(target)))
                vmax = float(max(np.max(source), np.max(target)))
                ax_source = row_axes[2 * col]
                ax_target = row_axes[2 * col + 1]

                pcm = _plot_field_2d(
                    ax_source,
                    x_grid,
                    y_grid,
                    source,
                    vmin=vmin,
                    vmax=vmax,
                    interface_positions=interface_meta["positions"],
                )
                _plot_field_2d(
                    ax_target,
                    x_grid,
                    y_grid,
                    target,
                    vmin=vmin,
                    vmax=vmax,
                    interface_positions=interface_meta["positions"],
                )

                text = f"Δt={lead_time:.3f}\n{forcing_label}\nR_c={float(R_c):.2f}"
                ax_target.text(
                    0.03,
                    0.97,
                    text,
                    transform=ax_target.transAxes,
                    va="top",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
                )
                ax_source.set_title(f"{split_name} | Sim {sim_id} | source t={t_grid[s]:.3f}")
                ax_target.set_title(f"target t={t_grid[j]:.3f}")
                fig.colorbar(pcm, ax=[ax_source, ax_target], label="Temperature", shrink=0.82)

        fig.suptitle("Snapshot Pair Samples — All-to-All Conditioning View")
        _save_figure(fig, save_path, "data", "snapshot_pair_samples", layout="constrained")


def plot_interface_error(
    model,
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_ids: np.ndarray,
    n_samples: int = 4,
    n_targets: int = 20,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
    dt: float | None = None,
):
    """Diagnose interface-jump prediction accuracy over y and target time."""
    config = _resolve_plot_config(config)
    interface_meta = _resolve_interface_metadata(config=config)

    rng = np.random.default_rng(seed)
    chosen_sims = rng.choice(sim_ids, size=min(n_samples, len(sim_ids)), replace=False)
    cases = []
    for sim_id in chosen_sims:
        s = int(rng.integers(0, len(t_grid) - 1))
        target_indices = _future_target_indices(s, n_targets, len(t_grid))
        case = _prepare_prediction_case(
            model,
            trajectories,
            x_grid,
            y_grid,
            t_grid,
            sim_params,
            sim_id,
            s,
            target_indices,
            config=config,
            dt=dt,
        )
        cases.append(case)

    truth_limits = []
    residual_limits = []
    for case in cases:
        Y_true = case["Y_true"]
        Y_pred = case["Y_pred"]
        left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, case["interface_x"])
        true_jump = _interface_jump_map(Y_true, left_node, right_node)
        pred_jump = _interface_jump_map(Y_pred, left_node, right_node)
        resid_jump = pred_jump - true_jump
        truth_limits.extend([float(np.min(true_jump)), float(np.max(true_jump)), float(np.min(pred_jump)), float(np.max(pred_jump))])
        residual_limits.extend([float(np.min(resid_jump)), float(np.max(resid_jump))])

    jump_abs = max(abs(min(truth_limits)), abs(max(truth_limits)), 1e-8)
    resid_abs = max(abs(min(residual_limits)), abs(max(residual_limits)), 1e-8)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(cases), 3, figsize=(18, 4.1 * len(cases)), squeeze=False)
        for row_idx, case in enumerate(cases):
            ax_left, ax_mid, ax_right = axes[row_idx]
            t_targets = case["t_targets"]
            Y_true = case["Y_true"]
            Y_pred = case["Y_pred"]
            left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, case["interface_x"])
            interface_metric_mask = _build_interface_mask_2d(
                x_grid, y_grid,
                case["interface_x"],
                case["interface_half_width"],
            )
            true_jump = _interface_jump_map(Y_true, left_node, right_node)
            pred_jump = _interface_jump_map(Y_pred, left_node, right_node)
            residual_jump = pred_jump - true_jump
            global_rel = _relative_l2_percent(Y_pred, Y_true)
            iface_rel = _relative_l2_percent(Y_pred[:, interface_metric_mask], Y_true[:, interface_metric_mask])
            jump_rel = _relative_l2_percent(pred_jump, true_jump)
            pcm_true = ax_left.pcolormesh(t_targets, y_grid, true_jump.T, cmap="coolwarm", vmin=-jump_abs, vmax=jump_abs, shading="auto")
            pcm_pred = ax_mid.pcolormesh(t_targets, y_grid, pred_jump.T, cmap="coolwarm", vmin=-jump_abs, vmax=jump_abs, shading="auto")
            pcm_resid = ax_right.pcolormesh(
                t_targets,
                y_grid,
                residual_jump.T,
                cmap="coolwarm",
                vmin=-resid_abs,
                vmax=resid_abs,
                shading="auto",
            )
            ax_left.set_xlabel("Time")
            ax_left.set_ylabel("y")
            ax_left.set_title(f"True Jump Map — x={case['interface_x']:.3f}")
            ax_mid.set_xlabel("Time")
            ax_mid.set_ylabel("y")
            ax_mid.set_title(f"Predicted Jump Map — x={case['interface_x']:.3f}")
            ax_right.set_xlabel("Time")
            ax_right.set_ylabel("y")
            ax_right.set_title("Jump Residual Map")

            ax_right.text(
                0.03,
                0.05,
                f"Sim {row_idx + 1}: A={case['amp']:.0f}, f={case['freq']:.2f}, R_c={case['R_c']:.2f}\n"
                f"Global rel. L2: {global_rel:.2f}%\nIface rel. L2: {iface_rel:.2f}%\nJump rel. err.: {jump_rel:.2f}%",
                transform=ax_right.transAxes,
                va="bottom",
                ha="left",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )
            fig.colorbar(pcm_true, ax=[ax_left, ax_mid], label="ΔT", shrink=0.82)
            fig.colorbar(pcm_resid, ax=ax_right, label="Pred - Truth", shrink=0.82)

        fig.suptitle("Interface Diagnostic")
        _save_figure(fig, save_path, "data", "interface_error", layout="constrained")


def plot_lead_time_coverage(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    bins: int = 20,
    save_path: str | Path | None = None,
):
    """Visualize train/test pair coverage across lead times and curriculum cutoffs."""
    coverage = _lead_time_coverage_counts(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    train_leads = coverage["train"]
    val_leads = coverage["val"]
    test_leads = coverage["test"]
    max_lead = max(
        float(train_leads.max()) if len(train_leads) > 0 else 0.0,
        float(val_leads.max()) if len(val_leads) > 0 else 0.0,
        float(test_leads.max()) if len(test_leads) > 0 else 0.0,
        float(t_grid[-1] - t_grid[0]),
    )
    hist_bins = np.linspace(0.0, max_lead, bins + 1)
    bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    warmup_epochs = int(config.get("training", {}).get("curriculum_warmup", 0))
    fractions = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    cutoff_positions = fractions * max_lead

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_hist, ax_frac) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        split_series = [
            ("Train", train_leads, "C0"),
            ("Val", val_leads, "C2"),
            ("Test", test_leads, "C3"),
        ]
        for label, values, color in split_series:
            counts, _ = np.histogram(values, bins=hist_bins)
            ax_hist.step(bin_centers, counts, where="mid", color=color, linewidth=1.8, label=f"{label} ({len(values)})")
            fractions_hist = counts / max(len(values), 1)
            ax_frac.step(bin_centers, fractions_hist, where="mid", color=color, linewidth=1.8, label=label)

        for frac, cutoff in zip(fractions, cutoff_positions):
            epoch_label = int(round(frac * warmup_epochs)) if warmup_epochs > 0 else 0
            label = "curriculum lead-time cutoff"
            if warmup_epochs > 0:
                label += f" (epoch {epoch_label})"
            ax_hist.axvline(cutoff, color="0.35", linestyle=":", alpha=0.8)
            ax_frac.axvline(cutoff, color="0.35", linestyle=":", alpha=0.8)
            ax_frac.text(cutoff, 1.02 * max(ax_frac.get_ylim()[1], 1.0), label, rotation=90, va="bottom", ha="center", fontsize=8)

        ax_hist.set_ylabel("Pair count")
        ax_hist.set_title("Lead-Time Coverage by Split")
        ax_hist.legend(loc="upper right")
        ax_hist.grid(True)

        ax_frac.set_xlabel("Lead time Δt")
        ax_frac.set_ylabel("Fraction of split")
        ax_frac.set_title("Normalized Coverage")
        ax_frac.legend(loc="upper right")
        ax_frac.grid(True)
        _save_figure(fig, save_path, "data", "lead_time_coverage", layout="constrained")


def plot_lead_time_error(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_bins: int = 8,
    max_samples: int = 256,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
):
    """Plot held-out error as a function of lead time using raw points, binned medians, and IQR bands."""
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        seed=seed,
    )
    centers, global_med, global_q25, global_q75, counts = _compute_binned_quantiles(
        records["lead_time"], records["global_rel_l2"], n_bins
    )
    _, iface_med, iface_q25, iface_q75, _ = _compute_binned_quantiles(
        records["lead_time"], records["iface_rel_l2"], n_bins
    )

    with plt.rc_context(PLOT_STYLE):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        ax_top.scatter(records["lead_time"], records["global_rel_l2"], s=16, alpha=0.18, color="0.55", label="Raw global")
        ax_top.scatter(records["lead_time"], records["iface_rel_l2"], s=16, alpha=0.12, color="C3", label="Raw interface")
        ax_top.plot(centers, global_med, marker="o", color="C0", label="Binned median global")
        ax_top.fill_between(centers, global_q25, global_q75, color="C0", alpha=0.15, label="Global IQR")
        ax_top.plot(centers, iface_med, marker="s", color="C3", label="Binned median interface")
        ax_top.fill_between(centers, iface_q25, iface_q75, color="C3", alpha=0.12, label="Interface IQR")
        ax_top.set_ylabel("Error (%)")
        ax_top.set_title("Held-Out Error vs Lead Time")
        ax_top.legend(loc="upper left")
        ax_top.grid(True)

        width = 0.8 * (centers[1] - centers[0]) if len(centers) > 1 else 0.02
        ax_bot.bar(centers, counts, width=width, color="0.65")
        ax_bot.set_xlabel("Lead time Δt")
        ax_bot.set_ylabel("Samples")
        ax_bot.grid(True, axis="y")

        _save_figure(fig, save_path, "data", "lead_time_error")


def plot_parameter_error_slices(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    max_samples: int = 256,
    n_bins: int = 6,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
):
    """Plot held-out error against amplitude, frequency, and contact resistance using medians and IQR bands."""
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        seed=seed,
    )
    specs = [
        ("amplitude", "Flux Amplitude (A)", False),
        ("frequency", "Flux Frequency (f)", True),
        ("R_c", "Contact Resistance (R_c)", False),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, (key, x_label, use_log) in zip(axes, specs):
            x_vals = records[key]
            ax.scatter(x_vals, records["global_rel_l2"], s=14, alpha=0.18, color="0.55", label="Raw global")
            ax.scatter(x_vals, records["iface_rel_l2"], s=14, alpha=0.12, color="C3", label="Raw interface")
            centers, global_med, global_q25, global_q75, _ = _compute_binned_quantiles(x_vals, records["global_rel_l2"], n_bins)
            _, iface_med, iface_q25, iface_q75, _ = _compute_binned_quantiles(x_vals, records["iface_rel_l2"], n_bins)
            ax.plot(centers, global_med, color="C0", linewidth=1.8, marker="o", label="Binned median global")
            ax.fill_between(centers, global_q25, global_q75, color="C0", alpha=0.14, label="Global IQR")
            ax.plot(centers, iface_med, color="C3", linewidth=1.8, marker="s", label="Binned median interface")
            ax.fill_between(centers, iface_q25, iface_q75, color="C3", alpha=0.12, label="Interface IQR")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Error (%)")
            ax.grid(True)
            if use_log:
                ax.set_xscale("log")

        axes[0].legend(loc="upper left")
        fig.suptitle("Held-Out Error Across Conditioning Dimensions")
        _save_figure(fig, save_path, "data", "parameter_error_slices")


def plot_dataset_summary(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    representative_sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Report-style overview of the 2D dataset."""
    config = _resolve_plot_config(config)
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)
    sim_id = _select_representative_sim_id(sim_params) if representative_sim_id is None else int(representative_sim_id)
    coverage = _lead_time_coverage_counts(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    interface_meta = _resolve_interface_metadata(config=config)
    left_node, right_node = _interface_flanking_nodes_from_grid(x_grid, interface_meta["interface_x"])
    rc_order = np.argsort(contact_resistance)
    low_idx = int(rc_order[len(rc_order) // 4])
    high_idx = int(rc_order[(3 * len(rc_order)) // 4])
    late_idx = max(1, int(round(0.75 * (len(t_grid) - 1))))
    ic_std = np.std(trajectories[:, 0], axis=0)
    low_jump = _interface_jump_profile(trajectories[low_idx, late_idx], left_node, right_node)
    high_jump = _interface_jump_profile(trajectories[high_idx, late_idx], left_node, right_node)
    max_lead = max(
        float(coverage["train"].max()) if len(coverage["train"]) > 0 else 0.0,
        float(coverage["val"].max()) if len(coverage["val"]) > 0 else 0.0,
        float(coverage["test"].max()) if len(coverage["test"]) > 0 else 0.0,
        float(t_grid[-1] - t_grid[0]),
    )
    hist_bins = np.linspace(0.0, max_lead, 16)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        temp_fields = np.stack([trajectories[sim_id, 0], trajectories[sim_id, -1]])
        temp_vmin = float(np.min(temp_fields))
        temp_vmax = float(np.max(temp_fields))
        std_vmax = float(np.max(ic_std))

        pcm = _plot_field_2d(
            axes[0, 0],
            x_grid,
            y_grid,
            trajectories[sim_id, 0],
            vmin=temp_vmin,
            vmax=temp_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 0].set_title(f"Representative Initial Field — Sim {sim_id}")

        _plot_field_2d(
            axes[0, 1],
            x_grid,
            y_grid,
            trajectories[sim_id, -1],
            vmin=temp_vmin,
            vmax=temp_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 1].set_title(f"Representative Final Field — t={t_grid[-1]:.3f}")

        pcm_std = _plot_field_2d(
            axes[0, 2],
            x_grid,
            y_grid,
            ic_std,
            cmap="viridis",
            vmin=0.0,
            vmax=std_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 2].set_title("Initial-Condition Std. Dev.")

        scatter = axes[1, 0].scatter(amplitudes, frequencies, c=contact_resistance, cmap="viridis", s=18, alpha=0.7, edgecolors="none")
        axes[1, 0].set_title("Parameter-Space Coverage")
        axes[1, 0].set_xlabel("Flux Amplitude (A)")
        axes[1, 0].set_ylabel("Flux Frequency (f)")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True)

        for label, values, color in [("Train", coverage["train"], "C0"), ("Val", coverage["val"], "C2"), ("Test", coverage["test"], "C3")]:
            axes[1, 1].hist(values, bins=hist_bins, histtype="step", linewidth=1.8, label=label, color=color)
        axes[1, 1].set_title("Lead-Time Coverage by Split")
        axes[1, 1].set_xlabel("Lead time Δt")
        axes[1, 1].set_ylabel("Pair count")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].grid(True)

        axes[1, 2].plot(y_grid, low_jump, color="C0", label=f"Low $R_c$={contact_resistance[low_idx]:.2f}")
        axes[1, 2].plot(y_grid, high_jump, color="C3", linestyle="--", label=f"High $R_c$={contact_resistance[high_idx]:.2f}")
        axes[1, 2].axhline(0.0, color="0.45", linestyle=":", linewidth=1.0)
        axes[1, 2].set_xlabel("y")
        axes[1, 2].set_ylabel("ΔT")
        axes[1, 2].set_title(f"Interface Jump Example — t={t_grid[late_idx]:.3f}")
        axes[1, 2].legend(loc="best")
        axes[1, 2].grid(True)

        fig.colorbar(pcm, ax=axes[0, :2].tolist(), label="Temperature", shrink=0.92)
        fig.colorbar(pcm_std, ax=axes[0, 2], label="Std. Dev.", shrink=0.92)
        fig.colorbar(scatter, ax=axes[1, 0], label="R_c", shrink=0.92)

        fig.suptitle("Dataset Summary")
        _save_figure(fig, save_path, "data", "dataset_summary", layout="constrained")


def plot_interface_jump_summary(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict,
    max_samples: int = 256,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Aggregate held-out interface-jump diagnostics over sampled test pairs."""
    config = _resolve_plot_config(config)
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        seed=seed,
    )

    lead_centers, lead_med, lead_q25, lead_q75, _ = _compute_binned_quantiles(records["lead_time"], records["jump_rel_l2"], 8)
    rc_centers, rc_med, rc_q25, rc_q75, _ = _compute_binned_quantiles(records["R_c"], records["jump_rel_l2"], 6)
    jump_rel_median = float(np.median(records["jump_rel_l2"]))
    jump_rel_p90 = float(np.percentile(records["jump_rel_l2"], 90))
    global_rel_median = float(np.median(records["global_rel_l2"]))
    iface_rel_median = float(np.median(records["iface_rel_l2"]))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

        axes[0, 0].scatter(records["lead_time"], records["jump_rel_l2"], s=16, alpha=0.18, color="0.55")
        axes[0, 0].plot(lead_centers, lead_med, color="C0", marker="o", label="Binned median jump rel. err.")
        axes[0, 0].fill_between(lead_centers, lead_q25, lead_q75, color="C0", alpha=0.15, label="Jump rel. err. IQR")
        axes[0, 0].set_title("Jump Relative Error vs Lead Time")
        axes[0, 0].set_xlabel("Lead time Δt")
        axes[0, 0].set_ylabel("Jump relative error (%)")
        axes[0, 0].legend(loc="upper left")
        axes[0, 0].grid(True)

        axes[0, 1].scatter(records["R_c"], records["jump_rel_l2"], s=16, alpha=0.18, color="0.55")
        axes[0, 1].plot(rc_centers, rc_med, color="C3", marker="o", label="Binned median jump rel. err.")
        axes[0, 1].fill_between(rc_centers, rc_q25, rc_q75, color="C3", alpha=0.15, label="Jump rel. err. IQR")
        axes[0, 1].set_title("Jump Relative Error vs R_c")
        axes[0, 1].set_xlabel("Contact Resistance (R_c)")
        axes[0, 1].set_ylabel("Jump relative error (%)")
        axes[0, 1].legend(loc="upper left")
        axes[0, 1].grid(True)

        min_jump = min(float(np.min(records["true_jump_rms"])), float(np.min(records["pred_jump_rms"])))
        max_jump = max(float(np.max(records["true_jump_rms"])), float(np.max(records["pred_jump_rms"])))
        axes[1, 0].scatter(records["true_jump_rms"], records["pred_jump_rms"], s=18, alpha=0.25, color="C2")
        axes[1, 0].plot([min_jump, max_jump], [min_jump, max_jump], color="black", linestyle="--", label="Identity")
        axes[1, 0].set_title("Predicted vs True Jump RMS")
        axes[1, 0].set_xlabel("True jump RMS")
        axes[1, 0].set_ylabel("Predicted jump RMS")
        axes[1, 0].legend(loc="upper left")
        axes[1, 0].grid(True)

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.02,
            0.98,
            "Aggregate Metrics\n\n"
            f"Median jump rel. err.: {jump_rel_median:.2f}%\n"
            f"90th pct jump rel. err.: {jump_rel_p90:.2f}%\n"
            f"Median global rel. L2: {global_rel_median:.2f}%\n"
            f"Median interface rel. L2: {iface_rel_median:.2f}%",
            va="top",
            ha="left",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "0.85"},
        )

        fig.suptitle("Interface Jump Summary")
        _save_figure(fig, save_path, "data", "interface_jump_summary", layout="constrained")


# ============================================================
# SOURCE BENCHMARK — helpers (ported from experiment/source-itr)
# ============================================================

def _patch_param_arrays(sim_params: np.ndarray) -> dict[str, np.ndarray]:
    """Pull per-sim patch parameters into arrays."""
    return {
        "R_c": np.array([float(p["R_c"]) for p in sim_params], dtype=np.float32),
        "A": np.array([float(p["A"]) for p in sim_params], dtype=np.float32),
        "x_h": np.array([float(p["x_h"]) for p in sim_params], dtype=np.float32),
        "y_h": np.array([float(p["y_h"]) for p in sim_params], dtype=np.float32),
        "regime": np.array([str(p.get("regime", "")) for p in sim_params], dtype=object),
    }


def _void_param_arrays(sim_params: np.ndarray) -> dict[str, np.ndarray]:
    """Pull source-ITR Gaussian-void parameters into arrays."""
    R_base = np.array([float(p["R_c_base"]) for p in sim_params], dtype=np.float64)
    R_amp = np.array([float(p["R_c_amp"]) for p in sim_params], dtype=np.float64)
    return {
        "R_c_base": R_base,
        "R_c_amp": R_amp,
        "R_c_y0": np.array([float(p["R_c_y0"]) for p in sim_params], dtype=np.float64),
        "R_c_sigma": np.array([float(p["R_c_sigma"]) for p in sim_params], dtype=np.float64),
        "R_c_peak": R_base + R_amp,
    }


def _interface_conductance_profile(
    y_grid: np.ndarray,
    R_c_y: np.ndarray,
    x_grid: np.ndarray | None = None,
    interface_x: float = INTERFACE_X,
    k_left: float = _DEFAULT_K_LEFT,
    k_right: float = _DEFAULT_K_RIGHT,
) -> np.ndarray:
    """Return the per-row interface conductance profile with solver-matching distances."""
    y = np.asarray(y_grid, dtype=np.float64)
    Rc = np.asarray(R_c_y, dtype=np.float64)
    if x_grid is not None:
        x = np.asarray(x_grid, dtype=np.float64)
        return np.asarray(
            _interface_conductance_G(x, interface_x, Rc, k_left, k_right),
            dtype=np.float64,
        )

    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    return 1.0 / ((0.5 * dy) / float(k_left) + Rc + (0.5 * dy) / float(k_right))


def _void_severity_arrays(
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return sampled void parameters plus integrated severity metrics."""
    arrs = _void_param_arrays(sim_params)
    y = np.asarray(y_grid, dtype=np.float64)
    excess_integral = np.zeros(len(sim_params), dtype=np.float64)
    conductance_deficit = np.zeros(len(sim_params), dtype=np.float64)

    for i, p in enumerate(sim_params):
        Rc_y = make_rc_void_profile(
            y,
            R_base=float(p["R_c_base"]),
            R_amp=float(p["R_c_amp"]),
            y0=float(p["R_c_y0"]),
            sigma=float(p["R_c_sigma"]),
        )
        base = np.full_like(Rc_y, float(p["R_c_base"]), dtype=np.float64)
        G_base = _interface_conductance_profile(
            y, base, x_grid=x_grid, interface_x=float(p.get("interface_x", INTERFACE_X))
        )
        G_y = _interface_conductance_profile(
            y, Rc_y, x_grid=x_grid, interface_x=float(p.get("interface_x", INTERFACE_X))
        )
        bounds = (0.0, 1.0)
        weights = interface_control_volume_weights(y, bounds)
        excess_integral[i] = integrated_excess_resistance(
            y, Rc_y, float(p["R_c_base"]), bounds=bounds
        )
        conductance_deficit[i] = float(np.sum(weights * (G_base - G_y)))

    arrs["R_c_excess_integral"] = excess_integral
    arrs["conductance_deficit"] = conductance_deficit
    arrs["is_void_active"] = arrs["R_c_amp"] > 1e-12
    return arrs


def _representative_void_sim_ids(
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> list[int]:
    """Pick deterministic low/median/high severity source-ITR samples."""
    severity = _void_severity_arrays(sim_params, y_grid, x_grid)["R_c_excess_integral"]
    if len(severity) == 0:
        return []
    order = np.argsort(severity)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    out: list[int] = []
    for sid in picks:
        if sid not in out:
            out.append(sid)
    return out


def _require_source_itr_params(sim_params: np.ndarray) -> None:
    required = ("R_c_base", "R_c_amp", "R_c_y0", "R_c_sigma")
    for i, p in enumerate(sim_params):
        missing = [key for key in required if key not in p]
        if missing:
            raise ValueError(f"sim_params[{i}] missing source-ITR keys {missing}.")


def _select_representative_source_sim_id(sim_params: np.ndarray) -> int:
    """Choose a deterministic representative source sim near the median patch params."""
    arrs = _patch_param_arrays(sim_params)
    A = arrs["A"]
    if len(A) == 0:
        return 0
    Rc = arrs["R_c"]
    xh = arrs["x_h"]
    yh = arrs["y_h"]

    def _norm(v: np.ndarray) -> np.ndarray:
        v_min, v_max = float(v.min()), float(v.max())
        return (v - v_min) / max(v_max - v_min, 1e-12)

    stacked = np.column_stack([_norm(A), _norm(Rc), _norm(xh), _norm(yh)])
    target = np.median(stacked, axis=0)
    return int(np.argmin(np.sum((stacked - target) ** 2, axis=1)))


def _format_patch_label(p: dict) -> str:
    """Compact label describing a sim's patch parameters."""
    return (
        f"x_h={float(p['x_h']):.2f}, y_h={float(p['y_h']):.2f}, "
        f"A={float(p['A']):.0f}"
    )


def _build_source_bin_channels(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    x_h: float,
    y_h: float,
    w_h: float,
    h_h: float,
    A: float,
    t_off: float,
    t_s: float,
    t_j: float,
    t_final: float,
    source_bins: int = SOURCE_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (S_h, Q_bins) matching the dataset's source-bin encoding.

    S_h     : (Nx, Ny) indicator
    Q_bins  : (Nx, Ny, source_bins) future-source bins divided by Q_REF.
    """
    A_max = float(PATCH_A_RANGE[1])
    q_ref = float(A_max * float(t_final) / float(source_bins))
    S_h = make_patch_indicator(X_grid, Y_grid, x_h, y_h, w_h, h_h)
    Nx, Ny = X_grid.shape
    Q = np.zeros((Nx, Ny, source_bins), dtype=np.float32)
    if t_j > t_s:
        edges = np.linspace(t_s, t_j, source_bins + 1, dtype=np.float64)
        for k in range(source_bins):
            integral = integrate_sin2_pulse(A, t_off, float(edges[k]), float(edges[k + 1]))
            Q[..., k] = S_h * np.float32(integral / q_ref)
    return S_h.astype(np.float32), Q


_REGIME_COLORS = {"left": "C0", "near": "C2", "right": "C3"}


def _regime_for_sim(p: dict, interface_x: float = INTERFACE_X) -> str:
    """Classify a sim's patch as left/near/right relative to the interface."""
    regime = p.get("regime", None)
    if regime in _REGIME_COLORS:
        return str(regime)
    x_h = float(p["x_h"])
    w_h = float(p["w_h"])
    if x_h + 0.5 * w_h < interface_x:
        return "left"
    if x_h - 0.5 * w_h > interface_x:
        return "right"
    return "near"


def _representative_sim_per_regime(
    sim_params: np.ndarray,
    interface_x: float = INTERFACE_X,
) -> dict[str, int]:
    """Pick one sim_id per regime, falling back gracefully."""
    selections: dict[str, int] = {}
    for r in ("left", "near", "right"):
        for i, p in enumerate(sim_params):
            if _regime_for_sim(p, interface_x) == r:
                selections[r] = i
                break
    return selections


def _extract_breakdown_payload(breakdown_path: str | Path) -> dict:
    """Load breakdown.json and return the per-seed[0] payload (or top-level dict)."""
    import json

    with open(breakdown_path, "r") as f:
        data = json.load(f)
    if "per_seed" in data:
        if not data["per_seed"]:
            raise ValueError(f"breakdown.json has empty per_seed: {breakdown_path}")
        return data["per_seed"][0]
    return data


def _grid_error_heatmap(
    ax, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    n_bins: int, xlabel: str, ylabel: str,
    x_log: bool = False, vmin: float | None = None, vmax: float | None = None,
):
    """Plot mean(z) over quantile bins of (x, y) and return the QuadMesh."""
    if x_log:
        x_safe = np.log10(np.clip(x, 1e-12, None))
    else:
        x_safe = x
    x_edges = np.quantile(x_safe, np.linspace(0, 1, n_bins + 1))
    y_edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)
    if len(x_edges) < 2 or len(y_edges) < 2:
        ax.text(0.5, 0.5, "(degenerate)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return None
    bx = np.clip(np.digitize(x_safe, x_edges[1:-1]), 0, len(x_edges) - 2)
    by = np.clip(np.digitize(y, y_edges[1:-1]), 0, len(y_edges) - 2)
    grid = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            mask = (bx == i) & (by == j)
            if np.any(mask):
                grid[i, j] = float(np.mean(z[mask]))
    pcm = ax.pcolormesh(x_edges, y_edges, grid.T, cmap="viridis",
                       shading="flat", vmin=vmin, vmax=vmax)
    if x_log:
        ax.set_xlabel(xlabel + " (log10)")
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return pcm


def _bilinear_resample(
    field: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
    xs: np.ndarray, ys: np.ndarray,
) -> np.ndarray:
    """Bilinear resample ``field`` of shape (Nx, Ny) at sample points (xs, ys).

    xs, ys arrays of the same shape. Returns the resampled field with that
    shape. Points outside the convex hull return NaN.
    """
    Nx = len(x_grid)
    Ny = len(y_grid)
    x_lo = float(x_grid[0])
    x_hi = float(x_grid[-1])
    y_lo = float(y_grid[0])
    y_hi = float(y_grid[-1])
    valid = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
    fx = (xs - x_lo) / max(x_hi - x_lo, 1e-12) * (Nx - 1)
    fy = (ys - y_lo) / max(y_hi - y_lo, 1e-12) * (Ny - 1)
    i0 = np.clip(np.floor(fx).astype(int), 0, Nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, Ny - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    wx = fx - i0
    wy = fy - j0
    f00 = field[i0, j0]
    f01 = field[i0, j1]
    f10 = field[i1, j0]
    f11 = field[i1, j1]
    out = (
        (1 - wx) * (1 - wy) * f00
        + (1 - wx) * wy * f01
        + wx * (1 - wy) * f10
        + wx * wy * f11
    )
    out = np.where(valid, out, np.nan)
    return out


# ============================================================
# SOURCE BENCHMARK — plots (ported from experiment/source-itr)
# ============================================================

def plot_source_itr_void_profiles(
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """Explain sampled Gaussian-void interface resistance profiles."""
    _require_source_itr_params(sim_params)
    y = np.asarray(y_grid, dtype=np.float64)
    severity = _void_severity_arrays(sim_params, y, x_grid)
    representative_ids = _representative_void_sim_ids(sim_params, y, x_grid)

    profile_rows = []
    for sid in representative_ids:
        p = sim_params[int(sid)]
        Rc_y = make_rc_void_profile(
            y,
            R_base=float(p["R_c_base"]),
            R_amp=float(p["R_c_amp"]),
            y0=float(p["R_c_y0"]),
            sigma=float(p["R_c_sigma"]),
        )
        G_y = _interface_conductance_profile(
            y, Rc_y, x_grid=x_grid, interface_x=float(p.get("interface_x", INTERFACE_X))
        )
        profile_rows.append((int(sid), p, Rc_y, rc_log_norm(Rc_y), G_y))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(profile_rows), 1)))
    active = severity["is_void_active"]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

        ax = axes[0, 0]
        for color, (sid, p, Rc_y, _Rc_norm, _G_y) in zip(colors, profile_rows):
            label = (
                f"sim {sid}: peak={float(p['R_c_base']) + float(p['R_c_amp']):.2f}, "
                f"area={severity['R_c_excess_integral'][sid]:.3f}"
            )
            ax.plot(y, Rc_y, color=color, label=label)
            if float(p["R_c_amp"]) > 1e-12:
                ax.axvline(float(p["R_c_y0"]), color=color, linestyle=":", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$R_c(y)$")
        ax.set_title("Physical interface resistance")
        ax.legend(loc="upper right")
        ax.grid(True)

        ax = axes[0, 1]
        for color, (sid, _p, _Rc_y, Rc_norm, _G_y) in zip(colors, profile_rows):
            ax.plot(y, Rc_norm, color=color, label=f"sim {sid}")
        ax.axhline(-1.0, color="0.7", linestyle=":", linewidth=1.0)
        ax.axhline(1.0, color="0.7", linestyle=":", linewidth=1.0)
        ax.set_xlabel("y")
        ax.set_ylabel("normalized channel")
        ax.set_title(r"Model input channel: log-normalized $R_c(y)$")
        ax.grid(True)

        ax = axes[1, 0]
        for color, (sid, _p, _Rc_y, _Rc_norm, G_y) in zip(colors, profile_rows):
            ax.plot(y, G_y, color=color, label=f"sim {sid}")
        ax.set_xlabel("y")
        ax.set_ylabel(r"$G(y)$")
        ax.set_title("Discrete interface conductance")
        ax.grid(True)

        ax = axes[1, 1]
        if np.any(active):
            peak_active = severity["R_c_peak"][active]
            peak_size = (peak_active - float(np.min(peak_active))) / max(
                float(np.max(peak_active) - np.min(peak_active)), 1e-12
            )
            scatter = ax.scatter(
                severity["R_c_y0"][active],
                severity["R_c_sigma"][active],
                c=severity["R_c_excess_integral"][active],
                s=35 + 70 * peak_size,
                cmap="magma",
                alpha=0.78,
                edgecolors="white",
                linewidths=0.4,
            )
            fig.colorbar(scatter, ax=ax, label=r"$\int (R_c(y)-R_{base})\,dy$")
        if np.any(~active):
            ax.scatter(
                np.full(int(np.sum(~active)), np.nanmean(severity["R_c_y0"])),
                np.full(int(np.sum(~active)), RC_VOID_RANGES["sigma"][0]),
                marker="x",
                color="0.45",
                s=32,
                alpha=0.8,
                label="flat profile",
            )
            ax.legend(loc="upper right")
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        sig_lo, sig_hi = RC_VOID_RANGES["sigma"]
        ax.set_xlim(y0_lo - 0.03, y0_hi + 0.03)
        ax.set_ylim(sig_lo * 0.85, sig_hi * 1.08)
        ax.set_xlabel(r"void center $y_0$")
        ax.set_ylabel(r"void width $\sigma$")
        ax.set_title("Void location/width coverage")
        ax.grid(True)

        fig.suptitle("Source-ITR Gaussian Void Profiles")
        _save_figure(fig, save_path, "source", "source_itr_void_profiles", layout="constrained")


def _aggregate_void_errors_by_sim(
    records: dict[str, np.ndarray],
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate sampled pair errors to one row per simulation."""
    severity = _void_severity_arrays(sim_params, y_grid, x_grid)
    sim_ids = np.asarray(records["sim_id"], dtype=np.int64)
    unique = np.unique(sim_ids)
    rows: dict[str, list] = {
        "sim_id": [],
        "iface_median": [],
        "iface_p90": [],
        "global_median": [],
        "lead_median": [],
        "pair_count": [],
    }
    for key in (
        "R_c_base",
        "R_c_amp",
        "R_c_y0",
        "R_c_sigma",
        "R_c_peak",
        "R_c_excess_integral",
        "conductance_deficit",
        "is_void_active",
    ):
        rows[key] = []

    for sid in unique:
        mask = sim_ids == int(sid)
        rows["sim_id"].append(int(sid))
        rows["iface_median"].append(float(np.median(records["iface_rel_l2"][mask])))
        rows["iface_p90"].append(float(np.percentile(records["iface_rel_l2"][mask], 90)))
        rows["global_median"].append(float(np.median(records["global_rel_l2"][mask])))
        rows["lead_median"].append(float(np.median(records["lead_time"][mask])))
        rows["pair_count"].append(int(np.sum(mask)))
        for key in (
            "R_c_base",
            "R_c_amp",
            "R_c_y0",
            "R_c_sigma",
            "R_c_peak",
            "R_c_excess_integral",
            "conductance_deficit",
            "is_void_active",
        ):
            rows[key].append(severity[key][int(sid)])

    out: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        dtype = bool if key == "is_void_active" else np.float64
        if key in {"sim_id", "pair_count"}:
            dtype = np.int64
        out[key] = np.asarray(values, dtype=dtype)
    return out


def _sim_horizon_void_rows(
    records: dict[str, np.ndarray],
    sim_params: np.ndarray,
    y_grid: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate pair errors by simulation and lead-time tertile."""
    severity = _void_severity_arrays(sim_params, y_grid, x_grid)
    sim_ids = np.asarray(records["sim_id"], dtype=np.int64)
    lead = np.asarray(records["lead_time"], dtype=np.float64)
    iface = np.asarray(records["iface_rel_l2"], dtype=np.float64)
    if len(lead) == 0:
        return {"severity": np.array([]), "lead_mid": np.array([]), "iface_median": np.array([]), "label": np.array([], dtype=object)}

    edges = np.unique(np.quantile(lead, np.linspace(0.0, 1.0, 4)))
    if len(edges) < 2:
        edges = np.array([float(np.min(lead)) - 1e-9, float(np.max(lead)) + 1e-9])

    rows = {"severity": [], "lead_mid": [], "iface_median": [], "label": []}
    for b in range(len(edges) - 1):
        lo = float(edges[b])
        hi = float(edges[b + 1])
        if b == len(edges) - 2:
            lead_mask = (lead >= lo) & (lead <= hi)
        else:
            lead_mask = (lead >= lo) & (lead < hi)
        label = f"{lo:.3f}-{hi:.3f}"
        for sid in np.unique(sim_ids[lead_mask]):
            mask = lead_mask & (sim_ids == int(sid))
            if not np.any(mask):
                continue
            rows["severity"].append(float(severity["R_c_excess_integral"][int(sid)]))
            rows["lead_mid"].append(0.5 * (lo + hi))
            rows["iface_median"].append(float(np.median(iface[mask])))
            rows["label"].append(label)

    return {
        "severity": np.asarray(rows["severity"], dtype=np.float64),
        "lead_mid": np.asarray(rows["lead_mid"], dtype=np.float64),
        "iface_median": np.asarray(rows["iface_median"], dtype=np.float64),
        "label": np.asarray(rows["label"], dtype=object),
    }


def plot_source_itr_error_vs_void_params(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict | None = None,
    max_samples: int = 256,
    batch_size: int = 64,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Plot source-ITR held-out error against Gaussian-void parameters."""
    _require_source_itr_params(dataset.sim_params)
    records = _compute_pair_error_records(
        model=model,
        dataset=dataset,
        x_grid=x_grid,
        y_grid=y_grid,
        config=config,
        max_samples=max_samples,
        batch_size=batch_size,
        seed=seed,
    )
    sim_rows = _aggregate_void_errors_by_sim(records, dataset.sim_params, y_grid, x_grid)
    horizon_rows = _sim_horizon_void_rows(records, dataset.sim_params, y_grid, x_grid)
    pair_severity = _void_severity_arrays(dataset.sim_params, y_grid, x_grid)["R_c_excess_integral"][
        np.asarray(records["sim_id"], dtype=np.int64)
    ]
    active = sim_rows["is_void_active"]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)

        ax = axes[0, 0]
        ax.scatter(pair_severity, records["iface_rel_l2"], s=10, alpha=0.08, color="0.25", edgecolors="none", label="pair")
        sim_scatter = ax.scatter(
            sim_rows["R_c_excess_integral"],
            sim_rows["iface_median"],
            c=sim_rows["lead_median"],
            cmap="viridis",
            s=52,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.5,
            label="sim median",
        )
        fig.colorbar(sim_scatter, ax=ax, label="Median lead time")
        centers, med, q25, q75, _ = _compute_binned_quantiles(
            sim_rows["R_c_excess_integral"], sim_rows["iface_median"], 6
        )
        if len(centers):
            ax.plot(centers, med, color="C3", marker="o", label="binned median")
            ax.fill_between(centers, q25, q75, color="C3", alpha=0.14, label="IQR")
        ax.set_xlabel(r"$\int (R_c(y)-R_{base})\,dy$")
        ax.set_ylabel("Interface rel. L2 (%)")
        ax.set_title("Per-simulation median vs integrated severity")
        ax.legend(loc="best")
        ax.grid(True)

        ax = axes[0, 1]
        scatter = ax.scatter(
            sim_rows["R_c_peak"],
            sim_rows["iface_p90"],
            c=sim_rows["conductance_deficit"],
            cmap="magma",
            s=56,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
        fig.colorbar(scatter, ax=ax, label=r"$\int (G_{base}-G(y))\,dy$")
        ax.set_xlabel(r"peak $R_c$")
        ax.set_ylabel("Interface rel. L2 p90 (%)")
        ax.set_title("Tail error vs peak resistance")
        ax.grid(True)

        ax = axes[0, 2]
        if np.any(active):
            scatter = ax.scatter(
                sim_rows["R_c_y0"][active],
                sim_rows["R_c_sigma"][active],
                c=sim_rows["iface_median"][active],
                s=48 + 80 * (
                    sim_rows["R_c_excess_integral"][active]
                    / max(float(np.max(sim_rows["R_c_excess_integral"][active])), 1e-12)
                ),
                cmap="plasma",
                alpha=0.82,
                edgecolors="white",
                linewidths=0.5,
            )
            fig.colorbar(scatter, ax=ax, label="Median interface rel. L2 (%)")
        if np.any(~active):
            ax.text(
                0.02, 0.04,
                f"{int(np.sum(~active))} flat profiles excluded from y0/sigma",
                transform=ax.transAxes,
                color="0.35",
                fontsize=8,
            )
        ax.set_xlabel(r"void center $y_0$")
        ax.set_ylabel(r"void width $\sigma$")
        ax.set_title("Location/width colored by error")
        ax.grid(True)

        ax = axes[1, 0]
        if len(horizon_rows["severity"]):
            for label in np.unique(horizon_rows["label"]):
                mask = horizon_rows["label"] == label
                centers, med, _q25, _q75, counts = _compute_binned_quantiles(
                    horizon_rows["severity"][mask],
                    horizon_rows["iface_median"][mask],
                    5,
                )
                if len(centers):
                    ax.plot(centers, med, marker="o", label=f"Δt {label} (n={int(np.sum(counts))})")
        ax.set_xlabel(r"$\int (R_c(y)-R_{base})\,dy$")
        ax.set_ylabel("Median interface rel. L2 (%)")
        ax.set_title("Severity trend by horizon bin")
        ax.legend(loc="best")
        ax.grid(True)

        ax = axes[1, 1]
        pcm = _grid_error_heatmap(
            ax,
            horizon_rows["severity"],
            horizon_rows["lead_mid"],
            horizon_rows["iface_median"],
            n_bins=5,
            xlabel=r"integrated void severity",
            ylabel="lead-time bin midpoint",
        )
        ax.set_title("Median error by severity × horizon")
        if pcm is not None:
            fig.colorbar(pcm, ax=ax, label="Median interface rel. L2 (%)")

        ax = axes[1, 2]
        scatter = ax.scatter(
            sim_rows["global_median"],
            sim_rows["iface_median"],
            c=sim_rows["R_c_excess_integral"],
            cmap="viridis",
            s=54,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.5,
        )
        fig.colorbar(scatter, ax=ax, label="Integrated severity")
        ax.set_xlabel("Global rel. L2 median (%)")
        ax.set_ylabel("Interface rel. L2 median (%)")
        ax.set_title("Interface error vs global error")
        ax.grid(True)

        fig.suptitle("Source-ITR Error vs Gaussian-Void Parameters")
        _save_figure(fig, save_path, "source", "source_itr_error_vs_void_params", layout="constrained")


def plot_source_error_vs_params(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    max_samples: int = 256,
    seed: int = 42,
    config: dict | None = None,
    save_path: str | Path | None = None,
):
    """Scatter rel-L2 against patch and contact parameters plus a regime box plot.

    Panels (2x3):
      (0,0) rel-L2 vs R_c
      (0,1) rel-L2 vs A (log x)
      (0,2) rel-L2 vs x_h
      (1,0) rel-L2 vs y_h
      (1,1) rel-L2 vs |x_h - 0.5|
      (1,2) box plot by regime {left, near, right}
    """
    records = _compute_pair_error_records(
        model=model, dataset=dataset,
        x_grid=x_grid, y_grid=y_grid,
        config=config, max_samples=max_samples, seed=seed,
    )
    rel = records["global_rel_l2"]
    dist = np.abs(records["x_h"] - float(INTERFACE_X))
    regimes = records["regime"]

    specs = [
        ("R_c", records["R_c"], False, "Contact resistance R_c"),
        ("A", records["A"], True, "Patch amplitude A"),
        ("x_h", records["x_h"], False, "Patch x-center x_h"),
        ("y_h", records["y_h"], False, "Patch y-center y_h"),
        ("|x_h - 0.5|", dist, False, "Distance to interface |x_h - 0.5|"),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        flat = axes.ravel()
        for ax, (key, xv, log_x, label) in zip(flat[:5], specs):
            ax.scatter(xv, rel, s=18, alpha=0.45, color="C0", edgecolors="none")
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("Global rel. L2 (%)")
            ax.grid(True)

        ax_box = flat[5]
        regime_order = ["left", "near", "right"]
        data = [rel[regimes == r] for r in regime_order]
        positions = list(range(1, len(regime_order) + 1))
        for pos, label, samples in zip(positions, regime_order, data):
            if len(samples) > 0:
                ax_box.boxplot(
                    [samples], positions=[pos], widths=0.6,
                    patch_artist=True,
                    boxprops={"facecolor": "C2", "alpha": 0.4},
                )
            else:
                ax_box.text(pos, 0.5, "(empty)", ha="center", va="center",
                            transform=ax_box.get_xaxis_transform(),
                            fontsize=8, color="0.5")
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(regime_order)
        ax_box.set_ylabel("Global rel. L2 (%)")
        ax_box.set_title("By regime")
        ax_box.grid(True, axis="y")

        fig.suptitle("Held-Out Error Across Patch/Contact Parameters")
        _save_figure(fig, save_path, "source", "source_error_vs_params", layout="constrained")


def plot_source_interface_zone_error(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict,
    max_samples: int = 256,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Lightweight interface-zone diagnostic.

    (a) Scatter of pred-vs-truth jump magnitude |T(0.5+) - T(0.5-)| (RMS over y).
    (b) Scatter of rel-L2 inside the band |x - 0.5| < 0.05 vs R_c.
    The jump is computed from the two cell centers adjacent to x=0.5, matching
    the FV face convention.
    """
    config = _resolve_plot_config(config)
    records = _compute_pair_error_records(
        model=model, dataset=dataset,
        x_grid=x_grid, y_grid=y_grid,
        config=config, max_samples=max_samples, seed=seed,
    )

    true_jump = records["true_jump_rms"]
    pred_jump = records["pred_jump_rms"]
    rc = records["R_c"]
    iface_rel = records["iface_rel_l2"]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

        ax = axes[0]
        if len(true_jump) > 0:
            lo = float(min(true_jump.min(), pred_jump.min()))
            hi = float(max(true_jump.max(), pred_jump.max()))
        else:
            lo, hi = 0.0, 1.0
        ax.scatter(true_jump, pred_jump, s=18, alpha=0.45, color="C2", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0, label="Identity")
        ax.set_xlabel(r"True $|T(0.5^+) - T(0.5^-)|$ RMS")
        ax.set_ylabel(r"Predicted $|T(0.5^+) - T(0.5^-)|$ RMS")
        ax.set_title("Jump magnitude — pred vs truth")
        ax.legend(loc="upper left")
        ax.grid(True)

        ax = axes[1]
        ax.scatter(rc, iface_rel, s=18, alpha=0.45, color="C3", edgecolors="none")
        ax.set_xlabel(r"Contact resistance $R_c$")
        ax.set_ylabel("Interface band rel. L2 (%)")
        ax.set_title(r"Band $|x - 0.5| < 0.05$ error vs $R_c$")
        ax.grid(True)

        fig.suptitle("Interface Zone Diagnostic")
        _save_figure(fig, save_path, "source", "source_interface_zone_error", layout="constrained")


def plot_source_dataset_summary(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    representative_sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Report-style overview of the source-itr dataset."""
    config = _resolve_plot_config(config)
    arrs = _patch_param_arrays(sim_params)
    sim_id = _select_representative_source_sim_id(sim_params) if representative_sim_id is None else int(representative_sim_id)
    coverage = _lead_time_coverage_counts(trajectories, x_grid, y_grid, t_grid, sim_params, config)
    interface_meta = _resolve_interface_metadata(config=config)
    ic_std = np.std(trajectories[:, 0], axis=0)
    max_lead = max(
        float(coverage["train"].max()) if len(coverage["train"]) > 0 else 0.0,
        float(coverage["val"].max()) if len(coverage["val"]) > 0 else 0.0,
        float(coverage["test"].max()) if len(coverage["test"]) > 0 else 0.0,
        float(t_grid[-1] - t_grid[0]),
    )
    hist_bins = np.linspace(0.0, max_lead, 16)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        temp_fields = np.stack([trajectories[sim_id, 0], trajectories[sim_id, -1]])
        temp_vmin = float(np.min(temp_fields))
        temp_vmax = float(np.max(temp_fields))
        std_vmax = float(np.max(ic_std))

        pcm = _plot_field_2d(
            axes[0, 0], x_grid, y_grid, trajectories[sim_id, 0],
            vmin=temp_vmin, vmax=temp_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 0].set_title(f"Representative IC — Sim {sim_id}")

        _plot_field_2d(
            axes[0, 1], x_grid, y_grid, trajectories[sim_id, -1],
            vmin=temp_vmin, vmax=temp_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 1].set_title(f"Representative Final Field — t={t_grid[-1]:.3f}")

        pcm_std = _plot_field_2d(
            axes[0, 2], x_grid, y_grid, ic_std,
            cmap="viridis", vmin=0.0, vmax=std_vmax,
            interface_positions=interface_meta["positions"],
        )
        axes[0, 2].set_title("Initial-Condition Std. Dev.")

        scatter = axes[1, 0].scatter(arrs["x_h"], arrs["y_h"], c=arrs["A"],
                                     cmap="viridis", s=18, alpha=0.7, edgecolors="none")
        axes[1, 0].axvline(INTERFACE_X, color="0.35", linestyle=":", linewidth=1.0)
        axes[1, 0].set_title("Patch x_h vs y_h (color = A)")
        axes[1, 0].set_xlabel("x_h")
        axes[1, 0].set_ylabel("y_h")
        axes[1, 0].grid(True)

        for label, values, color in [("Train", coverage["train"], "C0"), ("Val", coverage["val"], "C2"), ("Test", coverage["test"], "C3")]:
            axes[1, 1].hist(values, bins=hist_bins, histtype="step", linewidth=1.8, label=label, color=color)
        axes[1, 1].set_title("Lead-Time Coverage by Split")
        axes[1, 1].set_xlabel("Lead time Δt")
        axes[1, 1].set_ylabel("Pair count")
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].grid(True)

        axes[1, 2].hist(arrs["R_c"], bins=20, color="C2", edgecolor="white")
        for x_lim in RC_RANGE:
            axes[1, 2].axvline(x_lim, color="0.4", linestyle=":", linewidth=1.0)
        axes[1, 2].set_xlabel(r"$R_c$")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].set_title(r"Contact resistance $R_c$")
        axes[1, 2].grid(True)

        fig.colorbar(pcm, ax=axes[0, :2].tolist(), label="Temperature", shrink=0.92)
        fig.colorbar(pcm_std, ax=axes[0, 2], label="Std. Dev.", shrink=0.92)
        fig.colorbar(scatter, ax=axes[1, 0], label="A", shrink=0.92)

        fig.suptitle("Dataset Summary — source-itr")
        _save_figure(fig, save_path, "source", "source_dataset_summary", layout="constrained")


def plot_patch_param_scatter(
    sim_params: np.ndarray,
    split_assignments: dict[str, np.ndarray] | None = None,
    save_path: str | Path | None = None,
):
    """2x3 overview of patch parameter sampling and regime distribution."""
    arrs = _patch_param_arrays(sim_params)
    x_h = arrs["x_h"]
    y_h = arrs["y_h"]
    R_c = arrs["R_c"]
    A = arrs["A"]
    regimes = np.array([_regime_for_sim(p) for p in sim_params], dtype=object)

    if split_assignments is None:
        num_sims = len(sim_params)
        train_ids, val_ids, test_ids = split_sim_ids(num_sims, 0.7, 0.15, seed=0)
        split_assignments = {"train": train_ids, "val": val_ids, "test": test_ids}

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

        ax = axes[0, 0]
        for label, color in _REGIME_COLORS.items():
            mask = regimes == label
            if not np.any(mask):
                continue
            ax.scatter(
                x_h[mask], y_h[mask], c=color, s=18, alpha=0.75,
                edgecolors="none", label=label,
            )
        ax.axvline(INTERFACE_X, color="0.35", linestyle=":", linewidth=1.2)
        ax.axvspan(
            INTERFACE_X - 0.5 * PATCH_W, INTERFACE_X + 0.5 * PATCH_W,
            color="0.7", alpha=0.2,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("x_h")
        ax.set_ylabel("y_h")
        ax.set_title("Patch Position by Regime")
        ax.legend(loc="upper right")
        ax.grid(True)

        axes[0, 1].hist(R_c, bins=20, color="C2", edgecolor="white")
        axes[0, 1].set_xlabel(r"$R_c$")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Contact Resistance")
        axes[0, 1].grid(True)

        axes[0, 2].hist(A, bins=np.logspace(np.log10(A.min()), np.log10(A.max()), 20),
                       color="C1", edgecolor="white")
        axes[0, 2].set_xscale("log")
        axes[0, 2].set_xlabel("A")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].set_title("Patch Amplitude")
        axes[0, 2].grid(True, which="both")

        axes[1, 0].hist(x_h, bins=20, color="C0", edgecolor="white")
        axes[1, 0].axvline(INTERFACE_X, color="0.35", linestyle=":")
        axes[1, 0].set_xlabel("x_h")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Patch x")
        axes[1, 0].grid(True)

        axes[1, 1].hist(y_h, bins=20, color="C3", edgecolor="white")
        axes[1, 1].set_xlabel("y_h")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Patch y")
        axes[1, 1].grid(True)

        ax = axes[1, 2]
        split_labels = list(split_assignments.keys())
        regime_labels = ["left", "near", "right"]
        counts = np.zeros((len(regime_labels), len(split_labels)), dtype=int)
        for j, split in enumerate(split_labels):
            ids = np.asarray(split_assignments[split], dtype=int)
            if len(ids) == 0:
                continue
            split_regimes = regimes[ids]
            for i, r in enumerate(regime_labels):
                counts[i, j] = int(np.sum(split_regimes == r))
        bottom = np.zeros(len(split_labels), dtype=int)
        for i, r in enumerate(regime_labels):
            ax.bar(split_labels, counts[i], bottom=bottom,
                  color=_REGIME_COLORS[r], label=r, edgecolor="white")
            bottom = bottom + counts[i]
        ax.set_ylabel("Sim count")
        ax.set_title("Regime Counts by Split")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y")

        fig.suptitle("Patch Parameter Distributions")
        _save_figure(fig, save_path, "source", "patch_param_scatter", layout="constrained")


def plot_source_temporal_profile(
    sim_params: np.ndarray,
    t_grid: np.ndarray,
    save_path: str | Path | None = None,
):
    """1x2 overview of the patch's temporal amplitude."""
    arrs = _patch_param_arrays(sim_params)
    A = arrs["A"]
    if len(A) == 0:
        raise ValueError("sim_params is empty; cannot plot temporal profile.")
    t_off = float(sim_params[0]["t_off"])
    t_final = float(t_grid[-1])
    rep_idx = _select_representative_source_sim_id(sim_params)
    rep_A = float(sim_params[rep_idx]["A"])
    rep_t_off = float(sim_params[rep_idx]["t_off"])

    quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    A_samples = np.quantile(A, quantiles)

    t_dense = np.linspace(0.0, t_final, 400)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

        ax = axes[0]
        for q, A_val in zip(quantiles, A_samples):
            pulse = make_sin2_pulse(float(A_val), t_off)
            ax.plot(t_dense, [pulse(t) for t in t_dense],
                   label=f"A={A_val:.0f} (q={q:.2f})")
        ax.axvline(t_off, color="0.35", linestyle=":", label=r"$t_{off}$")
        ax.axvline(t_final, color="0.55", linestyle="--", label=r"$t_{final}$")
        ax.set_xlabel("t")
        ax.set_ylabel("a(t)")
        ax.set_title("Patch Amplitude Pulse")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

        ax = axes[1]
        edges = np.linspace(0.0, t_final, SOURCE_BINS + 1)
        integrals = np.array([
            integrate_sin2_pulse(rep_A, rep_t_off, float(edges[k]), float(edges[k + 1]))
            for k in range(SOURCE_BINS)
        ])
        ax.step(edges[:-1], integrals, where="post", color="C0", linewidth=1.8)
        ax.axvline(rep_t_off, color="0.35", linestyle=":", label=r"$t_{off}$")
        ax.axvline(t_final, color="0.55", linestyle="--", label=r"$t_{final}$")
        ax.set_xlabel("t bin start")
        ax.set_ylabel(r"$\int_{\tau_k}^{\tau_{k+1}} a(t)\,dt$")
        ax.set_title(f"Bin Integrals — sim {rep_idx}, A={rep_A:.0f}")
        ax.legend(loc="upper right")
        ax.grid(True)

        _save_figure(fig, save_path, "source", "source_temporal_profile", layout="constrained")


def plot_regime_error_breakdown(
    breakdown_path: str | Path,
    save_path: str | Path | None = None,
):
    """1x3 grouped bar chart of mean rel-L2 (physical) by regime / R_c / A bins."""
    payload = _extract_breakdown_payload(breakdown_path)

    def _bars_from_section(section: dict) -> tuple[list[str], list[float], list[float], list[int]]:
        labels, means, stds, counts = [], [], [], []
        for k, entry in section.items():
            if not isinstance(entry, dict):
                continue
            count = int(entry.get("count", 0))
            if count == 0:
                continue
            labels.append(str(k))
            means.append(float(entry.get("rel_l2_phys_mean", 0.0)))
            stds.append(float(entry.get("rel_l2_phys_std", 0.0)))
            counts.append(count)
        return labels, means, stds, counts

    sections = [
        ("by_regime", "By Regime"),
        ("by_R_c", r"By $R_c$ quartile"),
        ("by_A", "By A quartile"),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for ax, (key, title) in zip(axes, sections):
            section = payload.get(key, {})
            labels, means, stds, counts = _bars_from_section(section)
            if not labels:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                       transform=ax.transAxes)
                ax.set_title(title)
                continue
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=stds, color="C0", edgecolor="white",
                          capsize=4, alpha=0.85)
            for rect, c in zip(bars, counts):
                ax.text(rect.get_x() + rect.get_width() / 2.0,
                       rect.get_height(),
                       f"n={c}",
                       ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel("Relative L2 (%, physical)")
            ax.set_title(title)
            ax.grid(True, axis="y")

        fig.suptitle("Regime / Parameter Error Breakdown")
        _save_figure(fig, save_path, "source", "regime_error_breakdown", layout="constrained")


def plot_patch_error_slices(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: dict | None = None,
    sim_params: np.ndarray | None = None,
    max_samples: int = 512,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """2x2 heatmaps of mean rel-L2 over quantile bins of patch params x t_bar."""
    records = _compute_pair_error_records(
        model, dataset, x_grid, y_grid,
        config=config, max_samples=max_samples, seed=seed,
    )
    rel = records["global_rel_l2"]
    t_bar = records["lead_time"]

    vmax = float(np.nanpercentile(rel, 95))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

        pcm = _grid_error_heatmap(
            axes[0, 0], records["R_c"], t_bar, rel, n_bins=6,
            xlabel=r"$R_c$", ylabel=r"$\Delta t$", vmax=vmax,
        )
        axes[0, 0].set_title(r"Error vs $R_c$, $\Delta t$")

        _grid_error_heatmap(
            axes[0, 1], records["A"], t_bar, rel, n_bins=6,
            xlabel="A", ylabel=r"$\Delta t$", x_log=True, vmax=vmax,
        )
        axes[0, 1].set_title(r"Error vs A, $\Delta t$")

        _grid_error_heatmap(
            axes[1, 0], np.abs(records["x_h"] - INTERFACE_X), t_bar, rel, n_bins=6,
            xlabel=r"$|x_h - 0.5|$", ylabel=r"$\Delta t$", vmax=vmax,
        )
        axes[1, 0].set_title(r"Error vs interface offset, $\Delta t$")

        _grid_error_heatmap(
            axes[1, 1], records["y_h"], t_bar, rel, n_bins=6,
            xlabel=r"$y_h$", ylabel=r"$\Delta t$", vmax=vmax,
        )
        axes[1, 1].set_title(r"Error vs $y_h$, $\Delta t$")

        if pcm is not None:
            fig.colorbar(pcm, ax=axes.ravel().tolist(),
                        label="Mean rel L2 (%, physical)", shrink=0.85)
        fig.suptitle("Patch-Parameter Error Slices")
        _save_figure(fig, save_path, "source", "patch_error_slices", layout="constrained")


def plot_patch_region_error_map(
    model,
    dataset: SnapshotPairDataset,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict | None = None,
    window: float = 0.3,
    max_samples: int = 128,
    seed: int = 42,
    save_path: str | Path | None = None,
):
    """Mean per-pixel MAE recentered to each pair's patch center (x_h, y_h)."""
    import torch

    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot compute patch-region error map.")

    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, len(dataset))
    sample_indices = np.arange(len(dataset))
    if sample_size < len(dataset):
        sample_indices = np.sort(rng.choice(sample_indices, size=sample_size, replace=False))

    x_batch, c_batch, y_batch, stats_batch = [], [], [], []
    forcing_batch = []
    has_forcing = True
    for idx in sample_indices:
        item = dataset[idx]
        x_batch.append(item["spatial"])
        c_batch.append(item["cond_static"])
        fseq = item.get("forcing_seq")
        if fseq is None:
            has_forcing = False
        else:
            forcing_batch.append(fseq)
        y_batch.append(item["Y"])
        stats_batch.append(item["T_stats"])
    x_tensor = torch.stack(x_batch, dim=0)
    c_tensor = torch.stack(c_batch, dim=0)
    forcing_tensor = torch.stack(forcing_batch, dim=0) if has_forcing else None
    y_tensor = torch.stack(y_batch, dim=0)
    stats_tensor = torch.stack(stats_batch, dim=0)

    device = next(model.parameters()).device
    use_temporal = bool(getattr(model, "use_temporal_encoder", True))
    model.eval()
    preds = []
    batch_size = 32
    for start in range(0, sample_size, batch_size):
        stop = min(start + batch_size, sample_size)
        with torch.no_grad():
            if use_temporal and forcing_tensor is not None:
                p = model(x_tensor[start:stop].to(device),
                         c_tensor[start:stop].to(device),
                         forcing_tensor[start:stop].to(device)).cpu()
            else:
                p = model(x_tensor[start:stop].to(device),
                         c_tensor[start:stop].to(device)).cpu()
        preds.append(p)
    y_pred = torch.cat(preds, dim=0)
    mu_s = stats_tensor[:, 0][:, None, None, None]
    sigma_s = stats_tensor[:, 1][:, None, None, None]
    y_pred_phys = (y_pred * (sigma_s + T_EPS) + mu_s).numpy()[..., 0]
    y_true_phys = (y_tensor * (sigma_s + T_EPS) + mu_s).numpy()[..., 0]
    abs_err = np.abs(y_pred_phys - y_true_phys)

    sim_ids = np.array([dataset._pairs[i][0] for i in sample_indices], dtype=np.int64)

    local_n = 64
    grid_local = np.linspace(-0.5 * window, 0.5 * window, local_n)
    Xl, Yl = np.meshgrid(grid_local, grid_local, indexing="ij")

    accum = np.zeros((local_n, local_n), dtype=np.float64)
    weight = np.zeros((local_n, local_n), dtype=np.float64)
    used = 0
    for k, sid in enumerate(sim_ids):
        params = sim_params[int(sid)]
        x_h = float(params["x_h"])
        y_h = float(params["y_h"])
        if (x_h - 0.5 * window < x_grid[0] or x_h + 0.5 * window > x_grid[-1]
            or y_h - 0.5 * window < y_grid[0] or y_h + 0.5 * window > y_grid[-1]):
            continue
        sampled = _bilinear_resample(abs_err[k], x_grid, y_grid, Xl + x_h, Yl + y_h)
        valid = ~np.isnan(sampled)
        accum[valid] += sampled[valid]
        weight[valid] += 1.0
        used += 1
    mean_err = np.where(weight > 0, accum / np.maximum(weight, 1.0), np.nan)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        pcm = ax.pcolormesh(grid_local, grid_local, mean_err.T, cmap="magma",
                           shading="auto")
        fig.colorbar(pcm, ax=ax, label="Mean |pred - truth|")
        rect = Rectangle((-0.5 * PATCH_W, -0.5 * PATCH_H), PATCH_W, PATCH_H,
                        fill=False, edgecolor="cyan", linewidth=1.6)
        ax.add_patch(rect)
        ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(r"$x - x_h$")
        ax.set_ylabel(r"$y - y_h$")
        ax.set_aspect("equal")
        ax.set_title(
            f"Mean Absolute Error Centered on Patch ({used} pairs; interface fixed at x=0.5)"
        )
        _save_figure(fig, save_path, "source", "patch_region_error_map", layout="constrained")


def plot_source_field_snapshots(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """2x3 grid of Q(x,y,t) at representative times for one sim."""
    if sim_id is None:
        sim_id = _select_representative_source_sim_id(sim_params)
    params = sim_params[int(sim_id)]
    x_h = float(params["x_h"])
    y_h = float(params["y_h"])
    w_h = float(params["w_h"])
    h_h = float(params["h_h"])
    A = float(params["A"])
    t_off = float(params["t_off"])
    t_final = float(t_grid[-1])

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    source = build_patch_source(x_h, y_h, w_h, h_h, A, t_off)

    times = np.array([
        0.25 * t_off,
        0.5 * t_off,
        t_off,
        0.5 * (t_off + t_final),
        max(t_final - 1e-6, t_off + 1e-6),
        min(0.1 * t_off, t_off * 0.05 + 1e-6),
    ])
    times.sort()

    fields = np.stack([source(X, Y, float(t)) for t in times], axis=0)
    vmin = float(np.min(fields))
    vmax = float(np.max(fields))
    interface_meta = _resolve_interface_metadata(sim_params=sim_params, sim_id=int(sim_id))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        pcm = None
        for ax, t_val, fld in zip(axes.ravel(), times, fields):
            pcm = _plot_field_2d(
                ax, x_grid, y_grid, fld,
                vmin=vmin, vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            rect = Rectangle(
                (x_h - 0.5 * w_h, y_h - 0.5 * h_h), w_h, h_h,
                fill=False, edgecolor="cyan", linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.plot([x_h], [y_h], marker="x", color="cyan", markersize=8)
            ax.set_title(f"t = {t_val:.3f}")

        fig.colorbar(pcm, ax=axes.ravel().tolist(), label="Q(x, y, t)", shrink=0.92)
        fig.suptitle(f"Source Field Snapshots — sim {sim_id} ({_format_patch_label(params)})")
        _save_figure(fig, save_path, "source", "source_field_snapshots", layout="constrained")


def plot_source_input_channels(
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Render the SOURCE_BINS Q channels from a built dataset sample."""
    if sim_id is None:
        sim_id = _select_representative_source_sim_id(sim_params)
    params = sim_params[int(sim_id)]
    x_h = float(params["x_h"])
    y_h = float(params["y_h"])
    w_h = float(params["w_h"])
    h_h = float(params["h_h"])
    A = float(params["A"])
    t_off = float(params["t_off"])
    t_final = float(t_grid[-1])

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    _, Q_bins = _build_source_bin_channels(
        X, Y, x_h, y_h, w_h, h_h, A, t_off,
        t_s=0.0, t_j=t_final, t_final=t_final,
    )
    n_panels = Q_bins.shape[-1]
    nrows, ncols = _shared_thumbnail_grid(n_panels)

    vmax = float(np.max(Q_bins))
    interface_meta = _resolve_interface_metadata(sim_params=sim_params, sim_id=int(sim_id))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                                constrained_layout=True)
        axes_arr = np.atleast_2d(axes).ravel()
        pcm = None
        for k in range(n_panels):
            ax = axes_arr[k]
            pcm = _plot_field_2d(
                ax, x_grid, y_grid, Q_bins[..., k],
                cmap="magma", vmin=0.0, vmax=vmax,
                interface_positions=interface_meta["positions"],
            )
            rect = Rectangle(
                (x_h - 0.5 * w_h, y_h - 0.5 * h_h), w_h, h_h,
                fill=False, edgecolor="cyan", linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.set_title(f"Q_{k}")
        for k in range(n_panels, len(axes_arr)):
            axes_arr[k].axis("off")

        fig.colorbar(pcm, ax=list(axes_arr[:n_panels]),
                    label="Q_k (normalized)", shrink=0.92)
        fig.suptitle(
            f"Source Channels Q_0..Q_{n_panels - 1} "
            f"(SOURCE_BINS={SOURCE_BINS}, SPATIAL_IN_CHANNELS={SOURCE_SPATIAL_IN_CHANNELS} detected)"
        )
        _save_figure(fig, save_path, "source", "source_input_channels", layout="constrained")


def plot_patch_overlay_trajectory(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_ids: dict[str, int] | None = None,
    save_path: str | Path | None = None,
):
    """One row per regime, 4 snapshots each, with patch rectangle overlay."""
    if sim_ids is None:
        sim_ids = _representative_sim_per_regime(sim_params)
    if not sim_ids:
        sim_ids = {"sim": _select_representative_source_sim_id(sim_params)}

    snap_idx = np.linspace(0, len(t_grid) - 1, 4, dtype=int)
    interface_meta = _resolve_interface_metadata()

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(len(sim_ids), len(snap_idx),
                                figsize=(4.0 * len(snap_idx), 3.6 * len(sim_ids)),
                                constrained_layout=True,
                                squeeze=False)
        for row, (label, sid) in enumerate(sim_ids.items()):
            params = sim_params[int(sid)]
            x_h = float(params["x_h"])
            y_h = float(params["y_h"])
            w_h = float(params["w_h"])
            h_h = float(params["h_h"])
            traj = trajectories[int(sid)]
            row_fields = traj[snap_idx]
            vmin = float(np.min(row_fields))
            vmax = float(np.max(row_fields))
            for col, t_idx in enumerate(snap_idx):
                ax = axes[row, col]
                pcm = _plot_field_2d(
                    ax, x_grid, y_grid, traj[t_idx],
                    vmin=vmin, vmax=vmax,
                    interface_positions=interface_meta["positions"],
                )
                rect = Rectangle(
                    (x_h - 0.5 * w_h, y_h - 0.5 * h_h), w_h, h_h,
                    fill=False, edgecolor="cyan", linewidth=1.2,
                )
                ax.add_patch(rect)
                ax.plot([x_h], [y_h], marker="x", color="cyan", markersize=8)
                ax.set_title(
                    f"{label} (sim {sid}) — t={t_grid[t_idx]:.3f}"
                )
                if col == len(snap_idx) - 1:
                    fig.colorbar(pcm, ax=ax, shrink=0.8, label="T")

        fig.suptitle("Patch Overlay Trajectory by Regime")
        _save_figure(fig, save_path, "source", "patch_overlay_trajectory", layout="constrained")


def plot_energy_budget(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Compare injected, stored, and outgoing energy for one representative sim.

    Assumes rho = cp = 1 (from CLAUDE.md) and right-wall material k=1
    matching the source-itr experiment. Right-wall Dirichlet uses one-sided
    cell-center to ghost difference, consistent with the FV solver stencil.
    """
    if sim_id is None:
        sim_id = _select_representative_source_sim_id(sim_params)
    params = sim_params[int(sim_id)]
    x_h = float(params["x_h"])
    y_h = float(params["y_h"])
    w_h = float(params["w_h"])
    h_h = float(params["h_h"])
    A = float(params["A"])
    t_off = float(params["t_off"])

    Nx = len(x_grid)
    Ny = len(y_grid)
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])

    cell_dx = np.full(Nx, dx)
    cell_dx[0] = dx / 2.0
    cell_dx[-1] = dx / 2.0
    cell_dy = np.full(Ny, dy)
    cell_dy[0] = dy / 2.0
    cell_dy[-1] = dy / 2.0
    cell_area = cell_dx[:, None] * cell_dy[None, :]

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    indicator = make_patch_indicator(X, Y, x_h, y_h, w_h, h_h)
    pulse = make_sin2_pulse(A, t_off)
    a_t = np.array([pulse(float(t)) for t in t_grid], dtype=np.float64)
    Q_per_cell_t = a_t[:, None, None] * indicator[None, ...]
    integrand_t = np.sum(Q_per_cell_t * cell_area[None, ...], axis=(1, 2))
    dt = np.diff(t_grid)
    E_in = np.concatenate([[0.0], np.cumsum(0.5 * (integrand_t[1:] + integrand_t[:-1]) * dt)])

    traj = trajectories[int(sim_id)].astype(np.float64)
    T0_field = traj[0]
    delta = traj - T0_field[None, ...]
    E_stored = np.sum(delta * cell_area[None, ...], axis=(1, 2))

    T_right_target = 300.0
    k_right = 1.0
    T_ghost = 2.0 * T_right_target - traj[:, -1, :]
    flux_t = -k_right * (T_ghost - traj[:, -1, :]) / (dx / 2.0)
    out_per_t = np.sum(flux_t * cell_dy[None, :], axis=1)
    E_out_right = np.concatenate([[0.0], np.cumsum(0.5 * (out_per_t[1:] + out_per_t[:-1]) * dt)])

    residual = E_in - E_stored - E_out_right

    arrs = _patch_param_arrays(sim_params)
    max_dev = np.array([
        float(np.max(np.mean(trajectories[i] - trajectories[i, 0], axis=(1, 2))))
        for i in range(len(sim_params))
    ])

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        ax = axes[0]
        ax.plot(t_grid, E_in, color="C0", label=r"$E_{in}$")
        ax.plot(t_grid, E_stored, color="C2", label=r"$\Delta E_{stored}$")
        ax.plot(t_grid, E_out_right, color="C3", label=r"$E_{out}^{right}$")
        ax.set_xlabel("t")
        ax.set_ylabel("Energy")
        ax.legend(loc="upper left")
        ax.grid(True)
        ax.set_title(f"Energy Budget — sim {sim_id} ({_format_patch_label(params)})")
        ax2 = ax.twinx()
        ax2.plot(t_grid, residual, color="0.4", linestyle="--",
                 label=r"residual $E_{in} - \Delta E_{stored} - E_{out}$")
        ax2.set_ylabel("Residual")
        ax2.legend(loc="lower right")

        ax = axes[1]
        ax.scatter(arrs["A"], max_dev, c="C1", alpha=0.7, edgecolors="none", s=18)
        ax.set_xscale("log")
        ax.set_xlabel("A")
        ax.set_ylabel(r"$\max_t \langle T - T_0 \rangle_\Omega$")
        ax.set_title("Response Scaling vs A")
        ax.grid(True, which="both")

        fig.suptitle("Energy Budget and Response Scaling")
        _save_figure(fig, save_path, "source", "energy_budget", layout="constrained")


# ============================================================
# INTERFACES BENCHMARK — helpers
# ============================================================

_IC_FAMILY_ORDER = ("uniform_2d", "random_sinusoid_2d", "grf_2d", "hot_spot_2d")


def _interface_x_array(sim_params: np.ndarray) -> np.ndarray:
    """Return per-sim interface_x array."""
    return np.array([float(p["interface_x"]) for p in sim_params], dtype=np.float64)


def _pick_interface_quartile_sims(sim_params: np.ndarray) -> list[tuple[str, int]]:
    """Pick one sim per interface_x quartile (Q1, Q2, Q3, Q4).

    Returns a list of (label, sim_id). Order is from low to high interface_x.
    Matches the existing breakdown style (no controlling for other params).
    """
    interface_x = _interface_x_array(sim_params)
    n = len(interface_x)
    if n == 0:
        return []
    order = np.argsort(interface_x)
    picks: list[tuple[str, int]] = []
    for q in range(4):
        lo = q * n // 4
        hi = (q + 1) * n // 4 if q < 3 else n
        if hi <= lo:
            continue
        mid = lo + (hi - lo) // 2
        sid = int(order[mid])
        picks.append((f"Q{q + 1}", sid))
    return picks


def _pick_first_per_ic_family(sim_params: np.ndarray) -> list[tuple[str, int]]:
    """Pick the first sim for each IC family present in sim_params."""
    seen: dict[str, int] = {}
    for i, p in enumerate(sim_params):
        fam = p.get("ic_family")
        if fam in _IC_FAMILY_ORDER and fam not in seen:
            seen[fam] = i
    return [(fam, seen[fam]) for fam in _IC_FAMILY_ORDER if fam in seen]


# ============================================================
# INTERFACES BENCHMARK — plots
# ============================================================

def plot_interface_y_perturbation(
    trajectories: np.ndarray,
    sim_id: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray | None = None,
    t_window: tuple[float, float] = (0.0, 0.25),
    save_path: str | Path | None = None,
):
    """Plot the y-perturbation T(x, y, t) - <T>_y(x, t) for one simulation.

    Subtracting the y-mean exposes the y-structure imprinted by the spatial
    flux profile, regardless of the absolute temperature scale.
    """
    Nt = trajectories.shape[1]
    snap_main = _snap_indices_in_window(t_grid, t_window[0], t_window[1], 5)
    post = np.array([Nt - 1], dtype=int)
    snap_indices = np.unique(np.concatenate([snap_main, post]))

    fields = trajectories[sim_id, snap_indices].astype(np.float32)  # (S, Nx, Ny)
    perturbations = fields - fields.mean(axis=2, keepdims=True)
    abs_max = max(float(np.max(np.abs(perturbations))), 1e-12)
    interface_meta = _resolve_interface_metadata(sim_params=sim_params, sim_id=sim_id)

    family_str = ""
    if sim_params is not None:
        family_str = f" — {_format_spatial_family(sim_params[sim_id])}"

    with plt.rc_context(PLOT_STYLE):
        nrows = 2
        ncols = int(np.ceil(len(snap_indices) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        pcm = None
        for ax, t_idx, field in zip(axes_flat, snap_indices, perturbations):
            pcm = _plot_field_2d(
                ax,
                x_grid,
                y_grid,
                field,
                cmap="coolwarm",
                vmin=-abs_max,
                vmax=abs_max,
                interface_positions=interface_meta["positions"],
            )
            ax.set_title(f"t = {t_grid[t_idx]:.3f}")
        for ax in axes_flat[len(snap_indices):]:
            ax.set_axis_off()

        fig.colorbar(pcm, ax=axes_flat.tolist(), label=r"$T - \langle T \rangle_y$", shrink=0.9)
        fig.suptitle(f"y-Perturbation — Simulation {sim_id}{family_str}")
        _save_figure(fig, save_path, "interfaces", "interface_y_perturbation", layout="constrained")


def plot_interface_lhs_scatter(
    sim_params: np.ndarray,
    save_path: str | Path | None = None,
):
    """Coverage summary for sampled conditioning parameters.

    Amplitude / frequency are NaN for non-sin sims; nan-aware ops keep the sin
    marginals visible while ignoring NaN entries (matplotlib drops them too).
    """
    amplitudes, frequencies, contact_resistance = _parameter_arrays(sim_params)

    def _normalize(values: np.ndarray, log: bool = False) -> np.ndarray:
        v = np.log10(values) if log else values
        lo = np.nanmin(v)
        hi = np.nanmax(v)
        return (v - lo) / max(hi - lo, 1e-12)

    normalized_marginals = [
        ("A",        _normalize(amplitudes),                "C0"),
        ("log10(f)", _normalize(frequencies, log=True),     "C3"),
        ("R_c",      _normalize(contact_resistance),        "C2"),
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        scatter_specs = [
            (axes[0, 0], amplitudes, frequencies, "Flux Amplitude (A)", "Flux Frequency (f)"),
            (axes[0, 1], amplitudes, contact_resistance, "Flux Amplitude (A)", "Contact Resistance (R_c)"),
            (axes[1, 0], frequencies, contact_resistance, "Flux Frequency (f)", "Contact Resistance (R_c)"),
        ]
        for ax, x_vals, y_vals, x_label, y_label in scatter_specs:
            ax.scatter(x_vals, y_vals, s=16, alpha=0.45, color="C0", edgecolors="none")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            if "Frequency" in x_label:
                ax.set_xscale("log")
            if "Frequency" in y_label:
                ax.set_yscale("log")

        hist_ax = axes[1, 1]
        bins = np.linspace(0.0, 1.0, 12)
        for label, values, color in normalized_marginals:
            hist_ax.hist(values, bins=bins, histtype="step", linewidth=1.8, label=label, color=color)
        hist_ax.set_xlabel("Normalized parameter value")
        hist_ax.set_ylabel("Count")
        hist_ax.set_title("Marginal Coverage")
        hist_ax.legend(loc="upper center")
        hist_ax.grid(True)

        fig.suptitle(f"Conditioning Parameter Coverage — {len(amplitudes)} simulations")
        _save_figure(fig, save_path, "interfaces", "interface_lhs_scatter", layout="constrained")


def plot_interface_flux_profiles(
    t_on: float = 0.0,
    t_off: float = 0.2,
    t_final: float = 1.0,
    save_path: str | Path | None = None,
):
    """Tile the new q_L(y, t) representation across patch / gaussian / triangle.

    Row 0 (full width): shared temporal forcing a(t).
    Rows 1-3: per family — s(y) | q_left(y, t) heatmap | q_x at t_peak.
    """
    layers = [
        Layer2D(x_left=0.0, x_right=0.5, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=0.5, x_right=1.0, rho=1.0, cp=1.0, k=1.0),
    ]
    families = [
        ("patch",    dict(y_c=0.50, w=0.30)),
        ("gaussian", dict(y_c=0.50, sigma_y=0.08)),
        ("triangle", dict(y_c=0.50, ell=0.20)),
    ]
    Nx, Ny = 60, 60
    a_x, b_x, c_y, d_y = 0.0, 1.0, 0.0, 1.0
    flux_A, flux_f, phase = 175.0, 10.5, 0.0
    y_grid = np.linspace(c_y, d_y, Ny)

    runs = []
    for name, sp in families:
        q_fn, s_vec = build_qL(
            "sin",
            dict(A=flux_A, f=flux_f, t_on=t_on, t_off=t_off, phase=phase),
            name,
            sp,
            y_grid,
        )
        solver = FVSolver2D(
            a=a_x, b=b_x, c=c_y, d=d_y,
            Nx=Nx, Ny=Ny,
            lam_target=0.8,
            layers=layers,
            interface_R=[0.5],
            t_final=t_final,
            flux_f=flux_f,
            flux_A=flux_A,
            t_on=t_on,
            t_off=t_off,
            phase=phase,
            dt=0.005,
            q_left_fn=q_fn,
        )
        T0_demo = np.full((solver.Nx, solver.Ny), solver.T_right(0.0), dtype=float)
        _, _, _, T_hist = solver.solve(T0=T0_demo, store_trajectory=True)
        Q = _evaluate_q_left_field(solver)
        t_peak = int(np.argmax(np.max(np.abs(Q), axis=1)))
        dT = T_hist[t_peak, 1:, :] - T_hist[t_peak, :-1, :]
        q_x = -solver.G_x * dT
        runs.append({
            "name": name, "sp": sp, "solver": solver, "T_hist": T_hist,
            "Q": Q, "s_vec": s_vec, "q_x": q_x, "t_peak": t_peak,
        })

    run0 = runs[0]
    j0 = int(np.argmax(np.abs(run0["s_vec"])))
    s0 = float(run0["s_vec"][j0])
    a_t_shared = run0["Q"][:, j0] / s0 if abs(s0) > 1e-12 else run0["Q"][:, j0]

    q_x_global = max(float(np.max([np.max(np.abs(r["q_x"])) for r in runs])), 1e-8)
    interface_meta = _resolve_interface_metadata(solver=runs[0]["solver"])

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(16, 13), constrained_layout=True)
        gs = fig.add_gridspec(4, 3, height_ratios=[0.7, 1.0, 1.0, 1.0])

        ax_at = fig.add_subplot(gs[0, :])
        solver0 = runs[0]["solver"]
        ax_at.plot(solver0.t, a_t_shared, color="C3")
        ax_at.axvline(t_on, color="C2", linestyle=":", alpha=0.8, label=f"t_on={t_on}")
        ax_at.axvline(t_off, color="C1", linestyle=":", alpha=0.8, label=f"t_off={t_off}")
        ax_at.set_xlabel("Time")
        ax_at.set_ylabel(r"$a(t)$")
        ax_at.set_title("Temporal forcing (shared across families)")
        ax_at.legend(loc="upper right")
        ax_at.grid(True)

        qx_axes = []
        pcm_x = None
        for row, run in enumerate(runs, start=1):
            solver = run["solver"]
            ax_s = fig.add_subplot(gs[row, 0])
            ax_q = fig.add_subplot(gs[row, 1])
            ax_qx = fig.add_subplot(gs[row, 2])

            ax_s.plot(solver.grid_y, run["s_vec"], color="C0")
            sp_str = ", ".join(f"{k}={v:.2f}" for k, v in run["sp"].items())
            ax_s.set_xlabel("y")
            ax_s.set_ylabel(r"$s(y)$")
            ax_s.set_title(f"{run['name']}: {sp_str}")
            ax_s.set_ylim(-0.1, 1.15)
            ax_s.grid(True)

            q_abs_row = max(float(np.max(np.abs(run["Q"]))), 1e-12)
            pcm_q = ax_q.pcolormesh(
                solver.t, solver.grid_y, run["Q"].T,
                cmap="coolwarm", vmin=-q_abs_row, vmax=q_abs_row, shading="auto",
            )
            ax_q.axvline(t_on, color="0.2", linestyle=":", alpha=0.6)
            ax_q.axvline(t_off, color="0.2", linestyle=":", alpha=0.6)
            ax_q.set_xlabel("Time")
            ax_q.set_ylabel("y")
            ax_q.set_title(rf"$q_{{left}}(y,t)$ — {run['name']}")
            fig.colorbar(pcm_q, ax=ax_q, shrink=0.9)

            pcm_x = ax_qx.pcolormesh(
                solver.face_positions_x, solver.grid_y, run["q_x"].T,
                cmap="coolwarm", vmin=-q_x_global, vmax=q_x_global, shading="nearest",
            )
            _add_interface_lines(ax_qx, interface_meta["positions"])
            ax_qx.set_xlabel("x-face position")
            ax_qx.set_ylabel("y")
            ax_qx.set_title(rf"$q_x$ at t={solver.t[run['t_peak']]:.3f}")
            qx_axes.append(ax_qx)

        fig.colorbar(pcm_x, ax=qx_axes, label=r"$q_x$", shrink=0.85)
        fig.suptitle("Boundary Flux Demo — Spatial Family Tiling")
        _save_figure(fig, save_path, "interfaces", "interface_flux_profiles", layout="constrained")


def plot_vary_interface_lhs_scatter(
    sim_params: np.ndarray,
    save_path: str | Path | None = None,
):
    """Coverage summary for the vary-interfaces sampling space.

    The LHS sampler covers (R_c, interface_x) jointly; sin amplitude and
    frequency are drawn independently per sim. This plot surfaces all four
    axes so coverage gaps are obvious at a glance.
    """
    amplitudes, frequencies, R_c = _parameter_arrays(sim_params)
    interface_x = _interface_x_array(sim_params)
    n_sims = len(interface_x)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

        ax = axes[0, 0]
        ax.scatter(interface_x, R_c, s=18, alpha=0.55, color="C0", edgecolors="none")
        ax.set_xlabel(r"Interface location $x_I$")
        ax.set_ylabel(r"Contact resistance $R_c$")
        ax.set_title(r"LHS coverage: $x_I$ vs $R_c$")
        ax.grid(True)
        ax.set_xlim(INTERFACE_X_RANGE)
        ax.set_ylim(RC_RANGE)

        ax = axes[0, 1]
        ax.scatter(amplitudes, frequencies, s=18, alpha=0.55, color="C3", edgecolors="none")
        ax.set_xlabel(r"Sin amplitude $A$")
        ax.set_ylabel(r"Sin frequency $f$")
        ax.set_title(r"Per-sim sin sampler: $A$ vs $f$")
        ax.set_yscale("log")
        ax.grid(True)

        ax = axes[0, 2]
        ax.scatter(interface_x, amplitudes, s=18, alpha=0.55, color="C2", edgecolors="none")
        ax.set_xlabel(r"Interface location $x_I$")
        ax.set_ylabel(r"Sin amplitude $A$")
        ax.set_title(r"$x_I$ vs $A$ (independence check)")
        ax.grid(True)
        ax.set_xlim(INTERFACE_X_RANGE)

        ax = axes[1, 0]
        ax.hist(interface_x, bins=20, color="C0", edgecolor="white")
        for x_lim in INTERFACE_X_RANGE:
            ax.axvline(x_lim, color="0.4", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$x_I$")
        ax.set_ylabel("Count")
        ax.set_title(r"$x_I$ marginal")
        ax.grid(True)

        ax = axes[1, 1]
        ax.hist(R_c, bins=20, color="C2", edgecolor="white")
        for x_lim in RC_RANGE:
            ax.axvline(x_lim, color="0.4", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$R_c$")
        ax.set_ylabel("Count")
        ax.set_title(r"$R_c$ marginal")
        ax.grid(True)

        ax = axes[1, 2]
        amp_norm = (amplitudes - SIN_AMP_RANGE[0]) / max(SIN_AMP_RANGE[1] - SIN_AMP_RANGE[0], 1e-12)
        log_lo, log_hi = np.log(SIN_FREQ_RANGE[0]), np.log(SIN_FREQ_RANGE[1])
        freq_norm = (np.log(frequencies) - log_lo) / max(log_hi - log_lo, 1e-12)
        bins = np.linspace(0.0, 1.0, 16)
        ax.hist(amp_norm, bins=bins, histtype="step", linewidth=1.8, label="A (linear)", color="C3")
        ax.hist(freq_norm, bins=bins, histtype="step", linewidth=1.8, label="f (log)", color="C1")
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Count")
        ax.set_title("Sin A / f marginals")
        ax.legend(loc="upper center")
        ax.grid(True)

        fig.suptitle(f"Vary-Interfaces LHS Coverage — {n_sims} simulations", fontsize=13)
        _save_figure(fig, save_path, "interfaces", "vary_interface_lhs_scatter", layout="constrained")


def plot_interface_x_breakdown(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    t_window: tuple[float, float] = (0.0, 0.20),
    n_snaps: int = 4,
    save_path: str | Path | None = None,
):
    """Tile field snapshots across interface_x quartiles.

    Rows: one representative sim per interface_x quartile (Q1..Q4).
    Cols: ``n_snaps`` snapshots in the flux-active window.
    Each row uses its own colorbar and overlays its sim's interface_x.
    """
    picks = _pick_interface_quartile_sims(sim_params)
    if not picks:
        raise ValueError("No sims available for interface_x breakdown.")

    snap_indices = _snap_indices_in_window(t_grid, t_window[0], t_window[1], n_snaps)
    nrows = len(picks)
    ncols = len(snap_indices)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.8 * ncols + 1.5, 3.6 * nrows),
            constrained_layout=True, squeeze=False,
        )
        for r, (qlabel, sid) in enumerate(picks):
            fields = trajectories[sid, snap_indices].astype(np.float32)
            vmin = float(np.min(fields))
            vmax = float(np.max(fields))
            iface = _resolve_interface_metadata(sim_params=sim_params, sim_id=sid)
            row_pcm = None
            for c, (t_idx, field) in enumerate(zip(snap_indices, fields)):
                ax = axes[r, c]
                row_pcm = _plot_field_2d(
                    ax, x_grid, y_grid, field,
                    cmap="inferno", vmin=vmin, vmax=vmax,
                    interface_positions=iface["positions"],
                )
                if r == 0:
                    ax.set_title(f"t = {t_grid[t_idx]:.3f}")
                if c != 0:
                    ax.set_ylabel("")
            row_label = (
                f"{qlabel}\nsim {sid}\n"
                f"$x_I$={iface['interface_x']:.3f}\n"
                f"$R_c$={float(sim_params[sid]['R_c']):.2f}"
            )
            axes[r, 0].annotate(
                row_label,
                xy=(-0.32, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=10,
            )
            fig.colorbar(row_pcm, ax=axes[r, :].tolist(), label="Temperature", shrink=0.85)

        fig.suptitle(r"Interface-Location Breakdown — fields per $x_I$ quartile", fontsize=14)
        _save_figure(fig, save_path, "interfaces", "interface_x_breakdown", layout="constrained")


def plot_sin_forcing_profiles(
    t_on: float = 0.0,
    t_off: float = 0.2,
    t_final: float = 0.3,
    dt: float = 0.005,
    save_path: str | Path | None = None,
):
    """Visualize the half-wave-rectified windowed-sinusoid forcing space.

    Mirrors ``plot_interface_flux_profiles`` but focused on the sin family
    alone. Shows a(t) at four (A, f) corners, the uniform spatial profile, and
    the q_L(y,t) heatmap at the high-A/high-f corner. No external data — uses
    ``build_qL`` directly like the legacy flux profile plot.
    """
    Ny = 60
    y_grid = np.linspace(0.0, 1.0, Ny)
    t = np.arange(0.0, t_final + 1e-12, dt)

    A_lo, A_hi = SIN_AMP_RANGE
    f_lo, f_hi = SIN_FREQ_RANGE
    corners = [
        ("low-A / low-f",   A_lo, f_lo, "C0"),
        ("low-A / high-f",  A_lo, f_hi, "C1"),
        ("high-A / low-f",  A_hi, f_lo, "C2"),
        ("high-A / high-f", A_hi, f_hi, "C3"),
    ]

    a_t_per_corner: list[tuple[str, np.ndarray, str]] = []
    q_high_high = None
    s_vec_uniform = None
    for label, A, f, color in corners:
        q_fn, s_vec = build_qL(
            "sin",
            dict(A=A, f=f, t_on=t_on, t_off=t_off, phase=0.0, tukey_alpha=0.5),
            "uniform", {}, y_grid,
        )
        if s_vec_uniform is None:
            s_vec_uniform = np.asarray(s_vec, dtype=np.float64)
        Q = np.array([np.asarray(q_fn(ti), dtype=np.float64) for ti in t])
        if Q.ndim == 2:
            j0 = int(np.argmax(np.abs(s_vec_uniform)))
            a_t = Q[:, j0] / max(float(s_vec_uniform[j0]), 1e-12)
        else:
            a_t = Q
        a_t_per_corner.append((label, a_t, color))
        if label == "high-A / high-f":
            if Q.ndim == 1:
                Q = np.broadcast_to(Q[:, None], (len(t), Ny)).copy()
            q_high_high = Q

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(15, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

        ax_at = fig.add_subplot(gs[0, 0])
        for label, a_t, color in a_t_per_corner:
            ax_at.plot(t, a_t, color=color, label=label, linewidth=1.6)
        ax_at.axvline(t_on, color="0.45", linestyle=":", alpha=0.8, label=f"t_on={t_on}")
        ax_at.axvline(t_off, color="0.25", linestyle=":", alpha=0.8, label=f"t_off={t_off}")
        ax_at.set_xlabel("Time")
        ax_at.set_ylabel(r"$a(t)$")
        ax_at.set_title("Windowed-sinusoid temporal forcing at (A, f) corners")
        ax_at.legend(loc="upper right", fontsize=8)
        ax_at.grid(True)

        ax_sy = fig.add_subplot(gs[0, 1])
        ax_sy.plot(y_grid, s_vec_uniform, color="C0", linewidth=2.0)
        ax_sy.set_ylim(-0.1, 1.15)
        ax_sy.set_xlabel("y")
        ax_sy.set_ylabel(r"$s(y)$")
        ax_sy.set_title("Uniform spatial profile (only family in this experiment)")
        ax_sy.grid(True)

        ax_q = fig.add_subplot(gs[1, :])
        q_abs = max(float(np.max(np.abs(q_high_high))), 1e-12)
        pcm = ax_q.pcolormesh(
            t, y_grid, q_high_high.T,
            cmap="coolwarm", vmin=-q_abs, vmax=q_abs, shading="auto",
        )
        ax_q.axvline(t_on, color="0.2", linestyle=":", alpha=0.6)
        ax_q.axvline(t_off, color="0.2", linestyle=":", alpha=0.6)
        ax_q.set_xlabel("Time")
        ax_q.set_ylabel("y")
        ax_q.set_title(rf"$q_L(y, t)$ at high-A / high-f corner (A={A_hi:.0f}, f={f_hi:.1f})")
        fig.colorbar(pcm, ax=ax_q, label=r"$q_L$", shrink=0.85)

        fig.suptitle("Sin Forcing Profiles — vary-interfaces experiment", fontsize=13)
        _save_figure(fig, save_path, "interfaces", "sin_forcing_profiles", layout="constrained")


def plot_ic_family_trajectory_breakdown(
    trajectories: np.ndarray,
    sim_params: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    t_window: tuple[float, float] = (0.0, 0.20),
    n_snaps: int = 4,
    save_path: str | Path | None = None,
):
    """Tile T - <T>(t) snapshots for one representative sim per IC family.

    Rows: IC families (``uniform_2d``, ``random_sinusoid_2d``, ``grf_2d``,
    ``hot_spot_2d``). Cols: ``n_snaps`` snapshots in the flux-active window.
    Subtracts the per-snapshot scalar mean so uniform_2d still shows IC
    texture (unlike y-perturbation, which would render uniform as blank).
    Skips families that are not present in ``sim_params``.
    """
    picks = _pick_first_per_ic_family(sim_params)
    if not picks:
        raise ValueError("No sims with a recognized IC family.")

    snap_indices = _snap_indices_in_window(t_grid, t_window[0], t_window[1], n_snaps)
    nrows = len(picks)
    ncols = len(snap_indices)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.8 * ncols + 1.5, 3.6 * nrows),
            constrained_layout=True, squeeze=False,
        )
        for r, (fam, sid) in enumerate(picks):
            fields = trajectories[sid, snap_indices].astype(np.float32)
            scalar_means = fields.mean(axis=(1, 2), keepdims=True)
            perturbations = fields - scalar_means
            abs_max = max(float(np.max(np.abs(perturbations))), 1e-12)
            iface = _resolve_interface_metadata(sim_params=sim_params, sim_id=sid)
            row_pcm = None
            for c, (t_idx, field) in enumerate(zip(snap_indices, perturbations)):
                ax = axes[r, c]
                row_pcm = _plot_field_2d(
                    ax, x_grid, y_grid, field,
                    cmap="coolwarm", vmin=-abs_max, vmax=abs_max,
                    interface_positions=iface["positions"],
                )
                if r == 0:
                    ax.set_title(f"t = {t_grid[t_idx]:.3f}")
                if c != 0:
                    ax.set_ylabel("")
            row_label = (
                f"{fam}\nsim {sid}\n"
                f"$x_I$={iface['interface_x']:.3f}"
            )
            axes[r, 0].annotate(
                row_label,
                xy=(-0.32, 0.5), xycoords="axes fraction",
                ha="right", va="center", fontsize=10,
            )
            fig.colorbar(row_pcm, ax=axes[r, :].tolist(), label=r"$T - \langle T \rangle$", shrink=0.85)

        fig.suptitle("IC-Family Breakdown — per-snapshot mean removed", fontsize=14)
        _save_figure(fig, save_path, "interfaces", "ic_family_trajectory_breakdown", layout="constrained")


def plot_vary_interface_dataset_summary(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    config: dict,
    representative_sim_id: int | None = None,
    save_path: str | Path | None = None,
):
    """Report-style overview of the vary-interfaces dataset.

    Mirrors ``plot_dataset_summary`` but surfaces the new axes:
    interface_x, sin A/f histograms, and IC-family counts.
    """
    config = _resolve_plot_config(config)
    amplitudes, frequencies, R_c = _parameter_arrays(sim_params)
    interface_x = _interface_x_array(sim_params)
    sim_id = (
        _select_representative_sim_id(sim_params) if representative_sim_id is None
        else int(representative_sim_id)
    )
    iface = _resolve_interface_metadata(
        config=config, sim_params=sim_params, sim_id=sim_id,
    )
    coverage = _lead_time_coverage_counts(trajectories, x_grid, y_grid, t_grid, sim_params, config)

    ic_families = np.array(
        [str(p.get("ic_family", "?")) for p in sim_params], dtype=object,
    )
    fam_counts = {fam: int(np.sum(ic_families == fam)) for fam in _IC_FAMILY_ORDER}

    max_lead = max(
        float(coverage["train"].max()) if len(coverage["train"]) > 0 else 0.0,
        float(coverage["val"].max()) if len(coverage["val"]) > 0 else 0.0,
        float(coverage["test"].max()) if len(coverage["test"]) > 0 else 0.0,
        float(t_grid[-1] - t_grid[0]),
    )
    lead_bins = np.linspace(0.0, max_lead, 16)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

        temp_fields = np.stack([trajectories[sim_id, 0], trajectories[sim_id, -1]])
        temp_vmin = float(np.min(temp_fields))
        temp_vmax = float(np.max(temp_fields))
        pcm = _plot_field_2d(
            axes[0, 0], x_grid, y_grid, trajectories[sim_id, 0],
            vmin=temp_vmin, vmax=temp_vmax,
            interface_positions=iface["positions"],
        )
        axes[0, 0].set_title(f"Representative IC — sim {sim_id}, $x_I$={iface['interface_x']:.3f}")

        _plot_field_2d(
            axes[0, 1], x_grid, y_grid, trajectories[sim_id, -1],
            vmin=temp_vmin, vmax=temp_vmax,
            interface_positions=iface["positions"],
        )
        axes[0, 1].set_title(f"Representative Final — t={t_grid[-1]:.3f}")

        axes[0, 2].hist(interface_x, bins=20, color="C0", edgecolor="white")
        for x_lim in INTERFACE_X_RANGE:
            axes[0, 2].axvline(x_lim, color="0.4", linestyle=":", linewidth=1.0)
        axes[0, 2].set_xlabel(r"$x_I$")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].set_title(r"Interface location $x_I$")
        axes[0, 2].grid(True)

        axes[1, 0].hist(R_c, bins=20, color="C2", edgecolor="white")
        for x_lim in RC_RANGE:
            axes[1, 0].axvline(x_lim, color="0.4", linestyle=":", linewidth=1.0)
        axes[1, 0].set_xlabel(r"$R_c$")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title(r"Contact resistance $R_c$")
        axes[1, 0].grid(True)

        ax_a = axes[1, 1]
        ax_a.hist(amplitudes, bins=20, color="C3", alpha=0.55, edgecolor="white", label="A (linear)")
        ax_a.set_xlabel(r"$A$ (linear)")
        ax_a.set_ylabel("Count (A)", color="C3")
        ax_a.tick_params(axis="y", labelcolor="C3")
        ax_f = ax_a.twiny()
        log_lo, log_hi = np.log10(SIN_FREQ_RANGE[0]), np.log10(SIN_FREQ_RANGE[1])
        log_bins = np.logspace(log_lo, log_hi, 20)
        ax_f.hist(frequencies, bins=log_bins, color="C1", alpha=0.55,
                  edgecolor="white", label="f (log)")
        ax_f.set_xscale("log")
        ax_f.set_xlabel(r"$f$ (log)", color="C1")
        ax_f.tick_params(axis="x", labelcolor="C1")
        ax_a.set_title("Sin amplitude / frequency")
        ax_a.grid(True)

        labels = list(fam_counts.keys())
        counts = [fam_counts[f] for f in labels]
        bar_colors = ["C0", "C1", "C2", "C3"]
        axes[1, 2].bar(labels, counts, color=bar_colors, edgecolor="white")
        axes[1, 2].set_ylabel("Sim count")
        axes[1, 2].set_title("IC family counts")
        axes[1, 2].tick_params(axis="x", rotation=20)
        axes[1, 2].grid(True, axis="y")
        inset = axes[1, 2].inset_axes([0.55, 0.55, 0.43, 0.40])
        for label, values, color in [
            ("Train", coverage["train"], "C0"),
            ("Val", coverage["val"], "C2"),
            ("Test", coverage["test"], "C3"),
        ]:
            if len(values) > 0:
                inset.hist(values, bins=lead_bins, histtype="step", linewidth=1.2,
                           label=label, color=color)
        inset.set_title("Lead time", fontsize=8)
        inset.tick_params(axis="both", labelsize=7)
        inset.legend(loc="upper right", fontsize=6)

        fig.colorbar(pcm, ax=axes[0, :2].tolist(), label="Temperature", shrink=0.92)

        fig.suptitle(
            f"Vary-Interfaces Dataset Summary — {len(sim_params)} sims",
            fontsize=14,
        )
        _save_figure(fig, save_path, "interfaces", "vary_interface_dataset_summary", layout="constrained")
