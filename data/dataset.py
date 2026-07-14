import copy
from collections.abc import Callable
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    SPATIAL_BUILDERS,
    PATCH_W_RANGE,
    GAUSS_SIGMA_RANGE,
    TRIANGLE_ELL_RANGE,
    TEMPORAL_BUILDERS,
    TEMPORAL_FAMILY_ORDER,
    integrate_temporal_bins_signed,
    default_ramp_seconds,
    FORCING_BINS,
    SIN_AMP_RANGE,
)
from problems.base import ProblemSpec
from problems.registry import get_problem


def problem_from_config(config: dict) -> ProblemSpec:
    """Resolve the active ProblemSpec from a loaded config's benchmark section.

    Reads both the ``name`` and ``representation`` axes (the latter defaults to
    ``temporal_encoder``) so the resolved spec is the single source of truth for
    all representation-determined tensor dims.
    """
    benchmark = config.get("benchmark") or {}
    name = str(benchmark.get("name", "forcing"))
    representation = str(benchmark.get("representation", "temporal_encoder"))
    spec = get_problem(name, representation)
    if "rc_channel_mode" in benchmark:
        spec.rc_channel_mode = str(benchmark.get("rc_channel_mode"))
    if "rc_ell" in benchmark:
        spec.rc_ell = float(benchmark.get("rc_ell"))
    return spec


RC_RANGE = (0.05, 1.0)
T_EPS = 1e-6  # epsilon for temperature normalization

# Static conditioning layout (23 dims):
#   [0:3]    base:             t_bar_norm, t_s_norm, R_c_norm
#   [3:7]    spatial onehot:   uniform, patch, gaussian, triangle
#   [7:11]   spatial params:   y_c_norm, w_norm, sigma_y_norm, ell_norm
#   [11:15]  temporal onehot:  sin, exp, pulse_train, exp_train
#   [15:23]  forcing summary:  S1..S8 (signed/abs/pos/neg impulse, mean, RMS, peak, final)
#
# The forcing summary block gives the conditioning MLP global, interval-level
# scalar descriptors of a(t) over [t_s, t_j] — complementary to the spatial
# Q_y_bins channels and the learned TemporalForcingEncoder embedding.
SPATIAL_FAMILY_ORDER = ("uniform", "patch", "gaussian", "triangle")

_BASE_DIM             = 3
_SPATIAL_ONEHOT_DIM   = len(SPATIAL_FAMILY_ORDER)
_SPATIAL_PARAM_DIM    = 4
_TEMPORAL_ONEHOT_DIM  = len(TEMPORAL_FAMILY_ORDER)
_FORCING_SUMMARY_DIM  = 8
COND_STATIC_DIM = (
    _BASE_DIM + _SPATIAL_ONEHOT_DIM + _SPATIAL_PARAM_DIM
    + _TEMPORAL_ONEHOT_DIM + _FORCING_SUMMARY_DIM
)  # 23

# Defaults for the temporal forcing branch.
TEMPORAL_SAMPLES   = 64
TEMPORAL_TOKEN_DIM = 5
A_AMP_REF          = 300.0    # matches SIN_AMP_RANGE[1] from boundary_forcing

# log-uniform sigma_y is min-max normalized in log-space so coverage matches
# the sampler's log-uniform distribution.
_LOG_SIGMA_LO = float(np.log(GAUSS_SIGMA_RANGE[0]))
_LOG_SIGMA_HI = float(np.log(GAUSS_SIGMA_RANGE[1]))


def build_cond_vector(t_bar_norm: float, t_s_norm: float, R_c: float,
                      spatial_family: str, spatial_params: dict,
                      temporal_family: str,
                      forcing_summary: np.ndarray) -> np.ndarray:
    """Assemble the 23-dim static conditioning vector. Used by both the
    dataset and the inference plotting paths so they cannot drift apart.

    `R_c` is the raw contact resistance (not pre-normalized) — normalization
    happens here once. `forcing_summary` is the (_FORCING_SUMMARY_DIM,) vector
    built by `build_forcing_summary` from the same a(t) samples that feed
    `forcing_seq`; required (no default) so callers cannot silently ship a
    zero-padded conditioning vector.
    """
    R_c_norm = (R_c - RC_RANGE[0]) / (RC_RANGE[1] - RC_RANGE[0])
    base = np.array([t_bar_norm, t_s_norm, R_c_norm], dtype=np.float32)

    spatial_oh = np.zeros(_SPATIAL_ONEHOT_DIM, dtype=np.float32)
    spatial_oh[SPATIAL_FAMILY_ORDER.index(spatial_family)] = 1.0
    y_c_norm = w_norm = sigma_y_norm = ell_norm = 0.0
    if spatial_family == "patch":
        y_c_norm = float(spatial_params["y_c"])
        w_norm = (spatial_params["w"] - PATCH_W_RANGE[0]) / (PATCH_W_RANGE[1] - PATCH_W_RANGE[0])
    elif spatial_family == "gaussian":
        y_c_norm = float(spatial_params["y_c"])
        sigma_y_norm = (np.log(spatial_params["sigma_y"]) - _LOG_SIGMA_LO) / (_LOG_SIGMA_HI - _LOG_SIGMA_LO)
    elif spatial_family == "triangle":
        y_c_norm = float(spatial_params["y_c"])
        ell_norm = (spatial_params["ell"] - TRIANGLE_ELL_RANGE[0]) / (TRIANGLE_ELL_RANGE[1] - TRIANGLE_ELL_RANGE[0])
    spatial_p = np.array([y_c_norm, w_norm, sigma_y_norm, ell_norm], dtype=np.float32)

    temporal_oh = np.zeros(_TEMPORAL_ONEHOT_DIM, dtype=np.float32)
    temporal_oh[TEMPORAL_FAMILY_ORDER.index(temporal_family)] = 1.0

    fs = np.asarray(forcing_summary, dtype=np.float32).reshape(-1)
    if fs.shape[0] != _FORCING_SUMMARY_DIM:
        raise ValueError(
            f"forcing_summary must have shape ({_FORCING_SUMMARY_DIM},), got {fs.shape}"
        )

    return np.concatenate([base, spatial_oh, spatial_p, temporal_oh, fs]).astype(np.float32)


def _sample_a(q, t_s: float, t_j: float, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a(t) at M uniform samples spanning [t_s, t_j] (both endpoints).

    Returns (t_samples, a_m), both float32 arrays of shape (M,). Endpoint
    inclusion matters: the final sample is a(t_j), which the forcing-summary
    `S8` reads as the most recent forcing value.
    """
    t_bar = float(t_j) - float(t_s)
    r = np.linspace(0.0, 1.0, int(M), dtype=np.float32)
    t_samples = (float(t_s) + r * t_bar).astype(np.float32)

    try:
        a_m = np.asarray(q(t_samples), dtype=np.float32)
    except Exception:
        a_m = np.array([q(float(tk)) for tk in t_samples], dtype=np.float32)
    if a_m.shape == ():
        a_m = np.full_like(t_samples, float(a_m), dtype=np.float32)
    elif a_m.shape != t_samples.shape:
        a_m = np.array([q(float(tk)) for tk in t_samples], dtype=np.float32)
    return t_samples, a_m


def _forcing_seq_from_samples(
    t_samples: np.ndarray,
    a_m: np.ndarray,
    t_s: float,
    t_j: float,
    t_final: float,
    A_amp_ref: float,
    A_cum_ref: float,
) -> np.ndarray:
    t_bar = float(t_j) - float(t_s)
    M = t_samples.shape[0]
    r = np.linspace(0.0, 1.0, M, dtype=np.float32)

    # Signed cumulative trapezoid over the M sample grid. NO clamping — solver
    # uses signed flux, so the encoder must see the same signed values.
    A_cum = np.empty_like(a_m)
    A_cum[0] = 0.0
    A_cum[1:] = np.cumsum(0.5 * (a_m[1:] + a_m[:-1]) * np.diff(t_samples))

    tok0 = r
    tok1 = a_m / np.float32(A_amp_ref)
    tok2 = A_cum / np.float32(A_cum_ref)
    tok3 = ((1.0 - r) * t_bar / float(t_final)).astype(np.float32)
    tok4 = (t_samples / float(t_final)).astype(np.float32)
    return np.stack([tok0, tok1, tok2, tok3, tok4], axis=-1).astype(np.float32)


def build_forcing_seq(
    q,
    t_s: float,
    t_j: float,
    t_final: float,
    M: int = TEMPORAL_SAMPLES,
    A_amp_ref: float = A_AMP_REF,
    A_cum_ref: float | None = None,
) -> np.ndarray:
    """Sample a(t) on M points over [t_s, t_j] and build the (M, 5) token array.

    Tokens:
        [0] r_m = m / (M-1)
        [1] a_m / A_amp_ref
        [2] A_m / A_cum_ref            (signed trapezoidal cumulative)
        [3] (t_j - t_m) / t_final      (time to target)
        [4] t_m / t_final              (absolute normalized time)

    `q` is a callable returning a(t). We attempt vectorized evaluation and
    fall back to scalar iteration if the callable does not broadcast.
    """
    if A_cum_ref is None:
        A_cum_ref = float(A_amp_ref) * float(t_final)

    t_samples, a_m = _sample_a(q, t_s, t_j, int(M))
    return _forcing_seq_from_samples(t_samples, a_m, t_s, t_j, t_final, A_amp_ref, A_cum_ref)


def build_forcing_summary(
    a_vals: np.ndarray,
    t_vals: np.ndarray,
    t_s: float,
    t_j: float,
    t_final: float,
    A_amp_ref: float = A_AMP_REF,
) -> np.ndarray:
    """Global interval-level descriptors of a(t) over [t_s, t_j].

    Returns float32 shape (_FORCING_SUMMARY_DIM,):
        S1 = ∫a(t)dt / (A_ref * t_final)           signed impulse
        S2 = ∫|a(t)|dt / (A_ref * t_final)         absolute impulse
        S3 = ∫max(a,0)dt / (A_ref * t_final)       positive impulse
        S4 = ∫max(-a,0)dt / (A_ref * t_final)      negative impulse magnitude
        S5 = mean(a) / A_ref                       mean forcing
        S6 = sqrt(<a^2>) / A_ref                   RMS forcing
        S7 = max|a| / A_ref                         peak abs forcing
        S8 = a(t_j) / A_ref                         final forcing value

    `t_vals` must include both endpoints (t_s, t_j) so that `S8 = a_vals[-1]`
    is genuinely a(t_j). Caller is expected to share t_vals/a_vals with
    `build_forcing_seq` via `_sample_a` to guarantee both descriptors see the
    same underlying samples.
    """
    A_ref = float(A_amp_ref)
    dt_interval = max(float(t_j) - float(t_s), 1e-8)
    denom_impulse = A_ref * float(t_final)

    if hasattr(np, "trapezoid"):
        trapz = np.trapezoid
    else:
        trapz = np.trapz
    I_signed = float(trapz(a_vals, t_vals))
    I_abs    = float(trapz(np.abs(a_vals), t_vals))
    I_pos    = float(trapz(np.clip(a_vals, 0.0, None), t_vals))
    I_neg    = float(trapz(np.clip(-a_vals, 0.0, None), t_vals))
    I_sq     = float(trapz(a_vals ** 2, t_vals))

    S1 = I_signed / denom_impulse
    S2 = I_abs    / denom_impulse
    S3 = I_pos    / denom_impulse
    S4 = I_neg    / denom_impulse
    S5 = (I_signed / dt_interval) / A_ref
    S6 = np.sqrt(max(I_sq / dt_interval, 0.0)) / A_ref
    S7 = float(np.max(np.abs(a_vals))) / A_ref
    S8 = float(a_vals[-1]) / A_ref

    return np.array([S1, S2, S3, S4, S5, S6, S7, S8], dtype=np.float32)

# --------- SNAPSHOT PAIR DATASET ---------

class SnapshotPairDataset(Dataset):
    """All-to-all snapshot-pair dataset for time-conditioned FNO.

    Each sample is a (input, target) pair: given the temperature field at
    time t_s, predict the field at a future time t_j > t_s.

    Uses **global normalization**: T̃ = (T - μ_global) / σ_global, where
    μ_global and σ_global are computed once from the training set. This
    preserves absolute temperature scale in the field (unlike per-sample
    z-score) so the model can learn BC- and IC-dependent dynamics.

    When n_snapshots is provided, uniformly subsamples that many time steps
    from the full trajectory and enumerates all possible pairs per
    simulation.  Pairs are sorted by lead time to support curriculum slicing.

    Returns 5-tuple: (spatial, cond_static, forcing_seq, Y, T_stats)
        spatial      : (Nx, Ny, 20)       — [T̃_source, x_norm, y_norm, s_y, Q_y_bin_0, ..., Q_y_bin_15]
        cond_static  : (23,)              — see COND_STATIC_DIM layout above
        forcing_seq  : (M, 5)             — token-encoded a(t) over [t_s, t_j]
        Y            : (Nx, Ny, 1)        — T̃_target (globally normalized)
        T_stats      : (2,)               — [μ_global, σ_global] for denormalization
    """

    def __init__(
        self,
        trajectories: np.ndarray,
        t_grid: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        sim_ids: np.ndarray,
        sim_params: np.ndarray,
        mu_global: float,
        sigma_global: float,
        n_snapshots: int | None = None,
        noise_std: float = 0.0,
        dt: float | None = None,
        t_final: float | None = None,
        time_norm_horizon: float | None = None,
        ramp_seconds: float | None = None,
        temporal_samples: int = TEMPORAL_SAMPLES,
        problem: ProblemSpec | None = None,
        max_target_time: float | None = None,
        max_lead_time: float | None = None,
    ):
        self.problem = problem if problem is not None else get_problem("forcing")
        self.trajectories = trajectories  # (num_sims, Nt, Nx, Ny)
        self.sim_params = sim_params
        self.t_grid = t_grid.astype(np.float32)
        self.x_grid = x_grid.astype(np.float32)
        self.y_grid = y_grid.astype(np.float32)
        self.sim_ids = sim_ids.astype(np.int64)
        self.mu_global = np.float32(mu_global)
        self.sigma_global = np.float32(sigma_global)
        self.noise_std = noise_std

        self.num_sims, self.Nt, self.Nx, self.Ny = trajectories.shape

        # Defaults derived from t_grid match the generation pipeline whenever
        # the saved t_grid is uniform. `dt` is retained for downstream callers
        # that still want the solver step size; the forcing branch itself
        # consumes a(t) directly through TEMPORAL_BUILDERS, not via dt/tau.
        self.t_final = float(t_final) if t_final is not None else float(self.t_grid[-1])
        self.dt = float(dt) if dt is not None else float(self.t_grid[1] - self.t_grid[0])

        # TRAIN-split pair truncation (long-lead OOD adaptation). Both None =
        # legacy full-pair set. Applied in _build_all_pairs with a +eps tolerance
        # so a pair landing exactly on the cutoff is kept, not dropped by float
        # noise. Set only on the train dataset by create_dataloaders.
        self.max_target_time = (
            float(max_target_time) if max_target_time is not None else None
        )
        self.max_lead_time = (
            float(max_lead_time) if max_lead_time is not None else None
        )

        # Trained normalization horizon for temporal model-input features ONLY.
        # Defaults to the physical horizon so non-OOD behavior is identical; an
        # OOD time-extrapolation set passes the trained horizon (e.g. 0.30) while
        # t_final stays the true extended horizon (0.45), so a target past 0.30
        # yields lead/time features > 1.0. This NEVER drives pair construction,
        # validity filters, or forcing generation -- those keep reading t_final.
        self.time_norm_horizon = (
            float(time_norm_horizon) if time_norm_horizon is not None
            else self.t_final
        )

        # Physical startup-ramp width for q_L(0)=0. Read from the dataset's stored
        # value when available so the model's forcing conditioning uses the exact
        # ramp the solver used; fall back to the dt-derived default for datasets
        # generated before ramp_seconds was persisted.
        self.ramp_seconds = (
            float(ramp_seconds) if ramp_seconds is not None
            else default_ramp_seconds(self.dt)
        )

        # Number of a(t) samples the temporal branch consumes. Benchmark-
        # specific caches (lazy callables, spatial profiles, normalization
        # references) are populated by `self.problem.setup_dataset(self)` below.
        self.temporal_samples = int(temporal_samples)

        # Normalized spatial coordinates (fixed for all samples)
        self.x_norm = (
            (self.x_grid - self.x_grid[0]) / (self.x_grid[-1] - self.x_grid[0])
        ).astype(np.float32)

        self.y_norm = (
            (self.y_grid - self.y_grid[0]) / (self.y_grid[-1] - self.y_grid[0])
        ).astype(np.float32)

        # Broadcast to (Nx, Ny) so they can be stacked with T_source at sample time
        self.X_norm = np.broadcast_to(self.x_norm[:, None], (self.Nx, self.Ny)).astype(np.float32)
        self.Y_norm = np.broadcast_to(self.y_norm[None, :], (self.Nx, self.Ny)).astype(np.float32)

        self.problem.setup_dataset(self)

        # Determine which time indices to use
        if n_snapshots is not None and n_snapshots < self.Nt:
            self.t_indices = np.round(
                np.linspace(0, self.Nt - 1, n_snapshots)
            ).astype(int)
        else:
            self.t_indices = np.arange(self.Nt)

        # Build ALL valid (sim_id, s_idx, j_idx) pairs, sorted by lead time
        self._build_all_pairs()
        # active pairs represents the amount of pairs actually seen during a certain epoch
        self._active_len = len(self._pairs)

    def _build_all_pairs(self):
        """Enumerate all valid (sim_id, s, j) pairs from subsampled time indices.

        When ``max_lead_time`` / ``max_target_time`` are set (train-split
        truncation) a pair is skipped if its lead ``t_j - t_s`` or absolute
        target ``t_j`` exceeds the cap. The comparison uses a ``+eps = 0.5*dt``
        tolerance so a pair landing exactly on the cutoff survives float noise
        (and future dt changes) rather than being dropped.
        """
        eps = 0.5 * self.dt
        pairs = []
        for sim_id in self.sim_ids:
            for i, s_idx in enumerate(self.t_indices):
                for j_idx in self.t_indices[i + 1:]:
                    lead = float(self.t_grid[j_idx] - self.t_grid[s_idx])
                    if (self.max_lead_time is not None
                            and lead > self.max_lead_time + eps):
                        continue
                    if (self.max_target_time is not None
                            and float(self.t_grid[j_idx]) > self.max_target_time + eps):
                        continue
                    pairs.append((int(sim_id), int(s_idx), int(j_idx), lead))
        # Sort by lead time for curriculum slicing
        pairs.sort(key=lambda p: p[3])
        self._pairs = [(p[0], p[1], p[2]) for p in pairs]
        self._lead_times = np.array([p[3] for p in pairs], dtype=np.float32)

    def set_curriculum_fraction(self, frac: float):
        """Expose only pairs with lead time <= frac * max_lead_time.
        frac=1.0 means all pairs (no curriculum restriction)."""
        if frac >= 1.0:
            self._active_len = len(self._pairs)
        else:
            max_lead = self._lead_times[-1]
            cutoff = frac * max_lead
            self._active_len = max(1, int(np.searchsorted(self._lead_times, cutoff, side='right')))

    def __len__(self):
        return self._active_len

    def __getitem__(self, idx):
        sim_id, s, j = self._pairs[idx]
        item = self.problem.build_item(self, int(sim_id), int(s), int(j))
        return {k: torch.from_numpy(v) for k, v in item.items()}


def collate_fn(batch: list[dict]) -> dict:
    """Stack a list of per-item dicts into a batched dict.

    Each benchmark's items share the same keys (optional `forcing_seq` is
    present for every item of a benchmark that declares `has_forcing_seq`, and
    absent for every item otherwise), so stacking the first item's keys is
    sufficient.
    """
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def _dataset_with_pairs(dataset: SnapshotPairDataset, pairs: list[tuple[int, int, int]]) -> SnapshotPairDataset:
    out = copy.copy(dataset)
    out._pairs = list(pairs)
    out._lead_times = np.array(
        [float(out.t_grid[j] - out.t_grid[s]) for _, s, j in out._pairs],
        dtype=np.float32,
    )
    out._active_len = len(out._pairs)
    # Fresh worker-local cache on the copy so train/val don't share a mutable dict
    # (only benchmarks with a lazy a(t) callable cache populate this).
    if hasattr(out, "_q_callables"):
        out._q_callables = {}
    return out


def split_pairs_within_sims(
    dataset: SnapshotPairDataset,
    val_pair_frac: float = 0.1,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    val_indices = set()
    for sim_id in dataset.sim_ids:
        sim_pair_indices = [i for i, pair in enumerate(dataset._pairs) if pair[0] == int(sim_id)]
        n_val = int(round(len(sim_pair_indices) * val_pair_frac))
        if val_pair_frac > 0.0 and len(sim_pair_indices) > 0:
            n_val = max(1, n_val)
        if n_val > 0:
            chosen = rng.choice(sim_pair_indices, size=n_val, replace=False)
            val_indices.update(int(i) for i in chosen)

    train_pairs = [pair for i, pair in enumerate(dataset._pairs) if i not in val_indices]
    val_pairs = [pair for i, pair in enumerate(dataset._pairs) if i in val_indices]
    train_dataset = _dataset_with_pairs(dataset, train_pairs)
    val_dataset = _dataset_with_pairs(dataset, val_pairs)
    val_dataset.noise_std = 0.0
    return train_dataset, val_dataset


def long_lead_pairs(dataset: SnapshotPairDataset) -> list[tuple[int, int, int]]:
    """Return only full-span pairs: source at the first snapshot (t=0) and target
    at the last snapshot (t_final). One per sim; the maximum-lead pair."""
    s_first = int(dataset.t_indices[0])
    j_last = int(dataset.t_indices[-1])
    return [
        (int(sim), int(s), int(j))
        for (sim, s, j) in dataset._pairs
        if int(s) == s_first and int(j) == j_last
    ]


# ---- OOD time pair protocols -------------------------------------------------

# Names accepted by `build_protocol_pairs`. Each protocol pins a SINGLE
# source-state definition so a protocol curve never mixes two prediction tasks
# (aggregation in scripts/inspect_ood.py never pools across differing
# source_time_actual / lead_time_actual).
PROTOCOLS = ("fixed_initial", "anchored_from_horizon", "ood_local_fixed_lead")

# Default fixed lead (in physical time units) for `ood_local_fixed_lead`.
OOD_LOCAL_LEAD = 0.05


def _match_snapshot(t_indices: np.ndarray, t_grid: np.ndarray, requested: float,
                    atol: float) -> tuple[int, float]:
    """Return the (snapshot index, actual time) in `t_indices` whose `t_grid`
    value is nearest `requested`, or raise if none lies within `atol`.

    Records both requested and actual at the call site so a snapshot stride that
    misses a pinned anchor (e.g. exactly 0.30) is caught rather than silently
    shifting the anchor to a neighbouring snapshot.
    """
    times = t_grid[t_indices]
    k = int(np.argmin(np.abs(times - requested)))
    actual = float(times[k])
    if abs(actual - requested) > atol:
        raise ValueError(
            f"no snapshot within atol={atol:g} of requested time {requested:g}; "
            f"nearest is {actual:g}. Check the save stride / target_times."
        )
    return int(t_indices[k]), actual


def build_protocol_pairs(
    dataset: SnapshotPairDataset,
    protocols: list[str],
    target_times: list[float],
    *,
    horizon: float | None = None,
    local_lead: float = OOD_LOCAL_LEAD,
    atol: float | None = None,
) -> list[dict]:
    """Build OOD time-extrapolation pairs tagged by protocol.

    Generalizes `long_lead_pairs` into the three protocols from the OOD plan,
    each emitted for every test sim and tagged with its `protocol`,
    `source_time_*`, `target_time_*`, and `lead_time_actual`:

    - ``fixed_initial``        : source t_s = 0.0, target across all `target_times`
      (the only protocol that evaluates ID-reference targets).
    - ``anchored_from_horizon``: source t_s = `horizon`, target over OOD targets
      (`target_times` strictly above `horizon`).
    - ``ood_local_fixed_lead`` : source t_s > `horizon`, target t_s + `local_lead`
      (fixed lead), one pair per eligible source snapshot.

    Times are matched to snapshots by tolerance (`atol`, default a quarter of the
    snapshot spacing); both requested and realized times are recorded.
    """
    horizon = float(dataset.time_norm_horizon if horizon is None else horizon)
    t_grid = np.asarray(dataset.t_grid)
    t_indices = np.asarray(dataset.t_indices)
    if atol is None:
        times = t_grid[t_indices]
        spacing = float(np.min(np.diff(times))) if len(times) > 1 else 1.0
        atol = 0.25 * spacing

    out: list[dict] = []
    for name in protocols:
        if name not in PROTOCOLS:
            raise ValueError(f"unknown protocol {name!r}; expected one of {PROTOCOLS}.")

        if name == "fixed_initial":
            s_idx, s_act = _match_snapshot(t_indices, t_grid, 0.0, atol)
            targets = list(target_times)
        elif name == "anchored_from_horizon":
            s_idx, s_act = _match_snapshot(t_indices, t_grid, horizon, atol)
            targets = [t for t in target_times if t > horizon + atol]
        else:  # ood_local_fixed_lead
            targets = None  # source-driven; handled below

        if name in ("fixed_initial", "anchored_from_horizon"):
            for t_req in targets:
                j_idx, t_act = _match_snapshot(t_indices, t_grid, t_req, atol)
                if j_idx <= s_idx:
                    continue
                lead = t_act - s_act
                for sim in dataset.sim_ids:
                    out.append({
                        "sim_id": int(sim), "s_idx": s_idx, "j_idx": j_idx,
                        "protocol": name,
                        "source_time_requested": float(0.0 if name == "fixed_initial" else horizon),
                        "source_time_actual": s_act,
                        "target_time_requested": float(t_req),
                        "target_time_actual": t_act,
                        "lead_time_actual": float(lead),
                    })
        else:
            times = t_grid[t_indices]
            for k, t_s in enumerate(times):
                if t_s <= horizon + atol:
                    continue
                t_req = float(t_s) + local_lead
                try:
                    j_idx, t_act = _match_snapshot(t_indices, t_grid, t_req, atol)
                except ValueError:
                    continue
                s_idx = int(t_indices[k])
                if j_idx <= s_idx:
                    continue
                lead = t_act - float(t_s)
                for sim in dataset.sim_ids:
                    out.append({
                        "sim_id": int(sim), "s_idx": s_idx, "j_idx": j_idx,
                        "protocol": name,
                        "source_time_requested": float(t_s),
                        "source_time_actual": float(t_s),
                        "target_time_requested": t_req,
                        "target_time_actual": t_act,
                        "lead_time_actual": float(lead),
                    })
    return out


def apply_protocol_pairs(dataset: SnapshotPairDataset, tagged: list[dict]) -> None:
    """Install tagged protocol pairs onto `dataset`, replacing `_pairs`.

    Stores the parallel tag list as `dataset._pair_tags` (read by
    `write_test_records`), keeps `_pairs`/`_lead_times`/`_active_len` consistent
    with the existing pair machinery, and disables curriculum slicing.
    """
    if not tagged:
        raise ValueError("apply_protocol_pairs: empty pair list.")
    dataset._pairs = [(t["sim_id"], t["s_idx"], t["j_idx"]) for t in tagged]
    dataset._lead_times = np.array(
        [t["lead_time_actual"] for t in tagged], dtype=np.float32
    )
    dataset._pair_tags = list(tagged)
    dataset._active_len = len(dataset._pairs)


def one_step_physics_view(dataset: SnapshotPairDataset) -> SnapshotPairDataset:
    """Restrict `dataset` to consecutive one-step pairs (`j == s+1` in t_indices):
    O(Nt) pairs per sim instead of the O(Nt^2) all-to-all corpus. This is the W1
    physics-residual supply.

    The caller MUST pass a `save_stride=1` dataset with every snapshot retained
    (`n_snapshots=None`) so that adjacent `t_indices` are exactly one solver step
    `dt` apart — the CN-exact spacing the interior residual assumes. With a
    subsampled or strided grid the "one-step" pair would span several solver steps
    and the residual would be wrong.
    """
    t_idx = dataset.t_indices
    pairs = [
        (int(sim_id), int(t_idx[i]), int(t_idx[i + 1]))
        for sim_id in dataset.sim_ids
        for i in range(len(t_idx) - 1)
    ]
    return _dataset_with_pairs(dataset, pairs)


# --------- LOAD RAW SIM. DATA --------

def load_sim_data(
    sim_traj_path: str,
    x_grid_path: str,
    y_grid_path: str,
    t_grid_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    trajectories = np.load(sim_traj_path, mmap_mode="r")  # (num_sims, Nt, Nx, Ny)
    x_grid = np.load(x_grid_path) # (Nx,)
    y_grid = np.load(y_grid_path) # (Ny,)
    t_grid = np.load(t_grid_path) # (Nt,)

    # size safeguards
    if trajectories.ndim != 4:
        raise ValueError(f"Expected trajectories with ndim=4, got shape {trajectories.shape}")
    if x_grid.ndim != 1:
        raise ValueError(f"Expected x_grid with ndim=1, got shape {x_grid.shape}")
    if y_grid.ndim != 1:
        raise ValueError(f"Expected y_grid with ndim=1, got shape {y_grid.shape}")
    if t_grid.ndim != 1:
        raise ValueError(f"Expected t_grid with ndim=1, got shape {t_grid.shape}")

    # length safeguards
    n_sims, Nt_total, Nx, Ny = trajectories.shape
    if x_grid.shape[0] != Nx:
        raise ValueError(f"x_grid length {x_grid.shape[0]} does not match Nx={Nx}")
    if y_grid.shape[0] != Ny:
        raise ValueError(f"y_grid length {y_grid.shape[0]} does not match Ny={Ny}")
    if t_grid.shape[0] != Nt_total:
        raise ValueError(f"t_grid length {t_grid.shape[0]} does not match Nt_total={Nt_total}")

    return trajectories, x_grid, y_grid, t_grid


def load_solver_dt(t_grid_path: str | Path) -> float | None:
    """Return the solver dt saved alongside the trajectories, or None if absent.

    Looks for `dt.npy` in the same directory as `t_grid_path`. Needed because
    `t_grid[1] - t_grid[0]` is the saved snapshot cadence (= solver dt × save_stride),
    not the solver dt that the boundary-forcing samplers used to set tau / dt_n
    bounds. Mismatched dt produces cond-vec slots outside [0, 1].
    """
    dt_path = Path(t_grid_path).parent / "dt.npy"
    if not dt_path.exists():
        return None
    return float(np.load(dt_path))


def load_ramp_seconds(t_grid_path: str | Path) -> float | None:
    """Return the stored startup-ramp width saved with the trajectories, or None.

    Looks for `ramp_seconds.npy` in the same directory as `t_grid_path`. The
    resolved physical ramp is persisted at generation time so the model's
    forcing conditioning reproduces the exact q_L(t) the solver integrated,
    independent of `dt`. Absent for datasets generated before the ramp was added;
    callers fall back to `default_ramp_seconds(dt)`.
    """
    ramp_path = Path(t_grid_path).parent / "ramp_seconds.npy"
    if not ramp_path.exists():
        return None
    return float(np.load(ramp_path))


# ------- SLICE ALL SIMS INTO TRAIN/VAL/TEST SPLITS -------

def split_sim_ids(
    num_sims: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    ids = np.arange(num_sims)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_train = int(num_sims * train_frac)
    n_val = int(num_sims * val_frac)
    n_test = num_sims - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:n_train + n_val + n_test]

    return train_ids, val_ids, test_ids


def split_sim_ids_stratified(
    labels,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family-balanced train/val/test split for varying-IC datasets.

    ``labels`` is a per-sim sequence of category strings (here the IC family).
    Sims are shuffled within each family (seeded), then round-robin interleaved
    across families so any contiguous chunk of length ``L`` holds each family
    either ``floor(L/k)`` or ``ceil(L/k)`` times. The train/val/test cut is
    contiguous on that interleaving, so each split's per-family counts differ by
    at most one and every split with ``>= k`` samples contains all ``k``
    families. Falls back to the plain shuffled split behavior only in ordering;
    the fractions match :func:`split_sim_ids` exactly.
    """
    labels = np.asarray(labels)
    num_sims = int(labels.shape[0])
    rng = np.random.default_rng(seed)

    families = sorted({str(v) for v in labels.tolist()})
    queues: list[list[int]] = []
    for fam in families:
        fam_ids = np.nonzero(labels == fam)[0]
        rng.shuffle(fam_ids)
        queues.append(list(int(i) for i in fam_ids))

    order: list[int] = []
    idx = 0
    while len(order) < num_sims:
        q = queues[idx % len(queues)]
        if q:
            order.append(q.pop())
        idx += 1
    order_arr = np.asarray(order, dtype=int)

    n_train = int(num_sims * train_frac)
    n_val = int(num_sims * val_frac)
    train_ids = order_arr[:n_train]
    val_ids = order_arr[n_train:n_train + n_val]
    test_ids = order_arr[n_train + n_val:]
    return train_ids, val_ids, test_ids


def load_dataset_meta(t_grid_path: str | Path) -> dict | None:
    """Return the dataset ``meta.npy`` provenance dict, or None if absent.

    Looks for ``meta.npy`` beside the trajectory grids. Written at generation
    time for versioned (e.g. varying-IC) benchmarks with ``problem_version`` /
    ``ic_mode`` / ``ic_families``; absent for legacy fixed-IC datasets, which the
    load-time guard treats as a stale-dataset failure when a version is expected.
    """
    meta_path = Path(t_grid_path).parent / "meta.npy"
    if not meta_path.exists():
        return None
    return dict(np.load(meta_path, allow_pickle=True).item())


# ------- GLOBAL NORMALIZATION STATISTICS -------

def compute_global_stats(
    trajectories: np.ndarray,
    train_ids: np.ndarray,
    t_grid: np.ndarray | None = None,
    max_time: float | None = None,
) -> tuple[float, float]:
    """Compute global mean/std from training simulations only (avoids data leakage).

    When ``max_time`` (and ``t_grid``) are given, only snapshots with
    ``t <= max_time`` contribute — the windowed variant for a future absolute-
    time OOD study where late-time fields must not leak into normalization. The
    long-lead headline leaves ``max_time=None`` (full-trajectory stats, identical
    across every run), so this path is byte-identical to before by default.

    Returns (mu_global, sigma_global) as Python floats.
    """
    train_data = trajectories[train_ids]  # (N_train, Nt, Nx, Ny)
    if max_time is not None:
        if t_grid is None:
            raise ValueError("compute_global_stats: max_time requires t_grid.")
        keep = np.asarray(t_grid) <= float(max_time) + 1e-9
        if not keep.any():
            raise ValueError(
                f"compute_global_stats: max_time={max_time} keeps no snapshots."
            )
        train_data = train_data[:, keep]
    mu_global = float(train_data.mean())
    sigma_global = float(train_data.std())
    return mu_global, sigma_global


# ------- CREATE DATALOADERS -------

def create_dataloaders(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    batch_size: int,
    sim_params: np.ndarray,
    mu_global: float,
    sigma_global: float,
    n_snapshots: int = 15,
    n_snapshots_test: int | None = None,
    noise_std: float = 0.0,
    num_workers: int | None = None,
    dt: float | None = None,
    t_final: float | None = None,
    time_norm_horizon: float | None = None,
    ramp_seconds: float | None = None,
    temporal_samples: int = TEMPORAL_SAMPLES,
    world_size: int = 1,
    rank: int = 0,
    sampler_seed: int = 0,
    problem: ProblemSpec | None = None,
    train_max_target_time: float | None = None,
    train_max_lead_time: float | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    """Build train/val/test loaders.

    When `world_size > 1`:
    - `batch_size` is divided across ranks: per-GPU batch = `batch_size // world_size`.
      The total effective batch matches single-GPU baseline.
    - Train uses `CurriculumDistributedSampler` (curriculum-safe).
    - Val loader is built on **every rank**. Validation execution stays
      rank-0-only (the caller gates `validate()` on `is_main` and broadcasts the
      result), but the W2 collocation sampler reads `validation_set.dataset` on
      all ranks, so a `None` val loader on rank>0 crashes the DDP physics path.
    - Test loader is unchanged (eval is single-GPU and decoupled from training).
    """
    if world_size > 1:
        if batch_size % world_size != 0:
            raise ValueError(
                f"batch_size={batch_size} not divisible by world_size={world_size}. "
                f"Choose a batch_size that splits evenly across GPUs."
            )
        per_gpu_bs = batch_size // world_size
    else:
        per_gpu_bs = batch_size

    n_test = n_snapshots_test if n_snapshots_test is not None else n_snapshots

    if problem is None:
        problem = get_problem("forcing")

    train_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
        sim_ids=train_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=n_snapshots,
        noise_std=noise_std,
        dt=dt,
        t_final=t_final,
        time_norm_horizon=time_norm_horizon,
        ramp_seconds=ramp_seconds,
        temporal_samples=temporal_samples,
        problem=problem,
        max_target_time=train_max_target_time,
        max_lead_time=train_max_lead_time,
    )

    val_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
        sim_ids=val_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=n_snapshots,
        dt=dt,
        t_final=t_final,
        time_norm_horizon=time_norm_horizon,
        ramp_seconds=ramp_seconds,
        temporal_samples=temporal_samples,
        problem=problem,
    )

    test_dataset = SnapshotPairDataset(
        trajectories=trajectories,
        x_grid=x_grid,
        y_grid=y_grid,
        t_grid=t_grid,
        sim_ids=test_ids,
        sim_params=sim_params,
        mu_global=mu_global,
        sigma_global=sigma_global,
        n_snapshots=n_test,
        dt=dt,
        t_final=t_final,
        time_norm_horizon=time_norm_horizon,
        ramp_seconds=ramp_seconds,
        temporal_samples=temporal_samples,
        problem=problem,
    )

    pin = torch.cuda.is_available()
    workers = num_workers if num_workers is not None else (4 if pin else 0)
    persistent = workers > 0

    if world_size > 1:
        from data.distributed_sampler import CurriculumDistributedSampler

        train_sampler = CurriculumDistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=sampler_seed,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_bs,
            sampler=train_sampler,
            shuffle=False,
            pin_memory=pin,
            num_workers=workers,
            persistent_workers=persistent,
            collate_fn=collate_fn,
        )

        # Built on every rank: validation runs rank-0-only, but the W2
        # collocation sampler draws from `val_loader.dataset` on all ranks.
        # A None loader on rank>0 would crash setup and hang the surviving
        # ranks at the next collective. Workers spawn lazily on first
        # iteration, which non-main ranks never trigger, so this is cheap.
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_bs,
            shuffle=False,
            pin_memory=pin,
            num_workers=workers,
            persistent_workers=persistent,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_bs,
            shuffle=True,
            pin_memory=pin,
            num_workers=workers,
            persistent_workers=persistent,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_bs,
            shuffle=False,
            pin_memory=pin,
            num_workers=workers,
            persistent_workers=persistent,
            collate_fn=collate_fn,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=per_gpu_bs,
        shuffle=False,
        pin_memory=pin,
        num_workers=workers,
        persistent_workers=persistent,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=str(project_root / "data" / "trajectories.npy"),
        x_grid_path=str(project_root / "data" / "x_grid.npy"),
        y_grid_path=str(project_root / "data" / "y_grid.npy"),
        t_grid_path=str(project_root / "data" / "t_grid.npy"),
    )
    sim_params = np.load(str(project_root / "data" / "sim_params.npy"), allow_pickle=True)

    train_ids, val_ids, test_ids = split_sim_ids(num_sims=trajectories.shape[0], train_frac=0.7, val_frac=0.15, seed=0)
    mu_global, sigma_global = compute_global_stats(trajectories, train_ids)
    print(f"Global stats: mu={mu_global:.4f}, sigma={sigma_global:.4f}")

    batch_size = 64

    train_loader, val_loader, test_loader = create_dataloaders(
        trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        batch_size=batch_size, sim_params=sim_params,
        mu_global=mu_global, sigma_global=sigma_global,
    )

    # Verify shapes
    batch = next(iter(train_loader))
    print(f"spatial: {batch['spatial'].shape}")          # (B, Nx, Ny, 20)
    print(f"cond_static: {batch['cond_static'].shape}")  # (B, 23)
    print(f"forcing_seq: {batch['forcing_seq'].shape}")  # (B, M, 5)
    print(f"Y: {batch['Y'].shape}")                      # (B, Nx, Ny, 1)
    print(f"T_stats: {batch['T_stats'].shape}")          # (B, 2)
