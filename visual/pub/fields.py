"""Evaluate checkpoints on selected cases and return physical-Kelvin fields.

F05, F06, F09, F11, F13 and F15 all need the same three things: the run's
checkpoint, the trajectories that run was scored on, and the truth / prediction /
residual triple for a case that ``select.py`` chose. That is what this module
provides, once, so the figure modules compose rather than reimplement.

Everything here is in **Kelvin**. The model predicts in normalized space; the
de-normalization happens inside ``_prepare_prediction_case`` using the per-sample
``T_stats`` that the dataset emitted, exactly as evaluation does.

This module is deliberately not a ``fig_*`` module: it reads files and runs a
network, neither of which a figure module is allowed to do. It performs no
statistical reduction over ``test_records.csv`` -- every such reduction lives in
``stats.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from visual.pub import jump as jump_mod

_K_LEFT_DEFAULT = 2.0
_K_RIGHT_DEFAULT = 1.0
_K_LEFT_SOURCE = 3.0
_K_RIGHT_SOURCE = 35.0

CONTACT_JUMP_COHORT_SIZE = 24
CONTACT_JUMP_COHORT_SEED = 20260728


@dataclass
class RunBundle:
    """A loaded checkpoint plus the evaluation set it is scored against."""

    benchmark: str
    representation: str
    run_dir: Path
    model: object
    config: dict
    trajectories: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    t_grid: np.ndarray
    sim_params: np.ndarray
    dt: float | None
    # The training-set mean temperature baked into the checkpoint. Relative-L2 is
    # measured against `T - mu_global`, not against `T`: eval normalizes before it
    # sums (`target_sse_K2 = sum(y_true_norm^2) * sigma^2`, eval.py:1023), so the
    # denominator is the centred sum of squares. Dividing by absolute Kelvin
    # instead puts a ~300 K offset in the denominator and every case reads 0.00%.
    mu_global: float = 0.0

    @property
    def n_sims(self) -> int:
        return int(self.trajectories.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.trajectories.shape[1])


@dataclass
class CaseFields:
    """Truth, prediction and residual for one ``(sim_id, s, j...)`` case."""

    benchmark: str
    sim_id: int
    s: int
    targets: tuple[int, ...]
    truth: np.ndarray          # (n_targets, Nx, Ny) Kelvin
    pred: np.ndarray           # (n_targets, Nx, Ny) Kelvin
    mu_global: float
    x_grid: np.ndarray
    y_grid: np.ndarray
    t_source: float
    t_targets: np.ndarray
    lead_times: np.ndarray
    interface_x: float
    R_c: np.ndarray | float    # scalar, or (Ny,) for source_itr
    k_left: float
    k_right: float
    params: object

    @property
    def residual(self) -> np.ndarray:
        return self.pred - self.truth

    def contact_jumps(self) -> tuple[np.ndarray, np.ndarray]:
        """``(truth, prediction)`` contact jump ``R_c(y) q_n(y, t)``, each ``(n_targets, Ny)``."""
        args = (self.x_grid, self.interface_x, self.R_c, self.k_left, self.k_right)
        return (jump_mod.contact_jump_map(self.truth, *args),
                jump_mod.contact_jump_map(self.pred, *args))

    def contact_fluxes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(truth, prediction)`` through-interface flux ``q_n(y, t)``."""
        args = (self.x_grid, self.interface_x, self.R_c, self.k_left, self.k_right)
        return (jump_mod.contact_flux_map(self.truth, *args),
                jump_mod.contact_flux_map(self.pred, *args))


@dataclass(frozen=True)
class ContactJumpCohort:
    """Physical contact jumps for a fixed simulation cohort and lead grid."""

    benchmark: str
    sim_ids: tuple[int, ...]
    source_index: int
    target_indices: tuple[int, ...]
    lead_times: np.ndarray
    truth: np.ndarray          # (n_sims, n_targets, Ny) Kelvin
    pred: np.ndarray           # (n_sims, n_targets, Ny) Kelvin


# --------------------------------------------------------------------- loading

def checkpoint_path(run_dir: Path) -> Path | None:
    """The checkpoint of a run directory, preferring ``fno2d_best.pt``."""
    run_dir = Path(run_dir)
    best = run_dir / "fno2d_best.pt"
    if best.is_file():
        return best
    found = sorted(run_dir.glob("*.pt"))
    return found[0] if found else None


def run_dirs(source, benchmark: str | None = None) -> list[Path]:
    """Run directories a resolved :class:`FigureSource` carries, optionally filtered."""
    out: list[Path] = []
    for run in getattr(source, "runs", ()) or ():
        if benchmark is not None and run.benchmark != benchmark:
            continue
        out.append(Path(run.run_dir))
    return out


@lru_cache(maxsize=8)
def _load(run_dir_str: str) -> RunBundle:
    from visual.dataset_plots import _load_checkpoint_model, _load_plot_data

    run_dir = Path(run_dir_str)
    ckpt = checkpoint_path(run_dir)
    if ckpt is None:
        raise FileNotFoundError(f"{run_dir} has no .pt checkpoint")
    model, config = _load_checkpoint_model(ckpt)

    import torch
    mu_global = float(torch.load(ckpt, map_location="cpu", weights_only=True)
                      .get("mu_global", 0.0))

    # The checkpoint carries the config of the machine it trained on, whose data
    # paths do not exist here. The run directory's own config_used.yaml is what
    # the evaluation actually read, so its `data` block wins.
    used = run_dir / "config_used.yaml"
    if not used.is_file():
        used = run_dir.parent / "config_used.yaml"
    if used.is_file():
        import yaml
        run_config = yaml.safe_load(used.read_text()) or {}
        if run_config.get("data"):
            config = {**config, "data": run_config["data"]}
        for key in ("benchmark", "training", "physics"):
            if run_config.get(key):
                config = {**config, key: run_config[key]}

    data_cfg = config.get("data", {})
    trajectories, x_grid, y_grid, t_grid = _load_plot_data(
        data_cfg["trajectories.npy"], data_cfg["x_grid_path"],
        data_cfg["y_grid_path"], data_cfg["t_grid_path"],
    )
    sim_params = np.load(data_cfg["sim_params_path"], allow_pickle=True)

    dt = None
    dt_path = Path(data_cfg["trajectories.npy"]).parent / "dt.npy"
    if dt_path.is_file():
        dt = float(np.load(dt_path))

    bench_cfg = config.get("benchmark", {})
    return RunBundle(
        benchmark=str(bench_cfg.get("name", "unknown")),
        representation=str(bench_cfg.get("representation", "unknown")),
        run_dir=run_dir, model=model, config=config,
        trajectories=trajectories, x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
        sim_params=sim_params, dt=dt, mu_global=mu_global,
    )


def bundle(source, benchmark: str | None = None) -> RunBundle:
    """Load the checkpoint + evaluation set for ``benchmark`` out of ``source``."""
    dirs = run_dirs(source, benchmark)
    if not dirs:
        raise FileNotFoundError(
            f"no run directory for benchmark={benchmark!r} in {getattr(source, 'figure_key', '?')}"
        )
    return _load(str(dirs[0]))


def bundles(source) -> dict[str, RunBundle]:
    """Every benchmark's bundle, keyed by benchmark name."""
    out: dict[str, RunBundle] = {}
    for run in getattr(source, "runs", ()) or ():
        if run.benchmark and run.benchmark not in out:
            try:
                out[run.benchmark] = _load(str(run.run_dir))
            except (FileNotFoundError, KeyError):
                continue
    return out


# ------------------------------------------------------------------ evaluation

def layer_conductivities(bundle_: RunBundle) -> tuple[float, float]:
    if bundle_.benchmark in ("source", "source_itr", "source_itr_sin"):
        return _K_LEFT_SOURCE, _K_RIGHT_SOURCE
    layers = bundle_.config.get("physics", {}).get("layers")
    if layers and len(layers) >= 2:
        try:
            return float(layers[0]["k"]), float(layers[1]["k"])
        except (KeyError, TypeError, IndexError):
            pass
    return _K_LEFT_DEFAULT, _K_RIGHT_DEFAULT


def _param_names(params) -> tuple[str, ...]:
    """Field names of a per-simulation parameter record, dict or structured."""
    dtype_names = getattr(getattr(params, "dtype", None), "names", None)
    if dtype_names:
        return tuple(dtype_names)
    try:
        return tuple(params.keys())
    except AttributeError:
        return ()


def resistance(bundle_: RunBundle, sim_id: int) -> np.ndarray | float:
    """``R_c`` for one simulation: a scalar, or the ``(Ny,)`` void profile."""
    params = bundle_.sim_params[int(sim_id)]
    names = _param_names(params)
    if all(k in names for k in ("R_c_base", "R_c_amp", "R_c_y0", "R_c_sigma")):
        from src.physics.internal_source import make_rc_void_profile
        return make_rc_void_profile(
            np.asarray(bundle_.y_grid, dtype=np.float64),
            R_base=float(params["R_c_base"]), R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]), sigma=float(params["R_c_sigma"]),
        )
    if all(k in names for k in ("R_c_base", "R_c_A")):
        from src.physics.internal_source import make_rc_sin_profile
        return make_rc_sin_profile(
            np.asarray(bundle_.y_grid, dtype=np.float64),
            R_base=float(params["R_c_base"]), A=float(params["R_c_A"]),
        )
    return float(params["R_c"])


def void_profile(params) -> tuple[float, float] | None:
    """``(y0, sigma)`` of the Gaussian ``R_c(y)`` void, or ``None`` if uniform."""
    names = _param_names(params)
    if "R_c_y0" in names and "R_c_sigma" in names:
        return float(params["R_c_y0"]), float(params["R_c_sigma"])
    return None


def source_patch(params) -> tuple[float, float, float, float] | None:
    """``(x_h, y_h, w_h, h_h)`` of the volumetric heating patch, or ``None``.

    Only ``source`` and ``source_itr`` have one. Figures draw its outline so a
    reader can see where the heat entered the domain.
    """
    names = _param_names(params)
    if not all(k in names for k in ("x_h", "y_h", "w_h", "h_h")):
        return None
    return (float(params["x_h"]), float(params["y_h"]),
            float(params["w_h"]), float(params["h_h"]))


def interface_location(bundle_: RunBundle, sim_id: int) -> float:
    params = bundle_.sim_params[int(sim_id)]
    if "interface_x" in _param_names(params):
        return float(params["interface_x"])
    loss_cfg = bundle_.config.get("training", {}).get("loss", {})
    return float(loss_cfg.get("interface_x", 0.5))


def evaluate_case(bundle_: RunBundle, sim_id: int, s: int,
                  targets) -> CaseFields:
    """Run the checkpoint for one source snapshot and one or more target times."""
    from visual.dataset_plots import _prepare_prediction_case

    target_idx = np.asarray(list(targets), dtype=np.int64)
    case = _prepare_prediction_case(
        model=bundle_.model, trajectories=bundle_.trajectories,
        x_grid=bundle_.x_grid, y_grid=bundle_.y_grid, t_grid=bundle_.t_grid,
        sim_params=bundle_.sim_params, sim_id=int(sim_id), s=int(s),
        target_indices=target_idx, config=bundle_.config, dt=bundle_.dt,
    )
    k_left, k_right = layer_conductivities(bundle_)
    return CaseFields(
        benchmark=bundle_.benchmark, sim_id=int(sim_id), s=int(s),
        targets=tuple(int(t) for t in target_idx),
        truth=np.asarray(case["Y_true"], dtype=np.float64),
        pred=np.asarray(case["Y_pred"], dtype=np.float64),
        mu_global=bundle_.mu_global,
        x_grid=np.asarray(bundle_.x_grid), y_grid=np.asarray(bundle_.y_grid),
        t_source=float(case["source_time"]),
        t_targets=np.asarray(case["t_targets"], dtype=np.float64),
        lead_times=np.asarray(case["t_bars"], dtype=np.float64),
        interface_x=interface_location(bundle_, sim_id),
        R_c=resistance(bundle_, sim_id),
        k_left=k_left, k_right=k_right,
        params=bundle_.sim_params[int(sim_id)],
    )


def evaluate_contact_jump_cohort(
    bundle_: RunBundle,
    records_,
    *,
    n_sims: int = CONTACT_JUMP_COHORT_SIZE,
    source_index: int = 0,
    rng_seed: int = CONTACT_JUMP_COHORT_SEED,
) -> ContactJumpCohort:
    """Evaluate a reproducible simulation cohort on every future saved time.

    The cohort is sampled without replacement from simulation IDs present in
    the records and valid for the resolved run. No field or simulation-level
    reduction happens here; :mod:`visual.pub.stats` owns those operations.
    """
    frame = getattr(records_, "df", records_)
    if "sim_id" not in frame.columns:
        raise KeyError("contact-jump cohort selection requires sim_id")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    candidates = np.asarray(frame["sim_id"], dtype=np.int64)
    candidates = np.unique(candidates[
        (candidates >= 0) & (candidates < bundle_.n_sims)
    ])
    if candidates.size == 0:
        raise ValueError("no record simulation IDs are valid for the run bundle")

    rng = np.random.default_rng(rng_seed)
    size = min(int(n_sims), int(candidates.size))
    sim_ids = np.sort(rng.choice(candidates, size=size, replace=False))

    s = int(source_index)
    if not 0 <= s < bundle_.n_times - 1:
        raise ValueError(
            f"source_index must leave at least one future time; got {s} for "
            f"{bundle_.n_times} saved times"
        )
    targets = tuple(range(s + 1, bundle_.n_times))

    truth, pred = [], []
    lead_times = None
    for sim_id in sim_ids:
        case = evaluate_case(bundle_, int(sim_id), s, targets)
        truth_jump, pred_jump = case.contact_jumps()
        truth.append(truth_jump)
        pred.append(pred_jump)
        if lead_times is None:
            lead_times = case.lead_times
        elif not np.allclose(lead_times, case.lead_times):
            raise ValueError("contact-jump cohort cases do not share a lead grid")

    return ContactJumpCohort(
        benchmark=bundle_.benchmark,
        sim_ids=tuple(int(v) for v in sim_ids),
        source_index=s,
        target_indices=targets,
        lead_times=np.asarray(lead_times, dtype=np.float64),
        truth=np.stack(truth),
        pred=np.stack(pred),
    )


# ------------------------------------------------------- transverse structure

# The share of a truth field's spatial variance that survives removing its
# y-average. Zero means the field is exactly one-dimensional: constant along y,
# so every panel drawn from it is a set of vertical stripes and the reader
# learns nothing about how the operator behaves transverse to the interface.
#
# Large parts of every benchmark are like that, and not by accident. `forcing`
# with a `uniform` spatial profile is 1-D by construction. `source` and
# `source_itr` start 2-D at the patch and diffuse toward a 1-D profile, so the
# structure is gone by the late target times. `interfaces` drives all four
# benchmarks' least transverse variation of all: uniform/sin forcing, with only
# the initial condition breaking symmetry.
#
# 0.05 is a presentation threshold, not a physical one. It is recorded in the
# sidecar and in the figure footer so a reader can see the restriction rather
# than infer a 2-D benchmark from a hand-picked panel.
MIN_TRANSVERSE_FRACTION = 0.05


@lru_cache(maxsize=8)
def _transverse_grid(run_dir_str: str) -> np.ndarray:
    """``(n_sims, n_times)`` transverse variance fraction of every truth field."""
    traj = np.asarray(_load(run_dir_str).trajectories, dtype=np.float64)
    out = np.zeros(traj.shape[:2], dtype=np.float64)
    for i in range(traj.shape[0]):
        # Centre first. The identity below differences two mean-square terms,
        # and at 300 K those are ~9e4 while the variance of interest can be
        # ~1e-3, which cancels to noise if the offset is left in.
        f = traj[i]
        f = f - f.mean(axis=(1, 2), keepdims=True)
        total = (f ** 2).mean(axis=(1, 2))
        profile = f.mean(axis=2)                       # y-average, (T, Nx)
        transverse = total - (profile ** 2).mean(axis=1)
        out[i] = transverse / np.maximum(total, 1e-12)
    return np.clip(out, 0.0, 1.0)


def transverse_fraction(bundle_: RunBundle, sim_ids, targets) -> np.ndarray:
    """Transverse variance fraction of the truth field at each ``(sim_id, j)``."""
    grid = _transverse_grid(str(bundle_.run_dir))
    sim = np.asarray(sim_ids, dtype=np.int64)
    tgt = np.asarray(targets, dtype=np.int64)
    inside = (sim >= 0) & (sim < grid.shape[0]) & (tgt >= 0) & (tgt < grid.shape[1])
    out = np.zeros(sim.shape, dtype=np.float64)
    out[inside] = grid[sim[inside], tgt[inside]]
    return out


def transverse_pairs(bundle_: RunBundle, pair_df, *,
                     minimum: float = MIN_TRANSVERSE_FRACTION):
    """The subset of ``pair_df`` whose target truth field varies along ``y``.

    Returns the frame unchanged when the restriction would empty it, so a caller
    always gets something to draw.
    """
    pair_df = getattr(pair_df, "df", pair_df)
    if not {"sim_id", "j"} <= set(pair_df.columns):
        return pair_df
    r2d = transverse_fraction(bundle_, pair_df["sim_id"].to_numpy(),
                              pair_df["j"].to_numpy())
    kept = pair_df[r2d >= float(minimum)]
    return pair_df if kept.empty else kept


def select_transverse_case(bundle_: RunBundle, sims, pair_df, *,
                           minimum: float = MIN_TRANSVERSE_FRACTION,
                           min_leads: int = 1,
                           min_jump_contrast_K: float | None = None, **kwargs):
    """:func:`select.select_case`, restricted to cases that vary along ``y``.

    The restriction is applied to the two frames *before* selection rather than
    inside ``select.py``, so the selection protocol itself is untouched and
    ``PROTOCOL_VERSION`` stays where the golden test pins it. Within the
    restricted pool the rule is unchanged: the simulation at the requested
    quantile, ranked on its full pooled metric, and the median pair inside it.

    ``min_leads`` additionally requires the simulation to still have that many
    distinct lead times after the restriction. A figure whose columns *are* lead
    times needs one -- transverse structure decays, so the restriction strips
    the long leads and leaves a quarter of the simulations with a single lead,
    whose columns would then all be the same snapshot.

    ``min_jump_contrast_K`` narrows further to cases whose *contact jump* spans
    that many Kelvin along y. A figure with a jump row needs it: the transverse
    fraction is a property of the field, and clears easily for cases whose
    interface jump is still flat. Applied after the transverse filter, so a
    stratum with transverse cases but no contrasting jump keeps the transverse
    pool rather than dropping all the way back.

    Falls back to the unrestricted pool when nothing in the requested stratum
    survives. Some strata have no such case to find -- ``forcing`` with a
    ``uniform`` spatial profile has a rigorously constant jump -- and refusing
    to draw them would hide a real property of the benchmark. The returned flag
    says which pool was used and belongs in the sidecar.
    """
    from visual.pub import select

    sim_df = getattr(sims, "df", sims)
    pair_df = getattr(pair_df, "df", pair_df)
    kept = transverse_pairs(bundle_, pair_df, minimum=minimum)
    if min_jump_contrast_K is not None:
        kept = jump_contrast_pairs(bundle_, kept, minimum=min_jump_contrast_K)
    if kept is not pair_df:
        eligible = set(kept["sim_id"])
        if min_leads > 1 and "t_bar" in kept.columns:
            counts = kept.groupby("sim_id")["t_bar"].nunique()
            spanning = set(counts[counts >= min_leads].index)
            if spanning:
                eligible = spanning
                kept = kept[kept["sim_id"].isin(spanning)]
        keep_sims = sim_df[sim_df["sim_id"].isin(eligible)]
        if not keep_sims.empty:
            try:
                return select.select_case(keep_sims, kept, **kwargs), True
            except select.SelectionError:
                pass
    return select.select_case(sim_df, pair_df, **kwargs), False


# ------------------------------------------------------------- jump contrast

# Peak-to-peak spread of the truth contact jump along y, in Kelvin. The
# transverse fraction above measures structure in the *field*; this measures it
# in the quantity the jump row actually plots, and the two do not agree. A field
# can be strongly two-dimensional while its interface jump is nearly flat.
#
# 0.5 K is a presentation threshold. It sits above the ~0.06 K median of the
# forcing pool -- half of all pairs are essentially flat -- and still leaves 4
# to 11 simulations in every (temporal, spatial) cell that has any contrast at
# all, so the median rule inside the restricted pool still ranks a population
# rather than picking from a handful of extremes.
#
# Deliberately an absolute threshold, not ptp relative to the jump's mean. The
# relative measure is largest where the mean is near zero, so maximising it
# hunts the extremes the median protocol exists to avoid.
MIN_JUMP_CONTRAST_K = 0.5


@lru_cache(maxsize=8)
def _jump_contrast_grid(run_dir_str: str) -> np.ndarray:
    """``(n_sims, n_times)`` peak-to-peak truth contact jump, in Kelvin."""
    from visual.pub import jump as jump_mod

    b = _load(run_dir_str)
    traj = np.asarray(b.trajectories, dtype=np.float64)
    x_grid = np.asarray(b.x_grid, dtype=np.float64)
    k_left, k_right = layer_conductivities(b)
    out = np.zeros(traj.shape[:2], dtype=np.float64)
    for i in range(traj.shape[0]):
        profile = jump_mod.contact_jump_map(
            traj[i], x_grid, interface_location(b, i), resistance(b, i),
            k_left, k_right)
        out[i] = np.ptp(profile, axis=1)
    return out


def jump_contrast(bundle_: RunBundle, sim_ids, targets) -> np.ndarray:
    """Peak-to-peak truth contact jump at each ``(sim_id, j)``, in Kelvin."""
    grid = _jump_contrast_grid(str(bundle_.run_dir))
    sim = np.asarray(sim_ids, dtype=np.int64)
    tgt = np.asarray(targets, dtype=np.int64)
    inside = (sim >= 0) & (sim < grid.shape[0]) & (tgt >= 0) & (tgt < grid.shape[1])
    out = np.zeros(sim.shape, dtype=np.float64)
    out[inside] = grid[sim[inside], tgt[inside]]
    return out


def jump_contrast_pairs(bundle_: RunBundle, pair_df, *,
                        minimum: float = MIN_JUMP_CONTRAST_K):
    """The subset of ``pair_df`` whose truth contact jump varies along ``y``.

    Returns the frame unchanged when the restriction would empty it, matching
    :func:`transverse_pairs` so the two can be chained and the caller can still
    tell by identity whether anything was removed.
    """
    pair_df = getattr(pair_df, "df", pair_df)
    if not {"sim_id", "j"} <= set(pair_df.columns):
        return pair_df
    ptp = jump_contrast(bundle_, pair_df["sim_id"].to_numpy(),
                        pair_df["j"].to_numpy())
    kept = pair_df[ptp >= float(minimum)]
    return pair_df if kept.empty else kept


def jump_contrast_note(minimum: float = MIN_JUMP_CONTRAST_K) -> str:
    """One sidecar-ready sentence describing the jump-contrast restriction."""
    return (f"restricted to cases whose truth contact jump spans at least "
            f"{minimum:g} K along y, so the jump panel shows transverse "
            f"structure rather than a flat line; strata with no such case fall "
            f"back to the unrestricted pool")


def transverse_note(restricted: bool, minimum: float = MIN_TRANSVERSE_FRACTION) -> str:
    """One sidecar-ready sentence describing which pool a case was drawn from."""
    if restricted:
        return (f"restricted to cases whose truth field has at least "
                f"{minimum:.0%} of its spatial variance transverse to x, so the "
                f"panels show two-dimensional diffusion rather than a 1-D profile")
    return ("unrestricted: no case in this stratum clears the transverse-"
            "structure threshold, so the stratum is genuinely near-1-D")


def case_metrics(case: CaseFields) -> list[dict]:
    """Per-target-time error headlines for one case, in Kelvin and percent.

    These describe *one snapshot pair each*; they are annotations on a picture,
    not an estimate of model accuracy. Every population-level number in the paper
    comes from ``stats.py`` over ``test_records.csv``.
    """
    truth_jump, pred_jump = case.contact_jumps()
    out: list[dict] = []
    for i in range(case.truth.shape[0]):
        err = case.pred[i] - case.truth[i]
        sse = float(np.sum(err ** 2))
        n = int(err.size)
        denom = float(np.sum((case.truth[i] - case.mu_global) ** 2))
        djump = pred_jump[i] - truth_jump[i]
        out.append({
            "j": case.targets[i],
            "t_target": float(case.t_targets[i]),
            "lead_time": float(case.lead_times[i]),
            "rmse_K": float(np.sqrt(sse / max(n, 1))),
            "rel_l2_pct": 100.0 * float(np.sqrt(sse / max(denom, 1e-12))),
            "contact_jump_rmse_K": float(
                np.sqrt(float(np.sum(djump ** 2)) / max(djump.size, 1))),
            "contact_jump_rms_K": float(
                np.sqrt(float(np.sum(truth_jump[i] ** 2)) / max(truth_jump[i].size, 1))),
        })
    return out


__all__ = [
    "CONTACT_JUMP_COHORT_SEED",
    "CONTACT_JUMP_COHORT_SIZE",
    "MIN_JUMP_CONTRAST_K",
    "MIN_TRANSVERSE_FRACTION",
    "CaseFields",
    "ContactJumpCohort",
    "RunBundle",
    "bundle",
    "bundles",
    "case_metrics",
    "checkpoint_path",
    "evaluate_case",
    "evaluate_contact_jump_cohort",
    "interface_location",
    "jump_contrast",
    "jump_contrast_note",
    "jump_contrast_pairs",
    "layer_conductivities",
    "resistance",
    "run_dirs",
    "select_transverse_case",
    "source_patch",
    "transverse_fraction",
    "transverse_note",
    "transverse_pairs",
    "void_profile",
]
