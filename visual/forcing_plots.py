"""Forcing-function visualization for the separable-forcing benchmark.

These plots illustrate the forcing functions used in the
``q_L(y, t) = a(t) * s(y)`` benchmark across four temporal families
(``sin, exp, pulse_train, exp_train``) and four spatial profiles
(``uniform, patch, gaussian, triangle``).

The retained diagnostics split into two roles:

- **Physical-forcing plots** (``forcing_temporal_families``,
  ``forcing_spatial_profiles``, ``forcing_separable_assembly``) show
  ``a(t)``, ``s(y)``, and the assembled left-boundary flux ``q_L(y, t)``
  that the FV solver applies. This is NOT the exact tensor representation
  seen by the model.

- **Model-side encoding plot** (``forcing_seq_tokens``) shows the live
  ``(128, 3)`` temporal representation consumed by the model.

Token dimensions and encoding are imported from ``problems/forcing.py`` so the
figure tracks the active benchmark contract.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems.forcing import (
    A_AMP_REF,
    FORCING_TEMPORAL_SAMPLES,
    FORCING_TEMPORAL_TOKEN_DIM,
    _forcing_seq_3tok_from_samples,
    _sample_a,
)
from src.physics.boundary_forcing import (
    SPATIAL_BUILDERS,
    TEMPORAL_BUILDERS,
    TEMPORAL_FAMILY_ORDER,
    TEMPORAL_SAMPLERS,
    SPATIAL_SAMPLERS,
    build_qL,
    default_ramp_seconds,
    integrate_temporal_intervals_ramped_signed,
    ramped_temporal,
)
from visual._common import PLOT_STYLE, _save_figure

# Canonical sampling window from data/generate_dataset.py. The samplers need
# `dt` and `t_final`; sin additionally needs `t_on, t_off`.
DEFAULT_DT = 0.005
DEFAULT_T_FINAL = 0.3
DEFAULT_T_ON = 0.0
DEFAULT_T_OFF = 0.2

SPATIAL_FAMILY_ORDER = ("uniform", "patch", "gaussian", "triangle")
FAMILY_COLORS = {
    "sin":         "C0",
    "exp":         "C1",
    "pulse_train": "C2",
    "exp_train":   "C3",
}

# Token layout for forcing_seq. The semantics MUST stay aligned with
# problems/forcing.py:_forcing_seq_3tok_from_samples.
_FORCING_SEQ_TOKEN_LABELS = (
    r"$r_m = m / (M-1)$",
    rf"$a(t_m) / A_{{\mathrm{{ref}}}}$  ($A_{{\mathrm{{ref}}}}={A_AMP_REF:.0f}$)",
    r"$\overline{a}_{[t_m,t_{m+1}]} / A_{\mathrm{ref}}$",
)


# --------- HELPERS ----------

def _sample_temporal_params(
    family: str,
    rng: np.random.Generator,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
) -> dict:
    """Dispatch to the right sampler with the keyword args each one needs."""
    sampler = TEMPORAL_SAMPLERS[family]
    if family == "sin":
        return sampler(rng, dt=dt, t_final=t_final, t_on=t_on, t_off=t_off)
    return sampler(rng, dt=dt, t_final=t_final)


def _evaluate_temporal(family: str, params: dict, t: np.ndarray) -> np.ndarray:
    """Evaluate a(t) on a vector grid by scalar iteration (handles non-vectorized builders)."""
    builder = TEMPORAL_BUILDERS[family]
    q = builder(**params)
    out = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        val = q(float(ti))
        out[i] = float(np.asarray(val).reshape(-1)[0]) if np.ndim(val) else float(val)
    return out


def _evaluate_spatial(family: str, params: dict, y: np.ndarray) -> np.ndarray:
    builder = SPATIAL_BUILDERS[family]
    return np.asarray(builder(y, **params), dtype=float)


# --------- 1. TEMPORAL FAMILIES ----------

def plot_forcing_temporal_families(
    n_curves: int = 6,
    n_time: int = 400,
    seed: int = 0,
    save_path: str | Path | None = None,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
):
    """Overlay sampled a(t) curves per temporal family on a shared time axis."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_final, n_time)

    curves: dict[str, list[np.ndarray]] = {fam: [] for fam in TEMPORAL_FAMILY_ORDER}
    for fam in TEMPORAL_FAMILY_ORDER:
        for _ in range(n_curves):
            params = _sample_temporal_params(fam, rng, dt=dt, t_final=t_final,
                                             t_on=t_on, t_off=t_off)
            curves[fam].append(_evaluate_temporal(fam, params, t))

    y_max = max(float(np.max(np.abs(arr))) for fam in curves for arr in curves[fam])
    y_max = max(y_max, 1.0) * 1.1

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True,
                                 sharey=True)
        for ax, fam in zip(axes.ravel(), TEMPORAL_FAMILY_ORDER):
            for arr in curves[fam]:
                ax.plot(t, arr, color=FAMILY_COLORS[fam], alpha=0.55, linewidth=1.4)
            if fam == "sin":
                ax.axvspan(t_on, t_off, color="0.85", alpha=0.5,
                           label=f"window [{t_on:.2f}, {t_off:.2f}]")
                ax.legend(loc="upper right")
            ax.set_xlim(0.0, t_final)
            ax.set_ylim(-y_max, y_max)
            ax.set_title(f"{fam}  —  {n_curves} samples")
            ax.set_xlabel("t")
            ax.set_ylabel("a(t)")
            ax.grid(True)
            ax.axhline(0.0, color="0.5", linewidth=0.8)

        fig.suptitle("Temporal forcing families a(t)  —  physical boundary forcing")
        _save_figure(fig, save_path, "forcing", "forcing_temporal_families",
                     layout="constrained")


# --------- 2. SPATIAL PROFILES ----------

def plot_forcing_spatial_profiles(
    n_curves: int = 6,
    n_y: int = 400,
    seed: int = 0,
    save_path: str | Path | None = None,
):
    """Overlay sampled s(y) curves per spatial family on y in [0, 1]."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, n_y)

    curves: dict[str, list[tuple[dict, np.ndarray]]] = {
        fam: [] for fam in SPATIAL_FAMILY_ORDER
    }
    for fam in SPATIAL_FAMILY_ORDER:
        if fam == "uniform":
            curves[fam].append(({}, _evaluate_spatial(fam, {}, y)))
            continue
        for _ in range(n_curves):
            params = SPATIAL_SAMPLERS[fam](rng)
            curves[fam].append((params, _evaluate_spatial(fam, params, y)))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True,
                                 sharey=True)
        for ax, fam in zip(axes.ravel(), SPATIAL_FAMILY_ORDER):
            color = f"C{SPATIAL_FAMILY_ORDER.index(fam)}"
            for params, arr in curves[fam]:
                ax.plot(y, arr, color=color, alpha=0.6, linewidth=1.5)
                if "y_c" in params:
                    ax.axvline(params["y_c"], color=color, alpha=0.18,
                               linestyle=":", linewidth=0.9)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.05, 1.15)
            n_shown = len(curves[fam])
            ax.set_title(f"{fam}  —  {n_shown} sample{'s' if n_shown != 1 else ''}")
            ax.set_xlabel("y")
            ax.set_ylabel("s(y)")
            ax.grid(True)

        fig.suptitle("Spatial forcing profiles s(y)")
        _save_figure(fig, save_path, "forcing", "forcing_spatial_profiles",
                     layout="constrained")


# --------- 3. SEPARABLE ASSEMBLY ----------

def plot_forcing_separable_assembly(
    n_time: int = 200,
    n_y: int = 120,
    seed: int = 0,
    save_path: str | Path | None = None,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
):
    """4x4 grid of q_L(y, t) heatmaps for each (temporal, spatial) pair.

    This is the PHYSICAL left-boundary flux applied by the FV solver. The
    tensor representation seen by the model is shown in the model-side
    plot (forcing_seq_tokens).
    """
    rng = np.random.default_rng(seed)
    y_grid = np.linspace(0.0, 1.0, n_y)
    t = np.linspace(0.0, t_final, n_time)

    # One representative sample per temporal family (shared across spatial columns).
    temporal_samples = {
        fam: _sample_temporal_params(fam, rng, dt=dt, t_final=t_final,
                                     t_on=t_on, t_off=t_off)
        for fam in TEMPORAL_FAMILY_ORDER
    }
    # One representative sample per spatial family.
    spatial_samples = {
        fam: ({} if fam == "uniform" else SPATIAL_SAMPLERS[fam](rng))
        for fam in SPATIAL_FAMILY_ORDER
    }

    # Per-row symmetric scaling: scale by max |q_L| across the row.
    row_qmax: dict[str, float] = {}
    fields: dict[tuple[str, str], np.ndarray] = {}
    for tfam in TEMPORAL_FAMILY_ORDER:
        row_max = 0.0
        for sfam in SPATIAL_FAMILY_ORDER:
            q_fn, _s_vec = build_qL(tfam, temporal_samples[tfam],
                                    sfam, spatial_samples[sfam], y_grid)
            Q = np.stack([np.asarray(q_fn(float(ti)), dtype=float) for ti in t], axis=0)
            fields[(tfam, sfam)] = Q
            row_max = max(row_max, float(np.max(np.abs(Q))))
        row_qmax[tfam] = max(row_max, 1e-12)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            len(TEMPORAL_FAMILY_ORDER), len(SPATIAL_FAMILY_ORDER),
            figsize=(4.0 * len(SPATIAL_FAMILY_ORDER), 3.2 * len(TEMPORAL_FAMILY_ORDER)),
            constrained_layout=True, squeeze=False, sharex=True, sharey=True,
        )
        for r, tfam in enumerate(TEMPORAL_FAMILY_ORDER):
            qmax = row_qmax[tfam]
            row_pcm = None
            for c, sfam in enumerate(SPATIAL_FAMILY_ORDER):
                ax = axes[r, c]
                Q = fields[(tfam, sfam)]
                row_pcm = ax.pcolormesh(
                    t, y_grid, Q.T,
                    cmap="coolwarm", vmin=-qmax, vmax=qmax, shading="auto",
                )
                if r == 0:
                    ax.set_title(sfam)
                if c == 0:
                    ax.set_ylabel(f"{tfam}\ny")
                if r == len(TEMPORAL_FAMILY_ORDER) - 1:
                    ax.set_xlabel("t")
            fig.colorbar(row_pcm, ax=axes[r, :].tolist(), shrink=0.85,
                         label=f"q_L (max |·| = {qmax:.1f})")

        fig.suptitle(
            "Physical left-boundary flux $q_L(y, t) = a(t)\\,s(y)$  —  "
            "applied by FV solver (not the model-side tensor; see "
            "forcing_seq_tokens)"
        )
        _save_figure(fig, save_path, "forcing", "forcing_separable_assembly",
                     layout="constrained")

# --------- 4. OOD SINUSOID ASSEMBLY ----------

def plot_forcing_sinusoid_temporal_panels(
    n_time: int = 240,
    n_y: int = 160,
    seed: int = 1,
    save_path: str | Path | None = None,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
):
    """Four q_L(y, t) panels using one OOD sinusoid and all temporal families."""
    spatial_rng = np.random.default_rng(seed)
    temporal_rng = np.random.default_rng(seed + 1)
    spatial_params = SPATIAL_SAMPLERS["sinusoid"](spatial_rng)
    temporal_params = {
        family: _sample_temporal_params(
            family, temporal_rng, dt=dt, t_final=t_final,
            t_on=t_on, t_off=t_off,
        )
        for family in TEMPORAL_FAMILY_ORDER
    }

    y_grid = np.linspace(0.0, 1.0, n_y)
    t = np.linspace(0.0, t_final, n_time)
    t_ramp = default_ramp_seconds(dt)
    fields = {}
    for family in TEMPORAL_FAMILY_ORDER:
        q_fn, _ = build_qL(
            family, temporal_params[family], "sinusoid", spatial_params,
            y_grid, t_ramp=t_ramp,
        )
        fields[family] = np.stack(
            [np.asarray(q_fn(float(ti)), dtype=float) for ti in t], axis=0,
        )

    q_max = max(float(np.max(field)) for field in fields.values())
    family_labels = {
        "sin": "Sinusoid",
        "exp": "Exponential",
        "pulse_train": "Pulse train",
        "exp_train": "Exponential train",
    }
    modulation = spatial_params["c1"] / (spatial_params["c0"] + spatial_params["c1"])

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            1, len(TEMPORAL_FAMILY_ORDER), figsize=(12.0, 3.1),
            constrained_layout=True, sharex=True, sharey=True,
        )
        image = None
        for ax, family in zip(axes, TEMPORAL_FAMILY_ORDER):
            image = ax.pcolormesh(
                t, y_grid, fields[family].T, cmap="viridis",
                vmin=0.0, vmax=q_max, shading="auto",
            )
            ax.set_title(family_labels[family])
            ax.set_xlabel("time $t$")
            ax.set_xlim(0.0, t_final)
        axes[0].set_ylabel("boundary coordinate $y$")
        fig.colorbar(image, ax=axes.tolist(), label=r"heat flux $q_L(y,t)$")
        fig.suptitle(
            r"Unseen sinusoidal spatial profile with four temporal forcings "
            rf"($m={modulation:.2f}$, $f_y={spatial_params['f']:.2f}$)"
        )
        _save_figure(
            fig, save_path, "forcing", "forcing_sinusoid_temporal_panels",
            layout="constrained",
        )


# --------- 5. ZERO-SHOT FIELD CASES ----------

def _select_diffusive_sinusoid_cases(
    trajectories: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    sim_params: np.ndarray,
    minimum_target_fraction: float = 0.15,
    maximum_target_fraction: float = 0.35,
    n_cases: int = 2,
) -> list[dict]:
    """Select early, diffused 2D cases without using model predictions."""
    t_final = float(t_grid[-1])
    target_indices = np.flatnonzero(
        (np.asarray(t_grid, dtype=float) >= minimum_target_fraction * t_final)
        & (np.asarray(t_grid, dtype=float) <= maximum_target_fraction * t_final)
    )
    if target_indices.size == 0:
        target_indices = np.array([
            int(np.argmin(np.abs(
                np.asarray(t_grid, dtype=float)
                - maximum_target_fraction * t_final
            )))
        ])

    sinusoid_ids = [
        sid for sid, params in enumerate(sim_params)
        if params.get("spatial_family") == "sinusoid"
    ]
    if not sinusoid_ids:
        raise ValueError("No sinusoid spatial-profile simulations are available.")

    x = np.asarray(x_grid, dtype=float)
    candidates = []
    for sim_id in sinusoid_ids:
        params = sim_params[sim_id]
        baseline = float(np.mean(trajectories[sim_id, 0]))
        for target_index in target_indices:
            truth = np.asarray(trajectories[sim_id, target_index], dtype=np.float64)
            centered = truth - float(np.mean(truth))
            total_variance = float(np.mean(centered ** 2))
            x_profile = np.mean(centered, axis=1)
            transverse = max(
                total_variance - float(np.mean(x_profile ** 2)), 0.0
            ) / max(total_variance, 1e-12)
            interior = (x >= 0.2) & (x < 0.5)
            downstream = x >= 0.5
            interior_rms = float(np.sqrt(np.mean(
                (truth[interior] - baseline) ** 2
            )))
            downstream_rms = float(np.sqrt(np.mean(
                (truth[downstream] - baseline) ** 2
            )))
            penetration_rms = interior_rms + downstream_rms
            temperature_span = float(np.ptp(truth))
            score = (
                transverse
                * np.log1p(temperature_span)
                * np.log1p(penetration_rms)
            )
            candidates.append({
                "sim_id": int(sim_id),
                "source_index": 0,
                "target_index": int(target_index),
                "temporal_family": str(params.get("temporal_family", "unknown")),
                "score": float(score),
                "transverse_fraction": float(transverse),
                "penetration_rms_K": float(penetration_rms),
                "temperature_span_K": temperature_span,
            })
    ranked = sorted(
        candidates,
        key=lambda row: (-row["score"], row["sim_id"], row["target_index"]),
    )
    selected = [ranked[0]]
    for candidate in ranked[1:]:
        if candidate["sim_id"] == selected[0]["sim_id"]:
            continue
        if candidate["temporal_family"] == selected[0]["temporal_family"]:
            continue
        selected.append(candidate)
        if len(selected) == n_cases:
            return selected
    for candidate in ranked[1:]:
        if candidate["sim_id"] in {row["sim_id"] for row in selected}:
            continue
        selected.append(candidate)
        if len(selected) == n_cases:
            return selected
    raise ValueError(f"Need {n_cases} distinct sinusoid cases, found {len(selected)}.")


def plot_zero_shot_sinusoid_field_and_jump(
    cases: list[dict],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    mu_global: float,
    save_path: str | Path | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> dict:
    """Publication 3x2 field comparison for two sinusoid OOD cases."""
    from visual.pub import panels, style

    if len(cases) != 2:
        raise ValueError(f"Expected exactly two cases, got {len(cases)}.")

    prepared = []
    for case in cases:
        truth = np.asarray(case["truth"], dtype=np.float64)
        prediction = np.asarray(case["prediction"], dtype=np.float64)
        if truth.shape != prediction.shape or truth.ndim != 2:
            raise ValueError(
                "truth and prediction must be matching (Nx, Ny) fields, got "
                f"{truth.shape} and {prediction.shape}."
            )
        if truth.shape != (len(x_grid), len(y_grid)):
            raise ValueError(
                f"field shape {truth.shape} does not match grids "
                f"({len(x_grid)}, {len(y_grid)})."
            )
        residual = prediction - truth
        rmse_K = float(np.sqrt(np.mean(residual ** 2)))
        rel_l2_pct = 100.0 * float(np.sqrt(
            np.sum(residual ** 2)
            / max(float(np.sum((truth - float(mu_global)) ** 2)), 1e-12)
        ))
        params = case["sim_params"]
        spatial = params.get("spatial_params", {})
        c0 = float(spatial.get("c0", 1.0))
        c1 = float(spatial.get("c1", 0.0))
        modulation = c1 / (c0 + c1) if c0 + c1 != 0.0 else float("nan")
        prepared.append({
            **case,
            "truth": truth,
            "prediction": prediction,
            "residual": residual,
            "rmse_K": rmse_K,
            "rel_l2_pct": rel_l2_pct,
            "temporal_family": str(params.get("temporal_family", "unknown")),
            "spatial_family": str(params.get("spatial_family", "unknown")),
            "modulation_depth": float(modulation),
            "spatial_frequency": float(spatial.get("f", float("nan"))),
            "R_c": float(params.get("R_c", float("nan"))),
        })

    temperature_limits = style.shared_color_limits(
        *(item[key] for item in prepared for key in ("truth", "prediction")),
        mode="sequential", robust=False,
    )
    residual_limits = style.shared_color_limits(
        *(item["residual"] for item in prepared),
        mode="diverging", robust=False,
    )

    with style.pub_style():
        fig = plt.figure(figsize=(style.WIDTHS_IN["full_page"], 7.35))
        grid = fig.add_gridspec(
            3, 3,
            width_ratios=(1.0, 0.05, 1.0),
            left=0.08, right=0.98, bottom=0.075, top=0.85,
            hspace=0.28, wspace=0.22,
        )
        field_axes = np.array([
            [fig.add_subplot(grid[row, col]) for col in (0, 2)]
            for row in range(3)
        ])
        temperature_cax = fig.add_subplot(grid[0:2, 1])
        residual_cax = fig.add_subplot(grid[2, 1])

        images = np.empty((3, 2), dtype=object)
        row_specs = (
            ("truth", temperature_limits, "FV truth"),
            ("prediction", temperature_limits, "FNO prediction"),
            ("residual", residual_limits, r"Residual: $T_{FNO}-T_{FV}$"),
        )
        for col, item in enumerate(prepared):
            for row, (field_key, limits, title) in enumerate(row_specs):
                images[row, col] = panels.field_map(
                    field_axes[row, col], item[field_key], limits,
                    interface_x=float(item["interface_x"]), title=title,
                    xlabel="$x$" if row == 2 else "", ylabel="$y$",
                )
                if row != 2:
                    field_axes[row, col].tick_params(labelbottom=False)

        panels.attach_colorbar(
            fig, images[0, 0], field_axes[:2, :].ravel().tolist(), "$T$ [K]",
            cax=temperature_cax,
        )
        panels.attach_colorbar(
            fig, images[2, 0], field_axes[2, :].tolist(),
            r"$T_{FNO}-T_{FV}$ [K]",
            cax=residual_cax,
        )

        for col, item in enumerate(prepared):
            field_axes[2, col].text(
                0.97, 0.03,
                f"RMSE = {item['rmse_K']:.3f} K\n"
                f"rel. $L_2$ = {item['rel_l2_pct']:.2f}%",
                transform=field_axes[2, col].transAxes,
                ha="right", va="bottom", fontsize=5.8,
                bbox=dict(facecolor="w", alpha=0.78, edgecolor="none",
                          boxstyle="square,pad=0.18"),
            )

        fig.suptitle(
            "Zero-shot sinusoidal spatial forcing at early times",
            y=0.975, fontsize=9,
        )
        column_centers = [
            (field_axes[0, col].get_position().x0
             + field_axes[0, col].get_position().x1) / 2.0
            for col in range(2)
        ]
        temporal_labels = {
            "sin": "Sinusoidal",
            "exp": "Exponential",
            "pulse_train": "Pulse train",
            "exp_train": "Exponential train",
        }
        for col, item in enumerate(prepared):
            temporal_label = temporal_labels.get(
                item["temporal_family"],
                item["temporal_family"].replace("_", " ").title(),
            )
            fig.text(
                column_centers[col], 0.925,
                f"Case {chr(ord('A') + col)}: {temporal_label}, "
                f"$t={float(item['t_target']):.2f}$",
                ha="center", va="bottom", fontsize=7.2, color="0.15",
            )
            fig.text(
                column_centers[col], 0.902,
                f"$t_s={float(item['t_source']):.2f}$, "
                f"$m={item['modulation_depth']:.2f}$, "
                f"$f_y={item['spatial_frequency']:.2f}$, "
                f"$R_c={item['R_c']:.3f}$",
                ha="center", va="bottom", fontsize=6.2, color="0.30",
            )
        style.panel_letters(
            field_axes.ravel().tolist(),
            loc=(-0.16, 1.04),
        )

        if save_path is None:
            save_path = Path("visual/forcing/forcing_zero_shot_field_jump")
        resolved = Path(save_path)
        if resolved.suffix:
            paths = [_save_figure(
                fig, resolved, "forcing", "forcing_zero_shot_field_jump",
                layout="none",
            )]
        else:
            paths = style.save(
                fig, resolved.parent, resolved.name, formats=formats,
            )

    return {
        "paths": paths,
        "selections": [{
            "sim_id": int(item["sim_id"]),
            "source_index": int(item["source_index"]),
            "target_index": int(item["target_index"]),
            "t_source": float(item["t_source"]),
            "t_target": float(item["t_target"]),
            "temporal_family": item["temporal_family"],
            "spatial_family": item["spatial_family"],
            "modulation_depth": item["modulation_depth"],
            "spatial_frequency": item["spatial_frequency"],
            "R_c": item["R_c"],
        } for item in prepared],
        "metrics": [{
            "rmse_K": item["rmse_K"],
            "rel_l2_pct": item["rel_l2_pct"],
        } for item in prepared],
    }


def generate_zero_shot_sinusoid_field_and_jump(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    *,
    save_path: str | Path | None = None,
    sim_ids: tuple[int, int] | None = None,
    source_index: int = 0,
    target_indices: tuple[int, int] | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> dict:
    """Select, evaluate, and draw two early sinusoidal OOD field cases."""
    from visual.dataset_plots import _load_checkpoint_model, _prepare_prediction_case

    data_dir = Path(data_dir)
    required = {
        name: data_dir / name
        for name in (
            "trajectories.npy", "x_grid.npy", "y_grid.npy", "t_grid.npy",
            "sim_params.npy", "dt.npy",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Zero-shot field plot needs the standard dataset files; missing: "
            + ", ".join(missing)
        )

    model, config = _load_checkpoint_model(checkpoint_path)
    trajectories = np.load(required["trajectories.npy"], mmap_mode="r")
    x_grid = np.load(required["x_grid.npy"])
    y_grid = np.load(required["y_grid.npy"])
    t_grid = np.load(required["t_grid.npy"])
    sim_params = np.load(required["sim_params.npy"], allow_pickle=True)
    dt = float(np.load(required["dt.npy"]))

    if sim_ids is None or target_indices is None:
        selected = _select_diffusive_sinusoid_cases(
            trajectories, x_grid, t_grid, sim_params,
        )
        if sim_ids is None:
            sim_ids = tuple(row["sim_id"] for row in selected)
        if target_indices is None:
            target_indices = tuple(row["target_index"] for row in selected)
    sim_ids = tuple(int(value) for value in sim_ids)
    target_indices = tuple(int(value) for value in target_indices)
    if len(sim_ids) != 2 or len(target_indices) != 2:
        raise ValueError("sim_ids and target_indices must each contain two values.")

    cases = []
    for sim_id, target_index in zip(sim_ids, target_indices):
        predicted = _prepare_prediction_case(
            model=model, trajectories=trajectories,
            x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
            sim_params=sim_params, sim_id=sim_id, s=int(source_index),
            target_indices=np.array([target_index], dtype=np.int64),
            config=config, dt=dt,
        )
        cases.append({
            "truth": predicted["Y_true"][0],
            "prediction": predicted["Y_pred"][0],
            "interface_x": float(predicted["interface_x"]),
            "sim_params": sim_params[sim_id],
            "t_source": float(predicted["source_time"]),
            "t_target": float(predicted["t_targets"][0]),
            "sim_id": sim_id,
            "source_index": int(source_index),
            "target_index": target_index,
        })
    return plot_zero_shot_sinusoid_field_and_jump(
        cases, x_grid, y_grid,
        mu_global=float(model._mu_global),
        save_path=save_path, formats=formats,
    )



# --------- 4. FORCING_SEQ TOKENS ----------

def plot_forcing_seq_tokens(
    seed: int = 0,
    save_path: str | Path | None = None,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
    t_s: float = DEFAULT_T_ON,
    t_j: float | None = None,
):
    """Plot the live ``(128, 3)`` forcing tokens per temporal family.

    The plot uses the same ramped waveform, interval integration, and token
    builder as ``ForcingProblem.build_item``. A runtime shape assertion makes
    contract drift fail visibly.
    """
    if t_j is None:
        t_j = t_off
    rng = np.random.default_rng(seed)
    n_tokens = FORCING_TEMPORAL_TOKEN_DIM
    if len(_FORCING_SEQ_TOKEN_LABELS) != n_tokens:
        raise AssertionError(
            "forcing token labels do not match FORCING_TEMPORAL_TOKEN_DIM"
        )
    ramp_seconds = default_ramp_seconds(dt)

    seqs: dict[str, np.ndarray] = {}
    for fam in TEMPORAL_FAMILY_ORDER:
        params = _sample_temporal_params(fam, rng, dt=dt, t_final=t_final,
                                         t_on=t_on, t_off=t_off)
        q = ramped_temporal(fam, params, ramp_seconds)
        t_samples, a_m = _sample_a(q, t_s, t_j, FORCING_TEMPORAL_SAMPLES)
        seq = _forcing_seq_3tok_from_samples(
            t_samples,
            a_m,
            interval_integral_fn=lambda t_lo, t_hi, family=fam, p=params: (
                integrate_temporal_intervals_ramped_signed(
                    family, p, t_lo, t_hi, ramp_seconds
                )
            ),
            A_amp_ref=A_AMP_REF,
        )
        if seq.shape != (FORCING_TEMPORAL_SAMPLES, n_tokens):
            raise AssertionError(
                f"live forcing encoder returned shape {seq.shape}; "
                f"plot expects ({FORCING_TEMPORAL_SAMPLES}, {n_tokens})."
            )
        seqs[fam] = seq

    m = np.arange(FORCING_TEMPORAL_SAMPLES)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            n_tokens, len(TEMPORAL_FAMILY_ORDER),
            figsize=(3.6 * len(TEMPORAL_FAMILY_ORDER), 1.8 * n_tokens),
            constrained_layout=True, squeeze=False, sharex=True,
        )
        for c, fam in enumerate(TEMPORAL_FAMILY_ORDER):
            seq = seqs[fam]
            color = FAMILY_COLORS[fam]
            for r in range(n_tokens):
                ax = axes[r, c]
                ax.plot(m, seq[:, r], color=color, linewidth=1.4)
                ax.axhline(0.0, color="0.5", linewidth=0.6)
                ax.grid(True)
                if c == 0:
                    ax.set_ylabel(_FORCING_SEQ_TOKEN_LABELS[r], fontsize=8)
                if r == 0:
                    ax.set_title(fam)
                if r == n_tokens - 1:
                    ax.set_xlabel("token index m")

        fig.suptitle(
            f"forcing_seq tokens — shape ({FORCING_TEMPORAL_SAMPLES}, {n_tokens}) "
            f"on [t_s, t_j] = [{t_s:.2f}, {t_j:.2f}]  "
            "(live ForcingProblem encoding)"
        )
        _save_figure(fig, save_path, "forcing", "forcing_seq_tokens",
                     layout="constrained")
