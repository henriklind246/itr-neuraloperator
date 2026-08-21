"""Forcing-function visualization for the separable-forcing benchmark.

These plots illustrate the forcing functions used in the
``q_L(y, t) = a(t) * s(y)`` benchmark across four temporal families
(``sin, exp, pulse_train, exp_train``) and four spatial profiles
(``uniform, patch, gaussian, triangle``).

The plots split into two roles:

- **Physical-forcing plots** (``forcing_temporal_families``,
  ``forcing_spatial_profiles``, ``forcing_separable_assembly``) show
  ``a(t)``, ``s(y)``, and the assembled left-boundary flux ``q_L(y, t)``
  that the FV solver applies. This is NOT the exact tensor representation
  seen by the model.

- **Model-side encoding plots** (``forcing_seq_tokens``,
  ``forcing_summary_scalars``) show the representation consumed by the model:
  the ``(64, 5)`` ``forcing_seq`` tokens and the 8-scalar static-conditioning
  summary.

- **Sampling-coverage plots** (``forcing_param_distributions_design``,
  ``forcing_param_distributions_empirical``) compare the intended sampling
  design with what a particular ``sim_params.npy`` actually contains.

All numbers (token count, summary size, conditioning size) are
imported from ``data/dataset.py`` and ``src/physics/boundary_forcing.py``
so the figures track the active branch's contract.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.dataset import (
    A_AMP_REF,
    COND_STATIC_DIM,
    TEMPORAL_SAMPLES,
    _FORCING_SUMMARY_DIM,
    _sample_a,
    build_forcing_seq,
    build_forcing_summary,
)
from src.physics.boundary_forcing import (
    GAUSS_CENTER_RANGE,
    GAUSS_SIGMA_RANGE,
    NP_CHOICES,
    NP_MAX,
    PATCH_W_RANGE,
    PULSE_AMP_RANGE,
    SIN_AMP_RANGE,
    SIN_FREQ_RANGE,
    SPATIAL_BUILDERS,
    SPATIAL_FAMILIES,
    TEMPORAL_BUILDERS,
    TEMPORAL_FAMILY_ORDER,
    TEMPORAL_SAMPLERS,
    SPATIAL_SAMPLERS,
    TRIANGLE_ELL_RANGE,
    build_qL,
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
# data/dataset.py:build_forcing_seq — these labels are the *only* source of
# truth in the plot for what each column means.
_FORCING_SEQ_TOKEN_LABELS = (
    r"$r_m = m / (M-1)$",
    rf"$a(t_m) / A_{{\mathrm{{ref}}}}$  ($A_{{\mathrm{{ref}}}}={A_AMP_REF:.0f}$)",
    r"$A_{\mathrm{cum}}(t_m) / A_{\mathrm{cum,ref}}$",
    r"$(t_j - t_m) / t_{\mathrm{final}}$",
    r"$t_m / t_{\mathrm{final}}$",
)

_FORCING_SUMMARY_LABELS = (
    "S1: signed impulse",
    "S2: abs impulse",
    "S3: positive impulse",
    "S4: negative impulse",
    "S5: mean / A_ref",
    "S6: RMS / A_ref",
    "S7: peak / A_ref",
    "S8: final / A_ref",
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
    plots (forcing_seq_tokens / forcing_summary_scalars).
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
            "forcing_seq_tokens / forcing_summary_scalars)"
        )
        _save_figure(fig, save_path, "forcing", "forcing_separable_assembly",
                     layout="constrained")


# --------- 5. FORCING_SEQ TOKENS ----------

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
    """Plot the (TEMPORAL_SAMPLES, 5) forcing_seq tokens per temporal family.

    Token labels are taken from data/dataset.py (semantics: r, a/A_ref,
    A_cum/A_cum_ref, time-to-target, absolute time). Asserts at runtime
    that build_forcing_seq still returns shape (TEMPORAL_SAMPLES, 5) so
    this plot fails loudly if the contract changes.
    """
    if t_j is None:
        t_j = t_off
    rng = np.random.default_rng(seed)
    n_tokens = len(_FORCING_SEQ_TOKEN_LABELS)

    seqs: dict[str, np.ndarray] = {}
    for fam in TEMPORAL_FAMILY_ORDER:
        params = _sample_temporal_params(fam, rng, dt=dt, t_final=t_final,
                                         t_on=t_on, t_off=t_off)
        builder = TEMPORAL_BUILDERS[fam]
        q = builder(**params)
        seq = build_forcing_seq(q, t_s=t_s, t_j=t_j, t_final=t_final)
        if seq.shape != (TEMPORAL_SAMPLES, n_tokens):
            raise AssertionError(
                f"build_forcing_seq returned shape {seq.shape}; "
                f"plot expects ({TEMPORAL_SAMPLES}, {n_tokens}). "
                "Update _FORCING_SEQ_TOKEN_LABELS to match data/dataset.py."
            )
        seqs[fam] = seq

    m = np.arange(TEMPORAL_SAMPLES)

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
            f"forcing_seq tokens — shape ({TEMPORAL_SAMPLES}, {n_tokens}) "
            f"on [t_s, t_j] = [{t_s:.2f}, {t_j:.2f}]  "
            "(labels from data/dataset.py)"
        )
        _save_figure(fig, save_path, "forcing", "forcing_seq_tokens",
                     layout="constrained")


# --------- 6. FORCING SUMMARY SCALARS ----------

def plot_forcing_summary_scalars(
    n_samples: int = 800,
    n_bins: int = 30,
    seed: int = 0,
    save_path: str | Path | None = None,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
    t_s: float = DEFAULT_T_ON,
    t_j: float | None = None,
):
    """Histograms of the 8 build_forcing_summary scalars per family.

    Histograms (not KDEs) because several scalars are bounded, sign-fixed,
    or zero by construction for some families. Caption surfaces the active
    static-conditioning width so the plot is read against the live
    contract.
    """
    if t_j is None:
        t_j = t_off
    rng = np.random.default_rng(seed)

    summaries: dict[str, np.ndarray] = {}
    for fam in TEMPORAL_FAMILY_ORDER:
        rows = []
        builder = TEMPORAL_BUILDERS[fam]
        for _ in range(n_samples):
            params = _sample_temporal_params(fam, rng, dt=dt, t_final=t_final,
                                             t_on=t_on, t_off=t_off)
            q = builder(**params)
            t_samples, a_m = _sample_a(q, t_s, t_j, TEMPORAL_SAMPLES)
            rows.append(build_forcing_summary(a_m, t_samples, t_s, t_j, t_final))
        arr = np.asarray(rows, dtype=float)
        if arr.shape[1] != _FORCING_SUMMARY_DIM:
            raise AssertionError(
                f"build_forcing_summary returned width {arr.shape[1]}; "
                f"plot expects {_FORCING_SUMMARY_DIM}."
            )
        summaries[fam] = arr

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
        for k, label in enumerate(_FORCING_SUMMARY_LABELS):
            ax = axes.ravel()[k]
            data_by_fam = [summaries[fam][:, k] for fam in TEMPORAL_FAMILY_ORDER]
            lo = float(min(arr.min() for arr in data_by_fam))
            hi = float(max(arr.max() for arr in data_by_fam))
            if hi <= lo:
                hi = lo + 1e-6
            bins = np.linspace(lo, hi, n_bins + 1)
            for fam, arr in zip(TEMPORAL_FAMILY_ORDER, data_by_fam):
                ax.hist(
                    arr, bins=bins, histtype="step", linewidth=1.6,
                    color=FAMILY_COLORS[fam], label=fam,
                )
            ax.set_title(label)
            ax.grid(True)
            if k == 0:
                ax.legend(loc="best", fontsize=8)

        fig.suptitle(
            "Static-conditioning forcing summary (per family histograms)  —  "
            f"COND_STATIC_DIM={COND_STATIC_DIM}, "
            f"summary block has {_FORCING_SUMMARY_DIM} scalars "
            f"(N={n_samples} per family)"
        )
        _save_figure(fig, save_path, "forcing", "forcing_summary_scalars",
                     layout="constrained")


# --------- 7a. PARAMETER DISTRIBUTIONS — DESIGN ----------

def _gather_design_temporal_params(
    n_samples: int,
    seed: int,
    dt: float,
    t_final: float,
    t_on: float,
    t_off: float,
) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict]] = {fam: [] for fam in TEMPORAL_FAMILY_ORDER}
    for fam in TEMPORAL_FAMILY_ORDER:
        for _ in range(n_samples):
            out[fam].append(
                _sample_temporal_params(fam, rng, dt=dt, t_final=t_final,
                                        t_on=t_on, t_off=t_off)
            )
    return out


def _gather_design_spatial_params(n_samples: int, seed: int) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed + 7919)
    out: dict[str, list[dict]] = {fam: [] for fam in SPATIAL_FAMILY_ORDER}
    for fam in SPATIAL_FAMILY_ORDER:
        if fam == "uniform":
            continue
        for _ in range(n_samples):
            out[fam].append(SPATIAL_SAMPLERS[fam](rng))
    return out


def _render_param_distributions_figure(
    temporal_params: dict[str, list[dict]],
    spatial_params: dict[str, list[dict]],
    title: str,
    save_name: str,
    save_path: str | Path | None,
):
    """Shared renderer for design and empirical param-distribution figures."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)

        # Row 0: sin (A, log f) scatter + Np-style placeholder + 2 marginals
        sin_params = temporal_params.get("sin", [])
        if sin_params:
            A = np.array([p["A"] for p in sin_params])
            f = np.array([p["f"] for p in sin_params])
            ax = axes[0, 0]
            ax.scatter(A, f, s=8, alpha=0.45, color=FAMILY_COLORS["sin"])
            ax.set_yscale("log")
            ax.set_xlabel("A")
            ax.set_ylabel("f (log)")
            ax.set_title("sin: (A, f)")
            ax.grid(True)

            axes[0, 1].hist(A, bins=30, color=FAMILY_COLORS["sin"], alpha=0.7)
            axes[0, 1].set_xlabel("A"); axes[0, 1].set_title("sin A marginal")
            axes[0, 1].grid(True)
            axes[0, 2].hist(np.log10(f), bins=30, color=FAMILY_COLORS["sin"], alpha=0.7)
            axes[0, 2].set_xlabel(r"$\log_{10} f$"); axes[0, 2].set_title("sin log f marginal")
            axes[0, 2].grid(True)
        else:
            for c in (0, 1, 2):
                axes[0, c].set_visible(False)
        axes[0, 3].set_visible(False)

        # Row 1: exp (t0, log tau, A)
        exp_params = temporal_params.get("exp", [])
        if exp_params:
            t0 = np.array([p["t0"] for p in exp_params])
            tau = np.array([p["tau"] for p in exp_params])
            A = np.array([p["A"] for p in exp_params])
            axes[1, 0].hist(t0, bins=30, color=FAMILY_COLORS["exp"], alpha=0.7)
            axes[1, 0].set_xlabel("t0"); axes[1, 0].set_title("exp t0")
            axes[1, 0].grid(True)
            axes[1, 1].hist(np.log10(tau), bins=30, color=FAMILY_COLORS["exp"], alpha=0.7)
            axes[1, 1].set_xlabel(r"$\log_{10}\tau$"); axes[1, 1].set_title("exp tau")
            axes[1, 1].grid(True)
            axes[1, 2].hist(A, bins=30, color=FAMILY_COLORS["exp"], alpha=0.7)
            axes[1, 2].set_xlabel("A"); axes[1, 2].set_title("exp A")
            axes[1, 2].grid(True)
        else:
            for c in (0, 1, 2):
                axes[1, c].set_visible(False)
        axes[1, 3].set_visible(False)

        # Row 2: pulse_train — Np histogram + flattened per-pulse t, A, dt
        pt_params = temporal_params.get("pulse_train", [])
        if pt_params:
            Np = np.array([p["Np"] for p in pt_params])
            t_all = np.concatenate([np.asarray(p["t_list"], dtype=float) for p in pt_params])
            A_all = np.concatenate([np.asarray(p["A_list"], dtype=float) for p in pt_params])
            dt_all = np.concatenate([np.asarray(p["dt_list"], dtype=float) for p in pt_params])
            axes[2, 0].hist(Np, bins=np.arange(0.5, NP_MAX + 1.5),
                            color=FAMILY_COLORS["pulse_train"], alpha=0.7)
            axes[2, 0].set_xlabel("Np"); axes[2, 0].set_title("pulse_train Np")
            axes[2, 0].set_xticks(list(NP_CHOICES))
            axes[2, 0].grid(True)
            axes[2, 1].hist(t_all, bins=30, color=FAMILY_COLORS["pulse_train"], alpha=0.7)
            axes[2, 1].set_xlabel("t_n"); axes[2, 1].set_title("pulse_train onset")
            axes[2, 1].grid(True)
            axes[2, 2].hist(A_all, bins=30, color=FAMILY_COLORS["pulse_train"], alpha=0.7)
            axes[2, 2].set_xlabel("A_n"); axes[2, 2].set_title("pulse_train amp")
            axes[2, 2].grid(True)
            axes[2, 3].hist(np.log10(dt_all), bins=30, color=FAMILY_COLORS["pulse_train"], alpha=0.7)
            axes[2, 3].set_xlabel(r"$\log_{10}\Delta t_n$"); axes[2, 3].set_title("pulse_train dt")
            axes[2, 3].grid(True)
        else:
            for c in range(4):
                axes[2, c].set_visible(False)

        # Row 3: exp_train — Np histogram + flattened per-pulse t, A, tau
        et_params = temporal_params.get("exp_train", [])
        if et_params:
            Np = np.array([p["Np"] for p in et_params])
            t_all = np.concatenate([np.asarray(p["t_list"], dtype=float) for p in et_params])
            A_all = np.concatenate([np.asarray(p["A_list"], dtype=float) for p in et_params])
            tau_all = np.concatenate([np.asarray(p["tau_list"], dtype=float) for p in et_params])
            axes[3, 0].hist(Np, bins=np.arange(0.5, NP_MAX + 1.5),
                            color=FAMILY_COLORS["exp_train"], alpha=0.7)
            axes[3, 0].set_xlabel("Np"); axes[3, 0].set_title("exp_train Np")
            axes[3, 0].set_xticks(list(NP_CHOICES))
            axes[3, 0].grid(True)
            axes[3, 1].hist(t_all, bins=30, color=FAMILY_COLORS["exp_train"], alpha=0.7)
            axes[3, 1].set_xlabel("t_n"); axes[3, 1].set_title("exp_train onset")
            axes[3, 1].grid(True)
            axes[3, 2].hist(A_all, bins=30, color=FAMILY_COLORS["exp_train"], alpha=0.7)
            axes[3, 2].set_xlabel("A_n"); axes[3, 2].set_title("exp_train amp")
            axes[3, 2].grid(True)
            axes[3, 3].hist(np.log10(tau_all), bins=30, color=FAMILY_COLORS["exp_train"], alpha=0.7)
            axes[3, 3].set_xlabel(r"$\log_{10}\tau_n$"); axes[3, 3].set_title("exp_train tau")
            axes[3, 3].grid(True)
        else:
            for c in range(4):
                axes[3, c].set_visible(False)

        fig.suptitle(title)
        _save_figure(fig, save_path, "forcing", save_name, layout="constrained")


def _render_spatial_param_distributions(
    spatial_params: dict[str, list[dict]],
    title: str,
    save_name: str,
    save_path: str | Path | None,
):
    """Per-spatial-family distributions of (y_c, w/sigma_y/ell)."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)

        rows = [
            ("patch",    "w",       (PATCH_W_RANGE[0], PATCH_W_RANGE[1]),     "linear"),
            ("gaussian", "sigma_y", (GAUSS_SIGMA_RANGE[0], GAUSS_SIGMA_RANGE[1]), "log"),
            ("triangle", "ell",     (TRIANGLE_ELL_RANGE[0], TRIANGLE_ELL_RANGE[1]), "linear"),
        ]
        for r, (fam, scale_key, scale_range, scale_kind) in enumerate(rows):
            color = f"C{SPATIAL_FAMILY_ORDER.index(fam)}"
            params = spatial_params.get(fam, [])
            if not params:
                for c in (0, 1):
                    axes[r, c].set_visible(False)
                continue
            y_c = np.array([p["y_c"] for p in params])
            scale = np.array([p[scale_key] for p in params])
            axes[r, 0].hist(y_c, bins=30, color=color, alpha=0.7)
            axes[r, 0].set_xlabel("y_c"); axes[r, 0].set_title(f"{fam} y_c")
            axes[r, 0].grid(True)
            if scale_kind == "log":
                axes[r, 1].hist(np.log10(scale), bins=30, color=color, alpha=0.7)
                axes[r, 1].set_xlabel(rf"$\log_{{10}} {scale_key}$")
            else:
                axes[r, 1].hist(scale, bins=30, color=color, alpha=0.7)
                axes[r, 1].set_xlabel(scale_key)
            axes[r, 1].set_title(f"{fam} {scale_key}")
            axes[r, 1].grid(True)

        fig.suptitle(title)
        _save_figure(fig, save_path, "forcing", save_name, layout="constrained")


def plot_forcing_param_distributions_design(
    n_samples: int = 1000,
    seed: int = 0,
    save_path: str | Path | None = None,
    dt: float = DEFAULT_DT,
    t_final: float = DEFAULT_T_FINAL,
    t_on: float = DEFAULT_T_ON,
    t_off: float = DEFAULT_T_OFF,
    spatial_save_path: str | Path | None = None,
):
    """Sampler-driven coverage of temporal + spatial parameters.

    Calls the samplers directly — independent of any saved dataset.
    Renders two figures: one for temporal parameters and one for spatial.
    """
    temporal = _gather_design_temporal_params(
        n_samples, seed, dt=dt, t_final=t_final, t_on=t_on, t_off=t_off,
    )
    spatial = _gather_design_spatial_params(n_samples, seed)
    _render_param_distributions_figure(
        temporal_params=temporal,
        spatial_params=spatial,
        title=f"Forcing parameter design distributions  —  sampler-driven (N={n_samples} per family)",
        save_name="forcing_param_distributions_design",
        save_path=save_path,
    )
    if spatial_save_path is not None or save_path is not None:
        # Optional companion plot for spatial families (kept off the main
        # registry; saved next to the temporal figure for inspection).
        target = (
            Path(spatial_save_path)
            if spatial_save_path is not None
            else Path(save_path).with_name("forcing_param_distributions_design_spatial.png")
            if save_path is not None
            else None
        )
        _render_spatial_param_distributions(
            spatial_params=spatial,
            title=f"Spatial parameter design distributions (N={n_samples} per family)",
            save_name="forcing_param_distributions_design_spatial",
            save_path=target,
        )


# --------- 7b. PARAMETER DISTRIBUTIONS — EMPIRICAL ----------

def _split_sim_params_by_family(sim_params) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Bucket sim_params into per-family parameter dicts.

    Accepts either an object-array of per-sim dicts (the format saved by
    data/generate_dataset.py) or an iterable of such dicts. Returns
    (temporal_by_family, spatial_by_family) using the same keys as the
    design distribution renderer expects.
    """
    temporal: dict[str, list[dict]] = defaultdict(list)
    spatial: dict[str, list[dict]] = defaultdict(list)
    for entry in sim_params:
        record = entry if isinstance(entry, dict) else dict(entry)
        tfam = record.get("temporal_family")
        sfam = record.get("spatial_family")
        if tfam in TEMPORAL_FAMILY_ORDER:
            temporal[tfam].append(dict(record.get("temporal_params", {})))
        if sfam in SPATIAL_FAMILY_ORDER:
            spatial[sfam].append(dict(record.get("spatial_params", {})))
    return temporal, spatial


def plot_forcing_param_distributions_empirical(
    sim_params,
    save_path: str | Path | None = None,
    spatial_save_path: str | Path | None = None,
):
    """Empirical distribution of recorded forcing parameters in a dataset.

    `sim_params` is the object array saved by data/generate_dataset.py.
    Renders the same panels as the design plot, populated from the
    recorded entries.
    """
    temporal, spatial = _split_sim_params_by_family(sim_params)
    n = sum(len(v) for v in temporal.values())
    _render_param_distributions_figure(
        temporal_params=temporal,
        spatial_params=spatial,
        title=f"Forcing parameter empirical distributions  —  from sim_params (N={n} sims)",
        save_name="forcing_param_distributions_empirical",
        save_path=save_path,
    )
    if spatial_save_path is not None or save_path is not None:
        target = (
            Path(spatial_save_path)
            if spatial_save_path is not None
            else Path(save_path).with_name("forcing_param_distributions_empirical_spatial.png")
            if save_path is not None
            else None
        )
        _render_spatial_param_distributions(
            spatial_params=spatial,
            title=f"Spatial parameter empirical distributions (N={n} sims)",
            save_name="forcing_param_distributions_empirical_spatial",
            save_path=target,
        )
