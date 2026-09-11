"""F01 and F02: the governing problem and the model architecture.

Both figures are analytic. They require no run artifacts, so ``figures.yaml``
gives them an empty ``requires`` list and ``metric_space: none`` -- they are
still rendered through :func:`visual.pub.registry.render` and still write a
provenance sidecar, because a figure that cannot say which commit drew it is not
publishable either.

Neither figure hard-codes a tensor contract. Channel counts, conditioning
widths, and token shapes come from ``get_problem(...).dims`` at render time, so
the drawn architecture cannot drift away from the code the way ``CLAUDE.md``'s
table did.
"""

from __future__ import annotations

import numpy as np

from visual.pub import panels, style

_T_FINAL = 0.3
_DT = 0.005
_T_ON = 0.0
_T_OFF = 0.2

_TEMPORAL_ORDER = ("sin", "exp", "pulse_train", "exp_train")
_SPATIAL_ORDER = ("uniform", "patch", "gaussian", "triangle")
_BENCHMARK_ORDER = ("forcing", "source", "source_itr_sin", "interfaces")

_K_LEFT = 2.0
_K_RIGHT = 1.0
_INTERFACE_X = 0.5


# ============================================================
# F01 -- governing problem and benchmark suite
# ============================================================


def _draw_geometry(ax) -> None:
    """The two-slab domain, the thin-resistance interface, and the BCs."""
    import matplotlib as mpl

    panels.blank_canvas(ax, xlim=(-0.32, 1.32), ylim=(-0.30, 1.52))

    ax.add_patch(mpl.patches.Rectangle((0.0, 0.0), _INTERFACE_X, 1.0,
                                       facecolor="#cfe3f2", edgecolor="0.3",
                                       linewidth=0.8, zorder=1))
    ax.add_patch(mpl.patches.Rectangle((_INTERFACE_X, 0.0), 1.0 - _INTERFACE_X, 1.0,
                                       facecolor="#f7e3c8", edgecolor="0.3",
                                       linewidth=0.8, zorder=1))
    ax.plot([_INTERFACE_X, _INTERFACE_X], [0.0, 1.0], color="#b2182b",
            linewidth=1.6, zorder=3)

    ax.text(0.25, 0.55, f"$k_L = {_K_LEFT:g}$", ha="center", va="center", fontsize=7)
    ax.text(0.75, 0.55, f"$k_R = {_K_RIGHT:g}$", ha="center", va="center", fontsize=7)
    ax.text(0.25, 0.40, r"$\rho c_p = 1$", ha="center", va="center",
            fontsize=6.5, color="0.35")
    ax.text(0.75, 0.40, r"$\rho c_p = 1$", ha="center", va="center",
            fontsize=6.5, color="0.35")

    ax.annotate(r"$R_c(y)$: contact resistance",
                xy=(_INTERFACE_X, 0.72), xytext=(0.86, 1.14),
                fontsize=6.5, color="#b2182b", ha="center",
                arrowprops=dict(arrowstyle="-", color="#b2182b", linewidth=0.7))
    ax.text(_INTERFACE_X, -0.03, r"$x_I$", ha="center", va="top",
            fontsize=6.5, color="#b2182b")

    for y in (0.20, 0.45, 0.70, 0.92):
        panels.schematic_arrow(ax, (-0.22, y), (-0.015, y), color="#0072B2",
                               linewidth=1.0, style_str="-|>")
    ax.text(-0.24, 0.55, r"$q_L(y,t)$", ha="right", va="center", fontsize=7,
            color="#0072B2", rotation=90)

    ax.text(1.03, 0.5, r"$T = 300$ K", ha="left", va="center", fontsize=7,
            color="0.25")
    ax.plot([1.0, 1.0], [0.0, 1.0], color="0.25", linewidth=1.6, zorder=3)

    for y, va in ((1.0, "bottom"), (0.0, "top")):
        ax.plot([0.0, 1.0], [y, y], color="0.45", linewidth=1.0,
                linestyle=(0, (4, 2)), zorder=3)
    ax.text(0.22, 1.03, r"adiabatic:  $\partial T/\partial y = 0$", ha="center",
            va="bottom", fontsize=6.0, color="0.45")
    ax.text(0.22, -0.16, r"adiabatic:  $\partial T/\partial y = 0$", ha="center",
            va="center", fontsize=6.0, color="0.45")

    ax.text(0.5, 1.50,
            r"$\rho c_p\,\partial_t T = \nabla\!\cdot\!(k\nabla T) + f$"
            "\n"
            r"$-k\,\partial_x T|_{x_I^{\pm}} = q_n = \Delta T_{\mathrm{contact}}\,/\,R_c(y)$",
            ha="center", va="top", fontsize=6.8, linespacing=1.6)


def _sample_temporal(family: str, seed: int):
    from src.physics.boundary_forcing import TEMPORAL_SAMPLERS, ramped_temporal

    rng = np.random.default_rng(seed)
    params = TEMPORAL_SAMPLERS[family](rng, dt=_DT, t_final=_T_FINAL,
                                       t_on=_T_ON, t_off=_T_OFF)
    return ramped_temporal(family, params, 2.0 * _DT)


def _draw_forcing_panel(ax) -> None:
    """What varies in ``forcing``: four a(t) families x four s(y) profiles."""
    from src.physics.boundary_forcing import SPATIAL_BUILDERS

    t = np.linspace(0.0, _T_FINAL, 400)
    inset = ax.inset_axes((0.58, 0.62, 0.40, 0.34))

    for k, family in enumerate(_TEMPORAL_ORDER):
        a = np.asarray([float(_sample_temporal(family, 11 + k)(tv)) for tv in t])
        peak = float(np.max(np.abs(a))) or 1.0
        ax.plot(t, a / peak, linewidth=0.9,
                color=style.plt.get_cmap(style.CMAP_SEQ_ORDINAL)(k / 3.0),
                label=family)

    y = np.linspace(0.0, 1.0, 200)
    spatial_params = {
        "uniform": {},
        "patch": {"y_c": 0.42, "w": 0.30},
        "gaussian": {"y_c": 0.62, "sigma_y": 0.10},
        "triangle": {"y_c": 0.35, "ell": 0.22},
    }
    for k, family in enumerate(_SPATIAL_ORDER):
        s = np.asarray(SPATIAL_BUILDERS[family](y, **spatial_params[family]),
                       dtype=float)
        inset.plot(s, y, linewidth=0.8,
                   color=style.plt.get_cmap(style.CMAP_SEQ_ORDINAL)(k / 3.0))
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(r"$s(y)$", fontsize=5.5, pad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)

    ax.axhline(0.0, color="0.6", linewidth=0.5)
    ax.set_ylim(-0.08, 1.80)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$a(t)\,/\,\max|a|$")
    ax.set_title("forcing\n" r"$q_L = a(t)\,s(y)$,  scalar $R_c$",
                 fontsize=7, color=style.benchmark_color("forcing"))
    ax.legend(fontsize=5.0, loc="upper left", ncol=1, handlelength=1.1,
              labelspacing=0.25, borderpad=0.2)


def _draw_source_panel(ax) -> None:
    """What varies in ``source``: chip-patch position, size and amplitude."""
    import matplotlib as mpl
    from src.physics.internal_source import (
        PATCH_A_RANGE, PATCH_H, PATCH_W, PATCH_X_RANGE,
    )

    panels.blank_canvas(ax, xlim=(-0.06, 1.06), ylim=(-0.06, 1.20))
    ax.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, facecolor="0.96",
                                       edgecolor="0.4", linewidth=0.7))
    ax.axvline(_INTERFACE_X, color="#b2182b", linewidth=1.2)

    regimes = (("left", 0.24, 0.30), ("near", 0.44, 0.62), ("right", 0.74, 0.40))
    for name, x_h, y_h in regimes:
        ax.add_patch(mpl.patches.Rectangle(
            (x_h - PATCH_W / 2, y_h - PATCH_H / 2), PATCH_W, PATCH_H,
            facecolor=style.benchmark_color("source"), alpha=0.55,
            edgecolor="0.2", linewidth=0.7, zorder=2))
        ax.text(x_h, y_h - PATCH_H / 2 - 0.035, name, ha="center", va="top",
                fontsize=5.5, color="0.3")

    ax.text(0.5, 1.16,
            f"$x_h \\in [{PATCH_X_RANGE[0]:.2f}, {PATCH_X_RANGE[1]:.2f}]$,  "
            f"$A \\in [{PATCH_A_RANGE[0]:.0f}, {PATCH_A_RANGE[1]:.0f}]$",
            ha="center", va="top", fontsize=5.8, color="0.3")
    ax.set_title("source\n" r"volumetric patch $f(x,y)$, scalar $R_c$",
                 fontsize=7, color=style.benchmark_color("source"))


def _draw_source_itr_sin_panel(ax) -> None:
    from src.physics.internal_source import make_rc_sin_profile

    y = np.linspace(0.0, 1.0, 400)
    profiles = ((0.15, 0.5), (0.15, 2.4), (0.50, 1.0))
    cmap = style.plt.get_cmap(style.CMAP_SEQ_ORDINAL)
    for k, (base, amp) in enumerate(profiles):
        ax.plot(y, make_rc_sin_profile(y, base, amp), linewidth=1.0,
                color=cmap(k / (len(profiles) - 1)), label=f"A = {amp:.2f}")
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$R_c(y)$ [m$^2$ K/W]")
    ax.set_title("Source + ITR\n" r"$R_c(y)=R_b+A\sin(\pi y)$",
                 fontsize=7, color=style.benchmark_color("source_itr_sin"))
    ax.legend(fontsize=5.0, loc="upper right", handlelength=1.2, borderpad=0.2)


def _draw_interfaces_panel(ax) -> None:
    """What varies in ``interfaces``: the interface location and scalar R_c."""
    import matplotlib as mpl
    from problems.interfaces import INTERFACE_X_RANGE, RC_RANGE

    panels.blank_canvas(ax, xlim=(-0.06, 1.06), ylim=(-0.06, 1.20))
    ax.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, facecolor="0.96",
                                       edgecolor="0.4", linewidth=0.7))
    lo, hi = INTERFACE_X_RANGE
    ax.axvspan(lo, hi, ymin=0.0, ymax=1.0 / 1.26, color="#009E73", alpha=0.13,
               linewidth=0)
    for x_I, alpha in ((lo, 0.45), (0.5, 0.75), (hi, 1.0)):
        ax.plot([x_I, x_I], [0, 1], color="#b2182b", linewidth=1.2, alpha=alpha)
    ax.annotate("", xy=(lo, -0.03), xytext=(hi, -0.03),
                arrowprops=dict(arrowstyle="<|-|>", color="0.35", linewidth=0.7))
    ax.text(0.5, 1.16,
            f"$x_I \\in [{lo:g}, {hi:g}]$,  "
            f"$R_c \\in [{RC_RANGE[0]:g}, {RC_RANGE[1]:g}]$",
            ha="center", va="top", fontsize=5.8, color="0.3")
    ax.set_title("interfaces\n" r"$x_I$ varies, fixed sin/uniform $q_L$",
                 fontsize=7, color=style.benchmark_color("interfaces"))


def problem_schematic(*, source=None, spec=None, requirement=None):
    """F01 -- the governing problem and what each benchmark varies."""
    import matplotlib.pyplot as plt

    width = spec.width if spec is not None else "two_col"
    fig = plt.figure(figsize=style.figsize(width, rows=2, row_height="std",
                                           extra_in=0.15))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.15, 1.0), hspace=0.30,
                          wspace=0.46, left=0.07, right=0.985, top=0.97,
                          bottom=0.12)

    ax_geo = fig.add_subplot(gs[0, :])
    _draw_geometry(ax_geo)

    ax_forcing = fig.add_subplot(gs[1, 0])
    ax_source = fig.add_subplot(gs[1, 1])
    ax_itr = fig.add_subplot(gs[1, 2])
    ax_iface = fig.add_subplot(gs[1, 3])

    _draw_forcing_panel(ax_forcing)
    _draw_source_panel(ax_source)
    _draw_source_itr_sin_panel(ax_itr)
    _draw_interfaces_panel(ax_iface)

    style.panel_letters([ax_geo], loc=(-0.02, 1.00))
    style.panel_letters([ax_forcing, ax_source, ax_itr, ax_iface], start=1,
                        loc=(-0.20, 1.16))

    metric_definition = {
        "figure": "schematic; no measured quantity",
        "geometry": "two vertical slabs on [0,1]^2, interface at x = x_I",
        "boundary_conditions": "Neumann left q_L(y,t), Dirichlet right T = 300 K, "
                               "adiabatic top and bottom",
        "benchmarks": list(_BENCHMARK_ORDER),
    }
    return fig, None, metric_definition


# ============================================================
# F02 -- architecture
# ============================================================


def _real_forcing_tokens(seed: int = 3) -> tuple[np.ndarray, str]:
    """A genuine ``(M, 3)`` point-plus-interval-average token stream.

    Built with the same two functions the dataset uses, so the panel shows the
    live contract rather than a redrawing of it. Returns the tokens and the
    temporal family they came from.
    """
    from problems.forcing import (
        A_AMP_REF,
        _forcing_seq_3tok_from_samples,
        _sample_a,
    )
    from src.physics.boundary_forcing import (
        FORCING_TEMPORAL_SAMPLES,
        TEMPORAL_SAMPLERS,
        integrate_temporal_ramped_signed,
        ramped_temporal,
    )

    family = "pulse_train"
    params = TEMPORAL_SAMPLERS[family](
        np.random.default_rng(seed),
        dt=_DT,
        t_final=_T_FINAL,
        t_on=_T_ON,
        t_off=_T_OFF,
    )
    ramp_seconds = 2.0 * _DT
    q = ramped_temporal(family, params, ramp_seconds)
    t_samples, a_m = _sample_a(q, _T_ON, _T_OFF, FORCING_TEMPORAL_SAMPLES)
    return _forcing_seq_3tok_from_samples(
        t_samples,
        a_m,
        interval_integral_fn=lambda a, b: integrate_temporal_ramped_signed(
            family, params, a, b, ramp_seconds
        ),
        A_amp_ref=A_AMP_REF,
    ), family


def _draw_architecture_flow(ax, dims, n_layers: int, modes: tuple[int, int],
                            width_ch: int) -> None:
    panels.blank_canvas(ax, xlim=(0.0, 1.0), ylim=(0.05, 0.94))

    spatial = panels.schematic_box(
        ax, (0.01, 0.60), 0.15, 0.26,
        f"spatial\n$(N_x, N_y, {dims.in_channels})$",
        facecolor="#cfe3f2")
    lift = panels.schematic_box(ax, (0.24, 0.60), 0.11, 0.26,
                                f"lift\n1x1 conv\n$\\rightarrow {width_ch}$",
                                facecolor="0.93")
    blocks = panels.schematic_box(
        ax, (0.42, 0.56), 0.26, 0.34,
        f"{n_layers} x spectral block\n"
        f"modes $({modes[0]}, {modes[1]})$\n"
        "FFT $\\rightarrow$ $R_\\theta$ $\\rightarrow$ iFFT  $+$  1x1",
        facecolor="#e8e8e8")
    project = panels.schematic_box(ax, (0.75, 0.60), 0.11, 0.26,
                                   "project\n$\\rightarrow 1$", facecolor="0.93")
    out = panels.schematic_box(ax, (0.90, 0.60), 0.09, 0.26,
                               "$\\hat{T}$\n$(N_x, N_y)$", facecolor="#f7e3c8")

    # The CIN sits between the encoder and cond_static so that h_a and the
    # static conditioning meet it head on; nothing has to cross a box.
    seq = panels.schematic_box(
        ax, (0.01, 0.10), 0.15, 0.24,
        f"forcing_seq\n$({_token_count()}, {dims.temporal_token_dim})$\n"
        "$[r_m,\\,a_m/A_{ref}]$",
        facecolor="#d9ecd9")
    encoder = panels.schematic_box(ax, (0.24, 0.10), 0.11, 0.24,
                                   "temporal\nencoder\n$\\rightarrow h_a$",
                                   facecolor="#d9ecd9")
    cin = panels.schematic_box(ax, (0.44, 0.10), 0.14, 0.24,
                               "CIN MLP\n$(\\gamma, \\beta)$", facecolor="0.93")
    cond = panels.schematic_box(
        ax, (0.68, 0.10), 0.16, 0.24,
        f"cond_static\n$({dims.cond_static_dim},)$", facecolor="#cfe3f2")

    for a, b in ((spatial, lift), (lift, blocks), (blocks, project),
                 (project, out)):
        panels.schematic_arrow(ax, (a[0] + 0.055, a[1]), (b[0] - 0.055, b[1]))
    panels.schematic_arrow(ax, (seq[0] + 0.077, seq[1]),
                           (encoder[0] - 0.057, encoder[1]))
    panels.schematic_arrow(ax, (encoder[0] + 0.057, encoder[1]),
                           (cin[0] - 0.072, cin[1]), label=r"$h_a$")
    panels.schematic_arrow(ax, (cond[0] - 0.082, cond[1]),
                           (cin[0] + 0.072, cin[1]))
    panels.schematic_arrow(ax, (cin[0], cin[1] + 0.125),
                           (0.56, blocks[1] - 0.175))
    panels.schematic_arrow(
        ax, (encoder[0], encoder[1] + 0.125), (0.455, blocks[1] - 0.175),
        connection="arc3,rad=0.30")
    ax.text(0.30, 0.49, r"$s_y[\cdot,%d]\, z_a$" % dims.s_y_channel,
            fontsize=5.5, color="0.35", ha="left")
    ax.text(0.575, 0.45, r"$\gamma,\beta$ per block", fontsize=5.5,
            color="0.35", ha="left")


def _token_count() -> int:
    from src.physics.boundary_forcing import FORCING_TEMPORAL_SAMPLES

    return int(FORCING_TEMPORAL_SAMPLES)


def architecture(*, source=None, spec=None, requirement=None,
                 benchmark: str = "forcing"):
    """F02 -- the FNO with its temporal forcing encoder.

    Channel counts come from the live ``ProblemDims``; the token panel is a real
    ``forcing_seq``, not a redrawing of one.
    """
    import matplotlib.pyplot as plt
    from problems.registry import get_problem

    problem = get_problem(benchmark, "temporal_encoder")
    dims = problem.dims

    tokens, family = _real_forcing_tokens()
    n_layers, modes, width_ch = _model_defaults()

    width = spec.width if spec is not None else "two_col"
    fig = plt.figure(figsize=style.figsize(width, rows=2, row_height="std",
                                           extra_in=-0.15))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0), hspace=0.22,
                          wspace=0.38, left=0.08, right=0.98, top=0.97,
                          bottom=0.13)

    ax_flow = fig.add_subplot(gs[0, :])
    _draw_architecture_flow(ax_flow, dims, n_layers, modes, width_ch)

    m = np.arange(tokens.shape[0])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.plot(m, tokens[:, 0], color="0.35", linewidth=1.0)
    ax_r.set_xlabel("token index $m$")
    ax_r.set_ylabel(r"$r_m$")
    ax_r.set_title(r"token 0: $r_m = m/(M-1)$", fontsize=7)

    ax_a = fig.add_subplot(gs[1, 1])
    ax_a.plot(m, tokens[:, 1], color=style.benchmark_color(benchmark), linewidth=1.0)
    ax_a.axhline(0.0, color="0.6", linewidth=0.5)
    ax_a.set_xlabel("token index $m$")
    ax_a.set_ylabel(r"$a_m / A_{\mathrm{ref}}$")
    ax_a.set_title(f"token 1: amplitude ({family})", fontsize=7)

    ax_tab = fig.add_subplot(gs[1, 2])
    _draw_contract_table(ax_tab, benchmark, dims, tokens.shape)

    style.panel_letters([ax_flow], loc=(-0.01, 0.99))
    style.panel_letters([ax_r, ax_a, ax_tab], start=1, loc=(-0.22, 1.06))

    metric_definition = {
        "figure": "architecture schematic; no measured quantity",
        "benchmark": benchmark,
        "representation": "temporal_encoder",
        "in_channels": dims.in_channels,
        "cond_static_dim": dims.cond_static_dim,
        "forcing_seq_shape": list(tokens.shape),
        "temporal_token_dim": dims.temporal_token_dim,
        "s_y_channel": dims.s_y_channel,
        "use_forcing_time_aug": dims.use_forcing_time_aug,
        "tokens": "[r_m, a_m / A_ref] sampled on [t_s, t_j]",
        "dims_source": "problems.registry.get_problem(...).dims at render time",
    }
    return fig, None, metric_definition


def _model_defaults() -> tuple[int, tuple[int, int], int]:
    """Read layer count, modes, and width from ``conf/config.yaml``."""
    import yaml

    from visual.pub.manifest import PROJECT_ROOT

    cfg = yaml.safe_load((PROJECT_ROOT / "conf" / "config.yaml").read_text()) or {}
    params = (cfg.get("model") or {}).get("parameters") or {}
    return (int(params.get("n_layers", 4)),
            (int(params.get("modes1", 12)), int(params.get("modes2", 12))),
            int(params.get("width", 32)))


def _draw_contract_table(ax, benchmark: str, dims, token_shape) -> None:
    panels.blank_canvas(ax, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    rows = (
        ("benchmark", benchmark),
        ("representation", "temporal_encoder"),
        ("in_channels", str(dims.in_channels)),
        ("cond_static_dim", str(dims.cond_static_dim)),
        ("forcing_seq", f"{token_shape[0]} x {token_shape[1]}"),
        ("t_stats_dim", str(dims.t_stats_dim)),
        ("s_y_channel", str(dims.s_y_channel)),
        ("time aug", "yes" if dims.use_forcing_time_aug else "no"),
    )
    ax.set_title("live tensor contract", fontsize=7)
    for k, (name, value) in enumerate(rows):
        y = 0.94 - k * 0.118
        ax.text(0.02, y, name, fontsize=6.0, ha="left", va="center", color="0.35")
        ax.text(0.98, y, value, fontsize=6.0, ha="right", va="center",
                family="DejaVu Sans Mono")


__all__ = ["architecture", "problem_schematic"]
