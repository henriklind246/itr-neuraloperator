"""Conceptual two-panel figure: sinusoidal R_c(y) vs classical scalar R_c.

Illustrates the difference between the spatially varying sinusoidal
interfacial thermal resistance of the *_itr_sin benchmarks and the classical
spatially uniform scalar-ITR formulation. This is a representative schematic,
not a result plot.

The representative profile uses the center of the actual benchmark sampling
ranges (src/physics/internal_source.py RC_SIN_RANGES, R_PEAK_MAX and
problems/source_itr_sin.py sample_sim_params):

    R_c(y) = R_base + A sin(pi y),
    R_base = mid([0.05, 1.0]) = 0.525,
    A      = E[u_A] * (R_PEAK_MAX - R_base) = 0.5 * (3.0 - 0.525) = 1.2375.

The scalar panel uses the center of the uniform benchmarks' own sampling range
(problems/source.py / problems/forcing.py RC_RANGE = (0.05, 1.0)), i.e.
R_c = 0.525.

Run:  python -m visual.pub.fig_itr_schematic
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems.source import RC_RANGE
from src.physics.internal_source import RC_SIN_RANGES, R_PEAK_MAX, make_rc_sin_profile
from visual.pub import style

OUT_DIR = Path("visual/pub_out")
KEY = "itr_sin_vs_scalar_schematic"

COLOR_VARYING = "#0072B2"   # Okabe-Ito blue
COLOR_SCALAR = "#D55E00"    # Okabe-Ito vermillion


def representative_params() -> tuple[float, float]:
    """Return (R_base, A) at the center of the benchmark ranges."""
    base_lo, base_hi = RC_SIN_RANGES["R_base"]
    R_base = 0.5 * (base_lo + base_hi)
    # A is sampled as u_A * (R_PEAK_MAX - R_base) with u_A ~ U[0, 1].
    A = 0.5 * (R_PEAK_MAX - R_base)
    return R_base, A


def build() -> "plt.Figure":
    R_base, A = representative_params()
    y = np.linspace(0.0, 1.0, 401)
    rc = make_rc_sin_profile(y, R_base=R_base, A=A)

    fig, (ax_var, ax_uni) = plt.subplots(
        1, 2, sharey=True, figsize=style.figsize("one_half_col", extra_in=-0.2),
        constrained_layout=True,
    )

    xlim = (0.0, 2.0)
    xlabel = "$R_c$ [m$^2$K/W]"

    # ---- (a) spatially varying sinusoidal profile ----
    ax_var.plot(rc, y, color=COLOR_VARYING, linewidth=1.6, solid_capstyle="round")
    ax_var.text(
        0.975, 0.015,
        rf"$R_c(y) = R_\mathrm{{base}} + A\,\sin(\pi y)$"
        "\n"
        rf"$R_\mathrm{{base}} = {R_base:.3g},\ A = {A:.3g}$",
        transform=ax_var.transAxes, ha="right", va="bottom",
        fontsize=5.5, color="0.35", linespacing=1.35,
    )
    ax_var.set_title("Spatially varying ITR")
    ax_var.set_ylabel("$y$")

    # ---- (b) classical spatially uniform scalar ----
    rc_scalar = 0.5 * (RC_RANGE[0] + RC_RANGE[1])
    ax_uni.plot([rc_scalar, rc_scalar], [0.0, 1.0], color=COLOR_SCALAR,
                linewidth=1.6, solid_capstyle="round")
    ax_uni.text(rc_scalar + 0.06, 0.5, rf"$R_c = {rc_scalar:.3g}$",
                ha="left", va="center", rotation=90, fontsize=6, color="0.45")
    ax_uni.set_title("Spatially uniform ITR")

    ax_var.set_xlim(*xlim)
    ax_var.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_uni.set_xlim(0.0, 1.0)
    ax_uni.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    for ax in (ax_var, ax_uni):
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(xlabel)
        ax.tick_params(length=2.5, width=0.7)
    ax_var.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    style.panel_letters([ax_var, ax_uni], loc=(-0.10, 1.06))
    return fig


def main() -> None:
    with style.pub_style():
        fig = build()
        style.save(fig, OUT_DIR, KEY)


if __name__ == "__main__":
    main()
