import numpy as np
from typing import Callable

"""Internal volumetric heat-generation patch source.

Used by the variable-location chip-heating experiment. A single rectangular
patch centered at (x_h, y_h) of size (w, h) is heated by a smooth sin^2 pulse
between t=0 and t=t_off:

    Q(x, y, t) = a(t) * S_h(x, y)
    a(t)       = A * sin^2(pi * t / t_off)   for 0 <= t <= t_off, else 0
    S_h(x, y)  = indicator of the rectangular patch.

PATCH_A_RANGE is the (A_MIN, A_MAX) tuple consumed by downstream LHS sampling
and normalization, so there is a single source of truth. The range is chosen
so that peak temperature rises land inside the 5-150 K window used by
`scripts/calibrate_patch_amplitude.py`: a 4-point sweep at (A=300, A=3000)
across both interface regimes gave ~1 K and ~10 K peak rises, so linear
extrapolation places A_min at ~1500 (5 K floor) and A_max at ~15000
(~50 K rise, well below the 150 K ceiling).
"""

PATCH_W = 0.1
PATCH_H = 0.1

# Domain bounds of the conduction problem (the unit square). The admissible
# patch-center ranges are derived from these bounds and the patch size rather
# than hard-coded, so they stay correct if the domain ever changes:
#   x_h in [a + w/2, b - w/2],  y_h in [c + h/2, d - h/2].
DOMAIN_X: tuple[float, float] = (0.0, 1.0)
DOMAIN_Y: tuple[float, float] = (0.0, 1.0)


def patch_center_range(lo: float, hi: float, length: float) -> tuple[float, float]:
    """Center range that keeps a `length`-sized patch fully inside [lo, hi]."""
    half = 0.5 * length
    return (lo + half, hi - half)


PATCH_X_RANGE = patch_center_range(*DOMAIN_X, PATCH_W)
PATCH_Y_RANGE = patch_center_range(*DOMAIN_Y, PATCH_H)
PATCH_A_RANGE: tuple[float, float] = (1500.0, 15000.0)


def make_patch_indicator(
    X: np.ndarray,
    Y: np.ndarray,
    x_h: float,
    y_h: float,
    w: float,
    h: float,
) -> np.ndarray:
    """Return a float32 (Nx, Ny) indicator of the rectangular patch.

    Boundary convention is `<=` on both sides, matching the patch builder in
    `boundary_forcing.spatial_patch`. Cells with centers on the edge of the
    patch are included.
    """
    half_w = 0.5 * w
    half_h = 0.5 * h
    mask = (
        (X >= x_h - half_w) & (X <= x_h + half_w)
        & (Y >= y_h - half_h) & (Y <= y_h + half_h)
    )
    return mask.astype(np.float32)


def make_sin2_pulse(A: float, t_off: float) -> Callable[[float], float]:
    """Return a(t) = A * sin^2(pi * t / t_off), clamped to zero outside [0, t_off]."""
    if t_off <= 0:
        raise ValueError(f"t_off must be positive, got {t_off}.")

    def a(t: float) -> float:
        if t < 0.0 or t > t_off:
            return 0.0
        return float(A) * np.sin(np.pi * t / t_off) ** 2

    return a


def integrate_sin2_pulse(A: float, t_off: float, t0: float, t1: float) -> float:
    """Exact integral of a(t) = A * sin^2(pi t / t_off) over [t0, t1].

    Uses the closed form
        int sin^2(c t) dt = t/2 - sin(2 c t) / (4 c).
    Integration is clipped to [0, t_off] so the value outside the pulse
    window contributes nothing.
    """
    a = max(t0, 0.0)
    b = min(t1, t_off)
    if b <= a:
        return 0.0
    c = np.pi / t_off
    primitive = lambda u: 0.5 * u - np.sin(2.0 * c * u) / (4.0 * c)
    return float(A) * (primitive(b) - primitive(a))


def build_patch_source(
    x_h: float,
    y_h: float,
    w: float,
    h: float,
    A: float,
    t_off: float,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """Construct the solver-facing source(Xq, Yq, t) = a(t) * S_h(Xq, Yq).

    The returned callable matches FVSolver2D's expected source signature. The
    solver may pass sub-arrays (active y-rows), so the patch indicator is
    evaluated on the supplied query coordinates Xq, Yq on every call rather than
    against a buffered full-grid mask.
    """
    pulse = make_sin2_pulse(A, t_off)

    def source(Xq: np.ndarray, Yq: np.ndarray, t: float) -> np.ndarray:
        amp = pulse(t)
        if amp == 0.0:
            return np.zeros_like(Xq, dtype=np.float64)
        mask = make_patch_indicator(Xq, Yq, x_h, y_h, w, h)
        return amp * mask.astype(np.float64)

    return source


# ---------------------------------------------------------------------------
# Spatially-varying interface resistance (Gaussian void)
# ---------------------------------------------------------------------------
# R_c(y) = R_base + R_amp * exp(-((y - y0) / sigma)^2)
#
# Models a localized delamination / air-gap void in the thermal interface
# material along the fixed x = 0.5 interface: a single smooth positive bump in
# the contact resistance centered at y0 with width sigma. The amplitude bound is
# *dependent* on R_base so the profile is physical by construction (no clipping):
#   R_c(y) in [R_base, R_base + R_amp] subset [RC_MIN, R_PEAK_MAX].
# Sampling R_amp in [0, R_PEAK_MAX - R_base] makes the (R_base, R_amp) joint
# distribution triangular (high base => low admissible amp); this is an
# in-distribution correlation, so avoid independence/extrapolation claims on it.
RC_MIN: float = 0.05
R_PEAK_MAX: float = 3.0
RC_VOID_RANGES: dict[str, tuple[float, float]] = {
    "R_base": (0.05, 1.0),
    # Upper amp bound is resolved per-sample as (R_PEAK_MAX - R_base); the value
    # here is the global maximum used for normalization (R_base at its floor).
    "R_amp": (0.0, R_PEAK_MAX - RC_MIN),
    "y0": (0.1, 0.9),
    # sigma >= 0.05 spans ~5 cells on Ny = 100, keeping the void resolved.
    "sigma": (0.05, 0.2),
}


def make_rc_void_profile(
    y_grid: np.ndarray,
    R_base: float,
    R_amp: float,
    y0: float,
    sigma: float,
) -> np.ndarray:
    """Return the (Ny,) Gaussian-void interface-resistance profile R_c(y).

    Evaluated at the cell-center coordinates `y_grid`:
        R_c(y) = R_base + R_amp * exp(-((y - y0) / sigma)^2).
    With R_amp = 0 this returns a flat R_base profile (used by the parity /
    regression tests). Returns float64 so it feeds the solver's series-resistance
    formula at full precision.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    y = np.asarray(y_grid, dtype=np.float64)
    return R_base + R_amp * np.exp(-(((y - y0) / sigma) ** 2))
