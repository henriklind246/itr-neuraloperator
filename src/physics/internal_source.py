import numpy as np
from typing import Callable

"""Internal volumetric heat-generation patch source.

Used by the variable-location chip-heating experiment. A single rectangular
patch centered at (x_h, y_h) of size (w, h) is heated by a smooth sin^2 pulse
between t=0 and t=t_off:

    Q(x, y, t) = a(t) * S_h(x, y)
    a(t)       = A * sin^2(pi * t / t_off)   for 0 <= t <= t_off, else 0
    S_h(x, y)  = indicator of the rectangular patch.

PATCH_A_RANGE is in W/mm^3 for the Ti-6Al-4V / brass source benchmarks.
The shared 10:1 amplitude range is calibrated with
`scripts/calibrate_patch_amplitude.py` at 100x100, dt=0.005 s, t_final=0.3 s.
The 174-case screen gives peak rises of approximately 0.13-145 K over both
amplitude endpoints. A universal 5 K floor is incompatible with a 150 K
ceiling: patches next to the cold right wall heat much less than left patches.
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
PATCH_A_RANGE: tuple[float, float] = (8.0, 80.0)


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
# Spatially-varying interface resistance (sinusoidal profile)
# ---------------------------------------------------------------------------
RC_MIN: float = 0.05
R_PEAK_MAX: float = 3.0

RC_SIN_RANGES: dict[str, tuple[float, float]] = {
    "R_base": (0.05, 1.0),
    # Global max used for conditioning normalization (R_base at its floor); the
    # per-sample admissible upper bound is resolved as (R_PEAK_MAX - R_base).
    "A": (0.0, R_PEAK_MAX - RC_MIN),
}


def make_rc_sin_profile(
    y_grid: np.ndarray,
    R_base: float,
    A: float,
) -> np.ndarray:
    """Return the (Ny,) sinusoidal interface-resistance profile R_c(y).

    Evaluated at the interface-row coordinates `y_grid`:
        R_c(y) = R_base + A * sin(pi * y).
    With A = 0 this returns a flat R_base profile. Returns float64 so it feeds
    the solver's series-resistance formula at full precision. The exact peak
    R_base + A is attained at y = 0.5.
    """
    y = np.asarray(y_grid, dtype=np.float64)
    return R_base + A * np.sin(np.pi * y)


def interface_control_volume_weights(
    y_grid: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    """Return FV dual-control-volume widths associated with interface rows."""
    y = np.asarray(y_grid, dtype=np.float64)
    c, d = map(float, bounds)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("y_grid must be a one-dimensional array with at least two entries.")
    if not np.all(np.diff(y) > 0.0):
        raise ValueError("y_grid must be strictly increasing.")
    if y[0] < c or y[-1] > d:
        raise ValueError("y_grid entries must lie inside the supplied domain bounds.")

    edges = np.empty(y.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (y[:-1] + y[1:])
    edges[0] = c
    edges[-1] = d
    weights = np.diff(edges)
    if np.any(weights <= 0.0):
        raise ValueError("y_grid and bounds do not define positive control-volume widths.")
    return weights


def equivalent_scalar_resistance(
    y_grid: np.ndarray,
    Rc_profile: np.ndarray,
    *,
    bounds: tuple[float, float],
) -> float:
    """Conductance-matched scalar resistance on a general y-domain."""
    Rc = np.asarray(Rc_profile, dtype=np.float64)
    weights = interface_control_volume_weights(y_grid, bounds)
    if Rc.shape != weights.shape:
        raise ValueError(
            f"Rc_profile must have shape {weights.shape}, got {Rc.shape}."
        )
    if np.any(Rc <= 0.0):
        raise ValueError("Rc_profile must be strictly positive.")
    return float(np.sum(weights) / np.sum(weights / Rc))


def rc_log_norm(Rc_y: np.ndarray) -> np.ndarray:
    """Log-normalize a physical R_c(y) profile to roughly [-1, 1].

    Linear normalization over the 60x span [RC_MIN, R_PEAK_MAX] = [0.05, 3.0]
    wastes most of the range because realistic R_base clusters in [0.05, 1.0];
    log-spacing spends resolution where the samples are. R_c(y) >= R_base >=
    RC_MIN, so the log argument is always positive.
    """
    log_min = np.log(RC_MIN)
    log_max = np.log(R_PEAK_MAX)
    Rc = np.asarray(Rc_y, dtype=np.float64)
    return (2.0 * (np.log(Rc) - log_min) / (log_max - log_min) - 1.0).astype(np.float32)
