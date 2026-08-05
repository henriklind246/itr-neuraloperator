import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from src.physics.fv_solver_1d import windowed_sin_flux

"""
Separable left-boundary forcing q_L(y, t) = a(t) * s(y).

Spatial families: uniform, patch, gaussian, triangle. Each is normalized so
max_y s(y) = 1 analytically.

Temporal families: sin (Tukey-windowed half-wave-rectified sinusoid), exp
(single exponential decay), pulse_train (rectangular pulse train), exp_train
(exponential pulse train).

Sampling lives here, not in the solver. `build_qL` returns a callable
q_left(t) that the solver consumes via `q_left_fn`.
"""

SPATIAL_FAMILIES = {"uniform": 0, "patch": 1, "gaussian": 2, "triangle": 3}
TEMPORAL_FAMILIES = {"sin": 0, "exp": 1, "pulse_train": 2, "exp_train": 3}
TEMPORAL_FAMILY_ORDER = ("sin", "exp", "pulse_train", "exp_train")

SPATIAL_PARAM_ORDER = ("y_c", "w", "sigma_y", "ell")

# Spatial sampling bounds.
PATCH_W_RANGE = (0.10, 0.60)
GAUSS_SIGMA_RANGE = (0.03, 0.20)
GAUSS_CENTER_RANGE = (0.05, 0.95)
TRIANGLE_ELL_RANGE = (0.05, 0.35)

# Temporal sampling bounds. Train/exp ranges are expressed as fractions of
# (dt, t_final) and resolved per-call by the samplers.
SIN_AMP_RANGE = (50.0, 300.0)
SIN_FREQ_RANGE = (1.0, 20.0)
PULSE_AMP_RANGE = (50.0, 300.0)
EXP_T0_FRAC = 0.85
TAU_FRAC_LO = 5.0
TAU_EXP_FRAC_HI = 0.5
TAU_TRAIN_FRAC_HI = 0.3
DT_PULSE_FRAC_LO = 5.0
DT_PULSE_FRAC_HI = 0.2
NP_CHOICES = (1, 2, 3, 4)
NP_MAX = 4
PULSE_SLOTS = 4
FORCING_BINS = 16
SIN_INTEGRAL_SAMPLES = 2049

# Fixed flux scale shared by forcing representations. It is the predetermined
# maximum sampled forcing amplitude, not the 300 K temperature baseline.
A_REF_FLUX = 300.0
# Used by the FNO temporal representation.
FORCING_TEMPORAL_SAMPLES = 128
FORCING_SCHEMA_VERSION = 1
RAMP_SCHEMA_VERSION = 1

# Startup ramp on a(t) so q_L(0) = 0 (consistent with a zero left gradient at
# t=0). `ramp_seconds` is a physical dataset parameter persisted with the data;
# `default_ramp_seconds` is only the fallback initializer. 2*dt stays well under
# the shortest sampled pulse width (5*dt), so whole pulses are not swallowed.
RAMP_DT_MULT = 2.0


def default_ramp_seconds(dt: float) -> float:
    return RAMP_DT_MULT * float(dt)

# -------- SPATIAL PROFILE FUNCTIONS -----

def spatial_uniform(y: np.ndarray) -> np.ndarray:
    return np.ones_like(y, dtype=float)

def spatial_patch(y: np.ndarray, y_c: float, w: float) -> np.ndarray:
    half = 0.5 * w
    return ((y >= y_c - half) & (y <= y_c + half)).astype(float)

def spatial_gaussian(y: np.ndarray, y_c: float, sigma_y: float) -> np.ndarray:
    return np.exp(-((y - y_c) ** 2) / (2.0 * sigma_y ** 2))

def spatial_triangle(y: np.ndarray, y_c: float, ell: float) -> np.ndarray:
    return np.maximum(1.0 - np.abs(y - y_c) / ell, 0.0)

SPATIAL_BUILDERS = {
    "uniform":  lambda y: spatial_uniform(y),
    "patch":    lambda y, y_c, w: spatial_patch(y, y_c, w),
    "gaussian": lambda y, y_c, sigma_y: spatial_gaussian(y, y_c, sigma_y),
    "triangle": lambda y, y_c, ell: spatial_triangle(y, y_c, ell),
}

# -------- TEMPORAL FORCING FUNCTIONS ----------

def temporal_sin(A: float, f: float, t_on: float, t_off: float,
                 phase: float = 0.0, tukey_alpha: float = 0.5,
                 rectified: bool = True):
    """Tukey-windowed half-wave-rectified sinusoidal temporal forcing."""
    return windowed_sin_flux(f, A, t_on, t_off, phase, tukey_alpha, rectified=rectified)


def _windowed_sin_values(t, A: float, f: float, t_on: float, t_off: float,
                         phase: float = 0.0, tukey_alpha: float = 0.5,
                         rectified: bool = True) -> np.ndarray:
    """Vectorized evaluation of `temporal_sin` / `windowed_sin_flux` over `t`.

    Bit-for-bit equivalent to mapping the scalar `windowed_sin_flux` closure
    over `t`, but avoids a Python-level call per sample. The quadrature in the
    `sin` branch evaluates this at 2049 points per bin x 16 bins per item, so
    the scalar map dominated DataLoader item construction for `sin` forcing.
    Pinned against the scalar reference in tests/test_boundary_forcing.py.
    """
    if not rectified:
        raise ValueError("windowed_sin_flux now supports only rectified=True.")
    t = np.asarray(t, dtype=float)
    W = float(t_off) - float(t_on)
    tau = (t - float(t_on)) / W
    if tukey_alpha <= 0.0:
        w = np.ones_like(t)
    else:
        w = np.ones_like(t)
        left = tau < tukey_alpha / 2
        right = tau > 1.0 - tukey_alpha / 2
        w[left] = 0.5 * (1.0 - np.cos(2 * np.pi * tau[left] / tukey_alpha))
        w[right] = 0.5 * (
            1.0 + np.cos(2 * np.pi * (tau[right] - 1.0 + tukey_alpha / 2) / tukey_alpha)
        )
    vals = w * A * np.maximum(np.sin(2 * np.pi * f * t + phase), 0.0)
    vals[(t < float(t_on)) | (t > float(t_off))] = 0.0
    return vals

def temporal_exp(A: float, t0: float, tau: float):
    def q(t):
        if t < t0:
            return 0.0
        return A * np.exp(-(t - t0) / tau)
    return q

def temporal_pulse_train(A_list, t_list, dt_list, **_unused):
    A_arr = np.asarray(A_list, dtype=float)
    t_arr = np.asarray(t_list, dtype=float)
    dt_arr = np.asarray(dt_list, dtype=float)
    end_arr = t_arr + dt_arr
    def q(t):
        active = (t >= t_arr) & (t < end_arr)
        if not active.any():
            return 0.0
        return float(A_arr[active].sum())
    return q

def temporal_exp_train(A_list, t_list, tau_list, **_unused):
    A_arr = np.asarray(A_list, dtype=float)
    t_arr = np.asarray(t_list, dtype=float)
    tau_arr = np.asarray(tau_list, dtype=float)
    def q(t):
        mask = t >= t_arr
        if not mask.any():
            return 0.0
        contrib = A_arr[mask] * np.exp(-(t - t_arr[mask]) / tau_arr[mask])
        return float(contrib.sum())
    return q

TEMPORAL_BUILDERS = {
    "sin":         temporal_sin,
    "exp":         temporal_exp,
    "pulse_train": temporal_pulse_train,
    "exp_train":   temporal_exp_train,
}


def integrate_temporal(
    temporal_family: str,
    temporal_params: dict,
    t_lo: float,
    t_hi: float,
) -> float:
    """Integrate the nonnegative injected temporal forcing over [t_lo, t_hi]."""
    if t_hi <= t_lo:
        return 0.0

    if temporal_family == "sin":
        t = np.linspace(t_lo, t_hi, SIN_INTEGRAL_SAMPLES)
        values = _windowed_sin_values(t, **temporal_params)
        return float(_trapz(np.maximum(values, 0.0), t))

    if temporal_family == "exp":
        A = float(temporal_params["A"])
        t0 = float(temporal_params["t0"])
        tau = float(temporal_params["tau"])
        if t_hi <= t0:
            return 0.0
        a = max(float(t_lo), t0)
        b = float(t_hi)
        return float(A * tau * (np.exp(-(a - t0) / tau) - np.exp(-(b - t0) / tau)))

    if temporal_family == "pulse_train":
        total = 0.0
        for A, t_n, dt_n in zip(
            temporal_params["A_list"],
            temporal_params["t_list"],
            temporal_params["dt_list"],
        ):
            lo = max(float(t_lo), float(t_n))
            hi = min(float(t_hi), float(t_n) + float(dt_n))
            total += float(A) * max(0.0, hi - lo)
        return float(total)

    if temporal_family == "exp_train":
        total = 0.0
        for A, t_n, tau_n in zip(
            temporal_params["A_list"],
            temporal_params["t_list"],
            temporal_params["tau_list"],
        ):
            t0 = float(t_n)
            if t_hi <= t0:
                continue
            a = max(float(t_lo), t0)
            b = float(t_hi)
            tau = float(tau_n)
            total += float(A) * tau * (np.exp(-(a - t0) / tau) - np.exp(-(b - t0) / tau))
        return float(total)

    raise ValueError(f"Unknown temporal family: {temporal_family}")


def integrate_temporal_bins(
    temporal_family: str,
    temporal_params: dict,
    t_s: float,
    t_j: float,
    K: int = FORCING_BINS,
) -> np.ndarray:
    edges = np.linspace(float(t_s), float(t_j), int(K) + 1)
    return np.array(
        [
            integrate_temporal(temporal_family, temporal_params, edges[k], edges[k + 1])
            for k in range(int(K))
        ],
        dtype=float,
    )


if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz


def integrate_temporal_signed(
    temporal_family: str,
    temporal_params: dict,
    t_lo: float,
    t_hi: float,
) -> float:
    """Signed integral of a(t) over [t_lo, t_hi] — no clipping of negative parts."""
    if t_hi <= t_lo:
        return 0.0

    if temporal_family == "sin":
        t = np.linspace(t_lo, t_hi, SIN_INTEGRAL_SAMPLES)
        values = _windowed_sin_values(t, **temporal_params)
        return float(_trapz(values, t))

    if temporal_family == "exp":
        A = float(temporal_params["A"])
        t0 = float(temporal_params["t0"])
        tau = float(temporal_params["tau"])
        if t_hi <= t0:
            return 0.0
        a = max(float(t_lo), t0)
        b = float(t_hi)
        return float(A * tau * (np.exp(-(a - t0) / tau) - np.exp(-(b - t0) / tau)))

    if temporal_family == "pulse_train":
        total = 0.0
        for A, t_n, dt_n in zip(
            temporal_params["A_list"],
            temporal_params["t_list"],
            temporal_params["dt_list"],
        ):
            lo = max(float(t_lo), float(t_n))
            hi = min(float(t_hi), float(t_n) + float(dt_n))
            total += float(A) * max(0.0, hi - lo)
        return float(total)

    if temporal_family == "exp_train":
        total = 0.0
        for A, t_n, tau_n in zip(
            temporal_params["A_list"],
            temporal_params["t_list"],
            temporal_params["tau_list"],
        ):
            t0 = float(t_n)
            if t_hi <= t0:
                continue
            a = max(float(t_lo), t0)
            b = float(t_hi)
            tau = float(tau_n)
            total += float(A) * tau * (np.exp(-(a - t0) / tau) - np.exp(-(b - t0) / tau))
        return float(total)

    raise ValueError(f"Unknown temporal family: {temporal_family}")


def integrate_temporal_bins_signed(
    temporal_family: str,
    temporal_params: dict,
    t_s: float,
    t_j: float,
    K: int = FORCING_BINS,
) -> np.ndarray:
    edges = np.linspace(float(t_s), float(t_j), int(K) + 1)
    return np.array(
        [
            integrate_temporal_signed(temporal_family, temporal_params, edges[k], edges[k + 1])
            for k in range(int(K))
        ],
        dtype=float,
    )

# --------- STARTUP RAMP (q_L(0) = 0) ----------

def ramp_envelope(t, t_ramp: float):
    """Smoothstep startup envelope: 0 at t=0, 1 for t >= t_ramp, monotone.

    `t` may be scalar or array. For t_ramp <= 0 the envelope is identically 1.
    """
    t_ramp = float(t_ramp)
    if t_ramp <= 0.0:
        return np.ones_like(np.asarray(t, dtype=float)) if np.ndim(t) else 1.0
    s = np.clip(np.asarray(t, dtype=float) / t_ramp, 0.0, 1.0)
    env = 3.0 * s ** 2 - 2.0 * s ** 3
    return env if np.ndim(t) else float(env)


def ramped_temporal(temporal_family: str, temporal_params: dict, t_ramp: float):
    """Return a(t) callable with the startup ramp applied: ramp(t) * a(t)."""
    a_fn = TEMPORAL_BUILDERS[temporal_family](**temporal_params)

    def a_ramped(t):
        return ramp_envelope(t, t_ramp) * a_fn(t)

    return a_ramped


@dataclass(frozen=True)
class BoundaryForcingEvaluator:
    """Canonical separable left-boundary forcing evaluator."""

    temporal_family: str
    temporal_params: Mapping[str, Any] = field(repr=False)
    spatial_family: str
    spatial_params: Mapping[str, Any] = field(repr=False)
    t_ramp: float = 0.0
    schema_version: int = FORCING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "temporal_params", copy.deepcopy(dict(self.temporal_params)))
        object.__setattr__(self, "spatial_params", copy.deepcopy(dict(self.spatial_params)))
        object.__setattr__(self, "t_ramp", float(self.t_ramp))
        if int(self.schema_version) != FORCING_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported forcing schema version {self.schema_version}; "
                f"expected {FORCING_SCHEMA_VERSION}."
            )

    def _spatial(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(
            SPATIAL_BUILDERS[self.spatial_family](
                np.asarray(y, dtype=float), **self.spatial_params
            ),
            dtype=float,
        )

    def _temporal(self, t: np.ndarray) -> np.ndarray:
        a_fn = ramped_temporal(
            self.temporal_family, dict(self.temporal_params), self.t_ramp
        )
        t_arr = np.asarray(t, dtype=float)
        values = np.array(
            [float(a_fn(float(value))) for value in t_arr.reshape(-1)], dtype=float
        )
        return values.reshape(t_arr.shape)

    def evaluate_grid(
        self, y_axis: np.ndarray, t_axis: np.ndarray
    ) -> np.ndarray:
        """Return ``q_L`` on the outer-product grid ``(y, t)``."""
        y = np.asarray(y_axis, dtype=float).reshape(-1)
        t = np.asarray(t_axis, dtype=float).reshape(-1)
        return np.outer(self._spatial(y), self._temporal(t))

    def evaluate_points(
        self, y_points: np.ndarray, t_points: np.ndarray
    ) -> np.ndarray:
        """Return ``q_L`` at broadcast-compatible paired ``(y, t)`` points."""
        y, t = np.broadcast_arrays(
            np.asarray(y_points, dtype=float), np.asarray(t_points, dtype=float)
        )
        return self._spatial(y) * self._temporal(t)

    def integral(
        self, y_axis: np.ndarray, t_lo: float, t_hi: float
    ) -> np.ndarray:
        """Return the exact signed time integral of ``q_L`` over an interval."""
        a_int = integrate_temporal_ramped_signed(
            self.temporal_family,
            dict(self.temporal_params),
            float(t_lo),
            float(t_hi),
            self.t_ramp,
        )
        return float(a_int) * self._spatial(np.asarray(y_axis, dtype=float))


def _integrate_ramp_window(a_fn, t_lo: float, t_hi: float, t_ramp: float) -> float:
    """Numerically integrate ramp_envelope(t) * a_fn(t) over [t_lo, t_hi].

    Used only inside the ramp window [0, t_ramp]; this interval is ~2*dt wide,
    so a fine quadrature is cheap. Uses scipy.integrate.quad when available,
    falling back to a fine trapezoid otherwise.
    """
    if t_hi <= t_lo:
        return 0.0

    def integrand(t):
        return ramp_envelope(t, t_ramp) * a_fn(t)

    try:
        from scipy.integrate import quad
        val, _ = quad(integrand, float(t_lo), float(t_hi), limit=100)
        return float(val)
    except Exception:
        t = np.linspace(float(t_lo), float(t_hi), 1025)
        values = np.array([integrand(float(tn)) for tn in t], dtype=float)
        return float(_trapz(values, t))


def integrate_temporal_ramped_signed(
    temporal_family: str,
    temporal_params: dict,
    t_lo: float,
    t_hi: float,
    t_ramp: float,
) -> float:
    """Signed integral of ramp_envelope(t) * a(t) over [t_lo, t_hi].

    Split at t_ramp: numerical quadrature over the ramp window (where the
    envelope is nontrivial) plus the exact analytic `integrate_temporal_signed`
    beyond t_ramp (where the envelope is identically 1).
    """
    t_lo = float(t_lo)
    t_hi = float(t_hi)
    t_ramp = float(t_ramp)
    if t_hi <= t_lo:
        return 0.0
    if t_ramp <= 0.0:
        return integrate_temporal_signed(temporal_family, temporal_params, t_lo, t_hi)

    total = 0.0
    # Ramp window [t_lo, min(t_hi, t_ramp)] uses quadrature.
    hi_r = min(t_hi, t_ramp)
    if hi_r > t_lo:
        a_fn = TEMPORAL_BUILDERS[temporal_family](**temporal_params)
        total += _integrate_ramp_window(a_fn, t_lo, hi_r, t_ramp)
    # Beyond the ramp the envelope is 1: exact analytic integral.
    lo_a = max(t_lo, t_ramp)
    if t_hi > lo_a:
        total += integrate_temporal_signed(temporal_family, temporal_params, lo_a, t_hi)
    return float(total)


def integrate_temporal_bins_ramped_signed(
    temporal_family: str,
    temporal_params: dict,
    t_s: float,
    t_j: float,
    t_ramp: float,
    K: int = FORCING_BINS,
) -> np.ndarray:
    edges = np.linspace(float(t_s), float(t_j), int(K) + 1)
    return np.array(
        [
            integrate_temporal_ramped_signed(
                temporal_family, temporal_params, edges[k], edges[k + 1], t_ramp
            )
            for k in range(int(K))
        ],
        dtype=float,
    )

# --------- SAMPLER FUNCTIONS ----------

def sample_uniform_params(rng: np.random.Generator, **_unused) -> dict:
    return {}

def _span(c: float, d: float) -> float:
    span = float(d) - float(c)
    if span <= 0.0:
        raise ValueError(f"Expected d > c for y-domain, got c={c}, d={d}.")
    return span


def _scale_interval(frac_range: tuple[float, float], c: float, d: float) -> tuple[float, float]:
    span = _span(c, d)
    return float(c) + frac_range[0] * span, float(c) + frac_range[1] * span


def sample_patch_params(rng: np.random.Generator, c: float = 0.0, d: float = 1.0) -> dict:
    span = _span(c, d)
    w = float(rng.uniform(*PATCH_W_RANGE)) * span
    y_c = float(rng.uniform(float(c) + 0.5 * w, float(d) - 0.5 * w))
    return {"y_c": y_c, "w": w}

def sample_gauss_params(rng: np.random.Generator, c: float = 0.0, d: float = 1.0) -> dict:
    span = _span(c, d)
    lo, hi = GAUSS_SIGMA_RANGE
    u = rng.uniform(0.0, 1.0)
    sigma_y = float(lo * (hi / lo) ** u) * span
    y_lo, y_hi = _scale_interval(GAUSS_CENTER_RANGE, c, d)
    y_c = float(rng.uniform(y_lo, y_hi))
    return {"y_c": y_c, "sigma_y": sigma_y}

def sample_triangle_params(rng: np.random.Generator, c: float = 0.0, d: float = 1.0) -> dict:
    span = _span(c, d)
    ell = float(rng.uniform(*TRIANGLE_ELL_RANGE)) * span
    y_c = float(rng.uniform(float(c) + ell, float(d) - ell))
    return {"y_c": y_c, "ell": ell}

SPATIAL_SAMPLERS = {
    "uniform":  sample_uniform_params,
    "patch":    sample_patch_params,
    "gaussian": sample_gauss_params,
    "triangle": sample_triangle_params,
}

def sample_sin_params(rng: np.random.Generator, dt: float, t_final: float,
                      *, t_on: float, t_off: float,
                      phase: float = 0.0, tukey_alpha: float = 0.5) -> dict:
    A = float(rng.uniform(*SIN_AMP_RANGE))
    lo, hi = SIN_FREQ_RANGE
    u = rng.uniform(0.0, 1.0)
    f = float(lo * (hi / lo) ** u)
    return {"A": A, "f": f, "t_on": t_on, "t_off": t_off,
            "phase": phase, "tukey_alpha": tukey_alpha, "rectified": True}

def sample_exp_params(rng: np.random.Generator, dt: float, t_final: float,
                      **_unused) -> dict:
    A = float(rng.uniform(*PULSE_AMP_RANGE))
    t0 = float(rng.uniform(0.0, EXP_T0_FRAC * t_final))
    tau_lo = TAU_FRAC_LO * dt
    tau_hi = TAU_EXP_FRAC_HI * t_final
    u = rng.uniform(0.0, 1.0)
    tau = float(tau_lo * (tau_hi / tau_lo) ** u)
    return {"A": A, "t0": t0, "tau": tau}

def sample_pulse_train_params(rng: np.random.Generator, dt: float, t_final: float,
                              **_unused) -> dict:
    Np = int(rng.choice(NP_CHOICES))
    A_list, t_list, dt_list = [], [], []
    dt_lo = DT_PULSE_FRAC_LO * dt
    dt_hi = DT_PULSE_FRAC_HI * t_final
    for _ in range(Np):
        A_list.append(float(rng.uniform(*PULSE_AMP_RANGE)))
        u = rng.uniform(0.0, 1.0)
        dtn = float(dt_lo * (dt_hi / dt_lo) ** u)
        dt_list.append(dtn)
        t_list.append(float(rng.uniform(0.0, t_final - dtn)))
    order = np.argsort(t_list)
    return {
        "Np": Np,
        "A_list":  [A_list[i]  for i in order],
        "t_list":  [t_list[i]  for i in order],
        "dt_list": [dt_list[i] for i in order],
    }

def sample_exp_train_params(rng: np.random.Generator, dt: float, t_final: float,
                            **_unused) -> dict:
    Np = int(rng.choice(NP_CHOICES))
    A_list, t_list, tau_list = [], [], []
    tau_lo = TAU_FRAC_LO * dt
    tau_hi = TAU_TRAIN_FRAC_HI * t_final
    for _ in range(Np):
        A_list.append(float(rng.uniform(*PULSE_AMP_RANGE)))
        t_list.append(float(rng.uniform(0.0, EXP_T0_FRAC * t_final)))
        u = rng.uniform(0.0, 1.0)
        tau_list.append(float(tau_lo * (tau_hi / tau_lo) ** u))
    order = np.argsort(t_list)
    return {
        "Np": Np,
        "A_list":   [A_list[i]   for i in order],
        "t_list":   [t_list[i]   for i in order],
        "tau_list": [tau_list[i] for i in order],
    }

TEMPORAL_SAMPLERS = {
    "sin":         sample_sin_params,
    "exp":         sample_exp_params,
    "pulse_train": sample_pulse_train_params,
    "exp_train":   sample_exp_train_params,
}

def sample_spatial_family(rng: np.random.Generator,
                          probs: np.ndarray | None = None) -> str:
    families = list(SPATIAL_FAMILIES.keys())
    if probs is None:
        return str(rng.choice(families))
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(families, p=probs))

def sample_temporal_family(rng: np.random.Generator,
                           probs: np.ndarray | None = None) -> str:
    families = list(TEMPORAL_FAMILIES.keys())
    if probs is None:
        return str(rng.choice(families))
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(families, p=probs))

# --------- FLUX BUILDER -----------

def build_qL(temporal_family: str, temporal_params: dict,
             spatial_family: str, spatial_params: dict,
             y_grid: np.ndarray, t_ramp: float = 0.0):
    """
    Build a callable q_left(t) -> (Ny,) and return the static spatial profile.

    `temporal_params` keys must match `TEMPORAL_BUILDERS[temporal_family]`
    signature exactly. The samplers in `TEMPORAL_SAMPLERS` produce the
    canonical schema. `t_ramp > 0` applies the startup ramp so q_L(0) = 0.
    """
    forcing = reconstruct_qL(
        temporal_family, temporal_params, spatial_family, spatial_params, t_ramp
    )
    y_grid = np.asarray(y_grid, dtype=float)
    s_vec = forcing._spatial(y_grid)

    def q_left(t):
        return forcing.evaluate_points(y_grid, t)

    return q_left, s_vec


def build_qL_integral(temporal_family: str, temporal_params: dict,
                      spatial_family: str, spatial_params: dict,
                      y_grid: np.ndarray, t_ramp: float = 0.0):
    """
    Build a callable q_left_integral(t_lo, t_hi) -> (Ny,) returning the exact signed
    time-integral of q_L(y, t) over [t_lo, t_hi]. Uses the separable structure
    q_L(y, t) = a(t) * s(y), so the integral is
    integrate_temporal_ramped_signed(...) * s_vec. `t_ramp > 0` applies the
    startup ramp so q_L(0) = 0.
    """
    forcing = reconstruct_qL(
        temporal_family, temporal_params, spatial_family, spatial_params, t_ramp
    )
    y_grid = np.asarray(y_grid, dtype=float)
    s_vec = forcing._spatial(y_grid)

    def q_left_integral(t_lo, t_hi):
        return forcing.integral(y_grid, t_lo, t_hi)

    return q_left_integral, s_vec


def build_interface_forcing(
    temporal_family: str, temporal_params: dict,
    spatial_family: str, spatial_params: dict,
    y_grid: np.ndarray,
    t_n: float, t_np1: float, t_ramp: float,
):
    """Return physical FV endpoint fluxes and exact signed interval integral."""
    y_grid = np.asarray(y_grid, dtype=float)
    forcing = reconstruct_qL(
        temporal_family, temporal_params, spatial_family, spatial_params, t_ramp
    )
    qL_n = forcing.evaluate_points(y_grid, float(t_n)).astype(np.float64)
    qL_np1 = forcing.evaluate_points(y_grid, float(t_np1)).astype(np.float64)
    qL_int = forcing.integral(y_grid, float(t_n), float(t_np1)).astype(np.float64)
    return qL_n, qL_np1, qL_int


def reconstruct_qL(temporal_family: str, temporal_params: dict,
                   spatial_family: str, spatial_params: dict,
                   t_ramp: float = 0.0) -> BoundaryForcingEvaluator:
    """Build the canonical evaluator for one separable inward-flux record."""
    return BoundaryForcingEvaluator(
        temporal_family=temporal_family,
        temporal_params=temporal_params,
        spatial_family=spatial_family,
        spatial_params=spatial_params,
        t_ramp=t_ramp,
    )

# --------- PARAMETER ENCODING ----------

def encode_spatial_params(spatial_family: str, spatial_params: dict) -> np.ndarray:
    """
    Pack spatial-family parameters into a fixed 4-vector
    [y_c, w, sigma_y, ell] with unused slots set to 0.
    """
    y_c = w = sigma_y = ell = 0.0
    if spatial_family == "uniform":
        pass
    elif spatial_family == "patch":
        y_c = spatial_params["y_c"]
        w = spatial_params["w"]
    elif spatial_family == "gaussian":
        y_c = spatial_params["y_c"]
        sigma_y = spatial_params["sigma_y"]
    elif spatial_family == "triangle":
        y_c = spatial_params["y_c"]
        ell = spatial_params["ell"]
    else:
        raise ValueError(f"Unknown spatial family: {spatial_family}")
    return np.array([y_c, w, sigma_y, ell], dtype=float)

def encode_temporal_params(temporal_family: str, temporal_params: dict,
                           dt: float, t_final: float) -> np.ndarray:
    """
    Pack temporal-family parameters into a fixed (1 + PULSE_SLOTS*3)-vector.

    Layout:
      [0]              Np_norm = (Np-1)/(NP_MAX-1) for trains; 0 for sin/exp
      [1+3n : 1+3n+3]  pulse slot n, semantics depend on family:
        sin:         slot 0 = (A_norm, f_norm_log, 0)
        exp:         slot 0 = (t0_norm, tau_norm_log, A_norm)
        pulse_train: slot n = (t_n_norm, A_n_norm, dt_n_norm_log) for n < Np
        exp_train:   slot n = (t_n_norm, A_n_norm, tau_n_norm_log) for n < Np
      Unused slots are zero.

    Log-uniform parameters (f, tau, dt_n) are min-max normalized in log-space
    so coverage matches the sampler distributions.
    """
    out = np.zeros(1 + PULSE_SLOTS * 3, dtype=float)

    def _norm_amp(A):
        return (A - PULSE_AMP_RANGE[0]) / (PULSE_AMP_RANGE[1] - PULSE_AMP_RANGE[0])

    def _norm_t(t):
        return t / t_final

    def _norm_log(x, lo, hi):
        return (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))

    if temporal_family == "sin":
        A = temporal_params["A"]
        f = temporal_params["f"]
        out[1] = (A - SIN_AMP_RANGE[0]) / (SIN_AMP_RANGE[1] - SIN_AMP_RANGE[0])
        out[2] = _norm_log(f, SIN_FREQ_RANGE[0], SIN_FREQ_RANGE[1])
        return out

    if temporal_family == "exp":
        tau_lo = TAU_FRAC_LO * dt
        tau_hi = TAU_EXP_FRAC_HI * t_final
        out[1] = _norm_t(temporal_params["t0"])
        out[2] = _norm_log(temporal_params["tau"], tau_lo, tau_hi)
        out[3] = _norm_amp(temporal_params["A"])
        return out

    if temporal_family == "pulse_train":
        Np = temporal_params["Np"]
        out[0] = (Np - 1) / (NP_MAX - 1)
        dt_lo = DT_PULSE_FRAC_LO * dt
        dt_hi = DT_PULSE_FRAC_HI * t_final
        for n in range(Np):
            base = 1 + 3 * n
            out[base + 0] = _norm_t(temporal_params["t_list"][n])
            out[base + 1] = _norm_amp(temporal_params["A_list"][n])
            out[base + 2] = _norm_log(temporal_params["dt_list"][n], dt_lo, dt_hi)
        return out

    if temporal_family == "exp_train":
        Np = temporal_params["Np"]
        out[0] = (Np - 1) / (NP_MAX - 1)
        tau_lo = TAU_FRAC_LO * dt
        tau_hi = TAU_TRAIN_FRAC_HI * t_final
        for n in range(Np):
            base = 1 + 3 * n
            out[base + 0] = _norm_t(temporal_params["t_list"][n])
            out[base + 1] = _norm_amp(temporal_params["A_list"][n])
            out[base + 2] = _norm_log(temporal_params["tau_list"][n], tau_lo, tau_hi)
        return out

    raise ValueError(f"Unknown temporal family: {temporal_family}")
