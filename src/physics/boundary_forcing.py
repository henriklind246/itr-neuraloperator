import numpy as np
from src.physics.fv_solver_1d import windowed_sin_flux

"""
Separable left-boundary forcing q_L(y, t) = a(t) * s(y).

Spatial families: uniform, patch, gaussian, triangle. Each is normalized so
max_y s(y) = 1 analytically.

Temporal families: sin (windowed sinusoid), exp (single exponential decay),
pulse_train (rectangular pulse train), exp_train (exponential pulse train).

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
                 phase: float = 0.0, tukey_alpha: float = 0.5):
    return windowed_sin_flux(f, A, t_on, t_off, phase, tukey_alpha)

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
        q = temporal_sin(**temporal_params)
        t = np.linspace(t_lo, t_hi, 65)
        values = np.array([q(float(tn)) for tn in t], dtype=float)
        return float(np.trapezoid(np.maximum(values, 0.0), t))

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

# --------- SAMPLER FUNCTIONS ----------

def sample_uniform_params(rng: np.random.Generator) -> dict:
    return {}

def sample_patch_params(rng: np.random.Generator) -> dict:
    w = float(rng.uniform(*PATCH_W_RANGE))
    y_c = float(rng.uniform(0.5 * w, 1.0 - 0.5 * w))
    return {"y_c": y_c, "w": w}

def sample_gauss_params(rng: np.random.Generator) -> dict:
    lo, hi = GAUSS_SIGMA_RANGE
    u = rng.uniform(0.0, 1.0)
    sigma_y = float(lo * (hi / lo) ** u)
    y_c = float(rng.uniform(*GAUSS_CENTER_RANGE))
    return {"y_c": y_c, "sigma_y": sigma_y}

def sample_triangle_params(rng: np.random.Generator) -> dict:
    ell = float(rng.uniform(*TRIANGLE_ELL_RANGE))
    y_c = float(rng.uniform(ell, 1.0 - ell))
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
            "phase": phase, "tukey_alpha": tukey_alpha}

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
             y_grid: np.ndarray):
    """
    Build a callable q_left(t) -> (Ny,) and return the static spatial profile.

    `temporal_params` keys must match `TEMPORAL_BUILDERS[temporal_family]`
    signature exactly. The samplers in `TEMPORAL_SAMPLERS` produce the
    canonical schema.
    """
    a_fn = TEMPORAL_BUILDERS[temporal_family](**temporal_params)
    s_vec = SPATIAL_BUILDERS[spatial_family](y_grid, **spatial_params)
    s_vec = np.asarray(s_vec, dtype=float)

    def q_left(t):
        return a_fn(t) * s_vec

    return q_left, s_vec

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
