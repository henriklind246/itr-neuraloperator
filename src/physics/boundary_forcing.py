import numpy as np
from src.physics.fv_solver_1d import windowed_sin_flux

"""
Separable left-boundary forcing q_L(y, t) = a(t) * s(y).

Spatial families: uniform, patch, gaussian, triangle. Each is normalized so
max_y s(y) = 1 analytically. Temporal family: windowed sinusoid (only one for
now; the registry pattern leaves room for more).

Sampling lives here, not in the solver. `build_qL` returns a callable
q_left(t) that the solver consumes via `q_left_fn`.
"""

SPATIAL_FAMILIES = {"uniform": 0, "patch": 1, "gaussian": 2, "triangle": 3}
TEMPORAL_FAMILIES = {"sin": 0}

SPATIAL_PARAM_ORDER = ("y_c", "w", "sigma_y", "ell")

# Sampling bounds — kept here so dataset.py and tests share one source of truth.
PATCH_W_RANGE = (0.10, 0.60)
GAUSS_SIGMA_RANGE = (0.03, 0.20)
GAUSS_CENTER_RANGE = (0.05, 0.95)
TRIANGLE_ELL_RANGE = (0.05, 0.35)

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

TEMPORAL_BUILDERS = {
    "sin": temporal_sin,
}

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

def sample_spatial_family(rng: np.random.Generator,
                          probs: np.ndarray | None = None) -> str:
    families = list(SPATIAL_FAMILIES.keys())
    if probs is None:
        return str(rng.choice(families))
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(families, p=probs))

def sample_temporal_family(rng: np.random.Generator) -> str:
    families = list(TEMPORAL_FAMILIES.keys())
    return str(rng.choice(families))

# --------- FLUX BUILDER -----------

def build_qL(temporal_family: str, temporal_params: dict,
             spatial_family: str, spatial_params: dict,
             y_grid: np.ndarray):
    """
    Build a callable q_left(t) -> (Ny,) and return the static spatial profile.

    The solver only needs q_left; s_vec is returned for diagnostics/tests.
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
