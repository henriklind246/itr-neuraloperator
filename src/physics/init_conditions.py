import numpy as np

"""
Per-simulation initial-condition families for the 2D heat-conduction dataset.

Mirrors the spatial / temporal forcing pattern in `boundary_forcing.py`:
each family has a builder, a sampler, and entries in the IC_BUILDERS /
IC_SAMPLERS registries. `build_ic` is the top-level wrapper that adds the
T_right offset, tapers the deviation into the right Dirichlet wall, and pins
the last column exactly.

Boundary compatibility is split between the builders and the taper. The solver
imposes zero-Neumann at x = a, y = c, y = d and Dirichlet at x = b, so every
builder is even about those three Neumann walls by construction and only the
Dirichlet wall is tapered. The taper is a C2 smootherstep in x alone; a C1
smoothstep leaves a jump in the second derivative at the seam, which shows up
in rendered fields as a rectangular curvature crease inset by the taper width.
"""

IC_FAMILIES = {"uniform_2d": 0, "random_sinusoid_2d": 1, "grf_2d": 2, "hot_spot_2d": 3}

# Bump whenever a builder change can alter the final contiguous float32 field.
IC_BUILDER_SCHEMA_VERSION = 2
ONLINE_IC_SAMPLER_VERSION = "online_neumann_ic_v2"

UNIFORM_OFFSET_RANGE = (-20.0, 20.0)

SINU_N_CHOICES = (3, 4, 5)
# cos(pi n x) carries half the frequency of the sin(2 pi n x) basis this family
# used before, so the cap is doubled to preserve the achievable IC roughness.
SINU_KMAX = 8
SINU_AMP_RANGE = (1.0, 8.0)

GRF_ELL_RANGE = (0.05, 0.30)
GRF_SIGMA_RANGE = (2.0, 20.0)
# Truncation of the cosine series. Missing power at the worst case ell = 0.05
# is 5.7e-7.
RANDOM_FIELD_MODES = 32

HOT_N_CHOICES = (1, 2, 3, 4)
HOT_SIGMA_RANGE = (0.05, 0.20)
HOT_AMP_RANGE = (-30.0, 30.0)
HOT_MARGIN_FACTOR = 2.0

EDGE_TAPER_WIDTH = 0.10

# -------- BUILDER FUNCTIONS --------

def ic_uniform_2d(X: np.ndarray, Y: np.ndarray, T0_offset: float) -> np.ndarray:
    return np.full(X.shape, float(T0_offset), dtype=float)


def ic_random_sinusoid_2d(X: np.ndarray, Y: np.ndarray,
                          A_list, nx_list, ny_list) -> np.ndarray:
    """Cosine modes, even about x = 0, y = 0 and y = 1.

    `ny` must be an integer for evenness at *both* y-walls; `nx` may be any
    positive real because x = 1 is Dirichlet, so only f'(0) = 0 is required.
    Phase shifts are incompatible with exact even symmetry and were dropped;
    `A` is signed instead.
    """
    A = np.asarray(A_list, dtype=float)
    nx = np.asarray(nx_list, dtype=float)
    ny = np.asarray(ny_list, dtype=float)
    cx = np.cos(np.pi * X[..., None] * nx)
    cy = np.cos(np.pi * Y[..., None] * ny)
    return (A * cx * cy).sum(axis=-1)


def _separable_axis_coords(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1D coordinate vectors from a `meshgrid(..., indexing='ij')` mesh."""
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError(f"expected matching 2D meshes; got {X.shape} and {Y.shape}")
    x = X[:, 0]
    y = Y[0]
    if not np.array_equal(X, np.broadcast_to(x[:, None], X.shape)):
        raise ValueError("X is not constant along axis 1; expected indexing='ij'")
    if not np.array_equal(Y, np.broadcast_to(y[None, :], Y.shape)):
        raise ValueError("Y is not constant along axis 0; expected indexing='ij'")
    return x, y


def ic_random_cosine_field_2d(X: np.ndarray, Y: np.ndarray,
                              ell: float, sigma: float, wn_seed: int) -> np.ndarray:
    """Spectrally shaped random field with fixed continuum RMS `sigma`.

    A truncated cosine series on the unit square, so the field is even at
    x = 0, y = 0 and y = 1 by construction and is evaluated analytically at the
    requested coordinates (hence identical across grid resolutions).

    Normalisation is per realisation, which projects the coefficient vector
    onto a fixed-energy surface: the coefficients are no longer jointly
    Gaussian and this is *not* a GRF. That is deliberate — it gives a
    controlled IC amplitude — but it should not be described as Gaussian.
    """
    x, y = _separable_axis_coords(X, Y)
    n = np.arange(RANDOM_FIELD_MODES + 1)
    # <cos^2(pi n z)> over [0, 1]: 1 for n = 0, 1/2 otherwise.
    w = np.where(n == 0, 1.0, 0.5)
    P = np.exp(-(np.pi ** 2) * (n[:, None] ** 2 + n[None, :] ** 2) * ell ** 2 / 2.0)
    g = np.random.default_rng(int(wn_seed)).standard_normal(
        (RANDOM_FIELD_MODES + 1, RANDOM_FIELD_MODES + 1)
    )
    C = g * np.sqrt(P)
    C[0, 0] = 0.0
    norm_sq = float((C ** 2 * w[:, None] * w[None, :]).sum())
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise RuntimeError(
            f"degenerate random-field coefficient realization "
            f"(ell={ell!r}, sigma={sigma!r}, wn_seed={wn_seed!r})"
        )
    C *= sigma / np.sqrt(norm_sq)
    Cx = np.cos(np.pi * np.outer(x, n))
    Cy = np.cos(np.pi * np.outer(y, n))
    # Accelerate leaves the FPU status flags dirty after a BLAS matmul, so
    # numpy reports spurious divide/overflow/invalid here for any operands.
    # Suppress the flags and check the actual result instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        field = (Cx @ C) @ Cy.T
    if not np.isfinite(field).all():
        raise RuntimeError(
            f"non-finite random field (ell={ell!r}, sigma={sigma!r}, "
            f"wn_seed={wn_seed!r})"
        )
    return field


def ic_hot_spot_2d(X: np.ndarray, Y: np.ndarray,
                   A_list, mu_x_list, mu_y_list, sigma_list) -> np.ndarray:
    """Gaussian bumps made wall-compatible by a truncated method of images.

    The isotropic Gaussian factorises, so images apply per axis. The x axis
    gets the -mu_x reflection (zero-Neumann at x = 0) but deliberately *not*
    the 2 - mu_x one: x = 1 is Dirichlet, and that image would leak up to
    0.135 per unit A into the pinned wall.

    The y images are exact only in the infinite-period limit; a finite set
    cannot be exactly even about both y = 0 and y = 1. Within the sampler
    ranges the residual normal derivative is ~8e-10 per unit |A|, which is the
    tolerance the tests assert against.
    """
    A = np.asarray(A_list, dtype=float)
    mux = np.asarray(mu_x_list, dtype=float)
    muy = np.asarray(mu_y_list, dtype=float)
    s = np.asarray(sigma_list, dtype=float)
    two_s2 = 2.0 * s ** 2
    x = X[..., None]
    y = Y[..., None]
    gx = (np.exp(-(x - mux) ** 2 / two_s2)
          + np.exp(-(x + mux) ** 2 / two_s2))
    gy = (np.exp(-(y - muy) ** 2 / two_s2)
          + np.exp(-(y + muy) ** 2 / two_s2)
          + np.exp(-(y - (2.0 - muy)) ** 2 / two_s2))
    return (A * gx * gy).sum(axis=-1)


IC_BUILDERS = {
    "uniform_2d":         lambda X, Y, **p: ic_uniform_2d(X, Y, **p),
    "random_sinusoid_2d": lambda X, Y, **p: ic_random_sinusoid_2d(X, Y, **p),
    "grf_2d":             lambda X, Y, **p: ic_random_cosine_field_2d(X, Y, **p),
    "hot_spot_2d":        lambda X, Y, **p: ic_hot_spot_2d(X, Y, **p),
}

# -------- SAMPLER FUNCTIONS --------

def sample_uniform_ic_params(rng: np.random.Generator) -> dict:
    return {"T0_offset": float(rng.uniform(*UNIFORM_OFFSET_RANGE))}


def sample_random_sinusoid_ic_params(rng: np.random.Generator) -> dict:
    N = int(rng.choice(SINU_N_CHOICES))
    A_list, nx_list, ny_list = [], [], []
    lo, hi = SINU_AMP_RANGE
    for _ in range(N):
        # Signed amplitude; the phase freedom the old sin basis had is gone.
        A_list.append(float(rng.uniform(lo, hi)) * float(rng.choice((-1.0, 1.0))))
        nx_list.append(float(rng.uniform(1.0, SINU_KMAX)))
        ny_list.append(int(rng.integers(1, SINU_KMAX + 1)))
    return {"A_list": A_list, "nx_list": nx_list, "ny_list": ny_list}


def sample_random_field_ic_params(rng: np.random.Generator) -> dict:
    lo, hi = GRF_ELL_RANGE
    u = rng.uniform(0.0, 1.0)
    ell = float(lo * (hi / lo) ** u)
    sigma = float(rng.uniform(*GRF_SIGMA_RANGE))
    wn_seed = int(rng.integers(0, 2 ** 31))
    return {"ell": ell, "sigma": sigma, "wn_seed": wn_seed}


def sample_hot_spot_ic_params(rng: np.random.Generator) -> dict:
    N = int(rng.choice(HOT_N_CHOICES))
    A_list, mu_x_list, mu_y_list, sigma_list = [], [], [], []
    lo, hi = HOT_SIGMA_RANGE
    for _ in range(N):
        u = rng.uniform(0.0, 1.0)
        sigma = float(lo * (hi / lo) ** u)
        m = min(HOT_MARGIN_FACTOR * sigma, 0.45)
        A_list.append(float(rng.uniform(*HOT_AMP_RANGE)))
        mu_x_list.append(float(rng.uniform(m, 1.0 - m)))
        mu_y_list.append(float(rng.uniform(m, 1.0 - m)))
        sigma_list.append(sigma)
    return {"A_list": A_list, "mu_x_list": mu_x_list, "mu_y_list": mu_y_list, "sigma_list": sigma_list}


# No **kw: every sampler is now grid-independent, so a stale Nx=/Ny= call site
# must fail loudly rather than be silently swallowed.
IC_SAMPLERS = {
    "uniform_2d":         sample_uniform_ic_params,
    "random_sinusoid_2d": sample_random_sinusoid_ic_params,
    "grf_2d":             sample_random_field_ic_params,
    "hot_spot_2d":        sample_hot_spot_ic_params,
}


def sample_ic_family(rng: np.random.Generator,
                     probs: np.ndarray | None = None) -> str:
    families = list(IC_FAMILIES.keys())
    if probs is None:
        return str(rng.choice(families))
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(families, p=probs))


def balanced_ic_family_assignments(
    rng: np.random.Generator, batch_size: int,
) -> list[str]:
    """Balanced family labels with unbiased, without-replacement remainders."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    families = np.asarray(tuple(IC_FAMILIES), dtype=object)
    if batch_size < len(families):
        labels = rng.choice(families, size=batch_size, replace=False)
    else:
        quotient, remainder = divmod(batch_size, len(families))
        labels = np.repeat(families, quotient)
        if remainder:
            extra = rng.choice(families, size=remainder, replace=False)
            labels = np.concatenate((labels, extra))
    labels = np.asarray(labels, dtype=object)
    rng.shuffle(labels)
    return [str(label) for label in labels]


def canonical_ic_params(ic_family: str, ic_params: dict) -> dict:
    """Lossless, dtype-explicit IC parameters used by online descriptors."""
    p = dict(ic_params)
    if ic_family == "uniform_2d":
        return {"T0_offset": np.float64(p["T0_offset"])}
    if ic_family == "random_sinusoid_2d":
        return {
            "A_list": np.asarray(p["A_list"], dtype="<f8"),
            # nx is continuous (x = 1 is Dirichlet); ny must stay integral.
            "nx_list": np.asarray(p["nx_list"], dtype="<f8"),
            "ny_list": np.asarray(p["ny_list"], dtype="<i8"),
        }
    if ic_family == "grf_2d":
        return {
            "ell": np.float64(p["ell"]),
            "sigma": np.float64(p["sigma"]),
            "wn_seed": np.int64(p["wn_seed"]),
        }
    if ic_family == "hot_spot_2d":
        return {
            "A_list": np.asarray(p["A_list"], dtype="<f8"),
            "mu_x_list": np.asarray(p["mu_x_list"], dtype="<f8"),
            "mu_y_list": np.asarray(p["mu_y_list"], dtype="<f8"),
            "sigma_list": np.asarray(p["sigma_list"], dtype="<f8"),
        }
    raise ValueError(f"unknown IC family {ic_family!r}")


# -------- TAPER + WRAPPER --------

def _smootherstep_window(dist: np.ndarray, edge_width: float) -> np.ndarray:
    """C2 ramp: w' and w'' both vanish at s = 0 and s = 1.

    The C1 smoothstep 3s^2 - 2s^3 leaves a jump of 6/W^2 in w'' at the seam,
    which enters the diffusion operator directly and renders as a visible
    curvature crease at fixed distance from the wall.
    """
    s = np.clip(dist / edge_width, 0.0, 1.0)
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)


def dirichlet_edge_taper(X: np.ndarray, x_right: float = 1.0,
                         edge_width: float = EDGE_TAPER_WIDTH) -> np.ndarray:
    """Taper into the right Dirichlet wall only.

    `x_right` is the physical wall and must be passed explicitly: deriving it
    from `X.max()` breaks on slices, probes, and off-grid evaluation points.
    """
    return _smootherstep_window(x_right - X, edge_width)


def build_ic(ic_family: str, ic_params: dict,
             X: np.ndarray, Y: np.ndarray, T_right: float,
             b: float = 1.0, taper: bool = True,
             pin_right_edge: bool = True) -> np.ndarray:
    X64 = np.asarray(X, dtype=np.float64)
    Y64 = np.asarray(Y, dtype=np.float64)
    params = canonical_ic_params(ic_family, ic_params)
    dev = IC_BUILDERS[ic_family](X64, Y64, **params)
    if taper:
        dev = dev * dirichlet_edge_taper(X64, x_right=float(b))
    T0 = np.ascontiguousarray(dev + np.float64(T_right), dtype=np.float32)
    if pin_right_edge:
        T0[-1, :] = np.float32(T_right)
    return T0
