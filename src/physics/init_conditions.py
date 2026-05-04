import numpy as np

"""
Per-simulation initial-condition families for the 2D heat-conduction dataset.

Mirrors the spatial / temporal forcing pattern in `boundary_forcing.py`:
each family has a builder, a sampler, and entries in the IC_BUILDERS /
IC_SAMPLERS registries. `build_ic` is the top-level wrapper that adds the
T_right offset, applies a smoothstep taper at the right Dirichlet edge to
avoid a spurious initial discontinuity, and pins the last column exactly.
"""

IC_FAMILIES = {"uniform_2d": 0, "random_sinusoid_2d": 1, "grf_2d": 2, "hot_spot_2d": 3}

UNIFORM_OFFSET_RANGE = (-20.0, 20.0)

SINU_N_CHOICES = (3, 4, 5)
SINU_KMAX = 4
SINU_AMP_RANGE = (1.0, 8.0)

GRF_ELL_RANGE = (0.05, 0.30)
GRF_SIGMA_RANGE = (2.0, 20.0)

HOT_N_CHOICES = (1, 2, 3, 4)
HOT_SIGMA_RANGE = (0.05, 0.20)
HOT_AMP_RANGE = (-30.0, 30.0)
HOT_MARGIN_FACTOR = 2.0

EDGE_TAPER_WIDTH = 0.10

# -------- BUILDER FUNCTIONS --------

def ic_uniform_2d(X: np.ndarray, Y: np.ndarray, T0_offset: float) -> np.ndarray:
    return np.full(X.shape, float(T0_offset), dtype=float)


def ic_random_sinusoid_2d(X: np.ndarray, Y: np.ndarray,
                          A_list, nx_list, ny_list, phi_list,
                          Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
    A = np.asarray(A_list, dtype=float)
    nx = np.asarray(nx_list, dtype=float)
    ny = np.asarray(ny_list, dtype=float)
    phi = np.asarray(phi_list, dtype=float)
    # broadcast: (Nx, Ny, 1) * (N,) -> (Nx, Ny, N), sum along N
    kx = (2.0 * np.pi * nx / Lx)
    ky = (2.0 * np.pi * ny / Ly)
    arg = X[..., None] * kx + Y[..., None] * ky + phi
    return (A * np.sin(arg)).sum(axis=-1)


def ic_grf_2d(X: np.ndarray, Y: np.ndarray,
              ell: float, sigma: float, wn_seed: int,
              Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:
    Nx, Ny = X.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Pk = np.exp(-(KX ** 2 + KY ** 2) * (ell ** 2) / 2.0)
    W = np.random.default_rng(int(wn_seed)).standard_normal((Nx, Ny))
    field = np.real(np.fft.ifft2(np.fft.fft2(W) * np.sqrt(Pk)))
    field -= field.mean()
    std = field.std()
    if std > 0.0:
        field *= sigma / std
    return field


def ic_hot_spot_2d(X: np.ndarray, Y: np.ndarray,
                   A_list, mu_x_list, mu_y_list, sigma_list) -> np.ndarray:
    A = np.asarray(A_list, dtype=float)
    mux = np.asarray(mu_x_list, dtype=float)
    muy = np.asarray(mu_y_list, dtype=float)
    s = np.asarray(sigma_list, dtype=float)
    dx = X[..., None] - mux
    dy = Y[..., None] - muy
    bumps = A * np.exp(-(dx ** 2 + dy ** 2) / (2.0 * s ** 2))
    return bumps.sum(axis=-1)


IC_BUILDERS = {
    "uniform_2d":         lambda X, Y, **p: ic_uniform_2d(X, Y, **p),
    "random_sinusoid_2d": lambda X, Y, **p: ic_random_sinusoid_2d(X, Y, **p),
    "grf_2d":             lambda X, Y, **p: ic_grf_2d(X, Y, **p),
    "hot_spot_2d":        lambda X, Y, **p: ic_hot_spot_2d(X, Y, **p),
}

# -------- SAMPLER FUNCTIONS --------

def sample_uniform_ic_params(rng: np.random.Generator) -> dict:
    return {"T0_offset": float(rng.uniform(*UNIFORM_OFFSET_RANGE))}


def sample_random_sinusoid_ic_params(rng: np.random.Generator) -> dict:
    N = int(rng.choice(SINU_N_CHOICES))
    A_list, nx_list, ny_list, phi_list = [], [], [], []
    for _ in range(N):
        A_list.append(float(rng.uniform(*SINU_AMP_RANGE)))
        nx_list.append(int(rng.integers(1, SINU_KMAX + 1)))
        ny_list.append(int(rng.integers(1, SINU_KMAX + 1)))
        phi_list.append(float(rng.uniform(0.0, 2.0 * np.pi)))
    return {"A_list": A_list, "nx_list": nx_list, "ny_list": ny_list, "phi_list": phi_list}


def sample_grf_ic_params(rng: np.random.Generator, Nx: int, Ny: int) -> dict:
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


IC_SAMPLERS = {
    "uniform_2d":         lambda rng, **kw: sample_uniform_ic_params(rng),
    "random_sinusoid_2d": lambda rng, **kw: sample_random_sinusoid_ic_params(rng),
    "grf_2d":             lambda rng, Nx, Ny, **kw: sample_grf_ic_params(rng, Nx, Ny),
    "hot_spot_2d":        lambda rng, **kw: sample_hot_spot_ic_params(rng),
}


def sample_ic_family(rng: np.random.Generator,
                     probs: np.ndarray | None = None) -> str:
    families = list(IC_FAMILIES.keys())
    if probs is None:
        return str(rng.choice(families))
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(families, p=probs))


# -------- TAPER + WRAPPER --------

def right_edge_taper(X: np.ndarray, b: float = 1.0,
                     edge_width: float = EDGE_TAPER_WIDTH) -> np.ndarray:
    s = np.clip((b - X) / edge_width, 0.0, 1.0)
    return 3.0 * s ** 2 - 2.0 * s ** 3


def build_ic(ic_family: str, ic_params: dict,
             X: np.ndarray, Y: np.ndarray, T_right: float,
             b: float = 1.0, taper: bool = True,
             pin_right_edge: bool = True) -> np.ndarray:
    dev = IC_BUILDERS[ic_family](X, Y, **ic_params)
    if taper:
        dev = dev * right_edge_taper(X, b=b)
    T0 = (dev + T_right).astype(np.float32)
    if pin_right_edge:
        T0[-1, :] = np.float32(T_right)
    return T0
