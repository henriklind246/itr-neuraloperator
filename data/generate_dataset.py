import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.physics.fv_solver_1d import build_time_grid
from src.physics.fv_solver_2d import FVSolver2D, Layer2D
from src.physics.boundary_forcing import (
    SPATIAL_FAMILIES,
    TEMPORAL_FAMILIES,
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    default_ramp_seconds,
    sample_spatial_family,
    sample_temporal_family,
    build_qL,
    build_qL_integral,
)
from src.physics.init_conditions import (
    IC_FAMILIES,
)
from problems.registry import get_problem


DATA_DIR = Path(__file__).resolve().parent

# Separate seed for the balanced IC-family shuffle. Distinct from the rng=0
# (IC params) and rng_profile=1 (forcing params) streams so that changing IC
# balancing never perturbs the forcing-parameter sequence.
IC_ASSIGN_SEED = 2


def build_balanced_ic_families(
    num_sims: int, allowed_families: list[str], seed: int = IC_ASSIGN_SEED
) -> list[str]:
    """Return a length-``num_sims`` per-sim IC-family list with equal quotas.

    Each allowed family gets ``num_sims // n`` sims; the first ``num_sims % n``
    families get one extra, so per-family counts differ by at most one. The list
    is shuffled by an independent seeded RNG so the assignment order is
    decorrelated from the forcing/IC-parameter draws.
    """
    n = len(allowed_families)
    if n == 0:
        raise ValueError("allowed_families must be non-empty.")
    base, rem = divmod(num_sims, n)
    families: list[str] = []
    for i, fam in enumerate(allowed_families):
        families.extend([fam] * (base + (1 if i < rem else 0)))
    rng = np.random.default_rng(seed)
    rng.shuffle(families)
    return families


def generate_lhs_samples(num_sims: int, seed: int = 0) -> np.ndarray:
    # R_c is the only material parameter sampled here; forcing-family params
    # are drawn per-sim by the boundary-forcing samplers (see TEMPORAL_SAMPLERS).
    param_ranges = {"R_c": (0.05, 1.0)}
    param_names = list(param_ranges.keys())
    sample_dim = len(param_names)
    lower_bounds = np.array([param_ranges[name][0] for name in param_names], dtype=np.float32)
    upper_bounds = np.array([param_ranges[name][1] for name in param_names], dtype=np.float32)

    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, num_sims + 1, dtype=np.float64)
    widths = edges[1:] - edges[:-1]
    samples_unit = edges[:-1, None] + widths[:, None] * rng.random((num_sims, sample_dim))
    for j in range(sample_dim):
        rng.shuffle(samples_unit[:, j])

    samples_scaled = lower_bounds + samples_unit * (upper_bounds - lower_bounds)
    samples_scaled = samples_scaled.astype(np.float32)
    return samples_scaled


def build_sim_params(a: float, b: float, c: float, d: float, X: np.ndarray, Y: np.ndarray,
                     num_sims: int, rng, rng_profile,
                     dt: float, t_final: float, lhs_seed: int = 0,
                     t_on: float = 0.0, t_off: float = 0.2, phase: float = 0.0,
                     tukey_alpha: float = 0.5,
                     T_right: float = 300.0) -> list:

    samples_scaled = generate_lhs_samples(num_sims=num_sims, seed=lhs_seed)
    R_c_values = samples_scaled[:, 0]

    temporal_window = dict(t_on=t_on, t_off=t_off, phase=phase, tukey_alpha=tukey_alpha)

    # building full list of simulation parameters
    sim_params = []
    for i in range(num_sims):
        R_c = float(R_c_values[i])

        ic_family = "uniform_2d"
        ic_params = {"T0_offset": 0.0}
        T0 = np.full(X.shape, 300.0, dtype=np.float32)

        temporal_family = sample_temporal_family(rng_profile)
        temporal_params = TEMPORAL_SAMPLERS[temporal_family](rng_profile, dt=dt, t_final=t_final, **temporal_window)

        spatial_family = sample_spatial_family(rng_profile)
        spatial_params = SPATIAL_SAMPLERS[spatial_family](rng_profile, c=c, d=d)

        sim_params.append({
            "R_c": R_c,
            "T0": T0,
            "ic_family": ic_family,
            "ic_params": ic_params,
            "temporal_family": temporal_family,
            "temporal_params": temporal_params,
            "spatial_family": spatial_family,
            "spatial_params": spatial_params,
        })

    return sim_params


def build_base_setup(
    num_sims: int,
    save_stride: int,
    nx: int = 100,
    ny: int = 100,
    t_final: float = 0.3,
    dt: float = 0.005,
    ramp_seconds: float | None = None,
    lhs_seed: int = 0,
    ic_families: list[str] | None = None,
) -> dict:
    """Build the fixed geometry/time scaffolding shared by every simulation.

    Returns a bundle with the ``grids`` and ``time_cfg`` consumed by
    ``sample_sim_params``/``draw_latents``, the ``base_kwargs`` passed to
    ``configure_solver``, plus the resolved ``dt``, ``t_ramp``, and
    ``Nt_saved`` needed to size and save trajectories. ``t_final``/``dt`` are
    parameters (not hardcoded) so the OOD generator can solve to an extended
    horizon, but the defaults reproduce the standard dataset exactly.
    """
    a, b, c, d = 0.0, 1.0, 0.0, 1.0
    Nx, Ny = nx, ny

    # Physical startup-ramp width applied to q_L so q_L(0)=0. Resolve once here
    # (CLI override else dt-derived default) and persist the resolved value with
    # the dataset so the solver flux and the model's forcing conditioning use the
    # exact same ramp, independent of dt.
    t_ramp = float(ramp_seconds) if ramp_seconds is not None else default_ramp_seconds(dt)

    t_on, t_off = 0.0, 0.2
    phase = 0.0
    tukey_alpha = 0.5

    x_grid = np.linspace(a, b, Nx)
    y_grid = np.linspace(c, d, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    dt, t_grid_template = build_time_grid(t_final, dt, explicit_dt=True)
    Nt_saved = len(t_grid_template[::save_stride])

    x_mid = 0.5 * (a + b)
    layers = [
        Layer2D(x_left=a, x_right=x_mid, rho=1.0, cp=1.0, k=2.0),
        Layer2D(x_left=x_mid, x_right=b, rho=1.0, cp=1.0, k=1.0),
    ]

    grids = {"X": X, "Y": Y, "x_grid": x_grid, "y_grid": y_grid}
    time_cfg = dict(
        num_sims=num_sims, dt=dt, t_final=t_final, lhs_seed=lhs_seed,
        forcing_profile_seed=1,
        t_on=t_on, t_off=t_off, phase=phase, tukey_alpha=tukey_alpha,
        T_right=300.0, b=b, ic_families=ic_families,
    )
    base_kwargs = dict(
        a=a, b=b, c=c, d=d, Nx=Nx, Ny=Ny,
        lam_target=0.8, layers=layers, t_final=t_final,
        flux_f=0.0, flux_A=0.0, t_on=t_on, t_off=t_off, phase=phase,
        dt=dt, tukey_alpha=tukey_alpha, y_grid=y_grid, X=X, Y=Y,
        ramp_seconds=t_ramp,
    )
    return {
        "grids": grids,
        "time_cfg": time_cfg,
        "base_kwargs": base_kwargs,
        "dt": dt,
        "t_ramp": t_ramp,
        "Nt_saved": Nt_saved,
        "Nx": Nx,
        "Ny": Ny,
    }


def run_solves(
    spec,
    sim_params: list[dict],
    base_kwargs: dict,
    save_stride: int,
    Nt_saved: int,
    Nx: int,
    Ny: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve every ``sim_params`` entry and stack decimated trajectories.

    Returns ``(trajectories, t, x, y)`` where ``trajectories`` is
    ``(num_sims, Nt_saved, Nx, Ny)`` float32 and ``t/x/y`` come from the solver
    (identical across sims since the grid is shared).
    """
    num_sims = len(sim_params)
    trajectories = np.zeros((num_sims, Nt_saved, Nx, Ny), dtype=np.float32)
    t = x = y = None
    for i, params in enumerate(sim_params):
        sim = spec.configure_solver(params, base_kwargs)
        t, x, y, T_hist = sim.solve(T0=params["T0"], store_trajectory=True)
        trajectories[i] = T_hist[::save_stride].astype(np.float32)
        if verbose:
            print(f"Finished simulation {i}", flush=True)
    return trajectories, t, x, y


def save_dataset(
    save_path: Path | str,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    save_stride: int,
    dt: float,
    t_ramp: float,
    trajectories: np.ndarray,
    sim_params: list[dict],
    meta: dict | None = None,
) -> Path:
    """Write the standard dataset .npy files and return the save directory."""
    x_grid = x.astype(np.float32)
    y_grid = y.astype(np.float32)
    t_grid = t[::save_stride].astype(np.float32)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "x_grid.npy", x_grid)
    np.save(save_path / "y_grid.npy", y_grid)
    np.save(save_path / "t_grid.npy", t_grid)
    # Saved t_grid spacing != solver dt when save_stride > 1; persist solver dt
    # so the dataset can normalize tau / dt_n cond slots against the same bounds
    # used during sampling.
    np.save(save_path / "dt.npy", np.float64(dt))
    # Persist the resolved physical ramp width so build_item reproduces the exact
    # q_L(t) the solver integrated, decoupled from dt across regenerations.
    np.save(save_path / "ramp_seconds.npy", np.float64(t_ramp))
    np.save(save_path / "trajectories.npy", trajectories)
    np.save(save_path / "sim_params.npy", np.array(sim_params, dtype=object), allow_pickle=True)
    # Dataset-format tag + IC provenance. In-place editing of a varying-IC
    # benchmark keeps the same name and tensor shapes as the old fixed-IC set, so
    # meta.npy is the only signal that distinguishes them and is asserted on load.
    if meta is not None:
        np.save(save_path / "meta.npy", np.array(meta, dtype=object), allow_pickle=True)
    print("Saved to:", save_path, x_grid.shape, y_grid.shape, t_grid.shape, trajectories.shape, flush=True)
    return save_path


def generate_sim_data(
    num_sims: int = 2000,
    save_stride: int = 2,
    save_dir: Path | str | None = None,
    benchmark: str | None = None,
    nx: int = 100,
    ny: int = 100,
    ic_families: list[str] | None = None,
    ramp_seconds: float | None = None,
) -> None:
    benchmark = benchmark or os.environ.get("BENCHMARK", "forcing")
    spec = get_problem(benchmark)

    rng = np.random.default_rng(0)
    rng_profile = np.random.default_rng(1)

    setup = build_base_setup(
        num_sims=num_sims, save_stride=save_stride, nx=nx, ny=ny,
        ramp_seconds=ramp_seconds, lhs_seed=0, ic_families=ic_families,
    )

    print(f"Benchmark: {benchmark}", flush=True)
    print("Building simulation parameters.", flush=True)

    # Varying-IC benchmarks consume a per-sim balanced family assignment built by
    # a separate seeded RNG (so IC balancing never perturbs the forcing draws).
    # The spec reads a scalar per sim; it never decides the balance itself. Other
    # benchmarks (no ic_mode attribute) are byte-unaffected: the key is absent.
    allowed_ic_families = ic_families if ic_families is not None else list(IC_FAMILIES.keys())
    if getattr(spec, "ic_mode", None) == "varying":
        setup["time_cfg"]["ic_family_assignment"] = build_balanced_ic_families(
            num_sims=num_sims, allowed_families=allowed_ic_families, seed=IC_ASSIGN_SEED,
        )

    sim_params = spec.sample_sim_params(
        rng=rng, rng_profile=rng_profile,
        grids=setup["grids"], time_cfg=setup["time_cfg"],
    )

    trajectories, t, x, y = run_solves(
        spec, sim_params, setup["base_kwargs"], save_stride,
        setup["Nt_saved"], setup["Nx"], setup["Ny"],
    )

    required_times = getattr(spec, "required_observation_times", ())
    if required_times:
        saved_times = np.asarray(t, dtype=np.float64)[::save_stride]
        for requested in required_times:
            if not np.any(np.isclose(saved_times, requested, rtol=0.0, atol=1e-10)):
                raise ValueError(
                    f"benchmark {benchmark!r} requires saved snapshot time "
                    f"{requested:g}; generated schedule is {saved_times.tolist()}."
                )

    # Persist the dataset-format tag + IC provenance only when the spec declares a
    # version, so the load-time guard can reject a stale fixed-IC dataset that
    # shares this benchmark's name and shapes.
    meta = None
    problem_version = getattr(spec, "problem_version", None)
    if problem_version is not None:
        meta = {
            "problem_version": problem_version,
            "ic_mode": getattr(spec, "ic_mode", None),
            "ic_families": list(allowed_ic_families),
        }

    save_path = Path(save_dir) if save_dir is not None else DATA_DIR
    save_dataset(
        save_path, x, y, t, save_stride, setup["dt"], setup["t_ramp"],
        trajectories, sim_params, meta=meta,
    )

def main(argv: list[str] | None = None, generate_fn=generate_sim_data) -> int:
    parser = argparse.ArgumentParser(description="Generate FNO training data.")
    parser.add_argument("--num-sims", type=int, default=8000, help="Number of simulations to generate.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for generated .npy files. Defaults to this script's data directory.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=os.environ.get("BENCHMARK", "forcing"),
        help="Registered benchmark adapter to generate.",
    )
    parser.add_argument("--nx", type=int, default=100, help="Number of x-direction grid nodes.")
    parser.add_argument("--ny", type=int, default=100, help="Number of y-direction grid nodes.")
    parser.add_argument(
        "--save-stride",
        type=int,
        default=2,
        help=(
            "Snapshot decimation stride. Default 2. Use 1 for the "
            "physics-informed FV-residual set (consecutive snapshots one solver "
            "step dt apart; CN-exact). Note Nt_saved ~doubles and the all-to-all "
            "pair count is quadratic in Nt_saved -- generate a SMALL save-stride=1 "
            "set, not the full all-to-all corpus."
        ),
    )
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        default=None,
        help=(
            "Physical startup-ramp width (seconds) applied to q_L so q_L(0)=0. "
            "Defaults to 2*dt. The resolved value is persisted with the dataset."
        ),
    )
    parser.add_argument(
        "--exclude-ic",
        action="append",
        default=None,
        choices=list(IC_FAMILIES.keys()),
        help=(
            "IC family to exclude from sampling (repeatable). "
            "E.g. --exclude-ic grf_2d for the resolution-invariance test "
            "(grf_2d cannot be reproduced across grids). Default: all families."
        ),
    )
    args = parser.parse_args(argv)

    if args.exclude_ic:
        excluded = set(args.exclude_ic)
        ic_families = [f for f in IC_FAMILIES if f not in excluded]
        if not ic_families:
            parser.error("--exclude-ic cannot exclude every IC family.")
    else:
        ic_families = None

    generate_fn(
        num_sims=args.num_sims,
        save_dir=args.save_dir,
        benchmark=args.benchmark,
        nx=args.nx,
        ny=args.ny,
        save_stride=args.save_stride,
        ic_families=ic_families,
        ramp_seconds=args.ramp_seconds,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
