from __future__ import annotations

from typing import Any

import numpy as np

from problems.diffusion_forcing import DiffusionForcingProblem
from src.physics.boundary_forcing import SPATIAL_SAMPLERS, TEMPORAL_SAMPLERS
from src.physics.init_conditions import (
    IC_BUILDER_SCHEMA_VERSION,
    IC_FAMILIES,
    IC_SAMPLERS,
    ONLINE_IC_SAMPLER_VERSION,
    build_ic,
    canonical_ic_params,
    sample_ic_family,
)

# The single forcing family this benchmark pins.
FIXED_TEMPORAL_FAMILY = "sin"
FIXED_SPATIAL_FAMILY = "uniform"

# Dataset-format tag stamped into meta.npy at generation and asserted on load.
# In-place editing keeps the benchmark name and tensor shapes identical to the
# old fixed-300 K set, so this string is the only signal that distinguishes a
# varying-IC dataset from a stale fixed-IC one; bump it on any breaking change.
PROBLEM_VERSION = "forcing_single_varying_ic_v1"
IC_MODE = "varying"


class DiffusionForcingSingleProblem(DiffusionForcingProblem):
    """Single-slab forcing benchmark: sin/uniform forcing, varying IC.

    Identical physics to :class:`DiffusionForcingProblem` (single homogeneous
    slab, left-wall Neumann forcing) with the forcing locked to the ``sin``
    temporal family and ``uniform`` spatial profile, but the **initial condition
    varies per simulation** across all four families in
    ``src.physics.init_conditions`` (``uniform_2d``, ``random_sinusoid_2d``,
    ``grf_2d``, ``hot_spot_2d``). This mirrors the ``interfaces`` benchmark's
    IC + forcing setup without the interface, isolating the two-branch
    ForcingICCViT (forcing image + IC field) PINO path on a simpler problem.

    IC-family *balancing* lives in the generator, which hands this spec an
    explicit per-sim ``ic_family_assignment`` list via ``time_cfg`` (a separate
    seeded RNG shuffles it, so IC balancing never perturbs the forcing-parameter
    sequence). The spec only consumes ``ic_family_assignment[i]`` for sim ``i``;
    it never decides the balance itself. IC parameter draws use ``rng`` and the
    forcing draws use ``rng_profile`` so the two streams stay decoupled.
    """

    name = "diffusion_forcing_single"

    # Dataset version / mode advertised to the generator (meta.npy) and the
    # load-time guard. Present only on this spec.
    problem_version = PROBLEM_VERSION
    ic_mode = IC_MODE
    online_sampler_version = ONLINE_IC_SAMPLER_VERSION
    ic_builder_version = IC_BUILDER_SCHEMA_VERSION

    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        X = grids["X"]
        Y = grids["Y"]
        Nx, Ny = X.shape[0], X.shape[1]
        y_grid = grids.get("y_grid")
        if y_grid is None:
            c, d = float(np.min(Y)), float(np.max(Y))
        else:
            c, d = float(y_grid[0]), float(y_grid[-1])
        num_sims = int(time_cfg["num_sims"])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        b_temp = float(time_cfg.get("b", 1.0))
        T_right = float(time_cfg.get("T_right", 300.0))
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        # Per-sim IC family assignment. The generator supplies a balanced,
        # separately-shuffled list; fall back to an IID per-sim draw only when it
        # is absent (e.g. an ad-hoc caller) so behavior stays well-defined.
        assignment = time_cfg.get("ic_family_assignment")
        if assignment is not None and len(assignment) != num_sims:
            raise ValueError(
                f"ic_family_assignment length {len(assignment)} != num_sims "
                f"{num_sims} for benchmark {self.name!r}."
            )

        sim_params = []
        for i in range(num_sims):
            if assignment is not None:
                ic_family = str(assignment[i])
            else:
                ic_family = str(rng.choice(list(IC_FAMILIES.keys())))
            if ic_family not in IC_FAMILIES:
                raise ValueError(
                    f"ic_family {ic_family!r} for sim {i} is not a known family "
                    f"{tuple(IC_FAMILIES)}."
                )
            ic_params = IC_SAMPLERS[ic_family](rng, Nx=Nx, Ny=Ny)
            T0 = build_ic(ic_family, ic_params, X, Y, T_right=T_right, b=b_temp)

            temporal_family = FIXED_TEMPORAL_FAMILY
            temporal_params = TEMPORAL_SAMPLERS[temporal_family](
                rng_profile, dt=dt, t_final=t_final, **temporal_window
            )
            spatial_family = FIXED_SPATIAL_FAMILY
            spatial_params = SPATIAL_SAMPLERS[spatial_family](rng_profile, c=c, d=d)

            sim_params.append({
                "T0": T0,
                "ic_family": ic_family,
                "ic_params": ic_params,
                "temporal_family": temporal_family,
                "temporal_params": temporal_params,
                "spatial_family": spatial_family,
                "spatial_params": spatial_params,
            })
        return sim_params

    def sample_online_params(
        self,
        rng: np.random.Generator,
        n: int,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
        rng_profile: np.random.Generator | None = None,
        *,
        rng_streams: dict[str, np.random.Generator] | None = None,
        ic_family_assignment: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        streams = rng_streams or {}
        ic_family_rng = streams.get("ic_family", rng)
        ic_param_rng = streams.get("ic_params", rng)
        forcing_rng = streams.get(
            "forcing_params", rng_profile if rng_profile is not None else rng,
        )
        X = np.asarray(grids["X"], dtype=np.float64)
        Y = np.asarray(grids["Y"], dtype=np.float64)
        Nx, Ny = X.shape
        y_grid = np.asarray(grids.get("y_grid", Y[0]), dtype=np.float64)
        c, d = float(y_grid[0]), float(y_grid[-1])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        T_right = float(time_cfg.get("T_right", 300.0))
        b_temp = float(time_cfg.get("b", 1.0))
        temporal_window = {
            "t_on": float(time_cfg.get("t_on", 0.0)),
            "t_off": float(time_cfg.get("t_off", 0.2)),
            "phase": float(time_cfg.get("phase", 0.0)),
            "tukey_alpha": float(time_cfg.get("tukey_alpha", 0.5)),
        }
        if ic_family_assignment is not None and len(ic_family_assignment) != int(n):
            raise ValueError(
                "ic_family_assignment length must equal the online batch size"
            )

        records: list[dict] = []
        for i in range(int(n)):
            family = (
                str(ic_family_assignment[i])
                if ic_family_assignment is not None
                else sample_ic_family(ic_family_rng)
            )
            if family not in IC_FAMILIES:
                raise ValueError(f"unknown IC family {family!r}")
            ic_params = canonical_ic_params(
                family, IC_SAMPLERS[family](ic_param_rng, Nx=Nx, Ny=Ny),
            )
            temporal_params = TEMPORAL_SAMPLERS[FIXED_TEMPORAL_FAMILY](
                forcing_rng, dt=dt, t_final=t_final, **temporal_window,
            )
            spatial_params = SPATIAL_SAMPLERS[FIXED_SPATIAL_FAMILY](
                forcing_rng, c=c, d=d,
            )
            records.append({
                "T0": build_ic(
                    family, ic_params, X, Y, T_right=T_right, b=b_temp,
                ),
                "ic_family": family,
                "ic_params": ic_params,
                "temporal_family": FIXED_TEMPORAL_FAMILY,
                "temporal_params": temporal_params,
                "spatial_family": FIXED_SPATIAL_FAMILY,
                "spatial_params": spatial_params,
            })
        return records

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        # Parent enforces required keys and forbids interface_x / R_c.
        super().validate_schema(sim_params, sim_ids)
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            if entry["temporal_family"] != FIXED_TEMPORAL_FAMILY:
                raise ValueError(
                    f"sim_params[{int(sid)}] has temporal_family="
                    f"{entry['temporal_family']!r}; benchmark {self.name!r} admits "
                    f"only {FIXED_TEMPORAL_FAMILY!r}."
                )
            if entry["spatial_family"] != FIXED_SPATIAL_FAMILY:
                raise ValueError(
                    f"sim_params[{int(sid)}] has spatial_family="
                    f"{entry['spatial_family']!r}; benchmark {self.name!r} admits "
                    f"only {FIXED_SPATIAL_FAMILY!r}."
                )
            if entry["ic_family"] not in IC_FAMILIES:
                raise ValueError(
                    f"sim_params[{int(sid)}] has ic_family="
                    f"{entry['ic_family']!r}; benchmark {self.name!r} admits only "
                    f"{tuple(IC_FAMILIES)}."
                )
