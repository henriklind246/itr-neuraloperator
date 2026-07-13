from __future__ import annotations

from typing import Any

import numpy as np

from problems.diffusion_forcing import DiffusionForcingProblem
from src.physics.boundary_forcing import SPATIAL_SAMPLERS, TEMPORAL_SAMPLERS

# The single forcing family this benchmark pins.
FIXED_TEMPORAL_FAMILY = "sin"
FIXED_SPATIAL_FAMILY = "uniform"


class DiffusionForcingSingleProblem(DiffusionForcingProblem):
    """Single-slab forcing benchmark restricted to the sin/uniform family.

    Identical physics to :class:`DiffusionForcingProblem` (single homogeneous
    slab, fixed uniform 300 K IC, left-wall Neumann forcing), but the forcing is
    locked to the ``sin`` temporal family and ``uniform`` spatial profile so only
    the sinusoid amplitude/frequency vary per simulation. This mirrors the
    ``interfaces`` benchmark's forcing setup without the interface, isolating the
    ForcingCViT image/PINO path on a simpler problem.
    """

    name = "diffusion_forcing_single"

    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        X = grids["X"]
        y_grid = grids.get("y_grid")
        if y_grid is None:
            Y = grids["Y"]
            c, d = float(np.min(Y)), float(np.max(Y))
        else:
            c, d = float(y_grid[0]), float(y_grid[-1])
        num_sims = int(time_cfg["num_sims"])
        dt = float(time_cfg["dt"])
        t_final = float(time_cfg["t_final"])
        temporal_window = dict(
            t_on=float(time_cfg.get("t_on", 0.0)),
            t_off=float(time_cfg.get("t_off", 0.2)),
            phase=float(time_cfg.get("phase", 0.0)),
            tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)),
        )

        sim_params = []
        for _ in range(num_sims):
            # Fixed uniform 300 K IC: the forcing is the only per-sim signal.
            ic_family = "uniform_2d"
            ic_params = {"T0_offset": 0.0}
            T0 = np.full(X.shape, 300.0, dtype=np.float32)

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
