from itertools import product

import numpy as np

from problems.diffusion_forcing_single import DiffusionForcingSingleProblem
from problems.forcing import FORCING_TEMPORAL_SAMPLES
from src.physics.boundary_forcing import (
    TEMPORAL_FAMILIES, SPATIAL_FAMILIES, TEMPORAL_SAMPLERS, SPATIAL_SAMPLERS,
    sample_temporal_family, sample_spatial_family,
)


class ForcingHomogeneousProblem(DiffusionForcingSingleProblem):
    """Full forcing distribution on a homogeneous k=1 slab, starting at 300 K."""

    name = "forcing_homogeneous"
    problem_version = "forcing_homogeneous_v1"
    online_sampler_version = "forcing_homogeneous_online_v1"
    ic_builder_version = "uniform_300K_v1"
    ic_mode = "uniform"
    ic_families = ("uniform_2d",)
    temporal_families = tuple(TEMPORAL_FAMILIES)
    spatial_families = tuple(SPATIAL_FAMILIES)

    def sample_sim_params(self, rng, rng_profile, grids, time_cfg):
        return self.sample_online_params(rng, int(time_cfg["num_sims"]), grids,
                                         time_cfg, rng_profile)

    def sample_online_params(self, rng, n, grids, time_cfg, rng_profile=None, *,
                             rng_streams=None, forcing_family_assignment=None):
        streams = rng_streams or {}
        forcing_rng = streams.get("forcing_params", rng_profile if rng_profile is not None else rng)
        if forcing_family_assignment is not None and len(forcing_family_assignment) != n:
            raise ValueError("forcing_family_assignment length must equal the online batch size")
        y = np.asarray(grids.get("y_grid", grids["Y"][0]))
        window = dict(t_on=float(time_cfg.get("t_on", 0.0)),
                      t_off=float(time_cfg.get("t_off", 0.2)),
                      phase=float(time_cfg.get("phase", 0.0)),
                      tukey_alpha=float(time_cfg.get("tukey_alpha", 0.5)))
        records = []
        for i in range(n):
            temporal = (forcing_family_assignment[i][0] if forcing_family_assignment is not None
                        else sample_temporal_family(forcing_rng))
            temporal_params = TEMPORAL_SAMPLERS[temporal](
                forcing_rng, dt=float(time_cfg["dt"]), t_final=float(time_cfg["t_final"]), **window)
            spatial = (forcing_family_assignment[i][1] if forcing_family_assignment is not None
                       else sample_spatial_family(forcing_rng))
            spatial_params = SPATIAL_SAMPLERS[spatial](forcing_rng, c=float(y[0]), d=float(y[-1]))
            records.append(dict(T0=np.full(grids["X"].shape, 300.0, dtype=np.float32),
                                ic_family="uniform_2d", ic_params={"T0_offset": 0.0},
                                temporal_family=temporal, temporal_params=temporal_params,
                                spatial_family=spatial, spatial_params=spatial_params))
        return records

    def validation_assignments(self, count, rng):
        combinations = list(product(self.temporal_families, self.spatial_families))
        if not isinstance(count, int) or count < len(combinations) or count % len(combinations):
            raise ValueError("validation_cases must be a positive multiple of the 16 forcing combinations")
        assignments = combinations * (count // len(combinations))
        rng.shuffle(assignments)
        return {"forcing_family_assignment": assignments}

    def fixed_case_ids(self, params, train_ids):
        selected = {}
        for sid in train_ids:
            p = params[int(sid)]
            selected.setdefault((p["temporal_family"], p["spatial_family"]), int(sid))
        combinations = list(product(self.temporal_families, self.spatial_families))
        missing = [pair for pair in combinations if pair not in selected]
        if missing:
            raise ValueError(
                f"Fixed forcing screen needs all 16 combinations in the training split; "
                f"missing {missing}. Generate more prescribed cases."
            )
        return [selected[pair] for pair in combinations]

    def validate_schema(self, sim_params, sim_ids):
        super().validate_schema(sim_params, sim_ids)
        for sid in sim_ids:
            p = sim_params[int(sid)]
            if (p["ic_family"] != "uniform_2d" or p["ic_params"] != {"T0_offset": 0.0}
                    or not np.all(np.asarray(p["T0"]) == 300.0)):
                raise ValueError(f"sim_params[{int(sid)}] must prescribe a uniform 300 K initial condition")

    def build_step_inputs(self, state, coords, average_flux, dt, horizon):
        import torch

        batch, nx, ny = state.shape
        xy = coords.reshape(1, nx, ny, 2).expand(batch, -1, -1, -1)
        # Keep the complete signed boundary profile; a scalar mean loses patches
        # and a unit temporal amplitude also represents zero-flux intervals exactly.
        profile = average_flux[:, None, :, None].expand(-1, nx, -1, -1)
        r = torch.linspace(0, 1, FORCING_TEMPORAL_SAMPLES,
                           device=state.device, dtype=state.dtype).expand(batch, -1)
        return dict(spatial=torch.cat((state[..., None], xy, profile), dim=-1),
                    cond_static=state.new_full((batch, self.dims.cond_static_dim), dt / horizon),
                    forcing_seq=torch.stack((r, torch.ones_like(r), torch.ones_like(r)), dim=-1))
