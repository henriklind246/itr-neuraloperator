"""One-step Markov operator forward and free rollout for InterfaceCViT.

Production composition of the physics primitives in
`src.physics.one_step_objective` with the state-conditioned InterfaceCViT
encoder/decoder. `predict_one_step_field` maps a normalized state `T_n` and an
interval descriptor to `T_{n+1}` under the selected interface-flux closure;
`rollout_field` autoregresses that map for free-rollout validation (plan
Section 2).

Each case is a single-interface forward: the storage-projection and closure
heads read the shared face index `geom.face_idx[0]`, so a batched forward must
carry one geometry (uniform `interface_x`, `R_c`). The online runner therefore
draws one case per forward and averages gradients across cases (plan Section 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from problems.interfaces import normalize_interface_scalars
from src.operators.cvit import moving_interface_jump_enrichment
from src.operators.train_pino import (
    _decode_in_chunks,
    _interface_spatial_channels_from_normalized,
)
from src.physics.boundary_forcing import build_interface_forcing, build_qL_integral
from src.physics.fv_residual import build_cn_geom_per_interface
from src.physics.one_step_objective import (
    build_cn_tensors,
    conservative_energy_storage_projection,
    left_energy_interface_flux_closure,
    state_conditioned_spatial,
    two_sided_energy_interface_flux_closure,
)

__all__ = [
    "predict_one_step_field",
    "rollout_field",
    "build_interval_forcing_images",
    "material_channels",
    "OneStepCase",
    "prepare_interfaces_one_step_case",
    "OneStepStatePool",
]


def predict_one_step_field(
    model,
    state: torch.Tensor,
    interval_image: torch.Tensor,
    scalars: torch.Tensor,
    fixed_material_channels: torch.Tensor,
    coords: torch.Tensor,
    *,
    dt: float,
    query_chunk: int,
    interface_x: torch.Tensor | None = None,
    jump_scale: torch.Tensor | None = None,
    closure_geom=None,
    q_left_integral: torch.Tensor | None = None,
    resistance: float | None = None,
    sigma: float | None = None,
    q_ref: float | None = None,
    return_interface_flux: bool = False,
    return_storage_projection: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Map normalized `state` (T_n) to the next state field T_{n+1}.

    Builds the state-conditioned spatial stack, encodes the InterfaceCViT
    latent, decodes the smooth field at `dt`, then applies the selected
    interface-flux head (learned, left/two-sided energy closure, or the hard
    conservative storage projection). `state` is used only as input; the caller
    detaches it before the target-step objective (plan Section 4).
    """
    spatial = state_conditioned_spatial(state, fixed_material_channels)
    latent = model.encode(spatial, interval_image, scalars)
    time_query = torch.full(
        (state.shape[0], coords.shape[1], 1), float(dt), device=state.device
    )
    if not bool(getattr(model, "jump_enrichment", False)):
        prediction = _decode_in_chunks(
            model, latent, coords, time_query, int(query_chunk)
        )[..., 0].view(state.shape[0], state.shape[1], state.shape[2])
        if return_interface_flux:
            raise RuntimeError("interface flux requested without jump enrichment")
        if return_storage_projection:
            raise RuntimeError("storage projection requested without jump enrichment")
        return prediction
    if interface_x is None or jump_scale is None:
        raise ValueError("jump enrichment requires interface_x and jump_scale")
    flux_mode = str(getattr(model, "jump_flux_mode", "learned"))
    if flux_mode in {
        "left_energy_closure",
        "two_sided_energy_closure",
        "conservative_storage_projection",
    }:
        if any(value is None for value in (
            closure_geom, q_left_integral, resistance, sigma, q_ref
        )):
            raise ValueError("left-energy closure inputs are incomplete")
        smooth = _decode_in_chunks(
            model, latent, coords, time_query, int(query_chunk)
        )[..., 0].view(state.shape[0], state.shape[1], state.shape[2])
        if flux_mode == "left_energy_closure":
            q_physical = left_energy_interface_flux_closure(
                state,
                smooth,
                closure_geom,
                q_left_integral,
                resistance=float(resistance),
                sigma=float(sigma),
            )
        elif flux_mode == "two_sided_energy_closure":
            q_physical = two_sided_energy_interface_flux_closure(
                state,
                smooth,
                closure_geom,
                q_left_integral,
                resistance=float(resistance),
                right_value=float(model.t_right_tilde.detach().cpu()),
                q_ref=float(q_ref),
                sigma=float(sigma),
            )
        else:
            q_physical, smooth, correction, implied_flux_gap = (
                conservative_energy_storage_projection(
                    state,
                    smooth,
                    closure_geom,
                    q_left_integral,
                    resistance=float(resistance),
                    right_value=float(model.t_right_tilde.detach().cpu()),
                    sigma=float(sigma),
                )
            )
        q_normalized = q_physical[:, None].expand(-1, state.shape[2]) / float(q_ref)
        aligned_flux = q_normalized[:, None].expand_as(smooth)
        enriched = moving_interface_jump_enrichment(
            smooth.reshape(state.shape[0], -1, 1),
            coords,
            aligned_flux.reshape(state.shape[0], -1, 1),
            interface_x,
            jump_scale,
        )[..., 0].view_as(smooth)
        if return_storage_projection:
            return enriched, q_normalized, correction, implied_flux_gap
        if return_interface_flux:
            return enriched, q_normalized
        return enriched
    outputs = []
    fluxes = []
    chunk = int(query_chunk)
    for start in range(0, coords.shape[1], chunk):
        stop = min(start + chunk, coords.shape[1])
        output, flux = model.decode_enriched(
            latent,
            coords[:, start:stop],
            time_query[:, start:stop],
            interface_x,
            jump_scale,
        )
        outputs.append(output)
        fluxes.append(flux)
    prediction = torch.cat(outputs, dim=1)[..., 0].view(
        state.shape[0], state.shape[1], state.shape[2]
    )
    if not return_interface_flux:
        return prediction
    flux_grid = torch.cat(fluxes, dim=1)[..., 0].view_as(prediction)
    if return_storage_projection:
        zeros = prediction.new_zeros(prediction.shape)
        gap = prediction.new_zeros((prediction.shape[0],))
        return prediction, flux_grid[:, 0], zeros, gap
    return prediction, flux_grid[:, 0]


def rollout_field(
    model,
    initial_state: torch.Tensor,
    interval_images: torch.Tensor,
    scalars: torch.Tensor,
    fixed_material_channels: torch.Tensor,
    coords: torch.Tensor,
    *,
    dt: float,
    query_chunk: int,
    steps: int,
    interface_x: torch.Tensor | None = None,
    jump_scale: torch.Tensor | None = None,
    closure_geom=None,
    q_left_integrals: torch.Tensor | None = None,
    resistance: float | None = None,
    sigma: float | None = None,
    q_ref: float | None = None,
    return_interface_flux: bool = False,
    return_storage_projection: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Autoregressive free rollout from `initial_state` over `steps` intervals.

    Runs under `torch.no_grad()` in eval mode; each step feeds the previous
    detached prediction back as the next input state. Returns the stacked
    trajectory `(steps + 1, Nx, Ny)` and, when requested, the per-step interface
    flux / storage-projection diagnostics (plan Sections 2, 5).
    """
    was_training = model.training
    model.eval()
    states = [initial_state.detach()]
    fluxes = []
    corrections = []
    implied_flux_gaps = []
    with torch.no_grad():
        for n in range(int(steps)):
            result = predict_one_step_field(
                model,
                states[-1],
                interval_images[n : n + 1],
                scalars,
                fixed_material_channels,
                coords,
                dt=dt,
                query_chunk=query_chunk,
                interface_x=interface_x,
                jump_scale=jump_scale,
                closure_geom=closure_geom,
                q_left_integral=(
                    q_left_integrals[n : n + 1]
                    if q_left_integrals is not None else None
                ),
                resistance=resistance,
                sigma=sigma,
                q_ref=q_ref,
                return_interface_flux=return_interface_flux,
                return_storage_projection=return_storage_projection,
            )
            if return_storage_projection:
                prediction, flux, correction, implied_flux_gap = result
                fluxes.append(flux.detach())
                corrections.append(correction.detach())
                implied_flux_gaps.append(implied_flux_gap.detach())
            elif return_interface_flux:
                prediction, flux = result
                fluxes.append(flux.detach())
            else:
                prediction = result
            states.append(prediction.detach())
    model.train(was_training)
    rollout = torch.cat(states, dim=0)
    if return_storage_projection:
        return (
            rollout,
            torch.cat(fluxes, dim=0),
            torch.cat(corrections, dim=0),
            torch.cat(implied_flux_gaps, dim=0),
        )
    if return_interface_flux:
        return rollout, torch.cat(fluxes, dim=0)
    return rollout


def build_interval_forcing_images(
    params: dict,
    y_image: np.ndarray,
    time_grid: np.ndarray,
    *,
    t_ramp: float,
    q_ref: float,
    image_time_points: int,
) -> np.ndarray:
    """CN-averaged normalized left-flux image per interval, ``(Nt-1, 1, ny, nt)``.

    For each interval ``[t_n, t_{n+1}]`` the exact time-integral of the separable
    left flux is divided by ``dt * q_ref`` to yield the CN-average, then broadcast
    across the ``image_time_points`` temporal axis (the encoder's forcing image is
    constant in local time for a single CN interval).
    """
    images = []
    for n in range(len(time_grid) - 1):
        dt = float(time_grid[n + 1] - time_grid[n])
        _, _, q_integral = build_interface_forcing(
            params["temporal_family"],
            params["temporal_params"],
            params["spatial_family"],
            params["spatial_params"],
            y_image,
            float(time_grid[n]),
            float(time_grid[n + 1]),
            float(t_ramp),
        )
        average = q_integral / (dt * float(q_ref))
        images.append(np.broadcast_to(
            average[:, None], (len(y_image), int(image_time_points))
        ))
    return np.asarray(images, dtype=np.float32)[:, None]


def material_channels(
    right_state: float,
    interface_x: float,
    x_grid: np.ndarray,
    ny: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Fixed interface spatial channels for a constant-state baseline.

    Drops the leading temperature channel (index 0) so only the geometry-derived
    interface channels (signed distance, side indicator, material) remain; these
    are constant per case (they depend on ``interface_x``) and are re-stacked with
    the live state in ``state_conditioned_spatial``.
    """
    baseline = np.full((len(x_grid), ny), float(right_state), dtype=np.float32)
    channels = _interface_spatial_channels_from_normalized(
        baseline, float(interface_x), x_grid
    )[1:]
    return torch.from_numpy(channels).unsqueeze(0).to(device)


@dataclass
class OneStepCase:
    """A single online physics descriptor with its precomputed FV manifold.

    ``truth_states`` is the normalized FV trajectory ``(Nt, Nx, Ny)`` held on CPU
    (plan Section 9 memory preflight); per-step slices are moved to the GPU by the
    runner. All small per-case tensors (``cn``, ``geom``, ``scalars``,
    ``interface_x``, ``jump_scale``, ``q_left_integrals``, ``fixed_channels``)
    already live on the compute device. ``used_intervals`` tracks which one-step
    intervals have been drawn so a reuse always picks a fresh interval.
    """

    key: int
    params: dict
    n_intervals: int
    truth_states: torch.Tensor
    interval_images: torch.Tensor
    scalars: torch.Tensor
    interface_x: torch.Tensor
    jump_scale: torch.Tensor
    cn: dict
    geom: object
    q_left_integrals: torch.Tensor
    resistance: float
    fixed_channels: torch.Tensor
    forcing_norm: np.ndarray
    storage_change: np.ndarray
    interface_jump: np.ndarray
    uses: int = 0
    age: int = 0
    used_intervals: set = field(default_factory=set)


def prepare_interfaces_one_step_case(
    key: int,
    params: dict,
    solver_factory,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mu: float,
    sigma: float,
    t_ramp: float,
    q_ref: float,
    k_left: float,
    k_right: float,
    ny_image: int,
    image_time_points: int,
    right_value: float,
    device: torch.device,
) -> OneStepCase:
    """Solve the FV trajectory for one interfaces descriptor and pack a case.

    ``solver_factory(params) -> FVSolver2D`` builds the per-descriptor solver
    (varying ``interface_x`` / ``R_c`` / IC). The FV states are the on-manifold
    conditioning inputs; ``T_{n+1}`` is never used as a supervised label. Interval
    forcing images, CN coefficients, per-interface geometry, per-interval left-flux
    integrals, and the fixed interface channels are all precomputed here so the
    per-update loop only moves the selected state to the device and runs the
    forward.
    """
    solver = solver_factory(params)
    _, _, _, truth_K = solver.solve(
        np.asarray(params["T0"], dtype=np.float64), store_trajectory=True
    )
    truth_K = np.asarray(truth_K, dtype=np.float64)
    time_grid = np.asarray(solver.t, dtype=np.float64)
    n_intervals = len(time_grid) - 1

    normalized = ((truth_K - float(mu)) / (float(sigma) + 1e-8)).astype(np.float32)
    truth_states = torch.from_numpy(normalized)

    y_image = np.linspace(float(y_grid[0]), float(y_grid[-1]), int(ny_image))
    interval_images = torch.from_numpy(build_interval_forcing_images(
        params,
        y_image,
        time_grid,
        t_ramp=t_ramp,
        q_ref=q_ref,
        image_time_points=int(image_time_points),
    )).to(device)

    interface_x_value = float(params["interface_x"])
    resistance = float(params["R_c"])
    scalars = torch.from_numpy(normalize_interface_scalars(
        interface_x_value, resistance
    )).unsqueeze(0).to(device)
    interface_x = torch.tensor(
        [interface_x_value], device=device, dtype=torch.float32
    )
    jump_scale = torch.tensor(
        [resistance * float(q_ref) / (float(sigma) + 1e-8)],
        device=device,
        dtype=torch.float32,
    )
    cn = build_cn_tensors(solver, sigma=float(sigma), device=device, dtype=torch.float32)
    geom = build_cn_geom_per_interface(
        x_grid,
        y_grid,
        float(k_left),
        float(k_right),
        [interface_x_value],
        [resistance],
        float(solver.dt),
        sigma_global=float(sigma),
        rho=1.0,
        cp=1.0,
        device=device,
        dtype=torch.float32,
    )

    q_left_integral_fn, _ = build_qL_integral(
        params["temporal_family"],
        params["temporal_params"],
        params["spatial_family"],
        params["spatial_params"],
        y_grid,
        t_ramp=float(t_ramp),
    )
    q_left_integrals = []
    for step in range(n_intervals):
        integral = np.asarray(
            q_left_integral_fn(time_grid[step], time_grid[step + 1]),
            dtype=np.float64,
        )
        if integral.ndim == 0:
            integral = np.full(solver.Ny, float(integral))
        q_left_integrals.append(integral)
    q_left_integrals = torch.from_numpy(
        np.stack(q_left_integrals).astype(np.float32)
    ).to(device)

    fixed_channels = material_channels(
        float(right_value), interface_x_value, x_grid, solver.Ny, device=device
    )

    interface_face = int(geom.face_idx.reshape(-1)[0].item())
    forcing_norm = np.linalg.norm(
        np.stack(
            [np.asarray(q, dtype=np.float64) for q in q_left_integrals.cpu().numpy()]
        ).reshape(n_intervals, -1),
        axis=1,
    ).astype(np.float32)
    storage_change = np.abs(
        normalized[1:] - normalized[:-1]
    ).reshape(n_intervals, -1).mean(axis=1).astype(np.float32)
    jump = np.abs(
        normalized[1:, interface_face] - normalized[1:, interface_face + 1]
    ).reshape(n_intervals, -1).mean(axis=1).astype(np.float32)

    return OneStepCase(
        key=int(key),
        params=params,
        n_intervals=int(n_intervals),
        truth_states=truth_states,
        interval_images=interval_images,
        scalars=scalars,
        interface_x=interface_x,
        jump_scale=jump_scale,
        cn=cn,
        geom=geom,
        q_left_integrals=q_left_integrals,
        resistance=resistance,
        fixed_channels=fixed_channels,
        forcing_norm=forcing_norm,
        storage_change=storage_change,
        interface_jump=jump,
    )


class OneStepStatePool:
    """Continuously-refreshed, bounded-reuse pool of online physics cases.

    Holds ``pool_size`` :class:`OneStepCase` objects. Each ``refresh`` ages the
    pool, evicts cases that are exhausted (``uses >= max_uses_per_case`` or all
    intervals drawn), and retires the oldest cases so at least
    ``replace_per_update`` fresh descriptors enter per update. ``select`` returns
    ``n_cases`` distinct cases, each paired with a not-yet-used interval chosen by
    the time-index mixture (plan Section 4a); every non-uniform component falls
    back to uniform when it has no usable weight.
    """

    def __init__(
        self,
        case_factory,
        *,
        pool_size: int,
        n_cases: int,
        replace_per_update: int,
        max_uses_per_case: int,
        time_mixture: dict,
        rng: np.random.Generator,
    ) -> None:
        if int(n_cases) > int(pool_size):
            raise ValueError("n_cases cannot exceed pool_size")
        self._case_factory = case_factory
        self._pool_size = int(pool_size)
        self._n_cases = int(n_cases)
        self._replace_per_update = int(replace_per_update)
        self._max_uses_per_case = int(max_uses_per_case)
        self._rng = rng
        components = ("uniform", "active_forcing", "high_change")
        weights = np.array(
            [float(time_mixture.get(name, 0.0)) for name in components],
            dtype=np.float64,
        )
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            total = 1.0
        self._components = components
        self._mixture = weights / total
        self._next_key = 0
        self._cases: list[OneStepCase] = []
        self._fill()

    def _make_case(self) -> OneStepCase:
        case = self._case_factory(self._next_key)
        self._next_key += 1
        return case

    def _fill(self) -> None:
        while len(self._cases) < self._pool_size:
            self._cases.append(self._make_case())

    def _exhausted(self, case: OneStepCase) -> bool:
        return (
            case.uses >= self._max_uses_per_case
            or len(case.used_intervals) >= case.n_intervals
        )

    def refresh(self) -> None:
        for case in self._cases:
            case.age += 1
        self._cases = [c for c in self._cases if not self._exhausted(c)]
        n_evicted = self._pool_size - len(self._cases)
        target = min(self._replace_per_update, self._pool_size)
        n_extra = max(0, target - n_evicted)
        if n_extra > 0 and self._cases:
            self._cases.sort(key=lambda c: c.age, reverse=True)
            del self._cases[: min(n_extra, len(self._cases))]
        self._fill()

    def _select_interval(self, case: OneStepCase) -> int:
        available = [
            i for i in range(case.n_intervals) if i not in case.used_intervals
        ]
        if not available:
            available = list(range(case.n_intervals))
        component = self._components[
            int(self._rng.choice(len(self._components), p=self._mixture))
        ]
        if component == "uniform":
            return int(available[int(self._rng.integers(len(available)))])
        if component == "active_forcing":
            scores = case.forcing_norm[available]
        else:
            scores = case.storage_change[available] + case.interface_jump[available]
        scores = np.asarray(scores, dtype=np.float64)
        total = float(scores.sum())
        if not np.isfinite(total) or total <= 0.0:
            return int(available[int(self._rng.integers(len(available)))])
        probabilities = scores / total
        return int(available[int(self._rng.choice(len(available), p=probabilities))])

    def select(self) -> list[tuple[OneStepCase, int]]:
        if self._n_cases > len(self._cases):
            raise ValueError("pool has fewer cases than n_cases")
        chosen = self._rng.choice(len(self._cases), size=self._n_cases, replace=False)
        selection = []
        for index in chosen:
            case = self._cases[int(index)]
            interval = self._select_interval(case)
            case.used_intervals.add(interval)
            case.uses += 1
            selection.append((case, interval))
        return selection

    @property
    def cases(self) -> list[OneStepCase]:
        return list(self._cases)

    def diversity_stats(self) -> dict[str, float]:
        keys = [case.key for case in self._cases]
        return {
            "pool_distinct_keys": float(len(set(keys))),
            "pool_size": float(len(self._cases)),
            "pool_mean_uses": float(
                np.mean([case.uses for case in self._cases]) if self._cases else 0.0
            ),
            "pool_max_age": float(
                np.max([case.age for case in self._cases]) if self._cases else 0.0
            ),
        }

    def state_dict(self) -> dict:
        """Serialize the pool WITHOUT the heavy FV trajectories (plan Section 9).

        Stores the selection RNG state, the descriptor-key cursor, and per-case
        descriptors + reuse bookkeeping (``key``, ``params``, ``uses``, ``age``,
        ``used_intervals``) in pool order. On resume the FV manifolds are
        regenerated deterministically from ``params`` via ``load_state_dict``'s
        rebuild callable, so gigabyte trajectory tensors never enter a checkpoint.
        """
        return {
            "rng": self._rng.bit_generator.state,
            "next_key": int(self._next_key),
            "cases": [
                {
                    "key": int(c.key),
                    "params": c.params,
                    "uses": int(c.uses),
                    "age": int(c.age),
                    "used_intervals": sorted(int(i) for i in c.used_intervals),
                }
                for c in self._cases
            ],
        }

    def load_state_dict(self, state: dict, rebuild_fn) -> None:
        """Restore pool order + RNG and rebuild every case from its descriptor.

        ``rebuild_fn(key, params) -> OneStepCase`` re-solves the FV manifold from
        the stored descriptor (deterministic given ``params``); reuse counters and
        the drawn-interval set are then restored so the next ``refresh``/``select``
        matches an uninterrupted run.
        """
        self._rng.bit_generator.state = state["rng"]
        self._next_key = int(state["next_key"])
        self._cases = []
        for cs in state["cases"]:
            case = rebuild_fn(int(cs["key"]), cs["params"])
            case.uses = int(cs["uses"])
            case.age = int(cs["age"])
            case.used_intervals = set(int(i) for i in cs["used_intervals"])
            self._cases.append(case)
