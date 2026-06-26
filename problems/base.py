from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.physics.fv_solver_2d import FVSolver2D


def empty_forcing_seq() -> np.ndarray:
    """The placeholder ``forcing_seq`` for bins-mode items.

    Bins-mode benchmarks carry no temporal token stream, but the dataset item
    must keep a fixed set of keys so default collation does not break. They emit
    this explicit empty ``(0, 0)`` tensor (collated to ``(B, 0, 0)``); ``FNO2d``
    ignores ``forcing_seq`` whenever ``use_temporal_encoder`` is off.
    """
    return np.zeros((0, 0), dtype=np.float32)


@dataclass(frozen=True)
class ProblemDims:
    """Per-benchmark tensor contract.

    `in_channels` and `cond_static_dim` size the model's lift and conditioning
    MLP. `has_forcing_seq` / `use_temporal_encoder` toggle the temporal branch
    (bins mode has neither). `t_stats_dim` is the width of the denormalization
    stats vector (2 = [mu, sigma]; 3 adds a per-benchmark scalar such as
    interface_x). `s_y_channel` is the spatial channel the model multiplies the
    learned forcing weights against; `use_forcing_time_aug` augments the temporal
    embedding with the lead/start time before projecting those weights.
    """

    in_channels: int
    cond_static_dim: int
    has_forcing_seq: bool
    temporal_token_dim: int
    t_stats_dim: int
    use_temporal_encoder: bool
    s_y_channel: int = 3
    use_forcing_time_aug: bool = False


class ProblemSpec(ABC):
    """Adapter that makes one benchmark pluggable behind a fixed interface.

    The dataset, model, training/eval loops, and the data generator all route
    benchmark-specific behavior through this object so they never branch on the
    benchmark name themselves.
    """

    name: str
    dims: ProblemDims

    # ---- data generation ----

    @abstractmethod
    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        """Draw the per-simulation parameter dicts written to sim_params.npy.

        `rng` and `rng_profile` are two independent streams (IC vs forcing) so
        the call order matches the original per-benchmark generator exactly.
        """

    @abstractmethod
    def configure_solver(self, params: dict, base_kwargs: dict) -> "FVSolver2D":
        """Build the solver for one simulation from its sampled params.

        `base_kwargs` carries the fixed geometry/time shared by every sim
        (domain, grid sizes, dt, t_final, window) plus `y_grid`; the adapter
        adds the benchmark-specific wiring (`q_left_fn`, `source`, `layers`,
        `interface_R`).
        """

    # ---- dataset item ----

    def setup_dataset(self, ds) -> None:
        """Populate benchmark-specific caches on a `SnapshotPairDataset`.

        Called once at the end of `SnapshotPairDataset.__init__`. The default
        is a no-op; benchmarks override to precompute per-sim spatial profiles,
        normalization references, or lazy callable caches that `build_item`
        reads. `ds` is duck-typed (grids, sim_ids, sim_params, t_final,
        temporal_samples already set).
        """

    @abstractmethod
    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        """Construct one (source -> target) training item as a numpy dict.

        Keys: "spatial", "cond_static", "Y", "T_stats", and "forcing_seq" when
        `dims.has_forcing_seq`. The dataset wraps the arrays in tensors. `ds` is
        the owning `SnapshotPairDataset` (duck-typed); the adapter reads its
        grids, trajectories, normalization stats, and caches.
        """

    @abstractmethod
    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        """Raise if any referenced sim_params entry lacks required keys."""

    # ---- optional hooks (sane defaults) ----

    #: Extra per-pair conditioning columns this benchmark contributes to
    #: ``val_pairs.csv`` (on top of the universal ``epoch, sim_id, benchmark,
    #: t_s, t_bar, R_c, rel_l2, iface_rel_l2``). The training writer takes the
    #: union across all registered benchmarks as the CSV header and leaves
    #: columns a benchmark does not populate empty.
    val_pair_fields: tuple[str, ...] = ()

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        """Return this benchmark's extra val-pair columns for one (s -> j) pair.

        Keys must be a subset of ``val_pair_fields``. The default is empty; the
        universal columns are filled by the training loop. `ds` is the owning
        `SnapshotPairDataset` (duck-typed: grids, sim_params available).
        """
        return {}

    def diagnostics(self, model, batch: dict, ctx: dict) -> dict:
        """Benchmark-specific probes run during training diagnostics."""
        return {}

    def plot_label(self, params: dict) -> str:
        """Short human-readable label for a sim, used by visual modules."""
        return self.name

    # ---- optional collocation hooks (W2 physics path; default off) ----

    def collocation_geom_cfg(
        self, ds, phys_cfg: dict, mu_global: float, sigma_global: float, dt: float,
    ) -> dict[str, Any] | None:
        """Geometry/closure config for the full_bc collocation residual.

        Return a dict consumed by the training loop's collocation sampler
        (``x_grid``, ``y_grid``, ``k_left``, ``k_right``, ``interface_x``,
        ``dt``, ``sigma_global``, ``T_right_tilde``). The default ``None`` makes
        the training loop fall back to its built-in (forcing) geometry block, so
        benchmarks that do not opt in keep the existing behavior unchanged.
        """
        return None

    def collocation_closure(
        self, ds, sid: int, params: dict, t: float, t_dt: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
        """Per-item boundary closure for a collocation pair.

        Return ``(R_c, qL_n, qL_np1, qL_int)`` for the two collocation times, or
        ``None`` to use the training loop's built-in (forcing) closure.
        """
        return None

    def collocation_base_plan(self, ds, dt: float) -> dict[str, Any] | None:
        """Plan describing how collocation pairs are drawn.

        Return e.g. ``{"base_snapshot_index": 0, "on_grid_pairs": True}`` to pin
        the source snapshot and draw consecutive on-grid conditioning times, or
        ``None`` to use the training loop's built-in (random source / uniform
        lead) sampling.
        """
        return None
