from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.physics.fv_solver_2d import FVSolver2D


@dataclass(frozen=True)
class ProblemDims:
    """Per-benchmark tensor contract.

    `in_channels` and `cond_static_dim` size the model's lift and conditioning
    MLP. `has_forcing_seq` / `use_temporal_encoder` toggle the temporal branch
    (source has neither). `t_stats_dim` is the width of the denormalization
    stats vector (2 = [mu, sigma]; 3 adds a per-benchmark scalar such as
    interface_x).
    """

    in_channels: int
    cond_static_dim: int
    has_forcing_seq: bool
    temporal_token_dim: int
    t_stats_dim: int
    use_temporal_encoder: bool


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
