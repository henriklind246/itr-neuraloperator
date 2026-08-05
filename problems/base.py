from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

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
    embedding with lead time before projecting those weights.
    """

    in_channels: int
    cond_static_dim: int
    has_forcing_seq: bool
    temporal_token_dim: int
    t_stats_dim: int
    use_temporal_encoder: bool
    s_y_channel: int = 3
    use_forcing_time_aug: bool = False


OODKind = Literal["simulation_parameter", "evaluation_parameter", "compound"]


#: Train/eval ablation of the global spatial descriptor in ``cond_static``.
#:
#: The spatial forcing profile ``s(y)`` (or, for source benchmarks, the internal
#: source patch geometry) is visible to the model through two redundant paths:
#: the spatial input channel it multiplies against, and the global
#: ``cond_static`` descriptor (a one-hot family label plus continuous params for
#: forcing/forcing_itr; patch-geometry params for source/source_itr). These
#: modes zero out the descriptor entries in ``cond_static`` while leaving the
#: spatial channel intact, so the model must recover the profile from the field
#: alone. ``cond_static_dim`` is never changed, so the ConditioningMLP
#: architecture and parameter count are identical across modes.
#:
#: - ``full`` -- baseline; nothing zeroed.
#: - ``no_family`` -- zero only the one-hot family block (retain continuous
#:   params). A no-op for benchmarks with no family label (source/source_itr).
#: - ``spatial_field_only`` -- zero the whole spatial descriptor (family label +
#:   params, or patch params).
SpatialConditioningMode = Literal["full", "no_family", "spatial_field_only"]


@dataclass(frozen=True)
class OODAxis:
    """Declares one out-of-distribution sweep axis for a benchmark.

    An axis names a single controlled shift away from the training distribution.
    ``kind`` decides whether a swept value needs a new simulation or only a new
    evaluation query on an existing trajectory:

    - ``simulation_parameter``: one sim per ``(axis, value, repeat)`` cell; the
      value enters ``apply_ood_value`` and overrides a single sim-param field.
    - ``evaluation_parameter``: the value is a *target time*, not a sim param.
      One extended trajectory is generated per repeat (solved to the max target)
      and the eval stage expands it into target-time buckets. ``field`` is None.
    - ``compound``: one sim per cell, but more than one physical factor moves
      together (``family_transfer``, ``rc_severity``); CRN pairing is verified by
      the shared-latent fingerprint, not by byte-equal params.

    ``id_reference`` leads the value list so every sweep includes an exactly
    paired in-distribution anchor produced inside the same CRN sweep.
    ``dataset_t_final`` is the solver/trajectory horizon to solve to (0.30
    normally; 0.45 for the time axis). ``time_norm_horizon`` is the trained
    normalization horizon the model-input builders scale temporal features by
    (0.30), so a target past it yields lead/time features > 1.0 (the headline
    time-OOD signal). ``trained_range`` is the family's saved sampler range used
    to classify each realized value's 4-level ``distribution_class``;
    ``pinned_family`` is the temporal/spatial family this axis fixes so the swept
    field is meaningful. ``resolution_kind`` tags how the swept feature stresses
    the FV discretization (Step 6 resolution accounting).
    """

    name: str
    kind: OODKind
    id_reference: tuple[Any, ...]
    ood_values: tuple[Any, ...]
    field: str | None = None
    pinned_family: str | None = None
    trained_range: tuple[float, float] | None = None
    dataset_t_final: float = 0.30
    time_norm_horizon: float = 0.30
    resolution_kind: str = "none"
    compound: bool = False
    notes: str = ""

    def sweep_values(self) -> tuple[Any, ...]:
        """Full ordered value list: ID reference value(s) first, then OOD values.

        Sims are generated in this order (``repeat``-major) so a given repeat's
        shared-latent background is reused across every value, and the leading
        in-distribution reference gives an exactly-paired baseline.
        """
        return tuple(self.id_reference) + tuple(self.ood_values)


class ProblemSpec(ABC):
    """Adapter that makes one benchmark pluggable behind a fixed interface.

    The dataset, model, training/eval loops, and the data generator all route
    benchmark-specific behavior through this object so they never branch on the
    benchmark name themselves.
    """

    name: str
    dims: ProblemDims

    # ---- spatial-descriptor conditioning ablation ----
    #: Active ablation mode (see :data:`SpatialConditioningMode`). Set via
    #: :meth:`set_spatial_conditioning`; ``full`` leaves ``cond_static`` untouched.
    spatial_conditioning: str = "full"
    #: ``cond_static`` slice holding the one-hot family label, zeroed under
    #: ``no_family`` and ``spatial_field_only``. ``None`` for benchmarks with no
    #: family label (an empty mask). Declared next to each spec's cond layout as
    #: the single source of truth read by both the mask helper and the tests.
    family_cond_slice: "slice | None" = None
    #: ``cond_static`` slice holding the full spatial descriptor (family label +
    #: continuous params, or patch-geometry params), zeroed under
    #: ``spatial_field_only``. ``None`` for benchmarks with no descriptor.
    spatial_descriptor_cond_slice: "slice | None" = None

    def set_spatial_conditioning(self, mode: str) -> None:
        """Validate and store the spatial-descriptor conditioning mode.

        Raises immediately on an unknown mode so a bad config value fails at
        spec-resolution time rather than surviving into item construction.
        """
        valid = get_args(SpatialConditioningMode)
        if mode not in valid:
            raise ValueError(
                f"Unknown spatial_conditioning={mode!r}; "
                f"expected one of {valid}."
            )
        self.spatial_conditioning = mode

    def _apply_spatial_conditioning_mask(self, cond: np.ndarray) -> np.ndarray:
        """Zero the declared descriptor slice(s) for the active ablation mode.

        ``full`` returns ``cond`` unchanged. Ablated modes return a *copied*
        array (never aliasing a reused dict or shared-memory tensor) with the
        selected slice zeroed; a ``None`` slice is a no-op copy, so benchmarks
        without the relevant descriptor (e.g. ``no_family`` on source, or any
        mode on interfaces) are guaranteed no-ops.
        """
        mode = self.spatial_conditioning
        if mode == "full":
            return cond
        out = np.asarray(cond).copy()
        if mode == "no_family":
            sl = self.family_cond_slice
        elif mode == "spatial_field_only":
            sl = self.spatial_descriptor_cond_slice
        else:
            raise ValueError(
                f"Unknown spatial_conditioning={mode!r}; "
                f"expected one of {get_args(SpatialConditioningMode)}."
            )
        if sl is not None:
            out[sl] = 0.0
        return out

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

    # ---- optional OOD hooks (W3 out-of-distribution path; default off) ----

    def ood_axes(self) -> dict[str, OODAxis]:
        """Out-of-distribution sweep axes this benchmark supports.

        Default is empty so non-OOD benchmarks and the standard generation/eval
        paths are untouched. Benchmarks that opt in return a mapping from axis
        name to its :class:`OODAxis` and implement :meth:`draw_latents` and
        :meth:`apply_ood_value`.
        """
        return {}

    def draw_latents(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
        axis: OODAxis,
    ) -> dict[str, Any]:
        """Draw ONE shared-latent bundle for a single CRN repeat of ``axis``.

        Called once per ``repeat`` by the OOD generator (seeded by the repeat
        index). The returned bundle holds the *background* random draw shared by
        every swept value at this repeat: unit quantiles / standard normals for
        the parameters an axis maps through a family's inverse CDF, plus any
        non-swept physical fields (IC, ``T0``, the non-swept family draws) that
        every value reuses verbatim. It must NOT bake in the swept field --
        :meth:`apply_ood_value` injects that from the explicit value.

        Storing quantiles (not realized params) lets compound axes map the SAME
        background through different families' inverse CDFs, giving reproducible
        CRN coupling across otherwise-unrelated parameterizations.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares OOD axes but does not implement "
            "draw_latents()."
        )

    def apply_ood_value(
        self, latents: dict[str, Any], axis: OODAxis, value: Any,
    ) -> dict[str, Any]:
        """Map a shared-latent bundle + one swept ``value`` into a sim-param dict.

        Returns a single ``sim_params`` entry (same schema as
        :meth:`sample_sim_params` rows, passing :meth:`validate_schema` and
        :meth:`configure_solver`). The ``latents`` bundle is reused for every
        value of a repeat, so for ordinary axes every non-swept field is
        byte-identical across values and only the swept field changes; for
        compound axes the shared quantiles map through each value's families.

        When snapping/clipping/truncation can change another quantity (interface
        face-alignment, patch clipping at a boundary, void-profile truncation),
        the returned dict also records the realized geometry under
        ``*_actual``/``realized_*`` keys so the generator's sidecar can audit
        requested-vs-realized values.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares OOD axes but does not implement "
            "apply_ood_value()."
        )

    def resolution_scale(self, axis: OODAxis, params: dict) -> float | None:
        """Physical extent of the swept feature, for resolution accounting.

        Units follow ``axis.resolution_kind``: seconds for ``temporal_*`` (a
        sinusoid period, exponential timescale, or minimum pulse width) and
        domain length for ``spatial_*`` (a Gaussian sigma, triangle half-width,
        or patch extent). Read from the realized ``params`` so sampling/snapping
        is reflected. ``None`` (the default) when the axis does not stress the
        FV discretization, so the generator records no cells/steps for it.
        """
        return None
