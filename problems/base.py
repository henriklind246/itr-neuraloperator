from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

if TYPE_CHECKING:
    from src.physics.fv_solver_2d import FVSolver2D


@dataclass(frozen=True)
class ProblemDims:
    """Per-benchmark tensor contract.

    `in_channels` and `cond_static_dim` size the model's lift and conditioning
    MLP. `t_stats_dim` is the width of the denormalization stats vector
    (2 = [mu, sigma]; 3 adds a per-benchmark scalar such as interface_x).
    `s_y_channel` is the spatial channel the model multiplies the learned
    forcing weights against; `use_forcing_time_aug` augments the temporal
    embedding with lead time before projecting those weights.
    `forcing_extender_rc_cond_index` declares the normalized scalar-contact-
    resistance slot that an eligible boundary extender may add to its domain
    queries; ``None`` means the active benchmark does not expose that
    capability. `supports_diffusion_geometry_extender` is deliberately
    narrower: it marks the fixed-interface scalar-R_c contract supported by the
    first diffusion-inspired geometry-bias experiment.
    """

    in_channels: int
    cond_static_dim: int
    temporal_token_dim: int
    t_stats_dim: int
    s_y_channel: int = 3
    use_forcing_time_aug: bool = False
    forcing_extender_rc_cond_index: int | None = None
    supports_diffusion_geometry_extender: bool = False


OODKind = Literal["simulation_parameter", "evaluation_parameter", "compound"]


#: Train/eval ablation of the global spatial descriptor in ``cond_static``.
#:
#: For the source benchmarks the internal source patch geometry is visible to
#: the model through two redundant paths: the spatial input channels, and the
#: global ``cond_static`` patch-geometry params. ``spatial_field_only`` zeros the
#: descriptor entries in ``cond_static`` while leaving the spatial channels
#: intact, so the model must recover the geometry from the field alone.
#: ``cond_static_dim`` is never changed, so the ConditioningMLP architecture and
#: parameter count are identical across modes.
#:
#: - ``full`` -- baseline; nothing zeroed.
#: - ``spatial_field_only`` -- zero the whole spatial descriptor. A no-op for
#:   benchmarks with no descriptor (forcing family, interfaces).
SpatialConditioningMode = Literal["full", "spatial_field_only"]


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

    #: Only the scalar-R_c forcing benchmark supports duplicating normalized
    #: ITR and lead-time values as spatial fields. The discontinuous material-
    #: side field is geometry metadata and is available to every benchmark.
    supports_scalar_spatial_input: bool = False
    spatial_input_itr: bool = False
    spatial_input_lead_time: bool = False
    spatial_input_material_side: bool = False

    def configure_spatial_input(
        self,
        *,
        itr: bool,
        lead_time: bool,
        material_side: bool,
    ) -> None:
        """Configure optional spatial fields while preserving disabled defaults."""
        options = {
            "itr": itr,
            "lead_time": lead_time,
            "material_side": material_side,
        }
        for key, value in options.items():
            if not isinstance(value, bool):
                raise ValueError(
                    f"benchmark.spatial_input.{key} must be a boolean, got "
                    f"{value!r}."
                )
        if (itr or lead_time) and not self.supports_scalar_spatial_input:
            raise ValueError(
                "benchmark.spatial_input.itr and lead_time are supported only "
                "by the scalar-R_c forcing benchmark."
            )

        previous_extra = int(getattr(self, "spatial_input_itr", False)) + int(
            getattr(self, "spatial_input_lead_time", False)
        ) + int(getattr(self, "spatial_input_material_side", False))
        base_in_channels = self.dims.in_channels - previous_extra
        self.spatial_input_itr = itr
        self.spatial_input_lead_time = lead_time
        self.spatial_input_material_side = material_side
        self.dims = replace(
            self.dims,
            in_channels=(
                base_in_channels
                + int(self.spatial_input_itr)
                + int(self.spatial_input_lead_time)
                + int(self.spatial_input_material_side)
            ),
        )

    def _material_side_channel(self, ds, interface_x: float) -> np.ndarray:
        """Return the resolution-native indicator ``1[x >= interface_x]``."""
        side_x = (ds.x_grid >= float(interface_x)).astype(np.float32)
        return np.broadcast_to(side_x[:, None], (ds.Nx, ds.Ny))

    # ---- spatial-descriptor conditioning ablation ----
    #: Active ablation mode (see :data:`SpatialConditioningMode`). Set via
    #: :meth:`set_spatial_conditioning`; ``full`` leaves ``cond_static`` untouched.
    spatial_conditioning: str = "full"
    #: ``cond_static`` slice holding the spatial descriptor (patch-geometry
    #: params), zeroed under ``spatial_field_only``. ``None`` for benchmarks with
    #: no descriptor. Declared next to each spec's cond layout as the single
    #: source of truth read by both the mask helper and the tests.
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
        without a spatial descriptor (forcing family, interfaces) are guaranteed
        no-ops.
        """
        mode = self.spatial_conditioning
        if mode == "full":
            return cond
        if mode != "spatial_field_only":
            raise ValueError(
                f"Unknown spatial_conditioning={mode!r}; "
                f"expected one of {get_args(SpatialConditioningMode)}."
            )
        out = np.asarray(cond).copy()
        sl = self.spatial_descriptor_cond_slice
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

        Keys: "spatial", "cond_static", "forcing_seq", "Y", "T_stats". The
        dataset wraps the arrays in tensors. `ds` is the owning
        `SnapshotPairDataset` (duck-typed); the adapter reads its grids,
        trajectories, normalization stats, and caches.
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

        Storing quantiles (not realized params) lets compound axes that share a
        parameter space (e.g. ``rc_severity``: a shared peak fraction) map the
        SAME background through each value's parameterization, giving strict CRN
        coupling. Compound axes that span *incommensurable* family
        parameterizations (``family_transfer``: sin vs exp temporal, uniform vs
        patch spatial, which share no common latent) instead couple the swept
        profile through a single shared profile seed re-run per family. That
        pairing is reproducible and holds the non-family background byte-equal,
        but is a shared-fingerprint coupling, not a strict shared-quantile
        mapping of the profile latents.
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
        byte-identical across values and only the swept field changes;
        parameter-sharing compound axes (``rc_severity``) map the shared
        quantiles through each value's families, while incommensurable-family
        compound axes (``family_transfer``) hold the non-family background
        byte-equal and re-run each family's sampler from a shared profile seed.

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
