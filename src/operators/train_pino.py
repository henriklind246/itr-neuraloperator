from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import os
import platform
import random
import struct
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from data.dataset import (
    assert_dataset_problem_version,
    compute_global_stats,
    load_dataset_meta,
    load_ramp_seconds,
    load_sim_data,
    load_solver_dt,
    problem_from_config,
    split_sim_ids,
    split_sim_ids_stratified,
)
from problems.diffusion import T_RIGHT
from problems.diffusion_forcing import K_SLAB
from problems.forcing import (
    A_AMP_REF,
    INTERFACE_X as FORCING_INTERFACE_X,
    RC_RANGE as FORCING_RC_RANGE,
    normalize_interface_scalars as normalize_forcing_interface_scalars,
)
from problems.interfaces import (
    INTERFACE_X_RANGE,
    K_LEFT,
    K_RIGHT,
    RC_RANGE,
    normalize_interface_scalars,
)
from src.operators.cvit import (
    CViT,
    CViTEncoder,
    ForcingCViT,
    ForcingICCViT,
    ForcingTransitionCViT,
    InterfaceCViT,
    TransitionEncoding,
)
from src.operators.losses import full_bc_physics_loss, region_balanced_fv_rate_loss
from src.operators.train import (
    GradNormBalancer,
    _advance_scheduler,
    build_optimizer,
    build_scheduler,
    load_config,
    set_seed,
)
from src.operators.utils import resolve_device
from src.physics.boundary_forcing import (
    A_REF_FLUX,
    FORCING_SCHEMA_VERSION,
    RAMP_SCHEMA_VERSION,
    SPATIAL_SAMPLERS,
    TEMPORAL_SAMPLERS,
    build_interface_forcing,
    default_ramp_seconds,
    reconstruct_qL,
    sample_spatial_family,
    sample_temporal_family,
)
from src.physics.fv_residual import (
    FullBCData,
    block_energy_residual,
    build_cn_geom_batched,
    build_cn_geom_per_interface,
    build_homogeneous_cn_geom,
    interface_residual_rate,
    interface_trace_constraint_residuals,
    locate_interface,
)
from src.physics.init_conditions import (
    IC_BUILDER_SCHEMA_VERSION,
    IC_FAMILIES,
    ONLINE_IC_SAMPLER_VERSION,
    balanced_ic_family_assignments,
    build_ic,
)
from src.physics.pde_residual import (
    diffusion_residual,
    forcing_neumann_residual,
    ic_residual,
    neumann_residual,
)
from src.physics.one_step_objective import (
    build_cn_tensors_from_geom,
    defect_terms,
    explicit_cn_rhs,
    implicit_cn_action,
    one_step_objective,
    variational_objective,
)

# ---- Continuous ViT training on the diffusion benchmark suite ----------------
#
# Self-contained trainer: it reuses the data loaders, optimizer/scheduler
# builders, and seeding from the FNO path by import, but never constructs an FNO
# or a SnapshotPairDataset. Most runners are physics-only; the versioned
# diffusion_forcing_single benchmark can instead use saved trajectory values
# through its explicitly isolated supervised runner.

WALLS = ("left", "top", "bottom")

# Architecture-version stamp for the one-step interfaces PINO run. Bump when the
# encoder/decoder contract of the production InterfaceCViT changes so recorded
# run metadata and the SLURM preflight can reject a mismatched checkpoint.
ONE_STEP_ARCH_VERSION = 3

_ONLINE_NUMPY_STREAMS = (
    "ic_family",
    "ic_params",
    "forcing_family",
    "forcing_params",
    "interface_position",
    "contact_resistance",
    "other_materials",
    "fv_intervals",
    "evaluation",
)
_CANONICAL_EXCLUDED_KEYS = {
    "T0", "T0_sha256", "problem_key", "batch_key", "sample_key",
    "device", "runtime", "latent", "decoded",
}


@dataclass
class OnlineSamplerRNGs:
    """Independent online-sampling streams with checkpointable state."""

    numpy: dict[str, np.random.Generator]
    autodiff_collocation: torch.Generator
    evaluation: torch.Generator

    @classmethod
    def create(cls, seed: int, device: torch.device) -> "OnlineSamplerRNGs":
        children = np.random.SeedSequence(int(seed)).spawn(
            len(_ONLINE_NUMPY_STREAMS) + 2
        )
        numpy = {
            name: np.random.default_rng(child)
            for name, child in zip(_ONLINE_NUMPY_STREAMS, children)
        }

        def _torch_generator(child) -> torch.Generator:
            generator = torch.Generator(device=device)
            raw = int(child.generate_state(1, dtype=np.uint64)[0])
            generator.manual_seed(raw % (2**63 - 1))
            return generator

        return cls(
            numpy=numpy,
            autodiff_collocation=_torch_generator(children[-2]),
            evaluation=_torch_generator(children[-1]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "numpy": {
                name: copy.deepcopy(generator.bit_generator.state)
                for name, generator in self.numpy.items()
            },
            "autodiff_collocation": self.autodiff_collocation.get_state(),
            "evaluation": self.evaluation.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported OnlineSamplerRNGs checkpoint version")
        if tuple(state.get("numpy", {}).keys()) != _ONLINE_NUMPY_STREAMS:
            raise ValueError("online RNG stream names do not match this trainer")
        for name in _ONLINE_NUMPY_STREAMS:
            self.numpy[name].bit_generator.state = copy.deepcopy(state["numpy"][name])
        self.autodiff_collocation.set_state(state["autodiff_collocation"])
        self.evaluation.set_state(state["evaluation"])


@dataclass
class HybridForcingRNGs:
    """Independent common and backend-specific streams for the forcing screen."""

    forcing_selection: np.random.Generator
    contact_resistance: np.random.Generator
    interval_selection: np.random.Generator
    bulk_collocation: torch.Generator
    boundary_collocation: torch.Generator
    ic_collocation: torch.Generator
    validation: torch.Generator

    @classmethod
    def create(cls, seed: int, device: torch.device) -> "HybridForcingRNGs":
        children = np.random.SeedSequence(int(seed)).spawn(7)

        def _torch_generator(child) -> torch.Generator:
            generator = torch.Generator(device=device)
            raw = int(child.generate_state(1, dtype=np.uint64)[0])
            generator.manual_seed(raw % (2**63 - 1))
            return generator

        return cls(
            forcing_selection=np.random.default_rng(children[0]),
            contact_resistance=np.random.default_rng(children[1]),
            interval_selection=np.random.default_rng(children[2]),
            bulk_collocation=_torch_generator(children[3]),
            boundary_collocation=_torch_generator(children[4]),
            ic_collocation=_torch_generator(children[5]),
            validation=_torch_generator(children[6]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "forcing_selection": copy.deepcopy(
                self.forcing_selection.bit_generator.state
            ),
            "contact_resistance": copy.deepcopy(
                self.contact_resistance.bit_generator.state
            ),
            "interval_selection": copy.deepcopy(
                self.interval_selection.bit_generator.state
            ),
            "bulk_collocation": self.bulk_collocation.get_state(),
            "boundary_collocation": self.boundary_collocation.get_state(),
            "ic_collocation": self.ic_collocation.get_state(),
            "validation": self.validation.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in (
            "forcing_selection", "contact_resistance", "interval_selection",
        ):
            getattr(self, name).bit_generator.state = copy.deepcopy(state[name])
        for name in (
            "bulk_collocation", "boundary_collocation", "ic_collocation",
            "validation",
        ):
            getattr(self, name).set_state(state[name])


_FORCING_CONSTRAINT_NAMES = (
    "left_flux",
    "local_energy_interface",
    "local_energy_far",
    "interface_flux",
    "interface_contact",
)


class ForcingConstraintController:
    """Dead-band augmented-Lagrangian state for the forcing reachability gate."""

    def __init__(
        self, tolerances: dict[str, float], *, rho: float = 1.0,
        ema_decay: float = 0.9, dual_every: int = 10,
        multiplier_cap: float = 1000.0, dual_enabled: bool = True,
    ) -> None:
        if tuple(tolerances) != _FORCING_CONSTRAINT_NAMES:
            raise ValueError("forcing constraint tolerance names/order are invalid")
        if any(not math.isfinite(v) or v < 0.0 for v in tolerances.values()):
            raise ValueError("forcing constraint tolerances must be finite and non-negative")
        if rho <= 0.0 or not 0.0 <= ema_decay < 1.0 or dual_every < 1:
            raise ValueError("invalid forcing augmented-Lagrangian settings")
        self.tolerances = dict(tolerances)
        self.rho = float(rho)
        self.ema_decay = float(ema_decay)
        self.dual_every = int(dual_every)
        self.multiplier_cap = float(multiplier_cap)
        self.dual_enabled = bool(dual_enabled)
        self.multipliers = {name: 0.0 for name in _FORCING_CONSTRAINT_NAMES}
        self.ema = {name: 0.0 for name in _FORCING_CONSTRAINT_NAMES}
        self.ema_initialized = {name: False for name in _FORCING_CONSTRAINT_NAMES}

    def values(
        self, residuals: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        rms = {
            name: torch.sqrt(residuals[name].square().mean() + 1.0e-16)
            for name in _FORCING_CONSTRAINT_NAMES
        }
        violation = {
            name: torch.clamp(rms[name] - self.tolerances[name], min=0.0)
            for name in _FORCING_CONSTRAINT_NAMES
        }
        return rms, violation

    def primal(self, violation: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = [
            self.multipliers[name] * violation[name]
            + 0.5 * self.rho * violation[name].square()
            for name in _FORCING_CONSTRAINT_NAMES
        ]
        return torch.stack(terms).sum()

    def update(self, violation: dict[str, torch.Tensor], completed: int) -> None:
        for name in _FORCING_CONSTRAINT_NAMES:
            value = float(violation[name].detach().cpu())
            if self.ema_initialized[name]:
                self.ema[name] = (
                    self.ema_decay * self.ema[name]
                    + (1.0 - self.ema_decay) * value
                )
            else:
                self.ema[name] = value
                self.ema_initialized[name] = True
        if self.dual_enabled and completed % self.dual_every == 0:
            for name in _FORCING_CONSTRAINT_NAMES:
                self.multipliers[name] = min(
                    self.multiplier_cap,
                    self.multipliers[name] + self.rho * self.ema[name],
                )

    def reset_stage_history(self) -> None:
        self.ema = {name: 0.0 for name in _FORCING_CONSTRAINT_NAMES}
        self.ema_initialized = {name: False for name in _FORCING_CONSTRAINT_NAMES}

    def state_dict(self) -> dict[str, Any]:
        return {
            "tolerances": copy.deepcopy(self.tolerances), "rho": self.rho,
            "ema_decay": self.ema_decay, "dual_every": self.dual_every,
            "multiplier_cap": self.multiplier_cap,
            "dual_enabled": self.dual_enabled,
            "multipliers": copy.deepcopy(self.multipliers),
            "ema": copy.deepcopy(self.ema),
            "ema_initialized": copy.deepcopy(self.ema_initialized),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["tolerances"] != self.tolerances:
            raise ValueError("resume constraint tolerances differ from checkpoint")
        for key in ("multipliers", "ema", "ema_initialized"):
            setattr(self, key, copy.deepcopy(state[key]))


def _canonical_descriptor_value(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical_descriptor_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_canonical_descriptor_value(item) for item in value)
    if isinstance(value, list):
        if value and all(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool)
            for item in value
        ):
            return np.asarray(value, dtype="<i8")
        if value and all(isinstance(item, (float, np.floating)) for item in value):
            return np.asarray(value, dtype="<f8")
        return [_canonical_descriptor_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if isinstance(value, (float, np.floating)):
        return np.float64(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return np.int64(value)
    return copy.deepcopy(value)


def _canonical_binary_encode(value) -> bytes:
    chunks: list[bytes] = []

    def _length(n: int) -> bytes:
        return struct.pack("<Q", int(n))

    def _encode(item) -> None:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if item is None:
            chunks.append(b"N")
        elif isinstance(item, (bool, np.bool_)):
            chunks.append(b"B\x01" if bool(item) else b"B\x00")
        elif isinstance(item, (int, np.integer)):
            chunks.extend((b"I", struct.pack("<q", int(item))))
        elif isinstance(item, (float, np.floating)):
            chunks.extend((b"F", struct.pack("<d", float(item))))
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            chunks.extend((b"S", _length(len(encoded)), encoded))
        elif isinstance(item, bytes):
            chunks.extend((b"Y", _length(len(item)), item))
        elif isinstance(item, np.ndarray):
            arr = np.asarray(item)
            dtype = arr.dtype
            if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
                dtype = dtype.newbyteorder("<")
                arr = arr.astype(dtype, copy=False)
            elif dtype.byteorder == "=":
                dtype = dtype.newbyteorder("<")
                arr = arr.astype(dtype, copy=False)
            arr = np.ascontiguousarray(arr)
            dtype_name = dtype.str.encode("ascii")
            chunks.extend((b"A", _length(len(dtype_name)), dtype_name))
            chunks.append(_length(arr.ndim))
            for dim in arr.shape:
                chunks.append(struct.pack("<q", int(dim)))
            raw = arr.tobytes(order="C")
            chunks.extend((_length(len(raw)), raw))
        elif isinstance(item, dict):
            entries = sorted(
                ((str(key), value) for key, value in item.items()
                 if str(key) not in _CANONICAL_EXCLUDED_KEYS),
                key=lambda pair: pair[0],
            )
            chunks.extend((b"M", _length(len(entries))))
            for key, child in entries:
                _encode(key)
                _encode(child)
        elif isinstance(item, (list, tuple)):
            chunks.extend((b"L" if isinstance(item, list) else b"T", _length(len(item))))
            for child in item:
                _encode(child)
        else:
            raise TypeError(f"unsupported canonical descriptor value {type(item)!r}")

    _encode(value)
    return b"".join(chunks)


def _content_key(value) -> str:
    return hashlib.sha256(_canonical_binary_encode(value)).hexdigest()


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "numpy_version": np.__version__,
        "endianness": sys.byteorder,
        "architecture": platform.machine(),
        "ic_builder_version": IC_BUILDER_SCHEMA_VERSION,
    }


def _validate_environment_fingerprint(saved: dict, current: dict) -> None:
    for key in ("endianness", "architecture"):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"online IC environment {key} mismatch: "
                f"{saved.get(key)!r} != {current.get(key)!r}"
            )
    if saved.get("ic_builder_version") != current.get("ic_builder_version"):
        raise ValueError("online IC builder version mismatch")
    if saved.get("numpy_version") != current.get("numpy_version"):
        warnings.warn(
            "NumPy version differs from the online-IC checkpoint; exact "
            "reconstruction hashes will be verified before training.",
            RuntimeWarning,
        )


def _final_float32_hash(field: np.ndarray) -> str:
    field = np.ascontiguousarray(field, dtype=np.float32)
    return hashlib.sha256(field.tobytes(order="C")).hexdigest()


def _online_record_descriptor(record: dict) -> dict:
    T0 = np.ascontiguousarray(record["T0"], dtype=np.float32)
    descriptor = {
        key: _canonical_descriptor_value(value)
        for key, value in record.items() if key != "T0"
    }
    descriptor.update({
        "sampler_version": ONLINE_IC_SAMPLER_VERSION,
        "builder_version": np.int64(IC_BUILDER_SCHEMA_VERSION),
        "T0_sha256": _final_float32_hash(T0),
    })
    descriptor["problem_key"] = _content_key(descriptor)
    return descriptor


def _materialize_online_records(
    descriptors: list[dict], X: np.ndarray, Y: np.ndarray, *,
    T_right: float = 300.0, b: float = 1.0,
) -> list[dict]:
    records: list[dict] = []
    for descriptor in descriptors:
        if not isinstance(descriptor.get("builder_version"), np.integer):
            raise ValueError(
                "online IC descriptors must come from the lossless binary checkpoint; "
                "audit JSON is not an authoritative resume source"
            )
        if descriptor.get("sampler_version") != ONLINE_IC_SAMPLER_VERSION:
            raise ValueError("online IC sampler version mismatch")
        if int(descriptor.get("builder_version", -1)) != IC_BUILDER_SCHEMA_VERSION:
            raise ValueError("online IC builder version mismatch")
        T0 = build_ic(
            str(descriptor["ic_family"]), descriptor["ic_params"], X, Y,
            T_right=float(T_right), b=float(b),
        )
        if _final_float32_hash(T0) != descriptor["T0_sha256"]:
            raise ValueError(
                f"online IC reconstruction failed for {descriptor['problem_key']}"
            )
        record = {
            key: copy.deepcopy(value) for key, value in descriptor.items()
            if key not in {
                "sampler_version", "builder_version", "T0_sha256", "problem_key",
            }
        }
        record["T0"] = T0
        record["problem_key"] = descriptor["problem_key"]
        records.append(record)
    return records


def _normalized_online_ic_buffer(
    records: list[dict], mu: float, sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sigma32 = np.float32(sigma)
    if not np.isfinite(sigma32) or sigma32 <= 0.0:
        raise ValueError("online IC normalization requires finite sigma_global > 0")
    physical = np.ascontiguousarray(
        np.stack([np.asarray(record["T0"], dtype=np.float32) for record in records]),
        dtype=np.float32,
    )
    normalized = np.empty_like(physical, dtype=np.float32)
    np.subtract(physical, np.float32(mu), out=normalized)
    np.divide(normalized, sigma32, out=normalized)
    diagnostics: dict[str, Any] = {
        "max_abs_z": float(np.max(np.abs(normalized))),
        "frac_abs_z_gt_5": float(np.mean(np.abs(normalized) > 5.0)),
        "per_family": {},
    }
    families = np.asarray([str(record.get("ic_family", "saved")) for record in records])
    for family in np.unique(families):
        values = normalized[families == family]
        diagnostics["per_family"][str(family)] = {
            "min": float(values.min()), "max": float(values.max()),
        }
    return physical, normalized, diagnostics


def _transfer_normalized_ic(
    normalized: np.ndarray, device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(normalized).unsqueeze(1)
    if device.type == "cuda":
        tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def _batch_key(records: list[dict], collocation_descriptor) -> str:
    return _content_key({
        "problem_keys": [record["problem_key"] for record in records],
        "collocation": collocation_descriptor,
    })


# --------- collocation sampling ---------

def _leaf(shape, device, generator, scale: float = 1.0) -> torch.Tensor:
    t = torch.rand(shape, device=device, generator=generator) * scale
    return t.requires_grad_(True)


def _lhs_unit(n: int, dim: int, device, generator) -> torch.Tensor:
    """Latin-hypercube sample in the unit cube; (n, dim) in [0, 1).

    Each column is a stratified permutation of ``n`` equal bins with a uniform
    jitter inside the bin, so the marginal coverage of every axis is uniform for
    any ``n`` (broad per-iteration domain coverage; Chen et al. arXiv:2606.06164).
    Columns are permuted independently so the joint sample decorrelates.
    """
    cols = []
    for _ in range(dim):
        perm = torch.randperm(n, device=device, generator=generator).to(torch.float32)
        jitter = torch.rand(n, device=device, generator=generator)
        cols.append((perm + jitter) / float(n))
    return torch.stack(cols, dim=-1)


def _lhs_leaf(col: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Shape an LHS column (n,) into a (1, n, 1) leaf with requires_grad."""
    return (col.reshape(1, -1, 1) * scale).detach().requires_grad_(True)


def _collocation_bias_counts(
    n_r: int, wall_frac: float, lead_frac: float
) -> tuple[int, int, int]:
    """Interior sub-population sizes (n_wall, n_lead, n_full) summing to n_r.

    ``n_full`` absorbs the integer-rounding remainder so the total is exactly
    ``n_r`` (count-preserving). Pure integer arithmetic; tested in isolation.
    """
    n_wall = int(wall_frac * n_r)
    n_lead = int(lead_frac * n_r)
    n_full = n_r - n_wall - n_lead
    return n_wall, n_lead, n_full


def _validate_collocation_bias(bias: dict) -> tuple[float, float, float, float]:
    """Validate the collocation-bias spec and return (wall_frac, wall_x_cut,
    lead_frac, lead_t_lo). Raises ``ValueError`` on any out-of-range / non-finite
    value; the ``wall_frac + lead_frac <= 1`` guard prevents a negative n_full.
    """
    wall_frac = float(bias.get("wall_frac", 0.0))
    wall_x_cut = float(bias.get("wall_x_cut", 0.333))
    lead_frac = float(bias.get("lead_frac", 0.0))
    lead_t_lo = float(bias.get("lead_t_lo", 0.5))
    for name, v in (
        ("wall_frac", wall_frac), ("lead_frac", lead_frac),
        ("wall_x_cut", wall_x_cut), ("lead_t_lo", lead_t_lo),
    ):
        if not math.isfinite(v):
            raise ValueError(f"collocation_bias.{name} must be finite, got {v}")
    if not 0.0 <= wall_frac <= 1.0:
        raise ValueError(f"collocation_bias.wall_frac must be in [0,1], got {wall_frac}")
    if not 0.0 <= lead_frac <= 1.0:
        raise ValueError(f"collocation_bias.lead_frac must be in [0,1], got {lead_frac}")
    if wall_frac + lead_frac > 1.0:
        raise ValueError(
            "collocation_bias.wall_frac + lead_frac must be <= 1, got "
            f"{wall_frac} + {lead_frac}"
        )
    if not 0.0 < wall_x_cut <= 1.0:
        raise ValueError(f"collocation_bias.wall_x_cut must be in (0,1], got {wall_x_cut}")
    if not 0.0 <= lead_t_lo < 1.0:
        raise ValueError(f"collocation_bias.lead_t_lo must be in [0,1), got {lead_t_lo}")
    return wall_frac, wall_x_cut, lead_frac, lead_t_lo


def _resolve_collocation_bias(pino_cfg: dict) -> dict | None:
    """Build the validated collocation-bias spec from the ``training.pino`` config,
    or ``None`` when disabled. Returns ``None`` unless ``collocation_bias.enabled``
    is truthy AND at least one of ``wall_frac`` / ``lead_frac`` is > 0 (so the
    default and the enabled-but-zero cases both keep the legacy RNG-identical
    draw). Validates eagerly, so a bad enabled spec raises before training.
    """
    bias_cfg = pino_cfg.get("collocation_bias", {}) or {}
    if not bias_cfg.get("enabled"):
        return None
    wall_frac = float(bias_cfg.get("wall_frac", 0.0))
    lead_frac = float(bias_cfg.get("lead_frac", 0.0))
    if wall_frac <= 0.0 and lead_frac <= 0.0:
        return None
    coll_bias = {
        "wall_frac": wall_frac,
        "wall_x_cut": float(bias_cfg.get("wall_x_cut", 0.333)),
        "lead_frac": lead_frac,
        "lead_t_lo": float(bias_cfg.get("lead_t_lo", 0.5)),
    }
    _validate_collocation_bias(coll_bias)
    return coll_bias


def _biased_interior(
    n_r: int, t_final: float, device, generator, lhs: bool, bias: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interior x/y/t leaves with a wall/lead/full mixture (static collocation
    biasing). Each subgroup is drawn WITHOUT gradient (``_lhs_unit`` or
    ``torch.rand``); the biased axis is rescaled into its band; the three groups
    are concatenated (order wall->lead->full) and ONLY THEN made leaves via a
    single ``requires_grad_`` per axis. Subgroup-wise LHS (each group stratified
    in its own cube), not one global Latin hypercube.
    """
    wall_frac, wall_x_cut, lead_frac, lead_t_lo = _validate_collocation_bias(bias)
    n_wall, n_lead, n_full = _collocation_bias_counts(n_r, wall_frac, lead_frac)
    lead_lo = lead_t_lo * t_final
    lead_span = t_final - lead_lo

    def _draw(n: int, dim: int) -> torch.Tensor:
        if n == 0:
            return torch.empty((n, dim), device=device)
        if lhs:
            return _lhs_unit(n, dim, device, generator)
        return torch.rand((n, dim), device=device, generator=generator)

    xs, ys, ts = [], [], []
    # wall group: x in [0, wall_x_cut]; y, t full range.
    wall = _draw(n_wall, 3)
    xs.append(wall[:, 0] * wall_x_cut); ys.append(wall[:, 1]); ts.append(wall[:, 2] * t_final)
    # lead group: t in [lead_lo, t_final]; x, y full range.
    lead = _draw(n_lead, 3)
    xs.append(lead[:, 0]); ys.append(lead[:, 1]); ts.append(lead_lo + lead[:, 2] * lead_span)
    # full group: legacy full cube.
    full = _draw(n_full, 3)
    xs.append(full[:, 0]); ys.append(full[:, 1]); ts.append(full[:, 2] * t_final)

    x_r = torch.cat(xs).reshape(1, n_r, 1).requires_grad_(True)
    y_r = torch.cat(ys).reshape(1, n_r, 1).requires_grad_(True)
    t_r = torch.cat(ts).reshape(1, n_r, 1).requires_grad_(True)
    return x_r, y_r, t_r


def sample_collocation(
    n_r: int,
    n_ic: int,
    n_bc: int,
    t_final: float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
    dense_ic: bool = False,
    sampler: str = "uniform",
    bias: dict | None = None,
) -> dict[str, Any]:
    """Free space-time collocation for one training step (shared across sims).

    Returns leaf tensors (requires_grad where a derivative is taken), all with a
    leading batch dim of 1 so they broadcast across the sim minibatch (the model
    expands dim-1 query sets explicitly). Interior/wall points are random;
    IC points are drawn on native grid NODES (as index pairs) so per-sim IC
    targets can be read from the trajectories exactly, with no interpolation.
    The right wall (x=1) is omitted — it is enforced by the hard ansatz.

    ``dense_ic`` replaces the ``n_ic`` random IC nodes with EVERY grid node
    (``Nx*Ny`` points, no RNG draw): the constant field is the physics attractor
    for this benchmark, so a dense per-sim IC anchor is the primary stabilizer.
    The default (``False``) keeps the legacy random-node draw byte-identical.

    ``bias`` (optional) applies static, diagnostic-informed nonuniform biasing to
    the ``n_r`` interior points only (an implicit residual reweighting; total
    count preserved). When ``None`` — or when both ``wall_frac`` and ``lead_frac``
    are <= 0 — the legacy interior draw runs untouched (RNG-identical). IC/wall
    blocks are never biased.
    """
    lhs = sampler == "lhs"

    _wf = float(bias.get("wall_frac", 0.0)) if bias is not None else 0.0
    _lf = float(bias.get("lead_frac", 0.0)) if bias is not None else 0.0
    _biased = bias is not None and (_wf > 0.0 or _lf > 0.0)

    # interior: x, y ~ U(0,1); t ~ U(0, t_final). LHS stratifies the (x, y, t)
    # cube jointly for broader per-iteration coverage.
    if _biased:
        x_r, y_r, t_r = _biased_interior(
            n_r, t_final, device, generator, lhs, bias
        )
    elif lhs:
        cube = _lhs_unit(n_r, 3, device, generator)
        x_r = _lhs_leaf(cube[:, 0])
        y_r = _lhs_leaf(cube[:, 1])
        t_r = _lhs_leaf(cube[:, 2], scale=t_final)
    else:
        x_r = _leaf((1, n_r, 1), device, generator)
        y_r = _leaf((1, n_r, 1), device, generator)
        t_r = _leaf((1, n_r, 1), device, generator, scale=t_final)

    # IC: grid-node indices -> exact coords + a t=0 column
    Nx = int(x_grid.numel())
    Ny = int(y_grid.numel())
    if dense_ic:
        # Every node, row-major (ix varies slowest) -> exact full-grid anchor.
        ix = torch.arange(Nx, device=device).repeat_interleave(Ny)
        iy = torch.arange(Ny, device=device).repeat(Nx)
    else:
        ix = torch.randint(0, Nx, (n_ic,), device=device, generator=generator)
        iy = torch.randint(0, Ny, (n_ic,), device=device, generator=generator)
    n_ic_eff = int(ix.numel())
    ic_x = x_grid[ix].view(1, n_ic_eff, 1)
    ic_y = y_grid[iy].view(1, n_ic_eff, 1)
    ic_coords = torch.cat([ic_x, ic_y], dim=-1)  # (1, n_ic_eff, 2)
    ic_t = torch.zeros((1, n_ic_eff, 1), device=device)

    # walls: free coordinate ~ U(0,1); the pinned coordinate is fixed. All three
    # kept as separate leaves so neumann_residual differentiates unambiguously.
    def _wall(pin: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lhs:
            wc = _lhs_unit(n_bc, 2, device, generator)
            free = _lhs_leaf(wc[:, 0])
            tw = _lhs_leaf(wc[:, 1], scale=t_final)
        else:
            free = _leaf((1, n_bc, 1), device, generator)
            tw = _leaf((1, n_bc, 1), device, generator, scale=t_final)
        if pin == "left":       # x = 0
            xw = torch.zeros((1, n_bc, 1), device=device, requires_grad=True)
            return xw, free, tw
        if pin == "top":        # y = 1
            yw = torch.ones((1, n_bc, 1), device=device, requires_grad=True)
            return free, yw, tw
        # bottom: y = 0
        yw = torch.zeros((1, n_bc, 1), device=device, requires_grad=True)
        return free, yw, tw

    walls = {w: _wall(w) for w in WALLS}

    return {
        "interior": (x_r, y_r, t_r),
        "ic": {"coords": ic_coords, "t": ic_t, "ix": ix, "iy": iy},
        "walls": walls,
    }


# --------- data helpers ---------

def load_diffusion_data(config: dict) -> dict[str, Any]:
    """Load raw trajectories + grids and the train/val/test split + global stats.

    Mirrors the FNO data-load block but skips create_dataloaders /
    SnapshotPairDataset: PINO needs raw ICs and the saved trajectory grid, not
    snapshot pairs.
    """
    trajectories, x_grid, y_grid, t_grid = load_sim_data(
        sim_traj_path=config["data"]["trajectories.npy"],
        x_grid_path=config["data"]["x_grid_path"],
        y_grid_path=config["data"]["y_grid_path"],
        t_grid_path=config["data"]["t_grid_path"],
    )
    num_sims = int(trajectories.shape[0])

    # Versioned (varying-IC) benchmarks: assert the dataset was regenerated with
    # the matching problem_version and stratify the split by IC family. In-place
    # editing keeps the benchmark name and tensor shapes identical to the old
    # fixed-IC set, so this guard is the only thing catching a stale dataset.
    spec = problem_from_config(config)
    expected_version = getattr(spec, "problem_version", None)
    if expected_version is not None:
        assert_dataset_problem_version(spec, config["data"]["t_grid_path"])
        sim_params = np.load(
            Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy",
            allow_pickle=True,
        )
        if len(sim_params) != num_sims:
            raise ValueError(
                f"sim_params length {len(sim_params)} != num_sims {num_sims}."
            )
        ic_labels = np.array([str(sp["ic_family"]) for sp in sim_params])
        train_ids, val_ids, test_ids = split_sim_ids_stratified(
            ic_labels, train_frac=0.7, val_frac=0.15, seed=0,
        )
    else:
        train_ids, val_ids, test_ids = split_sim_ids(
            num_sims=num_sims, train_frac=0.7, val_frac=0.15, seed=0,
        )
    norm_max_time = config["data"].get("norm_max_time", None)
    mu_global, sigma_global = compute_global_stats(
        trajectories, train_ids, t_grid=t_grid, max_time=norm_max_time,
    )
    return {
        "trajectories": trajectories,
        "x_grid": np.asarray(x_grid),
        "y_grid": np.asarray(y_grid),
        "t_grid": np.asarray(t_grid),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "mu_global": float(mu_global),
        "sigma_global": float(sigma_global),
    }


def build_ic_batch(
    trajectories: np.ndarray,
    ids: np.ndarray,
    mu: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Encoder input u = normalized IC field (snapshot 0); (B, 1, Nx, Ny)."""
    ic = np.asarray(trajectories[ids, 0, :, :], dtype=np.float32)
    normalized = np.empty_like(ic)
    np.subtract(ic, np.float32(mu), out=normalized)
    np.divide(normalized, np.float32(sigma), out=normalized)
    return torch.from_numpy(normalized).unsqueeze(1).to(device)


def validate_forcing_ic_supervised_dataset(
    config: dict,
    data: dict[str, Any],
    sim_params: np.ndarray,
) -> dict[str, Any]:
    """Validate the saved-data contract before a supervised model is allocated."""
    data_cfg = config["data"]
    trajectory_path = Path(data_cfg["trajectories.npy"])
    t_grid_path = Path(data_cfg["t_grid_path"])
    required = {
        "trajectories": trajectory_path,
        "x_grid": Path(data_cfg["x_grid_path"]),
        "y_grid": Path(data_cfg["y_grid_path"]),
        "t_grid": t_grid_path,
        "sim_params": trajectory_path.parent / "sim_params.npy",
        "meta": t_grid_path.parent / "meta.npy",
        "ramp_seconds": t_grid_path.parent / "ramp_seconds.npy",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "diffusion_forcing_single supervised training requires the complete "
            f"saved dataset; missing {missing}."
        )

    meta = load_dataset_meta(t_grid_path)
    problem = problem_from_config(config)
    expected_version = getattr(problem, "problem_version", None)
    if meta is None or meta.get("problem_version") != expected_version:
        found = None if meta is None else meta.get("problem_version")
        raise ValueError(
            f"Dataset problem_version={found!r} != expected "
            f"{expected_version!r} for supervised forcing-IC training."
        )
    if meta.get("ic_mode") != "varying":
        raise ValueError("supervised forcing-IC data must declare ic_mode='varying'")

    ramp_value = np.asarray(np.load(required["ramp_seconds"]))
    if ramp_value.ndim != 0:
        raise ValueError(f"{required['ramp_seconds']} must contain a scalar")
    ramp_seconds = float(ramp_value)
    if not math.isfinite(ramp_seconds) or ramp_seconds <= 0.0:
        raise ValueError("ramp_seconds.npy must contain a finite positive value")

    trajectories = data["trajectories"]
    num_sims, Nt, Nx, Ny = (int(v) for v in trajectories.shape)
    if len(sim_params) != num_sims:
        raise ValueError(
            f"sim_params length {len(sim_params)} != num_sims {num_sims}."
        )
    if (
        len(data["t_grid"]) != Nt
        or len(data["x_grid"]) != Nx
        or len(data["y_grid"]) != Ny
    ):
        raise ValueError("trajectory and grid shapes disagree")
    if not (
        np.all(np.diff(data["t_grid"]) > 0.0)
        and np.all(np.diff(data["x_grid"]) > 0.0)
        and np.all(np.diff(data["y_grid"]) > 0.0)
    ):
        raise ValueError("saved x/y/t grids must be strictly increasing")
    if not math.isfinite(float(data["sigma_global"])) or float(data["sigma_global"]) <= 0.0:
        raise ValueError("training-only sigma_global must be finite and positive")

    all_ids = np.arange(num_sims, dtype=int)
    problem.validate_schema(sim_params, all_ids)
    expected_families = tuple(IC_FAMILIES)
    labels = np.asarray([str(p["ic_family"]) for p in sim_params])
    found_families = set(labels.tolist())
    if found_families != set(expected_families):
        raise ValueError(
            "supervised forcing-IC data must contain all and only the configured "
            f"IC families; found {sorted(found_families)}."
        )
    for sid, params in enumerate(sim_params):
        if np.asarray(params["T0"]).shape != (Nx, Ny):
            raise ValueError(
                f"sim_params[{sid}]['T0'] shape "
                f"{np.asarray(params['T0']).shape} != {(Nx, Ny)}."
            )

    split_names = ("train_ids", "val_ids", "test_ids")
    split_ids = [np.asarray(data[name], dtype=int) for name in split_names]
    expected_sizes = (
        int(0.70 * num_sims),
        int(0.15 * num_sims),
        num_sims - int(0.70 * num_sims) - int(0.15 * num_sims),
    )
    if tuple(len(ids) for ids in split_ids) != expected_sizes:
        raise ValueError(
            "saved-data split sizes are not the fixed 70/15/15 contract"
        )
    concatenated = np.concatenate(split_ids)
    if (
        len(np.unique(concatenated)) != num_sims
        or not np.array_equal(np.sort(concatenated), all_ids)
    ):
        raise ValueError("train/val/test simulation IDs must be disjoint and exhaustive")
    for name, ids in zip(split_names, split_ids):
        counts = [int(np.count_nonzero(labels[ids] == family)) for family in expected_families]
        if max(counts, default=0) - min(counts, default=0) > 1:
            raise ValueError(f"{name} is not stratified by IC family: {counts}")

    return {
        "problem_version": expected_version,
        "ic_mode": "varying",
        "shape": [num_sims, Nt, Nx, Ny],
        "dtype": str(trajectories.dtype),
        "split_sizes": list(expected_sizes),
        "ramp_seconds": ramp_seconds,
        "grid_key": _content_key({
            "x": np.asarray(data["x_grid"]),
            "y": np.asarray(data["y_grid"]),
            "t": np.asarray(data["t_grid"]),
        }),
        "sim_params_key": _content_key([dict(p) for p in sim_params]),
    }


def _ic_targets(
    trajectories: np.ndarray,
    ids: np.ndarray,
    ix: torch.Tensor,
    iy: torch.Tensor,
    mu: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-sim normalized IC values at the sampled grid nodes; (B, n_ic, 1)."""
    ix_np = ix.detach().cpu().numpy()
    iy_np = iy.detach().cpu().numpy()
    vals = np.asarray(trajectories[np.asarray(ids)][:, 0, :, :], dtype=np.float32)
    vals = vals[:, ix_np, iy_np]  # (B, n_ic)
    vals = (vals - mu) / (sigma + 1e-8)
    return torch.from_numpy(vals).unsqueeze(-1).to(device)


# --------- online forcing sampling (physics-only, no saved sim_params) ---------

def sample_forcing_params(
    rng: np.random.Generator,
    n: int,
    dt: float,
    t_final: float,
    *,
    c: float = 0.0,
    d: float = 1.0,
    temporal_window: dict | None = None,
    temporal_family: str | None = None,
    spatial_family: str | None = None,
) -> list[dict]:
    """Draw ``n`` fresh separable-forcing parameter sets ``q_L = a(t)*s(y)``.

    Uses the exact ProblemSpec samplers (``sample_temporal_family`` /
    ``sample_spatial_family`` + ``TEMPORAL_SAMPLERS`` / ``SPATIAL_SAMPLERS``) so
    the online training distribution matches the FV validation set. No
    ``sim_params.npy`` is read: physics-only training is not tied to a finite
    saved set (Chen et al. arXiv:2606.06164). ``c, d`` bound the y-domain the
    spatial profile lives on; ``temporal_window`` supplies the sin on/off window.

    ``temporal_family`` / ``spatial_family`` optionally pin the family: when
    ``None`` (default) the family is drawn as before (``sample_*_family(rng)``),
    preserving the existing RNG consumption; when a family string is given, the
    corresponding family draw is skipped and that fixed family is used (the
    ``diffusion_forcing_single`` benchmark pins ``sin`` / ``uniform``).
    """
    tw = temporal_window or {}
    win = dict(
        t_on=float(tw.get("t_on", 0.0)),
        t_off=float(tw.get("t_off", 0.2)),
        phase=float(tw.get("phase", 0.0)),
        tukey_alpha=float(tw.get("tukey_alpha", 0.5)),
    )
    out: list[dict] = []
    for _ in range(int(n)):
        tf = temporal_family if temporal_family is not None else sample_temporal_family(rng)
        tp = TEMPORAL_SAMPLERS[tf](rng, dt=dt, t_final=t_final, **win)
        sf = spatial_family if spatial_family is not None else sample_spatial_family(rng)
        sp = SPATIAL_SAMPLERS[sf](rng, c=c, d=d)
        out.append({
            "temporal_family": tf, "temporal_params": tp,
            "spatial_family": sf, "spatial_params": sp,
        })
    return out


def build_forcing_image(
    params: list[dict],
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    device: torch.device,
    t_ramp: float,
) -> torch.Tensor:
    """Encoder input: the forcing rendered as a space-time image; (B, 1, Ny, Nt).

    ``u_enc(y, t) = q_L(y, t) / a_ref`` with ``q_L`` reconstructed by the SAME
    ``reconstruct_qL`` helper the left-wall residual uses, so the encoder image
    and the residual forcing can never diverge. Axes are (rows = y, cols = t).
    """
    if float(a_ref) <= 0.0:
        raise ValueError(f"a_ref must be > 0; got {a_ref}.")
    B = len(params)
    Ny, Nt = int(y_img.shape[0]), int(t_img.shape[0])
    img = np.empty((B, 1, Ny, Nt), dtype=np.float32)
    for b, p in enumerate(params):
        forcing = reconstruct_qL(
            p["temporal_family"], p["temporal_params"],
            p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
        )
        img[b, 0] = np.asarray(
            forcing.evaluate_grid(y_img, t_img), dtype=np.float32
        ) / float(a_ref)
    return torch.from_numpy(img).to(device)


def build_forcing_transition_image(
    params: list[dict],
    y_img: np.ndarray,
    source_time: torch.Tensor | np.ndarray,
    lead_time: torch.Tensor | np.ndarray,
    nt_img: int,
    a_ref: float,
    device: torch.device,
    t_ramp: float,
    t_final: float,
) -> torch.Tensor:
    """Render a causal, source-relative forcing interval as three image channels."""
    if float(a_ref) <= 0.0:
        raise ValueError(f"a_ref must be > 0; got {a_ref}.")
    if int(nt_img) < 2:
        raise ValueError(f"nt_img must be at least 2; got {nt_img}.")
    if float(t_final) <= 0.0:
        raise ValueError(f"t_final must be > 0; got {t_final}.")
    source = np.asarray(
        torch.as_tensor(source_time).detach().cpu(), dtype=np.float64,
    ).reshape(-1)
    lead = np.asarray(
        torch.as_tensor(lead_time).detach().cpu(), dtype=np.float64,
    ).reshape(-1)
    if source.shape != lead.shape or source.shape[0] != len(params):
        raise ValueError(
            "params, source_time, and lead_time must have matching batch sizes"
        )
    if (
        not np.isfinite(source).all()
        or not np.isfinite(lead).all()
        or np.any(source < 0.0)
        or np.any(lead < 0.0)
        or np.any(source + lead > float(t_final) + 1.0e-7)
    ):
        raise ValueError(
            "source and lead times must be finite and satisfy "
            "0 <= source <= source + lead <= t_final"
        )

    relative = np.linspace(0.0, 1.0, int(nt_img), dtype=np.float64)
    y_values = np.asarray(y_img, dtype=np.float64).reshape(-1)
    image = np.empty(
        (len(params), 3, y_values.size, int(nt_img)), dtype=np.float32,
    )
    image[:, 1] = relative[None, None, :]
    for batch_index, p in enumerate(params):
        absolute = source[batch_index] + relative * lead[batch_index]
        forcing = reconstruct_qL(
            p["temporal_family"],
            p["temporal_params"],
            p["spatial_family"],
            p["spatial_params"],
            t_ramp=t_ramp,
        )
        image[batch_index, 0] = np.asarray(
            forcing.evaluate_grid(y_values, absolute), dtype=np.float32,
        ) / float(a_ref)
        image[batch_index, 2] = (
            absolute[None, :] / float(t_final)
        ).astype(np.float32)
    return torch.from_numpy(image).to(device)


def _snap_bin_value(value: float, edges: np.ndarray, tolerance: float) -> float:
    nearest = int(np.argmin(np.abs(edges - value)))
    if abs(float(edges[nearest]) - value) <= tolerance:
        return float(edges[nearest])
    return float(value)


def _right_closed_bin(
    value: float,
    edges: np.ndarray,
    tolerance: float,
) -> int | None:
    value = _snap_bin_value(value, edges, tolerance)
    index = int(np.searchsorted(edges, value, side="left") - 1)
    return index if 0 <= index < len(edges) - 1 else None


def _left_closed_bin(
    value: float,
    edges: np.ndarray,
    tolerance: float,
    *,
    include_final: bool,
) -> int | None:
    if include_final and abs(value - float(edges[-1])) <= tolerance:
        return len(edges) - 2
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return index if 0 <= index < len(edges) - 1 else None


class TransitionPairSchedule:
    """Index-only transition strata shared by training and validation."""

    def __init__(
        self,
        t_grid: np.ndarray,
        n_snapshots: int | None,
        lead_edges: list[float] | tuple[float, ...],
        source_edges: list[float] | tuple[float, ...],
        target_edges: list[float] | tuple[float, ...] | None = None,
    ):
        self.t_grid = np.asarray(t_grid, dtype=np.float64)
        if self.t_grid.ndim != 1 or self.t_grid.size < 2:
            raise ValueError("t_grid must be one-dimensional with at least two points")
        if not np.all(np.diff(self.t_grid) > 0.0):
            raise ValueError("t_grid must be strictly increasing")
        if n_snapshots is not None and int(n_snapshots) < self.t_grid.size:
            if int(n_snapshots) < 2:
                raise ValueError("n_snapshots must be at least two")
            self.snapshot_indices = np.unique(
                np.round(
                    np.linspace(0, self.t_grid.size - 1, int(n_snapshots))
                ).astype(int)
            )
        else:
            self.snapshot_indices = np.arange(self.t_grid.size, dtype=int)
        self.lead_edges = self._validate_edges(lead_edges, "lead_edges")
        self.source_edges = self._validate_edges(source_edges, "source_edges")
        self.target_edges = self._validate_edges(
            source_edges if target_edges is None else target_edges,
            "target_edges",
        )
        self.tolerance = 0.5 * float(np.min(np.diff(self.t_grid)))
        self.cells: dict[
            tuple[int, int], dict[int, tuple[int, ...]]
        ] = {}
        mutable: dict[tuple[int, int], dict[int, list[int]]] = {}
        for position, source_index in enumerate(self.snapshot_indices[:-1]):
            source_time = float(self.t_grid[source_index])
            source_bin = _left_closed_bin(
                source_time,
                self.source_edges,
                self.tolerance,
                include_final=False,
            )
            if source_bin is None:
                continue
            for target_index in self.snapshot_indices[position + 1:]:
                lead = float(
                    self.t_grid[target_index] - self.t_grid[source_index]
                )
                lead_bin = _right_closed_bin(
                    lead, self.lead_edges, self.tolerance,
                )
                if lead_bin is None:
                    continue
                mutable.setdefault((source_bin, lead_bin), {}).setdefault(
                    int(source_index), []
                ).append(int(target_index))
        self.cells = {
            cell: {
                source_index: tuple(targets)
                for source_index, targets in sources.items()
            }
            for cell, sources in mutable.items()
        }
        self.eligible_lead_bins = tuple(
            sorted({lead_bin for _, lead_bin in self.cells})
        )
        if not self.eligible_lead_bins:
            raise ValueError("the transition bin configuration contains no valid pairs")

    @staticmethod
    def _validate_edges(values, name: str) -> np.ndarray:
        edges = np.asarray(values, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0.0)
        ):
            raise ValueError(f"{name} must be a finite, strictly increasing sequence")
        return edges

    def cell_for_pair(
        self,
        source_index: int,
        target_index: int,
    ) -> tuple[int, int, int]:
        source = float(self.t_grid[int(source_index)])
        target = float(self.t_grid[int(target_index)])
        source_bin = _left_closed_bin(
            source, self.source_edges, self.tolerance, include_final=False,
        )
        lead_bin = _right_closed_bin(
            target - source, self.lead_edges, self.tolerance,
        )
        target_bin = _left_closed_bin(
            target, self.target_edges, self.tolerance, include_final=True,
        )
        if source_bin is None or lead_bin is None or target_bin is None:
            raise ValueError("pair lies outside the configured transition bins")
        return source_bin, lead_bin, target_bin

    def sample_pair(
        self,
        base_seed: int,
        epoch: int,
        sim_id: int,
    ) -> tuple[int, int, int, int]:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(base_seed), int(epoch), int(sim_id), 0],
            )
        )
        lead_bin = int(rng.choice(self.eligible_lead_bins))
        source_bins = tuple(
            sorted(
                source_bin
                for source_bin, candidate_lead in self.cells
                if candidate_lead == lead_bin
            )
        )
        source_bin = int(rng.choice(source_bins))
        sources = self.cells[(source_bin, lead_bin)]
        source_index = int(rng.choice(tuple(sorted(sources))))
        target_index = int(rng.choice(sources[source_index]))
        return source_index, target_index, source_bin, lead_bin

    def validation_records(
        self,
        sim_ids: np.ndarray,
        *,
        pairs_per_cell: int,
        base_seed: int,
    ) -> list[dict[str, int]]:
        if pairs_per_cell <= 0:
            raise ValueError("pairs_per_cell must be positive")
        records: list[dict[str, int]] = []
        for sim_id in np.asarray(sim_ids, dtype=int):
            for source_bin, lead_bin in sorted(self.cells):
                pairs = [
                    (source_index, target_index)
                    for source_index, targets in self.cells[
                        (source_bin, lead_bin)
                    ].items()
                    for target_index in targets
                ]
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            int(base_seed),
                            int(sim_id),
                            int(source_bin),
                            int(lead_bin),
                            1,
                        ]
                    )
                )
                count = min(int(pairs_per_cell), len(pairs))
                chosen = rng.choice(len(pairs), size=count, replace=False)
                for pair_index in np.atleast_1d(chosen):
                    source_index, target_index = pairs[int(pair_index)]
                    _, _, target_bin = self.cell_for_pair(
                        source_index, target_index,
                    )
                    records.append({
                        "sim_id": int(sim_id),
                        "source_index": int(source_index),
                        "target_index": int(target_index),
                        "source_bin": int(source_bin),
                        "lead_bin": int(lead_bin),
                        "target_bin": int(target_bin),
                    })
        return records


def transition_epoch_order(
    sim_ids: np.ndarray,
    *,
    base_seed: int,
    epoch: int,
) -> np.ndarray:
    ids = np.asarray(sim_ids, dtype=int)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(epoch), 2]),
    )
    return rng.permutation(ids)


def transition_manifest_hash(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_transition_manifest(
    path: Path,
    records: list[dict[str, Any]],
) -> str:
    digest = transition_manifest_hash(records)
    payload = {
        "schema_version": 1,
        "sha256": digest,
        "records": records,
    }
    _atomic_text(json.dumps(payload, indent=2) + "\n", Path(path))
    return digest


def left_wall_qL(
    params: list[dict],
    y_pts: torch.Tensor,
    t_pts: torch.Tensor,
    device: torch.device,
    t_ramp: float,
) -> torch.Tensor:
    """Precomputed inward flux ``q_L(y_w, t_w)`` at the left-wall points; (B, N, 1).

    ``y_pts`` / ``t_pts`` are the shared left-wall collocation leaves (any shape
    with ``N`` elements); ``q_L`` is evaluated per sim via the shared
    ``reconstruct_qL`` ``q_at`` at exactly those points. Returned detached (the
    residual only needs autograd through ``dT/dx``, never through ``q_L``).
    """
    yv = y_pts.detach().reshape(-1).cpu().numpy()
    tv = t_pts.detach().reshape(-1).cpu().numpy()
    B, N = len(params), int(yv.shape[0])
    q = np.empty((B, N), dtype=np.float32)
    for b, p in enumerate(params):
        forcing = reconstruct_qL(
            p["temporal_family"], p["temporal_params"],
            p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
        )
        q[b] = np.asarray(forcing.evaluate_points(yv, tv), dtype=np.float32)
    return torch.from_numpy(q).unsqueeze(-1).to(device)


# --------- causal residual weighting + time-bin diagnostics ---------

def _time_bin_index(t: torch.Tensor, t_final: float, n_bins: int) -> torch.Tensor:
    """Long bin index in [0, n_bins-1] for times t in [0, t_final]."""
    frac = t / max(float(t_final), 1e-12)
    idx = (frac * n_bins).long()
    return idx.clamp_(0, n_bins - 1)


def _bin_residual(
    r: torch.Tensor, t_r: torch.Tensor, t_final: float, n_bins: int
) -> torch.Tensor:
    """Detached per-time-bin mean squared residual, (n_bins,); empty bins -> 0."""
    sq = (r ** 2).mean(dim=0).reshape(-1).detach()
    idx = _time_bin_index(t_r.reshape(-1).detach(), t_final, n_bins)
    bin_sum = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    bin_cnt = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    bin_sum = bin_sum.index_add(0, idx, sq)
    bin_cnt = bin_cnt.index_add(0, idx, torch.ones_like(sq))
    return bin_sum / bin_cnt.clamp_min(1.0)


def _causal_bin_stats(
    r: torch.Tensor, t_r: torch.Tensor, t_final: float, n_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-input bin MSE, shared-time counts, and pointwise residual MSE."""
    sq = (r ** 2).reshape(r.shape[0], -1)
    idx = _time_bin_index(t_r.reshape(-1).detach(), t_final, n_bins)
    counts = torch.zeros(n_bins, device=r.device, dtype=sq.dtype)
    counts.index_add_(0, idx, torch.ones_like(idx, dtype=sq.dtype))
    sums = torch.zeros((sq.shape[0], n_bins), device=r.device, dtype=sq.dtype)
    sums.scatter_add_(1, idx.unsqueeze(0).expand(sq.shape[0], -1), sq)
    per_input = sums / counts.clamp_min(1.0).unsqueeze(0)
    return per_input.mean(dim=0), counts, sq.mean()


def _causal_weights(bin_mean: torch.Tensor, eps_causal: float) -> torch.Tensor:
    """Causal weights w_i = exp(-eps * sum_{j<i} L_j) from detached bin losses.

    Later time bins are only penalized once the earlier bins are resolved
    (Wang, Sankaran & Perdikaris 2022; Chen et al. arXiv:2606.06164): a small
    time-averaged residual that nonetheless violates the causal time evolution is
    the failure mode this counteracts.
    """
    cum_prev = torch.cumsum(bin_mean, 0) - bin_mean
    return torch.exp(-float(eps_causal) * cum_prev)


def _causal_residual_loss(
    r: torch.Tensor, t_r: torch.Tensor, t_final: float, n_bins: int, eps_causal: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causally weighted interior residual loss + detached per-bin residual.

    Bins the interior residual by query time, forms a differentiable per-bin mean
    squared residual, and returns a weighted mean over occupied bins with causal
    weights (detached, so they act as a mask, not an extra gradient path).
    """
    bin_mean, bin_cnt, pointwise = _causal_bin_stats(r, t_r, t_final, n_bins)
    if bool((bin_cnt > 0).all().item()):
        w = _causal_weights(bin_mean.detach(), eps_causal)
        loss = (w * bin_mean).mean()
    else:
        loss = pointwise
    return loss, bin_mean.detach()


def _resolve_forcing_causal(causal_cfg: dict | None) -> dict[str, Any]:
    cfg = dict(causal_cfg or {})
    initial = cfg.get("initial_eps")
    if initial is None:
        initial = cfg.get("eps_causal")
    if initial is None:
        initial = 1.0e-2
    resolved = {
        "enabled": bool(cfg.get("enabled", False)),
        "n_bins": int(cfg.get("n_bins", 24)),
        "initial_eps": float(initial),
        "min_eps": float(cfg.get("min_eps", 1.0e-12)),
        "max_eps": float(cfg.get("max_eps", 100.0)),
        "step_size": float(cfg.get("step_size", 5.0)),
        "min_mean_weight": float(cfg.get("min_mean_weight", 0.4)),
        "max_min_weight": float(cfg.get("max_min_weight", 0.99)),
    }
    if resolved["n_bins"] <= 0:
        raise ValueError("training.pino.causal.n_bins must be > 0")
    if not 0.0 < resolved["min_eps"] <= resolved["initial_eps"] <= resolved["max_eps"]:
        raise ValueError("causal epsilon values must satisfy 0 < min_eps <= initial_eps <= max_eps")
    if resolved["step_size"] <= 1.0:
        raise ValueError("training.pino.causal.step_size must be > 1")
    for key in ("min_mean_weight", "max_min_weight"):
        if not 0.0 <= resolved[key] <= 1.0:
            raise ValueError(f"training.pino.causal.{key} must be in [0, 1]")
    return resolved


def _adapt_causal_eps(
    eps: float, weights: torch.Tensor, cfg: dict[str, Any], *, populated: bool,
) -> tuple[float, str]:
    if not populated:
        return float(eps), "hold_empty"
    mean_weight = float(weights.mean().item())
    last_weight = float(weights[-1].item())
    if last_weight > cfg["max_min_weight"]:
        return min(float(cfg["max_eps"]), eps * float(cfg["step_size"])), "increase"
    if mean_weight < cfg["min_mean_weight"]:
        return max(float(cfg["min_eps"]), eps / float(cfg["step_size"])), "decrease"
    return float(eps), "hold"


# --------- loss ---------

def _ic_loss(
    ic: torch.Tensor,
    ic_target: torch.Tensor,
    *,
    mode: str = "mse",
    t_right_tilde: float = 0.0,
    eps: float = 1.0e-6,
    den: torch.Tensor | None = None,
) -> torch.Tensor:
    """IC term from the raw IC residual ``ic = T_hat_norm(0) - ic_target``.

    ``mode="mse"`` (legacy): plain ``mean(ic**2)`` -- a raw normalized MSE that
    goes numerically small on a near-300 K field even when the *relative* error
    in the nonconstant thermal signal is large (t0 rel-L2 ~0.75 on diffusion, the
    IC that seeds the whole trajectory).

    ``mode="rel"`` with ``den=None`` (legacy sampled): per-sim relative L2 with a
    *sampled* denominator ``sum_i (ic_target_i - t_right_tilde)**2``. This is the
    unstable form -- when the sampled IC nodes land in near-flat regions the
    denominator collapses and the ratio explodes, corrupting SOAP's second-moment
    preconditioner (observed loss_ic spikes to >2000).

    ``mode="rel"`` with ``den`` provided (stabilized full-grid): the denominator
    is a *fixed per-sim signal norm* precomputed on the whole grid,
    ``D_sim = mean_{x,y}[(T0_tilde - t_right_tilde)**2]`` (already floored by the
    caller). The numerator is the per-point mean squared IC residual over the
    sampled nodes, so numerator and denominator are both per-point mean-squares
    (dimensionally consistent regardless of ``n_ic``) and the ratio is bounded by
    the floor. ``den`` has shape ``(B,)``.
    """
    if mode == "rel":
        if den is not None:
            num = (ic ** 2).mean(dim=(1, 2))               # (B,) per-point MS
            return (num / den).mean()
        num = (ic ** 2).sum(dim=(1, 2))                    # (B,)
        dev = ic_target - t_right_tilde
        den = (dev ** 2).sum(dim=(1, 2)) + eps             # (B,)
        return (num / den).mean()
    return (ic ** 2).mean()


def pino_losses(
    model: CViT,
    u: torch.Tensor,
    batch: dict[str, Any],
    ic_target: torch.Tensor,
    alpha: float,
    *,
    causal_cfg: dict | None = None,
    t_final: float | None = None,
    res_bins: int = 0,
    ic_loss: str = "mse",
    t_right_tilde: float = 0.0,
    ic_eps: float = 1.0e-6,
    ic_den: torch.Tensor | None = None,
    compute_r: bool = True,
    compute_bc: bool = True,
    left_qL: torch.Tensor | None = None,
    sigma: float = 1.0,
    k_slab: float = 1.0,
    predict: Callable[..., torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Raw physics/IC/BC losses. ``causal_cfg.enabled`` swaps the plain
    ``mean(r^2)`` interior term for a causally time-weighted one (needs
    ``t_final``); ``res_bins > 0`` also returns a detached ``res_bins`` per-time
    residual vector for diagnostics. ``ic_loss="rel"`` (with ``t_right_tilde``)
    swaps the raw IC MSE for the per-sim relative IC L2 (see ``_ic_loss``);
    passing ``ic_den`` (shape ``(B,)``) selects the stabilized fixed full-grid
    denominator path. ``compute_r=False`` / ``compute_bc=False`` skip the
    (expensive, double-backward) residual / BC autodiff entirely and return a
    zero placeholder -- used for the IC-only diagnostic. All default off/on ->
    legacy behavior.

    ``predict`` is an optional decode closure ``predict(coords, t, q_left=None)``
    over a latent encoded ONCE per step (the two-branch :class:`ForcingICCViT`
    path). When given, every residual reuses that single latent and ``model``/
    ``u`` are only used for batch size / dtype; when ``None`` the monolithic
    ``model(u, ...)`` path is unchanged. The residual math (interior, soft-left
    forcing Neumann, homogeneous walls, IC) is identical on both paths.
    """
    bin_mean = None
    bin_count = None
    causal_weights = None
    pointwise_r = None
    r = None
    if compute_r:
        x_r, y_r, t_r = batch["interior"]
        r = diffusion_residual(model, u, x_r, y_r, t_r, alpha=alpha, predict=predict)
        pointwise_r = (r ** 2).mean()
        causal_on = bool(causal_cfg and causal_cfg.get("enabled", False))
        if causal_on:
            if t_final is None:
                raise ValueError("causal residual weighting requires t_final")
            n_bins = int(causal_cfg.get("n_bins", 24))
            eps = float(causal_cfg.get("current_eps", causal_cfg.get("initial_eps", 1e-2)))
            bin_mean_live, bin_count, pointwise_r = _causal_bin_stats(
                r, t_r, float(t_final), n_bins,
            )
            populated = bool((bin_count > 0).all().item())
            causal_weights = _causal_weights(bin_mean_live.detach(), eps)
            loss_r = (
                (causal_weights * bin_mean_live).mean()
                if populated else pointwise_r
            )
            bin_mean = bin_mean_live.detach()
        else:
            loss_r = pointwise_r
    else:
        loss_r = u.new_zeros(())

    ic = ic_residual(
        model, u, batch["ic"]["coords"], batch["ic"]["t"], ic_target, predict=predict,
    )
    loss_ic = _ic_loss(
        ic, ic_target, mode=ic_loss, t_right_tilde=t_right_tilde, eps=ic_eps,
        den=ic_den,
    )

    loss_bc_left = u.new_zeros(())
    loss_bc_hom = u.new_zeros(())
    if compute_bc:
        bc_sq = 0.0
        hom_sq = 0.0
        n_hom = 0
        for w in WALLS:
            xw, yw, tw = batch["walls"][w]
            if w == "left" and left_qL is not None:
                # Inhomogeneous forcing residual dT_tilde/dx + q_L/(k*sigma); the
                # left wall is now the actual forcing signal (top/bottom stay
                # homogeneous adiabatic). q_L is a precomputed constant tensor.
                nb = forcing_neumann_residual(
                    model, u, xw, yw, tw, left_qL, sigma, k=k_slab, predict=predict,
                )
                loss_bc_left = (nb ** 2).mean()
                bc_sq = bc_sq + loss_bc_left
            else:
                nb = neumann_residual(model, u, xw, yw, tw, w, predict=predict)
                wall_sq = (nb ** 2).mean()
                bc_sq = bc_sq + wall_sq
                hom_sq = hom_sq + wall_sq
                n_hom += 1
        loss_bc = bc_sq / len(WALLS)
        if n_hom > 0:
            loss_bc_hom = hom_sq / n_hom
    else:
        loss_bc = u.new_zeros(())

    out = {
        "r": loss_r,
        "ic": loss_ic,
        "bc": loss_bc,
        "bc_left": loss_bc_left,
        "bc_hom": loss_bc_hom,
    }
    if pointwise_r is not None:
        out["r_pointwise_mse"] = pointwise_r.detach()
    if causal_weights is not None and bin_mean is not None and bin_count is not None:
        populated = bool((bin_count > 0).all().item())
        equal_bin = bin_mean.mean() if populated else pointwise_r.detach()
        out.update({
            "causal_bin_losses": bin_mean,
            "causal_bin_counts": bin_count.detach(),
            "causal_weights": causal_weights.detach(),
            "causal_populated": populated,
            "r_equal_bin_mean": equal_bin,
            "r_causal_loss": loss_r.detach(),
            "causal_reduction_ratio": (
                loss_r.detach() / equal_bin.clamp_min(torch.finfo(equal_bin.dtype).tiny)
            ),
        })
    if res_bins and r is not None:
        if bin_mean is not None and bin_mean.numel() == res_bins:
            out["res_bins"] = bin_mean
        else:
            out["res_bins"] = _bin_residual(r, t_r, float(t_final), int(res_bins))
    return out


# --------- loss weighting (curriculum + GradNorm) ---------

def _curriculum_weights(
    epoch: int,
    lam_r: float,
    lam_ic: float,
    lam_bc: float,
    cfg: dict | None,
) -> tuple[float, float, float]:
    """Static per-term weights for this epoch under the IC-first curriculum.

    Returns the base ``lambda_*`` unchanged when the curriculum is disabled.

    ``mode="ic_only"`` (default, legacy): an IC-only phase (``w_r = w_bc = 0``)
    for ``ic_only_epochs`` steps, then a linear ramp of the residual/BC weights up
    to their ``lambda_*`` targets over ``ramp_epochs`` steps, then full weights.

    ``mode="ic_heavy"``: the residual/BC stay ON through the warm-up (Chen et al.
    arXiv:2606.06164 — the residual must see the IC-anchored field from ``t=0`` to
    propagate it forward, so zeroing it re-opens the constant-field basin). For
    ``warmup_epochs`` steps the weights are ``(warmup_lambda_r, lambda_ic,
    warmup_lambda_bc)``; then over ``decay_epochs`` the IC weight linearly relaxes
    ``lambda_ic -> lambda_ic_final`` while the residual/BC weights ramp
    ``warmup_lambda_* -> lambda_*``. The IC weight is always active.
    """
    if not cfg or not cfg.get("enabled", False):
        return lam_r, lam_ic, lam_bc
    mode = str(cfg.get("mode", "ic_only"))
    if mode == "ic_heavy":
        warm = int(cfg.get("warmup_epochs", 0))
        wr0 = float(cfg.get("warmup_lambda_r", lam_r))
        wbc0 = float(cfg.get("warmup_lambda_bc", lam_bc))
        decay = max(1, int(cfg.get("decay_epochs", 1)))
        ic_final = float(cfg.get("lambda_ic_final", lam_ic))
        if epoch < warm:
            return wr0, lam_ic, wbc0
        s = min(1.0, (epoch - warm) / decay)
        w_ic = lam_ic + s * (ic_final - lam_ic)
        w_r = wr0 + s * (lam_r - wr0)
        w_bc = wbc0 + s * (lam_bc - wbc0)
        return w_r, w_ic, w_bc
    ic_only = int(cfg.get("ic_only_epochs", 0))
    ramp = max(1, int(cfg.get("ramp_epochs", 1)))
    if epoch < ic_only:
        return 0.0, lam_ic, 0.0
    s = min(1.0, (epoch - ic_only) / ramp)
    return s * lam_r, lam_ic, s * lam_bc


def build_gradnorm(
    config: dict,
    lam_r: float | None = None,
    lam_ic: float | None = None,
    lam_bc: float | None = None,
    *,
    term_weights: dict[str, float] | None = None,
):
    """GradNormBalancer over the active PINO terms, or None when disabled.

    Reuses the FNO path's balancer (inverse gradient-norm multipliers, blended
    toward each new target by ``alpha_w``). ``term_weights`` gives an explicit ordered term -> static-weight map
    (its insertion order fixes ``term_names``); the forcing trainer passes the split
    set ``{r, ic, bc_left, bc_hom}`` so the left forcing wall is isolated from the
    near-satisfied adiabatic walls. When ``term_weights`` is None the legacy generic
    set ``{r, ic, bc}`` is used. A term is included when its static weight > 0 (an
    IC-first curriculum may zero ``r``/``bc`` early, but those still ramp on later,
    so they belong in ``term_names``). Opt-in ``w_min`` / ``w_max`` / ``floor``
    guardrails are threaded through from ``config["training"]["gradnorm"]``.
    """
    gn_cfg = config["training"].get("gradnorm", {}) or {}
    if not bool(gn_cfg.get("enabled", False)):
        return None
    if term_weights is None:
        term_weights = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
    terms = [n for n, w in term_weights.items() if w is not None and w > 0.0]
    floor_cfg = gn_cfg.get("floor", {}) or {}
    return GradNormBalancer(
        terms,
        alpha_w=float(gn_cfg.get("alpha_w", 1.0)),
        update_every=int(gn_cfg.get("update_every", 10)),
        eps=float(gn_cfg.get("eps", 1.0e-8)),
        w_min=None if gn_cfg.get("w_min") is None else float(gn_cfg["w_min"]),
        w_max=None if gn_cfg.get("w_max") is None else float(gn_cfg["w_max"]),
        floors={str(k): float(v) for k, v in dict(floor_cfg).items()},
    )


def _term_grad_norms(terms: dict[str, torch.Tensor], params) -> dict[str, float]:
    """L2 gradient norm of each raw loss term w.r.t. ``params`` on the live graph.

    Diagnostic only: exposes which objective dominates the shared backbone
    gradient (the quantity GradNorm balances). Uses ``retain_graph=True`` so the
    caller's subsequent ``loss.backward()`` still runs; call BEFORE that backward.
    """
    params = [p for p in params if p.requires_grad]
    out: dict[str, float] = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        sq = torch.zeros((), device=params[0].device) if params else torch.zeros(())
        for g in grads:
            if g is not None:
                sq = sq + g.detach().pow(2).sum()
        out[name] = float(torch.sqrt(sq))
    return out


def _term_grad_cosines(
    terms: dict[str, torch.Tensor],
    params,
    pairs: list[tuple[str, str]] | None = None,
    *,
    tiny: float = 1.0e-12,
) -> dict[str, float | None]:
    """Pairwise cosine similarity of raw-loss gradients over ``params``.

    Diagnostic: a near-zero cosine means the terms are mostly a MAGNITUDE problem
    (GradNorm-friendly); a strongly negative cosine means a DIRECTION conflict that
    GradNorm alone will not resolve. Dot products and squared norms are accumulated
    tensorwise (never concatenating one giant vector). A pair maps to ``None`` when
    either gradient norm < ``tiny`` (e.g. a near-satisfied ``bc_hom``) so the log
    carries an honest null instead of a NaN or an artificial 0. Uses
    ``retain_graph=True``; call BEFORE the caller's ``loss.backward()``.
    """
    params = [p for p in params if p.requires_grad]
    grads = {
        name: torch.autograd.grad(
            term, params, retain_graph=True, allow_unused=True,
        )
        for name, term in terms.items()
    }

    def _sq(g):
        s = None
        for t in g:
            if t is None:
                continue
            c = t.detach().pow(2).sum()
            s = c if s is None else s + c
        return s

    def _dot(ga, gb):
        s = None
        for a, b in zip(ga, gb):
            if a is None or b is None:
                continue
            c = (a.detach() * b.detach()).sum()
            s = c if s is None else s + c
        return s

    if pairs is None:
        keys = list(terms.keys())
        pairs = [
            (keys[i], keys[j])
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        ]
    sq_norms = {name: _sq(g) for name, g in grads.items()}
    out: dict[str, float | None] = {}
    for a, b in pairs:
        key = f"{a}|{b}"
        sa, sb = sq_norms.get(a), sq_norms.get(b)
        if sa is None or sb is None:
            out[key] = None
            continue
        na, nb = float(torch.sqrt(sa)), float(torch.sqrt(sb))
        if na < tiny or nb < tiny:
            out[key] = None
            continue
        d = _dot(grads[a], grads[b])
        out[key] = None if d is None else float(d) / (na * nb)
    return out


# --------- validation ---------

@torch.no_grad()
def validate_rel_l2(
    model: CViT,
    data: dict[str, Any],
    ids: np.ndarray,
    device: torch.device,
    query_batch: int = 8,
    return_per_time: bool = False,
) -> dict[str, float]:
    """Grid-query rel-L2 (normalized) + Kelvin RMSE over held-out sims.

    Queries the model on the full (x_grid, y_grid) mesh at every saved t_grid
    time and compares to the stored trajectories. ``return_per_time=True`` adds
    honest per-time / banded diagnostics for a near-uniform late-time field where
    the plain normalized rel-L2 denominator collapses toward the constant-300 K
    reference:
      - ``per_time``        normalized rel-L2 at each saved time (length Nt),
      - ``per_time_rmse_K`` Kelvin RMSE at each saved time,
      - ``per_time_rel_dev`` rel-L2 measured on the deviation (T - 300 K); sigma
        cancels so this equals the Kelvin rel-L2 on ``T - T_RIGHT``,
      - ``t0_rel_l2``       the t=0 rel-L2 (the IC pass/fail gate),
      - band scalars ``rel_l2_{early,mid,late}``, ``rmse_K_{early,mid,late}``,
        ``rel_dev_{early,mid,late}`` over t<=0.05 / 0.05<t<=0.15 / t>0.15.
    ``t_right_tilde = (T_RIGHT - mu)/sigma`` is the normalized constant-300 field.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu, sigma = data["mu_global"], data["sigma_global"]
    t_right_tilde = (T_RIGHT - mu) / (sigma + 1e-8)
    Nx, Ny, Nt = x_grid.numel(), y_grid.numel(), len(t_grid)

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)  # (1, Nx*Ny, 2)

    sq_err = 0.0
    sq_ref = 0.0
    se_K = 0.0
    n_pts = 0
    sq_err_t = np.zeros(Nt, dtype=np.float64)
    sq_ref_t = np.zeros(Nt, dtype=np.float64)
    dev_ref_t = np.zeros(Nt, dtype=np.float64)
    se_K_t = np.zeros(Nt, dtype=np.float64)
    n_pts_t = np.zeros(Nt, dtype=np.float64)
    ids = np.asarray(ids)
    with torch.no_grad():
        for start in range(0, len(ids), query_batch):
            chunk = ids[start:start + query_batch]
            u = build_ic_batch(data["trajectories"], chunk, mu, sigma, device)
            B = u.shape[0]
            coords = mesh.expand(B, -1, -1)
            pred = torch.empty((B, Nt, Nx, Ny), device=device)
            for k in range(Nt):
                tk = torch.full((B, Nx * Ny, 1), float(t_grid[k]), device=device)
                out = model(u, coords, tk)              # (B, Nx*Ny, 1)
                pred[:, k] = out[..., 0].view(B, Nx, Ny)
            truth = np.asarray(data["trajectories"][chunk], dtype=np.float32)  # (B,Nt,Nx,Ny)
            truth_t = torch.from_numpy(truth).to(device)
            truth_norm = (truth_t - mu) / (sigma + 1e-8)
            sq_err += ((pred - truth_norm) ** 2).sum().item()
            sq_ref += (truth_norm ** 2).sum().item()
            pred_K = pred * sigma + mu
            se_K += ((pred_K - truth_t) ** 2).sum().item()
            n_pts += truth_t.numel()
            if return_per_time:
                err_bt = ((pred - truth_norm) ** 2).sum(dim=(0, 2, 3))
                ref_bt = (truth_norm ** 2).sum(dim=(0, 2, 3))
                dev_bt = ((truth_norm - t_right_tilde) ** 2).sum(dim=(0, 2, 3))
                seK_bt = ((pred_K - truth_t) ** 2).sum(dim=(0, 2, 3))
                sq_err_t += err_bt.detach().cpu().numpy().astype(np.float64)
                sq_ref_t += ref_bt.detach().cpu().numpy().astype(np.float64)
                dev_ref_t += dev_bt.detach().cpu().numpy().astype(np.float64)
                se_K_t += seK_bt.detach().cpu().numpy().astype(np.float64)
                n_pts_t += float(B * Nx * Ny)

    rel_l2 = float(np.sqrt(sq_err / max(sq_ref, 1e-30)))
    rmse_K = float(np.sqrt(se_K / max(n_pts, 1)))
    out = {"val_rel_l2": rel_l2, "val_rmse_K": rmse_K}
    if return_per_time:
        per_time = np.sqrt(sq_err_t / np.maximum(sq_ref_t, 1e-30))
        per_time_rel_dev = np.sqrt(sq_err_t / np.maximum(dev_ref_t, 1e-30))
        per_time_rmse_K = np.sqrt(se_K_t / np.maximum(n_pts_t, 1.0))
        out["per_time"] = per_time
        out["per_time_rel_dev"] = per_time_rel_dev
        out["per_time_rmse_K"] = per_time_rmse_K
        out["t0_rel_l2"] = float(per_time[0])
        bands = {
            "early": t_grid <= 0.05,
            "mid": (t_grid > 0.05) & (t_grid <= 0.15),
            "late": t_grid > 0.15,
        }
        for name, mask in bands.items():
            if not mask.any():
                out[f"rel_l2_{name}"] = float("nan")
                out[f"rel_dev_{name}"] = float("nan")
                out[f"rmse_K_{name}"] = float("nan")
                continue
            out[f"rel_l2_{name}"] = float(
                np.sqrt(sq_err_t[mask].sum() / max(sq_ref_t[mask].sum(), 1e-30))
            )
            out[f"rel_dev_{name}"] = float(
                np.sqrt(sq_err_t[mask].sum() / max(dev_ref_t[mask].sum(), 1e-30))
            )
            out[f"rmse_K_{name}"] = float(
                np.sqrt(se_K_t[mask].sum() / max(n_pts_t[mask].sum(), 1.0))
            )
    return out


@torch.no_grad()
def validate_forcing_gnrmse(
    model: CViT,
    data: dict[str, Any],
    ids: np.ndarray,
    sim_params: np.ndarray,
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    t_ramp: float,
    device: torch.device,
    query_batch: int = 8,
) -> dict[str, float]:
    """Deviation-field globally-normalized RMSE over held-out forcing sims.

    The single-slab forcing field sits on a fixed 300 K baseline, so a raw
    rel-L2 denominator is dominated by that baseline (a constant-300 prediction
    scores deceptively well) and a per-sim normalized denominator explodes on
    weak forcings. This judges on the deviation field ``T - T_RIGHT`` with the
    FROZEN global ``sigma``:

        gnrmse = sqrt(mean_pts[(T_pred - T_true) ** 2]) / sigma_global

    (``T_RIGHT`` cancels in the deviation error, so this is ``rmse_K / sigma``),
    which is amplitude-fair. Per-sim results are stratified by forcing-amplitude
    tercile and by ``temporal_family`` so weak-forcing and pulse-family failures
    stay visible rather than averaged away. ``sim_params`` are the per-sim
    forcing records; each sim's encoder image is reconstructed by the SAME
    ``build_forcing_image`` / ``reconstruct_qL`` path the trainer uses.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny, Nt = x_grid.numel(), y_grid.numel(), len(t_grid)

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)
    ids = np.asarray(ids)
    per_sim_rmse: list[float] = []
    per_sim_amp: list[float] = []
    per_sim_fam: list[str] = []
    for start in range(0, len(ids), query_batch):
        chunk = ids[start:start + query_batch]
        params = [dict(sim_params[int(i)]) for i in chunk]
        u = build_forcing_image(params, y_img, t_img, a_ref, device, t_ramp)
        B = u.shape[0]
        coords = mesh.expand(B, -1, -1)
        pred = torch.empty((B, Nt, Nx, Ny), device=device)
        for k in range(Nt):
            tk = torch.full((B, Nx * Ny, 1), float(t_grid[k]), device=device)
            out = model(u, coords, tk)
            pred[:, k] = out[..., 0].view(B, Nx, Ny)
        pred_K = pred * sigma + mu
        truth = np.asarray(data["trajectories"][chunk], dtype=np.float32)
        truth_t = torch.from_numpy(truth).to(device)
        se = ((pred_K - truth_t) ** 2).sum(dim=(1, 2, 3))  # (B,)
        rmse = torch.sqrt(se / float(Nt * Nx * Ny)).detach().cpu().numpy()
        for b, p in enumerate(params):
            per_sim_rmse.append(float(rmse[b]))
            forcing = reconstruct_qL(
                p["temporal_family"], p["temporal_params"],
                p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
            )
            per_sim_amp.append(
                float(np.max(np.abs(forcing.evaluate_grid(y_img, t_img))))
            )
            per_sim_fam.append(str(p.get("temporal_family", "")))

    rmse_arr = np.asarray(per_sim_rmse, dtype=np.float64)
    amp_arr = np.asarray(per_sim_amp, dtype=np.float64)
    fam_arr = np.asarray(per_sim_fam)
    gnrmse = rmse_arr / (float(sigma) + 1e-8)

    out = {
        "val_gnrmse": float(gnrmse.mean()),
        "val_rmse_K": float(rmse_arr.mean()),
    }
    if len(amp_arr) >= 3:
        q1, q2 = np.quantile(amp_arr, [1.0 / 3.0, 2.0 / 3.0])
        strata = {
            "amp_low": amp_arr <= q1,
            "amp_mid": (amp_arr > q1) & (amp_arr <= q2),
            "amp_high": amp_arr > q2,
        }
        for name, mask in strata.items():
            out[f"gnrmse_{name}"] = (
                float(gnrmse[mask].mean()) if mask.any() else float("nan")
            )
    for fam in np.unique(fam_arr):
        out[f"gnrmse_fam_{fam}"] = float(gnrmse[fam_arr == fam].mean())
    return out


def forcing_data_loss(
    model: CViT,
    sim_params: np.ndarray,
    trajectories: np.ndarray,
    ids: np.ndarray,
    *,
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    t_ramp: float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    t_grid: np.ndarray,
    mu: float,
    sigma: float,
    n_sims: int,
    n_pts: int,
    device: torch.device,
    rng: np.random.Generator,
    gen: torch.Generator,
) -> torch.Tensor:
    """Supervised interior-field MSE against the saved FV forcing trajectories.

    Lever #1 for the diffusion_forcing gap. The pure-physics forcing objective
    couples ``q_L`` to the trainable field ONLY through the single soft left-wall
    Neumann residual, which competes with the flat-``T=300 K`` basin that
    minimizes every other term (interior residual, IC, homogeneous walls). This
    draws ``n_sims`` saved TRAIN sims, reconstructs each encoder image with the
    SAME :func:`build_forcing_image` path the physics batch uses, and matches
    ``model(u, x, y, t)`` to the FV field at ``n_pts`` grid nodes shared across
    the sim batch (normalized-space MSE). It is a plain value penalty -- no
    autodiff through the output -- so it directly pins the field level the
    derivative BC cannot, giving the reference benchmarks' value-supervised
    conditioning.
    """
    ids = np.asarray(ids)
    k = min(int(n_sims), int(len(ids)))
    chunk = rng.choice(ids, size=k, replace=False)
    params = [dict(sim_params[int(i)]) for i in chunk]
    u = build_forcing_image(params, y_img, t_img, a_ref, device, t_ramp)
    B = u.shape[0]
    Nx, Ny, Nt = int(x_grid.numel()), int(y_grid.numel()), int(len(t_grid))

    it = torch.randint(0, Nt, (n_pts,), device=device, generator=gen)
    ix = torch.randint(0, Nx, (n_pts,), device=device, generator=gen)
    iy = torch.randint(0, Ny, (n_pts,), device=device, generator=gen)
    coords = torch.stack([x_grid[ix], y_grid[iy]], dim=-1).view(1, n_pts, 2)
    coords = coords.expand(B, -1, -1)
    it_np = it.detach().cpu().numpy()
    ix_np = ix.detach().cpu().numpy()
    iy_np = iy.detach().cpu().numpy()
    t_np = np.asarray(t_grid, dtype=np.float32)[it_np]
    t = torch.from_numpy(t_np).to(device).view(1, n_pts, 1).expand(B, -1, -1)

    traj = np.asarray(trajectories[np.asarray(chunk)], dtype=np.float32)  # (B,Nt,Nx,Ny)
    truth = traj[:, it_np, ix_np, iy_np]  # (B, n_pts)
    truth = (truth - float(mu)) / (float(sigma) + 1e-8)
    truth_t = torch.from_numpy(truth).to(device).unsqueeze(-1)  # (B, n_pts, 1)

    pred = model(u, coords, t)  # (B, n_pts, 1)
    return ((pred - truth_t) ** 2).mean()


def forcing_ic_supervised_data_loss(
    model: ForcingICCViT,
    sim_params: np.ndarray,
    trajectories: np.ndarray,
    train_ids: np.ndarray,
    *,
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    t_ramp: float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    t_grid: np.ndarray,
    mu: float,
    sigma: float,
    n_sims: int,
    n_pts: int,
    query_chunk: int,
    device: torch.device,
    rng: np.random.Generator,
    gen: torch.Generator,
) -> tuple[torch.Tensor, np.ndarray]:
    """Supervised ``(T0, q_L) -> T(x,y,t)`` loss on saved training simulations."""
    ids = np.asarray(train_ids, dtype=int)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("train_ids must be a non-empty one-dimensional array")
    if n_sims <= 0 or n_sims > len(ids):
        raise ValueError(
            f"n_data_sims must be in [1, {len(ids)}], got {n_sims}."
        )
    if n_pts <= 0:
        raise ValueError("n_data_pts must be positive")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma_global must be finite and positive")
    if query_chunk < 0:
        raise ValueError("data_query_chunk must be non-negative")

    batch_ids = np.asarray(
        rng.choice(ids, size=int(n_sims), replace=False), dtype=int,
    )
    params = [dict(sim_params[int(i)]) for i in batch_ids]
    u_forcing = build_forcing_image(
        params, y_img, t_img, a_ref, device, t_ramp,
    )
    u_ic = build_ic_batch(trajectories, batch_ids, mu, sigma, device)

    B = len(batch_ids)
    Nt = int(len(t_grid))
    Nx = int(x_grid.numel())
    Ny = int(y_grid.numel())
    it = torch.randint(
        0, Nt, (B, n_pts), device=device, generator=gen,
    )
    ix = torch.randint(
        0, Nx, (B, n_pts), device=device, generator=gen,
    )
    iy = torch.randint(
        0, Ny, (B, n_pts), device=device, generator=gen,
    )
    coords = torch.stack([x_grid[ix], y_grid[iy]], dim=-1)
    t_values = torch.as_tensor(t_grid, dtype=torch.float32, device=device)
    query_t = t_values[it].unsqueeze(-1)

    trajectory_batch = np.asarray(trajectories[batch_ids], dtype=np.float32)
    it_np = it.detach().cpu().numpy()
    ix_np = ix.detach().cpu().numpy()
    iy_np = iy.detach().cpu().numpy()
    batch_np = np.arange(B, dtype=int)[:, None]
    truth = trajectory_batch[batch_np, it_np, ix_np, iy_np]
    truth = (truth - np.float32(mu)) / np.float32(sigma)
    truth_t = torch.from_numpy(np.asarray(truth, dtype=np.float32)).to(
        device,
    ).unsqueeze(-1)

    latent = model.encode(u_forcing, u_ic)
    prediction = _decode_in_chunks(
        model, latent, coords, query_t, int(query_chunk),
    )
    return (prediction - truth_t).square().mean(), batch_ids


# --------- single-seed training ---------

def build_cvit(
    config: dict,
    mu: float,
    sigma: float,
    grid_size: tuple[int, int],
    t_final: float = 1.0,
    variant: str = "cvit",
) -> CViT | InterfaceCViT | ForcingICCViT | ForcingTransitionCViT:
    """Construct the PINO surrogate. ``variant="cvit"`` (default) builds the
    diffusion :class:`CViT` conditioned on the IC field over ``grid_size =
    (Nx, Ny)``. ``variant="forcing"`` builds a :class:`ForcingCViT` whose encoder
    ingests the forcing space-time image over ``grid_size = (Ny_img, Nt_img)``;
    it reads ``model.forcing_cvit`` when present, falling back to ``model.cvit``.
    ``variant="interfaces"`` builds a multimodal :class:`InterfaceCViT` over
    ``grid_size = (Nx, Ny)`` whose three token streams (spatial field, forcing
    image, interface scalars) feed the decoder cross-attention; it
    reads ``model.interface_cvit`` when present, falling back to ``model.cvit``.
    ``variant="forcing_ic"`` builds a two-branch :class:`ForcingICCViT` over the
    IC data grid ``grid_size = (Nx, Ny)`` plus the forcing space-time image
    ``(Ny_img, Nt_img)`` read from ``training.pino.forcing``; it reads
    ``model.forcing_ic_cvit`` when present, falling back to ``model.cvit``.
    ``variant="forcing_transition"`` builds the causal transition operator over
    an arbitrary source snapshot and its source-relative forcing interval.
    """
    if variant == "forcing_transition":
        c = {
            **config["model"]["cvit"],
            **(config["model"].get("forcing_transition_cvit", {}) or {}),
        }
        forcing_cfg = (
            config.get("training", {}).get("pino", {}).get("forcing", {}) or {}
        )
        ny_img = int(
            forcing_cfg.get("ny_img")
            if forcing_cfg.get("ny_img") is not None
            else grid_size[1]
        )
        nt_img = int(
            forcing_cfg.get("nt_img")
            if forcing_cfg.get("nt_img") is not None
            else 128
        )
        t_norm = float(
            c.get("t_final", None)
            if c.get("t_final", None) is not None
            else t_final
        )
        default_time_freq = (
            c.get("fourier_freq_t")
            if c.get("fourier_freq_t") is not None
            else c.get("fourier_freq", 1.0)
        )
        return ForcingTransitionCViT(
            forcing_in_ch=int(c.get("forcing_in_ch", 3)),
            source_in_ch=int(c.get("source_in_ch", 1)),
            out_dim=int(c.get("out_dim", 1)),
            emb_dim=int(c.get("emb_dim", 256)),
            dec_emb_dim=c.get("dec_emb_dim", None),
            source_patch_size=int(
                c.get("source_patch_size", c.get("patch_size", 10))
            ),
            source_grid_size=grid_size,
            forcing_patch_size=int(c.get("forcing_patch_size", 8)),
            forcing_grid_size=(ny_img, nt_img),
            depth_enc=int(c.get("depth_enc", 4)),
            depth_dec=int(c.get("depth_dec", 2)),
            num_heads=int(c.get("num_heads", 8)),
            mlp_ratio=float(c.get("mlp_ratio", 2.0)),
            fourier_freq=float(c.get("fourier_freq", 1.0)),
            fourier_freq_source=float(
                c.get("fourier_freq_source", default_time_freq)
            ),
            fourier_freq_lead=float(
                c.get("fourier_freq_lead", default_time_freq)
            ),
            activation=str(c.get("activation", "gelu")),
            film_hidden_layers=int(c.get("film_hidden_layers", 2)),
            film_activation=str(c.get("film_activation", "silu")),
            head_hidden_layers=int(c.get("head_hidden_layers", 1)),
            head_activation=str(c.get("head_activation", "gelu")),
            query_time_conditioning=bool(
                c.get("query_time_conditioning", True)
            ),
            film_time_conditioning=bool(
                c.get("film_time_conditioning", True)
            ),
            film_init_std=float(c.get("film_init_std", 1.0e-3)),
            coord_tolerance=float(c.get("coord_tolerance", 1.0e-6)),
            t_final=t_norm,
        )
    if variant == "forcing_ic":
        c = {**config["model"]["cvit"], **(config["model"].get("forcing_ic_cvit", {}) or {})}
        forcing_cfg = config.get("training", {}).get("pino", {}).get("forcing", {}) or {}
        ny_img = int(forcing_cfg.get("ny_img") if forcing_cfg.get("ny_img") is not None else grid_size[1])
        nt_img = int(forcing_cfg.get("nt_img") if forcing_cfg.get("nt_img") is not None else 128)
        t_right_K = float(c.get("hard_right_dirichlet_t_right", T_RIGHT))
        hard_rd = bool(c.get("hard_right_dirichlet", True))
        t_right_tilde = (t_right_K - mu) / (sigma + 1e-8) if hard_rd else 0.0
        t_norm = float(
            c.get("t_final", None) if c.get("t_final", None) is not None else t_final
        )
        return ForcingICCViT(
            forcing_in_ch=int(c.get("forcing_in_ch", 1)),
            ic_in_ch=int(c.get("ic_in_ch", 1)),
            out_dim=int(c.get("out_dim", 1)),
            emb_dim=int(c.get("emb_dim", 256)),
            dec_emb_dim=c.get("dec_emb_dim", None),
            ic_patch_size=int(c.get("ic_patch_size", 10)),
            ic_grid_size=grid_size,
            forcing_patch_size=int(c.get("forcing_patch_size", 8)),
            forcing_grid_size=(ny_img, nt_img),
            depth_enc=int(c.get("depth_enc", 4)),
            depth_dec=int(c.get("depth_dec", 2)),
            num_heads=int(c.get("num_heads", 8)),
            mlp_ratio=float(c.get("mlp_ratio", 2.0)),
            fourier_freq=float(c.get("fourier_freq", 1.0)),
            fourier_freq_t=(
                None if c.get("fourier_freq_t", None) is None
                else float(c["fourier_freq_t"])
            ),
            activation=str(c.get("activation", "gelu")),
            film_hidden_layers=int(c.get("film_hidden_layers", 2)),
            film_activation=str(c.get("film_activation", "silu")),
            head_hidden_layers=int(c.get("head_hidden_layers", 1)),
            head_activation=str(c.get("head_activation", "gelu")),
            hard_right_dirichlet=hard_rd,
            t_right_tilde=t_right_tilde,
            t_final=t_norm,
        )
    if variant == "interfaces":
        interface_cfg = config["model"].get("interface_cvit", {}) or {}
        pino_cfg = config.get("training", {}).get("pino", {}) or {}
        legacy = [
            key for key in (
                "temporal_token_dim", "temporal_samples", "num_forcing_tokens",
                "forcing_hidden",
            )
            if key in interface_cfg
        ]
        if "forcing_samples" in pino_cfg:
            legacy.append("training.pino.forcing_samples")
        if legacy:
            raise ValueError(
                "InterfaceCViT no longer supports waveform-token settings "
                f"{legacy}; use the space-time image configuration and a fresh "
                "experiment name."
            )
        c = {**config["model"]["cvit"], **interface_cfg}
        forcing_cfg = config.get("training", {}).get("pino", {}).get("forcing", {}) or {}
        ny_img = int(forcing_cfg.get("ny_img") or 96)
        nt_img = int(forcing_cfg.get("nt_img") or 256)
        t_right_K = float(c.get("hard_right_dirichlet_t_right", T_RIGHT))
        hard_rd = bool(c.get("hard_right_dirichlet", True))
        t_right_tilde = (t_right_K - mu) / (sigma + 1e-8) if hard_rd else 0.0
        t_norm = float(
            c.get("t_final", None) if c.get("t_final", None) is not None else t_final
        )
        return InterfaceCViT(
            spatial_in_ch=int(c.get("spatial_in_ch", 3)),
            forcing_in_ch=int(c.get("forcing_in_ch", 1)),
            out_dim=int(c.get("out_dim", 1)),
            emb_dim=int(c.get("emb_dim", 256)),
            dec_emb_dim=c.get("dec_emb_dim", None),
            patch_size=int(c.get("patch_size", 10)),
            grid_size=grid_size,
            forcing_patch_size=int(c.get("forcing_patch_size", 8)),
            forcing_grid_size=(ny_img, nt_img),
            depth_enc=int(c.get("depth_enc", 4)),
            depth_dec=int(c.get("depth_dec", 2)),
            num_heads=int(c.get("num_heads", 8)),
            mlp_ratio=float(c.get("mlp_ratio", 2.0)),
            fourier_freq=float(c.get("fourier_freq", 1.0)),
            fourier_freq_t=(
                None if c.get("fourier_freq_t", None) is None
                else float(c["fourier_freq_t"])
            ),
            activation=str(c.get("activation", "gelu")),
            film_hidden_layers=int(c.get("film_hidden_layers", 2)),
            film_activation=str(c.get("film_activation", "silu")),
            head_hidden_layers=int(c.get("head_hidden_layers", 1)),
            head_activation=str(c.get("head_activation", "gelu")),
            hard_right_dirichlet=hard_rd,
            t_right_tilde=t_right_tilde,
            t_final=t_norm,
            n_param_scalars=int(c.get("n_param_scalars", 2)),
            num_param_tokens=int(c.get("num_param_tokens", 1)),
            param_hidden=int(c.get("param_hidden", 128)),
            jump_enrichment=bool(c.get("jump_enrichment", False)),
            jump_flux_depth=int(c.get("jump_flux_depth", 1)),
            jump_flux_conditioning=str(c.get("jump_flux_conditioning", "all")),
            jump_flux_mode=str(c.get("jump_flux_mode", "learned")),
            interface_aligned_domains=bool(
                c.get("interface_aligned_domains", False)
            ),
            interface_flux_gradient_scale=float(forcing_cfg.get("a_ref", 300.0))
            / (float(sigma) + 1.0e-8),
            k_left=K_LEFT,
            k_right=K_RIGHT,
            interface_x_range=INTERFACE_X_RANGE,
            resistance_range=RC_RANGE,
        )
    if variant == "forcing":
        c = {**config["model"]["cvit"], **config["model"].get("forcing_cvit", {})}
    else:
        c = config["model"]["cvit"]
    t_right_K = float(c.get("hard_right_dirichlet_t_right", T_RIGHT))
    hard_rd = bool(c.get("hard_right_dirichlet", True))
    t_right_tilde = (t_right_K - mu) / (sigma + 1e-8) if hard_rd else 0.0
    # Decoder time normalization horizon. Default to the data-derived t_final so
    # the temporal Fourier features live on [0, 1]; a config override wins.
    t_norm = float(c.get("t_final", None) if c.get("t_final", None) is not None else t_final)
    cls = ForcingCViT if variant == "forcing" else CViT
    return cls(
        in_ch=int(c.get("in_ch", 1)),
        out_dim=int(c.get("out_dim", 1)),
        emb_dim=int(c.get("emb_dim", 256)),
        dec_emb_dim=c.get("dec_emb_dim", None),
        patch_size=int(c.get("patch_size", 10)),
        grid_size=grid_size,
        depth_enc=int(c.get("depth_enc", 4)),
        depth_dec=int(c.get("depth_dec", 2)),
        num_heads=int(c.get("num_heads", 8)),
        mlp_ratio=float(c.get("mlp_ratio", 2.0)),
        fourier_freq=float(c.get("fourier_freq", 1.0)),
        fourier_freq_t=(
            None if c.get("fourier_freq_t", None) is None
            else float(c["fourier_freq_t"])
        ),
        activation=str(c.get("activation", "gelu")),
        film_hidden_layers=int(c.get("film_hidden_layers", 2)),
        film_activation=str(c.get("film_activation", "silu")),
        head_hidden_layers=int(c.get("head_hidden_layers", 1)),
        head_activation=str(c.get("head_activation", "gelu")),
        hard_right_dirichlet=hard_rd,
        t_right_tilde=t_right_tilde,
        t_final=t_norm,
    )


def _validate_interface_image_spec(config: dict, spec: Any) -> dict:
    if not isinstance(spec, dict):
        raise RuntimeError(
            "Incompatible InterfaceCViT checkpoint: missing interface_forcing "
            "space-time image specification. Legacy waveform-token checkpoints "
            "cannot be loaded; start a fresh experiment."
        )
    one_step_spec = isinstance(spec.get("one_step"), dict)
    expected = {
        "representation": "space_time_image",
        "version": 1,
        "forcing_schema_version": FORCING_SCHEMA_VERSION,
    }
    if not one_step_spec:
        expected.update({
            "axis_order": "channel_y_time",
            "dtype": "float32",
            "include_endpoints": True,
            "sign_convention": "positive_inward_left_flux",
            "normalization": "fixed_division",
            "clipping": False,
        })
    for key, value in expected.items():
        if spec.get(key) != value:
            raise RuntimeError(
                "Incompatible InterfaceCViT checkpoint image specification: "
                f"{key}={spec.get(key)!r}, expected {value!r}."
            )
    required = (
        ("ny_img", "nt_img", "a_ref", "ramp", "spatial_grid_size")
        if one_step_spec else
        (
            "ny_img", "nt_img", "patch_size", "y_min", "y_max", "t_min",
            "t_final", "a_ref", "ramp", "spatial_grid_size",
        )
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise RuntimeError(
            "Incomplete InterfaceCViT checkpoint image specification; missing "
            + ", ".join(missing)
            + "."
        )
    ramp = spec["ramp"]
    valid_ramp_type = isinstance(ramp, dict) and (
        one_step_spec or ramp.get("type") == "cubic_smoothstep"
    )
    if not valid_ramp_type or int(
        ramp.get("version", -1)
    ) != RAMP_SCHEMA_VERSION or "duration" not in ramp:
        raise RuntimeError(
            "Incompatible InterfaceCViT checkpoint ramp specification; expected "
            "cubic_smoothstep version 1 with a persisted duration."
        )
    ny_img, nt_img = int(spec["ny_img"]), int(spec["nt_img"])
    model_cfg = config.get("model", {}).get("interface_cvit", {}) or {}
    patch_size = int(
        model_cfg.get("forcing_patch_size", 0)
        if one_step_spec else spec["patch_size"]
    )
    if ny_img < 2 or nt_img < 2 or patch_size <= 0:
        raise RuntimeError(
            "Invalid InterfaceCViT checkpoint image dimensions or patch size; "
            f"got ({ny_img}, {nt_img}) and patch_size={patch_size}."
        )
    if ny_img % patch_size or nt_img % patch_size:
        raise RuntimeError(
            "Invalid InterfaceCViT checkpoint image dimensions: "
            f"({ny_img}, {nt_img}) is not divisible by patch_size={patch_size}."
        )
    if float(spec["a_ref"]) <= 0.0:
        raise RuntimeError("Invalid InterfaceCViT checkpoint: a_ref must be positive.")

    forcing_cfg = config.get("training", {}).get("pino", {}).get("forcing", {}) or {}
    comparisons = (
        ("training.pino.forcing.ny_img", forcing_cfg.get("ny_img"), ny_img),
        ("training.pino.forcing.nt_img", forcing_cfg.get("nt_img"), nt_img),
        ("training.pino.forcing.a_ref", forcing_cfg.get("a_ref"), float(spec["a_ref"])),
        ("model.interface_cvit.forcing_patch_size",
         model_cfg.get("forcing_patch_size"), patch_size),
        ("model.interface_cvit.forcing_in_ch", model_cfg.get("forcing_in_ch"), 1),
    )
    for name, configured, saved in comparisons:
        if configured is None or float(configured) != float(saved):
            raise RuntimeError(
                "InterfaceCViT checkpoint config/image-spec mismatch: "
                f"{name}={configured!r}, image specification requires {saved!r}."
            )
    if one_step_spec and ("patch_size" not in spec or "t_final" not in spec):
        spec = {
            **spec,
            "patch_size": patch_size,
            "t_final": float(spec.get("t_final", 1.0)),
        }
    return spec


def load_interface_cvit_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[InterfaceCViT, dict[str, Any]]:
    """Reconstruct an interface image-CViT solely from persisted checkpoint data."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("InterfaceCViT checkpoint is missing its saved config.")
    spec = _validate_interface_image_spec(config, checkpoint.get("interface_forcing"))
    grid_size_raw = spec["spatial_grid_size"]
    if not isinstance(grid_size_raw, (list, tuple)) or len(grid_size_raw) != 2:
        raise RuntimeError(
            "Invalid InterfaceCViT checkpoint spatial_grid_size; expected [Nx, Ny]."
        )
    model = build_cvit(
        config,
        float(checkpoint["mu_global"]),
        float(checkpoint["sigma_global"]),
        grid_size=(int(grid_size_raw[0]), int(grid_size_raw[1])),
        t_final=float(spec["t_final"]),
        variant="interfaces",
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), checkpoint


def run_one_seed_pino(config: dict, seed: int, run_dir: Path) -> dict[str, Any]:
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])
    t_final = float(data["t_grid"][-1])

    model = build_cvit(config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    pino = config["training"]["pino"]
    lam_r = float(pino["lambda_r"])
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    n_r = int(pino["n_r"])
    n_ic = int(pino["n_ic"])
    n_bc = int(pino["n_bc"])
    sim_batch = int(pino["sim_batch"])
    alpha = float(pino.get("alpha", 1.0))
    grad_clip_cfg = config["training"].get("grad_clip", None)
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    if grad_clip is not None and grad_clip <= 0.0:
        raise ValueError("training.grad_clip must be null or > 0")

    # IC term norm: "mse" (raw normalized MSE, legacy) or "rel" (per-sim relative
    # L2 vs the constant-300 K deviation). The relative form makes the IC anchor
    # target the same signal the validation rel-L2 measures on a near-300 K field.
    ic_loss = str(pino.get("ic_loss", "mse"))
    ic_eps = float(pino.get("ic_eps", 1.0e-6))
    t_right_tilde_ic = (T_RIGHT - mu) / (sigma + 1e-8)

    # Relative-IC denominator source. "sampled" (legacy): per-step sum over the
    # sampled IC nodes -- collapses when nodes land in the near-flat interior,
    # blowing the ratio up and corrupting the optimizer's second moment.
    # "full_grid": a FIXED per-sim signal norm D_sim = mean_{x,y}[(T0_tilde -
    # t_right_tilde)**2] precomputed on the whole grid and floored at a fraction
    # of the train-set median, so the ratio is bounded and amplitude-invariant.
    ic_denom = str(pino.get("ic_denom", "sampled"))
    ic_denom_floor_frac = float(pino.get("ic_denom_floor_frac", 0.01))
    d_sim_all: torch.Tensor | None = None
    ic_floor = 0.0
    if ic_loss == "rel" and ic_denom == "full_grid":
        T0_tilde = (
            np.asarray(data["trajectories"][:, 0, :, :], dtype=np.float64) - mu
        ) / (sigma + 1e-8)
        dev0 = T0_tilde - float(t_right_tilde_ic)
        d_np = (dev0 ** 2).reshape(dev0.shape[0], -1).mean(axis=1)   # (S,)
        _train_ids = np.asarray(data["train_ids"])
        ic_floor = ic_denom_floor_frac * float(np.median(d_np[_train_ids]))
        d_np = np.maximum(d_np, ic_floor)
        d_sim_all = torch.as_tensor(d_np, dtype=torch.float32, device=device)

    dense_ic = bool(pino.get("dense_ic", False))
    resample_every = max(1, int(pino.get("resample_every", 1)))
    causal_cfg = _resolve_forcing_causal(pino.get("causal", {}))
    causal_on = bool(causal_cfg["enabled"])
    causal_eps = float(causal_cfg["initial_eps"])
    res_n_bins = (
        int(causal_cfg.get("n_bins", 16)) if causal_on
        else int(pino.get("diag_time_bins", 16))
    )

    curr_cfg = pino.get("curriculum", {}) or {}
    gradnorm = build_gradnorm(config, lam_r, lam_ic, lam_bc)

    # IC-only fast path: when a term can never carry weight (static lambda 0 and
    # no curriculum that could ramp it up), skip its autodiff entirely. Dropping
    # the residual removes the double-backward that dominates step cost, so the
    # IC-only diagnostic runs at interactive speed.
    _curr_on_flags = bool(curr_cfg.get("enabled", False))
    compute_r = _curr_on_flags or lam_r > 0.0
    compute_bc = _curr_on_flags or lam_bc > 0.0

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_ids = np.asarray(data["train_ids"])

    metrics_path = run_dir / "train_metrics.csv"
    # Banded / t0 columns give an honest read on a near-300 K late field where the
    # plain normalized rel-L2 denominator collapses: val_t0_rel_l2 is the IC gate,
    # rel_dev_* is rel-L2 on (T-300 K), rmse_K_* is the Kelvin band error.
    band_cols = [
        f"{stem}_{band}"
        for stem in ("rel_l2", "rel_dev", "rmse_K")
        for band in ("early", "mid", "late")
    ]
    fieldnames = [
        "epoch", "loss", "loss_r", "loss_ic", "loss_bc",
        "w_r", "w_ic", "w_bc", "grad_norm_r", "grad_norm_ic", "grad_norm_bc",
        "causal_eps", "causal_action",
        "val_rel_l2", "val_rmse_K", "val_t0_rel_l2",
    ] + band_cols
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    # Per-time diagnostics (written on validation epochs only): rel-L2 at each
    # saved time exposes IC-fit-but-trajectory-drift; per-time-bin residual shows
    # whether the residual is uniformly small or concentrated at late times. The
    # rel-dev (rel-L2 on T-300 K) and RMSE_K per-time CSVs are the metric-honest
    # companions for the near-uniform late field.
    t_grid = np.asarray(data["t_grid"])
    Nt = int(t_grid.shape[0])
    relt_path = run_dir / "rel_l2_per_time.csv"
    relt_fields = ["epoch"] + [f"t{k}" for k in range(Nt)]
    with open(relt_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=relt_fields).writeheader()
    reldevt_path = run_dir / "rel_l2_dev_per_time.csv"
    with open(reldevt_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=relt_fields).writeheader()
    rmseKt_path = run_dir / "rmse_K_per_time.csv"
    with open(rmseKt_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=relt_fields).writeheader()
    resbin_path = run_dir / "residual_per_time_bin.csv"
    resbin_fields = ["epoch"] + [f"bin{b}" for b in range(res_n_bins)]
    with open(resbin_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=resbin_fields).writeheader()

    best_val = float("inf")
    history: list[dict[str, float]] = []

    curr_on = bool(curr_cfg.get("enabled", False))
    if curr_on and str(curr_cfg.get("mode", "ic_only")) == "ic_heavy":
        curr_desc = (
            f"ic_heavy warm={int(curr_cfg.get('warmup_epochs', 0))} "
            f"decay={int(curr_cfg.get('decay_epochs', 1))} "
            f"ic:{lam_ic}->{float(curr_cfg.get('lambda_ic_final', lam_ic))}"
        )
    elif curr_on:
        curr_desc = (
            f"ic_only={int(curr_cfg.get('ic_only_epochs', 0))} "
            f"ramp={int(curr_cfg.get('ramp_epochs', 1))}"
        )
    else:
        curr_desc = "off"
    gn_desc = (
        f"on(terms={gradnorm.term_names})" if gradnorm is not None else "off"
    )
    causal_desc = (
        f"on(n_bins={res_n_bins},eps={causal_eps})"
        if causal_on else "off"
    )
    print(
        f"[pino] seed={seed} device={device} epochs={epochs} "
        f"validate_every={validate_every} | grid={Nx}x{Ny} t_final={t_final:.4f} | "
        f"lambda_r={lam_r} lambda_ic={lam_ic} lambda_bc={lam_bc} | "
        f"n_r={n_r} n_ic={n_ic} n_bc={n_bc} sim_batch={sim_batch} alpha={alpha} | "
        f"ic_loss={ic_loss} ic_denom={ic_denom}"
        + (f"(floor={ic_floor:.3e})" if d_sim_all is not None else "")
        + f" dense_ic={dense_ic} resample_every={resample_every} | "
        f"compute_r={compute_r} compute_bc={compute_bc} | "
        f"curriculum={curr_desc} gradnorm={gn_desc} causal={causal_desc}",
        flush=True,
    )

    coll = None
    for epoch in range(epochs):
        model.train()
        batch_ids = rng.choice(train_ids, size=min(sim_batch, len(train_ids)), replace=False)
        u = build_ic_batch(data["trajectories"], batch_ids, mu, sigma, device)

        # Slower warm-up resampling: redraw the interior/wall collocation only
        # every ``resample_every`` steps. Per-step resampling can keep the IC loss
        # from converging (Chen et al.); reusing the collocation set lets the
        # anchor settle. IC targets are re-read every step (the sim minibatch
        # rotates); resample_every=1 restores per-step sampling (legacy).
        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        if coll is None or (epoch % resample_every == 0):
            coll = sample_collocation(
                n_r, n_ic, n_bc, t_final, x_grid_t, y_grid_t, device, gen,
                dense_ic=dense_ic,
            )
        ic_target = _ic_targets(
            data["trajectories"], batch_ids, coll["ic"]["ix"], coll["ic"]["iy"], mu, sigma, device,
        )

        ic_den = None
        if d_sim_all is not None:
            ic_den = d_sim_all[torch.as_tensor(batch_ids, device=device)]

        optimizer.zero_grad(set_to_none=True)
        eps_used = causal_eps
        causal_step_cfg = {**causal_cfg, "current_eps": eps_used}
        losses = pino_losses(
            model, u, coll, ic_target, alpha,
            causal_cfg=causal_step_cfg, t_final=t_final,
            res_bins=(res_n_bins if do_val else 0),
            ic_loss=ic_loss, t_right_tilde=t_right_tilde_ic, ic_eps=ic_eps,
            ic_den=ic_den, compute_r=compute_r, compute_bc=compute_bc,
        )

        # Static per-term weights for this epoch (IC-first curriculum or plain
        # lambda_*), then GradNorm multipliers on the RAW magnitudes. Only terms
        # with a positive static weight are measured; a zero-weighted term (IC-only
        # phase) is neither balanced nor added to the total.
        w = {"r": 0.0, "ic": 0.0, "bc": 0.0}
        w["r"], w["ic"], w["bc"] = _curriculum_weights(
            epoch, lam_r, lam_ic, lam_bc, curr_cfg,
        )
        if gradnorm is not None:
            active = {k: losses[k] for k in ("r", "ic", "bc") if w[k] > 0.0}
            mults = gradnorm.maybe_update(active, model.parameters())
        else:
            mults = {}
        w_eff = {k: w[k] * float(mults.get(k, 1.0)) for k in ("r", "ic", "bc")}

        # Per-term gradient norms (diagnostic; val epochs only to bound the extra
        # backward passes). Measured on the live graph before the combined backward.
        gnorms: dict[str, float] = {}
        if do_val:
            active_terms = {k: losses[k] for k in ("r", "ic", "bc") if w[k] > 0.0}
            if active_terms:
                gnorms = _term_grad_norms(active_terms, model.parameters())

        loss = (
            w_eff["r"] * losses["r"]
            + w_eff["ic"] * losses["ic"]
            + w_eff["bc"] * losses["bc"]
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite PINO loss at epoch {epoch}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(f"Non-finite PINO gradient at epoch {epoch}")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        causal_action = "off"
        if causal_on and "causal_weights" in losses:
            causal_eps, causal_action = _adapt_causal_eps(
                causal_eps,
                losses["causal_weights"],
                causal_cfg,
                populated=bool(losses["causal_populated"]),
            )

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_r": float(losses["r"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_bc": float(losses["bc"].detach().cpu()),
            "w_r": w_eff["r"],
            "w_ic": w_eff["ic"],
            "w_bc": w_eff["bc"],
            "grad_norm_r": gnorms.get("r", ""),
            "grad_norm_ic": gnorms.get("ic", ""),
            "grad_norm_bc": gnorms.get("bc", ""),
            "causal_eps": eps_used if causal_on else "",
            "causal_action": causal_action,
            "val_rel_l2": "",
            "val_rmse_K": "",
            "val_t0_rel_l2": "",
            **{c: "" for c in band_cols},
        }

        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(r={row['loss_r']:.6f}, ic={row['loss_ic']:.6f}, bc={row['loss_bc']:.6f}) "
            f"w=({w_eff['r']:.3g},{w_eff['ic']:.3g},{w_eff['bc']:.3g}) "
            f"lr={lr:.2e}",
            flush=True,
        )

        if do_val:
            model.eval()
            val = validate_rel_l2(
                model, data, data["val_ids"], device, return_per_time=True,
            )
            row["val_rel_l2"] = val["val_rel_l2"]
            row["val_rmse_K"] = val["val_rmse_K"]
            row["val_t0_rel_l2"] = val["t0_rel_l2"]
            for c in band_cols:
                row[c] = val[c]

            # per-time rel-L2 (normalized), rel-L2 on (T-300 K), RMSE_K, and
            # per-time-bin residual diagnostics
            with open(relt_path, "a", newline="") as f:
                rr = {"epoch": epoch}
                rr.update({f"t{k}": float(val["per_time"][k]) for k in range(Nt)})
                csv.DictWriter(f, fieldnames=relt_fields).writerow(rr)
            with open(reldevt_path, "a", newline="") as f:
                rr = {"epoch": epoch}
                rr.update({f"t{k}": float(val["per_time_rel_dev"][k]) for k in range(Nt)})
                csv.DictWriter(f, fieldnames=relt_fields).writerow(rr)
            with open(rmseKt_path, "a", newline="") as f:
                rr = {"epoch": epoch}
                rr.update({f"t{k}": float(val["per_time_rmse_K"][k]) for k in range(Nt)})
                csv.DictWriter(f, fieldnames=relt_fields).writerow(rr)
            if "res_bins" in losses:
                rb = losses["res_bins"].detach().cpu().numpy()
                with open(resbin_path, "a", newline="") as f:
                    rr = {"epoch": epoch}
                    rr.update({f"bin{b}": float(rb[b]) for b in range(res_n_bins)})
                    csv.DictWriter(f, fieldnames=resbin_fields).writerow(rr)
            if gnorms:
                gn_txt = " ".join(f"{k}={gnorms[k]:.3e}" for k in ("r", "ic", "bc") if k in gnorms)
                print(f"  grad_norms: {gn_txt}", flush=True)
            is_best = val["val_rel_l2"] < best_val
            if is_best:
                best_val = val["val_rel_l2"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "mu_global": mu,
                        "sigma_global": sigma,
                        "config": config,
                        "epoch": epoch,
                        "best_val": best_val,
                        "gradnorm_state": (
                            gradnorm.state_dict() if gradnorm is not None else None
                        ),
                    },
                    run_dir / "cvit_best.pt",
                )
            print(
                f"Validation for epoch {epoch}: "
                f"val_rel_l2={val['val_rel_l2'] * 100:.4f}% "
                f"val_rmse_K={val['val_rmse_K']:.4f}K "
                f"(best={best_val * 100:.4f}%)"
                + ("  [new best -> cvit_best.pt]" if is_best else ""),
                flush=True,
            )
            print(
                f"  t0_rel_l2={val['t0_rel_l2'] * 100:.2f}%  "
                f"rel_l2[e/m/l]={val['rel_l2_early'] * 100:.1f}/"
                f"{val['rel_l2_mid'] * 100:.1f}/{val['rel_l2_late'] * 100:.1f}%  "
                f"rel_dev[e/m/l]={val['rel_dev_early'] * 100:.1f}/"
                f"{val['rel_dev_mid'] * 100:.1f}/{val['rel_dev_late'] * 100:.1f}%  "
                f"rmse_K[e/m/l]={val['rmse_K_early']:.3f}/"
                f"{val['rmse_K_mid']:.3f}/{val['rmse_K_late']:.3f}K",
                flush=True,
            )

        history.append({k: (v if v != "" else None) for k, v in row.items()})
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    summary = {"seed": seed, "best_val_rel_l2": best_val, "epochs": epochs}
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _forcing_warmup_config(fcfg: dict) -> dict[str, Any]:
    cfg = dict(fcfg.get("warmup", {}) or {})
    steps = cfg.get("steps")
    if steps is None:
        steps = cfg.get("epochs", 0)
    return {
        "steps": max(0, int(steps)),
        "resample_every": max(1, int(cfg.get("resample_every", 1))),
        "r_mult": float(cfg.get("r_mult", 1.0)),
        "ic_mult": float(cfg.get("ic_mult", 1.0)),
        "bc_left_mult": float(cfg.get("bc_left_mult", 1.0)),
        "bc_hom_mult": float(cfg.get("bc_hom_mult", 1.0)),
    }


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _atomic_hashed_torch_save(payload: dict, path: Path) -> str:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = _sha256_file(tmp)
    if _sha256_file(tmp) != digest:
        tmp.unlink(missing_ok=True)
        raise OSError(f"hash verification failed while writing {path}")
    os.replace(tmp, path)
    _fsync_directory(path.parent)
    _atomic_text(digest + "\n", path.with_name(path.name + ".sha256"))
    return digest


def _load_hashed_calibration(
    path: Path, expected_metadata: dict[str, Any],
) -> tuple[dict, str]:
    hash_path = path.with_name(path.name + ".sha256")
    if not path.exists() or not hash_path.exists():
        raise FileNotFoundError("calibration artifact or SHA-256 sidecar is missing")
    expected_hash = hash_path.read_text().strip()
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError("region-standardization calibration SHA-256 mismatch")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    saved_metadata = artifact.get("metadata", {})
    saved_semantics = copy.deepcopy(saved_metadata)
    expected_semantics = copy.deepcopy(expected_metadata)
    saved_environment = saved_semantics.pop("environment", {})
    current_environment = expected_semantics.pop("environment", {})
    if saved_semantics != expected_semantics:
        raise ValueError("region-standardization calibration metadata mismatch")
    _validate_environment_fingerprint(saved_environment, current_environment)
    scales = artifact.get("std_scales", {})
    if set(scales) != set(expected_metadata.get("balanced_terms", scales)):
        raise ValueError("calibration scale terms do not match the active objective")
    if not scales or not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in scales.values()
    ):
        raise ValueError("calibration scales must all be finite and positive")
    return artifact, actual_hash


def _atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _pack_tensors(value):
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": value.detach().cpu(),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, dict):
        return {k: _pack_tensors(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_pack_tensors(v) for v in value]}
    if isinstance(value, list):
        return [_pack_tensors(v) for v in value]
    return copy.deepcopy(value)


def _unpack_tensors(value, device):
    if isinstance(value, dict) and "__tensor__" in value:
        return value["__tensor__"].to(device).detach().requires_grad_(value["requires_grad"])
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_unpack_tensors(v, device) for v in value["__tuple__"])
    if isinstance(value, dict):
        return {k: _unpack_tensors(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack_tensors(v, device) for v in value]
    return copy.deepcopy(value)


def _capture_forcing_rng(rng, gen, eval_gen) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_local": copy.deepcopy(rng.bit_generator.state),
        "torch_cpu": torch.get_rng_state(),
        "train_generator": gen.get_state(),
        "eval_generator": eval_gen.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_forcing_rng(state, rng, gen, eval_gen) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    rng.bit_generator.state = state["numpy_local"]
    torch.set_rng_state(state["torch_cpu"])
    gen.set_state(state["train_generator"])
    eval_gen.set_state(state["eval_generator"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _forcing_resume_config(config: dict, resolved_fv_dt: float | None = None) -> dict:
    normalized = copy.deepcopy(config)
    training = normalized.setdefault("training", {})
    pino = training.setdefault("pino", {})
    forcing = pino.setdefault("forcing", {})
    causal = _resolve_forcing_causal(pino.get("causal", {}))
    pino["causal"] = causal
    residual_method = str(pino.get("residual_method", "autodiff"))
    if residual_method == "autodiff":
        pino.pop("residual_method", None)
    if resolved_fv_dt is not None:
        pino["residual_method"] = "finite_volume"
        pino["dt"] = float(resolved_fv_dt)
    forcing["warmup"] = _forcing_warmup_config(forcing)
    for key in ("epochs", "validate_every", "run"):
        training.pop(key, None)
    for key in ("save_latest_every", "extend_completed"):
        forcing.pop(key, None)
    normalized.pop("experiment", None)
    normalized.pop("config_id", None)
    paths = normalized.get("paths")
    if isinstance(paths, dict):
        paths.pop("runs_root", None)
    return normalized


def _reconcile_forcing_metrics(path: Path, fieldnames: list[str], last_update: int) -> None:
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if int(r.get("completed_updates") or 0) <= last_update]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def run_one_seed_forcing_pino(config: dict, seed: int, run_dir: Path) -> dict[str, Any]:
    """Physics-only training of a :class:`ForcingCViT` on the single-slab forcing
    benchmark. Run-0 recipe: static Adam weights, ONLINE forcing sampling, LHS
    collocation, inhomogeneous left-wall Neumann forcing, and a fixed 300 K IC
    anchor. No GradNorm / causal / SOAP (layer those in later, one per run).

    The encoder conditions on the forcing rendered as a ``(Ny_img, Nt_img)``
    space-time image. Each step samples FRESH forcing params and FRESH
    collocation, so training is not tied to a finite saved set (Chen et al.
    arXiv:2606.06164). Saved FV trajectories + ``sim_params`` are used for
    VALIDATION only, judged by the deviation-field gnRMSE with a frozen global
    ``sigma`` (see :func:`validate_forcing_gnrmse`).
    """
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "cvit_latest.pt"
    best_path = run_dir / "cvit_best.pt"
    final_path = run_dir / "cvit_final.pt"
    complete_path = run_dir / "RUN_COMPLETE"

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])
    t_final = float(data["t_grid"][-1])
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    c_dom, d_dom = float(y_grid_np[0]), float(y_grid_np[-1])

    # Saved FV forcing records are VALIDATION-only. sim_params.npy lives beside
    # trajectories.npy (same directory the data generator writes to).
    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(str(sp_path), allow_pickle=True)

    pino = config["training"]["pino"]
    fcfg = pino.get("forcing", {}) or {}
    extend_completed = bool(fcfg.get("extend_completed", False))
    if complete_path.exists() and not extend_completed:
        summary_path = run_dir / "final_metrics.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return {"seed": seed, "status": "complete", "run_dir": str(run_dir)}
    resuming = latest_path.exists() and (not complete_path.exists() or extend_completed)
    causal_cfg = _resolve_forcing_causal(pino.get("causal", {}))
    warmup = _forcing_warmup_config(fcfg)
    save_latest_every = max(1, int(fcfg.get("save_latest_every", 25)))
    grad_clip_cfg = fcfg.get("grad_clip", None)
    if grad_clip_cfg is None:
        grad_clip_cfg = config["training"].get("grad_clip", None)
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    if grad_clip is not None and grad_clip <= 0.0:
        raise ValueError("training.pino.forcing.grad_clip must be null or > 0")
    # `null` in YAML resolves to a dynamic default here (Ny/A_AMP_REF/t_final are
    # not knowable statically), so coalesce None rather than trusting .get's
    # absent-key fallback.
    a_ref = float(fcfg.get("a_ref") if fcfg.get("a_ref") is not None else A_AMP_REF)
    ny_img = int(fcfg.get("ny_img") if fcfg.get("ny_img") is not None else Ny)
    nt_img = int(fcfg.get("nt_img") if fcfg.get("nt_img") is not None else 128)
    # Forcing image axes: rows = left-wall y-nodes, cols = time over [0, t_final].
    y_img = np.linspace(c_dom, d_dom, ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, t_final, nt_img, dtype=np.float64)

    # Frozen startup ramp: pin an ABSOLUTE constant shared by the online training
    # q_L AND the FV-baked validation q_L so early-time forcing matches exactly.
    # Config override wins; else reuse the ramp stored with the dataset; else fall
    # back to the dt-derived default.
    ramp_cfg = fcfg.get("ramp_seconds", None)
    if ramp_cfg is not None:
        t_ramp = float(ramp_cfg)
    else:
        t_ramp = load_ramp_seconds(config["data"]["t_grid_path"])
        if t_ramp is None:
            dt = load_solver_dt(config["data"]["t_grid_path"])
            t_ramp = default_ramp_seconds(dt if dt is not None else t_final / 100.0)

    temporal_window = dict(
        t_on=float(fcfg.get("t_on", 0.0)),
        t_off=float(fcfg.get("t_off", 0.2)),
        phase=float(fcfg.get("phase", 0.0)),
        tukey_alpha=float(fcfg.get("tukey_alpha", 0.5)),
    )
    # None -> draw all families (diffusion_forcing); a family string pins the
    # online sampler (diffusion_forcing_single pins sin/uniform).
    tf_fix = fcfg.get("temporal_family")
    sf_fix = fcfg.get("spatial_family")
    dt_sample = (
        float(fcfg["dt_sample"])
        if fcfg.get("dt_sample") is not None
        else t_final / max(nt_img - 1, 1)
    )
    sampler = str(fcfg.get("collocation") or "lhs")

    # Static, diagnostic-informed collocation biasing (implicit residual
    # reweighting of the interior points). Disabled by default -> coll_bias is
    # None, so sample_collocation runs the legacy RNG-identical interior draw.
    # The helper validates eagerly (raises before the training loop).
    coll_bias = _resolve_collocation_bias(pino)

    model = build_cvit(
        config, mu, sigma, grid_size=(ny_img, nt_img), t_final=t_final,
        variant="forcing",
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    lam_r = float(pino["lambda_r"])
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    # Optional per-term forcing-wall weight. None (default) keeps the legacy
    # single bc bucket (left+top+bottom averaged, weighted by lam_bc). When set,
    # the forcing left wall is pulled out and weighted by lam_bc_left, while the
    # homogeneous top/bottom walls stay on lam_bc via the bc_hom bucket.
    lam_bc_left_cfg = pino.get("lambda_bc_left", None)
    lam_bc_left = None if lam_bc_left_cfg is None else float(lam_bc_left_cfg)
    # Lever #1: optional supervised interior-field data term drawn from the saved
    # TRAIN forcing sims. 0.0 (default) = pure physics (off).
    lam_data = float(pino.get("lambda_data", 0.0))
    n_data_sims = int(pino.get("n_data_sims", 8))
    n_data_pts = int(pino.get("n_data_pts", 1024))
    t_grid_np = np.asarray(data["t_grid"], dtype=np.float64)
    n_r = int(pino["n_r"])
    n_ic = int(pino["n_ic"])
    n_bc = int(pino["n_bc"])
    sim_batch = int(pino["sim_batch"])
    alpha = float(pino.get("alpha", 1.0))
    # Fixed uniform 300 K IC in normalized space (mu ~ 300 -> ~0).
    t_right_tilde_ic = (T_RIGHT - mu) / (sigma + 1e-8)

    # GradNorm balances the RAW per-term gradient magnitudes so the forcing
    # left-wall residual is not swamped by the interior PDE gradient (SOAP only
    # sees the combined gradient). The term set matches the loss composition: the
    # split {r, ic, bc_left, bc_hom} isolates the forcing wall from the
    # near-satisfied adiabatic walls; the legacy single-bucket set is {r, ic, bc}.
    # None (disabled) is an exact no-op (multipliers == 1). Guardrails come from
    # config["training"]["gradnorm"] (w_min / w_max / floor).
    if lam_bc_left is None:
        gn_term_weights = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
        gn_cos_pairs = None
    else:
        gn_term_weights = {
            "r": lam_r, "ic": lam_ic, "bc_left": lam_bc_left, "bc_hom": lam_bc,
        }
        gn_cos_pairs = [("r", "bc_left"), ("ic", "bc_left"), ("bc_hom", "bc_left")]
    gradnorm = build_gradnorm(config, term_weights=gn_term_weights)

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed + 1_000_003)
    rng = np.random.default_rng(seed)

    metrics_path = run_dir / "train_metrics.csv"
    # loss_bc_left is logged as its OWN column: the left wall is the forcing
    # signal and must be watchable independently of the homogeneous top/bottom.
    # GradNorm diagnostics keep the existing w_* columns at their STATIC meaning
    # and add explicit new columns for the split term set {r, ic, bc_left, bc_hom}:
    # gn_mult_* (multiplier), w_eff_* (static x multiplier == what SOAP sees),
    # grad_norm_* (raw per-term gradient norm) and grad_norm_eff_* (effective).
    # These + JSON supplements are populated on validation epochs only.
    gn_cols = ["r", "ic", "bc_left", "bc_hom"]
    causal_cols = range(causal_cfg["n_bins"])
    fieldnames = [
        "epoch", "completed_updates", "lr_first", "lr_last",
        "loss", "loss_r", "loss_ic", "loss_bc", "loss_bc_left",
        "loss_data",
        "w_r", "w_ic", "w_bc", "w_bc_left", "w_data",
        *[f"warm_mult_{c}" for c in gn_cols],
        "r_pointwise_mse", "r_equal_bin_mean", "r_causal_loss",
        "causal_reduction_ratio", "causal_eps", "causal_eps_next",
        "causal_mean_weight", "causal_last_weight", "causal_log_last_weight",
        "causal_adaptation_action",
        *[f"causal_loss_bin_{i:02d}" for i in causal_cols],
        *[f"causal_weight_bin_{i:02d}" for i in causal_cols],
        *[f"causal_count_bin_{i:02d}" for i in causal_cols],
        "val_gnrmse", "val_rmse_K",
        "gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high",
        *[f"gn_mult_{c}" for c in gn_cols],
        *[f"w_eff_{c}" for c in gn_cols],
        *[f"grad_norm_{c}" for c in gn_cols],
        *[f"grad_norm_eff_{c}" for c in gn_cols],
        "gradnorm_mean_mult", "gradnorm_ms",
        "gradnorm_weights", "grad_cosines", "gradnorm_bound_hits",
    ]

    start_epoch = 0
    completed_updates = 0
    last_csv_update = 0
    best_val = float("inf")
    causal_eps = float(causal_cfg["initial_eps"])
    causal_updates = 0
    causal_calibrated = False
    coll = None
    params_batch: list[dict] | None = None

    if resuming:
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        saved_resume = ckpt.get("resume_config")
        current_resume = _forcing_resume_config(config)
        if saved_resume != current_resume:
            raise ValueError(
                "Incompatible diffusion_forcing resume configuration; use a fresh run directory."
            )
        saved_epochs = int(ckpt["config"]["training"]["epochs"])
        if epochs < int(ckpt["next_epoch"]):
            raise ValueError("training.epochs is below the checkpoint next_epoch")
        if epochs != saved_epochs:
            sched_type = str(config["training"]["scheduler"]["type"])
            if sched_type not in {"PICViTExponential", "StepLR"}:
                raise ValueError("Extending epochs requires a horizon-independent scheduler")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if gradnorm is not None:
            if ckpt.get("gradnorm_state") is None:
                raise ValueError("Resume checkpoint is missing GradNorm state")
            gradnorm.load_state_dict(ckpt["gradnorm_state"])
        start_epoch = int(ckpt["next_epoch"])
        completed_updates = int(ckpt["completed_updates"])
        last_csv_update = int(ckpt["last_csv_update"])
        best_val = float(ckpt["best_val"])
        causal_state = ckpt.get("causal_state") or {}
        if causal_cfg["enabled"]:
            if int(causal_state.get("n_bins", -1)) != causal_cfg["n_bins"]:
                raise ValueError("Resume causal n_bins does not match the active configuration")
            causal_eps = float(causal_state["eps"])
            causal_updates = int(causal_state.get("updates", 0))
            causal_calibrated = bool(causal_state.get("calibrated", False))
        cache = ckpt.get("forcing_cache") or {}
        params_batch = copy.deepcopy(cache.get("params_batch"))
        coll = _unpack_tensors(cache.get("coll"), device) if cache.get("coll") else None
        _restore_forcing_rng(ckpt["rng_state"], rng, gen, eval_gen)
        _reconcile_forcing_metrics(metrics_path, fieldnames, last_csv_update)
        if complete_path.exists():
            complete_path.unlink()
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()

    print(
        f"[pino-forcing] seed={seed} device={device} epochs={epochs} "
        f"validate_every={validate_every} | grid={Nx}x{Ny} t_final={t_final:.4f} | "
        f"img=({ny_img}x{nt_img}) a_ref={a_ref} t_ramp={t_ramp:.4g} "
        f"sampler={sampler} coll_bias={coll_bias} | lambda_r={lam_r} lambda_ic={lam_ic} "
        f"lambda_bc={lam_bc} lambda_bc_left={lam_bc_left} "
        f"lambda_data={lam_data} (n_data_sims={n_data_sims},n_data_pts={n_data_pts}) "
        f"n_r={n_r} n_ic={n_ic} n_bc={n_bc} "
        f"sim_batch={sim_batch} alpha={alpha} k_slab={K_SLAB} | "
        f"causal={causal_cfg} | warmup={warmup} grad_clip={grad_clip} "
        f"save_latest_every={save_latest_every} resume={resuming}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs):
        model.train()
        warming = completed_updates < warmup["steps"]
        # Computed up front: the cosine diagnostic runs on the training step that
        # coincides with a validation epoch, before backward.
        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        re = warmup["resample_every"] if warming else 1
        # Online resampling: fresh forcing batch + fresh collocation. During
        # warm-up they are held for ``re`` steps to let the IC anchor settle.
        if params_batch is None or (completed_updates % re == 0):
            params_batch = sample_forcing_params(
                rng, sim_batch, dt_sample, t_final,
                c=c_dom, d=d_dom, temporal_window=temporal_window,
                temporal_family=tf_fix, spatial_family=sf_fix,
            )
            coll = sample_collocation(
                n_r, n_ic, n_bc, t_final, x_grid_t, y_grid_t, device, gen,
                sampler=sampler, bias=coll_bias,
            )

        u = build_forcing_image(params_batch, y_img, t_img, a_ref, device, t_ramp)
        B = u.shape[0]
        ic_target = torch.full(
            (B, coll["ic"]["coords"].shape[1], 1), float(t_right_tilde_ic),
            device=device,
        )
        _, yw_left, tw_left = coll["walls"]["left"]
        left_qL = left_wall_qL(params_batch, yw_left, tw_left, device, t_ramp)

        optimizer.zero_grad(set_to_none=True)
        causal_step_cfg = {**causal_cfg, "current_eps": causal_eps}
        losses = pino_losses(
            model, u, coll, ic_target, alpha,
            causal_cfg=causal_step_cfg,
            t_final=t_final, ic_loss="mse", t_right_tilde=t_right_tilde_ic,
            left_qL=left_qL, sigma=float(sigma), k_slab=K_SLAB,
        )
        warm_mult = {
            "r": warmup["r_mult"] if warming else 1.0,
            "ic": warmup["ic_mult"] if warming else 1.0,
            "bc_left": warmup["bc_left_mult"] if warming else 1.0,
            "bc_hom": warmup["bc_hom_mult"] if warming else 1.0,
            "bc": warmup["bc_hom_mult"] if warming else 1.0,
        }
        # Static per-term weights ``sw`` (name -> lambda). Dict order preserves the
        # ORIGINAL left-to-right summation so the disabled path (gn_mults empty,
        # multiplier == 1.0) reproduces the previous ``loss`` exactly.
        if lam_bc_left is None:
            w_bc = lam_bc
            w_bc_left = lam_bc  # reported effective weight; left sits inside bc
            sw = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
        else:
            # Split BC: homogeneous top/bottom on lam_bc, forcing left on its own
            # weight so the forcing residual is not diluted by the 1/len(WALLS)
            # bucket average.
            w_bc = lam_bc
            w_bc_left = lam_bc_left
            sw = {"r": lam_r, "ic": lam_ic, "bc_hom": lam_bc, "bc_left": lam_bc_left}

        # GradNorm rebalances the RAW per-term gradient magnitudes BEFORE backward
        # (one extra autograd.grad per active term every ``update_every`` steps).
        # ``w_eff[k] = sw[k] * m_k`` -- static lambda priorities are preserved; the
        # multiplier equalizes the raw magnitudes SOAP sees. Disabled -> m_k == 1.
        gn_mults: dict[str, float] = {}
        gradnorm_ms: float | str = ""
        if gradnorm is not None:
            active = {k: losses[k] for k in gradnorm.term_names if k in losses}
            gn_params = [p for p in model.parameters() if p.requires_grad]
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            gn_mults = gradnorm.maybe_update(active, gn_params, dist_info=None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gradnorm_ms = (time.perf_counter() - _t0) * 1000.0
        # Cosine diagnostic on the decoder subset, only on the training step that
        # coincides with a validation epoch, on the LIVE graph before backward.
        grad_cosines: dict[str, float | None] | None = None
        if do_val and gradnorm is not None:
            grad_cosines = _term_grad_cosines(
                {k: losses[k] for k in sw}, model.decoder.parameters(),
                pairs=gn_cos_pairs,
            )
        w_eff = {
            k: sw[k] * float(gn_mults.get(k, 1.0)) * warm_mult[k] for k in sw
        }
        loss = None
        for k, wk in w_eff.items():
            term = wk * losses[k]
            loss = term if loss is None else loss + term

        loss_data = None
        if lam_data > 0.0:
            loss_data = forcing_data_loss(
                model, sim_params, data["trajectories"], data["train_ids"],
                y_img=y_img, t_img=t_img, a_ref=a_ref, t_ramp=t_ramp,
                x_grid=x_grid_t, y_grid=y_grid_t, t_grid=t_grid_np,
                mu=mu, sigma=sigma, n_sims=n_data_sims, n_pts=n_data_pts,
                device=device, rng=rng, gen=gen,
            )
            loss = loss + lam_data * loss_data
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Non-finite forcing PINO loss at epoch {epoch}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(f"Non-finite forcing PINO gradient at epoch {epoch}")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        completed_updates += 1

        eps_used = causal_eps
        causal_action = "off"
        if causal_cfg["enabled"]:
            populated = bool(losses["causal_populated"])
            causal_eps, causal_action = _adapt_causal_eps(
                causal_eps, losses["causal_weights"], causal_cfg, populated=populated,
            )
            if populated:
                causal_updates += 1
                if not causal_calibrated:
                    calibration = {}
                    bin_losses = losses["causal_bin_losses"]
                    for candidate in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                        cw = _causal_weights(bin_losses, candidate)
                        calibration[str(candidate)] = {
                            "mean": float(cw.mean().item()), "last": float(cw[-1].item()),
                        }
                    print(f"  causal epsilon calibration: {json.dumps(calibration)}", flush=True)
                    causal_calibrated = True

        row: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "lr_first": lr,
            "lr_last": lr,
            "loss": float(loss.detach().cpu()),
            "loss_r": float(losses["r"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_bc": float(losses["bc"].detach().cpu()),
            "loss_bc_left": float(losses["bc_left"].detach().cpu()),
            "loss_data": (
                float(loss_data.detach().cpu()) if loss_data is not None else ""
            ),
            "w_r": lam_r * warm_mult["r"],
            "w_ic": lam_ic * warm_mult["ic"],
            "w_bc": w_bc * warm_mult["bc_hom"],
            "w_bc_left": w_bc_left * warm_mult["bc_left"],
            "w_data": lam_data,
            "val_gnrmse": "", "val_rmse_K": "",
            "gnrmse_amp_low": "", "gnrmse_amp_mid": "", "gnrmse_amp_high": "",
        }
        for key in gn_cols:
            row[f"warm_mult_{key}"] = warm_mult[key]
        row["r_pointwise_mse"] = float(losses["r_pointwise_mse"].cpu())
        if causal_cfg["enabled"]:
            bins = losses["causal_bin_losses"].cpu()
            weights = losses["causal_weights"].cpu()
            counts = losses["causal_bin_counts"].cpu()
            row.update({
                "r_equal_bin_mean": float(losses["r_equal_bin_mean"].cpu()),
                "r_causal_loss": float(losses["r_causal_loss"].cpu()),
                "causal_reduction_ratio": float(losses["causal_reduction_ratio"].cpu()),
                "causal_eps": eps_used,
                "causal_eps_next": causal_eps,
                "causal_mean_weight": float(weights.mean()),
                "causal_last_weight": float(weights[-1]),
                "causal_log_last_weight": -eps_used * float(bins[:-1].sum()),
                "causal_adaptation_action": causal_action,
            })
            for i in causal_cols:
                row[f"causal_loss_bin_{i:02d}"] = float(bins[i])
                row[f"causal_weight_bin_{i:02d}"] = float(weights[i])
                row[f"causal_count_bin_{i:02d}"] = float(counts[i])
        data_txt = (
            f", data={row['loss_data']:.6f}" if loss_data is not None else ""
        )
        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(r={row['loss_r']:.6f}, ic={row['loss_ic']:.6f}, "
            f"bc={row['loss_bc']:.6f}, bc_left={row['loss_bc_left']:.6f}"
            f"{data_txt}) "
            f"lr={lr:.2e}" + ("  [warmup]" if warming else ""),
            flush=True,
        )

        if do_val:
            model.eval()
            val = validate_forcing_gnrmse(
                model, data, data["val_ids"], sim_params,
                y_img, t_img, a_ref, t_ramp, device,
            )
            row["val_gnrmse"] = val["val_gnrmse"]
            row["val_rmse_K"] = val["val_rmse_K"]
            for c in ("gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high"):
                row[c] = val.get(c, "")
            if gradnorm is not None:
                # GradNorm diagnostics logged on val epochs only. Raw per-term norms
                # come from the balancer's most-recent-update snapshot (fresh within
                # ``update_every``); effective = w_eff * raw is what SOAP sees.
                raw_norms = gradnorm.last_raw_norms
                for k in sw:
                    m = float(gn_mults.get(k, 1.0))
                    row[f"gn_mult_{k}"] = m
                    row[f"w_eff_{k}"] = w_eff[k]
                    g_raw = raw_norms.get(k)
                    if g_raw is not None:
                        row[f"grad_norm_{k}"] = g_raw
                        row[f"grad_norm_eff_{k}"] = w_eff[k] * g_raw
                row["gradnorm_mean_mult"] = gradnorm.last_mean_multiplier
                row["gradnorm_ms"] = gradnorm_ms
                row["gradnorm_weights"] = json.dumps(
                    {k: float(v) for k, v in gn_mults.items()}
                )
                row["gradnorm_bound_hits"] = json.dumps(gradnorm.bound_hit_counts)
                if grad_cosines is not None:
                    row["grad_cosines"] = json.dumps(grad_cosines)
            is_best = val["val_gnrmse"] < best_val
            if is_best:
                best_val = val["val_gnrmse"]
                # Full resumable best state is assembled after this row is flushed.
            fam_txt = " ".join(
                f"{k.split('gnrmse_fam_')[1]}={v * 100:.2f}%"
                for k, v in val.items() if k.startswith("gnrmse_fam_")
            )
            print(
                f"Validation for epoch {epoch}: "
                f"val_gnrmse={val['val_gnrmse'] * 100:.4f}% "
                f"val_rmse_K={val['val_rmse_K']:.4f}K (best={best_val * 100:.4f}%)"
                + ("  [new best -> cvit_best.pt]" if is_best else ""),
                flush=True,
            )
            print(
                f"  gnrmse[amp low/mid/high]="
                f"{val.get('gnrmse_amp_low', float('nan')) * 100:.2f}/"
                f"{val.get('gnrmse_amp_mid', float('nan')) * 100:.2f}/"
                f"{val.get('gnrmse_amp_high', float('nan')) * 100:.2f}%  "
                f"fam[{fam_txt}]",
                flush=True,
            )

        history.append({k: (v if v != "" else None) for k, v in row.items()})
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        last_csv_update = completed_updates

        def checkpoint_payload() -> dict[str, Any]:
            return {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "mu_global": mu, "sigma_global": sigma,
                "config": config, "resume_config": _forcing_resume_config(config),
                "epoch": epoch, "next_epoch": epoch + 1,
                "completed_updates": completed_updates,
                "last_csv_update": last_csv_update, "best_val": best_val,
                "gradnorm_state": gradnorm.state_dict() if gradnorm is not None else None,
                "causal_state": {
                    "eps": causal_eps, "n_bins": causal_cfg["n_bins"],
                    "updates": causal_updates, "calibrated": causal_calibrated,
                } if causal_cfg["enabled"] else None,
                "rng_state": _capture_forcing_rng(rng, gen, eval_gen),
                "forcing_cache": {
                    "params_batch": copy.deepcopy(params_batch),
                    "coll": _pack_tensors(coll),
                },
                "forcing_image": {
                    "ny_img": ny_img, "nt_img": nt_img, "a_ref": a_ref,
                    "t_ramp": t_ramp, "c_dom": c_dom, "d_dom": d_dom,
                    "t_final": t_final,
                },
            }

        payload = checkpoint_payload()
        if do_val and is_best:
            _atomic_torch_save(payload, best_path)
        if completed_updates % save_latest_every == 0 or epoch == epochs - 1:
            checkpoint_start = time.perf_counter()
            _atomic_torch_save(payload, latest_path)
            checkpoint_ms = (time.perf_counter() - checkpoint_start) * 1000.0
            checkpoint_mb = latest_path.stat().st_size / (1024.0 ** 2)
            print(
                f"  latest checkpoint: {checkpoint_ms:.1f} ms, {checkpoint_mb:.1f} MiB",
                flush=True,
            )

    summary = {"seed": seed, "best_val_gnrmse": best_val, "epochs": epochs}
    final_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    _atomic_torch_save(final_payload, final_path)
    _atomic_text(json.dumps(summary, indent=2) + "\n", run_dir / "final_metrics.json")
    _atomic_text("complete\n", complete_path)
    return summary


@torch.no_grad()
def validate_forcing_ic_gnrmse(
    model: ForcingICCViT,
    data: dict[str, Any],
    ids: np.ndarray,
    sim_params: np.ndarray,
    y_img: np.ndarray,
    t_img: np.ndarray,
    a_ref: float,
    t_ramp: float,
    device: torch.device,
    query_batch: int = 8,
    query_chunk: int = 0,
) -> dict[str, float]:
    """Two-branch deviation-field gnRMSE for the varying-IC single-slab benchmark.

    Identical amplitude-fair metric as :func:`validate_forcing_gnrmse`
    (``gnrmse = rmse_K / sigma_global`` on the ``T - T_RIGHT`` deviation field),
    but each held-out sim is reconstructed from BOTH its saved forcing image
    (``build_forcing_image`` / ``reconstruct_qL``) and its saved initial-condition
    field (snapshot 0), which the second encoder branch consumes. The latent is
    encoded ONCE per sim chunk and decoded slice-by-slice over the saved time
    grid (soft left wall -> no ``q_left`` term). Per-sim results are stratified by
    forcing-amplitude tercile and by ``ic_family`` so a family the model cannot
    fit stays visible rather than averaged away.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny, Nt = x_grid.numel(), y_grid.numel(), len(t_grid)
    if query_batch <= 0:
        raise ValueError("validation query_batch must be positive")
    if query_chunk < 0:
        raise ValueError("validation query_chunk must be non-negative")

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)

    ids = np.asarray(ids)
    per_sim_rmse: list[float] = []
    per_sim_amp: list[float] = []
    per_sim_fam: list[str] = []
    for start in range(0, len(ids), query_batch):
        chunk = ids[start:start + query_batch]
        params = [dict(sim_params[int(i)]) for i in chunk]
        u_forcing = build_forcing_image(params, y_img, t_img, a_ref, device, t_ramp)
        u_ic = build_ic_batch(data["trajectories"], chunk, mu, sigma, device)
        B = u_forcing.shape[0]
        latent = model.encode(u_forcing, u_ic)
        coords = mesh.expand(B, -1, -1)
        pred = torch.empty((B, Nt, Nx, Ny), device=device)
        for k in range(Nt):
            tk = torch.full((B, Nx * Ny, 1), float(t_grid[k]), device=device)
            out = _decode_in_chunks(model, latent, coords, tk, query_chunk)
            pred[:, k] = out[..., 0].view(B, Nx, Ny)
        pred_K = pred * sigma + mu
        truth = np.asarray(data["trajectories"][chunk], dtype=np.float32)
        truth_t = torch.from_numpy(truth).to(device)
        se = ((pred_K - truth_t) ** 2).sum(dim=(1, 2, 3))  # (B,)
        rmse = torch.sqrt(se / float(Nt * Nx * Ny)).detach().cpu().numpy()
        for b, p in enumerate(params):
            per_sim_rmse.append(float(rmse[b]))
            forcing = reconstruct_qL(
                p["temporal_family"], p["temporal_params"],
                p["spatial_family"], p["spatial_params"], t_ramp=t_ramp,
            )
            per_sim_amp.append(
                float(np.max(np.abs(forcing.evaluate_grid(y_img, t_img))))
            )
            per_sim_fam.append(str(p.get("ic_family", "")))

    rmse_arr = np.asarray(per_sim_rmse, dtype=np.float64)
    amp_arr = np.asarray(per_sim_amp, dtype=np.float64)
    fam_arr = np.asarray(per_sim_fam)
    gnrmse = rmse_arr / (float(sigma) + 1e-8)

    out = {
        "val_gnrmse": float(gnrmse.mean()),
        "val_rmse_K": float(rmse_arr.mean()),
    }
    if len(amp_arr) >= 3:
        q1, q2 = np.quantile(amp_arr, [1.0 / 3.0, 2.0 / 3.0])
        strata = {
            "amp_low": amp_arr <= q1,
            "amp_mid": (amp_arr > q1) & (amp_arr <= q2),
            "amp_high": amp_arr > q2,
        }
        for name, mask in strata.items():
            out[f"gnrmse_{name}"] = (
                float(gnrmse[mask].mean()) if mask.any() else float("nan")
            )
    for fam in np.unique(fam_arr):
        out[f"gnrmse_fam_{fam}"] = float(gnrmse[fam_arr == fam].mean())
    return out


def _forcing_ic_training_mode(pino: dict) -> tuple[str, dict[str, float]]:
    left = pino.get("lambda_bc_left", None)
    weights = {
        "data": float(pino.get("lambda_data", 0.0)),
        "r": float(pino.get("lambda_r", 0.0)),
        "ic": float(pino.get("lambda_ic", 0.0)),
        "bc": float(pino.get("lambda_bc", 0.0)),
        "bc_left": 0.0 if left is None else float(left),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("forcing_ic objective weights must be finite and non-negative")
    data_active = weights["data"] > 0.0
    physics_active = any(weights[name] > 0.0 for name in ("r", "ic", "bc", "bc_left"))
    if data_active and physics_active:
        raise ValueError(
            "diffusion_forcing_single supports supervised-only or physics-only "
            "training, not a mixed data-plus-physics objective"
        )
    if data_active:
        return "supervised", weights
    if physics_active:
        return "physics", weights
    raise ValueError("diffusion_forcing_single has no active training objective")


def _forcing_ic_supervised_resume_config(config: dict) -> dict[str, Any]:
    training = config["training"]
    pino = training["pino"]
    forcing = pino.get("forcing", {}) or {}
    return {
        "benchmark": copy.deepcopy(config.get("benchmark")),
        "model": copy.deepcopy(config["model"]),
        "training": {
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
            "optimizer": training.get("optimizer"),
            "soap": copy.deepcopy(training.get("soap")),
            "scheduler": copy.deepcopy(training.get("scheduler")),
            "grad_clip": training.get("grad_clip"),
            "pino": {
                "variant": pino.get("variant"),
                "lambda_data": pino.get("lambda_data"),
                "lambda_r": pino.get("lambda_r"),
                "lambda_ic": pino.get("lambda_ic"),
                "lambda_bc": pino.get("lambda_bc"),
                "lambda_bc_left": pino.get("lambda_bc_left"),
                "n_data_sims": pino.get("n_data_sims"),
                "n_data_pts": pino.get("n_data_pts"),
                "data_query_chunk": pino.get("data_query_chunk", 0),
                "forcing": {
                    key: forcing.get(key)
                    for key in (
                        "ny_img", "nt_img", "a_ref", "ramp_seconds",
                        "val_sim_batch", "val_query_chunk",
                    )
                },
            },
        },
    }


def run_one_seed_forcing_ic_supervised(
    config: dict,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Saved-data-only training of ``ForcingICCViT`` on varying-IC trajectories."""
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "cvit_latest.pt"
    best_path = run_dir / "cvit_best.pt"
    final_path = run_dir / "cvit_final.pt"
    complete_path = run_dir / "RUN_COMPLETE"

    pino = config["training"]["pino"]
    mode, weights = _forcing_ic_training_mode(pino)
    if mode != "supervised":
        raise ValueError("the supervised runner requires a data-only objective")

    fcfg = pino.get("forcing", {}) or {}
    extend_completed = bool(fcfg.get("extend_completed", False))
    if complete_path.exists() and not extend_completed:
        summary_path = run_dir / "final_metrics.json"
        if summary_path.exists():
            with open(summary_path) as stream:
                return json.load(stream)
        return {"seed": seed, "status": "complete", "run_dir": str(run_dir)}
    resuming = latest_path.exists() and (not complete_path.exists() or extend_completed)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    trajectory_path = Path(config["data"]["trajectories.npy"])
    sim_params = np.load(
        trajectory_path.parent / "sim_params.npy", allow_pickle=True,
    )
    data_signature = validate_forcing_ic_supervised_dataset(
        config, data, sim_params,
    )
    mu = float(data["mu_global"])
    sigma = float(data["sigma_global"])
    Nx = int(len(data["x_grid"]))
    Ny = int(len(data["y_grid"]))
    t_final = float(data["t_grid"][-1])

    stored_ramp = float(data_signature["ramp_seconds"])
    configured_ramp = fcfg.get("ramp_seconds", None)
    t_ramp = stored_ramp if configured_ramp is None else float(configured_ramp)
    if not math.isclose(t_ramp, stored_ramp, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "training.pino.forcing.ramp_seconds must match the saved "
            f"ramp_seconds.npy value {stored_ramp}"
        )
    a_ref = float(fcfg.get("a_ref") or A_AMP_REF)
    ny_img = int(fcfg.get("ny_img") if fcfg.get("ny_img") is not None else Ny)
    nt_img = int(fcfg.get("nt_img") if fcfg.get("nt_img") is not None else 128)
    y_img = np.linspace(
        float(data["y_grid"][0]), float(data["y_grid"][-1]),
        ny_img, dtype=np.float64,
    )
    t_img = np.linspace(0.0, t_final, nt_img, dtype=np.float64)

    n_data_sims = int(pino.get("n_data_sims", 8))
    n_data_pts = int(pino.get("n_data_pts", 1024))
    data_query_chunk = int(pino.get("data_query_chunk", 0) or 0)
    val_sim_batch = int(fcfg.get("val_sim_batch", 8))
    val_query_chunk = int(fcfg.get("val_query_chunk", 0) or 0)
    if val_sim_batch <= 0 or val_query_chunk < 0:
        raise ValueError(
            "forcing.val_sim_batch must be positive and val_query_chunk non-negative"
        )

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final,
        variant="forcing_ic",
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))
    if epochs <= 0 or validate_every <= 0:
        raise ValueError("training.epochs and validate_every must be positive")
    save_latest_every = max(1, int(fcfg.get("save_latest_every", 25)))
    grad_clip_cfg = config["training"].get("grad_clip", None)
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    if grad_clip is not None and grad_clip <= 0.0:
        raise ValueError("training.grad_clip must be null or positive")

    x_grid = torch.as_tensor(
        data["x_grid"], dtype=torch.float32, device=device,
    )
    y_grid = torch.as_tensor(
        data["y_grid"], dtype=torch.float32, device=device,
    )
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed + 1_000_003)

    family_columns = [f"gnrmse_fam_{family}" for family in IC_FAMILIES]
    fieldnames = [
        "epoch", "completed_updates", "objective", "lr",
        "loss", "loss_data", "w_data", "data_sim_ids",
        "val_gnrmse", "val_rmse_K",
        "gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high",
        *family_columns,
    ]
    metrics_path = run_dir / "train_metrics.csv"
    start_epoch = 0
    completed_updates = 0
    last_csv_update = 0
    best_val = float("inf")
    best_validation: dict[str, float] | None = None
    last_validation: dict[str, float] | None = None
    resume_config = _forcing_ic_supervised_resume_config(config)

    if resuming:
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False,
        )
        if checkpoint.get("objective") != "supervised":
            raise ValueError("resume checkpoint is not a supervised forcing-IC run")
        if checkpoint.get("resume_config") != resume_config:
            raise ValueError(
                "Incompatible supervised forcing-IC resume configuration; "
                "use a fresh experiment name."
            )
        if checkpoint.get("data_signature") != data_signature:
            raise ValueError(
                "The supervised forcing-IC dataset differs from the checkpoint."
            )
        saved_epochs = int(checkpoint["config"]["training"]["epochs"])
        if epochs < int(checkpoint["next_epoch"]):
            raise ValueError("training.epochs is below the checkpoint next_epoch")
        if epochs != saved_epochs:
            scheduler_type = str(config["training"]["scheduler"]["type"])
            if scheduler_type not in {"PICViTExponential", "StepLR"}:
                raise ValueError(
                    "Extending updates requires a horizon-independent scheduler"
                )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["next_epoch"])
        completed_updates = int(checkpoint["completed_updates"])
        last_csv_update = int(checkpoint["last_csv_update"])
        best_val = float(checkpoint["best_val"])
        best_validation = copy.deepcopy(checkpoint.get("best_validation"))
        last_validation = copy.deepcopy(checkpoint.get("last_validation"))
        _restore_forcing_rng(
            checkpoint["rng_state"], rng, gen, eval_gen,
        )
        _reconcile_forcing_metrics(
            metrics_path, fieldnames, last_csv_update,
        )
        if complete_path.exists():
            complete_path.unlink()
    else:
        with open(metrics_path, "w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    print(
        f"[supervised-forcing-ic] seed={seed} device={device} "
        f"updates={epochs} validate_every={validate_every} "
        f"grid={Nx}x{Ny} img={ny_img}x{nt_img} "
        f"n_data_sims={n_data_sims} n_data_pts={n_data_pts} "
        f"data_chunk={data_query_chunk} val_batch={val_sim_batch} "
        f"val_chunk={val_query_chunk} resume={resuming}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_loss, sampled_ids = forcing_ic_supervised_data_loss(
            model,
            sim_params,
            data["trajectories"],
            data["train_ids"],
            y_img=y_img,
            t_img=t_img,
            a_ref=a_ref,
            t_ramp=t_ramp,
            x_grid=x_grid,
            y_grid=y_grid,
            t_grid=t_grid,
            mu=mu,
            sigma=sigma,
            n_sims=n_data_sims,
            n_pts=n_data_pts,
            query_chunk=data_query_chunk,
            device=device,
            rng=rng,
            gen=gen,
        )
        loss = weights["data"] * data_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"Non-finite supervised forcing-IC loss at update {epoch}"
            )
        loss.backward()
        if not all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(
                f"Non-finite supervised forcing-IC gradient at update {epoch}"
            )
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip,
            )
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if not _all_finite(model.state_dict()) or not _all_finite(optimizer.state):
            raise FloatingPointError(
                f"Non-finite supervised forcing-IC state at update {epoch}"
            )
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        completed_updates += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "objective": "supervised",
            "lr": lr,
            "loss": float(loss.detach().cpu()),
            "loss_data": float(data_loss.detach().cpu()),
            "w_data": weights["data"],
            "data_sim_ids": json.dumps([int(i) for i in sampled_ids]),
            "val_gnrmse": "",
            "val_rmse_K": "",
            "gnrmse_amp_low": "",
            "gnrmse_amp_mid": "",
            "gnrmse_amp_high": "",
            **{column: "" for column in family_columns},
        }
        do_validation = (
            epoch % validate_every == 0 or epoch == epochs - 1
        )
        is_best = False
        if do_validation:
            model.eval()
            validation = validate_forcing_ic_gnrmse(
                model,
                data,
                data["val_ids"],
                sim_params,
                y_img,
                t_img,
                a_ref,
                t_ramp,
                device,
                query_batch=val_sim_batch,
                query_chunk=val_query_chunk,
            )
            last_validation = copy.deepcopy(validation)
            for key, value in validation.items():
                if key in row:
                    row[key] = value
            is_best = float(validation["val_gnrmse"]) < best_val
            if is_best:
                best_val = float(validation["val_gnrmse"])
                best_validation = copy.deepcopy(validation)
            print(
                f"Update {epoch}: data_mse={row['loss_data']:.6e} "
                f"val_gnrmse={100.0 * float(validation['val_gnrmse']):.4f}% "
                f"val_rmse_K={float(validation['val_rmse_K']):.4f}K "
                f"best={100.0 * best_val:.4f}%",
                flush=True,
            )
        else:
            print(
                f"Update {epoch}: data_mse={row['loss_data']:.6e} lr={lr:.2e}",
                flush=True,
            )

        with open(metrics_path, "a", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, extrasaction="ignore",
            )
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        last_csv_update = completed_updates

        payload = {
            "objective": "supervised",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "mu_global": mu,
            "sigma_global": sigma,
            "config": config,
            "resume_config": resume_config,
            "data_signature": data_signature,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "completed_updates": completed_updates,
            "last_csv_update": last_csv_update,
            "best_val": best_val,
            "best_validation": copy.deepcopy(best_validation),
            "last_validation": copy.deepcopy(last_validation),
            "rng_state": _capture_forcing_rng(rng, gen, eval_gen),
            "forcing_image": {
                "ny_img": ny_img,
                "nt_img": nt_img,
                "a_ref": a_ref,
                "t_ramp": t_ramp,
                "t_final": t_final,
            },
        }
        if is_best:
            _atomic_torch_save(payload, best_path)
        if completed_updates % save_latest_every == 0 or epoch == epochs - 1:
            _atomic_torch_save(payload, latest_path)

    final_payload = torch.load(
        latest_path, map_location="cpu", weights_only=False,
    )
    _atomic_torch_save(final_payload, final_path)
    summary = {
        "seed": seed,
        "objective": "supervised",
        "completed_updates": completed_updates,
        "best_val_gnrmse": best_val,
        "best_validation": best_validation,
        "last_validation": last_validation,
        "data_signature": data_signature,
        "n_data_sims": n_data_sims,
        "n_data_pts": n_data_pts,
        "test_set_evaluated": False,
    }
    _atomic_text(
        json.dumps(summary, indent=2) + "\n",
        run_dir / "final_metrics.json",
    )
    _atomic_text("complete\n", complete_path)
    return summary


def _resolve_forcing_residual_method(pino: dict) -> str:
    method = str(pino.get("residual_method", "autodiff"))
    if method not in {"autodiff", "finite_volume"}:
        raise ValueError(
            "training.pino.residual_method must be 'autodiff' or "
            f"'finite_volume'; got {method!r}."
        )
    return method


def _sample_online_problem_descriptors(
    problem,
    rngs: OnlineSamplerRNGs,
    batch_size: int,
    grids: dict[str, np.ndarray],
    time_cfg: dict[str, Any],
) -> list[dict]:
    assignments = balanced_ic_family_assignments(
        rngs.numpy["ic_family"], batch_size,
    )
    records = problem.sample_online_params(
        rngs.numpy["ic_params"], batch_size, grids, time_cfg,
        rng_profile=rngs.numpy["forcing_params"],
        rng_streams=rngs.numpy,
        ic_family_assignment=assignments,
    )
    descriptors = [_online_record_descriptor(record) for record in records]
    if [str(record["ic_family"]) for record in descriptors] != assignments:
        raise AssertionError("ProblemSpec changed the explicit IC-family assignment")
    return descriptors


def _sample_transition_multicontinuation_descriptors(
    problem,
    rngs: OnlineSamplerRNGs,
    batch_size: int,
    continuations_per_source: int,
    grids: dict[str, np.ndarray],
    time_cfg: dict[str, Any],
) -> list[dict]:
    batch_size = int(batch_size)
    continuations_per_source = int(continuations_per_source)
    if continuations_per_source < 1:
        raise ValueError(
            "forcing_continuations_per_source must be positive"
        )
    if batch_size % continuations_per_source != 0:
        raise ValueError(
            "forcing_continuations_per_source must divide transition.sim_batch"
        )
    source_count = batch_size // continuations_per_source
    source_assignments = balanced_ic_family_assignments(
        rngs.numpy["ic_family"], source_count,
    )
    assignments = [
        family
        for family in source_assignments
        for _ in range(continuations_per_source)
    ]
    records = problem.sample_online_params(
        rngs.numpy["ic_params"], batch_size, grids, time_cfg,
        rng_profile=rngs.numpy["forcing_params"],
        rng_streams=rngs.numpy,
        ic_family_assignment=assignments,
    )
    source_keys = ("T0", "ic_family", "ic_params")
    for start in range(0, batch_size, continuations_per_source):
        source = records[start]
        for index in range(start + 1, start + continuations_per_source):
            for key in source_keys:
                records[index][key] = copy.deepcopy(source[key])
    descriptors = [_online_record_descriptor(record) for record in records]
    if [str(record["ic_family"]) for record in descriptors] != assignments:
        raise AssertionError("ProblemSpec changed the explicit IC-family assignment")
    return descriptors


def _should_resample_online_batch(
    completed_updates: int, warmup_updates: int, resample_every: int,
) -> bool:
    if completed_updates < warmup_updates:
        return completed_updates % resample_every == 0
    return True


def _online_sampling_signature(problem, grid_shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "sampler_version": ONLINE_IC_SAMPLER_VERSION,
        "builder_version": IC_BUILDER_SCHEMA_VERSION,
        "problem": getattr(problem, "name", type(problem).__name__),
        "problem_version": getattr(problem, "problem_version", None),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
    }


def _all_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _online_failure_payload(
    *, phase: str, completed_updates: int, records: list[dict] | None,
    batch_key: str | None, coll, rng_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": str(phase),
        "completed_updates": int(completed_updates),
        "records": copy.deepcopy(records),
        "problem_keys": (
            [] if records is None else [record["problem_key"] for record in records]
        ),
        "batch_key": batch_key,
        "collocation": None if coll is None else _pack_tensors(coll),
        "sampler_version": ONLINE_IC_SAMPLER_VERSION,
        "builder_version": IC_BUILDER_SCHEMA_VERSION,
        "environment": _environment_fingerprint(),
        "rng_state": copy.deepcopy(rng_state),
    }


def _write_online_failure(
    run_dir: Path, *, phase: str, completed_updates: int,
    records: list[dict] | None, batch_key: str | None, coll,
    rng_state: dict[str, Any],
) -> None:
    _atomic_torch_save(
        _online_failure_payload(
            phase=phase,
            completed_updates=completed_updates,
            records=records,
            batch_key=batch_key,
            coll=coll,
            rng_state=rng_state,
        ),
        Path(run_dir) / "failed_online_batch.pt",
    )


def _resolve_forcing_fv_dt(
    config: dict, pino: dict, t_final: float,
) -> tuple[float, str, int]:
    dt_cfg = pino.get("dt", None)
    if dt_cfg is not None:
        if np.asarray(dt_cfg).ndim != 0:
            raise ValueError("training.pino.dt must be a scalar")
        dt = float(dt_cfg)
        source = "configuration"
    else:
        stored = load_solver_dt(config["data"]["t_grid_path"])
        if stored is None:
            raise ValueError(
                "finite_volume residual requires training.pino.dt or dt.npy "
                "beside the configured t_grid.npy"
            )
        dt = float(stored)
        source = "dt.npy"
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"finite-volume dt must be finite and > 0; got {dt!r}")
    if dt > t_final:
        raise ValueError(
            f"finite-volume dt={dt} exceeds t_final={t_final}"
        )
    ratio = float(t_final) / dt
    n_steps = int(round(ratio))
    if n_steps < 1 or not math.isclose(ratio, n_steps, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            "finite-volume solver-lattice sampling requires integral "
            f"t_final/dt; got {t_final}/{dt}={ratio}"
        )
    return dt, source, n_steps


def _sample_ic_nodes(
    n_ic: int,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    Nx, Ny = int(x_grid.numel()), int(y_grid.numel())
    ix = torch.randint(0, Nx, (n_ic,), device=device, generator=generator)
    iy = torch.randint(0, Ny, (n_ic,), device=device, generator=generator)
    coords = torch.stack((x_grid[ix], y_grid[iy]), dim=-1).unsqueeze(0)
    return {
        "coords": coords,
        "t": torch.zeros((1, n_ic, 1), device=device),
        "ix": ix,
        "iy": iy,
    }


def _sample_lattice_intervals(
    rng: np.random.Generator,
    batch_size: int,
    intervals_per_sim: int,
    n_steps: int,
    n_bins: int,
    stratified: bool,
    max_start_step: int | None = None,
) -> dict[str, torch.Tensor]:
    M = int(batch_size * intervals_per_sim)
    active_steps = n_steps if max_start_step is None else int(max_start_step) + 1
    if not 1 <= active_steps <= n_steps:
        raise ValueError("active causal interval count is outside the FV lattice")
    all_starts = np.arange(active_steps, dtype=np.int64)
    all_bins = np.minimum(
        ((all_starts.astype(np.float64) + 0.5) * n_bins / active_steps).astype(np.int64),
        n_bins - 1,
    )
    if stratified:
        available = np.unique(all_bins)
        if M < available.size:
            requested = np.asarray([
                rng.choice(group) for group in np.array_split(available, M)
            ])
        else:
            requested = np.resize(available, M)
        rng.shuffle(requested)
        starts = np.empty(M, dtype=np.int64)
        for m, bin_id in enumerate(requested):
            starts[m] = int(rng.choice(all_starts[all_bins == bin_id]))
    else:
        starts = rng.integers(0, n_steps, size=M, dtype=np.int64)
    bin_ids = all_bins[starts]
    return {
        "start_idx": torch.from_numpy(starts),
        "causal_bin_ids": torch.from_numpy(bin_ids),
        "sim_local": torch.from_numpy(
            np.repeat(np.arange(batch_size, dtype=np.int64), intervals_per_sim)
        ),
    }


def _full_grid_query_mesh(
    x_grid: torch.Tensor, y_grid: torch.Tensor,
) -> torch.Tensor:
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).unsqueeze(0)
    Nx, Ny = int(x_grid.numel()), int(y_grid.numel())
    if mesh.shape != (1, Nx * Ny, 2):
        raise AssertionError("full-grid query count does not match Nx*Ny")
    expected_x = x_grid.repeat_interleave(Ny)
    expected_y = y_grid.repeat(Nx)
    if not torch.equal(mesh[0, :, 0], expected_x) or not torch.equal(
        mesh[0, :, 1], expected_y
    ):
        raise AssertionError(
            "FV query topology must use indexing='ij' with y-index varying fastest"
        )
    return mesh


def _forcing_fv_signature(
    config: dict,
    geom_cfg: dict,
    *,
    dt: float,
    intervals_per_sim: int,
    grid_shape: tuple[int, int],
    causal_cfg: dict,
    ramp_seconds: float,
    stratified_time_sampling: bool,
) -> dict[str, Any]:
    spec = problem_from_config(config)
    material_keys = (
        "k", "rho", "cp", "k_left", "k_right", "interface_x", "R_c",
    )
    material = {
        key: geom_cfg[key] for key in material_keys
        if key in geom_cfg and np.isscalar(geom_cfg[key])
    }
    x_grid = np.asarray(geom_cfg["x_grid"], dtype=np.float64)
    y_grid = np.asarray(geom_cfg["y_grid"], dtype=np.float64)
    return {
        "residual_method": "finite_volume",
        "resolved_dt": float(dt),
        "intervals_per_sim": int(intervals_per_sim),
        "stratified_time_sampling": bool(stratified_time_sampling),
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "grid": {
            "x_min": float(x_grid[0]),
            "x_max": float(x_grid[-1]),
            "y_min": float(y_grid[0]),
            "y_max": float(y_grid[-1]),
            "hx": float((x_grid[-1] - x_grid[0]) / (x_grid.size - 1)),
            "hy": float((y_grid[-1] - y_grid[0]) / (y_grid.size - 1)),
        },
        "causal": {
            "enabled": bool(causal_cfg["enabled"]),
            "n_bins": int(causal_cfg["n_bins"]),
        },
        "benchmark": getattr(spec, "name", config.get("benchmark", {}).get("name")),
        "problem_version": getattr(spec, "problem_version", None),
        "geometry_kind": geom_cfg["geometry_kind"],
        "geometry_version": geom_cfg.get("geometry_version"),
        "material": material,
        "sigma_global": float(geom_cfg["sigma_global"]),
        "T_right_tilde": float(geom_cfg["T_right_tilde"]),
        "forcing_quadrature": geom_cfg["forcing_quadrature"],
        "forcing_schema_version": geom_cfg.get("forcing_schema_version"),
        "ramp_schema_version": geom_cfg.get("ramp_schema_version"),
        "ramp_seconds": float(ramp_seconds),
        "closure_identity": geom_cfg.get("closure_identity"),
    }


def _forcing_ic_fv_losses(
    *,
    model: ForcingICCViT,
    latent: torch.Tensor,
    coll: dict[str, Any],
    ic_target: torch.Tensor,
    params_batch: list[dict],
    ids_batch: np.ndarray | None,
    problem,
    collocation_ctx,
    geom_cfg: dict,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    dt: float,
    n_steps: int,
    sigma_global: float,
    causal_cfg: dict,
    causal_eps: float,
    chunk_r: int,
) -> dict[str, Any]:
    Nx, Ny = int(x_grid.numel()), int(y_grid.numel())
    mesh = _full_grid_query_mesh(x_grid, y_grid)
    Nq = Nx * Ny
    starts = coll["fv"]["start_idx"].to(device=latent.device, dtype=torch.long)
    bin_ids = coll["fv"]["causal_bin_ids"].to(device=latent.device, dtype=torch.long)
    sim_local = coll["fv"]["sim_local"].to(device=latent.device, dtype=torch.long)
    M = int(starts.numel())
    if M == 0 or sim_local.shape != starts.shape or bin_ids.shape != starts.shape:
        raise AssertionError("invalid FV interval descriptor shapes")
    if int(starts.min()) < 0 or int(starts.max()) >= n_steps:
        raise AssertionError("FV interval start index lies outside the solver lattice")

    ic_coords = coll["ic"]["coords"].expand(latent.shape[0], -1, -1)
    ic_t = coll["ic"]["t"].expand(latent.shape[0], -1, -1)
    ic_pred = _decode_in_chunks(model, latent, ic_coords, ic_t, chunk_r)
    loss_ic = (ic_pred - ic_target).square().mean()

    # MPS has no float64 tensor support.  The grid/query tensors are float32
    # on MPS (and already provide the residual's working dtype), so keep the
    # sampled CN times in that dtype instead of forcing a host-style float64.
    tn = starts.to(dtype=x_grid.dtype) * float(dt)
    tnp1 = tn + float(dt)
    latent_M = latent[sim_local]
    coords_M = mesh.expand(M, -1, -1)
    tn_q = tn.to(dtype=x_grid.dtype).view(M, 1, 1).expand(M, Nq, 1)
    tnp1_q = tnp1.to(dtype=x_grid.dtype).view(M, 1, 1).expand(M, Nq, 1)
    decoded_n = _decode_in_chunks(model, latent_M, coords_M, tn_q, chunk_r)
    decoded_np1 = _decode_in_chunks(model, latent_M, coords_M, tnp1_q, chunk_r)
    expected_shape = (M, Nq, 1)
    if decoded_n.shape != expected_shape or decoded_np1.shape != expected_shape:
        raise AssertionError(
            f"FV decoder output must have shape {expected_shape}; got "
            f"{tuple(decoded_n.shape)} and {tuple(decoded_np1.shape)}"
        )
    T_n = decoded_n[..., 0].reshape(M, Nx, Ny)
    T_np1 = decoded_np1[..., 0].reshape(M, Nx, Ny)
    if T_n.shape != (M, Nx, Ny) or T_np1.shape != (M, Nx, Ny):
        raise AssertionError("decoded FV fields do not match public (B,Nx,Ny) layout")

    qn = np.empty((M, Ny), dtype=np.float64)
    qnp1 = np.empty((M, Ny), dtype=np.float64)
    qint_values: list[np.ndarray | None] = []
    interface_x = np.empty(M, dtype=np.float64)
    resistance = np.empty(M, dtype=np.float64)
    for m in range(M):
        local = int(sim_local[m])
        closure = problem.collocation_closure(
            collocation_ctx,
            local if ids_batch is None else int(ids_batch[local]),
            params_batch[local],
            float(tn[m]), float(tnp1[m]),
        )
        if closure is None:
            raise ValueError("finite_volume residual requires a collocation_closure")
        rc, qn_m, qnp1_m, qint_m = closure
        qn[m] = np.asarray(qn_m, dtype=np.float64)
        qnp1[m] = np.asarray(qnp1_m, dtype=np.float64)
        qint_values.append(
            None if qint_m is None else np.asarray(qint_m, dtype=np.float64)
        )
        interface_x[m] = float(
            params_batch[local].get("interface_x", geom_cfg.get("interface_x", np.nan))
        )
        resistance[m] = float(rc)

    quadrature = str(geom_cfg["forcing_quadrature"])
    if quadrature == "exact_interval_integral":
        if any(value is None for value in qint_values):
            raise ValueError("exact_interval_integral closure returned qL_int=None")
        qint = np.stack(qint_values)
    elif quadrature == "endpoint_cn":
        if any(value is not None for value in qint_values):
            raise ValueError("endpoint_cn closure must return qL_int=None")
        qint = None
    else:
        raise ValueError(f"unknown forcing quadrature {quadrature!r}")

    geometry_kind = str(geom_cfg["geometry_kind"])
    if geometry_kind == "homogeneous":
        geom = build_homogeneous_cn_geom(
            geom_cfg["x_grid"], geom_cfg["y_grid"], geom_cfg["k"], dt,
            sigma_global=sigma_global, rho=geom_cfg["rho"], cp=geom_cfg["cp"],
            device=latent.device, dtype=T_n.dtype,
        )
    elif geometry_kind == "single_interface":
        if np.all(interface_x == interface_x[0]):
            geom = build_cn_geom_batched(
                geom_cfg["x_grid"], geom_cfg["y_grid"],
                geom_cfg["k_left"], geom_cfg["k_right"], interface_x[0],
                resistance, dt, sigma_global=sigma_global,
                rho=geom_cfg.get("rho", 1.0), cp=geom_cfg.get("cp", 1.0),
                device=latent.device, dtype=T_n.dtype,
            )
        else:
            geom = build_cn_geom_per_interface(
                geom_cfg["x_grid"], geom_cfg["y_grid"],
                geom_cfg["k_left"], geom_cfg["k_right"], interface_x,
                resistance, dt, sigma_global=sigma_global,
                rho=geom_cfg.get("rho", 1.0), cp=geom_cfg.get("cp", 1.0),
                device=latent.device, dtype=T_n.dtype,
            )
    else:
        raise ValueError(f"unknown FV geometry_kind {geometry_kind!r}")

    bc = FullBCData(
        T_right_tilde=torch.as_tensor(
            geom_cfg["T_right_tilde"], device=latent.device, dtype=T_n.dtype,
        ),
        qL_n=torch.from_numpy(qn).to(device=latent.device, dtype=T_n.dtype),
        qL_np1=torch.from_numpy(qnp1).to(device=latent.device, dtype=T_n.dtype),
        qL_int=(
            None if qint is None
            else torch.from_numpy(qint).to(device=latent.device, dtype=T_n.dtype)
        ),
    )
    phys = full_bc_physics_loss(
        T_n, T_np1, geom, bc, per_sample=True, dirichlet_both_ends=True,
    )
    interior = phys["interior_per_sample"]
    pointwise = interior.mean()
    loss_r = pointwise
    out: dict[str, torch.Tensor | bool] = {
        "r": loss_r,
        "ic": loss_ic,
        "bc_left": phys["phys_left_neumann_mse"],
        "bc_hom": phys["phys_topbot_adiabatic_mse"],
        "bc": (
            phys["phys_left_neumann_mse"]
            + 2.0 * phys["phys_topbot_adiabatic_mse"]
        ) / 3.0,
        "right_dir": phys["phys_right_dirichlet_mse"],
        "r_pointwise_mse": pointwise.detach(),
    }
    if causal_cfg["enabled"]:
        n_bins = int(causal_cfg["n_bins"])
        counts = torch.zeros(n_bins, device=latent.device, dtype=interior.dtype)
        sums = torch.zeros_like(counts)
        counts.index_add_(0, bin_ids, torch.ones_like(interior))
        sums.index_add_(0, bin_ids, interior)
        live_means = sums / counts.clamp_min(1.0)
        populated = bool((counts > 0).all().item())
        weights = _causal_weights(live_means.detach(), causal_eps)
        if populated:
            loss_r = (weights[bin_ids] * interior).mean()
        equal_bin = live_means.detach().mean() if populated else pointwise.detach()
        out.update({
            "r": loss_r,
            "causal_bin_losses": live_means.detach(),
            "causal_bin_counts": counts.detach(),
            "causal_weights": weights.detach(),
            "causal_populated": populated,
            "r_equal_bin_mean": equal_bin,
            "r_causal_loss": loss_r.detach(),
            "causal_reduction_ratio": (
                loss_r.detach() / equal_bin.clamp_min(torch.finfo(interior.dtype).tiny)
            ),
        })
    return out


def run_one_seed_forcing_ic_pino(config: dict, seed: int, run_dir: Path) -> dict[str, Any]:
    """Physics-only training of a :class:`ForcingICCViT` on the single-slab
    forcing benchmark with a VARYING initial condition.

    This is the two-branch sibling of :func:`run_one_seed_forcing_pino`. The
    default ``autodiff`` backend uses free continuous collocation and derivative
    boundary residuals. The exclusive ``finite_volume`` backend instead uses
    paired full-grid outputs on consecutive solver-lattice times and the exact
    Crank--Nicolson closure, while sharing the model, optimizer, validation, and
    checkpoint machinery.

    Each step encodes ONCE (``latent = model.encode(u_forcing, u_ic)``) and every
    residual reuses that latent. Complete IC + forcing problems are sampled online
    from lossless descriptors; saved trajectories are used only for frozen
    normalization and validation.
    """
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "cvit_latest.pt"
    best_path = run_dir / "cvit_best.pt"
    final_path = run_dir / "cvit_final.pt"
    complete_path = run_dir / "RUN_COMPLETE"

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])
    t_final = float(data["t_grid"][-1])
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    c_dom, d_dom = float(y_grid_np[0]), float(y_grid_np[-1])

    # Saved FV forcing records are VALIDATION-only. sim_params.npy lives beside
    # trajectories.npy (same directory the data generator writes to).
    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(str(sp_path), allow_pickle=True)

    pino = config["training"]["pino"]
    residual_method = _resolve_forcing_residual_method(pino)
    fv_dt = fv_dt_source = fv_n_steps = None
    if residual_method == "finite_volume":
        if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
            raise ValueError("finite_volume residual requires finite sigma_global > 0")
        fv_dt, fv_dt_source, fv_n_steps = _resolve_forcing_fv_dt(
            config, pino, t_final,
        )
    fcfg = pino.get("forcing", {}) or {}
    extend_completed = bool(fcfg.get("extend_completed", False))
    if complete_path.exists() and not extend_completed:
        summary_path = run_dir / "final_metrics.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return {"seed": seed, "status": "complete", "run_dir": str(run_dir)}
    resuming = latest_path.exists() and (not complete_path.exists() or extend_completed)
    causal_cfg = _resolve_forcing_causal(pino.get("causal", {}))
    warmup = _forcing_warmup_config(fcfg)
    save_latest_every = max(1, int(fcfg.get("save_latest_every", 25)))
    grad_clip_cfg = fcfg.get("grad_clip", None)
    if grad_clip_cfg is None:
        grad_clip_cfg = config["training"].get("grad_clip", None)
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    if grad_clip is not None and grad_clip <= 0.0:
        raise ValueError("training.pino.forcing.grad_clip must be null or > 0")
    a_ref = float(fcfg.get("a_ref") if fcfg.get("a_ref") is not None else A_AMP_REF)
    ny_img = int(fcfg.get("ny_img") if fcfg.get("ny_img") is not None else Ny)
    nt_img = int(fcfg.get("nt_img") if fcfg.get("nt_img") is not None else 128)
    y_img = np.linspace(c_dom, d_dom, ny_img, dtype=np.float64)
    t_img = np.linspace(0.0, t_final, nt_img, dtype=np.float64)

    ramp_cfg = fcfg.get("ramp_seconds", None)
    if ramp_cfg is not None:
        t_ramp = float(ramp_cfg)
    else:
        t_ramp = load_ramp_seconds(config["data"]["t_grid_path"])
        if t_ramp is None:
            dt = load_solver_dt(config["data"]["t_grid_path"])
            t_ramp = default_ramp_seconds(dt if dt is not None else t_final / 100.0)

    temporal_window = dict(
        t_on=float(fcfg.get("t_on", 0.0)),
        t_off=float(fcfg.get("t_off", 0.2)),
        phase=float(fcfg.get("phase", 0.0)),
        tukey_alpha=float(fcfg.get("tukey_alpha", 0.5)),
    )
    tf_fix = fcfg.get("temporal_family")
    sf_fix = fcfg.get("spatial_family")
    if tf_fix not in (None, "sin") or sf_fix not in (None, "uniform"):
        raise ValueError(
            "diffusion_forcing_single online sampling is fixed to sin/uniform"
        )
    dt_sample = (
        float(fcfg["dt_sample"])
        if fcfg.get("dt_sample") is not None
        else (
            float(fv_dt) if residual_method == "finite_volume"
            else t_final / max(nt_img - 1, 1)
        )
    )
    sampler = str(fcfg.get("collocation") or "lhs")
    coll_bias = _resolve_collocation_bias(pino)
    intervals_per_sim = int(pino.get("intervals_per_sim", 2))
    if intervals_per_sim <= 0:
        raise ValueError("training.pino.intervals_per_sim must be > 0")
    stratified_fv = bool(pino.get("stratified_time_sampling", True))
    chunk_r = int(pino.get("chunk_r", 0) or 0)
    if chunk_r < 0:
        raise ValueError("training.pino.chunk_r must be >= 0")

    problem = problem_from_config(config)
    X_online, Y_online = np.meshgrid(
        np.asarray(data["x_grid"], dtype=np.float64),
        np.asarray(data["y_grid"], dtype=np.float64),
        indexing="ij",
    )
    online_grids = {
        "X": X_online,
        "Y": Y_online,
        "x_grid": np.asarray(data["x_grid"], dtype=np.float64),
        "y_grid": np.asarray(data["y_grid"], dtype=np.float64),
    }
    online_time_cfg = {
        "dt": float(dt_sample),
        "t_final": t_final,
        "b": float(data["x_grid"][-1]),
        "T_right": float(T_RIGHT),
        **temporal_window,
    }
    b_online = float(online_grids["x_grid"][-1])
    online_signature = _online_sampling_signature(problem, (Nx, Ny))
    environment = _environment_fingerprint()
    collocation_ctx = SimpleNamespace(
        x_grid=np.asarray(data["x_grid"], dtype=np.float64),
        y_grid=np.asarray(data["y_grid"], dtype=np.float64),
        Nx=Nx,
        Ny=Ny,
        ramp_seconds=float(t_ramp),
    )
    fv_geom_cfg = fv_signature = None
    if residual_method == "finite_volume":
        fv_geom_cfg = problem.collocation_geom_cfg(
            collocation_ctx,
            config["training"].get("physics", {}) or {},
            float(mu),
            float(sigma),
            float(fv_dt),
        )
        if fv_geom_cfg is None:
            raise ValueError(
                f"benchmark {getattr(problem, 'name', '?')!r} does not define "
                "finite-volume collocation geometry"
            )
        if not np.array_equal(
            np.asarray(fv_geom_cfg["x_grid"]), collocation_ctx.x_grid,
        ) or not np.array_equal(
            np.asarray(fv_geom_cfg["y_grid"]), collocation_ctx.y_grid,
        ):
            raise ValueError("FV geometry grids must exactly match the decoder grids")
        if not math.isclose(float(fv_geom_cfg["dt"]), float(fv_dt), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("FV geometry dt does not match the resolved solver dt")
        fv_signature = _forcing_fv_signature(
            config,
            fv_geom_cfg,
            dt=float(fv_dt),
            intervals_per_sim=intervals_per_sim,
            grid_shape=(Nx, Ny),
            causal_cfg=causal_cfg,
            ramp_seconds=t_ramp,
            stratified_time_sampling=stratified_fv,
        )

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final,
        variant="forcing_ic",
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    lam_r = float(pino["lambda_r"])
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    lam_bc_left_cfg = pino.get("lambda_bc_left", None)
    lam_bc_left = None if lam_bc_left_cfg is None else float(lam_bc_left_cfg)
    # The two-branch path is physics-only: the supervised interior-field data term
    # calls the monolithic model(u, coords, t) signature and is incompatible with
    # ForcingICCViT.forward(u_forcing, u_ic, coords, t). Guard rather than silently
    # ignore a set lambda_data.
    lam_data = float(pino.get("lambda_data", 0.0))
    if lam_data > 0.0:
        raise ValueError(
            "lambda_data (supervised interior term) is not supported on the "
            "two-branch forcing_ic path; keep it 0.0 (physics-only)."
        )
    n_r = int(pino["n_r"])
    n_ic = int(pino["n_ic"])
    n_bc = int(pino["n_bc"])
    sim_batch = int(pino["sim_batch"])
    alpha = float(pino.get("alpha", 1.0))
    # IC anchor nodes gather from the same transferred normalized online field
    # that feeds the encoder, avoiding a separately normalized target path.
    physics_cfg = config["training"].get("physics", {}) or {}
    n_ic_points = int(
        physics_cfg.get("n_ic_points")
        if physics_cfg.get("n_ic_points") is not None
        else n_ic
    )
    # Right-wall Dirichlet baseline (300 K) for the hard ansatz. The IC anchor is
    # the SAMPLED T0_tilde (per-sim), NOT this constant -- t_right_tilde only feeds
    # the ic_loss="rel" denominator, unused on the "mse" path.
    t_right_tilde = (
        (T_RIGHT - mu) / sigma
        if residual_method == "finite_volume"
        else (T_RIGHT - mu) / (sigma + 1e-8)
    )

    if lam_bc_left is None:
        gn_term_weights = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
        gn_cos_pairs = None
    else:
        gn_term_weights = {
            "r": lam_r, "ic": lam_ic, "bc_left": lam_bc_left, "bc_hom": lam_bc,
        }
        gn_cos_pairs = [("r", "bc_left"), ("ic", "bc_left"), ("bc_hom", "bc_left")]
    gradnorm = build_gradnorm(config, term_weights=gn_term_weights)

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    online_rngs = OnlineSamplerRNGs.create(seed, device)
    gen = online_rngs.autodiff_collocation
    eval_gen = online_rngs.evaluation
    rng = online_rngs.numpy["forcing_params"]

    metrics_path = run_dir / "train_metrics.csv"
    gn_cols = ["r", "ic", "bc_left", "bc_hom"]
    causal_cols = range(causal_cfg["n_bins"])
    fieldnames = [
        "epoch", "completed_updates", "residual_method", "lr_first", "lr_last",
        "loss", "loss_r", "loss_ic", "loss_bc", "loss_bc_left", "loss_bc_hom",
        "loss_right_dir",
        "loss_data",
        "w_r", "w_ic", "w_bc", "w_bc_left", "w_data",
        *[f"warm_mult_{c}" for c in gn_cols],
        "r_pointwise_mse", "r_equal_bin_mean", "r_causal_loss",
        "causal_reduction_ratio", "causal_eps", "causal_eps_next",
        "causal_mean_weight", "causal_last_weight", "causal_log_last_weight",
        "causal_adaptation_action",
        *[f"causal_loss_bin_{i:02d}" for i in causal_cols],
        *[f"causal_weight_bin_{i:02d}" for i in causal_cols],
        *[f"causal_count_bin_{i:02d}" for i in causal_cols],
        "val_gnrmse", "val_rmse_K",
        "gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high",
        *[f"gn_mult_{c}" for c in gn_cols],
        *[f"w_eff_{c}" for c in gn_cols],
        *[f"grad_norm_{c}" for c in gn_cols],
        *[f"grad_norm_eff_{c}" for c in gn_cols],
        "gradnorm_mean_mult", "gradnorm_ms",
        "gradnorm_weights", "grad_cosines", "gradnorm_bound_hits",
        "online_max_abs_z", "online_frac_abs_z_gt_5", "online_family_z_ranges",
        "problem_keys", "batch_key",
        "online_sampling_fraction",
    ]

    start_epoch = 0
    completed_updates = 0
    last_csv_update = 0
    best_val = float("inf")
    causal_eps = float(causal_cfg["initial_eps"])
    causal_updates = 0
    causal_calibrated = False
    coll = None
    params_batch: list[dict] | None = None
    active_batch_key: str | None = None
    profile_cfg = dict(pino.get("online_sampling", {}) or {})
    profile_every = max(1, int(profile_cfg.get("profile_every", 100)))
    min_profile_samples = max(1, int(profile_cfg.get("min_profile_samples", 10)))
    warn_fraction = float(profile_cfg.get("warn_fraction", 0.10))
    if not math.isfinite(warn_fraction) or warn_fraction <= 0.0:
        raise ValueError("training.pino.online_sampling.warn_fraction must be > 0")
    profile_ratios: list[float] = []
    profile_warned = False

    if resuming:
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        saved_resume = ckpt.get("resume_config")
        current_resume = _forcing_resume_config(
            config, float(fv_dt) if residual_method == "finite_volume" else None,
        )
        if saved_resume != current_resume:
            raise ValueError(
                "Incompatible diffusion_forcing resume configuration; use a fresh run directory."
            )
        if residual_method == "finite_volume" and ckpt.get("fv_residual_signature") != fv_signature:
            raise ValueError(
                "Incompatible finite-volume residual semantics; use a fresh run directory."
            )
        if ckpt.get("online_sampling_signature") != online_signature:
            raise ValueError(
                "Incompatible online IC sampling semantics; use a fresh run directory."
            )
        _validate_environment_fingerprint(
            ckpt.get("online_environment", {}), environment,
        )
        saved_epochs = int(ckpt["config"]["training"]["epochs"])
        if epochs < int(ckpt["next_epoch"]):
            raise ValueError("training.epochs is below the checkpoint next_epoch")
        if epochs != saved_epochs:
            sched_type = str(config["training"]["scheduler"]["type"])
            if sched_type not in {"PICViTExponential", "StepLR"}:
                raise ValueError("Extending epochs requires a horizon-independent scheduler")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if gradnorm is not None:
            if ckpt.get("gradnorm_state") is None:
                raise ValueError("Resume checkpoint is missing GradNorm state")
            gradnorm.load_state_dict(ckpt["gradnorm_state"])
        start_epoch = int(ckpt["next_epoch"])
        completed_updates = int(ckpt["completed_updates"])
        last_csv_update = int(ckpt["last_csv_update"])
        best_val = float(ckpt["best_val"])
        causal_state = ckpt.get("causal_state") or {}
        if causal_cfg["enabled"]:
            if int(causal_state.get("n_bins", -1)) != causal_cfg["n_bins"]:
                raise ValueError("Resume causal n_bins does not match the active configuration")
            causal_eps = float(causal_state["eps"])
            causal_updates = int(causal_state.get("updates", 0))
            causal_calibrated = bool(causal_state.get("calibrated", False))
        cache = ckpt.get("online_cache") or ckpt.get("forcing_cache") or {}
        params_batch = copy.deepcopy(
            cache.get("records", cache.get("params_batch"))
        )
        coll = _unpack_tensors(cache.get("coll"), device) if cache.get("coll") else None
        active_batch_key = cache.get("batch_key")
        _restore_forcing_rng(ckpt["rng_state"], rng, gen, eval_gen)
        online_rngs.load_state_dict(ckpt["online_rng_state"])
        if params_batch is not None:
            _materialize_online_records(
                params_batch, X_online, Y_online,
                T_right=T_RIGHT, b=b_online,
            )
            reconstructed_key = _batch_key(params_batch, _pack_tensors(coll))
            if reconstructed_key != active_batch_key:
                raise ValueError("online batch descriptor key mismatch on resume")
        _reconcile_forcing_metrics(metrics_path, fieldnames, last_csv_update)
        if complete_path.exists():
            complete_path.unlink()
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()

    residual_detail = ""
    if residual_method == "finite_volume":
        residual_detail = (
            f"fv_dt={fv_dt}({fv_dt_source}) intervals_per_sim={intervals_per_sim} "
            f"stratified={stratified_fv} chunk_r={chunk_r} | "
        )
    print(
        f"[pino-forcing-ic] seed={seed} device={device} epochs={epochs} "
        f"residual_method={residual_method} "
        f"validate_every={validate_every} | grid={Nx}x{Ny} t_final={t_final:.4f} | "
        f"img=({ny_img}x{nt_img}) a_ref={a_ref} t_ramp={t_ramp:.4g} "
        f"sampler={sampler} coll_bias={coll_bias} | lambda_r={lam_r} lambda_ic={lam_ic} "
        f"lambda_bc={lam_bc} lambda_bc_left={lam_bc_left} | "
        f"n_r={n_r} n_ic={n_ic} n_ic_points={n_ic_points} n_bc={n_bc} "
        f"sim_batch={sim_batch} alpha={alpha} k_slab={K_SLAB} "
        f"{residual_detail}"
        f"online_ic={ONLINE_IC_SAMPLER_VERSION}/v{IC_BUILDER_SCHEMA_VERSION} | "
        f"causal={causal_cfg} | warmup={warmup} grad_clip={grad_clip} "
        f"save_latest_every={save_latest_every} resume={resuming}",
        flush=True,
    )

    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs):
        model.train()
        warming = completed_updates < warmup["steps"]
        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        profile_this = completed_updates % profile_every == 0
        descriptor_seconds = 0.0
        if params_batch is None or _should_resample_online_batch(
            completed_updates, warmup["steps"], warmup["resample_every"],
        ):
            descriptor_start = time.perf_counter()
            params_batch = _sample_online_problem_descriptors(
                problem, online_rngs, sim_batch, online_grids, online_time_cfg,
            )
            if residual_method == "autodiff":
                coll = sample_collocation(
                    n_r, n_ic_points, n_bc, t_final,
                    x_grid_t, y_grid_t, device, gen,
                    sampler=sampler, bias=coll_bias,
                )
            else:
                coll = {
                    "ic": _sample_ic_nodes(
                        n_ic_points, x_grid_t, y_grid_t, device, gen,
                    ),
                    "fv": _sample_lattice_intervals(
                        online_rngs.numpy["fv_intervals"], sim_batch,
                        intervals_per_sim, int(fv_n_steps),
                        causal_cfg["n_bins"], stratified_fv,
                    ),
                }
            active_batch_key = _batch_key(params_batch, _pack_tensors(coll))
            descriptor_seconds = time.perf_counter() - descriptor_start

        ic_build_start = time.perf_counter()
        online_records = _materialize_online_records(
            params_batch, X_online, Y_online,
            T_right=T_RIGHT, b=b_online,
        )
        _physical_ic, normalized_ic, online_diag = _normalized_online_ic_buffer(
            online_records, mu, sigma,
        )
        ic_build_seconds = time.perf_counter() - ic_build_start

        transfer_start_cpu = time.perf_counter()
        transfer_start_event = transfer_end_event = None
        if profile_this and device.type == "cuda":
            transfer_start_event = torch.cuda.Event(enable_timing=True)
            transfer_end_event = torch.cuda.Event(enable_timing=True)
            transfer_start_event.record()
        u_ic = _transfer_normalized_ic(normalized_ic, device)
        u_forcing = build_forcing_image(
            online_records, y_img, t_img, a_ref, device, t_ramp,
        )
        if transfer_end_event is not None:
            transfer_end_event.record()
        transfer_seconds_cpu = time.perf_counter() - transfer_start_cpu
        B = u_forcing.shape[0]
        ix = coll["ic"]["ix"].to(device=device, dtype=torch.long)
        iy = coll["ic"]["iy"].to(device=device, dtype=torch.long)
        batch_indices = torch.arange(B, device=device)[:, None]
        ic_target = u_ic[:, 0][
            batch_indices, ix[None, :], iy[None, :]
        ].unsqueeze(-1)
        if ic_target.shape != (B, int(ix.numel()), 1):
            raise AssertionError("online IC anchor gather returned the wrong shape")
        if residual_method == "finite_volume":
            left_qL = None
        else:
            _, yw_left, tw_left = coll["walls"]["left"]
            left_qL = left_wall_qL(
                online_records, yw_left, tw_left, device, t_ramp,
            )

        # Encode ONCE per step; every residual reuses this latent via the decode
        # closure. The latent is NOT detached (gradients flow to both encoders).
        train_start_cpu = time.perf_counter()
        train_start_event = train_end_event = None
        if profile_this and device.type == "cuda":
            train_start_event = torch.cuda.Event(enable_timing=True)
            train_end_event = torch.cuda.Event(enable_timing=True)
            train_start_event.record()
        latent = model.encode(u_forcing, u_ic)

        def _predict(coords, t, q_left=None, _latent=latent):
            return model.decode(_latent, coords, t)

        optimizer.zero_grad(set_to_none=True)
        causal_step_cfg = {**causal_cfg, "current_eps": causal_eps}
        if residual_method == "autodiff":
            losses = pino_losses(
                model, u_forcing, coll, ic_target, alpha,
                causal_cfg=causal_step_cfg,
                t_final=t_final, ic_loss="mse", t_right_tilde=t_right_tilde,
                left_qL=left_qL, sigma=float(sigma), k_slab=K_SLAB,
                predict=_predict,
            )
        else:
            losses = _forcing_ic_fv_losses(
                model=model,
                latent=latent,
                coll=coll,
                ic_target=ic_target,
                params_batch=online_records,
                ids_batch=None,
                problem=problem,
                collocation_ctx=collocation_ctx,
                geom_cfg=fv_geom_cfg,
                x_grid=x_grid_t,
                y_grid=y_grid_t,
                dt=float(fv_dt),
                n_steps=int(fv_n_steps),
                sigma_global=float(sigma),
                causal_cfg=causal_cfg,
                causal_eps=causal_eps,
                chunk_r=chunk_r,
            )
        warm_mult = {
            "r": warmup["r_mult"] if warming else 1.0,
            "ic": warmup["ic_mult"] if warming else 1.0,
            "bc_left": warmup["bc_left_mult"] if warming else 1.0,
            "bc_hom": warmup["bc_hom_mult"] if warming else 1.0,
            "bc": warmup["bc_hom_mult"] if warming else 1.0,
        }
        if lam_bc_left is None:
            w_bc = lam_bc
            w_bc_left = lam_bc
            sw = {"r": lam_r, "ic": lam_ic, "bc": lam_bc}
        else:
            w_bc = lam_bc
            w_bc_left = lam_bc_left
            sw = {"r": lam_r, "ic": lam_ic, "bc_hom": lam_bc, "bc_left": lam_bc_left}

        gn_mults: dict[str, float] = {}
        gradnorm_ms: float | str = ""
        if gradnorm is not None:
            active = {k: losses[k] for k in gradnorm.term_names if k in losses}
            gn_params = [p for p in model.parameters() if p.requires_grad]
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            gn_mults = gradnorm.maybe_update(active, gn_params, dist_info=None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gradnorm_ms = (time.perf_counter() - _t0) * 1000.0
        grad_cosines: dict[str, float | None] | None = None
        if do_val and gradnorm is not None:
            grad_cosines = _term_grad_cosines(
                {k: losses[k] for k in sw}, model.decoder.parameters(),
                pairs=gn_cos_pairs,
            )
        w_eff = {
            k: sw[k] * float(gn_mults.get(k, 1.0)) * warm_mult[k] for k in sw
        }
        loss = None
        for k, wk in w_eff.items():
            term = wk * losses[k]
            loss = term if loss is None else loss + term
        if not bool(torch.isfinite(loss).item()):
            _write_online_failure(
                run_dir,
                phase="pre_step",
                completed_updates=completed_updates,
                records=params_batch,
                batch_key=active_batch_key,
                coll=coll,
                rng_state={
                    "online": online_rngs.state_dict(),
                    "global": _capture_forcing_rng(rng, gen, eval_gen),
                },
            )
            raise FloatingPointError(f"Non-finite forcing-ic PINO loss at epoch {epoch}")
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            _write_online_failure(
                run_dir,
                phase="pre_step",
                completed_updates=completed_updates,
                records=params_batch,
                batch_key=active_batch_key,
                coll=coll,
                rng_state={
                    "online": online_rngs.state_dict(),
                    "global": _capture_forcing_rng(rng, gen, eval_gen),
                },
            )
            raise FloatingPointError(f"Non-finite forcing-ic PINO gradient at epoch {epoch}")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if not _all_finite(model.state_dict()) or not _all_finite(optimizer.state):
            _write_online_failure(
                run_dir,
                phase="post_step",
                completed_updates=completed_updates,
                records=params_batch,
                batch_key=active_batch_key,
                coll=coll,
                rng_state={
                    "online": online_rngs.state_dict(),
                    "global": _capture_forcing_rng(rng, gen, eval_gen),
                },
            )
            raise FloatingPointError(
                f"Non-finite forcing-ic model/optimizer state at epoch {epoch}"
            )
        if train_end_event is not None:
            train_end_event.record()
        train_seconds_cpu = time.perf_counter() - train_start_cpu
        sampling_fraction: float | str = ""
        if profile_this:
            if device.type == "cuda":
                train_end_event.synchronize()
                transfer_seconds = transfer_start_event.elapsed_time(
                    transfer_end_event
                ) / 1000.0
                train_seconds = train_start_event.elapsed_time(train_end_event) / 1000.0
            else:
                transfer_seconds = transfer_seconds_cpu
                train_seconds = train_seconds_cpu
            numerator = descriptor_seconds + ic_build_seconds + transfer_seconds
            sampling_fraction = numerator / max(train_seconds, 1e-12)
            profile_ratios.append(float(sampling_fraction))
            if (
                not profile_warned
                and len(profile_ratios) >= min_profile_samples
                and float(np.median(profile_ratios)) > warn_fraction
            ):
                warnings.warn(
                    "Online IC descriptor/build/normalization/transfer median "
                    f"overhead is {np.median(profile_ratios):.1%} of training time.",
                    RuntimeWarning,
                )
                profile_warned = True
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        completed_updates += 1

        eps_used = causal_eps
        causal_action = "off"
        if causal_cfg["enabled"]:
            populated = bool(losses["causal_populated"])
            causal_eps, causal_action = _adapt_causal_eps(
                causal_eps, losses["causal_weights"], causal_cfg, populated=populated,
            )
            if populated:
                causal_updates += 1
                if not causal_calibrated:
                    calibration = {}
                    bin_losses = losses["causal_bin_losses"]
                    for candidate in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                        cw = _causal_weights(bin_losses, candidate)
                        calibration[str(candidate)] = {
                            "mean": float(cw.mean().item()), "last": float(cw[-1].item()),
                        }
                    print(f"  causal epsilon calibration: {json.dumps(calibration)}", flush=True)
                    causal_calibrated = True

        row: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "residual_method": residual_method,
            "lr_first": lr,
            "lr_last": lr,
            "loss": float(loss.detach().cpu()),
            "loss_r": float(losses["r"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_bc": float(losses["bc"].detach().cpu()),
            "loss_bc_left": float(losses["bc_left"].detach().cpu()),
            "loss_bc_hom": float(losses["bc_hom"].detach().cpu()),
            "loss_right_dir": (
                float(losses["right_dir"].detach().cpu())
                if "right_dir" in losses else ""
            ),
            "loss_data": "",
            "w_r": lam_r * warm_mult["r"],
            "w_ic": lam_ic * warm_mult["ic"],
            "w_bc": w_bc * warm_mult["bc_hom"],
            "w_bc_left": w_bc_left * warm_mult["bc_left"],
            "w_data": 0.0,
            "val_gnrmse": "", "val_rmse_K": "",
            "gnrmse_amp_low": "", "gnrmse_amp_mid": "", "gnrmse_amp_high": "",
            "online_max_abs_z": online_diag["max_abs_z"],
            "online_frac_abs_z_gt_5": online_diag["frac_abs_z_gt_5"],
            "online_family_z_ranges": json.dumps(online_diag["per_family"]),
            "problem_keys": json.dumps([
                record["problem_key"] for record in params_batch
            ]),
            "batch_key": active_batch_key,
            "online_sampling_fraction": sampling_fraction,
        }
        for key in gn_cols:
            row[f"warm_mult_{key}"] = warm_mult[key]
        row["r_pointwise_mse"] = float(losses["r_pointwise_mse"].cpu())
        if causal_cfg["enabled"]:
            bins = losses["causal_bin_losses"].cpu()
            weights = losses["causal_weights"].cpu()
            counts = losses["causal_bin_counts"].cpu()
            row.update({
                "r_equal_bin_mean": float(losses["r_equal_bin_mean"].cpu()),
                "r_causal_loss": float(losses["r_causal_loss"].cpu()),
                "causal_reduction_ratio": float(losses["causal_reduction_ratio"].cpu()),
                "causal_eps": eps_used,
                "causal_eps_next": causal_eps,
                "causal_mean_weight": float(weights.mean()),
                "causal_last_weight": float(weights[-1]),
                "causal_log_last_weight": -eps_used * float(bins[:-1].sum()),
                "causal_adaptation_action": causal_action,
            })
            for i in causal_cols:
                row[f"causal_loss_bin_{i:02d}"] = float(bins[i])
                row[f"causal_weight_bin_{i:02d}"] = float(weights[i])
                row[f"causal_count_bin_{i:02d}"] = float(counts[i])
        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(r={row['loss_r']:.6f}, ic={row['loss_ic']:.6f}, "
            f"bc={row['loss_bc']:.6f}, bc_left={row['loss_bc_left']:.6f}) "
            f"lr={lr:.2e}" + ("  [warmup]" if warming else ""),
            flush=True,
        )

        if do_val:
            model.eval()
            val = validate_forcing_ic_gnrmse(
                model, data, data["val_ids"], sim_params,
                y_img, t_img, a_ref, t_ramp, device,
            )
            row["val_gnrmse"] = val["val_gnrmse"]
            row["val_rmse_K"] = val["val_rmse_K"]
            for c in ("gnrmse_amp_low", "gnrmse_amp_mid", "gnrmse_amp_high"):
                row[c] = val.get(c, "")
            if gradnorm is not None:
                raw_norms = gradnorm.last_raw_norms
                for k in sw:
                    m = float(gn_mults.get(k, 1.0))
                    row[f"gn_mult_{k}"] = m
                    row[f"w_eff_{k}"] = w_eff[k]
                    g_raw = raw_norms.get(k)
                    if g_raw is not None:
                        row[f"grad_norm_{k}"] = g_raw
                        row[f"grad_norm_eff_{k}"] = w_eff[k] * g_raw
                row["gradnorm_mean_mult"] = gradnorm.last_mean_multiplier
                row["gradnorm_ms"] = gradnorm_ms
                row["gradnorm_weights"] = json.dumps(
                    {k: float(v) for k, v in gn_mults.items()}
                )
                row["gradnorm_bound_hits"] = json.dumps(gradnorm.bound_hit_counts)
                if grad_cosines is not None:
                    row["grad_cosines"] = json.dumps(grad_cosines)
            is_best = val["val_gnrmse"] < best_val
            if is_best:
                best_val = val["val_gnrmse"]
            fam_txt = " ".join(
                f"{k.split('gnrmse_fam_')[1]}={v * 100:.2f}%"
                for k, v in val.items() if k.startswith("gnrmse_fam_")
            )
            print(
                f"Validation for epoch {epoch}: "
                f"val_gnrmse={val['val_gnrmse'] * 100:.4f}% "
                f"val_rmse_K={val['val_rmse_K']:.4f}K (best={best_val * 100:.4f}%)"
                + ("  [new best -> cvit_best.pt]" if is_best else ""),
                flush=True,
            )
            print(
                f"  gnrmse[amp low/mid/high]="
                f"{val.get('gnrmse_amp_low', float('nan')) * 100:.2f}/"
                f"{val.get('gnrmse_amp_mid', float('nan')) * 100:.2f}/"
                f"{val.get('gnrmse_amp_high', float('nan')) * 100:.2f}%  "
                f"fam[{fam_txt}]",
                flush=True,
            )
        else:
            is_best = False

        history.append({k: (v if v != "" else None) for k, v in row.items()})
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        last_csv_update = completed_updates

        def checkpoint_payload() -> dict[str, Any]:
            return {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "mu_global": mu, "sigma_global": sigma,
                "config": config,
                "resume_config": _forcing_resume_config(
                    config,
                    float(fv_dt) if residual_method == "finite_volume" else None,
                ),
                "epoch": epoch, "next_epoch": epoch + 1,
                "completed_updates": completed_updates,
                "last_csv_update": last_csv_update, "best_val": best_val,
                "gradnorm_state": gradnorm.state_dict() if gradnorm is not None else None,
                "causal_state": {
                    "eps": causal_eps, "n_bins": causal_cfg["n_bins"],
                    "updates": causal_updates, "calibrated": causal_calibrated,
                } if causal_cfg["enabled"] else None,
                "rng_state": _capture_forcing_rng(rng, gen, eval_gen),
                "online_rng_state": online_rngs.state_dict(),
                "ic_rng_state": copy.deepcopy(
                    online_rngs.numpy["ic_family"].bit_generator.state
                ),
                "online_sampling_signature": copy.deepcopy(online_signature),
                "online_environment": copy.deepcopy(environment),
                "fv_residual_signature": fv_signature,
                "fv_residual_metadata": (
                    {
                        "resolved_dt": float(fv_dt),
                        "dt_source": fv_dt_source,
                    }
                    if residual_method == "finite_volume" else None
                ),
                "online_cache": {
                    "records": copy.deepcopy(params_batch),
                    "batch_key": active_batch_key,
                    "coll": _pack_tensors(coll),
                },
                "forcing_cache": {
                    "params_batch": copy.deepcopy(params_batch),
                    "ids_batch": None,
                    "batch_key": active_batch_key,
                    "coll": _pack_tensors(coll),
                },
                "forcing_image": {
                    "ny_img": ny_img, "nt_img": nt_img, "a_ref": a_ref,
                    "t_ramp": t_ramp, "c_dom": c_dom, "d_dom": d_dom,
                    "t_final": t_final,
                },
            }

        payload = checkpoint_payload()
        if do_val and is_best:
            _atomic_torch_save(payload, best_path)
        if completed_updates % save_latest_every == 0 or epoch == epochs - 1:
            checkpoint_start = time.perf_counter()
            _atomic_torch_save(payload, latest_path)
            checkpoint_ms = (time.perf_counter() - checkpoint_start) * 1000.0
            checkpoint_mb = latest_path.stat().st_size / (1024.0 ** 2)
            print(
                f"  latest checkpoint: {checkpoint_ms:.1f} ms, {checkpoint_mb:.1f} MiB",
                flush=True,
            )

    summary = {
        "seed": seed,
        "best_val_gnrmse": best_val,
        "epochs": epochs,
        "residual_method": residual_method,
    }
    if residual_method == "finite_volume":
        summary["fv_dt"] = float(fv_dt)
        summary["fv_dt_source"] = fv_dt_source
    final_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    _atomic_torch_save(final_payload, final_path)
    _atomic_text(json.dumps(summary, indent=2) + "\n", run_dir / "final_metrics.json")
    _atomic_text("complete\n", complete_path)
    return summary


# ---- Physics-only (PINO) training of an InterfaceCViT on `interfaces` ---------
#
# The interfaces benchmark carries two materials (k_left=2, k_right=1) split by a
# per-sim interface location `interface_x` with per-sim contact resistance `R_c`,
# a fixed `sin`/`uniform` left-wall Neumann forcing, and a VARYING initial
# condition. The FV Crank-Nicolson residual (`full_bc_cn_residual` via
# `full_bc_physics_loss`) is the only physics; validation is against saved FV
# trajectories. The model is an `InterfaceCViT` (three token streams: spatial,
# forcing waveform, interface scalars) conditioned only on inference-available
# information -- it learns `(x_Gamma, R_c, a(.)) -> T`, never the synthetic
# generator parameters.
#
# UNITS CONTRACT (see `fv_residual.full_bc_cn_residual`): model outputs are
# NORMALIZED temperature (T_tilde); `FullBCData.qL_*` are PHYSICAL fluxes and
# `qL_int` is the EXACT step integral (the residual divides by `sigma_global`
# internally). The canonical `build_interface_forcing` emits both the sampled
# waveform the model sees and the physical flux the residual enforces from the
# same parameters, so they can never drift.
#
# NON-CAUSAL FORCING (documented): the model sees the COMPLETE sampled waveform
# over [0, t_final] while predicting earlier times -- valid for a fully-known
# loading schedule (forward/design), not a causal real-time predictor.


def _interface_spatial_channels(
    T0: np.ndarray, interface_x: float, x_grid: np.ndarray,
    mu: float, sigma: float,
) -> np.ndarray:
    """Spatial encoder channels `[T0_tilde, K_norm, D_norm]` -> `(3, Nx, Ny)`.

    Mirrors the FNO's interfaces spatial layout (`problems/interfaces`
    `_material_channel` / `_signed_distance_channel`): `T0_tilde` is the
    normalized IC field, `K_norm = (k(x) - 1.5)/0.5` the material map, and
    `D_norm = (x - x_Gamma)/(x[-1]-x[0])` the signed distance. `K_norm`/`D_norm`
    depend on x only and broadcast across y.
    """
    T0 = np.asarray(T0, dtype=np.float32)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    Ny = T0.shape[1]
    t0_tilde = np.empty_like(T0)
    np.subtract(T0, np.float32(mu), out=t0_tilde)
    np.divide(t0_tilde, np.float32(sigma), out=t0_tilde)
    kx = np.where(x_grid <= float(interface_x), K_LEFT, K_RIGHT)
    k_norm = ((kx - 1.5) / 0.5)[:, None] * np.ones((1, Ny))
    span = float(x_grid[-1] - x_grid[0])
    d_norm = ((x_grid - float(interface_x)) / span)[:, None] * np.ones((1, Ny))
    return np.stack([t0_tilde, k_norm, d_norm], axis=0).astype(np.float32)


def _interface_spatial_channels_from_normalized(
    T0_tilde: np.ndarray, interface_x: float, x_grid: np.ndarray,
) -> np.ndarray:
    T0_tilde = np.asarray(T0_tilde, dtype=np.float32)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    Ny = T0_tilde.shape[1]
    kx = np.where(x_grid <= float(interface_x), K_LEFT, K_RIGHT)
    k_norm = np.broadcast_to(((kx - 1.5) / 0.5)[:, None], T0_tilde.shape)
    span = float(x_grid[-1] - x_grid[0])
    d_norm = np.broadcast_to(
        ((x_grid - float(interface_x)) / span)[:, None], T0_tilde.shape,
    )
    return np.stack([T0_tilde, k_norm, d_norm], axis=0).astype(np.float32)


def _decode_in_chunks(
    model: InterfaceCViT, latent: torch.Tensor,
    coords: torch.Tensor, t: torch.Tensor, chunk: int,
) -> torch.Tensor:
    """Differentiable `model.decode` split over the query axis (dim 1).

    Cross-attention memory scales with the query count, so the full-grid FV
    residual decode is chunked to bound peak memory; chunk outputs are
    concatenated (autograd preserved) so the residual sees the whole grid. A
    non-positive `chunk` (or a query set that already fits) decodes in one shot.
    """
    Nq = coords.shape[1]
    if chunk is None or chunk <= 0 or Nq <= chunk:
        return model.decode(latent, coords, t)
    outs = []
    for s in range(0, Nq, chunk):
        e = min(s + chunk, Nq)
        outs.append(model.decode(latent, coords[:, s:e, :], t[:, s:e, :]))
    return torch.cat(outs, dim=1)


def _sample_interval_times(
    rng: np.random.Generator, batch_size: int, intervals_per_sim: int,
    dt: float, t_final: float, stratified: bool, n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw CN interval midpoints over `M = batch_size * intervals_per_sim`.

    Stratified sampling assigns bins across the WHOLE collection of intervals
    (not per sim), so every bin is populated whenever `M >= n_bins` -- the
    coverage causal weighting needs. Each `t_mid` is drawn uniformly inside its
    assigned bin's sub-window of the feasible midpoint range
    `[dt/2, t_final - dt/2]`, then `t_n = t_mid - dt/2`, `t_np1 = t_mid + dt/2`
    stay inside `[0, t_final]` by construction (no rejection). Non-stratified
    falls back to a uniform midpoint draw; `bin_ids` is then unused.
    """
    M = int(batch_size * intervals_per_sim)
    lo = dt / 2.0
    hi = t_final - dt / 2.0
    if stratified and M >= 1:
        bin_ids = np.arange(M) % int(n_bins)
        rng.shuffle(bin_ids)
        edges = np.linspace(lo, hi, int(n_bins) + 1)
        t_mid = np.empty(M, dtype=np.float64)
        for i in range(M):
            k = int(bin_ids[i])
            t_mid[i] = rng.uniform(edges[k], edges[k + 1])
    else:
        bin_ids = np.zeros(M, dtype=np.int64)
        t_mid = rng.uniform(lo, hi, size=M)
    t_n = t_mid - dt / 2.0
    t_np1 = t_mid + dt / 2.0
    return t_n, t_np1, t_mid, bin_ids


def _interface_face_idx(x_grid: np.ndarray, interface_x: float) -> int:
    """Discrete x-face carrying the series conductance (via `locate_interface`)."""
    return int(locate_interface(x_grid, float(interface_x)).face_idx)


def _compute_train_jump_scale(
    data: dict[str, Any], sim_params: np.ndarray, ids: np.ndarray,
) -> float:
    """Frozen `sigma_dT,train = RMS_train(T_Gamma^- - T_Gamma^+)` in Kelvin.

    The node jump uses the two interface-flanking nodes `[face_idx]`/`[face_idx+1]`
    from `locate_interface`. Computed ONCE from the training trajectories and
    frozen; the val jump metric normalizes by this so `E_zero` is stable and not
    coupled to the val split (plan Section 6).
    """
    x_grid = np.asarray(data["x_grid"], dtype=np.float64)
    traj = data["trajectories"]
    ssq = 0.0
    cnt = 0
    for i in np.asarray(ids):
        p = dict(sim_params[int(i)])
        fidx = _interface_face_idx(x_grid, p["interface_x"])
        tr = np.asarray(traj[int(i)], dtype=np.float64)          # (Nt, Nx, Ny)
        dT = tr[:, fidx, :] - tr[:, fidx + 1, :]                 # (Nt, Ny)
        ssq += float(np.sum(dT ** 2))
        cnt += int(dT.size)
    return math.sqrt(ssq / max(cnt, 1))


def validate_interfaces_gnrmse(
    model: InterfaceCViT,
    data: dict[str, Any],
    ids: np.ndarray,
    sim_params: np.ndarray,
    t_ramp: float,
    a_ref: float,
    y_img: np.ndarray,
    t_img: np.ndarray,
    sigma_dT_train: float,
    device: torch.device,
    query_batch: int = 8,
) -> dict[str, float]:
    """Deviation-field gnRMSE + interface-jump metrics over held-out sims.

    Beyond the global deviation gnRMSE (`rmse_K / sigma`, amplitude-fair on the
    300 K baseline), this reports the node interface jump error under the FROZEN
    `sigma_dT_train` denominator:

        E_model = RMS_val(dT_pred - dT_true) / sigma_dT_train
        E_zero  = RMS_val(dT_true)           / sigma_dT_train   (zero-jump baseline)

    so a good global fit that smooths the jump stays visible. Per-sim gnRMSE is
    stratified by `R_c` and by `interface_x` terciles. Inputs are rebuilt with
    the same spatial/image builders the trainer uses.
    """
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny, Nt = int(x_grid.numel()), int(y_grid.numel()), len(t_grid)

    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)

    ids = np.asarray(ids)
    per_sim_rmse: list[float] = []
    per_sim_rc: list[float] = []
    per_sim_ix: list[float] = []
    jump_err_ssq = 0.0
    jump_true_ssq = 0.0
    jump_cnt = 0
    for start in range(0, len(ids), query_batch):
        chunk = ids[start:start + query_batch]
        params = [dict(sim_params[int(i)]) for i in chunk]
        B = len(params)
        u_spatial = torch.from_numpy(
            np.stack([
                _interface_spatial_channels(
                    p["T0"], p["interface_x"], x_grid_np, mu, sigma
                ) for p in params
            ])
        ).to(device)
        forcing_image = build_forcing_image(
            params, y_img, t_img, a_ref, device, t_ramp
        )
        pscal = torch.from_numpy(
            np.stack([
                normalize_interface_scalars(p["interface_x"], p["R_c"])
                for p in params
            ])
        ).to(device)
        latent = model.encode(u_spatial, forcing_image, pscal)
        coords = mesh.expand(B, -1, -1)
        pred = torch.empty((B, Nt, Nx, Ny), device=device)
        for k in range(Nt):
            tk = torch.full((B, Nx * Ny, 1), float(t_grid[k]), device=device)
            out = model.decode(latent, coords, tk)
            pred[:, k] = out[..., 0].view(B, Nx, Ny)
        pred_K = (pred * sigma + mu).detach().cpu().numpy()
        truth = np.asarray(data["trajectories"][chunk], dtype=np.float64)
        se = ((pred_K - truth) ** 2).sum(axis=(1, 2, 3))          # (B,)
        rmse = np.sqrt(se / float(Nt * Nx * Ny))
        for b, p in enumerate(params):
            per_sim_rmse.append(float(rmse[b]))
            per_sim_rc.append(float(p["R_c"]))
            per_sim_ix.append(float(p["interface_x"]))
            fidx = _interface_face_idx(x_grid_np, p["interface_x"])
            dT_pred = pred_K[b][:, fidx, :] - pred_K[b][:, fidx + 1, :]  # (Nt,Ny)
            dT_true = truth[b][:, fidx, :] - truth[b][:, fidx + 1, :]
            jump_err_ssq += float(np.sum((dT_pred - dT_true) ** 2))
            jump_true_ssq += float(np.sum(dT_true ** 2))
            jump_cnt += int(dT_true.size)

    rmse_arr = np.asarray(per_sim_rmse, dtype=np.float64)
    rc_arr = np.asarray(per_sim_rc, dtype=np.float64)
    ix_arr = np.asarray(per_sim_ix, dtype=np.float64)
    gnrmse = rmse_arr / (float(sigma) + 1e-8)
    denom = float(sigma_dT_train) + 1e-12
    node_jump_rmse_K = math.sqrt(jump_err_ssq / max(jump_cnt, 1))
    out = {
        "val_gnrmse": float(gnrmse.mean()),
        "val_rmse_K": float(rmse_arr.mean()),
        "node_jump_rmse_K": node_jump_rmse_K,
        "E_model": node_jump_rmse_K / denom,
        "E_zero": math.sqrt(jump_true_ssq / max(jump_cnt, 1)) / denom,
    }
    for tag, arr in (("Rc", rc_arr), ("ix", ix_arr)):
        if len(arr) >= 3:
            q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
            strata = {
                "low": arr <= q1,
                "mid": (arr > q1) & (arr <= q2),
                "high": arr > q2,
            }
            for name, mask in strata.items():
                out[f"gnrmse_{tag}_{name}"] = (
                    float(gnrmse[mask].mean()) if mask.any() else float("nan")
                )
    return out


def _validate_forcing_layered_dataset(
    problem, data: dict[str, Any], sim_params: np.ndarray, dt: float,
) -> tuple[dict[str, Any], int, tuple[float, float]]:
    x = np.asarray(data["x_grid"], dtype=np.float64)
    y = np.asarray(data["y_grid"], dtype=np.float64)
    t = np.asarray(data["t_grid"], dtype=np.float64)
    if x.size < 7 or y.size < 3:
        raise ValueError("forcing hybrid residual requires Nx >= 7 and Ny >= 3")
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    if not np.allclose(np.diff(x), hx) or not np.allclose(np.diff(y), hy):
        raise ValueError("forcing hybrid residual requires uniform x/y grids")
    if not np.isclose(hx, hy, rtol=1e-10, atol=1e-12):
        raise ValueError("forcing hybrid residual requires the solver's isotropic grid")
    physics = problem.physics_parameters(
        float(x[0]), float(x[-1]), float(y[0]), float(y[-1])
    )
    interface_x = float(physics["interface_x"])
    location = locate_interface(x, interface_x)
    f = int(location.face_idx)
    if not np.isclose(interface_x, 0.5 * (x[f] + x[f + 1]), atol=1e-12):
        raise ValueError(
            f"interface_x={interface_x} is not face-aligned on the selected grid"
        )
    if f - 2 < 0 or f + 3 >= x.size:
        raise ValueError("interface trace reconstruction requires f-2 >= 0 and f+3 < Nx")
    if not np.isclose(float(t[-1]), 0.3, rtol=0.0, atol=5e-8):
        raise ValueError(f"forcing reference time expects t_final=0.3, got {t[-1]}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("forcing FV residual requires a positive solver dt")
    for sid, raw in enumerate(sim_params):
        params = dict(raw)
        if "interface_x" in params and not np.isclose(
            float(params["interface_x"]), interface_x, atol=1e-8
        ):
            raise ValueError(
                f"sim_params[{sid}] interface_x conflicts with ForcingProblem physics"
            )
        for key in ("k_left", "k_right", "rho_left", "rho_right", "cp_left", "cp_right"):
            if key in params and not np.isclose(float(params[key]), float(physics[key])):
                raise ValueError(
                    f"sim_params[{sid}] {key} conflicts with ForcingProblem physics"
                )
        if "T_right" in params and not np.isclose(
            float(params["T_right"]), float(physics["T_right"])
        ):
            raise ValueError(
                f"sim_params[{sid}] T_right conflicts with ForcingProblem physics"
            )
    exclusion = (0.5 * (x[f - 1] + x[f]), 0.5 * (x[f + 1] + x[f + 2]))
    return physics, f, exclusion


def _quadratic_trace_weights(nodes: np.ndarray, x_eval: float) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=np.float64)
    vandermonde = np.stack((np.ones(3), nodes, nodes ** 2), axis=1)
    inverse = np.linalg.inv(vandermonde)
    value = np.array([1.0, x_eval, x_eval ** 2]) @ inverse
    derivative = np.array([0.0, 1.0, 2.0 * x_eval]) @ inverse
    return value, derivative


def validate_forcing_interface_gnrmse(
    model: InterfaceCViT,
    data: dict[str, Any],
    ids: np.ndarray,
    sim_params: np.ndarray,
    *,
    physics: dict[str, Any],
    t_ramp: float,
    q_ref: float,
    y_img: np.ndarray,
    t_img: np.ndarray,
    t_ref: float,
    device: torch.device,
    query_batch: int = 8,
) -> dict[str, float]:
    x_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_np = np.asarray(data["y_grid"], dtype=np.float64)
    t_np = np.asarray(data["t_grid"], dtype=np.float64)
    x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    mu, sigma = float(data["mu_global"]), float(data["sigma_global"])
    Nx, Ny, Nt = x_np.size, y_np.size, t_np.size
    interface_x = float(physics["interface_x"])
    f = locate_interface(x_np, interface_x).face_idx
    if f - 2 < 0 or f + 3 >= Nx:
        raise ValueError("interface trace reconstruction stencil lies outside the grid")
    left_nodes = np.array([f - 2, f - 1, f])
    right_nodes = np.array([f + 1, f + 2, f + 3])
    left_value_w, left_deriv_w = _quadratic_trace_weights(
        x_np[left_nodes], interface_x
    )
    right_value_w, right_deriv_w = _quadratic_trace_weights(
        x_np[right_nodes], interface_x
    )
    gx, gy = torch.meshgrid(x_t, y_t, indexing="ij")
    mesh = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).unsqueeze(0)

    per_sim_rmse: list[float] = []
    per_sim_lead_rmse: list[np.ndarray] = []
    temporal_families: list[str] = []
    spatial_families: list[str] = []
    jump_ssq = flux_ssq = contact_ssq = energy_ssq = 0.0
    jump_count = flux_count = energy_count = 0
    for start in range(0, len(ids), query_batch):
        chunk = np.asarray(ids[start:start + query_batch])
        params = []
        for sid in chunk:
            record = dict(sim_params[int(sid)])
            record.setdefault("interface_x", interface_x)
            params.append(record)
        B = len(params)
        fixed_ic = np.full((Nx, Ny), float(physics["T_right"]), dtype=np.float32)
        u_spatial = torch.from_numpy(np.stack([
            _interface_spatial_channels(
                fixed_ic, interface_x, x_np, mu, sigma
            ) for _ in params
        ])).to(device)
        forcing_image = build_forcing_image(
            params, y_img, t_img, q_ref, device, t_ramp
        )
        pscal = torch.from_numpy(np.stack([
            normalize_forcing_interface_scalars(interface_x, p["R_c"])
            for p in params
        ])).to(device)
        latent = model.encode(u_spatial, forcing_image, pscal)
        coords = mesh.expand(B, -1, -1)
        pred = torch.empty((B, Nt, Nx, Ny), device=device)
        for index, time_value in enumerate(t_np):
            tq = torch.full(
                (B, Nx * Ny, 1), float(time_value), device=device
            )
            pred[:, index] = model.decode(latent, coords, tq)[..., 0].view(B, Nx, Ny)
        pred_norm = pred.detach().cpu().numpy().astype(np.float64)
        pred_K = pred_norm * sigma + mu
        truth = np.asarray(data["trajectories"][chunk], dtype=np.float64)
        rmse = np.sqrt(((pred_K - truth) ** 2).mean(axis=(1, 2, 3)))
        per_sim_rmse.extend(float(value) for value in rmse)
        per_sim_lead_rmse.extend(
            np.sqrt(((pred_K - truth) ** 2).mean(axis=(2, 3)))
        )
        temporal_families.extend(str(p["temporal_family"]) for p in params)
        spatial_families.extend(str(p["spatial_family"]) for p in params)

        jump_error = (
            pred_K[:, :, f, :] - pred_K[:, :, f + 1, :]
            - (truth[:, :, f, :] - truth[:, :, f + 1, :])
        )
        jump_ssq += float(np.square(jump_error).sum())
        jump_count += int(jump_error.size)

        left_field = pred_K[:, :, left_nodes, :]
        right_field = pred_K[:, :, right_nodes, :]
        T_minus = np.einsum("k,btkj->btj", left_value_w, left_field)
        T_plus = np.einsum("k,btkj->btj", right_value_w, right_field)
        dTdx_minus = np.einsum("k,btkj->btj", left_deriv_w, left_field)
        dTdx_plus = np.einsum("k,btkj->btj", right_deriv_w, right_field)
        q_minus = -float(physics["k_left"]) * dTdx_minus
        q_plus = -float(physics["k_right"]) * dTdx_plus
        flux_error = q_minus - q_plus
        rc = np.asarray([float(p["R_c"]) for p in params])[:, None, None]
        contact_error = T_minus - T_plus - rc * 0.5 * (q_minus + q_plus)
        flux_ssq += float(np.square(flux_error).sum())
        contact_ssq += float(np.square(contact_error).sum())
        flux_count += int(flux_error.size)

        if Nt > 1:
            dt_val = float(t_np[1] - t_np[0])
            interval_count = Nt - 1
            Tn = pred[:, :-1].reshape(B * interval_count, Nx, Ny)
            Tnp1 = pred[:, 1:].reshape(B * interval_count, Nx, Ny)
            rc_rep = np.repeat(np.asarray([float(p["R_c"]) for p in params]), interval_count)
            iface_rep = np.full(B * interval_count, interface_x)
            geom = build_cn_geom_per_interface(
                x_np, y_np, physics["k_left"], physics["k_right"],
                iface_rep, rc_rep, dt_val, sigma_global=sigma,
                rho=physics["rho_left"], cp=physics["cp_left"],
                device=device, dtype=pred.dtype,
            )
            energy = float(t_ref) * interface_residual_rate(Tn, Tnp1, geom)
            energy_ssq += float(energy.square().sum().cpu())
            energy_count += int(energy.numel())

    rmse_arr = np.asarray(per_sim_rmse, dtype=np.float64)
    lead_rmse = np.asarray(per_sim_lead_rmse, dtype=np.float64)
    out = {
        "val_gnrmse": float((rmse_arr / (sigma + 1e-8)).mean()),
        "val_rmse_K": float(rmse_arr.mean()),
        "node_jump_rmse_K": math.sqrt(jump_ssq / max(jump_count, 1)),
        "interface_flux_mismatch": math.sqrt(flux_ssq / max(flux_count, 1)) / q_ref,
        "interface_contact_rmse_K": math.sqrt(contact_ssq / max(flux_count, 1)),
        "interface_contact_rmse_norm": (
            math.sqrt(contact_ssq / max(flux_count, 1)) / (sigma + 1e-8)
        ),
        "interface_energy_rms": math.sqrt(energy_ssq / max(energy_count, 1)),
    }
    positive_leads = t_np > 0.0
    lead_edges = (t_np[-1] / 3.0, 2.0 * t_np[-1] / 3.0)
    lead_masks = {
        "short": positive_leads & (t_np <= lead_edges[0]),
        "mid": (t_np > lead_edges[0]) & (t_np <= lead_edges[1]),
        "long": t_np > lead_edges[1],
    }
    for name, mask in lead_masks.items():
        if mask.any():
            out[f"gnrmse_lead_{name}"] = float(
                (lead_rmse[:, mask] / (sigma + 1e-8)).mean()
            )
    for prefix, values in (
        ("temporal", np.asarray(temporal_families)),
        ("spatial", np.asarray(spatial_families)),
    ):
        for family in np.unique(values):
            mask = values == family
            out[f"gnrmse_{prefix}_{family}"] = float(
                (rmse_arr[mask] / (sigma + 1e-8)).mean()
            )
    return out


def _open_uniform(
    shape: tuple[int, ...], low: float, high: float, *,
    device: torch.device, generator: torch.Generator,
) -> torch.Tensor:
    if not high > low:
        raise ValueError(f"empty open sampling interval ({low}, {high})")
    eps = 16.0 * torch.finfo(torch.float32).eps
    unit = torch.rand(shape, device=device, generator=generator)
    unit = eps + (1.0 - 2.0 * eps) * unit
    return float(low) + (float(high) - float(low)) * unit


def _sample_hybrid_bulk(
    batch_size: int,
    n_points: int,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    exclusion: tuple[float, float],
    t_final: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = map(float, x_bounds)
    c, d = map(float, y_bounds)
    exclusion_left, exclusion_right = map(float, exclusion)
    left_width = exclusion_left - a
    right_width = b - exclusion_right
    if left_width <= 0.0 or right_width <= 0.0:
        raise ValueError("interface exclusion removes an entire material interior")
    side = torch.rand(
        (batch_size, n_points, 1), device=device, generator=generator
    ) < left_width / (left_width + right_width)
    x_left = _open_uniform(
        (batch_size, n_points, 1), a, exclusion_left,
        device=device, generator=generator,
    )
    x_right = _open_uniform(
        (batch_size, n_points, 1), exclusion_right, b,
        device=device, generator=generator,
    )
    x = torch.where(side, x_left, x_right)
    y = _open_uniform(
        (batch_size, n_points, 1), c, d,
        device=device, generator=generator,
    )
    t = _open_uniform(
        (batch_size, n_points, 1), 0.0, t_final,
        device=device, generator=generator,
    )
    return torch.cat((x, y), dim=-1).detach().requires_grad_(True), t.detach().requires_grad_(True)


def _decode_derivatives(
    model: InterfaceCViT,
    latent: torch.Tensor,
    coords: torch.Tensor,
    time_query: torch.Tensor,
    *,
    second_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    prediction = model.decode(latent, coords, time_query)
    ones = torch.ones_like(prediction)
    grad_xy = torch.autograd.grad(
        prediction, coords, ones, create_graph=True, retain_graph=True
    )[0]
    grad_t = torch.autograd.grad(
        prediction, time_query, ones, create_graph=True, retain_graph=True
    )[0]
    if not second_order:
        return prediction, grad_xy, grad_t, None, None
    grad_xx = torch.autograd.grad(
        grad_xy[..., 0:1], coords, torch.ones_like(grad_xy[..., 0:1]),
        create_graph=True, retain_graph=True,
    )[0][..., 0:1]
    grad_yy = torch.autograd.grad(
        grad_xy[..., 1:2], coords, torch.ones_like(grad_xy[..., 1:2]),
        create_graph=True, retain_graph=True,
    )[0][..., 1:2]
    return prediction, grad_xy, grad_t, grad_xx, grad_yy


def _forcing_q_at_batch(
    params: list[dict], y: torch.Tensor, t: torch.Tensor, t_ramp: float,
) -> torch.Tensor:
    y_np = y.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    values = np.empty(y_np.shape, dtype=np.float32)
    for index, record in enumerate(params):
        forcing = reconstruct_qL(
            record["temporal_family"], record["temporal_params"],
            record["spatial_family"], record["spatial_params"],
            t_ramp=t_ramp,
        )
        values[index, :, 0] = forcing.evaluate_points(
            y_np[index, :, 0], t_np[index, :, 0]
        )
    return torch.from_numpy(values).to(device=y.device, dtype=y.dtype)


def _forcing_hybrid_ad_losses(
    model: InterfaceCViT,
    latent: torch.Tensor,
    params: list[dict],
    *,
    physics: dict[str, Any],
    exclusion: tuple[float, float],
    sigma: float,
    q_ref: float,
    t_ref: float,
    t_final: float,
    t_ramp: float,
    n_r: int,
    n_bc: int,
    bulk_generator: torch.Generator,
    boundary_generator: torch.Generator,
    active_t_final: float | None = None,
) -> dict[str, torch.Tensor]:
    device = latent.device
    batch_size = latent.shape[0]
    x_bounds = (float(physics["a"]), float(physics["b"]))
    y_bounds = (float(physics["c"]), float(physics["d"]))
    interface_x = float(physics["interface_x"])

    sample_t_final = float(t_final if active_t_final is None else active_t_final)
    coords, tq = _sample_hybrid_bulk(
        batch_size, n_r, x_bounds=x_bounds, y_bounds=y_bounds,
        exclusion=exclusion, t_final=sample_t_final, device=device,
        generator=bulk_generator,
    )
    _, _, grad_t, grad_xx, grad_yy = _decode_derivatives(
        model, latent, coords, tq, second_order=True
    )
    alpha_left = float(physics["k_left"]) / (
        float(physics["rho_left"]) * float(physics["cp_left"])
    )
    alpha_right = float(physics["k_right"]) / (
        float(physics["rho_right"]) * float(physics["cp_right"])
    )
    alpha = torch.where(
        coords[..., 0:1] < interface_x,
        torch.as_tensor(alpha_left, device=device, dtype=coords.dtype),
        torch.as_tensor(alpha_right, device=device, dtype=coords.dtype),
    )
    bulk_rate = grad_t - alpha * (grad_xx + grad_yy)
    bulk = (float(t_ref) * bulk_rate).square().flatten(1).mean(dim=1).mean()

    def _boundary_coords(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        free = _open_uniform(
            (batch_size, n_bc, 1), 0.0, 1.0,
            device=device, generator=boundary_generator,
        )
        if kind != "left":
            interface = torch.as_tensor(
                interface_x, device=device, dtype=free.dtype
            )
            shift = 32.0 * torch.finfo(free.dtype).eps
            free = torch.where(
                (free - interface).abs() <= shift, free + shift, free
            )
        time = _open_uniform(
            (batch_size, n_bc, 1), 0.0, sample_t_final,
            device=device, generator=boundary_generator,
        )
        if kind == "left":
            xy = torch.cat((torch.zeros_like(free), free), dim=-1)
        elif kind == "top":
            xy = torch.cat((free, torch.ones_like(free)), dim=-1)
        elif kind == "bottom":
            xy = torch.cat((free, torch.zeros_like(free)), dim=-1)
        else:
            raise ValueError(f"unknown boundary {kind!r}")
        return xy.detach().requires_grad_(True), time.detach().requires_grad_(True)

    left_coords, left_t = _boundary_coords("left")
    _, left_grad, _, _, _ = _decode_derivatives(
        model, latent, left_coords, left_t, second_order=False
    )
    q_left = _forcing_q_at_batch(
        params, left_coords[..., 1:2], left_t, t_ramp
    )
    left_flux = -float(physics["k_left"]) * float(sigma) * left_grad[..., 0:1]
    left_residual = (left_flux - q_left) / float(q_ref)
    left = left_residual.square().flatten(1).mean(dim=1).mean()

    wall_losses = {}
    for wall in ("top", "bottom"):
        wall_coords, wall_t = _boundary_coords(wall)
        _, wall_grad, _, _, _ = _decode_derivatives(
            model, latent, wall_coords, wall_t, second_order=False
        )
        conductivity = torch.where(
            wall_coords[..., 0:1] < interface_x,
            torch.as_tensor(physics["k_left"], device=device, dtype=wall_coords.dtype),
            torch.as_tensor(physics["k_right"], device=device, dtype=wall_coords.dtype),
        )
        wall_flux = conductivity * float(sigma) * wall_grad[..., 1:2]
        wall_losses[wall] = (
            (wall_flux / float(q_ref)).square().flatten(1).mean(dim=1).mean()
        )

    return {
        "interior": bulk,
        "left_neumann": left,
        "top_adiabatic": wall_losses["top"],
        "bottom_adiabatic": wall_losses["bottom"],
        "topbot_adiabatic": 0.5 * (
            wall_losses["top"] + wall_losses["bottom"]
        ),
        "_constraint_left_flux": left_residual,
    }


def _mean_problem_intervals(
    values: torch.Tensor, batch_size: int, intervals_per_sim: int,
) -> torch.Tensor:
    if values.shape != (batch_size * intervals_per_sim,):
        raise ValueError(
            f"expected {batch_size * intervals_per_sim} interval losses, got {tuple(values.shape)}"
        )
    return values.view(batch_size, intervals_per_sim).mean(dim=1).mean()


def _forcing_interval_fluxes(
    params: list[dict],
    sim_local: torch.Tensor,
    y_grid: torch.Tensor,
    tn: torch.Tensor,
    tnp1: torch.Tensor,
    t_ramp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interval_count = int(sim_local.numel())
    Ny = int(y_grid.numel())
    qn = np.empty((interval_count, Ny), dtype=np.float64)
    qnp1 = np.empty_like(qn)
    qint = np.empty_like(qn)
    y_np = y_grid.detach().cpu().numpy()
    sim_np = sim_local.detach().cpu().numpy()
    tn_np = tn.detach().cpu().numpy()
    tnp1_np = tnp1.detach().cpu().numpy()
    for interval in range(interval_count):
        record = params[int(sim_np[interval])]
        qn_i, qnp1_i, qint_i = build_interface_forcing(
            record["temporal_family"], record["temporal_params"],
            record["spatial_family"], record["spatial_params"],
            y_np, float(tn_np[interval]), float(tnp1_np[interval]), t_ramp,
        )
        qn[interval], qnp1[interval], qint[interval] = qn_i, qnp1_i, qint_i
    return qn, qnp1, qint


def _forcing_energy_closure_residuals(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom: Any,
    qL_int: torch.Tensor,
    *,
    interface_face: int,
    physics: dict[str, Any],
    sigma: float,
    q_ref: float,
    scale_floor: float,
) -> dict[str, torch.Tensor]:
    interval_count, Nx, Ny = T_n.shape
    device, dtype = T_n.device, T_n.dtype
    dt_values = torch.full(
        (interval_count,), float(geom.dt), device=device, dtype=dtype,
    )
    dx = geom.dx
    if dx.dim() == 1:
        dx = dx.unsqueeze(0).expand(interval_count, -1)
    dy = geom.dy.to(device=device, dtype=dtype)
    cell_area = dx.to(device=device, dtype=dtype)[:, :, None] * dy[None, None, :]
    rho_cp = geom.rho_cp.to(device=device, dtype=dtype)
    if rho_cp.dim() == 2:
        rho_cp = rho_cp.unsqueeze(0)
    dE_cell = rho_cp * cell_area * (float(sigma) * (T_np1 - T_n))

    x_ids = torch.arange(Nx, device=device)[None, :]
    face_idx = torch.full(
        (interval_count, 1), int(interface_face), device=device, dtype=torch.long,
    )
    left_mask = x_ids <= face_idx
    right_mask = x_ids > face_idx
    dE_left = (dE_cell * left_mask[:, :, None].to(dtype)).sum(dim=(1, 2))
    dE_right = (dE_cell * right_mask[:, :, None].to(dtype)).sum(dim=(1, 2))

    g_interface = geom.G_x.to(device=device, dtype=dtype).gather(
        1, face_idx[:, :, None].expand(interval_count, 1, Ny)
    ).squeeze(1)
    Tn_jump = float(sigma) * (
        T_n[:, int(interface_face), :] - T_n[:, int(interface_face) + 1, :]
    )
    Tnp1_jump = float(sigma) * (
        T_np1[:, int(interface_face), :] - T_np1[:, int(interface_face) + 1, :]
    )
    q_gamma_integral = (
        0.5 * dt_values[:, None] * g_interface * (Tn_jump + Tnp1_jump)
    )
    Q_gamma = (q_gamma_integral * dy[None, :]).sum(dim=1)
    Q_left = (qL_int.to(device=device, dtype=dtype) * dy[None, :]).sum(dim=1)

    h_right = float(geom.hx)
    qR_n = -float(physics["k_right"]) * float(sigma) * (
        T_n[:, -1, :] - T_n[:, -2, :]
    ) / h_right
    qR_np1 = -float(physics["k_right"]) * float(sigma) * (
        T_np1[:, -1, :] - T_np1[:, -2, :]
    ) / h_right
    Q_right = (
        0.5 * dt_values[:, None] * (qR_n + qR_np1) * dy[None, :]
    ).sum(dim=1)

    characteristic = (
        float(q_ref) * dt_values * float(dy.sum().detach().cpu())
        * max(float(scale_floor), 0.0)
    )
    denom = (Q_left.detach().abs() + characteristic).clamp_min(1.0e-12)
    return {
        "energy_left_residual": (dE_left - (Q_left - Q_gamma)) / denom,
        "energy_right_residual": (dE_right - (Q_gamma - Q_right)) / denom,
        "energy_global_residual": (
            dE_left + dE_right - (Q_left - Q_right)
        ) / denom,
    }


def _forcing_energy_closure_losses(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom: Any,
    qL_int: torch.Tensor,
    *,
    interface_face: int,
    physics: dict[str, Any],
    sigma: float,
    q_ref: float,
    scale_floor: float,
) -> dict[str, torch.Tensor]:
    residuals = _forcing_energy_closure_residuals(
        T_n, T_np1, geom, qL_int,
        interface_face=interface_face, physics=physics, sigma=sigma,
        q_ref=q_ref, scale_floor=scale_floor,
    )
    loss_left = residuals["energy_left_residual"].square().mean()
    loss_right = residuals["energy_right_residual"].square().mean()
    loss_global = residuals["energy_global_residual"].square().mean()
    return {
        "energy_left": loss_left,
        "energy_right": loss_right,
        "energy_global": loss_global,
        "energy": loss_left + loss_right + loss_global,
    }


def _sample_online_layered_forcing_params(
    rngs: HybridForcingRNGs,
    n: int,
    *,
    dt: float,
    t_final: float,
    y_bounds: tuple[float, float],
    temporal_window: dict[str, float],
) -> list[dict]:
    params = sample_forcing_params(
        rngs.forcing_selection,
        n,
        dt,
        t_final,
        c=float(y_bounds[0]),
        d=float(y_bounds[1]),
        temporal_window=temporal_window,
    )
    resistance = rngs.contact_resistance.uniform(
        FORCING_RC_RANGE[0], FORCING_RC_RANGE[1], size=int(n)
    )
    for record, value in zip(params, resistance):
        record["R_c"] = float(value)
        record["interface_x"] = FORCING_INTERFACE_X
    return params


def _forcing_layered_batch_losses(
    model: InterfaceCViT,
    params: list[dict],
    *,
    residual_method: str,
    physics: dict[str, Any],
    interface_face: int,
    exclusion: tuple[float, float],
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    y_img: np.ndarray,
    t_img: np.ndarray,
    mu: float,
    sigma: float,
    q_ref: float,
    t_ref: float,
    t_final: float,
    t_ramp: float,
    dt: float,
    n_steps: int,
    intervals_per_sim: int,
    stratified: bool,
    n_bins: int,
    n_r: int,
    n_bc: int,
    n_ic: int,
    chunk_r: int,
    rngs: HybridForcingRNGs,
    energy_cfg: dict[str, Any] | None = None,
    reformulation_cfg: dict[str, Any] | None = None,
    max_start_step: int | None = None,
    active_t_final: float | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    device = x_grid.device
    batch_size = len(params)
    Nx, Ny = int(x_grid.numel()), int(y_grid.numel())
    x_np = x_grid.detach().cpu().numpy().astype(np.float64)
    fixed_ic_tilde = np.float32(
        (float(physics["T_right"]) - float(mu)) / float(sigma)
    )
    normalized_ic = np.full(
        (batch_size, Nx, Ny), fixed_ic_tilde, dtype=np.float32
    )
    u_spatial = torch.from_numpy(np.stack([
        _interface_spatial_channels_from_normalized(
            normalized_ic[index], float(physics["interface_x"]), x_np
        )
        for index in range(batch_size)
    ])).to(device)
    forcing_image = build_forcing_image(
        params, y_img, t_img, q_ref, device, t_ramp
    )
    pscal = torch.from_numpy(np.stack([
        normalize_forcing_interface_scalars(physics["interface_x"], p["R_c"])
        for p in params
    ])).to(device)
    latent = model.encode(u_spatial, forcing_image, pscal)

    ix = torch.randint(
        0, Nx, (batch_size, n_ic), device=device, generator=rngs.ic_collocation
    )
    iy = torch.randint(
        0, Ny, (batch_size, n_ic), device=device, generator=rngs.ic_collocation
    )
    ic_coords = torch.stack((x_grid[ix], y_grid[iy]), dim=-1)
    ic_time = torch.zeros((batch_size, n_ic, 1), device=device)
    ic_pred = _decode_in_chunks(model, latent, ic_coords, ic_time, chunk_r)
    batch_index = torch.arange(batch_size, device=device)[:, None]
    ic_target = u_spatial[batch_index, 0, ix, iy].unsqueeze(-1)
    loss_ic = (ic_pred - ic_target).square().flatten(1).mean(dim=1).mean()

    coll = _sample_lattice_intervals(
        rngs.interval_selection, batch_size, intervals_per_sim,
        n_steps, n_bins, stratified, max_start_step=max_start_step,
    )
    starts = coll["start_idx"].to(device=device, dtype=torch.long)
    sim_local = coll["sim_local"].to(device=device, dtype=torch.long)
    interval_count = int(starts.numel())
    # MPS has no float64 tensor support; keep CN query times in the active
    # grid dtype (float32 on MPS, float64 on CPU if callers choose it).
    tn = starts.to(dtype=x_grid.dtype) * float(dt)
    tnp1 = tn + float(dt)
    latent_intervals = latent[sim_local]

    interface_values = np.full(interval_count, float(physics["interface_x"]))
    resistance = np.asarray([
        float(params[int(index)]["R_c"])
        for index in sim_local.detach().cpu().numpy()
    ])
    geom = build_cn_geom_per_interface(
        x_np, y_grid.detach().cpu().numpy(),
        physics["k_left"], physics["k_right"],
        interface_values, resistance, dt,
        sigma_global=sigma, rho=physics["rho_left"], cp=physics["cp_left"],
        device=device, dtype=x_grid.dtype,
    )
    if not bool((geom.face_idx == int(interface_face)).all().item()):
        raise AssertionError("forcing interface face changed inside a training batch")

    reformulation_enabled = bool((reformulation_cfg or {}).get("enabled", False))
    if residual_method == "finite_volume" or reformulation_enabled:
        query_mesh = _full_grid_query_mesh(x_grid, y_grid)
        query_count = Nx * Ny
    else:
        local_x = x_grid[interface_face - 1:interface_face + 3]
        gx, gy = torch.meshgrid(local_x, y_grid, indexing="ij")
        query_mesh = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1).unsqueeze(0)
        query_count = 4 * Ny
    coords = query_mesh.expand(interval_count, -1, -1)
    tn_query = tn.to(dtype=x_grid.dtype).view(interval_count, 1, 1).expand(
        interval_count, query_count, 1
    )
    tnp1_query = tnp1.to(dtype=x_grid.dtype).view(interval_count, 1, 1).expand(
        interval_count, query_count, 1
    )
    T_n = _decode_in_chunks(
        model, latent_intervals, coords, tn_query, chunk_r
    )[..., 0].view(interval_count, -1, Ny)
    T_np1 = _decode_in_chunks(
        model, latent_intervals, coords, tnp1_query, chunk_r
    )[..., 0].view(interval_count, -1, Ny)

    energy_enabled = bool((energy_cfg or {}).get("enabled", False))
    qn_np = qnp1_np = qint_np = None
    T_n_energy = T_np1_energy = None
    if energy_enabled and residual_method == "hybrid" and not reformulation_enabled:
        full_mesh = _full_grid_query_mesh(x_grid, y_grid)
        full_count = Nx * Ny
        full_coords = full_mesh.expand(interval_count, -1, -1)
        tn_full = tn.to(dtype=x_grid.dtype).view(interval_count, 1, 1).expand(
            interval_count, full_count, 1
        )
        tnp1_full = tnp1.to(dtype=x_grid.dtype).view(interval_count, 1, 1).expand(
            interval_count, full_count, 1
        )
        T_n_energy = _decode_in_chunks(
            model, latent_intervals, full_coords, tn_full, chunk_r
        )[..., 0].view(interval_count, Nx, Ny)
        T_np1_energy = _decode_in_chunks(
            model, latent_intervals, full_coords, tnp1_full, chunk_r
        )[..., 0].view(interval_count, Nx, Ny)

    if residual_method == "finite_volume":
        qn_np, qnp1_np, qint_np = _forcing_interval_fluxes(
            params, sim_local, y_grid, tn, tnp1, t_ramp
        )
        bc = FullBCData(
            T_right_tilde=torch.as_tensor(
                (float(physics["T_right"]) - mu) / sigma,
                device=device, dtype=T_n.dtype,
            ),
            qL_n=torch.from_numpy(qn_np).to(device=device, dtype=T_n.dtype),
            qL_np1=torch.from_numpy(qnp1_np).to(device=device, dtype=T_n.dtype),
            qL_int=torch.from_numpy(qint_np).to(device=device, dtype=T_n.dtype),
        )
        fv = region_balanced_fv_rate_loss(
            T_n, T_np1, geom, bc, t_ref=t_ref, dirichlet_both_ends=True
        )
        losses = {
            "interior": _mean_problem_intervals(
                fv["interior_per_sample"], batch_size, intervals_per_sim
            ),
            "interface": _mean_problem_intervals(
                fv["interface_per_sample"], batch_size, intervals_per_sim
            ),
            "left_neumann": _mean_problem_intervals(
                fv["left_neumann_per_sample"], batch_size, intervals_per_sim
            ),
            "top_adiabatic": _mean_problem_intervals(
                fv["top_adiabatic_per_sample"], batch_size, intervals_per_sim
            ),
            "bottom_adiabatic": _mean_problem_intervals(
                fv["bottom_adiabatic_per_sample"], batch_size, intervals_per_sim
            ),
            "topbot_adiabatic": _mean_problem_intervals(
                fv["topbot_adiabatic_per_sample"], batch_size, intervals_per_sim
            ),
            "ic": loss_ic,
        }
        right_dir = fv["phys_right_dirichlet_mse"]
        if energy_enabled:
            T_n_energy, T_np1_energy = T_n, T_np1
    else:
        interface_rate = interface_residual_rate(T_n, T_np1, geom)
        interface_per_interval = (
            float(t_ref) * interface_rate
        ).square().flatten(1).mean(dim=1)
        ad = _forcing_hybrid_ad_losses(
            model, latent, params, physics=physics, exclusion=exclusion,
            sigma=sigma, q_ref=q_ref, t_ref=t_ref, t_final=t_final,
            t_ramp=t_ramp, n_r=n_r, n_bc=n_bc,
            bulk_generator=rngs.bulk_collocation,
            boundary_generator=rngs.boundary_collocation,
            active_t_final=active_t_final,
        )
        losses = {
            **ad,
            "interface": _mean_problem_intervals(
                interface_per_interval, batch_size, intervals_per_sim
            ),
            "ic": loss_ic,
        }
        right_dir = torch.zeros((), device=device, dtype=T_n.dtype)
    if reformulation_enabled:
        _, _, qint_np = _forcing_interval_fluxes(
            params, sim_local, y_grid, tn, tnp1, t_ramp
        )
        qint_t = torch.from_numpy(qint_np).to(device=device, dtype=T_n.dtype)
        blocks = block_energy_residual(
            T_n, T_np1, geom, qint_t,
            x_blocks_per_layer=int(reformulation_cfg.get("x_blocks_per_layer", 4)),
            y_blocks=int(reformulation_cfg.get("y_blocks", 8)),
            q_ref=q_ref,
            scale_floor=float(reformulation_cfg.get("scale_floor", 1.0)),
        )
        traces = interface_trace_constraint_residuals(
            T_n, T_np1, geom, x_np, resistance,
            k_left=float(physics["k_left"]),
            k_right=float(physics["k_right"]), sigma_global=sigma,
            q_ref=q_ref,
            jump_floor_K=float(reformulation_cfg.get("jump_floor_K", 1.0)),
        )
        losses.update({
            "_constraint_local_energy_interface": blocks["interface"],
            "_constraint_local_energy_far": blocks["far"],
            "_constraint_interface_flux": traces["flux"],
            "_constraint_interface_contact": traces["contact"],
            "local_energy_interface": blocks["interface"].square().mean(),
            "local_energy_far": blocks["far"].square().mean(),
            "interface_flux_constraint": traces["flux"].square().mean(),
            "interface_contact_constraint": traces["contact"].square().mean(),
            "local_energy_max_block": blocks["all"].abs().max(),
        })
    if energy_enabled:
        if qint_np is None:
            _, _, qint_np = _forcing_interval_fluxes(
                params, sim_local, y_grid, tn, tnp1, t_ramp
            )
        qint_t = torch.from_numpy(qint_np).to(device=device, dtype=T_n.dtype)
        losses.update(_forcing_energy_closure_losses(
            T_n_energy, T_np1_energy, geom, qint_t,
            interface_face=interface_face, physics=physics, sigma=sigma,
            q_ref=q_ref,
            scale_floor=float((energy_cfg or {}).get("scale_floor", 1.0)),
        ))
    return losses, right_dir


# ---- fixed forcing-screen instrumentation ----------------------------------

def _forcing_diagnostic_probe_params() -> list[dict[str, Any]]:
    """Return the small, deterministic forcing set used by the screen dashboard.

    The probes are deliberately outside the online sampler RNG stream.  They
    are fixed physical questions (a short pulse, a moderate smooth load, and
    two contact resistances), so a change in a diagnostic row means the model
    changed rather than the test problem changing.
    """
    base = {
        "interface_x": float(FORCING_INTERFACE_X),
        "spatial_family": "uniform",
        "spatial_params": {},
    }
    pulse = {
        **base,
        "temporal_family": "pulse_train",
        "temporal_params": {
            "Np": 1, "A_list": [250.0], "t_list": [0.04],
            "dt_list": [0.08],
        },
    }
    smooth = {
        **base,
        "temporal_family": "sin",
        "temporal_params": {
            "A": 180.0, "f": 4.0, "t_on": 0.0, "t_off": 0.2,
            "phase": 0.0, "tukey_alpha": 0.5, "rectified": True,
        },
    }
    return [
        {**pulse, "R_c": float(FORCING_RC_RANGE[0])},
        {**pulse, "R_c": float(FORCING_RC_RANGE[1])},
        {**smooth, "R_c": 0.5 * sum(FORCING_RC_RANGE)},
    ]


def _forcing_fixed_probe_case(case: str) -> dict[str, Any]:
    cases = {
        "pulse_low": 0,
        "pulse_high": 1,
        "smooth_mid": 2,
    }
    if case in cases:
        return copy.deepcopy(_forcing_diagnostic_probe_params()[cases[case]])
    compositional = {
        "pulse_mid": (0, 0.5 * sum(FORCING_RC_RANGE)),
        "smooth_low": (2, float(FORCING_RC_RANGE[0])),
        "smooth_high": (2, float(FORCING_RC_RANGE[1])),
    }
    if case not in compositional:
        raise ValueError(
            "training.pino.forcing.fixed_probe_case must be one of "
            f"{sorted([*cases, *compositional])}, got {case!r}"
        )
    source, resistance = compositional[case]
    params = copy.deepcopy(_forcing_diagnostic_probe_params()[source])
    params["R_c"] = float(resistance)
    return params


def _solve_forcing_diagnostic_truth(
    problem: Any,
    params: dict[str, Any],
    physics: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    dt: float,
    t_ramp: float,
) -> np.ndarray:
    """Solve one fixed probe with the same FV implementation as the dataset."""
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    t_grid = np.asarray(t_grid, dtype=np.float64)
    base_kwargs = {
        "a": float(physics["a"]), "b": float(physics["b"]),
        "c": float(physics["c"]), "d": float(physics["d"]),
        "Nx": int(x_grid.size), "Ny": int(y_grid.size),
        "lam_target": 0.8, "layers": physics["layers"],
        "t_final": float(round(float(t_grid[-1]), 6)),
        "flux_f": 0.0, "flux_A": 0.0,
        "t_on": 0.0, "t_off": 0.2, "phase": 0.0,
        "tukey_alpha": 0.5, "dt": float(dt), "y_grid": y_grid,
        "ramp_seconds": float(t_ramp),
    }
    solver = problem.configure_solver(params, base_kwargs)
    t_full, _, _, trajectory = solver.solve(
        np.full((x_grid.size, y_grid.size), float(physics["T_right"])),
        store_trajectory=True,
    )
    # The saved trajectory may be decimated relative to the solver.  The
    # diagnostic uses the exact saved times and selects the corresponding CN
    # states, with nearest-index fallback for custom local screens.
    indices = np.asarray(
        [int(np.argmin(np.abs(np.asarray(t_full) - value))) for value in t_grid],
        dtype=np.int64,
    )
    return np.asarray(trajectory[indices], dtype=np.float64)


def _calibrate_forcing_constraint_tolerances(
    problem: Any, probe_params: list[dict[str, Any]], physics: dict[str, Any],
    x_grid: np.ndarray, y_grid: np.ndarray, *, dt: float, t_final: float,
    t_ramp: float, sigma: float, q_ref: float,
    reformulation_cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure attainable residual floors on exact FV probe trajectories."""
    samples = {name: [] for name in _FORCING_CONSTRAINT_NAMES}
    for params in probe_params:
        base_kwargs = {
            "a": float(physics["a"]), "b": float(physics["b"]),
            "c": float(physics["c"]), "d": float(physics["d"]),
            "Nx": int(len(x_grid)), "Ny": int(len(y_grid)),
            "lam_target": 0.8, "layers": physics["layers"],
            "t_final": float(round(float(t_final), 6)), "flux_f": 0.0, "flux_A": 0.0,
            "t_on": 0.0, "t_off": 0.2, "phase": 0.0,
            "tukey_alpha": 0.5, "dt": float(dt), "y_grid": y_grid,
            "ramp_seconds": float(t_ramp),
        }
        solver = problem.configure_solver(params, base_kwargs)
        times, _, _, trajectory = solver.solve(
            np.full((len(x_grid), len(y_grid)), float(physics["T_right"])),
            store_trajectory=True,
        )
        field = torch.as_tensor(
            (np.asarray(trajectory) - float(physics["T_right"])) / float(sigma),
            dtype=torch.float64,
        )
        count = field.shape[0] - 1
        geom = build_cn_geom_per_interface(
            x_grid, y_grid, physics["k_left"], physics["k_right"],
            np.full(count, physics["interface_x"]),
            np.full(count, params["R_c"]), dt, sigma_global=sigma,
            rho=physics["rho_left"], cp=physics["cp_left"],
            device=torch.device("cpu"), dtype=torch.float64,
        )
        sim_local = torch.zeros(count, dtype=torch.long)
        tn = torch.as_tensor(times[:-1], dtype=torch.float64)
        tnp1 = torch.as_tensor(times[1:], dtype=torch.float64)
        _, _, qint_np = _forcing_interval_fluxes(
            [params], sim_local, torch.as_tensor(y_grid), tn, tnp1, t_ramp,
        )
        blocks = block_energy_residual(
            field[:-1], field[1:], geom, torch.as_tensor(qint_np),
            x_blocks_per_layer=int(reformulation_cfg.get("x_blocks_per_layer", 4)),
            y_blocks=int(reformulation_cfg.get("y_blocks", 8)), q_ref=q_ref,
            scale_floor=float(reformulation_cfg.get("scale_floor", 1.0)),
        )
        traces = interface_trace_constraint_residuals(
            field[:-1], field[1:], geom, x_grid,
            np.full(count, params["R_c"]), k_left=physics["k_left"],
            k_right=physics["k_right"], sigma_global=sigma, q_ref=q_ref,
            jump_floor_K=float(reformulation_cfg.get("jump_floor_K", 1.0)),
        )
        for name, values in (
            # The AD wall constraint is continuous. FV boundary nodes obey a
            # half-cell balance, so a nodal finite-difference gradient is not
            # its attainable floor and would create a large false dead band.
            ("left_flux", torch.zeros(1, dtype=torch.float64)),
            ("local_energy_interface", blocks["interface"]),
            ("local_energy_far", blocks["far"]),
            ("interface_flux", traces["flux"]),
            ("interface_contact", traces["contact"]),
        ):
            samples[name].append(float(torch.sqrt(values.square().mean()).cpu()))
    floors = {name: max(samples[name]) for name in _FORCING_CONSTRAINT_NAMES}
    margin = float(reformulation_cfg.get("tolerance_margin", 10.0))
    minimum = float(reformulation_cfg.get("tolerance_min", 1.0e-8))
    overrides = dict(reformulation_cfg.get("tolerance_override", {}) or {})
    tolerances = {
        name: float(overrides.get(name, max(margin * floors[name], minimum)))
        for name in _FORCING_CONSTRAINT_NAMES
    }
    return floors, tolerances


def _forcing_probe_query_indices(
    x_grid: np.ndarray, y_grid: np.ndarray, interface_face: int,
    x_samples: int, y_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_idx = np.linspace(0, len(x_grid) - 1, max(int(x_samples), 2), dtype=int)
    x_idx = np.unique(np.r_[x_idx, interface_face - 1, interface_face,
                            interface_face + 1, interface_face + 2])
    y_idx = np.linspace(0, len(y_grid) - 1, max(int(y_samples), 2), dtype=int)
    return x_idx, np.unique(y_idx)


def _forcing_probe_energy_losses(
    model: InterfaceCViT,
    latent: torch.Tensor,
    probe_params: list[dict[str, Any]],
    *,
    physics: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    time_idx: np.ndarray,
    q_ref: float,
    t_ramp: float,
    interface_face: int,
    sigma: float,
    scale_floor: float,
    query_chunk: int,
    device: torch.device,
) -> dict[str, float]:
    if len(time_idx) < 2:
        return {
            "probe_loss_energy_left": float("nan"),
            "probe_loss_energy_right": float("nan"),
            "probe_loss_energy_global": float("nan"),
            "probe_loss_energy": float("nan"),
        }
    Nx, Ny = int(x_grid.size), int(y_grid.size)
    gx, gy = np.meshgrid(x_grid, y_grid, indexing="ij")
    full_coords = torch.from_numpy(
        np.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1).astype(np.float32)
    ).unsqueeze(0).to(device)
    # The four rendered cases are [pulse-low, pulse-high, zeroed pulse-low,
    # smooth-mid]; energy is evaluated on the three physical nonzero probes.
    latent_energy = latent[torch.as_tensor([0, 1, 3], device=device)]
    states = []
    for grid_index in time_idx:
        time_value = float(t_grid[int(grid_index)])
        tq = torch.full(
            (len(probe_params), full_coords.shape[1], 1),
            time_value, device=device,
        )
        states.append(
            _decode_in_chunks(
                model, latent_energy,
                full_coords.expand(len(probe_params), -1, -1), tq,
                int(query_chunk),
            )[..., 0].view(len(probe_params), Nx, Ny)
        )
    field = torch.stack(states, dim=1)
    slabs = len(time_idx) - 1
    T_n = field[:, :-1].reshape(len(probe_params) * slabs, Nx, Ny)
    T_np1 = field[:, 1:].reshape(len(probe_params) * slabs, Nx, Ny)

    t_values = np.asarray(t_grid[time_idx], dtype=np.float64)
    t_lo = np.tile(t_values[:-1], len(probe_params))
    t_hi = np.tile(t_values[1:], len(probe_params))
    dt_values = t_hi - t_lo
    interval_params = [
        probe_params[case_index]
        for case_index in range(len(probe_params))
        for _ in range(slabs)
    ]
    sim_local = torch.arange(len(interval_params), device=device)
    _, _, qint_np = _forcing_interval_fluxes(
        interval_params, sim_local,
        torch.as_tensor(y_grid, dtype=T_n.dtype, device=device),
        torch.as_tensor(t_lo, dtype=T_n.dtype, device=device),
        torch.as_tensor(t_hi, dtype=T_n.dtype, device=device),
        t_ramp,
    )
    qint = torch.from_numpy(qint_np).to(device=device, dtype=T_n.dtype)

    losses = {
        "probe_loss_energy_left": 0.0,
        "probe_loss_energy_right": 0.0,
        "probe_loss_energy_global": 0.0,
    }
    total_count = 0
    interval_x = np.full(len(interval_params), float(physics["interface_x"]))
    interval_rc = np.asarray(
        [float(record["R_c"]) for record in interval_params], dtype=np.float64,
    )
    for dt_value in np.unique(np.round(dt_values, 12)):
        mask_np = np.isclose(dt_values, float(dt_value), rtol=0.0, atol=1e-12)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        geom = build_cn_geom_per_interface(
            x_grid, y_grid, physics["k_left"], physics["k_right"],
            interval_x[mask_np], interval_rc[mask_np], float(dt_value),
            sigma_global=sigma, rho=physics["rho_left"], cp=physics["cp_left"],
            device=device, dtype=T_n.dtype,
        )
        if not bool((geom.face_idx == int(interface_face)).all().item()):
            raise AssertionError("fixed probe interface face changed")
        residuals = _forcing_energy_closure_residuals(
            T_n[mask], T_np1[mask], geom, qint[mask],
            interface_face=interface_face, physics=physics, sigma=sigma,
            q_ref=q_ref, scale_floor=scale_floor,
        )
        count = int(mask_np.sum())
        losses["probe_loss_energy_left"] += float(
            residuals["energy_left_residual"].square().sum().detach().cpu()
        )
        losses["probe_loss_energy_right"] += float(
            residuals["energy_right_residual"].square().sum().detach().cpu()
        )
        losses["probe_loss_energy_global"] += float(
            residuals["energy_global_residual"].square().sum().detach().cpu()
        )
        total_count += count
    for key in list(losses):
        losses[key] /= max(total_count, 1)
    losses["probe_loss_energy"] = (
        losses["probe_loss_energy_left"]
        + losses["probe_loss_energy_right"]
        + losses["probe_loss_energy_global"]
    )
    return losses


def _forcing_probe_model_outputs(
    model: InterfaceCViT,
    probe_params: list[dict[str, Any]],
    *,
    mu: float,
    sigma: float,
    physics: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_grid: np.ndarray,
    y_img: np.ndarray,
    t_img: np.ndarray,
    q_ref: float,
    t_ramp: float,
    interface_face: int,
    x_samples: int,
    y_samples: int,
    time_samples: int,
    query_chunk: int,
    device: torch.device,
    energy_scale_floor: float = 1.0,
    truth_cache: list[np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Evaluate fixed physical probes and their zero-forcing twins."""
    was_training = model.training
    model.eval()
    x_idx, y_idx = _forcing_probe_query_indices(
        x_grid, y_grid, interface_face, x_samples, y_samples,
    )
    time_idx = np.linspace(
        0, len(t_grid) - 1, max(int(time_samples), 2), dtype=int,
    )
    time_idx = np.unique(time_idx)
    gx, gy = np.meshgrid(x_grid[x_idx], y_grid[y_idx], indexing="ij")
    sparse_coords = torch.from_numpy(
        np.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1).astype(np.float32)
    ).unsqueeze(0).to(device)
    jump_x = np.asarray([x_grid[interface_face], x_grid[interface_face + 1]])
    jx, jy = np.meshgrid(jump_x, y_grid, indexing="ij")
    jump_coords = torch.from_numpy(
        np.stack((jx.reshape(-1), jy.reshape(-1)), axis=-1).astype(np.float32)
    ).unsqueeze(0).to(device)

    truth_cache_out: list[np.ndarray | None] = []
    for probe_index, p in enumerate(probe_params):
        if truth_cache is None:
            truth_cache_out.append(_solve_forcing_diagnostic_truth(
                problem=physics["problem"], params=p, physics=physics,
                x_grid=x_grid, y_grid=y_grid, t_grid=t_grid,
                dt=physics["fv_dt"], t_ramp=t_ramp,
            ))
        else:
            truth_cache_out.append(truth_cache[probe_index])
    # Exactly four cases: low-R pulse, high-R pulse, zeroed low-R pulse image,
    # and moderate sinusoid.  The zero case intentionally retains the low-R
    # parameter token so only the forcing image changes.
    all_params = [probe_params[0], probe_params[1], probe_params[0], probe_params[2]]
    zero_flags = [False, False, True, False]
    truths: list[np.ndarray | None] = [
        truth_cache_out[0], truth_cache_out[1], None, truth_cache_out[2],
    ]

    fixed_ic = np.full(
        (len(all_params), x_grid.size, y_grid.size),
        np.float32((float(physics["T_right"]) - mu) / sigma),
    )
    u_spatial = torch.from_numpy(np.stack([
        _interface_spatial_channels_from_normalized(
            fixed_ic[index], float(physics["interface_x"]), x_grid,
        )
        for index in range(len(all_params))
    ])).to(device)
    forcing_image = build_forcing_image(
        all_params, y_img, t_img, q_ref, device, t_ramp,
    )
    for index, is_zero in enumerate(zero_flags):
        if is_zero:
            forcing_image[index].zero_()
    pscal = torch.from_numpy(np.stack([
        normalize_forcing_interface_scalars(
            physics["interface_x"], p["R_c"],
        ) for p in all_params
    ])).to(device)
    with torch.no_grad():
        latent = model.encode(u_spatial, forcing_image, pscal)
        probe_energy_losses = _forcing_probe_energy_losses(
            model, latent, probe_params, physics=physics,
            x_grid=x_grid, y_grid=y_grid, t_grid=t_grid, time_idx=time_idx,
            q_ref=q_ref, t_ramp=t_ramp, interface_face=interface_face,
            sigma=sigma, scale_floor=float(energy_scale_floor),
            query_chunk=int(query_chunk), device=device,
        )
        sparse_prediction = np.empty(
            (len(all_params), len(time_idx), len(x_idx), len(y_idx)),
            dtype=np.float64,
        )
        jump_prediction = np.empty(
            (len(all_params), len(time_idx), 2, len(y_grid)), dtype=np.float64,
        )
        for ti, grid_index in enumerate(time_idx):
            time_value = float(t_grid[grid_index])
            tq_sparse = torch.full(
                (len(all_params), sparse_coords.shape[1], 1),
                time_value, device=device,
            )
            tq_jump = torch.full(
                (len(all_params), jump_coords.shape[1], 1),
                time_value, device=device,
            )
            sparse_prediction[:, ti] = (
                _decode_in_chunks(
                    model, latent,
                    sparse_coords.expand(len(all_params), -1, -1), tq_sparse,
                    int(query_chunk),
                )[..., 0].view(len(all_params), len(x_idx), len(y_idx))
                .detach().cpu().numpy() * sigma + mu
            )
            jump_prediction[:, ti] = (
                _decode_in_chunks(
                    model, latent,
                    jump_coords.expand(len(all_params), -1, -1), tq_jump,
                    int(query_chunk),
                )[..., 0].view(len(all_params), 2, len(y_grid))
                .detach().cpu().numpy() * sigma + mu
            )
    truth_sparse: list[np.ndarray | None] = []
    truth_jump: list[np.ndarray | None] = []
    for truth in truths:
        if truth is None:
            truth_sparse.append(None)
            truth_jump.append(None)
        else:
            truth_sparse.append(truth[time_idx][:, x_idx][:, :, y_idx])
            truth_jump.append(truth[time_idx][:, [interface_face, interface_face + 1]])
    if was_training:
        model.train()
    return {
        "sparse_prediction": sparse_prediction,
        "jump_prediction": jump_prediction,
        "truth_sparse": truth_sparse,
        "truth_jump": truth_jump,
        "truth_full": truth_cache_out,
        "time_idx": time_idx,
        "zero_flags": np.asarray(zero_flags, dtype=bool),
        "probe_energy_losses": probe_energy_losses,
    }


def _forcing_probe_metrics(
    outputs: dict[str, np.ndarray],
    init_outputs: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    pred = outputs["sparse_prediction"]
    jumps = outputs["jump_prediction"]
    zero = outputs["zero_flags"]
    truths = outputs["truth_sparse"]
    truth_jumps = outputs["truth_jump"]
    nonzero = ~zero
    field_errors = []
    departure = []
    true_jumps = []
    pred_jumps = []
    jump_errors = []
    for index in np.flatnonzero(nonzero):
        truth = np.asarray(truths[index], dtype=np.float64)
        field_errors.append(pred[index] - truth)
        departure.append(pred[index] - 300.0)
        truth_jump = np.asarray(truth_jumps[index], dtype=np.float64)
        true_jumps.append(truth_jump[:, 0] - truth_jump[:, 1])
        pred_jump = jumps[index, :, 0] - jumps[index, :, 1]
        pred_jumps.append(pred_jump)
        jump_errors.append(pred_jump - (truth_jump[:, 0] - truth_jump[:, 1]))
    field_errors_arr = np.concatenate([value.reshape(-1) for value in field_errors])
    departure_arr = np.concatenate([value.reshape(-1) for value in departure])
    true_jump_arr = np.concatenate([value.reshape(-1) for value in true_jumps])
    pred_jump_arr = np.concatenate([value.reshape(-1) for value in pred_jumps])
    jump_error_arr = np.concatenate([value.reshape(-1) for value in jump_errors])

    # Entries are [pulse-low, pulse-high, pulse-low-zero, smooth-mid].
    forcing_sensitivity = pred[0] - pred[2]
    true_forcing = np.asarray(truths[0], dtype=np.float64) - 300.0
    rc_sensitivity = pred[1] - pred[0]
    truth_rc_sensitivity = (
        np.asarray(truths[1], dtype=np.float64)
        - np.asarray(truths[0], dtype=np.float64)
    )
    out = {
        "drift_from_init_K": float("nan"),
        "departure_from_300_K": float(np.sqrt(np.mean(departure_arr ** 2))),
        "field_rmse_K": float(np.sqrt(np.mean(field_errors_arr ** 2))),
        "forcing_sensitivity_K": float(np.sqrt(np.mean(forcing_sensitivity ** 2))),
        "forcing_response_ratio": float(
            np.sqrt(np.mean(forcing_sensitivity ** 2))
            / max(np.sqrt(np.mean(true_forcing ** 2)), 1e-12)
        ),
        "rc_sensitivity_K": float(np.sqrt(np.mean(rc_sensitivity ** 2))),
        "rc_response_ratio": float(
            np.sqrt(np.mean(rc_sensitivity ** 2))
            / max(np.sqrt(np.mean(truth_rc_sensitivity ** 2)), 1e-12)
        ),
        "pred_jump_rms_K": float(np.sqrt(np.mean(pred_jump_arr ** 2))),
        "true_jump_rms_K": float(np.sqrt(np.mean(true_jump_arr ** 2))),
        "node_jump_rmse_K": float(np.sqrt(np.mean(jump_error_arr ** 2))),
    }
    out.update({
        "probe_loss_energy_left": float("nan"),
        "probe_loss_energy_right": float("nan"),
        "probe_loss_energy_global": float("nan"),
        "probe_loss_energy": float("nan"),
        **{
            key: float(value)
            for key, value in dict(outputs.get("probe_energy_losses", {})).items()
        },
    })
    if init_outputs is not None:
        out["drift_from_init_K"] = float(np.sqrt(np.mean(
            (pred[nonzero] - init_outputs["sparse_prediction"][nonzero]) ** 2
        )))
    return out


def _gradient_norm_and_cosine(
    model: torch.nn.Module, loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[float, torch.Tensor]:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
    )
    pieces = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None
        else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    vector = torch.cat(pieces) if pieces else torch.zeros(1, device=loss.device)
    return float(torch.linalg.vector_norm(vector).detach().cpu()), vector


def _forcing_pair_gradient_metrics(
    l_discriminating: torch.Tensor,
    l_stiff: torch.Tensor,
    l_homogeneous: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    suffix: str = "",
) -> dict[str, float]:
    norm_d, grad_d = _gradient_norm_and_cosine(
        None, l_discriminating, parameters
    )
    norm_p, grad_p = _gradient_norm_and_cosine(None, l_stiff, parameters)
    norm_h, _ = _gradient_norm_and_cosine(None, l_homogeneous, parameters)
    grad_d_norm = torch.linalg.vector_norm(grad_d)
    grad_p_norm = torch.linalg.vector_norm(grad_p)
    cosine = float(
        (torch.dot(grad_d, grad_p) / (grad_d_norm * grad_p_norm + 1e-12))
        .detach().cpu()
    )
    combined = grad_d + grad_p
    projection_d = float(
        (torch.dot(combined, grad_d) / (grad_d_norm + 1e-12)).detach().cpu()
    )
    projection_p = float(
        (torch.dot(combined, grad_p) / (grad_p_norm + 1e-12)).detach().cpu()
    )
    total = norm_d + norm_p + norm_h
    return {
        f"grad_norm_discriminating{suffix}": norm_d,
        f"grad_norm_stiff_physics{suffix}": norm_p,
        f"grad_norm_homogeneous{suffix}": norm_h,
        f"grad_ratio_discriminating_to_physics{suffix}": (
            norm_d / max(norm_p, 1e-12)
        ),
        f"grad_cos_discriminating_physics{suffix}": cosine,
        f"gradient_share_discriminating{suffix}": (
            norm_d / max(norm_d + norm_p, 1e-12)
        ),
        f"effective_gradient_share_discriminating{suffix}": (
            norm_d / max(total, 1e-12)
        ),
        f"grad_projection_on_discriminating{suffix}": projection_d,
        f"grad_projection_on_physics{suffix}": projection_p,
    }


def _forcing_pair_metrics_from_vectors(
    grad_d: torch.Tensor,
    grad_p: torch.Tensor,
    norm_h: float,
    *,
    suffix: str = "",
) -> dict[str, float]:
    norm_d = float(torch.linalg.vector_norm(grad_d).detach().cpu())
    norm_p = float(torch.linalg.vector_norm(grad_p).detach().cpu())
    grad_d_norm = torch.linalg.vector_norm(grad_d)
    grad_p_norm = torch.linalg.vector_norm(grad_p)
    cosine = float(
        (torch.dot(grad_d, grad_p) / (grad_d_norm * grad_p_norm + 1e-12))
        .detach().cpu()
    )
    combined = grad_d + grad_p
    projection_d = float(
        (torch.dot(combined, grad_d) / (grad_d_norm + 1e-12)).detach().cpu()
    )
    projection_p = float(
        (torch.dot(combined, grad_p) / (grad_p_norm + 1e-12)).detach().cpu()
    )
    total = norm_d + norm_p + float(norm_h)
    return {
        f"grad_norm_discriminating{suffix}": norm_d,
        f"grad_norm_stiff_physics{suffix}": norm_p,
        f"grad_norm_homogeneous{suffix}": float(norm_h),
        f"grad_ratio_discriminating_to_physics{suffix}": (
            norm_d / max(norm_p, 1e-12)
        ),
        f"grad_cos_discriminating_physics{suffix}": cosine,
        f"gradient_share_discriminating{suffix}": (
            norm_d / max(norm_d + norm_p, 1e-12)
        ),
        f"effective_gradient_share_discriminating{suffix}": (
            norm_d / max(total, 1e-12)
        ),
        f"grad_projection_on_discriminating{suffix}": projection_d,
        f"grad_projection_on_physics{suffix}": projection_p,
    }


def _forcing_branch_parameters(
    model: torch.nn.Module, prefix: str,
) -> list[torch.nn.Parameter]:
    return [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    ]


def _forcing_branch_gradient_metrics(
    l_discriminating: torch.Tensor,
    l_stiff: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    branch: str,
) -> dict[str, float]:
    if not parameters:
        return {
            f"grad_norm_discriminating_{branch}": 0.0,
            f"grad_norm_stiff_physics_{branch}": 0.0,
            f"grad_cos_discriminating_physics_{branch}": 0.0,
        }
    norm_d, grad_d = _gradient_norm_and_cosine(
        None, l_discriminating, parameters
    )
    norm_p, grad_p = _gradient_norm_and_cosine(None, l_stiff, parameters)
    cosine = float(
        (torch.dot(grad_d, grad_p) / (
            torch.linalg.vector_norm(grad_d) * torch.linalg.vector_norm(grad_p)
            + 1e-12
        )).detach().cpu()
    )
    return {
        f"grad_norm_discriminating_{branch}": norm_d,
        f"grad_norm_stiff_physics_{branch}": norm_p,
        f"grad_cos_discriminating_physics_{branch}": cosine,
    }


def _forcing_group_losses(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
    *,
    energy_weight: float,
) -> dict[str, torch.Tensor]:
    zero = losses["left_neumann"].new_zeros(())
    energy = losses.get("energy", zero)
    return {
        "discriminating": (
            float(weights["left_neumann"]) * losses["left_neumann"]
            + float(energy_weight) * energy
        ),
        "stiff_physics": (
            float(weights["interior"]) * losses["interior"]
            + float(weights["interface"]) * losses["interface"]
        ),
        "homogeneous": (
            float(weights["ic"]) * losses["ic"]
            + float(weights["topbot_adiabatic"]) * losses["topbot_adiabatic"]
        ),
    }


def _forcing_diagnostic_gradient_metrics(
    model: InterfaceCViT,
    *,
    residual_method: str,
    physics: dict[str, Any],
    interface_face: int,
    exclusion: tuple[float, float],
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    y_img: np.ndarray,
    t_img: np.ndarray,
    mu: float,
    sigma: float,
    q_ref: float,
    t_ref: float,
    t_final: float,
    t_ramp: float,
    dt: float,
    n_steps: int,
    intervals_per_sim: int,
    stratified: bool,
    n_bins: int,
    n_r: int,
    n_bc: int,
    n_ic: int,
    chunk_r: int,
    seed: int,
    weights: dict[str, float],
    energy_cfg: dict[str, Any],
    gradnorm_multipliers: dict[str, float] | None = None,
    reformulation_cfg: dict[str, Any] | None = None,
    constraint_controller: ForcingConstraintController | None = None,
) -> dict[str, float]:
    """Audit raw group gradients on a deterministic, non-training batch."""
    params = _forcing_diagnostic_probe_params()[:2]
    rngs = HybridForcingRNGs.create(int(seed), x_grid.device)
    was_training = model.training
    model.eval()
    try:
        losses, _ = _forcing_layered_batch_losses(
            model, params, residual_method=residual_method, physics=physics,
            interface_face=interface_face, exclusion=exclusion,
            x_grid=x_grid, y_grid=y_grid, y_img=y_img, t_img=t_img,
            mu=mu, sigma=sigma, q_ref=q_ref, t_ref=t_ref,
            t_final=t_final, t_ramp=t_ramp, dt=dt, n_steps=n_steps,
            intervals_per_sim=int(intervals_per_sim), stratified=stratified,
            n_bins=n_bins, n_r=max(int(n_r), 1), n_bc=max(int(n_bc), 1),
            n_ic=max(int(n_ic), 1), chunk_r=chunk_r, rngs=rngs,
            energy_cfg=energy_cfg,
            reformulation_cfg=reformulation_cfg,
        )
        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        energy_weight = (
            float(energy_cfg.get("lambda", 1.0))
            if bool(energy_cfg.get("enabled", False)) else 0.0
        )
        if constraint_controller is not None:
            constraint_residuals = {
                name: losses[f"_constraint_{name}"]
                for name in _FORCING_CONSTRAINT_NAMES
            }
            constraint_rms, violation = constraint_controller.values(constraint_residuals)
            groups = {
                "discriminating": constraint_controller.primal(violation),
                "stiff_physics": (
                    float(weights["interior"]) * losses["interior"]
                    + float(weights["interface"]) * losses["interface"]
                ),
                "homogeneous": (
                    float(weights["ic"]) * losses["ic"]
                    + float(weights["topbot_adiabatic"])
                    * losses["topbot_adiabatic"]
                ),
            }
            constraint_metrics = {}
            for name in _FORCING_CONSTRAINT_NAMES:
                constraint_metrics[f"constraint_rms_{name}"] = float(
                    constraint_rms[name].detach().cpu()
                )
                constraint_metrics[f"constraint_violation_{name}"] = float(
                    violation[name].detach().cpu()
                )
                constraint_metrics[f"constraint_multiplier_{name}"] = float(
                    constraint_controller.multipliers[name]
                )
            constraint_metrics["constraint_max_block"] = float(
                losses["local_energy_max_block"].detach().cpu()
            )
        else:
            groups = _forcing_group_losses(
                losses, weights, energy_weight=energy_weight,
            )
        multipliers = {
            "discriminating": 1.0,
            "stiff_physics": 1.0,
            **dict(gradnorm_multipliers or {}),
        }
        norm_d, grad_d = _gradient_norm_and_cosine(
            None, groups["discriminating"], parameters
        )
        norm_p, grad_p = _gradient_norm_and_cosine(
            None, groups["stiff_physics"], parameters
        )
        norm_h, _ = _gradient_norm_and_cosine(
            None, groups["homogeneous"], parameters
        )
        out = _forcing_pair_metrics_from_vectors(
            grad_d, grad_p, norm_h,
        )
        out.update(_forcing_pair_metrics_from_vectors(
            float(multipliers["discriminating"]) * grad_d,
            float(multipliers["stiff_physics"]) * grad_p,
            norm_h, suffix="_weighted",
        ))
        for branch, prefix in (
            ("decoder", "decoder."),
            ("forcing_encoder", "forcing_encoder."),
            ("param_encoder", "param_encoder."),
        ):
            out.update(_forcing_branch_gradient_metrics(
                groups["discriminating"], groups["stiff_physics"],
                _forcing_branch_parameters(model, prefix), branch,
            ))
        out["gradnorm_multiplier_discriminating"] = float(
            multipliers["discriminating"]
        )
        out["gradnorm_multiplier_stiff_physics"] = float(
            multipliers["stiff_physics"]
        )
        if constraint_controller is not None:
            out.update(constraint_metrics)
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
    return out


def run_one_seed_forcing_interface_pino(
    config: dict, seed: int, run_dir: Path,
) -> dict[str, Any]:
    """Physics-only forcing InterfaceCViT with online problems and saved validation."""
    set_seed(seed)
    run_dir = Path(run_dir)
    stale_names = (
        "train_metrics.csv", "diagnostics.csv", "cvit_init.pt",
        "cvit_best_global.pt", "cvit_last.pt", "final_metrics.json",
    )
    stale = [run_dir / name for name in stale_names if (run_dir / name).exists()]
    if stale:
        raise FileExistsError(
            "forcing InterfaceCViT residual screens require a fresh experiment name; found "
            + ", ".join(str(path) for path in stale)
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = float(data["mu_global"]), float(data["sigma_global"])
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("forcing InterfaceCViT requires a positive global temperature scale")
    x_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_np = np.asarray(data["y_grid"], dtype=np.float64)
    t_np = np.asarray(data["t_grid"], dtype=np.float64)
    Nx, Ny = x_np.size, y_np.size
    t_final = float(t_np[-1])
    t_ref = 0.3
    q_ref = float(A_AMP_REF)

    sim_params_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(sim_params_path, allow_pickle=True)
    if len(sim_params) != data["trajectories"].shape[0]:
        raise ValueError("sim_params length does not match trajectories")
    problem = problem_from_config(config)
    if getattr(problem, "name", None) != "forcing":
        raise ValueError("forcing InterfaceCViT runner requires benchmark=forcing")
    val_ids = np.asarray(data["val_ids"])
    problem.validate_schema(sim_params, val_ids)

    pino = config["training"]["pino"]
    residual_method = str(pino.get("residual_method", "finite_volume"))
    if residual_method not in {"finite_volume", "hybrid"}:
        raise ValueError(
            "forcing InterfaceCViT residual_method must be 'finite_volume' or "
            f"'hybrid', got {residual_method!r}"
        )
    if str(pino.get("collocation_source", "online")) != "online":
        raise ValueError(
            "the forcing residual comparison requires collocation_source=online"
        )
    fv_weighting = dict(
        (pino.get("finite_volume", {}) or {}).get(
            "interface_residual_weighting", {}
        ) or {}
    )
    if residual_method == "finite_volume" and not bool(fv_weighting.get("enabled", False)):
        raise ValueError(
            "finite_volume forcing mode requires "
            "training.pino.finite_volume.interface_residual_weighting.enabled=true"
        )
    if bool((pino.get("causal", {}) or {}).get("enabled", False)):
        raise ValueError("causal weighting must be disabled for the matched residual screen")
    if bool((pino.get("region_standardize", {}) or {}).get("enabled", False)):
        raise ValueError("region standardization must be disabled for the matched residual screen")
    if float(pino.get("lambda_data", 0.0)) != 0.0:
        raise ValueError("forcing residual comparison is physics-only; lambda_data must be zero")

    lam_r = float(pino["lambda_r"])
    lam_interface = float(pino.get("lambda_interface", 1.0))
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    lam_bc_left_cfg = pino.get("lambda_bc_left", None)
    lam_bc_left = lam_bc if lam_bc_left_cfg is None else float(lam_bc_left_cfg)
    weights = {
        "interior": lam_r,
        "interface": lam_interface,
        "left_neumann": lam_bc_left,
        "topbot_adiabatic": lam_bc,
        "ic": lam_ic,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("forcing residual weights must be finite and non-negative")
    if lam_interface <= 0.0:
        raise ValueError("lambda_interface must be positive")
    energy_cfg = dict(pino.get("energy", {}) or {})
    energy_enabled = bool(energy_cfg.get("enabled", False))
    energy_weight = float(energy_cfg.get("lambda", 1.0)) if energy_enabled else 0.0
    if energy_weight < 0.0 or not math.isfinite(energy_weight):
        raise ValueError("training.pino.energy.lambda must be finite and non-negative")
    reformulation_cfg = dict(pino.get("reformulation", {}) or {})
    reformulation_enabled = bool(reformulation_cfg.get("enabled", False))
    if reformulation_enabled:
        if residual_method != "hybrid":
            raise ValueError("forcing reformulation requires residual_method=hybrid")
        if energy_enabled:
            raise ValueError("legacy energy losses and forcing reformulation are mutually exclusive")
        if bool((config["training"].get("gradnorm", {}) or {}).get("enabled", False)):
            raise ValueError("GradNorm and forcing augmented-Lagrangian are mutually exclusive")

    fcfg = dict(pino.get("forcing", {}) or {})
    fixed_probe_case = fcfg.get("fixed_probe_case", None)
    fixed_probe_cases = fcfg.get("fixed_probe_cases", None)
    if fixed_probe_case is not None and fixed_probe_cases is not None:
        raise ValueError("set fixed_probe_case or fixed_probe_cases, not both")
    if fixed_probe_cases is not None:
        fixed_probe_cases = [str(value) for value in fixed_probe_cases]
        if not fixed_probe_cases:
            raise ValueError("fixed_probe_cases cannot be empty")
        for value in fixed_probe_cases:
            _forcing_fixed_probe_case(value)
    if (
        fixed_probe_case is None
        and (
            fcfg.get("temporal_family") is not None
            or fcfg.get("spatial_family") is not None
        )
    ):
        raise ValueError(
            "layered forcing residual screens require unpinned online temporal "
            "and spatial families"
        )
    if fixed_probe_case is not None and (
        fcfg.get("temporal_family") is not None
        or fcfg.get("spatial_family") is not None
    ):
        raise ValueError(
            "training.pino.forcing.fixed_probe_case cannot be combined with "
            "temporal_family/spatial_family pins"
        )
    fixed_train_param = (
        _forcing_fixed_probe_case(str(fixed_probe_case))
        if fixed_probe_case is not None else None
    )
    configured_ref = float(fcfg.get("a_ref", q_ref))
    if not math.isclose(configured_ref, q_ref, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"forcing image a_ref={configured_ref} must equal q_ref={q_ref} W/m^2"
        )
    ny_img = int(fcfg.get("ny_img") or 96)
    nt_img = int(fcfg.get("nt_img") or 256)
    y_img = np.linspace(float(y_np[0]), float(y_np[-1]), ny_img)
    t_img = np.linspace(0.0, t_final, nt_img)
    temporal_window = {
        "t_on": float(fcfg.get("t_on", 0.0)),
        "t_off": float(fcfg.get("t_off", 0.2)),
        "phase": float(fcfg.get("phase", 0.0)),
        "tukey_alpha": float(fcfg.get("tukey_alpha", 0.5)),
    }
    ramp_cfg = fcfg.get("ramp_seconds", None)
    t_ramp = float(ramp_cfg) if ramp_cfg is not None else load_ramp_seconds(
        config["data"]["t_grid_path"]
    )
    fv_dt, fv_dt_source, fv_n_steps = _resolve_forcing_fv_dt(
        config, pino, t_final
    )
    stored_dt = load_solver_dt(config["data"]["t_grid_path"])
    if stored_dt is None:
        raise ValueError("forcing residual comparison requires dataset dt.npy metadata")
    if not math.isclose(float(stored_dt), fv_dt, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError(
            f"configured FV dt={fv_dt} conflicts with dataset dt.npy={stored_dt}"
        )
    if t_ramp is None:
        t_ramp = default_ramp_seconds(fv_dt)
    physics, interface_face, exclusion = _validate_forcing_layered_dataset(
        problem, data, sim_params[val_ids], fv_dt
    )
    if reformulation_enabled:
        f = int(interface_face)
        if f - 3 < 0 or f + 4 >= Nx:
            raise ValueError("six-column interface trace guard does not fit this grid")
        exclusion = (
            0.5 * (x_np[f - 3] + x_np[f - 2]),
            0.5 * (x_np[f + 3] + x_np[f + 4]),
        )
    if not math.isclose(float(physics["T_right"]), T_RIGHT):
        raise ValueError("ForcingProblem right boundary conflicts with the CViT contract")

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final,
        variant="interfaces",
    ).to(device)
    if not model.hard_right_dirichlet:
        raise ValueError("forcing residual comparison requires hard_right_dirichlet=true")
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    x_grid = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(y_np, dtype=torch.float32, device=device)

    sim_batch = int(pino["sim_batch"])
    intervals_per_sim = int(pino.get("intervals_per_sim", 2))
    n_r = int(pino["n_r"])
    n_bc = int(pino["n_bc"])
    n_ic = int(pino["n_ic"])
    n_bins = int(pino.get("causal_num_bins", 6))
    stratified = bool(pino.get("stratified_time_sampling", True))
    chunk_r = int(pino.get("chunk_r", 0))
    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))
    grad_clip_cfg = config["training"].get("grad_clip", None)
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    rngs = HybridForcingRNGs.create(seed, device)
    gradnorm = build_gradnorm(
        config,
        term_weights={"discriminating": 1.0, "stiff_physics": 1.0},
    )
    constraint_controller = None
    constraint_floors: dict[str, float] = {}
    if reformulation_enabled:
        constraint_floors, constraint_tolerances = _calibrate_forcing_constraint_tolerances(
            problem, _forcing_diagnostic_probe_params(), physics, x_np, y_np,
            dt=fv_dt, t_final=t_final, t_ramp=t_ramp, sigma=sigma,
            q_ref=q_ref, reformulation_cfg=reformulation_cfg,
        )
        constraint_controller = ForcingConstraintController(
            constraint_tolerances,
            rho=float(reformulation_cfg.get("rho", 1.0)),
            ema_decay=float(reformulation_cfg.get("ema_decay", 0.9)),
            dual_every=int(reformulation_cfg.get("dual_every", 10)),
            multiplier_cap=float(reformulation_cfg.get("multiplier_cap", 1000.0)),
            dual_enabled=bool(reformulation_cfg.get("dual_enabled", True)),
        )

    diagnostics_cfg = dict(pino.get("diagnostics", {}) or {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", False))
    diagnostics_every = int(diagnostics_cfg.get("every_updates", 50))
    if diagnostics_enabled and diagnostics_every < 1:
        raise ValueError("training.pino.diagnostics.every_updates must be >= 1")
    diagnostic_fields = [
        "epoch", "completed_updates", "drift_from_init_K",
        "departure_from_300_K", "field_rmse_K", "forcing_sensitivity_K",
        "forcing_response_ratio", "rc_sensitivity_K", "rc_response_ratio",
        "pred_jump_rms_K", "true_jump_rms_K", "node_jump_rmse_K",
        "probe_loss_energy_left", "probe_loss_energy_right",
        "probe_loss_energy_global", "probe_loss_energy",
        "grad_norm_discriminating", "grad_norm_stiff_physics",
        "grad_norm_homogeneous", "grad_ratio_discriminating_to_physics",
        "grad_cos_discriminating_physics", "gradient_share_discriminating",
        "effective_gradient_share_discriminating",
        "grad_projection_on_discriminating", "grad_projection_on_physics",
        "grad_norm_discriminating_weighted",
        "grad_norm_stiff_physics_weighted", "grad_norm_homogeneous_weighted",
        "grad_ratio_discriminating_to_physics_weighted",
        "grad_cos_discriminating_physics_weighted",
        "gradient_share_discriminating_weighted",
        "effective_gradient_share_discriminating_weighted",
        "grad_projection_on_discriminating_weighted",
        "grad_projection_on_physics_weighted",
        "grad_norm_discriminating_decoder",
        "grad_norm_stiff_physics_decoder",
        "grad_cos_discriminating_physics_decoder",
        "grad_norm_discriminating_forcing_encoder",
        "grad_norm_stiff_physics_forcing_encoder",
        "grad_cos_discriminating_physics_forcing_encoder",
        "grad_norm_discriminating_param_encoder",
        "grad_norm_stiff_physics_param_encoder",
        "grad_cos_discriminating_physics_param_encoder",
        "gradnorm_multiplier_discriminating",
        "gradnorm_multiplier_stiff_physics",
        "causal_stage", "causal_fraction", "constraint_max_block",
        *[f"constraint_rms_{name}" for name in _FORCING_CONSTRAINT_NAMES],
        *[f"constraint_violation_{name}" for name in _FORCING_CONSTRAINT_NAMES],
        *[f"constraint_multiplier_{name}" for name in _FORCING_CONSTRAINT_NAMES],
    ]
    diagnostics_path = run_dir / "diagnostics.csv"
    probe_params = _forcing_diagnostic_probe_params()
    diagnostic_physics = {**physics, "problem": problem, "fv_dt": fv_dt}
    init_probe_outputs = None
    if diagnostics_enabled:
        init_probe_outputs = _forcing_probe_model_outputs(
            model, probe_params, mu=mu, sigma=sigma, physics=diagnostic_physics,
            x_grid=x_np, y_grid=y_np, t_grid=t_np, y_img=y_img, t_img=t_img,
            q_ref=q_ref, t_ramp=t_ramp, interface_face=interface_face,
            x_samples=int(diagnostics_cfg.get("probe_x_samples", 9)),
            y_samples=int(diagnostics_cfg.get("probe_y_samples", 9)),
            time_samples=int(diagnostics_cfg.get("probe_time_samples", 7)),
            query_chunk=int(diagnostics_cfg.get("probe_query_chunk", 256)),
            device=device,
            energy_scale_floor=float(energy_cfg.get("scale_floor", 1.0)),
        )
        _atomic_torch_save({
            "model_state": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "mu_global": mu,
            "sigma_global": sigma,
            "config": config,
            "completed_updates": 0,
            "probe_spec": {
                "params": copy.deepcopy(probe_params),
                "interface_face": int(interface_face),
                "t_grid": t_np.tolist(),
            },
        }, run_dir / "cvit_init.pt")
        with open(diagnostics_path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=diagnostic_fields).writeheader()

        def write_diagnostics(epoch: int, completed: int) -> None:
            nonlocal init_probe_outputs
            probe_outputs = _forcing_probe_model_outputs(
                model, probe_params, mu=mu, sigma=sigma,
                physics=diagnostic_physics, x_grid=x_np, y_grid=y_np,
                t_grid=t_np, y_img=y_img, t_img=t_img, q_ref=q_ref,
                t_ramp=t_ramp, interface_face=interface_face,
                x_samples=int(diagnostics_cfg.get("probe_x_samples", 9)),
                y_samples=int(diagnostics_cfg.get("probe_y_samples", 9)),
                time_samples=int(diagnostics_cfg.get("probe_time_samples", 7)),
                query_chunk=int(diagnostics_cfg.get("probe_query_chunk", 256)),
                device=device,
                energy_scale_floor=float(energy_cfg.get("scale_floor", 1.0)),
                truth_cache=init_probe_outputs["truth_full"],
            )
            metrics = _forcing_probe_metrics(probe_outputs, init_probe_outputs)
            gradients = _forcing_diagnostic_gradient_metrics(
                model, residual_method=residual_method,
                physics=physics, interface_face=interface_face,
                exclusion=exclusion, x_grid=x_grid, y_grid=y_grid,
                y_img=y_img, t_img=t_img, mu=mu, sigma=sigma, q_ref=q_ref,
                t_ref=t_ref, t_final=t_final, t_ramp=t_ramp, dt=fv_dt,
                n_steps=fv_n_steps,
                intervals_per_sim=max(int(intervals_per_sim), 1),
                stratified=stratified, n_bins=n_bins,
                n_r=int(diagnostics_cfg.get("gradient_n_r", 128)),
                n_bc=int(diagnostics_cfg.get("gradient_n_bc", 64)),
                n_ic=int(diagnostics_cfg.get("gradient_n_ic", 64)),
                chunk_r=chunk_r,
                seed=seed + 1000003,
                weights=weights,
                energy_cfg=energy_cfg,
                gradnorm_multipliers=(
                    gradnorm.multipliers_for(
                        ["discriminating", "stiff_physics"]
                    ) if gradnorm is not None else None
                ),
                reformulation_cfg=reformulation_cfg,
                constraint_controller=constraint_controller,
            )
            row = {"epoch": int(epoch), "completed_updates": int(completed)}
            if reformulation_enabled:
                stage_updates = list(reformulation_cfg.get(
                    "stage_updates", [0, 200, 400, 600, 800]
                ))
                fractions = list(reformulation_cfg.get(
                    "stage_fractions", [0.1, 0.25, 0.5, 0.75, 1.0]
                ))
                stage = max(i for i, start in enumerate(stage_updates) if completed >= start)
                causal_enabled = bool(reformulation_cfg.get("causal_enabled", True))
                row.update(
                    causal_stage=(stage if causal_enabled else 0),
                    causal_fraction=(fractions[stage] if causal_enabled else 1.0),
                )
            row.update(metrics)
            row.update(gradients)
            with open(diagnostics_path, "a", newline="") as handle:
                csv.DictWriter(
                    handle, fieldnames=diagnostic_fields,
                    extrasaction="ignore",
                ).writerow(row)

        write_diagnostics(epoch=0, completed=0)

    image_spec = {
        "representation": "space_time_image",
        "version": 1,
        "forcing_schema_version": FORCING_SCHEMA_VERSION,
        "axis_order": "channel_y_time",
        "dtype": "float32",
        "ny_img": ny_img,
        "nt_img": nt_img,
        "patch_size": int(model.forcing_patch_size),
        "include_endpoints": True,
        "y_min": float(y_img[0]),
        "y_max": float(y_img[-1]),
        "t_min": 0.0,
        "t_final": t_final,
        "sign_convention": "positive_inward_left_flux",
        "normalization": "fixed_division",
        "a_ref": q_ref,
        "clipping": False,
        "ramp": {
            "type": "cubic_smoothstep", "version": RAMP_SCHEMA_VERSION,
            "duration": float(t_ramp),
        },
        "fv_dt": float(fv_dt),
        "spatial_grid_size": [Nx, Ny],
    }

    family_columns = [
        *[f"gnrmse_temporal_{name}" for name in ("sin", "exp", "pulse_train", "exp_train")],
        *[f"gnrmse_spatial_{name}" for name in ("uniform", "patch", "gaussian", "triangle")],
    ]
    lead_columns = [
        "gnrmse_lead_short", "gnrmse_lead_mid", "gnrmse_lead_long",
    ]
    fieldnames = [
        "epoch", "completed_updates", "residual_method", "loss",
        "loss_interior", "loss_interface", "loss_left_neumann",
        "loss_top", "loss_bottom", "loss_topbot", "loss_ic", "loss_right_dir",
        "loss_energy_left", "loss_energy_right", "loss_energy_global",
        "loss_energy", "loss_group_discriminating",
        "loss_group_stiff_physics", "loss_group_homogeneous",
        "gradnorm_multiplier_discriminating",
        "gradnorm_multiplier_stiff_physics",
        "causal_stage", "causal_fraction", "constraint_max_block",
        *[f"constraint_rms_{name}" for name in _FORCING_CONSTRAINT_NAMES],
        *[f"constraint_violation_{name}" for name in _FORCING_CONSTRAINT_NAMES],
        *[f"constraint_multiplier_{name}" for name in _FORCING_CONSTRAINT_NAMES],
        "lr", "epoch_seconds", "elapsed_seconds",
        "val_gnrmse", "val_rmse_K", "node_jump_rmse_K",
        "interface_flux_mismatch", "interface_contact_rmse_K",
        "interface_contact_rmse_norm", "interface_energy_rms",
        *lead_columns,
        *family_columns,
    ]
    metrics_path = run_dir / "train_metrics.csv"
    with open(metrics_path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    print(
        f"[pino-forcing-interface] seed={seed} method={residual_method} device={device} "
        f"grid={Nx}x{Ny} dt={fv_dt} t_ref={t_ref} q_ref={q_ref} "
        f"interface_face={interface_face} exclusion={exclusion} "
        f"online_forcing={('fixed_probe:' + str(fixed_probe_case)) if fixed_train_param is not None else 'all_families'} "
        f"R_c_range={FORCING_RC_RANGE} "
        f"weights={weights} energy={energy_cfg if energy_enabled else 'off'} "
        f"gradnorm={'two_group' if gradnorm is not None else 'off'} "
        f"reformulation={reformulation_cfg if reformulation_enabled else 'off'} "
        f"constraint_floors={constraint_floors}",
        flush=True,
    )
    best_val = float("inf")
    best_validation: dict[str, float] | None = None
    completed_updates = 0
    stage_updates = [int(v) for v in reformulation_cfg.get(
        "stage_updates", [0, 200, 400, 600, 800]
    )]
    stage_fractions = [float(v) for v in reformulation_cfg.get(
        "stage_fractions", [0.1, 0.25, 0.5, 0.75, 1.0]
    )]
    if len(stage_updates) != len(stage_fractions) or stage_updates[0] != 0:
        raise ValueError("causal stage updates/fractions must align and start at zero")
    if stage_updates != sorted(stage_updates) or any(
        not 0.0 < value <= 1.0 for value in stage_fractions
    ):
        raise ValueError("invalid causal stage schedule")
    previous_stage = 0
    start_time = time.perf_counter()
    last_payload = None
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        if fixed_probe_cases is not None:
            selected_case = fixed_probe_cases[completed_updates % len(fixed_probe_cases)]
            selected = _forcing_fixed_probe_case(selected_case)
            params = [copy.deepcopy(selected) for _ in range(sim_batch)]
        elif fixed_train_param is None:
            params = _sample_online_layered_forcing_params(
                rngs,
                sim_batch,
                dt=fv_dt,
                t_final=t_final,
                y_bounds=(float(y_np[0]), float(y_np[-1])),
                temporal_window=temporal_window,
            )
        else:
            params = [
                copy.deepcopy(fixed_train_param) for _ in range(sim_batch)
            ]

        stage = max(i for i, start in enumerate(stage_updates) if completed_updates >= start)
        if stage != previous_stage and constraint_controller is not None:
            constraint_controller.reset_stage_history()
        previous_stage = stage
        causal_on = reformulation_enabled and bool(
            reformulation_cfg.get("causal_enabled", True)
        )
        active_fraction = stage_fractions[stage] if causal_on else 1.0
        active_steps = max(1, int(math.ceil(fv_n_steps * active_fraction)))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, right_dir = _forcing_layered_batch_losses(
            model, params, residual_method=residual_method, physics=physics,
            interface_face=interface_face, exclusion=exclusion,
            x_grid=x_grid, y_grid=y_grid, y_img=y_img, t_img=t_img,
            mu=mu, sigma=sigma, q_ref=q_ref, t_ref=t_ref,
            t_final=t_final, t_ramp=t_ramp, dt=fv_dt,
            n_steps=fv_n_steps, intervals_per_sim=intervals_per_sim,
            stratified=stratified, n_bins=n_bins, n_r=n_r, n_bc=n_bc,
            n_ic=n_ic, chunk_r=chunk_r, rngs=rngs, energy_cfg=energy_cfg,
            reformulation_cfg=reformulation_cfg,
            max_start_step=(active_steps - 1 if causal_on else None),
            active_t_final=(active_steps * fv_dt if causal_on else None),
        )
        if constraint_controller is not None:
            constraint_residuals = {
                name: losses[f"_constraint_{name}"]
                for name in _FORCING_CONSTRAINT_NAMES
            }
            constraint_rms, constraint_violation = constraint_controller.values(
                constraint_residuals
            )
            groups = {
                "discriminating": constraint_controller.primal(constraint_violation),
                "stiff_physics": (
                    weights["interior"] * losses["interior"]
                    + weights["interface"] * losses["interface"]
                ),
                "homogeneous": (
                    weights["ic"] * losses["ic"]
                    + weights["topbot_adiabatic"] * losses["topbot_adiabatic"]
                ),
            }
        else:
            constraint_rms = constraint_violation = {}
            groups = _forcing_group_losses(
                losses, weights, energy_weight=energy_weight,
            )
        if gradnorm is not None:
            group_multipliers = gradnorm.maybe_update(
                {
                    "discriminating": groups["discriminating"],
                    "stiff_physics": groups["stiff_physics"],
                },
                model.parameters(),
            )
        else:
            group_multipliers = {}
        multiplier_d = float(group_multipliers.get("discriminating", 1.0))
        multiplier_p = float(group_multipliers.get("stiff_physics", 1.0))
        loss = (
            multiplier_d * groups["discriminating"]
            + multiplier_p * groups["stiff_physics"]
            + groups["homogeneous"]
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite forcing {residual_method} loss")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"non-finite forcing {residual_method} gradients")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        completed_updates += 1
        if constraint_controller is not None:
            constraint_controller.update(constraint_violation, completed_updates)

        row: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "residual_method": residual_method,
            "loss": float(loss.detach().cpu()),
            "loss_interior": float(losses["interior"].detach().cpu()),
            "loss_interface": float(losses["interface"].detach().cpu()),
            "loss_left_neumann": float(losses["left_neumann"].detach().cpu()),
            "loss_top": float(losses["top_adiabatic"].detach().cpu()),
            "loss_bottom": float(losses["bottom_adiabatic"].detach().cpu()),
            "loss_topbot": float(losses["topbot_adiabatic"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_right_dir": float(right_dir.detach().cpu()),
            "loss_energy_left": float(
                losses.get("energy_left", loss.new_zeros(())).detach().cpu()
            ),
            "loss_energy_right": float(
                losses.get("energy_right", loss.new_zeros(())).detach().cpu()
            ),
            "loss_energy_global": float(
                losses.get("energy_global", loss.new_zeros(())).detach().cpu()
            ),
            "loss_energy": float(
                losses.get("energy", loss.new_zeros(())).detach().cpu()
            ),
            "loss_group_discriminating": float(
                groups["discriminating"].detach().cpu()
            ),
            "loss_group_stiff_physics": float(
                groups["stiff_physics"].detach().cpu()
            ),
            "loss_group_homogeneous": float(
                groups["homogeneous"].detach().cpu()
            ),
            "gradnorm_multiplier_discriminating": multiplier_d,
            "gradnorm_multiplier_stiff_physics": multiplier_p,
            "causal_stage": stage,
            "causal_fraction": active_fraction,
            "constraint_max_block": float(
                losses.get("local_energy_max_block", loss.new_zeros(())).detach().cpu()
            ),
            "lr": lr,
        }
        for name in _FORCING_CONSTRAINT_NAMES:
            if constraint_controller is not None:
                row[f"constraint_rms_{name}"] = float(constraint_rms[name].detach().cpu())
                row[f"constraint_violation_{name}"] = float(
                    constraint_violation[name].detach().cpu()
                )
                row[f"constraint_multiplier_{name}"] = float(
                    constraint_controller.multipliers[name]
                )
        do_validation = epoch % validate_every == 0 or epoch == epochs - 1
        is_best = False
        if do_validation:
            model.eval()
            with torch.no_grad():
                validation = validate_forcing_interface_gnrmse(
                    model, data, data["val_ids"], sim_params,
                    physics=physics, t_ramp=t_ramp, q_ref=q_ref,
                    y_img=y_img, t_img=t_img, t_ref=t_ref, device=device,
                )
            row.update(validation)
            is_best = validation["val_gnrmse"] < best_val
            if is_best:
                best_val = validation["val_gnrmse"]
                best_validation = {
                    name: float(value) for name, value in validation.items()
                }

        row["epoch_seconds"] = time.perf_counter() - epoch_start
        row["elapsed_seconds"] = time.perf_counter() - start_time
        with open(metrics_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row)

        manifest = {
            "residual_method": residual_method,
            "weights": copy.deepcopy(weights),
            "energy": {
                "enabled": bool(energy_enabled),
                "lambda": float(energy_weight),
                "scale_floor": float(energy_cfg.get("scale_floor", 1.0)),
            },
            "reformulation": ({
                "enabled": True,
                "config": copy.deepcopy(reformulation_cfg),
                "measured_floors": copy.deepcopy(constraint_floors),
                "tolerances": copy.deepcopy(constraint_controller.tolerances),
                "controller_state": constraint_controller.state_dict(),
                "ad_exclusion": [float(exclusion[0]), float(exclusion[1])],
            } if constraint_controller is not None else {"enabled": False}),
            "t_ref": float(t_ref),
            "q_ref": float(q_ref),
            "mu_global": mu,
            "sigma_global": sigma,
            "training_problem_sampling": {
                "forcing": (
                    f"fixed_probe:{fixed_probe_case}"
                    if fixed_train_param is not None else (
                        "fixed_probe_cycle:" + ",".join(fixed_probe_cases)
                        if fixed_probe_cases is not None else "online_all_families"
                    )
                ),
                "contact_resistance": (
                    "fixed_probe"
                    if fixed_train_param is not None or fixed_probe_cases is not None
                    else "online_uniform"
                ),
                "R_c_range": list(FORCING_RC_RANGE),
                "initial_condition_K": float(physics["T_right"]),
            },
            "rng_state": rngs.state_dict(),
        }
        last_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "mu_global": mu,
            "sigma_global": sigma,
            "config": config,
            "epoch": epoch,
            "completed_updates": completed_updates,
            "best_val_gnrmse": best_val,
            "best_validation_metrics": copy.deepcopy(best_validation),
            "interface_forcing": copy.deepcopy(image_spec),
            "physics_manifest": manifest,
            "gradnorm_state": (
                gradnorm.state_dict() if gradnorm is not None else None
            ),
            "constraint_state": (
                constraint_controller.state_dict()
                if constraint_controller is not None else None
            ),
            "hybrid_rng_state": rngs.state_dict(),
            "causal_stage": stage,
        }
        if is_best:
            _atomic_torch_save(last_payload, run_dir / "cvit_best_global.pt")
        save_latest_every = int(fcfg.get("save_latest_every", 25))
        if save_latest_every < 1:
            raise ValueError("training.pino.forcing.save_latest_every must be >= 1")
        if completed_updates % save_latest_every == 0:
            _atomic_torch_save(last_payload, run_dir / "cvit_last.pt")
        if diagnostics_enabled and (
            completed_updates % diagnostics_every == 0 or epoch == epochs - 1
        ):
            write_diagnostics(epoch=epoch, completed=completed_updates)
        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(bulk={row['loss_interior']:.6f}, interface={row['loss_interface']:.6f}, "
            f"left={row['loss_left_neumann']:.6f}, tb={row['loss_topbot']:.6f}, "
            f"ic={row['loss_ic']:.6f}, energy={row['loss_energy']:.6f})"
            + (
                f" wD/P={multiplier_d:.3g}/{multiplier_p:.3g}"
                if gradnorm is not None else ""
            )
            + (
                f" val={row['val_gnrmse'] * 100:.4f}%"
                if do_validation else ""
            ),
            flush=True,
        )

    if last_payload is None:
        raise ValueError("training.epochs must be positive")
    _atomic_torch_save(last_payload, run_dir / "cvit_last.pt")
    wall_seconds = time.perf_counter() - start_time
    summary = {
        "seed": int(seed),
        "residual_method": residual_method,
        "best_val_gnrmse": best_val,
        "epochs": epochs,
        "completed_updates": completed_updates,
        "wall_seconds": wall_seconds,
        "gpu_hours": wall_seconds / 3600.0,
        "fv_dt": float(fv_dt),
        "fv_dt_source": fv_dt_source,
        "t_ref": float(t_ref),
        "q_ref": float(q_ref),
        "best_validation_metrics": best_validation,
    }
    _atomic_text(json.dumps(summary, indent=2) + "\n", run_dir / "final_metrics.json")
    return summary


def run_one_seed_interfaces_pino(
    config: dict, seed: int, run_dir: Path
) -> dict[str, Any]:
    """Physics-only training of an :class:`InterfaceCViT` on the `interfaces`
    benchmark with an FV Crank-Nicolson interface residual.

    `online` collocation (default) draws interface, resistance, forcing, and a
    balanced IC family batch from independent streams. `saved_train` remains an
    explicit diagnostic that reads complete records from the saved training
    split. Both score freshly sampled CN intervals, and validation remains fixed
    to the saved validation trajectories. GradNorm balances only the nontrivial terms
    `{interior, left_neumann, topbot_adiabatic, ic}`; `right_dirichlet` is exact
    under the hard ansatz (~0 gradient) so it is logged, not balanced. Causal
    interior weighting (stratified `t_mid` bins) is optional and off for the tiny
    debug run. Validation uses :func:`validate_interfaces_gnrmse` and three
    checkpoints with a lexicographic jump gate (plan Section 6).
    """
    if bool(
        (config.get("model", {}).get("interface_cvit", {}) or {}).get(
            "jump_enrichment", False
        )
    ):
        raise ValueError(
            "jump_enrichment is not consumed by the collapse interfaces runner; "
            "set training.pino.mode=one_step or disable the enrichment."
        )
    set_seed(seed)
    run_dir = Path(run_dir)
    stale_artifacts = [
        run_dir / name for name in (
            "train_metrics.csv", "cvit_best_global.pt", "cvit_best_jump.pt",
            "cvit_last.pt", "final_metrics.json",
        )
        if (run_dir / name).exists()
    ]
    if stale_artifacts:
        raise FileExistsError(
            "InterfaceCViT image runs require a fresh experiment name; found "
            + ", ".join(str(path) for path in stale_artifacts)
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid_np = np.asarray(data["t_grid"], dtype=np.float64)
    Nx, Ny = int(x_grid_np.shape[0]), int(y_grid_np.shape[0])
    Nq = Nx * Ny
    t_final = float(t_grid_np[-1])

    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(str(sp_path), allow_pickle=True)

    pino = config["training"]["pino"]
    fcfg = pino.get("forcing", {}) or {}
    a_ref = float(fcfg.get("a_ref") if fcfg.get("a_ref") is not None else A_REF_FLUX)
    if a_ref <= 0.0:
        raise ValueError(f"training.pino.forcing.a_ref must be > 0; got {a_ref}.")
    ny_img = int(fcfg.get("ny_img") or 96)
    nt_img = int(fcfg.get("nt_img") or 256)
    if ny_img < 2 or nt_img < 2:
        raise ValueError(
            "training.pino.forcing.ny_img and nt_img must both be >= 2 to "
            f"include both schedule endpoints; got {ny_img} and {nt_img}."
        )
    y_img = np.linspace(float(y_grid_np[0]), float(y_grid_np[-1]), ny_img)
    t_img = np.linspace(0.0, t_final, nt_img)

    # Frozen startup ramp shared by the model image and the residual flux
    # (config override wins; else the dataset's stored ramp; else dt-derived).
    ramp_cfg = fcfg.get("ramp_seconds", None)
    if ramp_cfg is not None:
        t_ramp = float(ramp_cfg)
    else:
        t_ramp = load_ramp_seconds(config["data"]["t_grid_path"])
        if t_ramp is None:
            _dt0 = load_solver_dt(config["data"]["t_grid_path"])
            t_ramp = default_ramp_seconds(_dt0 if _dt0 is not None else t_final / 100.0)

    # CN step `dt`: config override, else the solver dt stored with the dataset,
    # else the trajectory time spacing (so the residual matches the FV scheme).
    dt_cfg = pino.get("dt", None)
    if dt_cfg is not None:
        dt = float(dt_cfg)
    else:
        dt = load_solver_dt(config["data"]["t_grid_path"])
        if dt is None:
            dt = t_final / max(len(t_grid_np) - 1, 1)

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final,
        variant="interfaces",
    ).to(device)
    image_spec = {
        "representation": "space_time_image",
        "version": 1,
        "forcing_schema_version": FORCING_SCHEMA_VERSION,
        "axis_order": "channel_y_time",
        "dtype": "float32",
        "ny_img": ny_img,
        "nt_img": nt_img,
        "patch_size": int(model.forcing_patch_size),
        "include_endpoints": True,
        "y_min": float(y_img[0]),
        "y_max": float(y_img[-1]),
        "t_min": float(t_img[0]),
        "t_final": float(t_img[-1]),
        "sign_convention": "positive_inward_left_flux",
        "normalization": "fixed_division",
        "a_ref": a_ref,
        "clipping": False,
        "ramp": {
            "type": "cubic_smoothstep",
            "version": RAMP_SCHEMA_VERSION,
            "duration": float(t_ramp),
        },
        "fv_dt": float(dt),
        "spatial_grid_size": [Nx, Ny],
    }
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    lam_r = float(pino["lambda_r"])
    lam_ic = float(pino["lambda_ic"])
    lam_bc = float(pino["lambda_bc"])
    lam_bc_left_cfg = pino.get("lambda_bc_left", None)
    lam_bc_left = lam_bc if lam_bc_left_cfg is None else float(lam_bc_left_cfg)
    sim_batch = int(pino["sim_batch"])
    intervals_per_sim = int(pino.get("intervals_per_sim", 2))
    stratified = bool(pino.get("stratified_time_sampling", True))
    causal = dict(pino.get("causal", {}) or {})
    causal_enabled = bool(causal.get("enabled", False))
    n_bins = int(pino.get("causal_num_bins", causal.get("n_bins", 6)))
    eps_causal = (
        float(_resolve_forcing_causal(causal)["initial_eps"])
        if causal_enabled else 1.0
    )
    chunk_r = int(pino.get("chunk_r", 0))
    collocation_source = str(pino.get("collocation_source", "online"))
    if collocation_source not in ("saved_train", "online"):
        raise ValueError(
            f"training.pino.collocation_source must be 'saved_train' or 'online'; "
            f"got {collocation_source!r}."
        )
    profile_cfg = dict(pino.get("online_sampling", {}) or {})
    profile_every = max(1, int(profile_cfg.get("profile_every", 100)))
    min_profile_samples = max(1, int(profile_cfg.get("min_profile_samples", 10)))
    warn_fraction = float(profile_cfg.get("warn_fraction", 0.10))
    if not math.isfinite(warn_fraction) or warn_fraction <= 0.0:
        raise ValueError("training.pino.online_sampling.warn_fraction must be > 0")
    profile_ratios: list[float] = []
    profile_warned = False

    # Right wall is 300 K here too (matches the model's hard right-Dirichlet).
    t_right_tilde = (T_RIGHT - mu) / (sigma + 1e-8)
    t_right_tilde_t = torch.tensor(
        float(t_right_tilde), dtype=torch.float32, device=device
    )

    # Interface-band term (Remedy #1): split the two interior node columns
    # flanking the interface face into a disjoint, separately balanced term so
    # GradNorm sees the jump sub-signal isolated from bulk diffusion. Default OFF.
    iface_band_cfg = dict(
        (pino.get("finite_volume", {}) or {}).get(
            "interface_residual_weighting", pino.get("interface_band", {})
        ) or {}
    )
    band_enabled = bool(iface_band_cfg.get("enabled", False))
    lam_band = float(pino.get("lambda_interface", iface_band_cfg.get("lambda", 1.0)))
    if band_enabled and lam_band <= 0.0:
        raise ValueError(
            "training.pino.lambda_interface must be > 0 when FV interface "
            "residual weighting is enabled "
            f"(got {lam_band}); a non-positive weight would remove the band "
            "columns from 'interior' yet drop 'interface_band' from GradNorm, "
            "leaving the interface-adjacent residual completely unpenalized."
        )

    # GradNorm over the nontrivial terms only; right_dirichlet excluded (its hard
    # ansatz makes the loss ~0 with ~0 gradient -> inverse-norm weighting would
    # divide by ~0). None (disabled) is an exact no-op.
    gn_term_weights = {
        "interior": lam_r, "left_neumann": lam_bc_left,
        "topbot_adiabatic": lam_bc, "ic": lam_ic,
    }
    if band_enabled:
        gn_term_weights["interface_band"] = lam_band
    balanced_terms = list(gn_term_weights.keys())

    # Per-region standardization (Remedy #2): frozen calibration scales, default
    # OFF. Not a moving EMA -- the constant per-term scale cancels in GradNorm's
    # relative-progress ratio L_k(t)/L_k(0) only if L_k(0) is standardized too, so
    # scales are frozen by a calibration pass BEFORE GradNorm is built (below).
    std_cfg = dict(pino.get("region_standardize", {}) or {})
    std_enabled = bool(std_cfg.get("enabled", False))
    calibration_steps = int(std_cfg.get("calibration_steps", 50))
    std_eps = float(std_cfg.get("eps", 1e-8))
    if std_enabled:
        if calibration_steps <= 0:
            raise ValueError(
                "training.pino.region_standardize.calibration_steps must be > 0 "
                f"when enabled; got {calibration_steps}."
            )
        if std_eps <= 0.0:
            raise ValueError(
                "training.pino.region_standardize.eps must be > 0 when enabled; "
                f"got {std_eps}."
            )
    std_scales: dict[str, float] = {k: 1.0 for k in balanced_terms}
    std_calib_done = not std_enabled

    # GradNorm is constructed AFTER the calibration pass (§4) so its reference
    # L_k(0) is captured on the same standardized scale used in training.
    gradnorm = None

    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))

    x_grid_t = torch.as_tensor(x_grid_np, dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(y_grid_np, dtype=torch.float32, device=device)
    gx, gy = torch.meshgrid(x_grid_t, y_grid_t, indexing="ij")
    mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)

    rng = np.random.default_rng(seed)
    online_rngs = OnlineSamplerRNGs.create(seed, device)
    train_ids = np.asarray(data["train_ids"])

    # Online collocation draws fresh (interface_x, R_c, IC, forcing) each step
    # from the SAME target distributions the generator uses (IID, not the fixed
    # LHS design of the saved set). Validation always uses the saved val
    # trajectories, so sim_params stays loaded regardless of the source.
    online_spec = problem_from_config(config)
    gxm, gym = np.meshgrid(x_grid_np, y_grid_np, indexing="ij")
    online_grids = {
        "X": gxm, "Y": gym, "x_grid": x_grid_np, "y_grid": y_grid_np,
    }
    online_time_cfg = {
        "dt": dt, "t_final": t_final, "b": 1.0, "T_right": float(T_RIGHT),
        "t_on": float(fcfg.get("t_on", 0.0)),
        "t_off": float(fcfg.get("t_off", 0.2)),
        "phase": float(fcfg.get("phase", 0.0)),
        "tukey_alpha": float(fcfg.get("tukey_alpha", 0.5)),
    }
    if collocation_source == "online":
        online_signature = _online_sampling_signature(online_spec, (Nx, Ny))
        saved_record_sampler = None
    else:
        online_signature = None
        def saved_record_sampler(sample_rng):
            replace = len(train_ids) < sim_batch
            sim_ids = sample_rng.choice(
                train_ids, size=sim_batch, replace=replace,
            )
            return [dict(sim_params[int(i)]) for i in sim_ids]

    # Frozen train jump scale for E_model / E_zero (computed once).
    sigma_dT_train = _compute_train_jump_scale(data, sim_params, train_ids)

    metrics_path = run_dir / "train_metrics.csv"
    gn_cols = list(gn_term_weights.keys())
    band_loss_cols = ["loss_interface_band"] if band_enabled else []
    band_w_cols = ["w_interface_band"] if band_enabled else []
    fieldnames = [
        "epoch", "completed_updates", "loss",
        "loss_interior", "loss_left_neumann", "loss_topbot", "loss_ic",
        "loss_right_dir",
        *band_loss_cols,
        "w_interior", "w_left_neumann", "w_topbot", "w_ic",
        *band_w_cols,
        "val_gnrmse", "val_rmse_K", "node_jump_rmse_K", "E_model", "E_zero",
        "gnrmse_Rc_low", "gnrmse_Rc_mid", "gnrmse_Rc_high",
        "gnrmse_ix_low", "gnrmse_ix_mid", "gnrmse_ix_high",
        *[f"gn_mult_{c}" for c in gn_cols],
        *[f"w_eff_{c}" for c in gn_cols],
        *([f"loss_opt_{c}" for c in gn_cols] if std_enabled else []),
        *([f"std_scale_{c}" for c in gn_cols] if std_enabled else []),
        "gradnorm_weights",
        "online_max_abs_z", "online_frac_abs_z_gt_5", "online_family_z_ranges",
        "problem_keys", "batch_key",
        "online_sampling_fraction",
    ]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()

    print(
        f"[pino-interfaces] seed={seed} device={device} epochs={epochs} "
        f"validate_every={validate_every} | grid={Nx}x{Ny} t_final={t_final:.4f} "
        f"dt={dt:.4g} t_ramp={t_ramp:.4g} "
        f"forcing_image={ny_img}x{nt_img}/p{model.forcing_patch_size} "
        f"a_ref={a_ref} | "
        f"sim_batch={sim_batch} intervals_per_sim={intervals_per_sim} "
        f"collocation={collocation_source} "
        f"stratified={stratified} causal={causal_enabled}(n_bins={n_bins}) | "
        f"lambda_r={lam_r} lambda_ic={lam_ic} lambda_bc={lam_bc} "
        f"lambda_bc_left={lam_bc_left} | k=({K_LEFT},{K_RIGHT}) "
        f"sigma_dT_train={sigma_dT_train:.4g} | "
        f"interface_residual_weighting={band_enabled}(lambda={lam_band})",
        flush=True,
    )

    # Constant static per-term weights (config lambdas); GradNorm multiplies these.
    sw = {
        "interior": lam_r, "left_neumann": lam_bc_left,
        "topbot_adiabatic": lam_bc, "ic": lam_ic,
    }
    if band_enabled:
        sw["interface_band"] = lam_band

    last_online_batch: dict[str, Any] | None = None
    last_online_diagnostics: dict[str, Any] | None = None
    last_online_timing = {"descriptor": 0.0, "build": 0.0, "transfer": 0.0}

    def _forward_losses(
        sample_rng=None, *, online_batch: dict[str, Any] | None = None,
        rng_bundle: OnlineSamplerRNGs | None = None, profile: bool = False,
    ):
        """One collocation batch -> raw (pre-standardization) balanced-term
        losses dict + right-Dirichlet diagnostic. Shared by the §4 calibration
        pass and the training loop so both use identical reductions."""
        nonlocal last_online_batch, last_online_diagnostics, last_online_timing
        descriptor_start = time.perf_counter()
        if collocation_source == "online":
            bundle = online_rngs if rng_bundle is None else rng_bundle
            if online_batch is None:
                descriptors = _sample_online_problem_descriptors(
                    online_spec, bundle, sim_batch, online_grids, online_time_cfg,
                )
                t_n, t_np1, t_mid, bin_ids = _sample_interval_times(
                    bundle.numpy["fv_intervals"], sim_batch, intervals_per_sim,
                    dt, t_final, stratified, n_bins,
                )
                interval_descriptor = {
                    "t_n": np.asarray(t_n, dtype="<f8"),
                    "t_np1": np.asarray(t_np1, dtype="<f8"),
                    "t_mid": np.asarray(t_mid, dtype="<f8"),
                    "causal_bin_ids": np.asarray(bin_ids, dtype="<i8"),
                    "sim_local": np.repeat(
                        np.arange(sim_batch, dtype="<i8"), intervals_per_sim,
                    ),
                }
                online_batch = {
                    "records": descriptors,
                    "intervals": interval_descriptor,
                    "batch_key": _batch_key(descriptors, interval_descriptor),
                }
            descriptors = copy.deepcopy(online_batch["records"])
            params = _materialize_online_records(
                descriptors, online_grids["X"], online_grids["Y"],
                T_right=T_RIGHT, b=float(x_grid_np[-1]),
            )
            intervals = online_batch["intervals"]
            t_n = np.asarray(intervals["t_n"], dtype=np.float64)
            t_np1 = np.asarray(intervals["t_np1"], dtype=np.float64)
            t_mid = np.asarray(intervals["t_mid"], dtype=np.float64)
            bin_ids = np.asarray(intervals["causal_bin_ids"], dtype=np.int64)
            last_online_batch = copy.deepcopy(online_batch)
        else:
            if sample_rng is None:
                raise ValueError("saved_train collocation requires a sampling RNG")
            params = saved_record_sampler(sample_rng)
            last_online_batch = None
        B = len(params)
        descriptor_seconds = time.perf_counter() - descriptor_start

        build_start = time.perf_counter()
        _physical_ic, normalized_ic, _online_diag = _normalized_online_ic_buffer(
            params, mu, sigma,
        )
        last_online_diagnostics = _online_diag
        spatial_cpu = np.ascontiguousarray(np.stack([
            _interface_spatial_channels_from_normalized(
                normalized_ic[index], p["interface_x"], x_grid_np,
            )
            for index, p in enumerate(params)
        ]), dtype=np.float32)
        build_seconds = time.perf_counter() - build_start
        if profile and device.type == "cuda":
            torch.cuda.synchronize()
        transfer_start = time.perf_counter()
        spatial_tensor = torch.from_numpy(spatial_cpu)
        if device.type == "cuda":
            spatial_tensor = spatial_tensor.pin_memory()
            u_spatial = spatial_tensor.to(device, non_blocking=True)
        else:
            u_spatial = spatial_tensor.to(device)
        forcing_image = build_forcing_image(
            params, y_img, t_img, a_ref, device, t_ramp
        )
        pscal = torch.from_numpy(
            np.stack([
                normalize_interface_scalars(p["interface_x"], p["R_c"])
                for p in params
            ])
        ).to(device)
        if profile and device.type == "cuda":
            torch.cuda.synchronize()
        transfer_seconds = time.perf_counter() - transfer_start
        last_online_timing = {
            "descriptor": descriptor_seconds,
            "build": build_seconds,
            "transfer": transfer_seconds,
        }
        ic_target = u_spatial[:, 0].reshape(B, Nq, 1)
        interface_x = np.array([float(p["interface_x"]) for p in params])
        R_c = np.array([float(p["R_c"]) for p in params])

        latent = model.encode(u_spatial, forcing_image, pscal)

        # IC anchor: decode at t=0 against the (varying) normalized IC field.
        t0q = torch.zeros((B, Nq, 1), device=device)
        ic_pred = _decode_in_chunks(model, latent, mesh.expand(B, -1, -1), t0q, chunk_r)
        ic_loss = ((ic_pred - ic_target) ** 2).mean()

        # CN intervals over M = B * intervals_per_sim.
        if collocation_source != "online":
            t_n, t_np1, t_mid, bin_ids = _sample_interval_times(
                sample_rng, B, intervals_per_sim, dt, t_final, stratified, n_bins
            )
        M = B * intervals_per_sim
        sim_local = np.repeat(np.arange(B), intervals_per_sim)      # (M,)
        latent_M = latent[torch.as_tensor(sim_local, device=device)]

        qL_n = np.empty((M, Ny), dtype=np.float64)
        qL_np1 = np.empty((M, Ny), dtype=np.float64)
        qL_int = np.empty((M, Ny), dtype=np.float64)
        for m in range(M):
            p = params[int(sim_local[m])]
            qn, qnp1, qint = build_interface_forcing(
                p["temporal_family"], p["temporal_params"],
                p["spatial_family"], p["spatial_params"],
                y_grid_np, float(t_n[m]), float(t_np1[m]), t_ramp,
            )
            qL_n[m] = qn
            qL_np1[m] = qnp1
            qL_int[m] = qint

        coords_M = mesh.expand(M, -1, -1)
        tn_q = (torch.from_numpy(t_n).to(torch.float32).to(device)
                .view(M, 1, 1).expand(M, Nq, 1))
        tnp1_q = (torch.from_numpy(t_np1).to(torch.float32).to(device)
                  .view(M, 1, 1).expand(M, Nq, 1))
        T_n = _decode_in_chunks(model, latent_M, coords_M, tn_q, chunk_r).view(M, Nx, Ny)
        T_np1 = _decode_in_chunks(model, latent_M, coords_M, tnp1_q, chunk_r).view(M, Nx, Ny)

        geom = build_cn_geom_per_interface(
            x_grid_np, y_grid_np, K_LEFT, K_RIGHT,
            interface_x[sim_local], R_c[sim_local], dt,
            sigma_global=float(sigma), device=device, dtype=torch.float32,
        )
        bc = FullBCData(
            T_right_tilde=t_right_tilde_t,
            qL_n=torch.from_numpy(qL_n).to(torch.float32).to(device),
            qL_np1=torch.from_numpy(qL_np1).to(torch.float32).to(device),
            qL_int=torch.from_numpy(qL_int).to(torch.float32).to(device),
        )
        phys = full_bc_physics_loss(
            T_n, T_np1, geom, bc, per_sample=True, interface_band=band_enabled,
        )
        interior_ps = phys["interior_per_sample"]                  # (M,) bulk-only if band

        # Shared causal weights: derived once from the interior bin means and
        # applied identically to the band, so the two disjoint terms share the
        # same temporal (causal-front) and spatial weighting for a clean
        # one-hypothesis comparison and to avoid training late-time interface
        # equations ahead of the causal front.
        if causal_enabled and stratified:
            bins_t = torch.as_tensor(bin_ids, device=device, dtype=torch.long)
            ones = torch.ones_like(interior_ps)
            bcnt = torch.zeros(n_bins, device=device, dtype=interior_ps.dtype)
            bcnt = bcnt.index_add(0, bins_t, ones)
            bsum_int = torch.zeros(n_bins, device=device, dtype=interior_ps.dtype)
            bsum_int = bsum_int.index_add(0, bins_t, interior_ps)
            bmean_int = bsum_int / bcnt.clamp_min(1.0)
            causal_w = _causal_weights(bmean_int.detach(), eps_causal) * (bcnt > 0)

            def _causal_reduce(ps: torch.Tensor) -> torch.Tensor:
                bsum = torch.zeros(n_bins, device=device, dtype=ps.dtype)
                bsum = bsum.index_add(0, bins_t, ps)
                bmean = bsum / bcnt.clamp_min(1.0)
                return (causal_w * bmean).sum() / causal_w.sum().clamp_min(1e-12)
        else:
            def _causal_reduce(ps: torch.Tensor) -> torch.Tensor:
                return ps.mean()

        losses = {
            "interior": _causal_reduce(interior_ps),
            "left_neumann": phys["phys_left_neumann_mse"],
            "topbot_adiabatic": phys["phys_topbot_adiabatic_mse"],
            "ic": ic_loss,
        }
        if band_enabled:
            losses["interface_band"] = _causal_reduce(phys["interface_band_per_sample"])
        right_dir = phys["phys_right_dirichlet_mse"]               # logged only
        return losses, right_dir

    calibration_path = run_dir / "region_standardize_calibration.pt"
    calibration_metadata = {
        "version": 1,
        "collocation_source": collocation_source,
        "seed": int(seed + 1_000_003),
        "calibration_steps": calibration_steps,
        "descriptor_count": int(calibration_steps * sim_batch),
        "sim_batch": sim_batch,
        "intervals_per_sim": intervals_per_sim,
        "stratified_time_sampling": stratified,
        "causal_num_bins": n_bins,
        "dt": float(dt),
        "grid_shape": [Nx, Ny],
        "balanced_terms": list(balanced_terms),
        "eps": std_eps,
        "online_sampling_signature": online_signature,
        "environment": _environment_fingerprint(),
        "model_config_key": _content_key(config.get("model", {})),
    }
    calibration_hash: str | None = None

    # --- §4 calibration pass: freeze per-region scales on the INITIAL model,
    # before any optimizer step and before GradNorm is built. Observationally
    # inert: separate rng, snapshot/restore torch RNG, model state asserted
    # unchanged. This finite-volume path needs only scalar loss values, so retaining
    # decoder graphs would waste the dominant GPU memory.
    if std_enabled:
        if calibration_path.exists():
            artifact, calibration_hash = _load_hashed_calibration(
                calibration_path, calibration_metadata,
            )
            if collocation_source == "online":
                calibration_batches = artifact.get("batches", [])
                descriptor_count = sum(
                    len(batch.get("records", [])) for batch in calibration_batches
                )
                if descriptor_count != calibration_metadata["descriptor_count"]:
                    raise ValueError(
                        "region-standardization calibration descriptor count mismatch"
                    )
                for batch in calibration_batches:
                    descriptors = batch["records"]
                    _materialize_online_records(
                        descriptors, online_grids["X"], online_grids["Y"],
                        T_right=T_RIGHT, b=float(x_grid_np[-1]),
                    )
                    if _batch_key(descriptors, batch["intervals"]) != batch["batch_key"]:
                        raise ValueError(
                            "region-standardization calibration batch key mismatch"
                        )
            std_scales = {
                key: float(value) for key, value in artifact["std_scales"].items()
            }
            std_calib_done = True
            calib_wall = 0.0
        else:
            state_before = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            cpu_rng_before = torch.get_rng_state()
            cuda_rng_before = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            was_training = model.training
            model.train()
            calib_rngs = OnlineSamplerRNGs.create(seed + 1_000_003, device)
            calib_saved_rng = np.random.default_rng(seed + 1_000_003)
            accum = {k: 0.0 for k in balanced_terms}
            calibration_batches: list[dict[str, Any]] = []
            t_calib0 = time.perf_counter()
            with torch.no_grad():
                for _calib_step in range(calibration_steps):
                    losses_c, right_dir_c = _forward_losses(
                        calib_saved_rng, rng_bundle=calib_rngs,
                    )
                    if collocation_source == "online":
                        calibration_batches.append(copy.deepcopy(last_online_batch))
                    for k in balanced_terms:
                        accum[k] += float(losses_c[k].cpu())
                    del losses_c, right_dir_c
            calib_wall = time.perf_counter() - t_calib0
            for k in balanced_terms:
                std_scales[k] = max(accum[k] / calibration_steps, std_eps)
            std_calib_done = True
            model.train(was_training)
            torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
            state_after = model.state_dict()
            for k, v0 in state_before.items():
                if not torch.equal(v0, state_after[k].to(v0.device)):
                    raise RuntimeError(
                        f"region_standardize calibration mutated model state {k!r}; "
                        "the calibration pass must be observationally inert."
                    )
            artifact = {
                "metadata": copy.deepcopy(calibration_metadata),
                "std_scales": dict(std_scales),
                "batches": calibration_batches,
            }
            calibration_hash = _atomic_hashed_torch_save(
                artifact, calibration_path,
            )
            del state_before, state_after, cpu_rng_before, cuda_rng_before
        print(
            "[pino-interfaces] region_standardize: "
            f"calibration_steps={calibration_steps} wall={calib_wall:.2f}s "
            "scales="
            + json.dumps({k: round(float(v), 8) for k, v in std_scales.items()}),
            flush=True,
        )

    calibration_checkpoint = (
        None if not std_enabled else {
            "relative_path": str(calibration_path.relative_to(run_dir)),
            "sha256": calibration_hash,
            "configuration": copy.deepcopy(calibration_metadata),
            "seed": calibration_metadata["seed"],
            "descriptor_count": calibration_metadata["descriptor_count"],
            "sampler_version": ONLINE_IC_SAMPLER_VERSION,
            "builder_version": IC_BUILDER_SCHEMA_VERSION,
            "environment": _environment_fingerprint(),
            "std_scales": dict(std_scales),
        }
    )

    # GradNorm now, after calibration, so its reference L_k(0) sees standardized
    # losses from optimizer step 0.
    gradnorm = build_gradnorm(config, term_weights=gn_term_weights)

    best_global = float("inf")
    best_jump = float("inf")
    completed_updates = 0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        do_val = (epoch % validate_every == 0) or (epoch == epochs - 1)
        profile_this = completed_updates % profile_every == 0
        if profile_this and device.type == "cuda":
            torch.cuda.synchronize()
        profiled_update_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        # Raw (pre-standardization) balanced-term losses via the shared closure.
        losses, right_dir = _forward_losses(rng, profile=profile_this)
        # §4: standardize each balanced term by its frozen calibration scale
        # (identity when region_standardize is disabled). Both L_k(t) here and
        # GradNorm's reference L_k(0) carry the same constant s_k, so it cancels
        # in the relative-progress ratio.
        if std_enabled:
            losses_opt = {k: losses[k] / std_scales[k] for k in balanced_terms}
        else:
            losses_opt = losses

        gn_mults: dict[str, float] = {}
        if gradnorm is not None:
            active = {k: losses_opt[k] for k in gradnorm.term_names if k in losses_opt}
            gn_params = [p for p in model.parameters() if p.requires_grad]
            gn_mults = gradnorm.maybe_update(active, gn_params, dist_info=None)
        w_eff = {k: sw[k] * float(gn_mults.get(k, 1.0)) for k in sw}
        loss = None
        for k, wk in w_eff.items():
            term = wk * losses_opt[k]
            loss = term if loss is None else loss + term
        failure_records = (
            None if last_online_batch is None else last_online_batch["records"]
        )
        failure_key = (
            None if last_online_batch is None else last_online_batch["batch_key"]
        )
        failure_coll = (
            None if last_online_batch is None else last_online_batch["intervals"]
        )
        if not bool(torch.isfinite(loss).item()):
            _write_online_failure(
                run_dir,
                phase="pre_step",
                completed_updates=completed_updates,
                records=failure_records,
                batch_key=failure_key,
                coll=failure_coll,
                rng_state={
                    "online": online_rngs.state_dict(),
                    "saved_train": copy.deepcopy(rng.bit_generator.state),
                },
            )
            raise FloatingPointError(
                f"Non-finite interfaces PINO loss at epoch {epoch}"
            )
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            _write_online_failure(
                run_dir,
                phase="pre_step",
                completed_updates=completed_updates,
                records=failure_records,
                batch_key=failure_key,
                coll=failure_coll,
                rng_state={
                    "online": online_rngs.state_dict(),
                    "saved_train": copy.deepcopy(rng.bit_generator.state),
                },
            )
            raise FloatingPointError(
                f"Non-finite interfaces PINO gradient at epoch {epoch}"
            )
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if not _all_finite(model.state_dict()) or not _all_finite(optimizer.state):
            _write_online_failure(
                run_dir,
                phase="post_step",
                completed_updates=completed_updates,
                records=failure_records,
                batch_key=failure_key,
                coll=failure_coll,
                rng_state={
                    "online": online_rngs.state_dict(),
                    "saved_train": copy.deepcopy(rng.bit_generator.state),
                },
            )
            raise FloatingPointError(
                f"Non-finite interfaces model/optimizer state at epoch {epoch}"
            )
        sampling_fraction: float | str = ""
        if profile_this:
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds = time.perf_counter() - profiled_update_start
            numerator = sum(last_online_timing.values())
            denominator = max(total_seconds - numerator, 1e-12)
            sampling_fraction = numerator / denominator
            profile_ratios.append(float(sampling_fraction))
            if (
                not profile_warned
                and len(profile_ratios) >= min_profile_samples
                and float(np.median(profile_ratios)) > warn_fraction
            ):
                warnings.warn(
                    "Online IC descriptor/build/normalization/transfer median "
                    f"overhead is {np.median(profile_ratios):.1%} of training time.",
                    RuntimeWarning,
                )
                profile_warned = True
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        completed_updates += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "loss": float(loss.detach().cpu()),
            "loss_interior": float(losses["interior"].detach().cpu()),
            "loss_left_neumann": float(losses["left_neumann"].detach().cpu()),
            "loss_topbot": float(losses["topbot_adiabatic"].detach().cpu()),
            "loss_ic": float(losses["ic"].detach().cpu()),
            "loss_right_dir": float(right_dir.detach().cpu()),
            "w_interior": lam_r, "w_left_neumann": lam_bc_left,
            "w_topbot": lam_bc, "w_ic": lam_ic,
            "online_max_abs_z": last_online_diagnostics["max_abs_z"],
            "online_frac_abs_z_gt_5": last_online_diagnostics["frac_abs_z_gt_5"],
            "online_family_z_ranges": json.dumps(
                last_online_diagnostics["per_family"]
            ),
            "problem_keys": (
                "" if last_online_batch is None else json.dumps([
                    record["problem_key"] for record in last_online_batch["records"]
                ])
            ),
            "batch_key": (
                "" if last_online_batch is None else last_online_batch["batch_key"]
            ),
            "online_sampling_fraction": sampling_fraction,
        }
        if band_enabled:
            row["loss_interface_band"] = float(losses["interface_band"].detach().cpu())
            row["w_interface_band"] = lam_band
        if std_enabled:
            for k in gn_cols:
                row[f"loss_opt_{k}"] = float(losses_opt[k].detach().cpu())
                row[f"std_scale_{k}"] = float(std_scales[k])
        band_str = (
            f", band={row['loss_interface_band']:.6f}" if band_enabled else ""
        )
        print(
            f"Epoch {epoch}: loss={row['loss']:.6f} "
            f"(int={row['loss_interior']:.6f}, "
            f"left={row['loss_left_neumann']:.6f}, "
            f"tb={row['loss_topbot']:.6f}, ic={row['loss_ic']:.6f}, "
            f"rd={row['loss_right_dir']:.2e}{band_str}) lr={lr:.2e}",
            flush=True,
        )

        if do_val:
            model.eval()
            with torch.no_grad():
                val = validate_interfaces_gnrmse(
                    model, data, data["val_ids"], sim_params,
                    t_ramp, a_ref, y_img, t_img, sigma_dT_train, device,
                )
            for key in (
                "val_gnrmse", "val_rmse_K", "node_jump_rmse_K", "E_model", "E_zero",
                "gnrmse_Rc_low", "gnrmse_Rc_mid", "gnrmse_Rc_high",
                "gnrmse_ix_low", "gnrmse_ix_mid", "gnrmse_ix_high",
            ):
                row[key] = val.get(key, "")
            if gradnorm is not None:
                for k in gn_cols:
                    row[f"gn_mult_{k}"] = float(gn_mults.get(k, 1.0))
                    row[f"w_eff_{k}"] = w_eff[k]
                row["gradnorm_weights"] = json.dumps(
                    {k: float(v) for k, v in gn_mults.items()}
                )

            ckpt = {
                "model_state": model.state_dict(),
                "mu_global": mu, "sigma_global": sigma,
                "config": config, "epoch": epoch,
                "best_val_gnrmse": val["val_gnrmse"], "E_model": val["E_model"],
                "E_zero": val["E_zero"], "sigma_dT_train": sigma_dT_train,
                "interface_forcing": copy.deepcopy(image_spec),
                "gradnorm_state": (
                    gradnorm.state_dict() if gradnorm is not None else None
                ),
                "region_standardize": {
                    "enabled": std_enabled,
                    "calibration_steps": calibration_steps,
                    "eps": std_eps,
                    "std_scales": dict(std_scales),
                    "std_calib_done": std_calib_done,
                    "artifact": copy.deepcopy(calibration_checkpoint),
                },
                "online_sampling_signature": copy.deepcopy(online_signature),
                "online_environment": _environment_fingerprint(),
            }
            is_best_global = val["val_gnrmse"] < best_global
            if is_best_global:
                best_global = val["val_gnrmse"]
                _atomic_torch_save(ckpt, run_dir / "cvit_best_global.pt")
            # Lexicographic jump gate: only a model that beats the explicit gate
            # E_model <= 0.5 * E_zero is eligible as the jump-best checkpoint.
            passes_gate = val["E_model"] <= 0.5 * val["E_zero"]
            is_best_jump = passes_gate and (val["E_model"] < best_jump)
            if is_best_jump:
                best_jump = val["E_model"]
                _atomic_torch_save(ckpt, run_dir / "cvit_best_jump.pt")

            print(
                f"Validation epoch {epoch}: "
                f"val_gnrmse={val['val_gnrmse'] * 100:.4f}% "
                f"rmse_K={val['val_rmse_K']:.4f}K "
                f"jump_rmse_K={val['node_jump_rmse_K']:.4f}K "
                f"E_model={val['E_model']:.4f} E_zero={val['E_zero']:.4f} "
                f"(gate={'pass' if passes_gate else 'fail'})"
                + ("  [best global]" if is_best_global else "")
                + ("  [best jump]" if is_best_jump else ""),
                flush=True,
            )

        history.append({k: (v if v != "" else None) for k, v in row.items()})
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "mu_global": mu, "sigma_global": sigma,
            "config": config, "epoch": epochs - 1,
            "sigma_dT_train": sigma_dT_train,
            "interface_forcing": copy.deepcopy(image_spec),
            "region_standardize": {
                "enabled": std_enabled,
                "calibration_steps": calibration_steps,
                "eps": std_eps,
                "std_scales": dict(std_scales),
                "std_calib_done": std_calib_done,
                "artifact": copy.deepcopy(calibration_checkpoint),
            },
            "online_sampling_signature": copy.deepcopy(online_signature),
            "online_environment": _environment_fingerprint(),
        },
        run_dir / "cvit_last.pt",
    )

    summary = {
        "seed": seed,
        "best_val_gnrmse": best_global,
        "best_E_model": (best_jump if best_jump != float("inf") else None),
        "sigma_dT_train": sigma_dT_train,
        "epochs": epochs,
    }
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _fixed_interval_indices(n_intervals: int, num_states: int) -> list[int]:
    if int(num_states) < 1:
        raise ValueError("fixed_simulation.num_states must be >= 1")
    if int(num_states) > int(n_intervals):
        raise ValueError(
            "fixed_simulation.num_states cannot exceed the number of FV intervals "
            f"({n_intervals}); got {num_states}."
        )
    return np.rint(
        np.linspace(0, int(n_intervals) - 1, int(num_states))
    ).astype(np.int64).tolist()


def _evaluate_interfaces_fixed_states(
    model: InterfaceCViT,
    case,
    interval_indices: list[int],
    *,
    sigma: float,
    dt: float,
    coords: torch.Tensor,
    query_chunk: int,
    output_parameterization: str,
    a_ref: float,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Evaluate only the frozen training states; FV targets never enter training."""
    from src.operators.one_step import predict_one_step_field

    was_training = model.training
    model.eval()
    steps = torch.as_tensor(interval_indices, dtype=torch.long)
    state_n = case.truth_states.index_select(0, steps).to(device)
    truth_np1 = case.truth_states.index_select(0, steps + 1).to(device)
    batch_size = int(steps.numel())
    fixed_channels = case.fixed_channels.expand(batch_size, -1, -1, -1)
    scalars = case.scalars.expand(batch_size, -1)
    interface_x = case.interface_x.expand(batch_size)
    jump_scale = case.jump_scale.expand(batch_size)

    with torch.no_grad():
        prediction = predict_one_step_field(
            model,
            state_n,
            case.interval_images.index_select(0, steps.to(case.interval_images.device)),
            scalars,
            fixed_channels,
            coords,
            dt=dt,
            query_chunk=int(query_chunk),
            output_parameterization=output_parameterization,
            interface_x=interface_x,
            jump_scale=jump_scale,
            closure_geom=case.geom,
            q_left_integral=case.q_left_integrals.index_select(
                0, steps.to(case.q_left_integrals.device)
            ),
            resistance=case.resistance,
            sigma=sigma,
            q_ref=a_ref,
        )

    error = prediction - truth_np1
    transition = truth_np1 - state_n
    error_ssq = error.square().reshape(batch_size, -1).sum(dim=1)
    transition_ssq = transition.square().reshape(batch_size, -1).sum(dim=1)
    counts = error[0].numel()
    rmse_K = (error_ssq / counts).sqrt() * float(sigma)
    transition_rel = (error_ssq / transition_ssq.clamp_min(1.0e-20)).sqrt()
    aggregate = {
        "fixed_train_rmse_K": float(
            (error_ssq.sum() / (batch_size * counts)).sqrt().cpu() * float(sigma)
        ),
        "fixed_train_transition_rel": float(
            (error_ssq.sum() / transition_ssq.sum().clamp_min(1.0e-20)).sqrt().cpu()
        ),
        "fixed_train_max_rmse_K": float(rmse_K.max().cpu()),
        "fixed_train_max_transition_rel": float(transition_rel.max().cpu()),
        "_error_ssq": float(error_ssq.sum().cpu()),
        "_transition_ssq": float(transition_ssq.sum().cpu()),
        "_value_count": float(batch_size * counts),
    }
    rows = [
        {
            "interval": float(step),
            "t_n": float(step) * float(dt),
            "rmse_K": float(rmse_K[index].cpu()),
            "transition_rel": float(transition_rel[index].cpu()),
        }
        for index, step in enumerate(interval_indices)
    ]
    model.train(was_training)
    return aggregate, rows


def validate_interfaces_one_step_gnrmse(
    model: InterfaceCViT,
    cases: list,
    *,
    mu: float,
    sigma: float,
    sigma_dT_train: float,
    dt: float,
    coords: torch.Tensor,
    query_chunk: int,
    output_parameterization: str = "absolute",
    a_ref: float,
    right_value: float,
    device: torch.device,
) -> dict[str, float]:
    """One-step operator validation: teacher-forced one-step + free rollout.

    ``cases`` is a frozen list of :class:`OneStepCase` (never sampled for
    training). Two modes per case (plan Section 5):

    - teacher-forced one-step: FV-truth ``T_n -> hat T_{n+1}`` vs FV-truth
      ``T_{n+1}`` (a diagnostic that isolates the learned one-step map from
      rollout instability);
    - free rollout: autoregress ``rollout_field`` from ``T_0`` over the whole
      trajectory (the primary gate signal ``val_gnrmse``).

    gnRMSE is the normalized-field RMSE (``rmse_K / sigma``); on the normalized
    states that is exactly ``sqrt(mean (pred - truth)^2)``. The node interface
    jump uses the FROZEN ``sigma_dT_train`` denominator so ``E_zero`` stays
    comparable with the collapse runner. Emits per-case ``R_c`` / ``interface_x``
    strata plus rollout-stability diagnostics (max stepwise / final-time gnRMSE,
    rollout/teacher ratio, max ``|T|``, non-finite count).
    """
    from src.operators.one_step import predict_one_step_field, rollout_field

    was_training = model.training
    model.eval()

    # The hard global-storage residual is only defined when the conservative
    # storage-projection head is active; the learned/closure heads report NaN so
    # an inactive constraint cannot look numerically satisfied.
    want_storage = (
        str(getattr(model, "jump_flux_mode", "learned"))
        == "conservative_storage_projection"
    )

    per_case_rollout: list[float] = []
    per_case_teacher: list[float] = []
    per_case_rc: list[float] = []
    per_case_ix: list[float] = []
    per_case_max_step: list[float] = []
    per_case_final: list[float] = []
    jump_err_ssq = 0.0
    jump_true_ssq = 0.0
    jump_cnt = 0
    max_abs_T_K = 0.0
    nonfinite = 0
    hard_storage_error = 0.0 if want_storage else float("nan")

    with torch.no_grad():
        for case in cases:
            n_intervals = int(case.n_intervals)
            if n_intervals < 1:
                continue
            truth = case.truth_states.to(device)              # (Nt, Nx, Ny)
            face = int(case.geom.face_idx.reshape(-1)[0].item())

            teacher_ssq = 0.0
            teacher_cnt = 0
            for n in range(n_intervals):
                pred = predict_one_step_field(
                    model,
                    truth[n : n + 1],
                    case.interval_images[n : n + 1],
                    case.scalars,
                    case.fixed_channels,
                    coords,
                    dt=dt,
                    query_chunk=int(query_chunk),
                    output_parameterization=output_parameterization,
                    interface_x=case.interface_x,
                    jump_scale=case.jump_scale,
                    closure_geom=case.geom,
                    q_left_integral=case.q_left_integrals[n : n + 1],
                    resistance=case.resistance,
                    sigma=sigma,
                    q_ref=a_ref,
                )
                diff = pred - truth[n + 1 : n + 2]
                teacher_ssq += float(diff.square().sum().cpu())
                teacher_cnt += int(diff.numel())
            per_case_teacher.append(math.sqrt(teacher_ssq / max(teacher_cnt, 1)))

            roll_out = rollout_field(
                model,
                truth[0:1],
                case.interval_images,
                case.scalars,
                case.fixed_channels,
                coords,
                dt=dt,
                query_chunk=int(query_chunk),
                output_parameterization=output_parameterization,
                steps=n_intervals,
                interface_x=case.interface_x,
                jump_scale=case.jump_scale,
                closure_geom=case.geom,
                q_left_integrals=case.q_left_integrals,
                resistance=case.resistance,
                sigma=sigma,
                q_ref=a_ref,
                return_storage_projection=want_storage,
            )                                                  # (Nt, Nx, Ny)
            if want_storage:
                roll, _flux, _corr, implied_flux_gap = roll_out
                gap = float(implied_flux_gap.abs().max().cpu()) / (float(a_ref) + 1e-12)
                if math.isfinite(gap):
                    hard_storage_error = max(hard_storage_error, gap)
            else:
                roll = roll_out
            finite = bool(torch.isfinite(roll).all().item())
            nonfinite += int((~torch.isfinite(roll)).sum().cpu())
            roll_safe = torch.nan_to_num(roll, nan=0.0, posinf=0.0, neginf=0.0)
            diff = roll_safe - truth
            e_k = diff.reshape(diff.shape[0], -1).square().mean(dim=1).sqrt()
            per_case_rollout.append(float(e_k.square().mean().sqrt().cpu()))
            per_case_max_step.append(float(e_k[1:].max().cpu()) if n_intervals >= 1 else 0.0)
            per_case_final.append(float(e_k[-1].cpu()))
            per_case_rc.append(float(case.resistance))
            per_case_ix.append(float(case.interface_x.reshape(-1)[0].cpu()))

            T_K = roll_safe * float(sigma) + float(mu)
            max_abs_T_K = max(max_abs_T_K, float(T_K.abs().max().cpu()) if finite else float("inf"))

            dT_pred = (roll_safe[:, face] - roll_safe[:, face + 1]) * float(sigma)
            dT_true = (truth[:, face] - truth[:, face + 1]) * float(sigma)
            jump_err_ssq += float((dT_pred - dT_true).square().sum().cpu())
            jump_true_ssq += float(dT_true.square().sum().cpu())
            jump_cnt += int(dT_true.numel())

    model.train(was_training)

    roll_arr = np.asarray(per_case_rollout, dtype=np.float64)
    teacher_arr = np.asarray(per_case_teacher, dtype=np.float64)
    rc_arr = np.asarray(per_case_rc, dtype=np.float64)
    ix_arr = np.asarray(per_case_ix, dtype=np.float64)
    denom = float(sigma_dT_train) + 1e-12
    node_jump_rmse_K = math.sqrt(jump_err_ssq / max(jump_cnt, 1))
    val_gnrmse = float(roll_arr.mean()) if roll_arr.size else float("nan")
    val_gnrmse_teacher = float(teacher_arr.mean()) if teacher_arr.size else float("nan")

    out: dict[str, float] = {
        "val_gnrmse": val_gnrmse,
        "val_gnrmse_rollout": val_gnrmse,
        "val_gnrmse_teacher": val_gnrmse_teacher,
        "val_rmse_K": val_gnrmse * float(sigma),
        "val_rmse_K_teacher": val_gnrmse_teacher * float(sigma),
        "node_jump_rmse_K": node_jump_rmse_K,
        "E_model": node_jump_rmse_K / denom,
        "E_zero": math.sqrt(jump_true_ssq / max(jump_cnt, 1)) / denom,
        "rollout_max_stepwise_gnrmse": (
            float(np.mean(per_case_max_step)) if per_case_max_step else float("nan")
        ),
        "rollout_final_gnrmse": (
            float(np.mean(per_case_final)) if per_case_final else float("nan")
        ),
        "rollout_teacher_ratio": (
            val_gnrmse / (val_gnrmse_teacher + 1e-12)
            if math.isfinite(val_gnrmse_teacher) else float("inf")
        ),
        "rollout_max_abs_T_K": max_abs_T_K,
        "rollout_nonfinite": float(nonfinite),
        "hard_storage_active": float(want_storage),
        "hard_global_storage_error": float(hard_storage_error),
    }
    for tag, arr in (("Rc", rc_arr), ("ix", ix_arr)):
        if arr.size >= 3:
            q1, q2 = np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0])
            strata = {
                "low": arr <= q1,
                "mid": (arr > q1) & (arr <= q2),
                "high": arr > q2,
            }
            for name, mask in strata.items():
                out[f"gnrmse_{tag}_{name}"] = (
                    float(roll_arr[mask].mean()) if mask.any() else float("nan")
                )
    return out


def _through_origin_slope(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    """Least-squares slope of ``pred`` on ``true`` forced through the origin.

    ``s = sum(true * pred) / sum(true^2)`` (plan Section 6). Differencing paired
    probes removes the shared 300 K baseline before this call, so a through-origin
    fit is the response of the prediction to the true increment. Returns ``nan``
    when the true signal is degenerate (``sum(true^2)`` ~ 0), which the gate
    treats as ineligible.
    """
    t = np.asarray(true_vals, dtype=np.float64).reshape(-1)
    p = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    denom = float(np.dot(t, t))
    if not math.isfinite(denom) or denom <= 1e-30:
        return float("nan")
    return float(np.dot(t, p) / denom)


def interfaces_one_step_response_slopes(
    model: InterfaceCViT,
    probe_pairs: list,
    *,
    dt: float,
    coords: torch.Tensor,
    query_chunk: int,
    output_parameterization: str = "absolute",
    a_ref: float,
    sigma: float,
    device: torch.device,
) -> dict[str, float]:
    """Paired-amplitude forcing / jump response slopes (anti-collapse probe).

    Each element of ``probe_pairs`` is a ``(case_lo, case_hi)`` tuple built from
    the SAME descriptor (identical IC, ``interface_x``, ``R_c``) but two forcing
    amplitudes. Both one-step predictions are conditioned on the SHARED initial
    state ``T_0`` so the 300 K baseline cancels in the difference (plan Section
    6):

    - ``forcing_response_slope``: through-origin slope of
      ``dT_pred = pred_hi - pred_lo`` against ``dT_true = truth_hi[1] -
      truth_lo[1]`` over the whole field.
    - ``jump_response_slope``: through-origin slope of the interface node jump
      ``T[face] - T[face+1]`` of the same paired differences, plus the jump RMSE
      (K), correlation, and correct-sign fraction.

    A constant-field (collapsed) operator yields ``dT_pred ~ 0`` and hence a
    slope near zero, which the gate rejects.
    """
    from src.operators.one_step import predict_one_step_field

    was_training = model.training
    model.eval()

    field_true: list[np.ndarray] = []
    field_pred: list[np.ndarray] = []
    jump_true: list[np.ndarray] = []
    jump_pred: list[np.ndarray] = []

    with torch.no_grad():
        for case_lo, case_hi in probe_pairs:
            if int(case_lo.n_intervals) < 1 or int(case_hi.n_intervals) < 1:
                continue
            face = int(case_lo.geom.face_idx.reshape(-1)[0].item())
            state0 = case_lo.truth_states[0:1].to(device)

            def _predict(case):
                return predict_one_step_field(
                    model,
                    state0,
                    case.interval_images[0:1],
                    case.scalars,
                    case.fixed_channels,
                    coords,
                    dt=dt,
                    query_chunk=int(query_chunk),
                    output_parameterization=output_parameterization,
                    interface_x=case.interface_x,
                    jump_scale=case.jump_scale,
                    closure_geom=case.geom,
                    q_left_integral=case.q_left_integrals[0:1],
                    resistance=case.resistance,
                    sigma=sigma,
                    q_ref=a_ref,
                )

            pred_lo = _predict(case_lo)[0]
            pred_hi = _predict(case_hi)[0]
            dpred = (pred_hi - pred_lo) * float(sigma)
            dtrue = (
                case_hi.truth_states[1].to(device)
                - case_lo.truth_states[1].to(device)
            ) * float(sigma)

            field_true.append(dtrue.reshape(-1).cpu().numpy())
            field_pred.append(dpred.reshape(-1).cpu().numpy())
            jump_true.append(
                (dtrue[face] - dtrue[face + 1]).reshape(-1).cpu().numpy()
            )
            jump_pred.append(
                (dpred[face] - dpred[face + 1]).reshape(-1).cpu().numpy()
            )

    model.train(was_training)

    if not field_true:
        return {
            "forcing_response_slope": float("nan"),
            "jump_response_slope": float("nan"),
            "jump_rmse_K": float("nan"),
            "jump_correlation": float("nan"),
            "jump_sign_fraction": float("nan"),
        }

    ft = np.concatenate(field_true)
    fp = np.concatenate(field_pred)
    jt = np.concatenate(jump_true)
    jp = np.concatenate(jump_pred)

    jump_rmse_K = float(np.sqrt(np.mean((jp - jt) ** 2)))
    if jt.size >= 2 and np.std(jt) > 1e-12 and np.std(jp) > 1e-12:
        jump_corr = float(np.corrcoef(jt, jp)[0, 1])
    else:
        jump_corr = float("nan")
    active = np.abs(jt) > 1e-9
    if active.any():
        sign_frac = float(np.mean(np.sign(jt[active]) == np.sign(jp[active])))
    else:
        sign_frac = float("nan")

    return {
        "forcing_response_slope": _through_origin_slope(ft, fp),
        "jump_response_slope": _through_origin_slope(jt, jp),
        "jump_rmse_K": jump_rmse_K,
        "jump_correlation": jump_corr,
        "jump_sign_fraction": sign_frac,
    }


def anti_collapse_eligible(
    metrics: dict[str, float],
    *,
    forcing_slope_min: float,
    jump_slope_min: float,
    max_abs_T_K: float | None = None,
    hard_storage_tol: float | None = None,
    require_finite: bool = True,
) -> bool:
    """Anti-collapse checkpoint eligibility gate (plan Section 6).

    A checkpoint is eligible only if it shows live transport and stays
    numerically sane. ``None`` thresholds skip that condition (e.g.
    ``max_abs_temperature_K: null``). The response-slope minima reject the 300 K
    constant-field collapse; the interface-flux / contact-law diagnostics are NOT
    gate conditions for the initial screen.
    """
    slope_f = float(metrics.get("forcing_response_slope", float("nan")))
    slope_j = float(metrics.get("jump_response_slope", float("nan")))
    if not (math.isfinite(slope_f) and slope_f >= float(forcing_slope_min)):
        return False
    if not (math.isfinite(slope_j) and slope_j >= float(jump_slope_min)):
        return False
    if require_finite:
        if float(metrics.get("rollout_nonfinite", 0.0)) > 0.0:
            return False
        if not math.isfinite(float(metrics.get("val_gnrmse", float("nan")))):
            return False
    if max_abs_T_K is not None:
        max_T = float(metrics.get("rollout_max_abs_T_K", float("inf")))
        if not (math.isfinite(max_T) and max_T <= float(max_abs_T_K)):
            return False
    if hard_storage_tol is not None and bool(metrics.get("hard_storage_active", False)):
        err = float(metrics.get("hard_global_storage_error", float("inf")))
        if not (math.isfinite(err) and err <= float(hard_storage_tol)):
            return False
    return True


def _resolve_interfaces_one_step_config(config: dict) -> dict:
    resolved = copy.deepcopy(config)
    os_cfg = resolved["training"]["pino"].setdefault("one_step", {})
    output_parameterization = str(os_cfg.get("output_parameterization", "rate"))
    if output_parameterization not in {"rate", "increment", "deviation", "absolute"}:
        raise ValueError(
            "training.pino.one_step.output_parameterization must be 'rate', "
            "'increment', 'deviation', or 'absolute'."
        )
    conservation = str(os_cfg.get("conservation", "learned"))
    allowed = {
        "none",
        "learned",
        "left_energy_closure",
        "two_sided_energy_closure",
        "conservative_storage_projection",
    }
    if conservation not in allowed:
        raise ValueError(
            "training.pino.one_step.conservation must be one of "
            f"{sorted(allowed)}; got {conservation!r}."
        )
    interface_cfg = resolved["model"].setdefault("interface_cvit", {})
    interface_cfg["jump_enrichment"] = conservation != "none"
    interface_cfg["jump_flux_mode"] = (
        "learned" if conservation == "none" else conservation
    )
    os_cfg["output_parameterization"] = output_parameterization
    os_cfg["conservation"] = conservation
    if bool(interface_cfg.get("interface_aligned_domains", False)) and (
        output_parameterization != "rate" or conservation != "learned"
    ):
        raise ValueError(
            "interface_aligned_domains requires output_parameterization='rate' "
            "and conservation='learned'"
        )
    return resolved


def run_one_seed_interfaces_one_step_pino(
    config: dict, seed: int, run_dir: Path
) -> dict[str, Any]:
    """Physics-only one-step Markov training of an :class:`InterfaceCViT`.

    Maps ``(T_n, interval forcing, R_c, interface_x) -> T_{n+1}`` under a
    variational or preconditioned-defect CN objective (never using ``T_{n+1}``
    as a supervised label). Each update draws ``n_cases`` distinct online
    descriptors from a continuously-refreshed :class:`OneStepStatePool`, solves
    the FV manifold once per descriptor, conditions on a detached on-manifold
    ``T_n``, and averages the per-case gradients (single ``zero_grad`` /
    ``optimizer.step``; each ``loss_c`` pre-divided by ``n_cases``). Validation
    is :func:`validate_interfaces_one_step_gnrmse` with free-rollout gnRMSE as
    the primary scalar gate (plan Sections 4-6).
    """
    from src.operators.one_step import (
        OneStepStatePool,
        predict_one_step_field,
        prepare_interfaces_one_step_case,
    )
    from src.physics.one_step_objective import one_step_objective

    config = _resolve_interfaces_one_step_config(config)
    set_seed(seed)
    run_dir = Path(run_dir)
    resume_requested = bool(
        (((config.get("training", {}) or {}).get("pino", {}) or {})
         .get("one_step", {}) or {}).get("resume", False)
    )
    resume_path = run_dir / "resume_state.pt"
    resuming = resume_requested and resume_path.exists()
    stale_artifacts = [
        run_dir / name for name in (
            "train_metrics.csv", "cvit_best_global.pt", "cvit_last.pt",
            "final_metrics.json",
        )
        if (run_dir / name).exists()
    ]
    if not resuming and stale_artifacts:
        raise FileExistsError(
            "InterfaceCViT one-step runs require a fresh experiment name; found "
            + ", ".join(str(path) for path in stale_artifacts)
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    x_grid_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_grid_np = np.asarray(data["y_grid"], dtype=np.float64)
    t_grid_np = np.asarray(data["t_grid"], dtype=np.float64)
    Nx, Ny = int(x_grid_np.shape[0]), int(y_grid_np.shape[0])
    Nq = Nx * Ny
    t_final = float(t_grid_np[-1])

    pino = config["training"]["pino"]
    os_cfg = dict(pino.get("one_step", {}) or {})
    fcfg = pino.get("forcing", {}) or {}
    a_ref = float(fcfg.get("a_ref") if fcfg.get("a_ref") is not None else A_REF_FLUX)
    if a_ref <= 0.0:
        raise ValueError(f"training.pino.forcing.a_ref must be > 0; got {a_ref}.")
    ny_img = int(fcfg.get("ny_img") or 96)
    nt_img = int(fcfg.get("nt_img") or 256)

    ramp_cfg = fcfg.get("ramp_seconds", None)
    if ramp_cfg is not None:
        t_ramp = float(ramp_cfg)
    else:
        t_ramp = load_ramp_seconds(config["data"]["t_grid_path"])
        if t_ramp is None:
            _dt0 = load_solver_dt(config["data"]["t_grid_path"])
            t_ramp = default_ramp_seconds(_dt0 if _dt0 is not None else t_final / 100.0)

    dt_cfg = pino.get("dt", None)
    if dt_cfg is not None:
        dt = float(dt_cfg)
    else:
        dt = load_solver_dt(config["data"]["t_grid_path"])
        if dt is None:
            dt = t_final / max(len(t_grid_np) - 1, 1)

    # ``t_final`` is reconstructed from the float32-saved t_grid, so it carries
    # ~1e-8 quantization noise (e.g. 0.3 -> 0.30000001192). FVSolver2D treats a
    # non-None dt as explicit and enforces exact divisibility (fv_solver_2d.py),
    # so that noise makes an otherwise-integer step count fail. Snap t_final back
    # onto the dt grid when the ratio is within float32 noise of an integer;
    # leave a genuinely non-dividing dt untouched so the solver raises its
    # precise error.
    _ratio = t_final / dt
    _n_steps = round(_ratio)
    if _n_steps >= 1 and abs(_ratio - _n_steps) <= 1.0e-5 * _n_steps:
        t_final = _n_steps * dt

    # One-step temporal contract (plan Section 0): one model step == one CN step.
    step_stride = int(os_cfg.get("step_stride", 1))
    if step_stride != 1:
        raise ValueError(
            "training.pino.one_step.step_stride must be 1 for this experiment "
            "(one model step == one FV CN step); a coarse saved-output interval "
            f"is not a single CN interval. Got {step_stride}."
        )
    output_parameterization = str(os_cfg["output_parameterization"])

    kind = str(os_cfg.get("objective", "variational"))
    if kind not in ("variational", "defect"):
        raise ValueError(
            f"training.pino.one_step.objective must be 'variational' or 'defect'; "
            f"got {kind!r}."
        )
    defect_sweeps = int(os_cfg.get("defect_sweeps", 1))
    defect_omega = float(os_cfg.get("defect_omega", 2.0 / 3.0))
    n_cases = int(os_cfg.get("n_cases", 8))
    pool_size = int(os_cfg.get("state_pool_size", 128))
    replace_per_update = int(os_cfg.get("state_pool_replace_per_update", 2))
    max_uses_per_case = int(os_cfg.get("max_uses_per_case", 8))
    fixed_cfg = dict(os_cfg.get("fixed_simulation", {}) or {})
    fixed_enabled = bool(fixed_cfg.get("enabled", False))
    fixed_num_simulations = int(fixed_cfg.get("num_simulations", 1))
    if fixed_num_simulations < 1:
        raise ValueError("fixed_simulation.num_simulations must be >= 1")
    fixed_num_states = int(fixed_cfg.get("num_states", 16))
    normalize_fixed_defect = bool(
        fixed_cfg.get("normalize_per_state_defect", False)
    )
    normalization_floor_fraction = float(
        fixed_cfg.get("normalization_floor_fraction", 0.01)
    )
    if normalization_floor_fraction <= 0.0:
        raise ValueError(
            "fixed_simulation.normalization_floor_fraction must be > 0"
        )
    if normalize_fixed_defect and (not fixed_enabled or kind != "defect"):
        raise ValueError(
            "fixed-simulation per-state normalization requires "
            "fixed_simulation.enabled=true and objective=defect"
        )
    init_checkpoint_cfg = os_cfg.get("init_from_checkpoint", None)
    init_checkpoint_path = (
        None if init_checkpoint_cfg is None
        else Path(str(init_checkpoint_cfg)).expanduser().resolve()
    )
    time_mixture = dict(os_cfg.get("time_mixture", {}) or {
        "uniform": 0.5, "active_forcing": 0.25, "high_change": 0.25
    })
    updates = int(os_cfg.get("updates", int(config["training"].get("epochs", 8000))))
    validate_every = int(os_cfg.get("validate_every", config["training"].get("validate_every", 500)))
    fast_validation_cases = int(os_cfg.get("fast_validation_cases", 16))
    chunk_r = int(pino.get("chunk_r", 0)) or Nq
    grad_clip_cfg = os_cfg.get("grad_clip", config["training"].get("grad_clip"))
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    if grad_clip is not None and grad_clip <= 0.0:
        raise ValueError("training.pino.one_step.grad_clip must be null or > 0")

    # Anti-collapse checkpoint gate (plan Section 6). The response-slope minima
    # reject the 300 K constant-field collapse; ``None`` thresholds skip that
    # condition. ``keep_top_k_eligible`` best eligible checkpoints are retained
    # for the deterministic full-development selection procedure.
    ac_cfg = dict(os_cfg.get("anti_collapse", {}) or {})
    rollout_cfg = dict(os_cfg.get("rollout", {}) or {})
    forcing_slope_min = float(ac_cfg.get("forcing_response_slope_min", 0.25))
    jump_slope_min = float(ac_cfg.get("jump_response_slope_min", 0.10))
    hard_storage_tol = ac_cfg.get("hard_global_storage_tol", 1.0e-6)
    hard_storage_tol = None if hard_storage_tol is None else float(hard_storage_tol)
    max_abs_T_K_gate = rollout_cfg.get("max_abs_temperature_K", None)
    max_abs_T_K_gate = None if max_abs_T_K_gate is None else float(max_abs_T_K_gate)
    require_finite_gate = bool(rollout_cfg.get("require_finite", True))
    keep_top_k_eligible = int(os_cfg.get("keep_top_k_eligible", 5))
    n_probe_pairs = int(os_cfg.get("probe_pairs", 8))
    probe_amp_ratio = float(os_cfg.get("probe_amp_ratio", 0.5))

    model = build_cvit(
        config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final,
        variant="interfaces",
    ).to(device)
    # Encoder guard (plan Section 10): the one-step run MUST use the current 2D
    # forcing-image InterfaceCViT, never the legacy waveform-token encoder. Check
    # the model + forcing-encoder class, the (Ny_img, Nt_img) image grid, and the
    # derived patch-token count (384 for the 96x256/patch-8 production encoder).
    forcing_encoder = getattr(model, "forcing_encoder", None)
    if not isinstance(model, InterfaceCViT) or not isinstance(
        forcing_encoder, CViTEncoder
    ):
        raise ValueError(
            "one-step run requires the 2D forcing-image InterfaceCViT with a "
            f"CViTEncoder forcing branch; got model={type(model).__name__} "
            f"forcing_encoder={type(forcing_encoder).__name__}."
        )
    if tuple(getattr(model, "forcing_grid_size", ())) != (ny_img, nt_img):
        raise ValueError(
            "one-step run requires the 2D forcing-image InterfaceCViT with "
            f"forcing_grid_size=({ny_img}, {nt_img}); got "
            f"{getattr(model, 'forcing_grid_size', None)}."
        )
    forcing_patch_size = int(getattr(model, "forcing_patch_size", 0))
    expected_forcing_tokens = (
        (ny_img // forcing_patch_size) * (nt_img // forcing_patch_size)
        if forcing_patch_size > 0 else -1
    )
    num_forcing_tokens = int(getattr(model, "num_forcing_tokens", -1))
    if num_forcing_tokens != expected_forcing_tokens:
        raise ValueError(
            "one-step forcing-image patch-token count mismatch: expected "
            f"{expected_forcing_tokens} for forcing_grid_size=({ny_img}, "
            f"{nt_img}) patch={forcing_patch_size}; got {num_forcing_tokens}."
        )
    init_checkpoint = None
    init_fixed_state = None
    if init_checkpoint_path is not None:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(init_checkpoint_path)
        init_checkpoint = torch.load(
            init_checkpoint_path, map_location="cpu", weights_only=False
        )
        for name, current in (("mu_global", mu), ("sigma_global", sigma)):
            source = float(init_checkpoint[name])
            if not math.isclose(source, float(current), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(
                    f"one-step warm start {name} mismatch: source={source}, "
                    f"current={current}"
                )
        model.load_state_dict(init_checkpoint["model_state"], strict=True)
        init_fixed_state = init_checkpoint.get("fixed_simulation_state")
        if fixed_enabled and init_fixed_state is None:
            raise ValueError(
                "fixed-simulation warm start requires a checkpoint containing "
                "fixed_simulation_state"
            )
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    t_right_tilde = float((T_RIGHT - mu) / (sigma + 1e-8))

    online_spec = problem_from_config(config)
    gxm, gym = np.meshgrid(x_grid_np, y_grid_np, indexing="ij")
    online_grids = {"X": gxm, "Y": gym, "x_grid": x_grid_np, "y_grid": y_grid_np}
    online_time_cfg = {
        "dt": dt, "t_final": t_final, "b": 1.0, "T_right": float(T_RIGHT),
        "t_on": float(fcfg.get("t_on", 0.0)),
        "t_off": float(fcfg.get("t_off", 0.2)),
        "phase": float(fcfg.get("phase", 0.0)),
        "tukey_alpha": float(fcfg.get("tukey_alpha", 0.5)),
    }

    base_kwargs = {
        "a": float(x_grid_np[0]), "b": float(x_grid_np[-1]),
        "c": float(y_grid_np[0]), "d": float(y_grid_np[-1]),
        "Nx": Nx, "Ny": Ny, "lam_target": 0.8,
        "t_final": t_final, "flux_f": 0.0, "flux_A": 0.0,
        "t_on": online_time_cfg["t_on"], "t_off": online_time_cfg["t_off"],
        "phase": online_time_cfg["phase"], "tukey_alpha": online_time_cfg["tukey_alpha"],
        "dt": dt, "y_grid": y_grid_np, "ramp_seconds": t_ramp,
    }

    def solver_factory(params: dict):
        return online_spec.configure_solver(params, base_kwargs)

    def build_case_from_params(key: int, params: dict):
        # Deterministic case build from a stored descriptor. This is the resume
        # ``rebuild_fn`` (plan Section 9): FV trajectories are regenerated from
        # the descriptor, never serialized into the checkpoint.
        return prepare_interfaces_one_step_case(
            key, params, solver_factory,
            x_grid=x_grid_np, y_grid=y_grid_np, mu=mu, sigma=sigma,
            t_ramp=t_ramp, q_ref=a_ref, k_left=K_LEFT, k_right=K_RIGHT,
            ny_image=ny_img, image_time_points=nt_img,
            right_value=t_right_tilde, device=device,
        )

    def make_case(key: int, sampler_rng: np.random.Generator):
        params = online_spec.sample_online_params(
            sampler_rng, 1, online_grids, online_time_cfg,
        )[0]
        return build_case_from_params(key, params)

    pool_rng = np.random.default_rng(seed + 101)
    fixed_cases = []
    fixed_case = None
    fixed_intervals: list[int] = []
    pool = None
    if fixed_enabled:
        if init_fixed_state is None:
            fixed_params = online_spec.sample_online_params(
                pool_rng,
                fixed_num_simulations,
                online_grids,
                online_time_cfg,
            )
            fixed_cases = [
                build_case_from_params(key, params)
                for key, params in enumerate(fixed_params)
            ]
            fixed_case = fixed_cases[0]
            fixed_intervals = _fixed_interval_indices(
                fixed_case.n_intervals, fixed_num_states
            )
        else:
            saved_params = init_fixed_state.get(
                "case_params", [init_fixed_state["params"]]
            )
            if len(saved_params) != fixed_num_simulations:
                raise ValueError(
                    "fixed-simulation warm start descriptor count mismatch: "
                    f"checkpoint={len(saved_params)}, "
                    f"config={fixed_num_simulations}"
                )
            fixed_cases = [
                build_case_from_params(key, params)
                for key, params in enumerate(saved_params)
            ]
            fixed_case = fixed_cases[0]
            fixed_intervals = [int(i) for i in init_fixed_state["intervals"]]
            if len(fixed_intervals) != fixed_num_states:
                raise ValueError(
                    "fixed-simulation warm start interval count mismatch: "
                    f"checkpoint={len(fixed_intervals)}, config={fixed_num_states}"
                )
        if any(
            max(fixed_intervals) >= case.n_intervals for case in fixed_cases
        ):
            raise ValueError(
                "fixed-simulation interval selection exceeds a case trajectory"
            )
    else:
        pool = OneStepStatePool(
            lambda key: make_case(key, pool_rng),
            pool_size=pool_size, n_cases=n_cases,
            replace_per_update=replace_per_update,
            max_uses_per_case=max_uses_per_case,
            time_mixture=time_mixture, rng=np.random.default_rng(seed + 202),
        )

    # Frozen validation cases (never sampled for training; plan Section 6).
    val_rng = np.random.default_rng(seed + 777)
    val_cases = [make_case(-1 - i, val_rng) for i in range(fast_validation_cases)]

    def make_probe_pair(key: int, sampler_rng: np.random.Generator):
        """Paired-amplitude probe: two cases sharing one descriptor / IC.

        The base descriptor is drawn once, then the sin forcing amplitude ``A``
        is scaled by ``probe_amp_ratio`` for the low probe and left at the
        sampled value for the high probe. Everything else (IC, ``interface_x``,
        ``R_c``, spatial profile) is identical, so differencing the two one-step
        predictions isolates the forcing / jump response (plan Section 6).
        """
        base = online_spec.sample_online_params(
            sampler_rng, 1, online_grids, online_time_cfg,
        )[0]
        a_high = float(base["temporal_params"]["A"])
        pair = []
        for tag, amp in (("lo", probe_amp_ratio * a_high), ("hi", a_high)):
            params = copy.deepcopy(base)
            params["temporal_params"] = dict(params["temporal_params"])
            params["temporal_params"]["A"] = float(amp)
            pair.append(prepare_interfaces_one_step_case(
                (key * 2) + (0 if tag == "lo" else 1), params, solver_factory,
                x_grid=x_grid_np, y_grid=y_grid_np, mu=mu, sigma=sigma,
                t_ramp=t_ramp, q_ref=a_ref, k_left=K_LEFT, k_right=K_RIGHT,
                ny_image=ny_img, image_time_points=nt_img,
                right_value=t_right_tilde, device=device,
            ))
        return (pair[0], pair[1])

    # Frozen anti-collapse probe pairs (never sampled for training).
    probe_rng = np.random.default_rng(seed + 909)
    probe_pairs = [
        make_probe_pair(-1000 - i, probe_rng) for i in range(n_probe_pairs)
    ]

    x_grid_t = torch.as_tensor(x_grid_np, dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(y_grid_np, dtype=torch.float32, device=device)
    gx, gy = torch.meshgrid(x_grid_t, y_grid_t, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)

    sim_batch_ids = np.asarray(data["train_ids"])
    sp_path = Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy"
    sim_params = np.load(str(sp_path), allow_pickle=True)
    sigma_dT_train = _compute_train_jump_scale(data, sim_params, sim_batch_ids)

    metrics_path = run_dir / "train_metrics.csv"
    fieldnames = [
        "update", "completed_updates", "loss", "energy", "defect_rms",
        "val_gnrmse", "val_gnrmse_rollout", "val_gnrmse_teacher",
        "val_rmse_K", "val_rmse_K_teacher",
        "node_jump_rmse_K", "E_model", "E_zero",
        "gnrmse_Rc_low", "gnrmse_Rc_mid", "gnrmse_Rc_high",
        "gnrmse_ix_low", "gnrmse_ix_mid", "gnrmse_ix_high",
        "rollout_max_stepwise_gnrmse", "rollout_final_gnrmse",
        "rollout_teacher_ratio", "rollout_max_abs_T_K", "rollout_nonfinite",
        "hard_storage_active", "hard_global_storage_error",
        "forcing_response_slope", "jump_response_slope",
        "jump_rmse_K", "jump_correlation", "jump_sign_fraction",
        "anti_collapse_eligible",
        "pool_distinct_keys", "pool_size", "pool_mean_uses", "pool_max_age",
        "fixed_train_rmse_K", "fixed_train_transition_rel",
        "fixed_train_max_rmse_K", "fixed_train_max_transition_rel",
    ]
    if not resuming:
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writeheader()
    fixed_metrics_path = run_dir / "fixed_state_metrics.csv"
    fixed_metric_fields = [
        "update", "simulation", "interval", "t_n", "rmse_K", "transition_rel"
    ]
    if fixed_enabled and not resuming:
        with open(fixed_metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fixed_metric_fields).writeheader()

    print(
        f"[pino-interfaces-one-step] seed={seed} device={device} "
        f"updates={updates} validate_every={validate_every} | grid={Nx}x{Ny} "
        f"t_final={t_final:.4f} dt={dt:.4g} t_ramp={t_ramp:.4g} "
        f"forcing_image={ny_img}x{nt_img} a_ref={a_ref} | objective={kind} "
        + (
            f"fixed_simulation=on(simulations={len(fixed_cases)},"
            f"states={len(fixed_intervals)},"
            f"intervals={fixed_intervals})"
            if fixed_enabled else
            f"n_cases={n_cases} pool={pool_size}(replace={replace_per_update},"
            f"max_uses={max_uses_per_case})"
        )
        + f" | k=({K_LEFT},{K_RIGHT}) "
        f"sigma_dT_train={sigma_dT_train:.4g} jump_flux={getattr(model, 'jump_flux_mode', 'learned')}",
        flush=True,
    )

    image_spec = {
        "representation": "space_time_image",
        "version": 1,
        "forcing_schema_version": FORCING_SCHEMA_VERSION,
        "axis_order": "channel_y_time",
        "dtype": "float32",
        "ny_img": ny_img, "nt_img": nt_img,
        "patch_size": forcing_patch_size,
        "include_endpoints": True,
        "y_min": float(y_grid_np[0]), "y_max": float(y_grid_np[-1]),
        "t_min": 0.0, "t_final": float(t_final),
        "sign_convention": "positive_inward_left_flux",
        "normalization": "fixed_division",
        "clipping": False,
        "a_ref": a_ref, "fv_dt": float(dt),
        "ramp": {
            "type": "cubic_smoothstep",
            "version": RAMP_SCHEMA_VERSION,
            "duration": float(t_ramp),
        },
        "spatial_grid_size": [Nx, Ny],
        "one_step": {
            "objective": kind,
            "step_stride": step_stride,
            "output_parameterization": output_parameterization,
            "conservation": os_cfg["conservation"],
            "fixed_simulation": {
                "enabled": fixed_enabled,
                "num_simulations": len(fixed_cases),
                "num_states": len(fixed_intervals),
                "intervals": fixed_intervals,
                "normalize_per_state_defect": normalize_fixed_defect,
                "normalization_floor_fraction": normalization_floor_fraction,
            },
            "init_from_checkpoint": (
                None if init_checkpoint_path is None else str(init_checkpoint_path)
            ),
        },
        # Concrete encoder class + architecture version (plan Section 10) so the
        # SLURM preflight and any resumed checkpoint can reject a run that was not
        # produced by the current 2D forcing-image InterfaceCViT.
        "architecture": {
            "version": ONE_STEP_ARCH_VERSION,
            "model_class": type(model).__name__,
            "forcing_encoder_class": type(model.forcing_encoder).__name__,
            "forcing_grid_size": [ny_img, nt_img],
            "forcing_patch_size": forcing_patch_size,
            "num_forcing_tokens": num_forcing_tokens,
            "interface_aligned_domains": bool(
                getattr(model, "interface_aligned_domains", False)
            ),
        },
    }
    with open(run_dir / "architecture.json", "w") as f:
        json.dump(image_spec["architecture"], f, indent=2, sort_keys=True)

    best_eligible = float("inf")
    best_ungated = float("inf")
    best_fixed = float("inf")
    final_fixed_metrics: dict[str, float] = {}
    fixed_case_weights: torch.Tensor | None = None
    fixed_reference_energy: torch.Tensor | None = None
    any_eligible = False
    eligible_dir = run_dir / "eligible"
    eligible_top: list[dict[str, Any]] = []  # top-k eligible: val_gnrmse ascending
    completed_updates = 0
    start_update = 0

    def save_resume_state(next_update: int) -> None:
        # Exact-resume snapshot (plan Section 9). Heavy FV trajectories are NOT
        # serialized; only descriptors are stored and regenerated on resume via
        # ``build_case_from_params``. Everything needed to continue the online
        # stream identically is captured: model/optimizer/scheduler, the pool
        # (selection rng + per-case descriptors/uses/ages/intervals), the pool
        # factory rng, and the global torch/numpy/python rng states.
        _atomic_torch_save(
            {
                "completed_updates": completed_updates,
                "next_update": int(next_update),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "pool_state": None if pool is None else pool.state_dict(),
                "fixed_simulation_state": (
                    {
                        "params": fixed_case.params,
                        "case_params": [case.params for case in fixed_cases],
                        "intervals": fixed_intervals,
                    }
                    if fixed_case is not None else None
                ),
                "pool_factory_rng": pool_rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "best_eligible": best_eligible,
                "best_ungated": best_ungated,
                "best_fixed": best_fixed,
                "final_fixed_metrics": final_fixed_metrics,
                "fixed_case_weights": (
                    None if fixed_case_weights is None
                    else fixed_case_weights.detach().cpu()
                ),
                "fixed_reference_energy": (
                    None if fixed_reference_energy is None
                    else fixed_reference_energy.detach().cpu()
                ),
                "any_eligible": any_eligible,
                "eligible_top": eligible_top,
                "objective": kind,
            },
            resume_path,
        )

    if resuming:
        rs = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["model_state"])
        optimizer.load_state_dict(rs["optimizer_state"])
        scheduler.load_state_dict(rs["scheduler_state"])
        if fixed_enabled:
            fixed_state = rs.get("fixed_simulation_state")
            if fixed_state is None:
                raise ValueError(
                    "resume checkpoint does not contain fixed-simulation state"
                )
            saved_params = fixed_state.get(
                "case_params", [fixed_state["params"]]
            )
            if len(saved_params) != fixed_num_simulations:
                raise ValueError(
                    "resume fixed-simulation descriptor count mismatch: "
                    f"checkpoint={len(saved_params)}, "
                    f"config={fixed_num_simulations}"
                )
            fixed_cases = [
                build_case_from_params(key, params)
                for key, params in enumerate(saved_params)
            ]
            fixed_case = fixed_cases[0]
            fixed_intervals = [int(i) for i in fixed_state["intervals"]]
        else:
            pool.load_state_dict(rs["pool_state"], build_case_from_params)
        pool_rng.bit_generator.state = rs["pool_factory_rng"]
        torch.set_rng_state(rs["torch_rng"])
        np.random.set_state(rs["numpy_rng"])
        random.setstate(rs["python_rng"])
        best_eligible = float(rs["best_eligible"])
        best_ungated = float(rs["best_ungated"])
        best_fixed = float(rs.get("best_fixed", float("inf")))
        final_fixed_metrics = dict(rs.get("final_fixed_metrics", {}) or {})
        if rs.get("fixed_case_weights") is not None:
            fixed_case_weights = rs["fixed_case_weights"].to(device)
            fixed_reference_energy = rs["fixed_reference_energy"].to(device)
        any_eligible = bool(rs["any_eligible"])
        eligible_top = list(rs["eligible_top"])
        completed_updates = int(rs["completed_updates"])
        start_update = int(rs["next_update"])
        print(
            f"[pino-interfaces-one-step] RESUME start_update={start_update} "
            f"completed_updates={completed_updates}",
            flush=True,
        )

    if normalize_fixed_defect and fixed_case_weights is None:
        model.eval()
        steps = torch.as_tensor(fixed_intervals, dtype=torch.long)
        batch_size = int(steps.numel())
        references = []
        with torch.no_grad():
            for case in fixed_cases:
                state_n = case.truth_states.index_select(0, steps).to(device)
                prediction = predict_one_step_field(
                    model,
                    state_n,
                    case.interval_images.index_select(
                        0, steps.to(case.interval_images.device)
                    ),
                    case.scalars.expand(batch_size, -1),
                    case.fixed_channels.expand(batch_size, -1, -1, -1),
                    coords,
                    dt=dt,
                    query_chunk=chunk_r,
                    output_parameterization=output_parameterization,
                    interface_x=case.interface_x.expand(batch_size),
                    jump_scale=case.jump_scale.expand(batch_size),
                    closure_geom=case.geom,
                    q_left_integral=case.q_left_integrals.index_select(
                        0, steps.to(case.q_left_integrals.device)
                    ),
                    resistance=case.resistance,
                    sigma=sigma,
                    q_ref=a_ref,
                )
                cn_batch = {
                    **case.cn,
                    "forcing": case.cn["forcing"].index_select(
                        0, steps.to(case.cn["forcing"].device)
                    ),
                }
                _, reference = one_step_objective(
                    kind,
                    prediction,
                    state_n,
                    cn_batch,
                    right_value=t_right_tilde,
                    defect_sweeps=defect_sweeps,
                    defect_omega=defect_omega,
                )
                references.append(reference["energy_per_case"].to(device))
        fixed_reference_energy = torch.stack(references)
        median = fixed_reference_energy.median().clamp_min(1.0e-20)
        scale = fixed_reference_energy.clamp_min(
            median * normalization_floor_fraction
        )
        inverse = scale.reciprocal()
        fixed_case_weights = inverse / inverse.mean()
        weight_rows = [
            {
                "simulation": simulation,
                "interval": int(step),
                "reference_energy": float(
                    fixed_reference_energy[simulation, index].cpu()
                ),
                "weight": float(fixed_case_weights[simulation, index].cpu()),
            }
            for simulation in range(len(fixed_cases))
            for index, step in enumerate(fixed_intervals)
        ]
        with open(run_dir / "fixed_state_weights.json", "w") as f:
            json.dump(weight_rows, f, indent=2)
        print(
            "[pino-interfaces-one-step] frozen per-state defect weights "
            f"range=({fixed_case_weights.min().item():.4g}, "
            f"{fixed_case_weights.max().item():.4g})",
            flush=True,
        )

    for update in range(start_update, updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if fixed_enabled:
            steps = torch.as_tensor(fixed_intervals, dtype=torch.long)
            batch_size = int(steps.numel())
            loss_sum = 0.0
            energy_sum = 0.0
            defect_sum = 0.0
            for simulation, case in enumerate(fixed_cases):
                state_n = case.truth_states.index_select(0, steps).to(device)
                pred = predict_one_step_field(
                    model, state_n,
                    case.interval_images.index_select(
                        0, steps.to(case.interval_images.device)
                    ),
                    case.scalars.expand(batch_size, -1),
                    case.fixed_channels.expand(batch_size, -1, -1, -1),
                    coords,
                    dt=dt, query_chunk=chunk_r,
                    output_parameterization=output_parameterization,
                    interface_x=case.interface_x.expand(batch_size),
                    jump_scale=case.jump_scale.expand(batch_size),
                    closure_geom=case.geom,
                    q_left_integral=case.q_left_integrals.index_select(
                        0, steps.to(case.q_left_integrals.device)
                    ),
                    resistance=case.resistance, sigma=sigma, q_ref=a_ref,
                )
                cn_batch = {
                    **case.cn,
                    "forcing": case.cn["forcing"].index_select(
                        0, steps.to(case.cn["forcing"].device)
                    ),
                }
                weights = (
                    None if fixed_case_weights is None
                    else fixed_case_weights[simulation]
                )
                loss, m = one_step_objective(
                    kind, pred, state_n.detach(), cn_batch,
                    right_value=t_right_tilde,
                    defect_sweeps=defect_sweeps, defect_omega=defect_omega,
                    case_weights=weights,
                )
                (loss / len(fixed_cases)).backward()
                loss_sum += float(loss.detach().cpu()) / len(fixed_cases)
                energy_sum += float(m["energy"].cpu()) / len(fixed_cases)
                defect_sum += float(m["defect_rms"].cpu()) / len(fixed_cases)
            diversity = {
                "pool_distinct_keys": float(len(fixed_cases)),
                "pool_size": float(len(fixed_cases)),
                "pool_mean_uses": float(completed_updates + 1),
                "pool_max_age": float(completed_updates + 1),
            }
        else:
            pool.refresh()
            selection = pool.select()
            loss_sum = 0.0
            energy_sum = 0.0
            defect_sum = 0.0
            for case, step in selection:
                state_n = case.truth_states[step : step + 1].to(device)
                pred = predict_one_step_field(
                    model, state_n,
                    case.interval_images[step : step + 1],
                    case.scalars, case.fixed_channels, coords,
                    dt=dt, query_chunk=chunk_r,
                    output_parameterization=output_parameterization,
                    interface_x=case.interface_x, jump_scale=case.jump_scale,
                    closure_geom=case.geom,
                    q_left_integral=case.q_left_integrals[step : step + 1],
                    resistance=case.resistance, sigma=sigma, q_ref=a_ref,
                )
                cn_step = {
                    **case.cn, "forcing": case.cn["forcing"][step : step + 1]
                }
                loss_c, m = one_step_objective(
                    kind, pred, state_n.detach(), cn_step,
                    right_value=t_right_tilde,
                    defect_sweeps=defect_sweeps, defect_omega=defect_omega,
                )
                (loss_c / n_cases).backward()
                loss_sum += float(loss_c.detach().cpu()) / n_cases
                energy_sum += float(m["energy"].cpu()) / n_cases
                defect_sum += float(m["defect_rms"].cpu()) / n_cases
            diversity = pool.diversity_stats()
        if not math.isfinite(loss_sum):
            raise FloatingPointError(
                f"Non-finite one-step objective at update {update}"
            )
        gradients_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise FloatingPointError(
                f"Non-finite one-step gradient at update {update}"
            )
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        completed_updates += 1

        row: dict[str, Any] = {
            "update": update,
            "completed_updates": completed_updates,
            "loss": loss_sum,
            "energy": energy_sum,
            "defect_rms": defect_sum,
            **diversity,
        }

        do_val = (update % validate_every == 0) or (update == updates - 1)
        if do_val:
            val = validate_interfaces_one_step_gnrmse(
                model, val_cases,
                mu=mu, sigma=sigma, sigma_dT_train=sigma_dT_train,
                dt=dt, coords=coords, query_chunk=chunk_r,
                output_parameterization=output_parameterization,
                a_ref=a_ref, right_value=t_right_tilde, device=device,
            )
            slopes = interfaces_one_step_response_slopes(
                model, probe_pairs,
                dt=dt, coords=coords, query_chunk=chunk_r,
                output_parameterization=output_parameterization,
                a_ref=a_ref, sigma=sigma, device=device,
            )
            val.update(slopes)
            fixed_rows: list[dict[str, float]] = []
            if fixed_enabled:
                per_simulation_metrics = []
                for simulation, case in enumerate(fixed_cases):
                    metrics, simulation_rows = _evaluate_interfaces_fixed_states(
                        model,
                        case,
                        fixed_intervals,
                        sigma=sigma,
                        dt=dt,
                        coords=coords,
                        query_chunk=chunk_r,
                        output_parameterization=output_parameterization,
                        a_ref=a_ref,
                        device=device,
                    )
                    per_simulation_metrics.append(metrics)
                    fixed_rows.extend(
                        {"simulation": simulation, **simulation_row}
                        for simulation_row in simulation_rows
                    )
                error_ssq = sum(m["_error_ssq"] for m in per_simulation_metrics)
                transition_ssq = sum(
                    m["_transition_ssq"] for m in per_simulation_metrics
                )
                value_count = sum(
                    m["_value_count"] for m in per_simulation_metrics
                )
                fixed_metrics = {
                    "fixed_train_rmse_K": (
                        math.sqrt(error_ssq / max(value_count, 1.0)) * float(sigma)
                    ),
                    "fixed_train_transition_rel": math.sqrt(
                        error_ssq / max(transition_ssq, 1.0e-20)
                    ),
                    "fixed_train_max_rmse_K": max(
                        m["fixed_train_max_rmse_K"]
                        for m in per_simulation_metrics
                    ),
                    "fixed_train_max_transition_rel": max(
                        m["fixed_train_max_transition_rel"]
                        for m in per_simulation_metrics
                    ),
                }
                row.update(fixed_metrics)
                final_fixed_metrics = dict(fixed_metrics)
                with open(fixed_metrics_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fixed_metric_fields)
                    for fixed_row in fixed_rows:
                        writer.writerow({"update": update, **fixed_row})
            eligible = anti_collapse_eligible(
                val,
                forcing_slope_min=forcing_slope_min,
                jump_slope_min=jump_slope_min,
                max_abs_T_K=max_abs_T_K_gate,
                hard_storage_tol=hard_storage_tol,
                require_finite=require_finite_gate,
            )
            for key in (
                "val_gnrmse", "val_gnrmse_rollout", "val_gnrmse_teacher",
                "val_rmse_K", "val_rmse_K_teacher",
                "node_jump_rmse_K", "E_model", "E_zero",
                "gnrmse_Rc_low", "gnrmse_Rc_mid", "gnrmse_Rc_high",
                "gnrmse_ix_low", "gnrmse_ix_mid", "gnrmse_ix_high",
                "rollout_max_stepwise_gnrmse", "rollout_final_gnrmse",
                "rollout_teacher_ratio", "rollout_max_abs_T_K",
                "rollout_nonfinite", "hard_storage_active", "hard_global_storage_error",
                "forcing_response_slope", "jump_response_slope",
                "jump_rmse_K", "jump_correlation", "jump_sign_fraction",
            ):
                row[key] = val.get(key, "")
            row["anti_collapse_eligible"] = int(eligible)

            ckpt = {
                "model_state": model.state_dict(),
                "mu_global": mu, "sigma_global": sigma,
                "config": config, "update": update,
                "best_val_gnrmse": val["val_gnrmse"],
                "anti_collapse_eligible": bool(eligible),
                "forcing_response_slope": val["forcing_response_slope"],
                "jump_response_slope": val["jump_response_slope"],
                "hard_global_storage_error": val["hard_global_storage_error"],
                "sigma_dT_train": sigma_dT_train,
                "interface_forcing": copy.deepcopy(image_spec),
                "fixed_simulation_metrics": copy.deepcopy(final_fixed_metrics),
                "fixed_simulation_state": (
                    {
                        "params": fixed_case.params,
                        "case_params": [case.params for case in fixed_cases],
                        "intervals": fixed_intervals,
                    }
                    if fixed_case is not None else None
                ),
                "source_checkpoint": (
                    None if init_checkpoint_path is None else str(init_checkpoint_path)
                ),
                "fixed_case_weights": (
                    None if fixed_case_weights is None
                    else fixed_case_weights.detach().cpu()
                ),
            }
            v = float(val["val_gnrmse"])
            saved_best = False
            if fixed_enabled:
                fixed_score = float(final_fixed_metrics["fixed_train_transition_rel"])
                if math.isfinite(fixed_score) and fixed_score < best_fixed:
                    best_fixed = fixed_score
                    _atomic_torch_save(ckpt, run_dir / "cvit_best_fixed.pt")
            if eligible and math.isfinite(v):
                # Eligible checkpoints take over cvit_best_global.pt; the top-k
                # eligible pool feeds the deterministic selection (plan Section 6).
                if v < best_eligible:
                    best_eligible = v
                    _atomic_torch_save(ckpt, run_dir / "cvit_best_global.pt")
                    saved_best = True
                any_eligible = True
                if keep_top_k_eligible > 0:
                    eligible_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = eligible_dir / f"ckpt_u{update:07d}.pt"
                    _atomic_torch_save(ckpt, ckpt_path)
                    eligible_top.append(
                        {"val_gnrmse": v, "update": int(update),
                         "path": str(ckpt_path),
                         "forcing_response_slope": float(val["forcing_response_slope"]),
                         "jump_response_slope": float(val["jump_response_slope"]),
                         "jump_rmse_K": float(val["jump_rmse_K"]),
                         "rollout_final_gnrmse": float(val["rollout_final_gnrmse"])}
                    )
                    eligible_top.sort(key=lambda r: r["val_gnrmse"])
                    for stale in eligible_top[keep_top_k_eligible:]:
                        Path(stale["path"]).unlink(missing_ok=True)
                    eligible_top = eligible_top[:keep_top_k_eligible]
            elif (not any_eligible) and math.isfinite(v) and v < best_ungated:
                # Fallback while nothing has passed the gate: keep the best raw
                # gnRMSE so a resumable checkpoint always exists.
                best_ungated = v
                _atomic_torch_save(ckpt, run_dir / "cvit_best_global.pt")
                saved_best = True
            print(
                f"Update {update}: loss={loss_sum:.6e} energy={energy_sum:.6e} "
                f"defect_rms={defect_sum:.6e} | "
                f"val_gnrmse={val['val_gnrmse'] * 100:.4f}% "
                f"teacher={val['val_gnrmse_teacher'] * 100:.4f}% "
                f"final={val['rollout_final_gnrmse'] * 100:.4f}% "
                f"fslope={val['forcing_response_slope']:.3f} "
                f"jslope={val['jump_response_slope']:.3f} "
                f"storage={val['hard_global_storage_error']:.2e} "
                f"elig={int(eligible)}"
                + (
                    f" fixed_rmse={final_fixed_metrics['fixed_train_rmse_K']:.6f}K"
                    f" fixed_transition={100.0 * final_fixed_metrics['fixed_train_transition_rel']:.4f}%"
                    if fixed_enabled else ""
                )
                + ("  [best]" if saved_best else ""),
                flush=True,
            )
        else:
            print(
                f"Update {update}: loss={loss_sum:.6e} energy={energy_sum:.6e} "
                f"defect_rms={defect_sum:.6e}",
                flush=True,
            )

        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)

        if do_val:
            save_resume_state(update + 1)

    save_resume_state(updates)
    _atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "mu_global": mu, "sigma_global": sigma,
            "config": config, "update": updates - 1,
            "sigma_dT_train": sigma_dT_train,
            "interface_forcing": copy.deepcopy(image_spec),
            "fixed_simulation_metrics": copy.deepcopy(final_fixed_metrics),
            "fixed_simulation_state": (
                {
                    "params": fixed_case.params,
                    "case_params": [case.params for case in fixed_cases],
                    "intervals": fixed_intervals,
                }
                if fixed_case is not None else None
            ),
            "source_checkpoint": (
                None if init_checkpoint_path is None else str(init_checkpoint_path)
            ),
            "fixed_case_weights": (
                None if fixed_case_weights is None
                else fixed_case_weights.detach().cpu()
            ),
        },
        run_dir / "cvit_last.pt",
    )

    summary = {
        "seed": seed,
        "best_val_gnrmse": best_eligible if any_eligible else best_ungated,
        "best_eligible_val_gnrmse": best_eligible if any_eligible else None,
        "best_ungated_val_gnrmse": best_ungated,
        "any_eligible": bool(any_eligible),
        "eligible_top_k": eligible_top,
        "sigma_dT_train": sigma_dT_train,
        "updates": updates,
        "objective": kind,
        "fixed_simulation": fixed_enabled,
        "fixed_num_simulations": len(fixed_cases),
        "fixed_intervals": fixed_intervals,
        "fixed_state_defect_normalization": normalize_fixed_defect,
        "source_checkpoint": (
            None if init_checkpoint_path is None else str(init_checkpoint_path)
        ),
        "best_fixed_transition_rel": (
            best_fixed if fixed_enabled and math.isfinite(best_fixed) else None
        ),
        **final_fixed_metrics,
    }
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _decode_transition_in_chunks(
    model: ForcingTransitionCViT,
    encoding: TransitionEncoding,
    source: torch.Tensor,
    coords: torch.Tensor,
    source_time: torch.Tensor,
    lead_time: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0 or coords.shape[1] <= chunk_size:
        return model.decode(
            encoding, source, coords, source_time, lead_time,
        )
    return torch.cat(
        [
            model.decode(
                encoding,
                source,
                coords[:, start:start + chunk_size],
                source_time,
                lead_time,
            )
            for start in range(0, coords.shape[1], chunk_size)
        ],
        dim=1,
    )


def _transition_source_batch(
    trajectories: np.ndarray,
    records: list[dict[str, int]],
    mu: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    source = np.stack(
        [
            np.asarray(
                trajectories[record["sim_id"], record["source_index"]],
                dtype=np.float32,
            )
            for record in records
        ]
    )
    source = (source - np.float32(mu)) / np.float32(sigma)
    return torch.from_numpy(source).unsqueeze(1).to(device)


def _transition_batch_loss(
    model: ForcingTransitionCViT,
    records: list[dict[str, int]],
    *,
    sim_params: np.ndarray,
    trajectories: np.ndarray,
    t_grid: np.ndarray,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    y_img: np.ndarray,
    nt_img: int,
    a_ref: float,
    t_ramp: float,
    t_final: float,
    mu: float,
    sigma: float,
    n_queries: int,
    query_chunk: int,
    base_seed: int,
    epoch: int,
    device: torch.device,
) -> torch.Tensor:
    if not records:
        raise ValueError("transition training batch must not be empty")
    if n_queries <= 0:
        raise ValueError("transition n_queries must be positive")
    source_time_np = np.asarray(
        [t_grid[record["source_index"]] for record in records],
        dtype=np.float32,
    )
    target_time_np = np.asarray(
        [t_grid[record["target_index"]] for record in records],
        dtype=np.float32,
    )
    lead_time_np = target_time_np - source_time_np
    params = [dict(sim_params[record["sim_id"]]) for record in records]
    source = _transition_source_batch(
        trajectories, records, mu, sigma, device,
    )
    forcing = build_forcing_transition_image(
        params,
        y_img,
        source_time_np,
        lead_time_np,
        nt_img,
        a_ref,
        device,
        t_ramp,
        t_final,
    )

    nx, ny = int(x_grid.numel()), int(y_grid.numel())
    ix = np.empty((len(records), n_queries), dtype=np.int64)
    iy = np.empty_like(ix)
    for batch_index, record in enumerate(records):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    int(base_seed),
                    int(epoch),
                    int(record["sim_id"]),
                    3,
                ]
            )
        )
        ix[batch_index] = rng.integers(0, nx, size=n_queries)
        iy[batch_index] = rng.integers(0, ny, size=n_queries)
    ix_t = torch.from_numpy(ix).to(device)
    iy_t = torch.from_numpy(iy).to(device)
    coords = torch.stack([x_grid[ix_t], y_grid[iy_t]], dim=-1)

    truth = np.empty((len(records), n_queries), dtype=np.float32)
    for batch_index, record in enumerate(records):
        truth[batch_index] = np.asarray(
            trajectories[
                record["sim_id"],
                record["target_index"],
                ix[batch_index],
                iy[batch_index],
            ],
            dtype=np.float32,
        )
    truth = (truth - np.float32(mu)) / np.float32(sigma)
    source_time = torch.from_numpy(source_time_np).to(device).unsqueeze(-1)
    lead_time = torch.from_numpy(lead_time_np).to(device).unsqueeze(-1)
    encoding = model.encode(forcing, source)
    prediction = _decode_transition_in_chunks(
        model,
        encoding,
        source,
        coords,
        source_time,
        lead_time,
        query_chunk,
    )
    target = torch.from_numpy(truth).to(device).unsqueeze(-1)
    return (prediction - target).square().mean()


def transition_pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    sigma_global: float,
    signal_floor_fraction: float,
) -> dict[str, float | bool | None]:
    if (
        prediction.shape != target.shape
        or prediction.shape != source.shape
        or prediction.ndim < 1
    ):
        raise ValueError("prediction, target, and source must have matching shapes")
    if sigma_global <= 0.0 or signal_floor_fraction < 0.0:
        raise ValueError("metric scales must be positive/non-negative")
    error_rms = torch.sqrt((prediction - target).square().mean())
    increment_rms = torch.sqrt((target - source).square().mean())
    gated = bool(
        (increment_rms >= float(signal_floor_fraction)).detach().cpu().item()
    )
    relative = None
    copy_skill = None
    if gated:
        error_l2 = torch.linalg.vector_norm(prediction - target)
        increment_l2 = torch.linalg.vector_norm(target - source)
        relative = float((error_l2 / increment_l2).detach().cpu())
        copy_skill = 1.0 - relative ** 2
    return {
        "target_rmse_K": float(error_rms.detach().cpu()) * float(sigma_global),
        "target_gnrmse": float(error_rms.detach().cpu()),
        "copy_rmse_K": float(increment_rms.detach().cpu()) * float(sigma_global),
        "copy_gnrmse": float(increment_rms.detach().cpu()),
        "increment_signal_rms_K": (
            float(increment_rms.detach().cpu()) * float(sigma_global)
        ),
        "increment_gated": gated,
        "increment_rel_l2": relative,
        "copy_skill": copy_skill,
    }


def _mean_finite(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def aggregate_transition_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty transition validation set")
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"s{int(row['source_bin'])}_l{int(row['lead_bin'])}"
        cells.setdefault(key, []).append(row)
    cell_gnrmse = {
        key: float(np.mean([float(row["target_gnrmse"]) for row in group]))
        for key, group in cells.items()
    }
    gated_count = sum(bool(row["increment_gated"]) for row in rows)
    out: dict[str, Any] = {
        "num_pairs": len(rows),
        "target_rmse_K": float(
            np.mean([float(row["target_rmse_K"]) for row in rows])
        ),
        "target_gnrmse": float(
            np.mean([float(row["target_gnrmse"]) for row in rows])
        ),
        "copy_rmse_K": _mean_finite(rows, "copy_rmse_K"),
        "copy_gnrmse": _mean_finite(rows, "copy_gnrmse"),
        "macro_cell_gnrmse": float(np.mean(list(cell_gnrmse.values()))),
        "increment_rel_l2": _mean_finite(rows, "increment_rel_l2"),
        "copy_skill": _mean_finite(rows, "copy_skill"),
        "low_frequency_error_fraction": _mean_finite(
            rows, "low_frequency_error_fraction",
        ),
        "gated_coverage": gated_count / len(rows),
        "cell_gnrmse": cell_gnrmse,
    }
    for field in (
        "source_bin",
        "lead_bin",
        "target_bin",
        "source_lead_cell",
        "ic_family",
        "forcing_amplitude_tercile",
    ):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[field]), []).append(row)
        out[f"by_{field}"] = {
            key: {
                "count": len(group),
                "target_gnrmse": float(
                    np.mean(
                        [float(item["target_gnrmse"]) for item in group]
                    )
                ),
                "copy_gnrmse": _mean_finite(group, "copy_gnrmse"),
                "increment_rel_l2": _mean_finite(
                    group, "increment_rel_l2",
                ),
                "copy_skill": _mean_finite(group, "copy_skill"),
                "low_frequency_error_fraction": _mean_finite(
                    group, "low_frequency_error_fraction",
                ),
                "gated_coverage": (
                    sum(bool(item["increment_gated"]) for item in group)
                    / len(group)
                ),
            }
            for key, group in groups.items()
        }
    return out


def _write_transition_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate_forcing_transition(
    model: ForcingTransitionCViT,
    data: dict[str, Any],
    sim_params: np.ndarray,
    records: list[dict[str, int]],
    *,
    y_img: np.ndarray,
    nt_img: int,
    a_ref: float,
    t_ramp: float,
    device: torch.device,
    sim_batch: int,
    query_chunk: int,
    signal_floor_fraction: float,
    pair_csv_path: Path | None = None,
) -> dict[str, Any]:
    if sim_batch <= 0 or query_chunk < 0:
        raise ValueError("validation batch must be positive and chunk non-negative")
    x_grid = torch.as_tensor(
        data["x_grid"], dtype=torch.float32, device=device,
    )
    y_grid = torch.as_tensor(
        data["y_grid"], dtype=torch.float32, device=device,
    )
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack(
        [gx.reshape(-1), gy.reshape(-1)], dim=-1,
    ).unsqueeze(0)
    trajectories = data["trajectories"]
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    mu = float(data["mu_global"])
    sigma = float(data["sigma_global"])
    t_final = float(t_grid[-1])
    rows: list[dict[str, Any]] = []

    amplitudes: list[float] = []
    for record in records:
        source_time = float(t_grid[record["source_index"]])
        lead_time = float(
            t_grid[record["target_index"]] - source_time
        )
        image = build_forcing_transition_image(
            [dict(sim_params[record["sim_id"]])],
            y_img,
            [source_time],
            [lead_time],
            nt_img,
            a_ref,
            torch.device("cpu"),
            t_ramp,
            t_final,
        )
        amplitudes.append(float(image[:, 0].abs().max()) * float(a_ref))
    if len(amplitudes) >= 3:
        amp_low, amp_high = np.quantile(amplitudes, [1.0 / 3.0, 2.0 / 3.0])
    else:
        amp_low = amp_high = float(np.median(amplitudes))

    for start in range(0, len(records), sim_batch):
        batch_records = records[start:start + sim_batch]
        params = [
            dict(sim_params[record["sim_id"]]) for record in batch_records
        ]
        source_time_np = np.asarray(
            [t_grid[record["source_index"]] for record in batch_records],
            dtype=np.float32,
        )
        target_time_np = np.asarray(
            [t_grid[record["target_index"]] for record in batch_records],
            dtype=np.float32,
        )
        lead_time_np = target_time_np - source_time_np
        source = _transition_source_batch(
            trajectories, batch_records, mu, sigma, device,
        )
        forcing = build_forcing_transition_image(
            params,
            y_img,
            source_time_np,
            lead_time_np,
            nt_img,
            a_ref,
            device,
            t_ramp,
            t_final,
        )
        batch_size = len(batch_records)
        coords = mesh.expand(batch_size, -1, -1)
        source_time = torch.from_numpy(source_time_np).to(device).unsqueeze(-1)
        lead_time = torch.from_numpy(lead_time_np).to(device).unsqueeze(-1)
        encoding = model.encode(forcing, source)
        prediction = _decode_transition_in_chunks(
            model,
            encoding,
            source,
            coords,
            source_time,
            lead_time,
            query_chunk,
        )[..., 0]
        target_np = np.stack(
            [
                np.asarray(
                    trajectories[
                        record["sim_id"], record["target_index"]
                    ],
                    dtype=np.float32,
                )
                for record in batch_records
            ]
        )
        target = torch.from_numpy(
            (target_np - np.float32(mu)) / np.float32(sigma)
        ).to(device).reshape(batch_size, -1)
        source_queries = source[:, 0].reshape(batch_size, -1)
        for batch_index, record in enumerate(batch_records):
            metrics = transition_pair_metrics(
                prediction[batch_index],
                target[batch_index],
                source_queries[batch_index],
                sigma_global=sigma,
                signal_floor_fraction=signal_floor_fraction,
            )
            error_field = (
                prediction[batch_index] - target[batch_index]
            ).reshape(int(x_grid.numel()), int(y_grid.numel()))
            spectrum = torch.fft.fft2(error_field, norm="ortho")
            total_spectral_error = spectrum.abs().square().sum()
            low_spectral_error = spectrum[:4, :4].abs().square().sum()
            metrics["low_frequency_error_fraction"] = float(
                (
                    low_spectral_error
                    / total_spectral_error.clamp_min(
                        torch.finfo(total_spectral_error.dtype).tiny
                    )
                ).detach().cpu()
            )
            amplitude = amplitudes[start + batch_index]
            if amplitude <= amp_low:
                tercile = "low"
            elif amplitude <= amp_high:
                tercile = "mid"
            else:
                tercile = "high"
            rows.append({
                **record,
                "source_time": float(source_time_np[batch_index]),
                "lead_time": float(lead_time_np[batch_index]),
                "target_time": float(target_time_np[batch_index]),
                "source_lead_cell": (
                    f"s{record['source_bin']}_l{record['lead_bin']}"
                ),
                "ic_family": str(
                    sim_params[record["sim_id"]].get("ic_family", "")
                ),
                "forcing_amplitude": amplitude,
                "forcing_amplitude_tercile": tercile,
                **metrics,
            })
    if pair_csv_path is not None:
        _write_transition_rows(Path(pair_csv_path), rows)
    result = aggregate_transition_metrics(rows)
    result["manifest_hash"] = transition_manifest_hash(records)
    return result


def _ic_balanced_subset(
    ids: np.ndarray,
    sim_params: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(ids, dtype=int)
    if limit <= 0 or limit >= len(ids):
        return ids.copy()
    groups: dict[str, list[int]] = {}
    for sim_id in ids:
        family = str(sim_params[int(sim_id)].get("ic_family", ""))
        groups.setdefault(family, []).append(int(sim_id))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 4]))
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    names = sorted(groups)
    position = 0
    while len(selected) < limit:
        progressed = False
        for name in names:
            if position < len(groups[name]) and len(selected) < limit:
                selected.append(groups[name][position])
                progressed = True
        if not progressed:
            break
        position += 1
    return np.asarray(selected, dtype=int)


def _transition_checkpoint_compatibility(
    config: dict,
    *,
    nx: int,
    ny: int,
    ny_img: int,
    nt_img: int,
    a_ref: float,
    t_final: float,
    schedule: TransitionPairSchedule,
) -> dict[str, Any]:
    c = {
        **config["model"]["cvit"],
        **(config["model"].get("forcing_transition_cvit", {}) or {}),
    }
    default_time_freq = (
        c.get("fourier_freq_t")
        if c.get("fourier_freq_t") is not None
        else c.get("fourier_freq", 1.0)
    )
    return {
        "variant": "forcing_transition",
        "forcing_channels": 3,
        "forcing_resolution": [int(ny_img), int(nt_img)],
        "source_resolution": [int(nx), int(ny)],
        "forcing_patch_size": int(c.get("forcing_patch_size", 8)),
        "source_patch_size": int(
            c.get("source_patch_size", c.get("patch_size", 10))
        ),
        "a_ref": float(a_ref),
        "t_final": float(t_final),
        "fourier_freq_source": float(
            c.get("fourier_freq_source", default_time_freq)
        ),
        "fourier_freq_lead": float(
            c.get("fourier_freq_lead", default_time_freq)
        ),
        "source_edges": schedule.source_edges.tolist(),
        "lead_edges": schedule.lead_edges.tolist(),
        "target_edges": schedule.target_edges.tolist(),
        "coordinate_tolerance": float(c.get("coord_tolerance", 1.0e-6)),
        "align_corners": True,
        "query_time_conditioning": bool(
            c.get("query_time_conditioning", True)
        ),
        "film_time_conditioning": bool(
            c.get("film_time_conditioning", True)
        ),
        "film_initialization": {
            "distribution": "normal",
            "mean": 0.0,
            "std": float(c.get("film_init_std", 1.0e-3)),
            "bias": 0.0,
            "convention": "(1+gamma)*h+beta",
        },
        "residual_gate": "(1-x)*(lead_time/t_final)",
    }


@torch.no_grad()
def transition_counterfactual_diagnostics(
    model: ForcingTransitionCViT,
    data: dict[str, Any],
    sim_params: np.ndarray,
    records: list[dict[str, int]],
    *,
    y_img: np.ndarray,
    nt_img: int,
    a_ref: float,
    t_ramp: float,
    device: torch.device,
    query_chunk: int,
) -> dict[str, float | None]:
    if len({record["sim_id"] for record in records}) < 2:
        return {
            "source_state_swap_rms_K": None,
            "in_window_forcing_swap_rms_K": None,
            "source_departure_ratio": None,
            "forcing_response_ratio": None,
        }
    first = records[0]
    second = next(
        record for record in records
        if record["sim_id"] != first["sim_id"]
    )
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    source_time = float(t_grid[first["source_index"]])
    lead_time = float(t_grid[first["target_index"]] - source_time)
    source_records = [
        first,
        {
            **first,
            "sim_id": int(second["sim_id"]),
            "source_index": int(first["source_index"]),
            "target_index": int(first["target_index"]),
        },
    ]
    source = _transition_source_batch(
        data["trajectories"],
        source_records,
        float(data["mu_global"]),
        float(data["sigma_global"]),
        device,
    )
    forcing = build_forcing_transition_image(
        [dict(sim_params[first["sim_id"]])] * 2,
        y_img,
        [source_time, source_time],
        [lead_time, lead_time],
        nt_img,
        a_ref,
        device,
        t_ramp,
        float(t_grid[-1]),
    )
    x_grid = torch.as_tensor(
        data["x_grid"], dtype=torch.float32, device=device,
    )
    y_grid = torch.as_tensor(
        data["y_grid"], dtype=torch.float32, device=device,
    )
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    coords = torch.stack(
        [gx.reshape(-1), gy.reshape(-1)], dim=-1,
    ).unsqueeze(0).expand(2, -1, -1)
    source_times = torch.full((2, 1), source_time, device=device)
    lead_times = torch.full((2, 1), lead_time, device=device)
    source_prediction = _decode_transition_in_chunks(
        model,
        model.encode(forcing, source),
        source,
        coords,
        source_times,
        lead_times,
        query_chunk,
    )
    source_sensitivity = torch.sqrt(
        (source_prediction[0] - source_prediction[1]).square().mean()
    )

    forcing_swap = build_forcing_transition_image(
        [
            dict(sim_params[first["sim_id"]]),
            dict(sim_params[second["sim_id"]]),
        ],
        y_img,
        [source_time, source_time],
        [lead_time, lead_time],
        nt_img,
        a_ref,
        device,
        t_ramp,
        float(t_grid[-1]),
    )
    repeated_source = source[0:1].expand(2, -1, -1, -1)
    forcing_prediction = _decode_transition_in_chunks(
        model,
        model.encode(forcing_swap, repeated_source),
        repeated_source,
        coords,
        source_times,
        lead_times,
        query_chunk,
    )
    forcing_sensitivity = torch.sqrt(
        (forcing_prediction[0] - forcing_prediction[1]).square().mean()
    )
    target_np = np.asarray(
        data["trajectories"][first["sim_id"], first["target_index"]],
        dtype=np.float32,
    )
    target = torch.from_numpy(
        (target_np - np.float32(data["mu_global"]))
        / np.float32(data["sigma_global"])
    ).to(device).reshape(-1, 1)
    source_query = source[0, 0].reshape(-1, 1)
    true_increment = torch.sqrt((target - source_query).square().mean())
    predicted_departure = torch.sqrt(
        (source_prediction[0] - source_query).square().mean()
    )
    denominator = float(true_increment.cpu())
    sigma = float(data["sigma_global"])
    return {
        "source_state_swap_rms_K": float(source_sensitivity.cpu()) * sigma,
        "in_window_forcing_swap_rms_K": (
            float(forcing_sensitivity.cpu()) * sigma
        ),
        "source_departure_ratio": (
            None if denominator == 0.0
            else float(predicted_departure.cpu()) / denominator
        ),
        "forcing_response_ratio": (
            None if denominator == 0.0
            else float(forcing_sensitivity.cpu()) / denominator
        ),
    }


def _transition_parameter_group_norm(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
    *,
    gradients: bool,
) -> float:
    squares = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefixes):
            continue
        value = parameter.grad if gradients else parameter
        if value is not None:
            squares.append(value.detach().double().square().sum())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def _transition_initial_parameters(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    }


def _transition_parameter_displacement(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> tuple[float, float]:
    current = dict(model.named_parameters())
    displacement_sq = 0.0
    initial_sq = 0.0
    for name, reference in initial.items():
        delta = current[name].detach().cpu().double() - reference.double()
        displacement_sq += float(delta.square().sum())
        initial_sq += float(reference.double().square().sum())
    displacement = math.sqrt(displacement_sq)
    relative = displacement / max(math.sqrt(initial_sq), np.finfo(float).tiny)
    return displacement, relative


def _build_transition_forcing_probe(
    problem,
    *,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dt: float,
    t_final: float,
    t_ramp: float,
    time_cfg: dict[str, float],
    mu: float,
    sigma: float,
) -> dict[str, Any]:
    common = {
        "t_on": float(time_cfg["t_on"]),
        "t_off": float(time_cfg["t_off"]),
        "phase": float(time_cfg["phase"]),
        "tukey_alpha": float(time_cfg["tukey_alpha"]),
        "rectified": True,
    }
    cases = [
        ("zero", 0.0, 4.0),
        ("low", 75.0, 2.0),
        ("high", 250.0, 12.0),
    ]
    solver_steps = int(round(float(t_final) / float(dt)))
    solver_t_final = solver_steps * float(dt)
    target_steps = sorted({
        max(1, min(int(round(value / dt)), solver_steps))
        for value in (0.05, 0.10, 0.20, t_final)
    })
    source_K = np.full(
        (len(x_grid), len(y_grid)), 300.0, dtype=np.float64,
    )
    base_kwargs = {
        "a": float(x_grid[0]),
        "b": float(x_grid[-1]),
        "c": float(y_grid[0]),
        "d": float(y_grid[-1]),
        "Nx": int(len(x_grid)),
        "Ny": int(len(y_grid)),
        "lam_target": 0.5,
        "t_final": solver_t_final,
        "flux_f": 0.0,
        "t_on": float(time_cfg["t_on"]),
        "t_off": float(time_cfg["t_off"]),
        "phase": float(time_cfg["phase"]),
        "dt": float(dt),
        "tukey_alpha": float(time_cfg["tukey_alpha"]),
        "y_grid": np.asarray(y_grid, dtype=np.float64),
        "ramp_seconds": float(t_ramp),
    }
    records = []
    for name, amplitude, frequency in cases:
        params = {
            "temporal_family": "sin",
            "temporal_params": {
                "A": amplitude,
                "f": frequency,
                **common,
            },
            "spatial_family": "uniform",
            "spatial_params": {},
        }
        solver = problem.configure_solver(params, base_kwargs)
        _, _, _, trajectory_K = solver.solve(
            T0=source_K, store_trajectory=True,
        )
        truth = (
            np.asarray(trajectory_K, dtype=np.float64)[target_steps]
            - float(mu)
        ) / float(sigma)
        records.append({
            "name": name,
            "params": params,
            "truth": truth,
        })
    return {
        "source": (source_K - float(mu)) / float(sigma),
        "target_steps": target_steps,
        "lead_times": np.asarray(target_steps, dtype=np.float32) * float(dt),
        "cases": records,
    }


@torch.no_grad()
def transition_designed_forcing_diagnostics(
    model: ForcingTransitionCViT,
    probe: dict[str, Any],
    *,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    y_img: np.ndarray,
    nt_img: int,
    a_ref: float,
    t_ramp: float,
    t_final: float,
    sigma: float,
    device: torch.device,
    query_chunk: int,
) -> dict[str, float]:
    cases = list(probe["cases"])
    leads = np.asarray(probe["lead_times"], dtype=np.float32)
    num_cases = len(cases)
    num_leads = len(leads)
    params = [
        case["params"]
        for case in cases
        for _ in range(num_leads)
    ]
    lead_values = np.tile(leads, num_cases)
    source_times = np.zeros_like(lead_values)
    forcing = build_forcing_transition_image(
        params,
        y_img,
        source_times,
        lead_values,
        nt_img,
        a_ref,
        device,
        t_ramp,
        t_final,
    )
    zero_forcing = forcing.clone()
    zero_forcing[:, 0] = 0.0
    shuffled_forcing = (
        forcing.reshape(
            num_cases, num_leads, *forcing.shape[1:],
        )
        .roll(shifts=1, dims=0)
        .reshape_as(forcing)
    )
    source = torch.as_tensor(
        probe["source"], dtype=torch.float32, device=device,
    ).unsqueeze(0).unsqueeze(0)
    source = source.expand(
        num_cases * num_leads, -1, -1, -1,
    ).contiguous()
    gx, gy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    mesh = torch.stack(
        [gx.reshape(-1), gy.reshape(-1)], dim=-1,
    ).unsqueeze(0).expand(num_cases * num_leads, -1, -1)
    source_time = torch.zeros(
        num_cases * num_leads, 1, device=device,
    )
    lead_time = torch.from_numpy(lead_values).to(device).unsqueeze(-1)

    def predict(image: torch.Tensor) -> torch.Tensor:
        prediction = _decode_transition_in_chunks(
            model,
            model.encode(image, source),
            source,
            mesh,
            source_time,
            lead_time,
            query_chunk,
        )[..., 0]
        return prediction.reshape(
            num_cases, num_leads, len(x_grid), len(y_grid),
        )

    correct = predict(forcing)
    zeroed = predict(zero_forcing)
    shuffled = predict(shuffled_forcing)
    truth = torch.as_tensor(
        np.stack([case["truth"] for case in cases]),
        dtype=correct.dtype,
        device=device,
    )
    nonzero = slice(1, None)

    def rmse_K(prediction: torch.Tensor) -> float:
        return float(
            torch.sqrt(
                (prediction[nonzero] - truth[nonzero]).square().mean()
            ).cpu()
        ) * float(sigma)

    predicted_response = correct[2] - correct[0]
    true_response = truth[2] - truth[0]
    predicted_norm = torch.linalg.vector_norm(predicted_response)
    true_norm = torch.linalg.vector_norm(true_response)
    denominator = true_norm.clamp_min(torch.finfo(true_norm.dtype).tiny)
    cosine_denominator = (
        predicted_norm * true_norm
    ).clamp_min(torch.finfo(true_norm.dtype).tiny)
    correct_rmse = rmse_K(correct)
    zero_rmse = rmse_K(zeroed)
    shuffled_rmse = rmse_K(shuffled)
    return {
        "correct_forcing_rmse_K": correct_rmse,
        "zero_forcing_rmse_K": zero_rmse,
        "shuffled_forcing_rmse_K": shuffled_rmse,
        "forcing_permutation_gap_K": shuffled_rmse - correct_rmse,
        "designed_response_gain": float(
            (predicted_norm / denominator).cpu()
        ),
        "designed_response_cosine": float(
            (
                (predicted_response * true_response).sum()
                / cosine_denominator
            ).cpu()
        ),
        "predicted_response_rms_K": float(
            torch.sqrt(predicted_response.square().mean()).cpu()
        ) * float(sigma),
        "true_response_rms_K": float(
            torch.sqrt(true_response.square().mean()).cpu()
        ) * float(sigma),
    }


def load_verified_transition_gate(
    path: str | Path,
    *,
    expected_stage: str,
    require_passed: bool = True,
) -> tuple[dict[str, Any], str]:
    gate_path = Path(path).expanduser().resolve()
    with gate_path.open() as stream:
        summary = json.load(stream)
    recorded = str(summary.get("gate_sha256", ""))
    unhashed = copy.deepcopy(summary)
    unhashed.pop("gate_sha256", None)
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not recorded or computed != recorded:
        raise ValueError(f"gate summary hash mismatch: {gate_path}")
    if str(summary.get("stage")) != str(expected_stage):
        raise ValueError(
            f"expected gate stage {expected_stage!r}, got "
            f"{summary.get('stage')!r}"
        )
    if require_passed and not bool(summary.get("passed", False)):
        raise ValueError(f"gate did not pass: {gate_path}")
    return summary, recorded


def transition_screen_skill(
    error: float,
    floor_error: float,
    copy_error: float,
    *,
    separation: float = 1.0e3,
) -> float:
    values = np.asarray(
        [error, floor_error, copy_error], dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("screen errors must be finite and non-negative")
    if copy_error < float(separation) * floor_error:
        raise ValueError("copy-to-floor separation prerequisite failed")
    denominator = copy_error - floor_error
    if denominator <= 0.0:
        raise ValueError("copy error must exceed the reconstruction floor")
    return float(1.0 - (error - floor_error) / denominator)


def transition_consecutive_interval_trigger(
    *,
    overall_passed: bool,
    longest_passed: bool,
    copy_skills: list[float] | tuple[float, ...],
    forcing_response_ratio: float,
    source_departure_ratio: float,
    finite_and_defect_passed: bool,
    low_frequency_fractions: list[float] | tuple[float, ...],
) -> bool:
    fractions = np.asarray(low_frequency_fractions, dtype=np.float64)
    skills = np.asarray(copy_skills, dtype=np.float64)
    if fractions.size < 3 or skills.size == 0:
        return False
    last = fractions[-3:]
    monotone = bool(np.all(last[1:] >= 0.95 * last[:-1]))
    drift = bool(fractions[-1] >= 2.0 * fractions[0])
    return bool(
        overall_passed
        and not longest_passed
        and np.isfinite(skills).all()
        and bool(np.all(skills > 0.0))
        and float(forcing_response_ratio) >= 0.10
        and float(source_departure_ratio) >= 0.10
        and finite_and_defect_passed
        and np.isfinite(fractions).all()
        and monotone
        and drift
    )


def transition_production_screen_decision(
    records: dict[int, dict[str, float]],
    *,
    minimum_improvement: float = 0.10,
) -> dict[str, Any]:
    required = (500, 750, 1000, 1500, 1750, 2000)
    missing = [update for update in required if update not in records]
    if missing:
        raise ValueError(f"production screen is missing updates {missing}")

    def combined(update: int) -> float:
        record = records[update]
        return 0.5 * (
            float(record["overall_skill"]) + float(record["long_skill"])
        )

    early = float(np.median([combined(update) for update in required[:3]]))
    late = float(np.median([combined(update) for update in required[3:]]))
    final = records[2000]
    finite = bool(
        all(
            np.isfinite(float(value))
            for record in records.values()
            for value in record.values()
        )
    )
    progress = bool(
        finite
        and float(final["overall_skill"]) > 0.0
        and float(final["long_skill"]) > 0.0
        and late - early >= float(minimum_improvement)
    )
    return {
        "early_combined_skill": early,
        "late_combined_skill": late,
        "skill_improvement": late - early,
        "finite": finite,
        "passed": progress,
    }


def sample_transition_source_steps(
    rng: np.random.Generator,
    *,
    batch_size: int,
    dt: float,
    t_final: float,
    source_edges: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(source_edges, dtype=np.float64)
    lattice = np.arange(
        int(round(float(t_final) / float(dt))), dtype=np.int64,
    )
    times = lattice.astype(np.float64) * float(dt)
    groups = [
        lattice[
            (times >= edges[index] - 1.0e-12)
            & (times < edges[index + 1] - 1.0e-12)
        ]
        for index in range(len(edges) - 1)
    ]
    eligible = [index for index, values in enumerate(groups) if values.size]
    if not eligible:
        raise ValueError("source-time bins contain no feasible lattice points")
    requested = np.resize(np.asarray(eligible, dtype=np.int64), int(batch_size))
    rng.shuffle(requested)
    steps = np.asarray(
        [int(rng.choice(groups[int(index)])) for index in requested],
        dtype=np.int64,
    )
    return steps, requested


def transition_curriculum_max_lead(
    completed_updates: int,
    total_updates: int,
    feasible_horizon: float,
) -> float:
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    fraction = (int(completed_updates) + 1) / int(total_updates)
    if fraction <= 0.10:
        requested = 0.05
    elif fraction <= 0.25:
        requested = 0.10
    elif fraction <= 0.50:
        requested = 0.20
    else:
        requested = 0.30
    return min(float(requested), float(feasible_horizon))


def sample_transition_physics_intervals(
    rng: np.random.Generator,
    *,
    source_steps: np.ndarray,
    dt: float,
    t_final: float,
    lead_edges: list[float] | tuple[float, ...],
    include_anchor_interval: bool,
    intervals_per_cell: int,
    consecutive_intervals: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    if int(intervals_per_cell) <= 0:
        raise ValueError("intervals_per_cell must be positive")
    edges = np.asarray(lead_edges, dtype=np.float64)
    total_steps = int(round(float(t_final) / float(dt)))
    cfg = dict(consecutive_intervals or {})
    enabled = bool(cfg.get("enabled", False))
    probability = float(cfg.get("probability", 0.25))
    min_length = int(cfg.get("min_length", 2))
    max_length = int(cfg.get("max_length", 4))
    if (
        not 0.0 <= probability <= 1.0
        or min_length < 2
        or max_length < min_length
    ):
        raise ValueError("invalid consecutive_intervals configuration")

    rows: list[tuple[int, int, int, bool]] = []
    for sim_local, source_step in enumerate(
        np.asarray(source_steps, dtype=np.int64),
    ):
        feasible = total_steps - int(source_step)
        if feasible <= 0:
            continue
        if include_anchor_interval:
            rows.append((sim_local, 0, 0, True))
        end_steps = np.arange(1, feasible + 1, dtype=np.int64)
        leads = end_steps.astype(np.float64) * float(dt)
        for lead_bin in range(len(edges) - 1):
            candidates = end_steps[
                (leads > edges[lead_bin] + 1.0e-12)
                & (leads <= edges[lead_bin + 1] + 1.0e-12)
            ]
            if candidates.size == 0:
                continue
            for _ in range(int(intervals_per_cell)):
                end_step = int(rng.choice(candidates))
                start = end_step - 1
                rows.append((sim_local, start, lead_bin, start == 0))
                if enabled and rng.random() < probability:
                    length = int(rng.integers(min_length, max_length + 1))
                    first = min(start, max(0, feasible - length))
                    for offset in range(length):
                        interval = first + offset
                        if interval < feasible:
                            endpoint_lead = float(interval + 1) * float(dt)
                            bundle_bin = int(
                                np.searchsorted(
                                    edges, endpoint_lead, side="left",
                                ) - 1
                            )
                            bundle_bin = min(
                                max(bundle_bin, 0), len(edges) - 2,
                            )
                            rows.append(
                                (
                                    sim_local,
                                    interval,
                                    bundle_bin,
                                    interval == 0,
                                )
                            )
    unique = list(dict.fromkeys(rows))
    if not unique:
        raise ValueError("no feasible transition physics intervals")
    return {
        "sim_local": torch.tensor(
            [row[0] for row in unique], dtype=torch.long,
        ),
        "start_step": torch.tensor(
            [row[1] for row in unique], dtype=torch.long,
        ),
        "lead_bin": torch.tensor(
            [row[2] for row in unique], dtype=torch.long,
        ),
        "is_anchor": torch.tensor(
            [row[3] for row in unique], dtype=torch.bool,
        ),
    }


def _repeat_transition_intervals_by_continuation(
    intervals: dict[str, torch.Tensor],
    continuations_per_source: int,
) -> dict[str, torch.Tensor]:
    continuations_per_source = int(continuations_per_source)
    if continuations_per_source < 1:
        raise ValueError(
            "forcing_continuations_per_source must be positive"
        )
    sim_local = intervals["sim_local"]
    count = int(sim_local.numel())
    for value in intervals.values():
        if value.ndim != 1 or int(value.numel()) != count:
            raise ValueError("transition interval fields must be aligned vectors")
    offsets = torch.arange(
        continuations_per_source,
        dtype=sim_local.dtype,
        device=sim_local.device,
    ).repeat(count)
    expanded = {
        "sim_local": (
            sim_local.repeat_interleave(continuations_per_source)
            * continuations_per_source
            + offsets
        ),
    }
    expanded.update({
        key: value.repeat_interleave(continuations_per_source)
        for key, value in intervals.items()
        if key != "sim_local"
    })
    return expanded


def _macro_transition_cell_mean(
    values: torch.Tensor,
    source_bins: torch.Tensor,
    lead_bins: torch.Tensor,
    physical_defect: torch.Tensor,
    *,
    causal_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    unique_leads = torch.unique(lead_bins, sorted=True)
    lead_defects = torch.stack([
        physical_defect[lead_bins == lead].mean()
        for lead in unique_leads
    ])
    lead_weights = _causal_weights(
        lead_defects.detach(), float(causal_epsilon),
    )
    sample_weights = torch.ones_like(values)
    for index, lead in enumerate(unique_leads):
        sample_weights[lead_bins == lead] = lead_weights[index]
    cells = torch.stack((source_bins, lead_bins), dim=1)
    unique_cells = torch.unique(cells, dim=0)
    cell_values = []
    for cell in unique_cells:
        mask = (cells == cell).all(dim=1)
        cell_values.append((sample_weights[mask] * values[mask]).mean())
    return torch.stack(cell_values).mean(), lead_defects


def forcing_transition_physics_loss(
    *,
    model: ForcingTransitionCViT,
    source_fields: torch.Tensor,
    params: list[dict],
    source_steps: np.ndarray,
    source_bins: np.ndarray,
    intervals: dict[str, torch.Tensor],
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    y_img: np.ndarray,
    nt_img: int,
    a_ref: float,
    t_ramp: float,
    t_final: float,
    dt: float,
    sigma_global: float,
    right_value: float,
    objective: str,
    causal_epsilon: float,
    query_chunk: int,
    defect_sweeps: int = 1,
    defect_omega: float = 2.0 / 3.0,
) -> dict[str, torch.Tensor | bool]:
    device = source_fields.device
    dtype = source_fields.dtype
    if source_fields.ndim != 4 or source_fields.shape[1] != 1:
        raise ValueError("source_fields must have shape (B,1,Nx,Ny)")
    sim_local = intervals["sim_local"].to(device=device, dtype=torch.long)
    start_step = intervals["start_step"].to(device=device, dtype=torch.long)
    lead_bins = intervals["lead_bin"].to(device=device, dtype=torch.long)
    M = int(sim_local.numel())
    if M == 0:
        raise ValueError("transition physics batch is empty")
    Nx, Ny = int(source_fields.shape[-2]), int(source_fields.shape[-1])
    mesh = _full_grid_query_mesh(x_grid, y_grid).expand(M * 2, -1, -1)
    endpoint_sim = torch.cat((sim_local, sim_local), dim=0)
    endpoint_steps = torch.cat((start_step, start_step + 1), dim=0)
    positive = endpoint_steps > 0
    predictions: list[torch.Tensor | None] = [None] * (2 * M)
    for index in torch.nonzero(~positive, as_tuple=False).flatten().tolist():
        predictions[index] = source_fields[int(endpoint_sim[index]), 0]

    if bool(positive.any().item()):
        selected = torch.nonzero(positive, as_tuple=False).flatten()
        selected_sim = endpoint_sim.index_select(0, selected)
        selected_steps = endpoint_steps.index_select(0, selected)
        source_times = torch.as_tensor(
            np.asarray(source_steps, dtype=np.float64) * float(dt),
            device=device,
            dtype=dtype,
        ).index_select(0, selected_sim)
        lead_times = selected_steps.to(dtype=dtype) * float(dt)
        selected_params = [params[int(index)] for index in selected_sim.cpu()]
        forcing = build_forcing_transition_image(
            selected_params,
            y_img,
            source_times,
            lead_times,
            int(nt_img),
            float(a_ref),
            device,
            float(t_ramp),
            float(t_final),
        ).to(dtype=dtype)
        source_tokens = model.encode_source(source_fields)
        forcing_tokens = model.encode_forcing(forcing)
        encoding = model.fuse(
            source_tokens.index_select(0, selected_sim),
            forcing_tokens,
        )
        decoded = model.decode(
            encoding,
            source_fields.index_select(0, selected_sim),
            mesh.index_select(0, selected),
            source_times,
            lead_times,
        )[..., 0].reshape(-1, Nx, Ny)
        for local, index in enumerate(selected.tolist()):
            predictions[index] = decoded[local]
    if any(value is None for value in predictions):
        raise AssertionError("not every direct endpoint was decoded")
    T_n = torch.stack(
        [value for value in predictions[:M] if value is not None],
    )
    T_np1 = torch.stack(
        [value for value in predictions[M:] if value is not None],
    )

    source_time_values = (
        np.asarray(source_steps, dtype=np.float64)[sim_local.cpu().numpy()]
        * float(dt)
    )
    start_values = start_step.cpu().numpy()
    qn = np.empty((M, Ny), dtype=np.float64)
    qnp1 = np.empty((M, Ny), dtype=np.float64)
    qint = np.empty((M, Ny), dtype=np.float64)
    history_active = np.empty(M, dtype=bool)
    y_values = np.asarray(y_img, dtype=np.float64)
    if y_values.size != Ny or not np.allclose(
        y_values, y_grid.detach().cpu().numpy(), rtol=0.0, atol=1.0e-12,
    ):
        forcing_y = y_grid.detach().cpu().numpy()
    else:
        forcing_y = y_values
    for row in range(M):
        record = params[int(sim_local[row])]
        forcing = reconstruct_qL(
            record["temporal_family"],
            record["temporal_params"],
            record["spatial_family"],
            record["spatial_params"],
            t_ramp=float(t_ramp),
        )
        absolute_n = (
            source_time_values[row] + float(start_values[row]) * float(dt)
        )
        absolute_np1 = absolute_n + float(dt)
        qn[row] = forcing.evaluate_points(forcing_y, absolute_n)
        qnp1[row] = forcing.evaluate_points(forcing_y, absolute_np1)
        qint[row] = forcing.integral(
            forcing_y, absolute_n, absolute_np1,
        )
        history_integral = forcing.integral(
            forcing_y, source_time_values[row], absolute_np1,
        )
        history_active[row] = bool(
            np.max(np.abs(history_integral)) > 1.0e-12
        )

    geom = build_homogeneous_cn_geom(
        x_grid.detach().cpu().numpy(),
        y_grid.detach().cpu().numpy(),
        K_SLAB,
        float(dt),
        sigma_global=float(sigma_global),
        rho=1.0,
        cp=1.0,
        device=device,
        dtype=dtype,
    )
    forcing_increment = (
        2.0 * qint / (float(geom.hx) * float(sigma_global))
    )
    cn = build_cn_tensors_from_geom(
        geom,
        torch.from_numpy(forcing_increment).to(device=device, dtype=dtype),
    )
    if objective == "raw_ls":
        bc = FullBCData(
            T_right_tilde=torch.as_tensor(
                float(right_value),
                device=device,
                dtype=dtype,
            ),
            qL_n=torch.from_numpy(qn).to(device=device, dtype=dtype),
            qL_np1=torch.from_numpy(qnp1).to(device=device, dtype=dtype),
            qL_int=torch.from_numpy(qint).to(device=device, dtype=dtype),
        )
        phys = full_bc_physics_loss(
            T_n,
            T_np1,
            geom,
            bc,
            per_sample=True,
            dirichlet_both_ends=True,
        )
        raw_per_sample = {
            "interior": phys["interior_per_sample"],
            "left_neumann": phys["left_neumann_per_sample"],
            "topbot_adiabatic": phys["topbot_adiabatic_per_sample"],
            "right_dirichlet": phys["right_dirichlet_per_sample"],
        }
        per_sample = sum(raw_per_sample.values())
        physical_defect = phys["physics_loss_allcell_mean"].new_empty(M)
        deviation_np1 = T_np1 - float(right_value)
        deviation_n = T_n - float(right_value)
        implicit = implicit_cn_action(deviation_np1, cn)
        rhs = explicit_cn_rhs(
            deviation_n,
            torch.arange(M, device=device, dtype=torch.long),
            {**cn, "forcing": cn["forcing"]},
        )
        residual = implicit - rhs
        physical_defect = residual.square().flatten(1).mean(dim=1)
    elif objective in {"variational", "defect"}:
        if objective == "variational":
            per_sample, residual = variational_objective(
                T_np1, T_n, cn, right_value=float(right_value),
            )
        else:
            per_sample, residual = defect_terms(
                T_np1,
                T_n,
                cn,
                right_value=float(right_value),
                sweeps=int(defect_sweeps),
                omega=float(defect_omega),
            )
        physical_defect = residual.square().flatten(1).mean(dim=1)
    else:
        raise ValueError(
            "transition physics objective must be raw_ls, variational, or defect"
        )
    interval_source_bins = torch.as_tensor(
        np.asarray(source_bins, dtype=np.int64),
        device=device,
        dtype=torch.long,
    ).index_select(0, sim_local)
    loss, lead_defects = _macro_transition_cell_mean(
        per_sample,
        interval_source_bins,
        lead_bins,
        physical_defect,
        causal_epsilon=float(causal_epsilon),
    )
    local_active = torch.from_numpy(
        np.max(np.abs(qint), axis=1) > 1.0e-12
    ).to(device=device)
    history_active_tensor = torch.from_numpy(history_active).to(device=device)
    detached_objective = per_sample.detach()

    def masked_mean(
        values: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        if bool(mask.any().item()):
            return values[mask].mean()
        return values.new_tensor(float("nan"))

    defect_squared = residual.detach().square()
    boundary_defect = defect_squared[:, 0, :].mean()
    interior_defect = (
        defect_squared[:, 1:, :].mean()
        if defect_squared.shape[-2] > 1
        else defect_squared.new_tensor(float("nan"))
    )
    out: dict[str, torch.Tensor | bool | dict[str, torch.Tensor]] = {
        "loss": loss,
        "physical_defect_mse": physical_defect.mean().detach(),
        "boundary_defect_mse": boundary_defect,
        "interior_defect_mse": interior_defect,
        "local_forcing_active_fraction": local_active.float().mean(),
        "history_forcing_active_fraction": (
            history_active_tensor.float().mean()
        ),
        "active_forcing_objective": masked_mean(
            detached_objective, local_active,
        ),
        "inactive_forcing_objective": masked_mean(
            detached_objective, ~local_active,
        ),
        "lead_defect_mse": lead_defects,
        "finite": bool(
            torch.isfinite(loss).item()
            and torch.isfinite(T_n).all().item()
            and torch.isfinite(T_np1).all().item()
        ),
        "no_rollout": True,
        "previous_endpoint_detached": objective != "raw_ls",
        "decoded_endpoint_count": torch.tensor(
            int(positive.sum().item()), device=device,
        ),
    }
    if objective == "raw_ls":
        out["raw_terms"] = {
            name: _macro_transition_cell_mean(
                values,
                interval_source_bins,
                lead_bins,
                physical_defect,
                causal_epsilon=float(causal_epsilon),
            )[0]
            for name, values in raw_per_sample.items()
        }
    return out


def run_one_seed_forcing_transition(
    config: dict,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Supervised causal transition training for ``diffusion_forcing_single``."""
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "cvit_latest.pt"
    best_path = run_dir / "cvit_best.pt"
    final_path = run_dir / "cvit_final.pt"
    complete_path = run_dir / "RUN_COMPLETE"
    summary_path = run_dir / "final_metrics.json"

    training = config["training"]
    pino = training["pino"]
    transition = pino.get("transition", {}) or {}
    forcing_cfg = pino.get("forcing", {}) or {}
    if float(training.get("noise_std", 0.0) or 0.0) != 0.0:
        raise ValueError(
            "forcing_transition V1 does not support source noise or denoising"
        )
    if float(transition.get("source_noise_std", 0.0) or 0.0) != 0.0:
        raise ValueError(
            "forcing_transition V1 does not support source noise or denoising"
        )
    extend_completed = bool(transition.get("extend_completed", False))
    if complete_path.exists() and not extend_completed:
        if summary_path.exists():
            with open(summary_path) as stream:
                return json.load(stream)
        return {"seed": seed, "status": "complete", "run_dir": str(run_dir)}

    device = resolve_device(training.get("device", "auto"))
    data = load_diffusion_data(config)
    trajectory_path = Path(config["data"]["trajectories.npy"])
    sim_params = np.load(
        trajectory_path.parent / "sim_params.npy", allow_pickle=True,
    )
    data_signature = validate_forcing_ic_supervised_dataset(
        config, data, sim_params,
    )
    trajectories = data["trajectories"]
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    nx, ny = int(len(data["x_grid"])), int(len(data["y_grid"]))
    t_final = float(t_grid[-1])
    mu = float(data["mu_global"])
    sigma = float(data["sigma_global"])
    n_snapshots = training.get("n_snapshots", None)
    schedule = TransitionPairSchedule(
        t_grid,
        None if n_snapshots is None else int(n_snapshots),
        transition.get("lead_edges", [0.0, 0.05, 0.10, 0.20, 0.30]),
        transition.get(
            "source_edges", [0.0, 0.075, 0.15, 0.225, 0.30],
        ),
        transition.get(
            "target_edges", [0.0, 0.075, 0.15, 0.225, 0.30],
        ),
    )

    stored_ramp = float(data_signature["ramp_seconds"])
    configured_ramp = forcing_cfg.get("ramp_seconds", None)
    t_ramp = (
        stored_ramp if configured_ramp is None else float(configured_ramp)
    )
    if not math.isclose(
        t_ramp, stored_ramp, rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError(
            "training.pino.forcing.ramp_seconds must match the saved dataset"
        )
    a_ref = float(forcing_cfg.get("a_ref") or A_AMP_REF)
    ny_img = int(
        forcing_cfg.get("ny_img")
        if forcing_cfg.get("ny_img") is not None
        else ny
    )
    nt_img = int(
        forcing_cfg.get("nt_img")
        if forcing_cfg.get("nt_img") is not None
        else 128
    )
    y_img = np.linspace(
        float(data["y_grid"][0]),
        float(data["y_grid"][-1]),
        ny_img,
        dtype=np.float64,
    )
    sim_batch = int(transition.get("sim_batch", pino.get("sim_batch", 16)))
    n_queries = int(
        transition.get("n_queries", pino.get("n_data_pts", 1024))
    )
    query_chunk = int(transition.get("query_chunk", 0) or 0)
    val_batch = int(
        transition.get(
            "validation_batch_size",
            forcing_cfg.get("val_sim_batch", 8),
        )
    )
    val_chunk = int(
        transition.get(
            "validation_query_chunk",
            forcing_cfg.get("val_query_chunk", 2048),
        )
        or 0
    )
    if (
        sim_batch <= 0
        or n_queries <= 0
        or query_chunk < 0
        or val_batch <= 0
        or val_chunk < 0
    ):
        raise ValueError("transition batch/query settings are invalid")

    fast_ids = _ic_balanced_subset(
        data["val_ids"],
        sim_params,
        limit=int(transition.get("fast_val_max_sims", 64)),
        seed=seed,
    )
    fast_records = schedule.validation_records(
        fast_ids,
        pairs_per_cell=int(
            transition.get("fast_val_pairs_per_cell", 1)
        ),
        base_seed=seed + 10_000,
    )
    final_records = schedule.validation_records(
        data["val_ids"],
        pairs_per_cell=int(
            transition.get("final_val_pairs_per_cell", 2)
        ),
        base_seed=seed + 20_000,
    )
    fast_hash = write_transition_manifest(
        run_dir / "fast_validation_manifest.json", fast_records,
    )
    final_hash = write_transition_manifest(
        run_dir / "final_validation_manifest.json", final_records,
    )

    model = build_cvit(
        config,
        mu,
        sigma,
        grid_size=(nx, ny),
        t_final=t_final,
        variant="forcing_transition",
    ).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    compatibility = _transition_checkpoint_compatibility(
        config,
        nx=nx,
        ny=ny,
        ny_img=ny_img,
        nt_img=nt_img,
        a_ref=a_ref,
        t_final=t_final,
        schedule=schedule,
    )
    resume_config = {
        "compatibility": compatibility,
        "optimizer": training.get("optimizer"),
        "learning_rate": training.get("learning_rate"),
        "weight_decay": training.get("weight_decay"),
        "scheduler": copy.deepcopy(training.get("scheduler")),
        "n_snapshots": n_snapshots,
        "sim_batch": sim_batch,
        "n_queries": n_queries,
        "query_chunk": query_chunk,
        "fast_manifest_hash": fast_hash,
        "final_manifest_hash": final_hash,
    }

    epochs = int(training["epochs"])
    validate_every = int(training.get("validate_every", 10))
    save_every = max(
        1, int(transition.get("save_latest_every_updates", 25)),
    )
    signal_floor = float(
        transition.get("increment_signal_floor_sigma", 0.01)
    )
    loss_weight = float(transition.get("loss_weight", 1.0))
    grad_clip_value = training.get("grad_clip", None)
    grad_clip = (
        None if grad_clip_value is None else float(grad_clip_value)
    )
    if (
        epochs <= 0
        or validate_every <= 0
        or signal_floor < 0.0
        or loss_weight <= 0.0
        or (grad_clip is not None and grad_clip <= 0.0)
    ):
        raise ValueError("transition training settings are invalid")
    train_ids = np.asarray(data["train_ids"], dtype=int)
    num_batches = int(math.ceil(len(train_ids) / sim_batch))
    x_grid = torch.as_tensor(
        data["x_grid"], dtype=torch.float32, device=device,
    )
    y_grid = torch.as_tensor(
        data["y_grid"], dtype=torch.float32, device=device,
    )
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed + 1_000_003)

    metrics_path = run_dir / "train_metrics.csv"
    fieldnames = [
        "epoch",
        "completed_updates",
        "loss",
        "lr",
        "fast_macro_cell_gnrmse",
        "fast_target_gnrmse",
        "fast_target_rmse_K",
        "fast_increment_rel_l2",
        "fast_copy_skill",
        "fast_gated_coverage",
    ]
    start_epoch = 0
    start_batch = 0
    completed_updates = 0
    last_csv_epoch = -1
    partial_loss_sum = 0.0
    partial_batches = 0
    best_metric = float("inf")
    best_epoch: int | None = None
    best_validation: dict[str, Any] | None = None
    resuming = latest_path.exists() and (
        not complete_path.exists() or extend_completed
    )
    if resuming:
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False,
        )
        if checkpoint.get("resume_config") != resume_config:
            raise ValueError(
                "Incompatible forcing-transition resume configuration; "
                "use a fresh experiment name."
            )
        if checkpoint.get("data_signature") != data_signature:
            raise ValueError(
                "The forcing-transition dataset differs from the checkpoint."
            )
        saved_epochs = int(checkpoint["config"]["training"]["epochs"])
        if epochs < int(checkpoint["next_epoch"]):
            raise ValueError(
                "training.epochs is below the transition checkpoint cursor"
            )
        if (
            epochs != saved_epochs
            and str(training["scheduler"]["type"])
            not in {"PICViTExponential", "StepLR"}
        ):
            raise ValueError(
                "Extending transition training requires a "
                "horizon-independent scheduler"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["next_epoch"])
        start_batch = int(checkpoint["next_batch"])
        completed_updates = int(checkpoint["completed_updates"])
        last_csv_epoch = int(checkpoint["last_csv_epoch"])
        partial_loss_sum = float(checkpoint.get("partial_loss_sum", 0.0))
        partial_batches = int(checkpoint.get("partial_batches", 0))
        best_metric = float(checkpoint["best_fast_validation_metric"])
        best_epoch = checkpoint.get("best_checkpoint_epoch")
        best_validation = copy.deepcopy(checkpoint.get("best_validation"))
        _restore_forcing_rng(
            checkpoint["stochastic_state"], rng, gen, eval_gen,
        )
        if metrics_path.exists():
            with open(metrics_path, newline="") as stream:
                rows = [
                    row
                    for row in csv.DictReader(stream)
                    if int(row["epoch"]) <= last_csv_epoch
                ]
            with open(metrics_path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        if complete_path.exists():
            complete_path.unlink()
    else:
        with open(metrics_path, "w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    def checkpoint_payload(
        *,
        epoch_value: int,
        next_epoch: int,
        next_batch: int,
        loss_sum: float,
        batches_done: int,
    ) -> dict[str, Any]:
        return {
            "objective": "supervised_transition",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "mu_global": mu,
            "sigma_global": sigma,
            "config": config,
            "resume_config": resume_config,
            "checkpoint_compatibility": compatibility,
            "data_signature": data_signature,
            "epoch": int(epoch_value),
            "next_epoch": int(next_epoch),
            "next_batch": int(next_batch),
            "completed_updates": int(completed_updates),
            "last_csv_epoch": int(last_csv_epoch),
            "partial_loss_sum": float(loss_sum),
            "partial_batches": int(batches_done),
            "best_fast_validation_metric": float(best_metric),
            "best_checkpoint_epoch": best_epoch,
            "best_validation": copy.deepcopy(best_validation),
            "stochastic_state": _capture_forcing_rng(
                rng, gen, eval_gen,
            ),
            "schedule_state": {
                "base_seed": int(seed),
                "epoch": int(next_epoch),
                "next_batch": int(next_batch),
                "num_batches": int(num_batches),
            },
            "validation_manifests": {
                "fast": fast_hash,
                "final": final_hash,
            },
        }

    print(
        f"[forcing-transition] seed={seed} device={device} epochs={epochs} "
        f"sims/epoch={len(train_ids)} batches/epoch={num_batches} "
        f"batch={sim_batch} queries={n_queries} img={ny_img}x{nt_img} "
        f"resume={resuming}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs):
        order = transition_epoch_order(
            train_ids, base_seed=seed, epoch=epoch,
        )
        batch_begin = start_batch if epoch == start_epoch else 0
        epoch_loss_sum = (
            partial_loss_sum if epoch == start_epoch else 0.0
        )
        epoch_batches = (
            partial_batches if epoch == start_epoch else 0
        )
        model.train()
        lr = float(optimizer.param_groups[0]["lr"])
        for batch_index in range(batch_begin, num_batches):
            batch_ids = order[
                batch_index * sim_batch:(batch_index + 1) * sim_batch
            ]
            records = []
            for sim_id in batch_ids:
                source_index, target_index, source_bin, lead_bin = (
                    schedule.sample_pair(seed, epoch, int(sim_id))
                )
                _, _, target_bin = schedule.cell_for_pair(
                    source_index, target_index,
                )
                records.append({
                    "sim_id": int(sim_id),
                    "source_index": int(source_index),
                    "target_index": int(target_index),
                    "source_bin": int(source_bin),
                    "lead_bin": int(lead_bin),
                    "target_bin": int(target_bin),
                })
            optimizer.zero_grad(set_to_none=True)
            data_loss = _transition_batch_loss(
                model,
                records,
                sim_params=sim_params,
                trajectories=trajectories,
                t_grid=t_grid,
                x_grid=x_grid,
                y_grid=y_grid,
                y_img=y_img,
                nt_img=nt_img,
                a_ref=a_ref,
                t_ramp=t_ramp,
                t_final=t_final,
                mu=mu,
                sigma=sigma,
                n_queries=n_queries,
                query_chunk=query_chunk,
                base_seed=seed,
                epoch=epoch,
                device=device,
            )
            loss = loss_weight * data_loss
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"non-finite transition loss at epoch {epoch}, "
                    f"batch {batch_index}"
                )
            loss.backward()
            if not all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"non-finite transition gradient at epoch {epoch}, "
                    f"batch {batch_index}"
                )
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip,
                )
            lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            _advance_scheduler(
                scheduler, unit="update", successful_updates=1,
            )
            completed_updates += 1
            epoch_loss_sum += float(data_loss.detach().cpu())
            epoch_batches += 1
            next_batch = batch_index + 1
            if (
                completed_updates % save_every == 0
                and next_batch < num_batches
            ):
                _atomic_torch_save(
                    checkpoint_payload(
                        epoch_value=epoch,
                        next_epoch=epoch,
                        next_batch=next_batch,
                        loss_sum=epoch_loss_sum,
                        batches_done=epoch_batches,
                    ),
                    latest_path,
                )

        _advance_scheduler(
            scheduler, unit="epoch", successful_updates=1,
        )
        do_validation = (
            epoch % validate_every == 0 or epoch == epochs - 1
        )
        validation: dict[str, Any] | None = None
        is_best = False
        if do_validation:
            model.eval()
            validation = validate_forcing_transition(
                model,
                data,
                sim_params,
                fast_records,
                y_img=y_img,
                nt_img=nt_img,
                a_ref=a_ref,
                t_ramp=t_ramp,
                device=device,
                sim_batch=val_batch,
                query_chunk=val_chunk,
                signal_floor_fraction=signal_floor,
            )
            metric = float(validation["macro_cell_gnrmse"])
            is_best = metric < best_metric
            if is_best:
                best_metric = metric
                best_epoch = epoch
                best_validation = copy.deepcopy(validation)
        row = {
            "epoch": epoch,
            "completed_updates": completed_updates,
            "loss": epoch_loss_sum / epoch_batches,
            "lr": lr,
            "fast_macro_cell_gnrmse": (
                "" if validation is None
                else validation["macro_cell_gnrmse"]
            ),
            "fast_target_gnrmse": (
                "" if validation is None else validation["target_gnrmse"]
            ),
            "fast_target_rmse_K": (
                "" if validation is None else validation["target_rmse_K"]
            ),
            "fast_increment_rel_l2": (
                "" if validation is None
                else validation["increment_rel_l2"]
            ),
            "fast_copy_skill": (
                "" if validation is None else validation["copy_skill"]
            ),
            "fast_gated_coverage": (
                "" if validation is None
                else validation["gated_coverage"]
            ),
        }
        with open(metrics_path, "a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        last_csv_epoch = epoch
        payload = checkpoint_payload(
            epoch_value=epoch,
            next_epoch=epoch + 1,
            next_batch=0,
            loss_sum=0.0,
            batches_done=0,
        )
        if is_best:
            _atomic_torch_save(payload, best_path)
        _atomic_torch_save(payload, latest_path)
        print(
            f"Epoch {epoch}: loss={row['loss']:.6e} "
            + (
                f"macro_cell_gnrmse={100.0 * float(validation['macro_cell_gnrmse']):.4f}% "
                f"copy_skill={validation['copy_skill']}"
                if validation is not None
                else f"lr={lr:.2e}"
            ),
            flush=True,
        )
        start_batch = 0
        partial_loss_sum = 0.0
        partial_batches = 0

    if not best_path.exists():
        raise RuntimeError("forcing-transition training produced no best checkpoint")
    best_checkpoint = torch.load(
        best_path, map_location="cpu", weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()
    final_validation = validate_forcing_transition(
        model,
        data,
        sim_params,
        final_records,
        y_img=y_img,
        nt_img=nt_img,
        a_ref=a_ref,
        t_ramp=t_ramp,
        device=device,
        sim_batch=val_batch,
        query_chunk=val_chunk,
        signal_floor_fraction=signal_floor,
        pair_csv_path=run_dir / "final_val_pairs.csv",
    )
    diagnostics = transition_counterfactual_diagnostics(
        model,
        data,
        sim_params,
        final_records,
        y_img=y_img,
        nt_img=nt_img,
        a_ref=a_ref,
        t_ramp=t_ramp,
        device=device,
        query_chunk=val_chunk,
    )
    final_payload = copy.deepcopy(best_checkpoint)
    final_payload["final_validation"] = copy.deepcopy(final_validation)
    final_payload["counterfactual_diagnostics"] = diagnostics
    _atomic_torch_save(final_payload, final_path)
    summary = {
        "seed": seed,
        "objective": "supervised_transition",
        "formulation": "forcing_transition",
        "completed_updates": completed_updates,
        "best_fast_validation_metric": best_metric,
        "best_checkpoint_epoch": best_epoch,
        "best_validation": best_validation,
        "final_validation": final_validation,
        "counterfactual_diagnostics": diagnostics,
        "fast_validation_manifest_hash": fast_hash,
        "final_validation_manifest_hash": final_hash,
        "training_sigma_global": sigma,
        "test_set_evaluated": False,
        "comparison_scope": (
            "legacy CViT versus transition CViT is a formulation comparison"
        ),
    }
    _atomic_text(json.dumps(summary, indent=2) + "\n", summary_path)
    _atomic_text("complete\n", complete_path)
    return summary


def _transition_gate_floor(
    direct_gate: dict[str, Any],
    metric: str,
    *,
    sigma_global: float,
) -> float:
    values = [
        float(case["floor_metrics"][metric])
        for case in direct_gate["cases"]
        if metric in case.get("floor_metrics", {})
    ]
    if not values:
        raise ValueError(f"direct-state gate has no floor metric {metric!r}")
    return max(values) / float(sigma_global)


def _transition_validation_screen_metrics(
    validation: dict[str, Any],
    direct_gate: dict[str, Any] | None,
    *,
    sigma_global: float,
    anti_collapse_passed: bool,
    defect_passed: bool,
) -> dict[str, Any]:
    lead_groups = validation["by_lead_bin"]
    lead_keys = sorted(lead_groups, key=lambda value: int(value))
    if not lead_keys:
        raise ValueError("transition validation has no populated lead bins")
    longest = lead_groups[lead_keys[-1]]
    if direct_gate is None:
        overall_floor = 0.0
        long_floor = 0.0
    else:
        overall_floor = _transition_gate_floor(
            direct_gate, "overall_field", sigma_global=sigma_global,
        )
        long_floor = _transition_gate_floor(
            direct_gate, "longest_lead", sigma_global=sigma_global,
        )
    overall_skill = transition_screen_skill(
        float(validation["target_gnrmse"]),
        overall_floor,
        float(validation["copy_gnrmse"]),
    )
    long_skill = transition_screen_skill(
        float(longest["target_gnrmse"]),
        long_floor,
        float(longest["copy_gnrmse"]),
    )
    low_fractions = [
        float(lead_groups[key]["low_frequency_error_fraction"])
        for key in lead_keys
        if lead_groups[key]["low_frequency_error_fraction"] is not None
    ]
    drift = bool(
        len(low_fractions) >= 3
        and all(
            low_fractions[index + 1] >= 0.95 * low_fractions[index]
            for index in range(len(low_fractions) - 3, len(low_fractions) - 1)
        )
        and low_fractions[-1] >= 2.0 * low_fractions[0]
        and overall_skill > 0.0
        and long_skill <= 0.0
        and bool(anti_collapse_passed)
        and bool(defect_passed)
    )
    return {
        "overall_skill": overall_skill,
        "long_skill": long_skill,
        "overall_floor_gnrmse": overall_floor,
        "long_floor_gnrmse": long_floor,
        "floor_source": (
            "zero" if direct_gate is None else "dense_direct_state_gate"
        ),
        "low_frequency_fractions": low_fractions,
        "low_frequency_long_lead_drift": drift,
    }


def _transition_source_coverage(
    online_source: np.ndarray,
    data: dict[str, Any],
    validation_records: list[dict[str, int]],
    *,
    max_sources: int = 16,
) -> dict[str, float]:
    online = np.asarray(online_source, dtype=np.float64)[:max_sources]
    unique_records = []
    seen = set()
    for record in validation_records:
        key = (int(record["sim_id"]), int(record["source_index"]))
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
        if len(unique_records) >= max_sources:
            break
    validation = np.stack([
        np.asarray(
            data["trajectories"][
                int(record["sim_id"]), int(record["source_index"])
            ],
            dtype=np.float64,
        )
        for record in unique_records
    ])
    online_flat = online.reshape(len(online), -1)
    validation_flat = validation.reshape(len(validation), -1)
    distances = np.sqrt(np.mean(
        (
            validation_flat[:, None, :]
            - online_flat[None, :, :]
        ) ** 2,
        axis=-1,
    ))
    nearest = distances.min(axis=1)
    return {
        "validation_to_online_nearest_rmse_K_mean": float(nearest.mean()),
        "validation_to_online_nearest_rmse_K_max": float(nearest.max()),
        "online_sources": int(len(online)),
        "validation_sources": int(len(validation)),
    }


def run_one_seed_forcing_transition_physics(
    config: dict,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "cvit_latest.pt"
    best_path = run_dir / "cvit_best.pt"
    final_path = run_dir / "cvit_final.pt"
    complete_path = run_dir / "RUN_COMPLETE"
    summary_path = run_dir / "final_metrics.json"
    forcing_init_path = run_dir / "cvit_forcing_init.pt"

    training = config["training"]
    compact_stdout = bool(
        (training.get("run", {}) or {}).get("compact_stdout", False)
    )
    pino = training["pino"]
    transition = dict(pino.get("transition", {}) or {})
    physics_cfg = dict(transition.get("physics", {}) or {})
    if str(transition.get("objective", "supervised")) != "physics_only":
        raise ValueError(
            "physics transition runner requires "
            "training.pino.transition.objective=physics_only"
        )
    for key, expected in (
        ("residual_method", "finite_volume"),
        ("source_state_mode", "online_local_ivp"),
        ("time_bundle", "lead_bins"),
    ):
        if str(physics_cfg.get(key)) != expected:
            raise ValueError(
                f"training.pino.transition.physics.{key} must be {expected!r}"
            )
    if not bool(physics_cfg.get("include_anchor_interval", True)):
        raise ValueError("physics-only transition training requires the anchor interval")
    legacy_curriculum = dict(pino.get("curriculum", {}) or {})
    if (
        bool(legacy_curriculum.get("enabled", False))
        and str(legacy_curriculum.get("mode", "ic_only")) == "ic_only"
        and int(legacy_curriculum.get("ic_only_epochs", 0)) > 0
    ):
        raise ValueError(
            "IC-only curricula are incompatible with the hard transition anchor"
        )
    if float(training.get("noise_std", 0.0) or 0.0) != 0.0:
        raise ValueError("physics-only transition V1 does not support denoising")

    gate_path = physics_cfg.get("gate_summary")
    network_path = physics_cfg.get("network_gate_summary")
    direct_gate = None
    direct_hash = None
    if gate_path not in (None, ""):
        direct_gate, direct_hash = load_verified_transition_gate(
            gate_path, expected_stage="dense_direct_state_gate",
        )
    network_gate = None
    network_hash = None
    if network_path not in (None, ""):
        network_gate, network_hash = load_verified_transition_gate(
            network_path, expected_stage="single_instance_network_gate",
        )
    if (
        direct_gate is not None
        and network_gate is not None
        and str(network_gate.get("direct_state_gate_sha256")) != direct_hash
    ):
        raise ValueError(
            "network gate was not authorized by the supplied direct gate"
        )
    configured_objective = physics_cfg.get("objective")
    if network_gate is None:
        if configured_objective is None:
            raise ValueError(
                "training.pino.transition.physics.objective is required when "
                "no network gate summary is supplied"
            )
        selected_objective = str(configured_objective)
        if direct_gate is not None:
            qualified = tuple(direct_gate.get("qualified_objectives", ()))
            if qualified and selected_objective not in qualified:
                raise ValueError(
                    "configured physics objective was not qualified by the "
                    "supplied direct-state gate"
                )
        if not compact_stdout:
            print(
                "[pino-transition] network gate not supplied; using configured "
                f"objective={selected_objective!r}",
                flush=True,
            )
    else:
        selected_objective = str(network_gate["selected_objective"])
        if (
            configured_objective is not None
            and str(configured_objective) != selected_objective
        ):
            raise ValueError(
                "configured physics objective differs from the network-gate "
                "selection"
            )
    if selected_objective not in {"raw_ls", "variational", "defect"}:
        raise ValueError("physics transition objective is unsupported")
    consecutive = dict(physics_cfg.get("consecutive_intervals", {}) or {})
    if (
        network_gate is not None
        and consecutive != dict(network_gate.get("consecutive_intervals", {}))
    ):
        raise ValueError(
            "production consecutive-interval policy differs from the network gate"
        )
    if direct_gate is None and not compact_stdout:
        print(
            "[pino-transition] direct-state gate not supplied; production "
            "screen skills use a zero numerical floor",
            flush=True,
        )

    extend_completed = bool(transition.get("extend_completed", False))
    if complete_path.exists() and not extend_completed:
        if summary_path.exists():
            with summary_path.open() as stream:
                return json.load(stream)
        return {"seed": seed, "status": "complete", "run_dir": str(run_dir)}
    device = resolve_device(training.get("device", "auto"))
    data = load_diffusion_data(config)
    mu = float(data["mu_global"])
    sigma = float(data["sigma_global"])
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("physics-only transition requires a positive normalizer")
    gate_sigma = (
        None
        if direct_gate is None
        else direct_gate.get("configuration", {}).get(
            "resolved_sigma_global",
        )
    )
    if gate_sigma is not None and not math.isclose(
        float(gate_sigma), sigma, rel_tol=1.0e-7, abs_tol=1.0e-10,
    ):
        raise ValueError(
            "direct-state gate used a different training normalizer"
        )
    sim_params = np.load(
        Path(config["data"]["trajectories.npy"]).parent / "sim_params.npy",
        allow_pickle=True,
    )
    data_signature = validate_forcing_ic_supervised_dataset(
        config, data, sim_params,
    )
    problem = problem_from_config(config)
    x_np = np.asarray(data["x_grid"], dtype=np.float64)
    y_np = np.asarray(data["y_grid"], dtype=np.float64)
    X, Y = np.meshgrid(x_np, y_np, indexing="ij")
    t_grid = np.asarray(data["t_grid"], dtype=np.float64)
    t_final = float(t_grid[-1])
    dt_cfg = physics_cfg.get("dt")
    dt = (
        float(dt_cfg)
        if dt_cfg is not None
        else float(load_solver_dt(config["data"]["t_grid_path"]))
    )
    n_steps = int(round(t_final / dt))
    if not math.isclose(
        n_steps * dt, t_final, rel_tol=1.0e-6, abs_tol=1.0e-8,
    ):
        raise ValueError("physics transition dt must divide t_final")

    forcing_cfg = dict(pino.get("forcing", {}) or {})
    t_ramp = forcing_cfg.get("ramp_seconds")
    if t_ramp is None:
        t_ramp = data_signature["ramp_seconds"]
    t_ramp = float(t_ramp)
    a_ref = float(forcing_cfg.get("a_ref") or A_AMP_REF)
    ny_img = int(
        forcing_cfg.get("ny_img")
        if forcing_cfg.get("ny_img") is not None else len(y_np)
    )
    nt_img = int(
        forcing_cfg.get("nt_img")
        if forcing_cfg.get("nt_img") is not None else 128
    )
    y_img = np.linspace(y_np[0], y_np[-1], ny_img, dtype=np.float64)
    x_grid = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    right_value = (300.0 - mu) / sigma

    schedule = TransitionPairSchedule(
        t_grid,
        training.get("n_snapshots"),
        transition.get("lead_edges", [0.0, 0.05, 0.10, 0.20, 0.30]),
        transition.get(
            "source_edges", [0.0, 0.075, 0.15, 0.225, 0.30],
        ),
        transition.get(
            "target_edges", [0.0, 0.075, 0.15, 0.225, 0.30],
        ),
    )
    fast_ids = _ic_balanced_subset(
        data["val_ids"],
        sim_params,
        limit=int(transition.get("fast_val_max_sims", 64)),
        seed=seed,
    )
    fast_records = schedule.validation_records(
        fast_ids,
        pairs_per_cell=int(
            transition.get("fast_val_pairs_per_cell", 1)
        ),
        base_seed=seed + 10_000,
    )
    fast_hash = write_transition_manifest(
        run_dir / "fast_validation_manifest.json", fast_records,
    )

    model = build_cvit(
        config,
        mu,
        sigma,
        grid_size=(len(x_np), len(y_np)),
        t_final=t_final,
        variant="forcing_transition",
    ).to(device)
    source_prefixes = ("source_encoder.",)
    forcing_prefixes = (
        "forcing_encoder.",
        "forcing_context_projection.",
    )
    initial_forcing_parameters = _transition_initial_parameters(
        model, forcing_prefixes,
    )
    if latest_path.exists() and forcing_init_path.exists():
        initial_forcing_parameters = torch.load(
            forcing_init_path, map_location="cpu", weights_only=True,
        )
    else:
        _atomic_torch_save(
            initial_forcing_parameters, forcing_init_path,
        )
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    causal_cfg = _resolve_forcing_causal(pino.get("causal", {}))
    causal_epsilon = (
        float(causal_cfg["initial_eps"]) if causal_cfg["enabled"] else 0.0
    )
    warmup = _forcing_warmup_config(forcing_cfg)
    raw_gradnorm = (
        build_gradnorm(
            config,
            term_weights={
                "interior": 1.0,
                "left_neumann": 1.0,
                "topbot_adiabatic": 1.0,
                "right_dirichlet": 1.0,
            },
        )
        if selected_objective == "raw_ls" else None
    )
    gradnorm_inert = bool(
        selected_objective != "raw_ls"
        and bool((training.get("gradnorm", {}) or {}).get("enabled", False))
    )

    screen_cfg = dict(physics_cfg.get("production_screen", {}) or {})
    screen_enabled = bool(screen_cfg.get("enabled", True))
    registered_screen_updates = int(screen_cfg.get("updates", 2000))
    updates_per_epoch = physics_cfg.get("updates_per_epoch")
    updates_per_epoch = (
        1 if updates_per_epoch is None else int(updates_per_epoch)
    )
    total_updates = (
        registered_screen_updates
        if screen_enabled
        else int(training["epochs"]) * updates_per_epoch
    )
    if screen_enabled and registered_screen_updates != 2000:
        raise ValueError("the production screen budget is preregistered at 2000")
    validate_every = int(screen_cfg.get("validate_every", 250))
    anti_collapse_every = int(screen_cfg.get("anti_collapse_every", 100))
    collapse_update = int(screen_cfg.get("collapse_update", 500))
    minimum_skill_improvement = float(
        screen_cfg.get("minimum_skill_improvement", 0.10)
    )
    if screen_enabled and (
        validate_every != 250
        or anti_collapse_every != 100
        or collapse_update != 500
        or not math.isclose(
            minimum_skill_improvement, 0.10, rel_tol=0.0, abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "production screen cadence and skill threshold are preregistered"
        )
    if (
        total_updates <= 0
        or validate_every <= 0
        or anti_collapse_every <= 0
    ):
        raise ValueError("invalid physics transition update schedule")
    sim_batch = int(transition.get("sim_batch", pino.get("sim_batch", 16)))
    val_batch = int(transition.get("validation_batch_size", 8))
    query_chunk = int(
        transition.get("validation_query_chunk", 2048) or 0
    )
    intervals_per_cell = int(physics_cfg.get("intervals_per_cell", 1))
    continuations_per_source = int(
        physics_cfg.get("forcing_continuations_per_source", 2)
    )
    if continuations_per_source < 1:
        raise ValueError(
            "forcing_continuations_per_source must be positive"
        )
    if sim_batch % continuations_per_source != 0:
        raise ValueError(
            "forcing_continuations_per_source must divide transition.sim_batch"
        )
    source_states_per_batch = sim_batch // continuations_per_source
    pairing_strategy = (
        "multi_continuation"
        if continuations_per_source > 1
        else "one_to_one_control"
    )
    grad_clip_cfg = training.get("grad_clip")
    grad_clip = None if grad_clip_cfg is None else float(grad_clip_cfg)
    save_every = max(
        1, int(transition.get("save_latest_every_updates", 25)),
    )

    online_rngs = OnlineSamplerRNGs.create(seed, device)
    compatibility = {
        "direct_gate_sha256": direct_hash,
        "network_gate_sha256": network_hash,
        "selected_objective": selected_objective,
        "network_gate_budget": (
            None
            if network_gate is None
            else int(network_gate["matched_budget"])
        ),
        "screen_floor_source": (
            "zero" if direct_gate is None else "dense_direct_state_gate"
        ),
        "consecutive_intervals": consecutive,
        "dt": dt,
        "source_state_mode": "online_local_ivp",
        "time_bundle": "lead_bins",
        "include_anchor_interval": True,
        "intervals_per_cell": intervals_per_cell,
        "forcing_continuations_per_source": continuations_per_source,
        "source_states_per_batch": source_states_per_batch,
        "pairing_strategy": pairing_strategy,
        "total_updates": total_updates,
        "updates_per_epoch": updates_per_epoch,
        "production_screen": copy.deepcopy(screen_cfg),
        "fast_manifest_hash": fast_hash,
        "optimizer": training.get("optimizer"),
        "scheduler": copy.deepcopy(training.get("scheduler")),
        "causal": causal_cfg,
        "warmup": warmup,
        "gradnorm_mode": (
            "raw_multi_term" if raw_gradnorm is not None
            else "inert_single_term" if gradnorm_inert else "disabled"
        ),
        "phase1_diagnostics_schema": 1,
    }
    metrics_path = run_dir / "train_metrics.csv"
    diagnostics_path = run_dir / "diagnostics.csv"
    fields = [
        "update", "loss", "physical_defect_mse", "lr",
        "fast_target_gnrmse", "fast_copy_gnrmse",
        "overall_skill", "long_skill", "copy_skill",
        "forcing_swap_rms_K", "source_swap_rms_K",
        "forcing_response_ratio", "source_departure_ratio",
        "low_frequency_long_lead_drift",
    ]
    designed_probe_fields = [
        "correct_forcing_rmse_K",
        "zero_forcing_rmse_K",
        "shuffled_forcing_rmse_K",
        "forcing_permutation_gap_K",
        "designed_response_gain",
        "designed_response_cosine",
        "predicted_response_rms_K",
        "true_response_rms_K",
    ]
    diagnostic_fields = [
        "update",
        "source_encoder_grad_norm",
        "forcing_encoder_grad_norm",
        "forcing_context_grad_norm",
        "forcing_branch_grad_norm",
        "forcing_to_source_grad_ratio",
        "forcing_parameter_displacement_l2",
        "forcing_parameter_relative_displacement",
        "local_forcing_active_fraction",
        "history_forcing_active_fraction",
        "active_forcing_objective",
        "inactive_forcing_objective",
        "boundary_defect_mse",
        "interior_defect_mse",
        "boundary_to_interior_defect_ratio",
        *designed_probe_fields,
    ]
    completed_updates = 0
    best_metric = float("inf")
    best_update = None
    best_validation = None
    screen_records: dict[int, dict[str, float]] = {}
    active_descriptors = None
    active_source_steps = None
    active_source_bins = None
    active_intervals = None
    source_coverage = None
    collapse_detected_at_500 = False
    latest_counterfactual = {
        "in_window_forcing_swap_rms_K": float("nan"),
        "source_state_swap_rms_K": float("nan"),
        "forcing_response_ratio": float("nan"),
        "source_departure_ratio": float("nan"),
    }
    latest_designed_probe = {
        key: float("nan") for key in designed_probe_fields
    }
    resuming = latest_path.exists() and (
        not complete_path.exists() or extend_completed
    )
    if resuming:
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False,
        )
        if checkpoint["resume_config"] != compatibility:
            raise ValueError("incompatible physics transition resume configuration")
        if checkpoint.get("data_signature") != data_signature:
            raise ValueError("physics transition dataset differs from checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if raw_gradnorm is not None:
            raw_gradnorm.load_state_dict(checkpoint["gradnorm_state"])
        online_rngs.load_state_dict(checkpoint["online_rng_state"])
        completed_updates = int(checkpoint["completed_updates"])
        best_metric = float(checkpoint["best_metric"])
        best_update = checkpoint.get("best_update")
        best_validation = copy.deepcopy(checkpoint.get("best_validation"))
        screen_records = {
            int(key): value
            for key, value in checkpoint.get("screen_records", {}).items()
        }
        active_descriptors = checkpoint.get("active_descriptors")
        active_source_steps = checkpoint.get("active_source_steps")
        active_source_bins = checkpoint.get("active_source_bins")
        packed_intervals = checkpoint.get("active_intervals")
        if packed_intervals is not None:
            active_intervals = {
                key: torch.as_tensor(value)
                for key, value in packed_intervals.items()
            }
        source_coverage = checkpoint.get("source_coverage")
        collapse_detected_at_500 = bool(
            checkpoint.get("collapse_detected_at_500", False)
        )
        latest_counterfactual = copy.deepcopy(
            checkpoint.get("latest_counterfactual", latest_counterfactual)
        )
        latest_designed_probe = copy.deepcopy(
            checkpoint.get("latest_designed_probe", latest_designed_probe)
        )
        if not diagnostics_path.exists():
            with diagnostics_path.open("w", newline="") as stream:
                csv.DictWriter(
                    stream, fieldnames=diagnostic_fields,
                ).writeheader()
    else:
        with metrics_path.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
        with diagnostics_path.open("w", newline="") as stream:
            csv.DictWriter(
                stream, fieldnames=diagnostic_fields,
            ).writeheader()

    def checkpoint_payload() -> dict[str, Any]:
        return {
            "objective": "physics_only_transition",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "gradnorm_state": (
                None if raw_gradnorm is None else raw_gradnorm.state_dict()
            ),
            "mu_global": mu,
            "sigma_global": sigma,
            "config": config,
            "resume_config": compatibility,
            "data_signature": data_signature,
            "completed_updates": completed_updates,
            "best_metric": best_metric,
            "best_update": best_update,
            "best_validation": copy.deepcopy(best_validation),
            "screen_records": copy.deepcopy(screen_records),
            "online_rng_state": online_rngs.state_dict(),
            "active_descriptors": copy.deepcopy(active_descriptors),
            "active_source_steps": active_source_steps,
            "active_source_bins": active_source_bins,
            "active_intervals": (
                None if active_intervals is None
                else {
                    key: value.detach().cpu()
                    for key, value in active_intervals.items()
                }
            ),
            "source_coverage": copy.deepcopy(source_coverage),
            "collapse_detected_at_500": collapse_detected_at_500,
            "latest_counterfactual": copy.deepcopy(latest_counterfactual),
            "latest_designed_probe": copy.deepcopy(latest_designed_probe),
            "provenance": {
                "optimization_uses_solution_fields": False,
                "normalization_uses_training_trajectories": True,
                "checkpoint_selection_uses_validation_targets": True,
                "direct_state_gate_uses_fv_reference_solutions": (
                    direct_gate is not None
                ),
                "network_gate_uses_supervised_capacity_baseline": (
                    network_gate is not None
                ),
                "diagnostics_use_fv_reference_solutions": True,
                "physics_only_claim_scope": "optimization_objective",
            },
        }

    grids = {
        "X": X,
        "Y": Y,
        "x_grid": x_np,
        "y_grid": y_np,
    }
    time_cfg = {
        "dt": dt,
        "t_final": t_final,
        "T_right": 300.0,
        "a": float(x_np[0]),
        "b": float(x_np[-1]),
        "c": float(y_np[0]),
        "d": float(y_np[-1]),
        "t_on": float(forcing_cfg.get("t_on", 0.0)),
        "t_off": float(forcing_cfg.get("t_off", 0.2)),
        "phase": float(forcing_cfg.get("phase", 0.0)),
        "tukey_alpha": float(forcing_cfg.get("tukey_alpha", 0.5)),
    }
    source_edges = transition.get(
        "source_edges", [0.0, 0.075, 0.15, 0.225, 0.30],
    )
    lead_edges = transition.get(
        "lead_edges", [0.0, 0.05, 0.10, 0.20, 0.30],
    )
    designed_probe = _build_transition_forcing_probe(
        problem,
        x_grid=x_np,
        y_grid=y_np,
        dt=dt,
        t_final=t_final,
        t_ramp=t_ramp,
        time_cfg=time_cfg,
        mu=mu,
        sigma=sigma,
    )
    last_loss = float("nan")
    last_defect = float("nan")
    while completed_updates < total_updates:
        should_resample = (
            active_descriptors is None
            or _should_resample_online_batch(
                completed_updates,
                int(warmup["steps"]),
                int(warmup["resample_every"]),
            )
        )
        if should_resample:
            active_descriptors = _sample_transition_multicontinuation_descriptors(
                problem,
                online_rngs,
                sim_batch,
                continuations_per_source,
                grids,
                time_cfg,
            )
            source_steps, source_bins = (
                sample_transition_source_steps(
                    online_rngs.numpy["fv_intervals"],
                    batch_size=source_states_per_batch,
                    dt=dt,
                    t_final=t_final,
                    source_edges=source_edges,
                )
            )
            source_intervals = sample_transition_physics_intervals(
                online_rngs.numpy["fv_intervals"],
                source_steps=source_steps,
                dt=dt,
                t_final=t_final,
                lead_edges=lead_edges,
                include_anchor_interval=True,
                intervals_per_cell=intervals_per_cell,
                consecutive_intervals=consecutive,
            )
            active_source_steps = np.repeat(
                source_steps, continuations_per_source,
            )
            active_source_bins = np.repeat(
                source_bins, continuations_per_source,
            )
            active_intervals = _repeat_transition_intervals_by_continuation(
                source_intervals, continuations_per_source,
            )
        records = _materialize_online_records(
            active_descriptors, X, Y, T_right=300.0, b=float(x_np[-1]),
        )
        physical_source, normalized, _ = _normalized_online_ic_buffer(
            records, mu, sigma,
        )
        if source_coverage is None:
            source_coverage = _transition_source_coverage(
                physical_source[::continuations_per_source],
                data,
                fast_records,
            )
        source_fields = torch.from_numpy(normalized).unsqueeze(1).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        max_lead = transition_curriculum_max_lead(
            completed_updates, total_updates, t_final,
        )
        interval_mask = (
            (active_intervals["start_step"] + 1).to(dtype=torch.float64)
            * float(dt)
            <= max_lead + 1.0e-12
        )
        training_intervals = {
            key: value[interval_mask]
            for key, value in active_intervals.items()
        }
        result = forcing_transition_physics_loss(
            model=model,
            source_fields=source_fields,
            params=records,
            source_steps=np.asarray(active_source_steps, dtype=np.int64),
            source_bins=np.asarray(active_source_bins, dtype=np.int64),
            intervals=training_intervals,
            x_grid=x_grid,
            y_grid=y_grid,
            y_img=y_img,
            nt_img=nt_img,
            a_ref=a_ref,
            t_ramp=t_ramp,
            t_final=t_final,
            dt=dt,
            sigma_global=sigma,
            right_value=right_value,
            objective=selected_objective,
            causal_epsilon=causal_epsilon,
            query_chunk=query_chunk,
        )
        if not bool(result["finite"]):
            _atomic_torch_save(checkpoint_payload(), run_dir / "failed_batch.pt")
            raise FloatingPointError("non-finite physics transition objective")
        if raw_gradnorm is not None:
            raw_terms = result["raw_terms"]
            multipliers = raw_gradnorm.maybe_update(
                raw_terms, model.parameters(),
            )
            loss = sum(
                float(multipliers[name]) * raw_terms[name]
                for name in raw_terms
            )
        else:
            loss = result["loss"]
        loss.backward()
        if not all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            _atomic_torch_save(checkpoint_payload(), run_dir / "failed_batch.pt")
            raise FloatingPointError("non-finite physics transition gradient")
        do_counterfactual = (
            (completed_updates + 1) % anti_collapse_every == 0
            or completed_updates + 1 == total_updates
        )
        if do_counterfactual:
            source_grad_norm = _transition_parameter_group_norm(
                model, source_prefixes, gradients=True,
            )
            forcing_encoder_grad_norm = _transition_parameter_group_norm(
                model, ("forcing_encoder.",), gradients=True,
            )
            forcing_context_grad_norm = _transition_parameter_group_norm(
                model, ("forcing_context_projection.",), gradients=True,
            )
            forcing_grad_norm = _transition_parameter_group_norm(
                model, forcing_prefixes, gradients=True,
            )
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip,
            )
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        completed_updates += 1
        if completed_updates % updates_per_epoch == 0:
            _advance_scheduler(scheduler, unit="epoch", successful_updates=1)
        last_loss = float(loss.detach().cpu())
        last_defect = float(result["physical_defect_mse"].detach().cpu())

        if do_counterfactual:
            model.eval()
            latest_counterfactual = transition_counterfactual_diagnostics(
                model,
                data,
                sim_params,
                fast_records,
                y_img=y_img,
                nt_img=nt_img,
                a_ref=a_ref,
                t_ramp=t_ramp,
                device=device,
                query_chunk=query_chunk,
            )
            latest_designed_probe = transition_designed_forcing_diagnostics(
                model,
                designed_probe,
                x_grid=x_grid,
                y_grid=y_grid,
                y_img=y_img,
                nt_img=nt_img,
                a_ref=a_ref,
                t_ramp=t_ramp,
                t_final=t_final,
                sigma=sigma,
                device=device,
                query_chunk=query_chunk,
            )
            displacement, relative_displacement = (
                _transition_parameter_displacement(
                    model, initial_forcing_parameters,
                )
            )
            boundary_defect = float(
                result["boundary_defect_mse"].detach().cpu()
            )
            interior_defect = float(
                result["interior_defect_mse"].detach().cpu()
            )
            diagnostic_row = {
                "update": completed_updates,
                "source_encoder_grad_norm": source_grad_norm,
                "forcing_encoder_grad_norm": forcing_encoder_grad_norm,
                "forcing_context_grad_norm": forcing_context_grad_norm,
                "forcing_branch_grad_norm": forcing_grad_norm,
                "forcing_to_source_grad_ratio": (
                    forcing_grad_norm
                    / max(source_grad_norm, np.finfo(float).tiny)
                ),
                "forcing_parameter_displacement_l2": displacement,
                "forcing_parameter_relative_displacement": (
                    relative_displacement
                ),
                "local_forcing_active_fraction": float(
                    result["local_forcing_active_fraction"].detach().cpu()
                ),
                "history_forcing_active_fraction": float(
                    result["history_forcing_active_fraction"].detach().cpu()
                ),
                "active_forcing_objective": float(
                    result["active_forcing_objective"].detach().cpu()
                ),
                "inactive_forcing_objective": float(
                    result["inactive_forcing_objective"].detach().cpu()
                ),
                "boundary_defect_mse": boundary_defect,
                "interior_defect_mse": interior_defect,
                "boundary_to_interior_defect_ratio": (
                    boundary_defect
                    / max(interior_defect, np.finfo(float).tiny)
                ),
                **latest_designed_probe,
            }
            with diagnostics_path.open("a", newline="") as stream:
                csv.DictWriter(
                    stream, fieldnames=diagnostic_fields,
                ).writerow(diagnostic_row)
        do_validation = (
            completed_updates % validate_every == 0
            or completed_updates == total_updates
        )
        validation = None
        screen_metrics = None
        if do_validation:
            model.eval()
            validation = validate_forcing_transition(
                model,
                data,
                sim_params,
                fast_records,
                y_img=y_img,
                nt_img=nt_img,
                a_ref=a_ref,
                t_ramp=t_ramp,
                device=device,
                sim_batch=val_batch,
                query_chunk=query_chunk,
                signal_floor_fraction=float(
                    transition.get("increment_signal_floor_sigma", 0.01)
                ),
            )
            screen_metrics = _transition_validation_screen_metrics(
                validation,
                direct_gate,
                sigma_global=sigma,
                anti_collapse_passed=bool(
                    latest_counterfactual["forcing_response_ratio"] is not None
                    and latest_counterfactual["source_departure_ratio"] is not None
                    and float(latest_counterfactual[
                        "forcing_response_ratio"
                    ]) >= 0.10
                    and float(latest_counterfactual[
                        "source_departure_ratio"
                    ]) >= 0.10
                ),
                defect_passed=math.isfinite(last_defect),
            )
            screen_records[completed_updates] = {
                "overall_skill": float(screen_metrics["overall_skill"]),
                "long_skill": float(screen_metrics["long_skill"]),
            }
            metric = float(validation["macro_cell_gnrmse"])
            if metric < best_metric:
                best_metric = metric
                best_update = completed_updates
                best_validation = copy.deepcopy(validation)
                _atomic_torch_save(checkpoint_payload(), best_path)
            if completed_updates == collapse_update:
                collapse_detected_at_500 = bool(
                    screen_metrics["overall_skill"] <= 0.0
                    or screen_metrics["long_skill"] <= 0.0
                    or validation.get("copy_skill") is None
                    or float(validation["copy_skill"]) <= 0.0
                    or not math.isfinite(
                        float(latest_counterfactual[
                            "in_window_forcing_swap_rms_K"
                        ])
                    )
                    or float(latest_counterfactual[
                        "in_window_forcing_swap_rms_K"
                    ]) <= 0.0
                    or latest_counterfactual["forcing_response_ratio"] is None
                    or float(latest_counterfactual[
                        "forcing_response_ratio"
                    ]) < 0.10
                    or latest_counterfactual["source_departure_ratio"] is None
                    or float(latest_counterfactual[
                        "source_departure_ratio"
                    ]) < 0.10
                )
        row = {
            "update": completed_updates,
            "loss": last_loss,
            "physical_defect_mse": last_defect,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "fast_target_gnrmse": (
                "" if validation is None else validation["target_gnrmse"]
            ),
            "fast_copy_gnrmse": (
                "" if validation is None else validation["copy_gnrmse"]
            ),
            "overall_skill": (
                "" if screen_metrics is None
                else screen_metrics["overall_skill"]
            ),
            "long_skill": (
                "" if screen_metrics is None else screen_metrics["long_skill"]
            ),
            "copy_skill": (
                "" if validation is None else validation["copy_skill"]
            ),
            "forcing_swap_rms_K": latest_counterfactual[
                "in_window_forcing_swap_rms_K"
            ],
            "source_swap_rms_K": latest_counterfactual[
                "source_state_swap_rms_K"
            ],
            "forcing_response_ratio": latest_counterfactual[
                "forcing_response_ratio"
            ],
            "source_departure_ratio": latest_counterfactual[
                "source_departure_ratio"
            ],
            "low_frequency_long_lead_drift": (
                "" if screen_metrics is None
                else screen_metrics["low_frequency_long_lead_drift"]
            ),
        }
        with metrics_path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        print(
            f"Update {completed_updates}: loss={last_loss:.6e}",
            flush=True,
        )
        if validation is not None:
            print(
                f"Validation for update {completed_updates}: "
                f"val_gnrmse="
                f"{100.0 * float(validation['macro_cell_gnrmse']):.4f}% "
                f"val_rmse_K={float(validation['target_rmse_K']):.4f}K "
                f"(best={100.0 * best_metric:.4f}%)",
                flush=True,
            )
        if (
            completed_updates % save_every == 0
            or completed_updates == total_updates
        ):
            _atomic_torch_save(checkpoint_payload(), latest_path)
        if collapse_detected_at_500:
            _atomic_torch_save(checkpoint_payload(), latest_path)
            break

    screen_decision = None
    if screen_enabled:
        if collapse_detected_at_500:
            screen_decision = {
                "passed": False,
                "reason": "collapse_at_update_500",
                "finite": True,
            }
        else:
            screen_decision = transition_production_screen_decision(
                screen_records,
                minimum_improvement=minimum_skill_improvement,
            )
    if not best_path.exists():
        _atomic_torch_save(checkpoint_payload(), best_path)
    best_checkpoint = torch.load(
        best_path, map_location="cpu", weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state"])
    final_payload = checkpoint_payload()
    final_payload["screen_decision"] = screen_decision
    _atomic_torch_save(final_payload, final_path)
    unexpected_failure = bool(
        screen_decision is not None and not screen_decision["passed"]
    )
    summary = {
        "seed": seed,
        "stage": "production_distribution_screen",
        "objective": "physics_only_transition",
        "selected_physics_objective": selected_objective,
        "completed_updates": completed_updates,
        "best_fast_validation_metric": best_metric,
        "best_checkpoint_update": best_update,
        "best_validation": best_validation,
        "screen_decision": screen_decision,
        "unexpected_screen_failure": unexpected_failure,
        "collapse_detected_at_500": collapse_detected_at_500,
        "next_action": (
            "run_20x20_16_source_mini_operator"
            if unexpected_failure else "production_screen_passed"
        ),
        "source_distribution_coverage": source_coverage,
        "counterfactual_diagnostics": latest_counterfactual,
        "designed_forcing_diagnostics": latest_designed_probe,
        "diagnostics_path": str(diagnostics_path),
        "phase1_diagnostics_schema": 1,
        "direct_state_gate_sha256": direct_hash,
        "network_gate_sha256": network_hash,
        "network_gate_budget": (
            None
            if network_gate is None
            else int(network_gate["matched_budget"])
        ),
        "consecutive_intervals": consecutive,
        "forcing_continuations_per_source": continuations_per_source,
        "source_states_per_batch": source_states_per_batch,
        "transition_cases_per_batch": sim_batch,
        "pairing_strategy": pairing_strategy,
        "screen_floor_source": compatibility["screen_floor_source"],
        "gradnorm_mode": compatibility["gradnorm_mode"],
        "optimization_uses_solution_fields": False,
        "normalization_uses_training_trajectories": True,
        "checkpoint_selection_uses_validation_targets": True,
        "direct_state_gate_uses_fv_reference_solutions": direct_gate is not None,
        "network_gate_uses_supervised_capacity_baseline": network_gate is not None,
        "diagnostics_use_fv_reference_solutions": True,
        "physics_only_claim_scope": "optimization_objective",
        "test_set_evaluated": False,
    }
    _atomic_text(json.dumps(summary, indent=2) + "\n", summary_path)
    _atomic_text("complete\n", complete_path)
    return summary


def run_config_seeds_pino(
    config: dict, base_run_dir: Path, seeds: list[int]
) -> dict[str, Any]:
    base_run_dir = Path(base_run_dir)
    # Dispatch on the benchmark: forcing and interfaces use the layered
    # InterfaceCViT paths; the constant-IC single-material forcing benchmark
    # (diffusion_forcing) trains a ForcingCViT on an
    # online-sampled forcing image; the varying-IC single-slab benchmark
    # (diffusion_forcing_single) defaults to the two-branch ForcingICCViT and
    # exposes the causal transition variant as an explicit opt-in; every other
    # benchmark uses the IC-conditioned diffusion CViT path.
    bench = str(config.get("benchmark", {}).get("name", "diffusion"))
    if bench == "diffusion_forcing_single":
        declared = (config.get("training", {}).get("pino", {}) or {}).get("variant")
        declared = "forcing_ic" if declared is None else str(declared)
        if declared not in {"forcing_ic", "forcing_transition"}:
            raise ValueError(
                "benchmark=diffusion_forcing_single requires "
                "training.pino.variant='forcing_ic' or "
                f"'forcing_transition', got {declared!r}."
            )
        if declared == "forcing_transition":
            forcing_ic_mode = "transition"
            transition_objective = str(
                (
                    config["training"]["pino"].get("transition", {}) or {}
                ).get("objective", "supervised")
            )
            if transition_objective not in {"supervised", "physics_only"}:
                raise ValueError(
                    "training.pino.transition.objective must be "
                    "'supervised' or 'physics_only'"
                )
        else:
            forcing_ic_mode, _ = _forcing_ic_training_mode(
                config["training"]["pino"],
            )
            transition_objective = None
    else:
        forcing_ic_mode = None
        transition_objective = None
    pino_cfg = (config.get("training", {}).get("pino", {}) or {})
    interfaces_mode = str(pino_cfg.get("mode", "collapse"))
    if bench == "interfaces" and interfaces_mode not in ("collapse", "one_step"):
        raise ValueError(
            "benchmark=interfaces training.pino.mode must be 'collapse' "
            f"(full-trajectory decoder) or 'one_step' but got {interfaces_mode!r}."
        )
    if bench == "interfaces":
        interfaces_runner = (
            run_one_seed_interfaces_one_step_pino
            if interfaces_mode == "one_step"
            else run_one_seed_interfaces_pino
        )
        if interfaces_mode == "one_step":
            config = _resolve_interfaces_one_step_config(config)
    else:
        interfaces_runner = run_one_seed_interfaces_pino
    runner = (
        run_one_seed_forcing_interface_pino if bench == "forcing"
        else interfaces_runner if bench == "interfaces"
        else (
            (
                run_one_seed_forcing_transition_physics
                if transition_objective == "physics_only"
                else run_one_seed_forcing_transition
            )
            if forcing_ic_mode == "transition"
            else (
                run_one_seed_forcing_ic_supervised
                if forcing_ic_mode == "supervised"
                else run_one_seed_forcing_ic_pino
            )
        ) if bench == "diffusion_forcing_single"
        else run_one_seed_forcing_pino if bench == "diffusion_forcing"
        else run_one_seed_pino
    )
    results = {}
    for seed in seeds:
        seed_dir = base_run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config), seed_dir / "config_used.yaml")
        results[str(seed)] = runner(config, int(seed), seed_dir)
    return {"run_dir": str(base_run_dir), "seeds": results}


__all__ = [
    "OnlineSamplerRNGs",
    "HybridForcingRNGs",
    "sample_collocation",
    "sample_forcing_params",
    "build_forcing_image",
    "build_forcing_transition_image",
    "load_verified_transition_gate",
    "transition_screen_skill",
    "transition_consecutive_interval_trigger",
    "transition_production_screen_decision",
    "sample_transition_source_steps",
    "transition_curriculum_max_lead",
    "sample_transition_physics_intervals",
    "forcing_transition_physics_loss",
    "TransitionPairSchedule",
    "transition_epoch_order",
    "transition_manifest_hash",
    "write_transition_manifest",
    "left_wall_qL",
    "load_diffusion_data",
    "build_ic_batch",
    "pino_losses",
    "_ic_loss",
    "_curriculum_weights",
    "_causal_weights",
    "_causal_residual_loss",
    "_bin_residual",
    "_term_grad_norms",
    "build_gradnorm",
    "validate_rel_l2",
    "validate_forcing_gnrmse",
    "validate_forcing_ic_gnrmse",
    "validate_forcing_ic_supervised_dataset",
    "build_cvit",
    "forcing_ic_supervised_data_loss",
    "transition_pair_metrics",
    "aggregate_transition_metrics",
    "validate_forcing_transition",
    "transition_counterfactual_diagnostics",
    "load_interface_cvit_checkpoint",
    "run_one_seed_forcing_interface_pino",
    "run_one_seed_pino",
    "run_one_seed_forcing_pino",
    "run_one_seed_forcing_ic_pino",
    "run_one_seed_forcing_ic_supervised",
    "run_one_seed_forcing_transition",
    "run_one_seed_forcing_transition_physics",
    "run_one_seed_interfaces_pino",
    "run_one_seed_interfaces_one_step_pino",
    "validate_interfaces_one_step_gnrmse",
    "run_config_seeds_pino",
]
