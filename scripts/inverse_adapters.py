from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from problems.forcing import RC_RANGE
from problems.forcing import ForcingProblem
from problems.registry import get_problem
from problems.source_itr_sin import RC_Y_CHANNEL
from src.physics.fv_solver_2d import Layer2D
from src.physics.internal_source import (
    RC_MIN,
    RC_SIN_RANGES,
    R_PEAK_MAX,
    equivalent_scalar_resistance,
    make_rc_sin_profile,
)


SIN_PARAM_NAMES = ("R_base", "A")
RELATIVE_ERROR_FLOOR_FRACTION = 1e-6


def relative_error_percent(estimate: float, truth: float, parameter_range: float) -> float:
    """Percentage error, undefined within 1e-6 of the admissible range of zero."""
    if abs(truth) <= RELATIVE_ERROR_FLOOR_FRACTION * parameter_range:
        return float("nan")
    return 100.0 * abs(estimate - truth) / abs(truth)


_GEN_DOMAIN = dict(a=0.0, b=1.0, c=0.0, d=1.0)
_GEN_LAM_TARGET = 0.8
_GEN_FLUX_F = 0.0
_GEN_FLUX_A = 0.0
_GEN_T_ON = 0.0
_GEN_T_OFF = 0.2
_GEN_PHASE = 0.0
_GEN_TUKEY_ALPHA = 0.5


def _as_theta_1d(theta: torch.Tensor, theta_dim: int) -> torch.Tensor:
    theta = torch.as_tensor(theta)
    if theta.shape != (theta_dim,):
        raise ValueError(f"Expected theta shape ({theta_dim},), got {tuple(theta.shape)}")
    return theta


def _to_numpy_theta(theta, theta_dim: int) -> np.ndarray:
    if torch.is_tensor(theta):
        arr = theta.detach().cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(theta, dtype=np.float64)
    if arr.shape != (theta_dim,):
        raise ValueError(f"Expected theta shape ({theta_dim},), got {arr.shape}")
    return arr


def _dtype_eps(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return float(torch.finfo(dtype).eps)
    return float(np.finfo(np.float32).eps)


def _logit(p: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    if eps is None:
        eps = _dtype_eps(p.dtype)
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def _lhs_unit_samples(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, n + 1)
    lo = cut[:n]
    hi = cut[1:]
    pts = lo[:, None] + rng.uniform(size=(n, d)) * (hi - lo)[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def _snap_t_final(ds) -> float:
    if ds.dt is None:
        raise ValueError(
            "FV refinement needs the solver dt; dt.npy is missing from the data dir."
        )
    dt = float(ds.dt)
    n_steps = int(round(float(ds.t_final) / dt))
    return n_steps * dt


class InverseAdapter(ABC):
    benchmark: str
    theta_dim: int
    param_names: tuple[str, ...]

    @classmethod
    def from_config(cls, config: dict) -> "InverseAdapter":
        bench = config.get("benchmark", {})
        name = bench.get("name", "forcing") if isinstance(bench, dict) else str(bench)
        adapters = {
            "source_itr_sin": SourceItrSinAdapter,
            "forcing": ForcingAdapter,
            "forcing_itr_sin": ForcingItrSinAdapter,
        }
        adapter_cls = adapters.get(name)
        if adapter_cls is not None:
            return adapter_cls()
        raise ValueError(
            f"Unsupported inverse benchmark {name!r}; expected one of "
            f"{sorted(adapters)}."
        )


    @property
    def supports_equivalent_scalar(self) -> bool:
        return False

    @abstractmethod
    def theta_from_unconstrained(self, u: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def unconstrained_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        ...

    def lhs_starts_unconstrained(
        self, n_starts: int, rng: np.random.Generator, eps: float = 1e-3
    ) -> np.ndarray:
        f = _lhs_unit_samples(n_starts, self.theta_dim, rng)
        f = np.clip(f, eps, 1.0 - eps)
        u = np.log(f / (1.0 - f))
        return u.astype(np.float32)

    @abstractmethod
    def theta_from_sim_params(
        self, sim_params: dict, *, dtype: torch.dtype = torch.float32, device=None
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def cond_slice_indices(self) -> tuple[int, int]:
        ...

    @abstractmethod
    def cond_slice_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def spatial_channel_index(self) -> Optional[int]:
        ...

    @abstractmethod
    def spatial_channel_from_theta(
        self, theta: torch.Tensor, y_grid: torch.Tensor, Nx: int
    ) -> Optional[torch.Tensor]:
        ...

    @abstractmethod
    def inject_theta_into_sim_params(self, sim_params: dict, theta) -> dict:
        ...

    @abstractmethod
    def fv_base_kwargs(self, ds, grid_size: Optional[int] = None) -> dict:
        ...

    def fv_initial_condition(self, ds, sim_params: dict, solver) -> np.ndarray:
        T0 = np.asarray(sim_params["T0"], dtype=np.float64)
        shape = (int(solver.Nx), int(solver.Ny))
        if T0.shape == shape:
            return T0
        if np.allclose(T0, T0.flat[0]):
            return np.full(shape, float(T0.flat[0]), dtype=np.float64)
        raise ValueError(
            "Refined FV inversion requires a grid-independent initial-condition "
            "builder for nonuniform T0."
        )

    def equivalent_scalar_values(
        self, theta: torch.Tensor, y_grid: torch.Tensor
    ) -> tuple[float, float]:
        raise ValueError(f"{type(self).__name__} has no equivalent-scalar diagnostic.")

    def build_scalar_fv_solver(
        self, ds, sim_params: dict, resistance: float, base_kwargs: dict
    ):
        raise ValueError(f"{type(self).__name__} has no scalar FV comparison.")

    def build_fv_solver(self, ds, sim_params: dict, theta):
        updated = self.inject_theta_into_sim_params(sim_params, theta)
        return ds.problem.configure_solver(updated, self.fv_base_kwargs(ds))

    @abstractmethod
    def validate_dataset(self, ds) -> None:
        ...

    def validate_model(self, loaded_model) -> None:
        config = loaded_model.config
        bench = config.get("benchmark", {})
        name = bench.get("name", "forcing") if isinstance(bench, dict) else str(bench)
        if name != self.benchmark:
            raise ValueError(
                f"Adapter {type(self).__name__} expected checkpoint benchmark "
                f"{self.benchmark!r}, got {name!r}."
            )
        representation = (
            bench.get("representation")
            if isinstance(bench, dict)
            else config.get("representation", {}).get("name")
        )
        representation = representation or "temporal_encoder"
        expected = get_problem(self.benchmark, representation).dims
        dims = loaded_model.dims
        if dims.in_channels != expected.in_channels:
            raise ValueError(
                f"Checkpoint dims.in_channels={dims.in_channels}, expected "
                f"{expected.in_channels} for {self.benchmark}/{representation}."
            )
        if dims.cond_static_dim != expected.cond_static_dim:
            raise ValueError(
                f"Checkpoint dims.cond_static_dim={dims.cond_static_dim}, expected "
                f"{expected.cond_static_dim} for {self.benchmark}/{representation}."
            )
        model = loaded_model.model
        checks = {
            "in_channels": expected.in_channels,
            "cond_static_dim": expected.cond_static_dim,
            "s_y_channel": expected.s_y_channel,
        }
        for attr, exp in checks.items():
            got = getattr(model, attr, None)
            if got != exp:
                raise ValueError(
                    f"Model {attr}={got!r}, expected {exp!r} for "
                    f"{self.benchmark}/{representation}."
                )
        ch = self.spatial_channel_index()
        if ch is not None and ch >= expected.in_channels:
            raise ValueError(
                f"Adapter spatial channel {ch} is outside model input width "
                f"{expected.in_channels}."
            )

    @abstractmethod
    def profile_bounds(self, param_index: int) -> tuple[float, float]:
        ...

    @property
    def param_scales(self) -> np.ndarray:
        """Physical range (hi - lo) per parameter, the normalized-SVD scaling.

        Derived from ``profile_bounds`` (the source of truth) so the artifact
        records the exact coordinate system used to build
        ``J_scaled = J . diag(param_scales)`` for unit-invariant identifiability.
        """
        scales = [
            float(self.profile_bounds(i)[1] - self.profile_bounds(i)[0])
            for i in range(self.theta_dim)
        ]
        return np.asarray(scales, dtype=np.float64)

    @abstractmethod
    def theta_profile(
        self, u: torch.Tensor, fixed_index: int, fixed_value: float
    ) -> torch.Tensor:
        ...


    @abstractmethod
    def summarize_result(self, result, y_grid: torch.Tensor) -> dict:
        ...

    @abstractmethod
    def sensitivity_summary(self, report: dict) -> dict:
        ...

    @abstractmethod
    def fv_refine_summary(self, res, obs, y_grid: torch.Tensor) -> dict:
        ...

    @abstractmethod
    def laplace_summary(self, spec: dict) -> dict:
        ...


class ForcingAdapter(InverseAdapter):
    benchmark = "forcing"
    theta_dim = 1
    param_names = ("R_c",)

    def theta_from_unconstrained(self, u: torch.Tensor) -> torch.Tensor:
        lo, hi = RC_RANGE
        s = torch.sigmoid(u)
        R_c = lo + (hi - lo) * s[..., 0]
        return R_c.unsqueeze(-1)

    def unconstrained_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        lo, hi = RC_RANGE
        p = (theta[..., 0] - lo) / (hi - lo)
        return _logit(p).unsqueeze(-1)

    def theta_from_sim_params(
        self, sim_params: dict, *, dtype: torch.dtype = torch.float32, device=None
    ) -> torch.Tensor:
        return torch.tensor([float(sim_params["R_c"])], dtype=dtype, device=device)

    def cond_slice_indices(self) -> tuple[int, int]:
        return (1, 2)

    def cond_slice_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        lo, hi = RC_RANGE
        R_c = theta[..., 0]
        return ((R_c - lo) / (hi - lo)).unsqueeze(-1)

    def spatial_channel_index(self) -> Optional[int]:
        return None

    def spatial_channel_from_theta(
        self, theta: torch.Tensor, y_grid: torch.Tensor, Nx: int
    ) -> Optional[torch.Tensor]:
        return None

    def inject_theta_into_sim_params(self, sim_params: dict, theta) -> dict:
        th = _to_numpy_theta(theta, self.theta_dim)
        params = dict(sim_params)
        params["R_c"] = float(th[0])
        return params

    def fv_base_kwargs(self, ds, grid_size: Optional[int] = None) -> dict:
        a, b = _GEN_DOMAIN["a"], _GEN_DOMAIN["b"]
        x_mid = 0.5 * (a + b)
        size = int(grid_size) if grid_size is not None else int(ds.Nx)
        layers = [
            Layer2D(x_left=a, x_right=x_mid, rho=1.0, cp=1.0, k=2.0),
            Layer2D(x_left=x_mid, x_right=b, rho=1.0, cp=1.0, k=1.0),
        ]
        return dict(
            a=a, b=b, c=_GEN_DOMAIN["c"], d=_GEN_DOMAIN["d"],
            Nx=size, Ny=size,
            lam_target=_GEN_LAM_TARGET,
            layers=layers,
            t_final=_snap_t_final(ds),
            flux_f=_GEN_FLUX_F, flux_A=_GEN_FLUX_A,
            t_on=_GEN_T_ON, t_off=_GEN_T_OFF, phase=_GEN_PHASE,
            dt=float(ds.dt), tukey_alpha=_GEN_TUKEY_ALPHA,
            y_grid=np.linspace(
                _GEN_DOMAIN["c"], _GEN_DOMAIN["d"], size, dtype=np.float64
            ),
            ramp_seconds=getattr(ds, "ramp_seconds", None),
        )

    def validate_dataset(self, ds) -> None:
        if getattr(ds.problem, "name", None) != self.benchmark:
            raise ValueError(
                f"ForcingAdapter expected dataset benchmark 'forcing', got "
                f"{getattr(ds.problem, 'name', None)!r}."
            )
        if getattr(ds.problem, "rc_channel_mode", None) is not None:
            raise ValueError("forcing dataset unexpectedly exposes rc_channel_mode.")
        first_sid = int(ds.sim_ids[0])
        sample = ds.problem.build_item(ds, first_sid, 0, min(1, ds.Nt - 1))
        spatial_channels = int(sample["spatial"].shape[-1])
        if ds.problem.dims.in_channels != spatial_channels:
            raise ValueError(
                f"Dataset spatial schema has {spatial_channels} channels, "
                f"expected {ds.problem.dims.in_channels}."
            )
        required = {"R_c", "temporal_family", "temporal_params", "spatial_family", "spatial_params"}
        missing = required - set(dict(ds.sim_params[first_sid]).keys())
        if missing:
            raise ValueError(f"forcing sim_params missing keys: {sorted(missing)}")

    def profile_bounds(self, param_index: int) -> tuple[float, float]:
        if param_index != 0:
            raise ValueError("forcing profile param_index must be 0 for R_c")
        return tuple(RC_RANGE)

    def theta_profile(
        self, u: torch.Tensor, fixed_index: int, fixed_value: float
    ) -> torch.Tensor:
        if fixed_index != 0:
            raise ValueError("forcing profile fixed_index must be 0 for R_c")
        return torch.as_tensor([fixed_value], dtype=u.dtype, device=u.device)


    def summarize_result(self, result, y_grid: torch.Tensor) -> dict:
        out = {"loss": result.loss, "R_c_map": float(result.theta_hat[0])}
        if result.theta_true is not None:
            out["R_c_true"] = float(result.theta_true[0])
            out["R_c_abs_error"] = abs(out["R_c_map"] - out["R_c_true"])
            out["R_c_rel_error_pct"] = relative_error_percent(
                out["R_c_map"], out["R_c_true"], self.param_scales[0]
            )
        return out

    def sensitivity_summary(self, report: dict) -> dict:
        S = report["singular_values"]
        least = report["least_identified_dir"]
        sens = report["param_sensitivity"]
        return {
            "cond_number": report["cond_number"],
            "sv_0": float(S[0]) if len(S) else float("nan"),
            "sens_R_c": float(sens[0]),
            "least_dir_R_c": float(least[0]),
        }

    def fv_refine_summary(self, res, obs, y_grid: torch.Tensor) -> dict:
        out = {
            "fno_resid": res.fno_resid,
            "fv_resid": res.fv_resid,
            "fno_vs_fv_resid": res.fno_vs_fv_resid,
        }
        if res.theta_fv_polish is not None:
            out["fv_polish_resid"] = res.fv_polish_resid
            out["fv_polish_evals"] = res.fv_polish_evals
            out["R_c_fvpolish"] = float(res.theta_fv_polish[0])
            if obs.theta_true is not None:
                out["R_c_fvpolish_abs_error"] = abs(
                    out["R_c_fvpolish"] - float(obs.theta_true[0])
                )
        return out

    def laplace_summary(self, spec: dict) -> dict:
        eig = spec["eigenvalues"]
        least = spec["least_identified_dir"]
        return {
            "laplace_cond": spec["cond_number"],
            "laplace_eig_0": float(eig[0]) if len(eig) else float("nan"),
            "laplace_least_dir_R_c": float(least[0]),
        }


class SourceItrSinAdapter(InverseAdapter):
    """Inverse adapter for theta=(R_base, A) with a dependent amplitude bound.

    Conditioning uses A/(R_PEAK_MAX-RC_MIN), matching the forward model.
    """

    benchmark = "source_itr_sin"
    theta_dim = 2
    param_names = SIN_PARAM_NAMES


    def theta_from_unconstrained(self, u: torch.Tensor) -> torch.Tensor:
        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        s = torch.sigmoid(u)
        R_base = base_lo + (base_hi - base_lo) * s[..., 0]
        A = s[..., 1] * (R_PEAK_MAX - R_base)
        return torch.stack([R_base, A], dim=-1)

    def unconstrained_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        R_base = theta[..., 0]
        A = theta[..., 1]
        p_base = (R_base - base_lo) / (base_hi - base_lo)
        ceil = (R_PEAK_MAX - R_base).clamp_min(_dtype_eps(theta.dtype))
        p_A = A / ceil
        return torch.stack([_logit(p_base), _logit(p_A)], dim=-1)

    def theta_from_sim_params(
        self, sim_params: dict, *, dtype: torch.dtype = torch.float32, device=None
    ) -> torch.Tensor:
        return torch.tensor(
            [float(sim_params["R_c_base"]), float(sim_params["R_c_A"])],
            dtype=dtype,
            device=device,
        )

    def cond_slice_indices(self) -> tuple[int, int]:
        return (1, 3)

    def cond_slice_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        A_lo, A_hi = RC_SIN_RANGES["A"]
        R_base = theta[..., 0]
        A = theta[..., 1]
        R_base_norm = (R_base - base_lo) / (base_hi - base_lo)
        A_norm = (A - A_lo) / (A_hi - A_lo)  # global norm (matches forward)
        return torch.stack([R_base_norm, A_norm], dim=-1)

    def spatial_channel_from_theta(
        self, theta: torch.Tensor, y_grid: torch.Tensor, Nx: int
    ) -> torch.Tensor:
        R_base = theta[..., 0:1]
        A = theta[..., 1:2]
        y = y_grid.to(dtype=theta.dtype, device=theta.device)
        Rc_y = R_base + A * torch.sin(np.pi * y)

        log_min = float(np.log(RC_MIN))
        log_max = float(np.log(R_PEAK_MAX))
        Rc_y_norm = 2.0 * (torch.log(Rc_y) - log_min) / (log_max - log_min) - 1.0
        return Rc_y_norm.unsqueeze(-2).expand(*Rc_y_norm.shape[:-1], Nx, y.shape[0])

    def inject_theta_into_sim_params(self, sim_params: dict, theta) -> dict:
        th = _to_numpy_theta(theta, self.theta_dim)
        params = dict(sim_params)
        params["R_c_base"] = float(th[0])
        params["R_c_A"] = float(th[1])
        params["R_c"] = float(th[0])
        return params

    def validate_dataset(self, ds) -> None:
        if getattr(ds.problem, "name", None) != self.benchmark:
            raise ValueError(
                f"SourceItrSinAdapter expected dataset benchmark "
                f"{self.benchmark!r}, got {getattr(ds.problem, 'name', None)!r}."
            )
        if getattr(ds.problem, "rc_channel_mode", None) != "broadcast":
            raise ValueError(
                f"{self.benchmark} inversion only supports "
                f"rc_channel_mode='broadcast'; got "
                f"{getattr(ds.problem, 'rc_channel_mode', None)!r}."
            )
        first_sid = int(ds.sim_ids[0])
        sample = ds.problem.build_item(ds, first_sid, 0, min(1, ds.Nt - 1))
        spatial_channels = int(sample["spatial"].shape[-1])
        if ds.problem.dims.in_channels != spatial_channels:
            raise ValueError(
                f"Dataset spatial schema has {spatial_channels} channels, "
                f"expected {ds.problem.dims.in_channels}."
            )
        required = {"R_c_base", "R_c_A"}
        missing = required - set(dict(ds.sim_params[first_sid]).keys())
        if missing:
            raise ValueError(
                f"{self.benchmark} sim_params missing keys: {sorted(missing)}"
            )

    def profile_bounds(self, param_index: int) -> tuple[float, float]:
        if param_index == 0:
            return tuple(RC_SIN_RANGES["R_base"])
        if param_index == 1:
            return (0.0, R_PEAK_MAX - RC_MIN)
        raise ValueError(f"profile param_index {param_index} out of range")

    def theta_profile(
        self, u: torch.Tensor, fixed_index: int, fixed_value: float
    ) -> torch.Tensor:
        base_lo, base_hi = RC_SIN_RANGES["R_base"]
        s = torch.sigmoid(u)
        if fixed_index == 1:
            A = torch.as_tensor(fixed_value, dtype=u.dtype, device=u.device)
            base_ceiling = min(base_hi, R_PEAK_MAX - float(fixed_value))
            R_base = base_lo + (base_ceiling - base_lo) * s[..., 0]
        else:
            R_base = (
                torch.as_tensor(fixed_value, dtype=u.dtype, device=u.device)
                if fixed_index == 0
                else base_lo + (base_hi - base_lo) * s[..., 0]
            )
            A = s[..., 1] * (R_PEAK_MAX - R_base)
        return torch.stack([R_base, A], dim=-1)


    def summarize_result(self, result, y_grid: torch.Tensor) -> dict:
        out = {
            "loss": result.loss,
            "R_base_hat": float(result.theta_hat[0]),
            "A_hat": float(result.theta_hat[1]),
        }
        if result.theta_true is not None:
            t = result.theta_true
            out.update(
                {
                    "R_base_true": float(t[0]),
                    "A_true": float(t[1]),
                    "R_base_abserr": abs(float(result.theta_hat[0]) - float(t[0])),
                    "A_abserr": abs(float(result.theta_hat[1]) - float(t[1])),
                }
            )
            for i, name in enumerate(self.param_names):
                out[f"{name}_rel_error_pct"] = relative_error_percent(
                    out[f"{name}_hat"], out[f"{name}_true"], self.param_scales[i]
                )
        return out


    def spatial_channel_index(self) -> Optional[int]:
        return RC_Y_CHANNEL


    def fv_base_kwargs(self, ds, grid_size: Optional[int] = None) -> dict:
        size = int(grid_size) if grid_size is not None else int(ds.Nx)
        y_grid = np.linspace(
            _GEN_DOMAIN["c"], _GEN_DOMAIN["d"], size, dtype=np.float64
        )
        return dict(
            a=_GEN_DOMAIN["a"], b=_GEN_DOMAIN["b"],
            c=_GEN_DOMAIN["c"], d=_GEN_DOMAIN["d"],
            Nx=size, Ny=size,
            lam_target=_GEN_LAM_TARGET,
            t_final=_snap_t_final(ds),
            flux_f=_GEN_FLUX_F, flux_A=_GEN_FLUX_A,
            t_on=_GEN_T_ON, phase=_GEN_PHASE,
            dt=float(ds.dt), tukey_alpha=_GEN_TUKEY_ALPHA,
            y_grid=y_grid,
        )


    def sensitivity_summary(self, report: dict) -> dict:
        S = report["singular_values"]
        least = report["least_identified_dir"]
        sens = report["param_sensitivity"]
        out = {
            "cond_number": report["cond_number"],
        }
        for i, name in enumerate(self.param_names):
            out[f"sv_{i}"] = float(S[i]) if i < len(S) else float("nan")
            out[f"sens_{name}"] = float(sens[i])
            out[f"least_dir_{name}"] = float(least[i])
        return out


    def fv_refine_summary(self, res, obs, y_grid: torch.Tensor) -> dict:
        out = {
            "fno_resid": res.fno_resid,
            "fv_resid": res.fv_resid,
            "fno_vs_fv_resid": res.fno_vs_fv_resid,
        }
        if res.theta_fv_polish is not None:
            tp = res.theta_fv_polish
            out["fv_polish_resid"] = res.fv_polish_resid
            out["fv_polish_evals"] = res.fv_polish_evals
            for i, name in enumerate(self.param_names):
                out[f"{name}_fvpolish"] = float(tp[i])
            if obs.theta_true is not None:
                t = obs.theta_true
                for i, name in enumerate(self.param_names):
                    out[f"{name}_fvpolish_abserr"] = abs(float(tp[i]) - float(t[i]))
        return out


    def laplace_summary(self, spec: dict) -> dict:
        eig = spec["eigenvalues"]
        least = spec["least_identified_dir"]
        out = {
            "laplace_cond": spec["cond_number"],
        }
        for i in range(len(eig)):
            out[f"laplace_eig_{i}"] = float(eig[i])
        for i, nm in enumerate(self.param_names):
            out[f"laplace_least_dir_{nm}"] = float(least[i])
        return out


class ForcingItrSinAdapter(SourceItrSinAdapter):
    """Inverse adapter for the sinusoid interface-resistance forcing benchmark.

    Mirrors :class:`ForcingItrSinAdapter` for the 2-param sinusoid: reuses the
    forcing FV domain/layers, exposes the equivalent-scalar diagnostic on the
    sin profile, and builds the scalar comparison solver through
    :class:`ForcingProblem`.
    """

    benchmark = "forcing_itr_sin"

    @property
    def supports_equivalent_scalar(self) -> bool:
        return True

    def fv_base_kwargs(self, ds, grid_size: Optional[int] = None) -> dict:
        return ForcingAdapter().fv_base_kwargs(ds, grid_size=grid_size)

    def validate_dataset(self, ds) -> None:
        if getattr(ds.problem, "name", None) != self.benchmark:
            raise ValueError(
                f"ForcingItrSinAdapter expected dataset benchmark "
                f"{self.benchmark!r}, got {getattr(ds.problem, 'name', None)!r}."
            )
        if getattr(ds.problem, "rc_channel_mode", None) != "broadcast":
            raise ValueError(
                f"{self.benchmark} inversion only supports "
                f"rc_channel_mode='broadcast'."
            )
        first_sid = int(ds.sim_ids[0])
        sample = ds.problem.build_item(ds, first_sid, 0, min(1, ds.Nt - 1))
        if int(sample["spatial"].shape[-1]) != ds.problem.dims.in_channels:
            raise ValueError(
                f"{self.benchmark} dataset spatial schema does not match its spec."
            )
        required = {
            "R_c_base", "R_c_A",
            "temporal_family", "temporal_params", "spatial_family", "spatial_params",
        }
        missing = required - set(dict(ds.sim_params[first_sid]))
        if missing:
            raise ValueError(
                f"{self.benchmark} sim_params missing keys: {sorted(missing)}"
            )

    def equivalent_scalar_values(
        self, theta: torch.Tensor, y_grid: torch.Tensor
    ) -> tuple[float, float]:
        values = _to_numpy_theta(theta, self.theta_dim)
        y = y_grid.detach().cpu().numpy().astype(np.float64)
        profile = make_rc_sin_profile(y, float(values[0]), float(values[1]))
        req = equivalent_scalar_resistance(
            y, profile, bounds=(_GEN_DOMAIN["c"], _GEN_DOMAIN["d"])
        )
        return float(values[0]), req

    def build_scalar_fv_solver(
        self, ds, sim_params: dict, resistance: float, base_kwargs: dict
    ):
        params = dict(sim_params)
        params["R_c"] = float(resistance)
        return ForcingProblem(self._representation(ds)).configure_solver(
            params, base_kwargs
        )

    @staticmethod
    def _representation(ds) -> str:
        return str(getattr(ds.problem, "representation", "temporal_encoder"))
