from __future__ import annotations

import copy
from typing import Any

import numpy as np

from problems.base import OODAxis, ProblemDims, empty_forcing_seq
from problems.forcing import (
    FORCING_TEMPORAL_TOKEN_DIM,
    T_EPS,
    _forcing_seq_2tok_from_samples,
    _sample_a,
)
from problems.source import (
    SourceProblem,
    _classify_regime,
    _lhs_unit,
    _patch_center_ranges,
    _validate_patch_bounds,
)
from src.physics.internal_source import (
    RC_MIN,
    RC_VOID_RANGES,
    R_PEAK_MAX,
    make_rc_void_profile,
    make_sin2_pulse,
)
from src.physics.fv_solver_2d import FVSolver2D, Layer2D

# ----- representation constants (source_itr tensor contract) -------------------

# cond_static: base [t_bar, t_s] + 4 void scalars [R_base, R_amp, y0, sigma]
# + 4 patch scalars [x_h, y_h, w_h, h_h]. The single R_c slot of `source` is
# expanded to the four Gaussian-void parameters; A still never conditions.
COND_STATIC_DIM = 10

# Base spatial channels [T_tilde, x, y, S_h, Rc_y_norm]; bins mode appends the
# 16 source integral bins after the base. S_h stays at index 3; the new
# normalized R_c(y) channel is index 4.
SOURCE_BINS = 16
SPATIAL_CHANNELS_TEMPORAL = 5
SPATIAL_CHANNELS_BINS = SPATIAL_CHANNELS_TEMPORAL + SOURCE_BINS  # 21
S_Y_CHANNEL = 3
RC_Y_CHANNEL = 4

# Default x-localization length for rc_channel_mode="localized" (only used in
# that mode). The slab is [0, 1] with the interface at x = 0.5, so ~0.05 keeps
# the envelope tight around the interface.
DEFAULT_RC_ELL = 0.05


def rc_log_norm(Rc_y: np.ndarray) -> np.ndarray:
    """Log-normalize a physical R_c(y) profile to roughly [-1, 1].

    Linear normalization over the 60x span [RC_MIN, R_PEAK_MAX] = [0.05, 3.0]
    wastes most of the range because realistic R_base clusters in [0.05, 1.0];
    log-spacing spends resolution where the samples are. R_c(y) >= R_base >=
    RC_MIN, so the log argument is always positive.
    """
    log_min = np.log(RC_MIN)
    log_max = np.log(R_PEAK_MAX)
    Rc = np.asarray(Rc_y, dtype=np.float64)
    return (2.0 * (np.log(Rc) - log_min) / (log_max - log_min) - 1.0).astype(np.float32)


def build_cond_vector_itr(
    t_bar_norm: float,
    t_s_norm: float,
    R_base: float,
    R_amp: float,
    y0: float,
    sigma: float,
    x_h: float,
    y_h: float,
    w_h: float,
    h_h: float,
    *,
    x_center_range: tuple[float, float],
    y_center_range: tuple[float, float],
    x_length_scale: float,
    y_length_scale: float,
) -> np.ndarray:
    """Assemble the 10-dim static conditioning vector for source_itr.

    Layout: [t_bar_norm, t_s_norm, R_base_norm, R_amp_norm, y0_norm, sigma_norm,
    x_h_norm, y_h_norm, w_h_norm, h_h_norm]. Each void scalar is normalized over
    its own sampling range; the patch scalars match `source.build_cond_vector`.
    """
    base_lo, base_hi = RC_VOID_RANGES["R_base"]
    amp_lo, amp_hi = RC_VOID_RANGES["R_amp"]
    y0_lo, y0_hi = RC_VOID_RANGES["y0"]
    sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

    R_base_norm = (R_base - base_lo) / (base_hi - base_lo)
    R_amp_norm = (R_amp - amp_lo) / (amp_hi - amp_lo)
    y0_norm = (y0 - y0_lo) / (y0_hi - y0_lo)
    sigma_norm = (sigma - sig_lo) / (sig_hi - sig_lo)

    x_h_norm = (x_h - x_center_range[0]) / (x_center_range[1] - x_center_range[0])
    y_h_norm = (y_h - y_center_range[0]) / (y_center_range[1] - y_center_range[0])
    w_h_norm = w_h / x_length_scale
    h_h_norm = h_h / y_length_scale

    return np.array(
        [
            t_bar_norm, t_s_norm,
            R_base_norm, R_amp_norm, y0_norm, sigma_norm,
            x_h_norm, y_h_norm, w_h_norm, h_h_norm,
        ],
        dtype=np.float32,
    )


def _void_unit_severity(y0: float, sigma: float, y_grid: np.ndarray) -> float:
    """Integrated unit void shape int exp(-((y - y0)/sigma)^2) dy over `y_grid`.

    Trapezoidal on the actual cell-center grid so the integral matches the
    discretization the solver sees (a tiny sigma is honestly under-integrated at
    coarse Ny rather than evaluated against an idealized continuum). The result
    is strictly positive for sigma > 0, so it is a safe denominator when solving
    R_amp to hold integrated severity fixed.
    """
    y = np.asarray(y_grid, dtype=np.float64)
    shape = np.exp(-(((y - float(y0)) / float(sigma)) ** 2))
    return float(np.trapz(shape, y))


def _void_severity(
    R_amp: float, y0: float, sigma: float, y_grid: np.ndarray,
) -> float:
    """Integrated void conductance deficit R_amp * int exp(...) dy on `y_grid`."""
    return float(R_amp) * _void_unit_severity(y0, sigma, y_grid)


class SourceItrProblem(SourceProblem):
    """Source benchmark with a spatially-varying interface resistance R_c(y).

    Subclasses :class:`SourceProblem`, reusing its patch/amplitude/IC sampling,
    patch-mask cache, and source-bin channels. The interface at x = 0.5 carries
    a Gaussian-void resistance profile
        R_c(y) = R_base + R_amp * exp(-((y - y0)/sigma)^2)
    instead of a scalar. The model sees both a normalized R_c(y) spatial channel
    (index 4, broadcast across x, or x-localized) and the four void scalars in
    `cond_static`. The constant-R_c `source` benchmark is untouched.
    """

    name = "source_itr"

    def __init__(self, representation: str = "temporal_encoder"):
        self.representation = representation
        # rc_channel_mode / rc_ell are model-input knobs (build_item only); they
        # do not affect data generation. Defaults here; problem_from_config may
        # override from the benchmark config.
        self.rc_channel_mode = "broadcast"
        self.rc_ell = DEFAULT_RC_ELL

        if representation == "temporal_encoder":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_TEMPORAL,
                cond_static_dim=COND_STATIC_DIM,
                has_forcing_seq=True,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=3,
                use_temporal_encoder=True,
                s_y_channel=S_Y_CHANNEL,
                use_forcing_time_aug=False,
            )
        elif representation == "bins":
            self.dims = ProblemDims(
                in_channels=SPATIAL_CHANNELS_BINS,
                cond_static_dim=COND_STATIC_DIM,
                has_forcing_seq=False,
                temporal_token_dim=FORCING_TEMPORAL_TOKEN_DIM,
                t_stats_dim=3,
                use_temporal_encoder=False,
                s_y_channel=S_Y_CHANNEL,
                use_forcing_time_aug=False,
            )
        else:
            raise ValueError(
                f"Unknown representation {representation!r} for benchmark {self.name!r}."
            )

    # ---- data generation ----

    def sample_sim_params(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
    ) -> list[dict]:
        # Reuse the parent's patch/A/IC sampling unchanged so the patch
        # distribution stays identical to `source`.
        sim_params = super().sample_sim_params(rng, rng_profile, grids, time_cfg)
        num_sims = len(sim_params)
        lhs_seed = int(time_cfg.get("lhs_seed", 0))

        # Independent LHS stream for the four void params; offset so it does not
        # alias the parent's R_c/x_h/y_h/A LHS (which keys off lhs_seed).
        void_rng = np.random.default_rng(lhs_seed + 991)
        u = _lhs_unit(num_sims, 4, void_rng)

        base_lo, base_hi = RC_VOID_RANGES["R_base"]
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

        R_base = base_lo + u[:, 0] * (base_hi - base_lo)
        amp_unit = u[:, 1]
        y0 = y0_lo + u[:, 2] * (y0_hi - y0_lo)
        sigma = sig_lo + u[:, 3] * (sig_hi - sig_lo)
        # Dependent amp bound keeps R_c(y) in [RC_MIN, R_PEAK_MAX] by
        # construction (no clipping); makes (R_base, R_amp) jointly triangular.
        R_amp = amp_unit * (R_PEAK_MAX - R_base)

        for i, params in enumerate(sim_params):
            R_c_base = float(R_base[i])
            params["R_c_base"] = R_c_base
            params["R_c_amp"] = float(R_amp[i])
            params["R_c_y0"] = float(y0[i])
            params["R_c_sigma"] = float(sigma[i])
            # Keep params["R_c"] populated (= R_base) so the universal R_c
            # val-pair column and any params["R_c"] reads stay valid.
            params["R_c"] = R_c_base

        return sim_params

    def configure_solver(self, params: dict, base_kwargs: dict) -> FVSolver2D:
        y_grid = base_kwargs["y_grid"]
        x_I = float(params["interface_x"])
        a = float(base_kwargs["a"])
        b = float(base_kwargs["b"])
        layers = [
            Layer2D(x_left=a, x_right=x_I, rho=1.0, cp=1.0, k=3.0),
            Layer2D(x_left=x_I, x_right=b, rho=1.0, cp=1.0, k=35.0),
        ]
        q_left_fn = lambda t: np.zeros_like(y_grid)
        # Build the source via the parent's patch builder by delegating the
        # rest of the wiring; only interface_R changes (scalar -> R_c(y)).
        from src.physics.internal_source import build_patch_source

        source = build_patch_source(
            x_h=params["x_h"], y_h=params["y_h"],
            w=params["w_h"], h=params["h_h"],
            A=params["A"], t_off=params["t_off"],
        )
        Rc_profile = make_rc_void_profile(
            y_grid,
            R_base=float(params["R_c_base"]),
            R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]),
            sigma=float(params["R_c_sigma"]),
        )
        return FVSolver2D(
            a=base_kwargs["a"], b=base_kwargs["b"],
            c=base_kwargs["c"], d=base_kwargs["d"],
            Nx=base_kwargs["Nx"], Ny=base_kwargs["Ny"],
            lam_target=base_kwargs["lam_target"],
            layers=layers,
            t_final=base_kwargs["t_final"],
            flux_f=base_kwargs["flux_f"], flux_A=base_kwargs["flux_A"],
            t_on=base_kwargs["t_on"], t_off=float(params["t_off"]),
            phase=base_kwargs["phase"],
            dt=base_kwargs["dt"], tukey_alpha=base_kwargs["tukey_alpha"],
            interface_R=[Rc_profile],
            q_left_fn=q_left_fn,
            source=source,
        )

    # ---- dataset item ----

    def _rc_channel(self, ds, params: dict) -> np.ndarray:
        """Build the (Nx, Ny) normalized R_c(y) input channel for one sim."""
        Rc_y = make_rc_void_profile(
            ds.y_grid,
            R_base=float(params["R_c_base"]),
            R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]),
            sigma=float(params["R_c_sigma"]),
        )
        Rc_y_norm = rc_log_norm(Rc_y)  # (Ny,)
        if self.rc_channel_mode == "broadcast":
            channel = np.broadcast_to(Rc_y_norm[None, :], (ds.Nx, ds.Ny))
        elif self.rc_channel_mode == "localized":
            x_I = float(params.get("interface_x", 0.5))
            env = np.exp(-(((ds._X - x_I) / self.rc_ell) ** 2))
            channel = Rc_y_norm[None, :] * env
        else:
            raise ValueError(
                f"Unknown rc_channel_mode {self.rc_channel_mode!r}; "
                f"expected 'broadcast' or 'localized'."
            )
        return np.ascontiguousarray(channel, dtype=np.float32)

    def build_item(self, ds, sid: int, s: int, j: int) -> dict[str, np.ndarray]:
        sid = int(sid)
        params = ds.sim_params[sid]
        interface_x = float(params.get("interface_x", 0.5))

        T_source = ds.trajectories[sid, s, :, :]
        T_target = ds.trajectories[sid, j, :, :]

        T_source_norm = (T_source - ds.mu_global) / (ds.sigma_global + T_EPS)
        T_target_norm = (T_target - ds.mu_global) / (ds.sigma_global + T_EPS)

        if ds.noise_std > 0:
            T_source_norm = T_source_norm + np.random.randn(
                *T_source_norm.shape
            ).astype(np.float32) * ds.noise_std

        t_s_val = float(ds.t_grid[s])
        t_j_val = float(ds.t_grid[j])
        t_bar_norm = (t_j_val - t_s_val) / ds.time_norm_horizon
        t_s_norm = t_s_val / ds.time_norm_horizon

        S_h = self._patch_mask(ds, sid)
        Rc_channel = self._rc_channel(ds, params)
        spatial_base = np.stack(
            [T_source_norm, ds.X_norm, ds.Y_norm, S_h, Rc_channel], axis=-1,
        ).astype(np.float32)

        x_lo, x_hi = float(ds.x_grid[0]), float(ds.x_grid[-1])
        y_lo, y_hi = float(ds.y_grid[0]), float(ds.y_grid[-1])
        x_center_range, y_center_range = _patch_center_ranges(
            x_lo, x_hi, y_lo, y_hi, float(params["w_h"]), float(params["h_h"]),
        )
        cond_static = build_cond_vector_itr(
            t_bar_norm=float(t_bar_norm), t_s_norm=float(t_s_norm),
            R_base=float(params["R_c_base"]), R_amp=float(params["R_c_amp"]),
            y0=float(params["R_c_y0"]), sigma=float(params["R_c_sigma"]),
            x_h=float(params["x_h"]), y_h=float(params["y_h"]),
            w_h=float(params["w_h"]), h_h=float(params["h_h"]),
            x_center_range=x_center_range,
            y_center_range=y_center_range,
            x_length_scale=(x_hi - x_lo),
            y_length_scale=(y_hi - y_lo),
        )

        Y = T_target_norm[:, :, None].astype(np.float32)
        T_stats = np.array(
            [ds.mu_global, ds.sigma_global, interface_x], dtype=np.float32
        )

        if self.representation == "temporal_encoder":
            A = float(params["A"])
            t_off = float(params["t_off"])
            if sid not in ds._q_callables:
                ds._q_callables[sid] = make_sin2_pulse(A, t_off)
            q = ds._q_callables[sid]
            t_samples, a_m = _sample_a(q, t_s_val, t_j_val, ds.temporal_samples)
            forcing_seq = _forcing_seq_2tok_from_samples(
                t_samples, a_m, A_amp_ref=ds.a_amp_ref,
            )
            spatial = spatial_base
        else:
            Q_bins = self._source_bin_channels(ds, sid, t_s_val, t_j_val)
            spatial = np.concatenate([spatial_base, Q_bins], axis=-1).astype(np.float32)
            forcing_seq = empty_forcing_seq()

        return {
            "spatial": spatial,
            "cond_static": cond_static,
            "forcing_seq": forcing_seq,
            "Y": Y,
            "T_stats": T_stats,
        }

    # ---- val-pair logging ----

    # Keep the source patch columns and add the void severity/location columns
    # so error-vs-void-severity (the headline figure) is plottable per pair;
    # the inherited row logs only the universal R_c (= R_c_base), which hides it.
    val_pair_fields = ("x_h", "y_h", "A", "regime", "R_c_amp", "R_c_y0", "R_c_sigma")

    def val_pair_row(self, ds, sim_id: int, s: int, j: int) -> dict[str, Any]:
        p = ds.sim_params[int(sim_id)]
        return {
            "x_h": float(p["x_h"]),
            "y_h": float(p["y_h"]),
            "A": float(p["A"]),
            "regime": p.get("regime", ""),
            "R_c_amp": float(p["R_c_amp"]),
            "R_c_y0": float(p["R_c_y0"]),
            "R_c_sigma": float(p["R_c_sigma"]),
        }

    # ---- schema ----

    def validate_schema(self, sim_params: np.ndarray, sim_ids: np.ndarray) -> None:
        required = (
            "R_c", "interface_x", "x_h", "y_h", "A", "t_off", "w_h", "h_h",
            "R_c_base", "R_c_amp", "R_c_y0", "R_c_sigma",
        )
        forbidden = ("temporal_family", "temporal_params",
                     "spatial_family", "spatial_params")
        for sid in sim_ids:
            entry = sim_params[int(sid)]
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"sim_params[{int(sid)}] missing keys {missing} for benchmark "
                    f"{self.name!r}."
                )
            present_legacy = [k for k in forbidden if k in entry]
            if present_legacy:
                raise ValueError(
                    f"sim_params[{int(sid)}] still contains legacy keys "
                    f"{present_legacy}; regenerate with benchmark {self.name!r}."
                )

    # ---- OOD hooks (W3 out-of-distribution path) ----

    def ood_axes(self) -> dict[str, OODAxis]:
        base_lo, base_hi = RC_VOID_RANGES["R_base"]
        amp_lo, amp_hi = RC_VOID_RANGES["R_amp"]
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        sig_lo, sig_hi = RC_VOID_RANGES["sigma"]
        return {
            "rc_base": OODAxis(
                name="rc_base",
                kind="simulation_parameter",
                id_reference=(0.5,),
                ood_values=(0.1, 1.25, 1.5, 2.0),
                field="R_c_base",
                trained_range=(base_lo, base_hi),
                notes="void floor R_base; reject < RC_MIN (log-normalized R_c(y) "
                "channel hits log(0)); R_amp held at the shared background value.",
            ),
            "rc_amp": OODAxis(
                name="rc_amp",
                kind="simulation_parameter",
                id_reference=(0.5 * (amp_lo + amp_hi),),
                ood_values=(0.0, 3.5, 4.0),
                field="R_c_amp",
                trained_range=(amp_lo, amp_hi),
                notes="void depth R_amp; R_base/y0/sigma held at background. "
                "R_amp=0 is the flat-profile limiting case.",
            ),
            "rc_sigma": OODAxis(
                name="rc_sigma",
                kind="simulation_parameter",
                id_reference=(0.5 * (sig_lo + sig_hi),),
                ood_values=(0.02, 0.30),
                field="R_c_sigma",
                trained_range=(sig_lo, sig_hi),
                resolution_kind="spatial_gaussian_sigma",
                notes="pure void width study: R_amp re-solved to hold integrated "
                "severity fixed across sigma; record realized_severity.",
            ),
            "rc_y0": OODAxis(
                name="rc_y0",
                kind="simulation_parameter",
                id_reference=(0.5,),
                ood_values=(0.0, 0.05, 0.95, 1.0),
                field="R_c_y0",
                trained_range=(y0_lo, y0_hi),
                notes="pure void location study: R_amp re-solved to hold "
                "integrated severity fixed across y0 (edge truncation "
                "compensated); record realized_severity.",
            ),
            "rc_severity": OODAxis(
                name="rc_severity",
                kind="compound",
                id_reference=(0.5,),
                ood_values=(0.1, 1.0, 1.5),
                field="R_c_amp",
                trained_range=(0.0, 1.0),
                compound=True,
                notes="severity-fraction of the no-clip headroom "
                "(R_amp = value * (R_PEAK_MAX - R_base)); value > 1.0 drives the "
                "profile peak past R_PEAK_MAX (extreme). CRN paired on latents.",
            ),
        }

    def draw_latents(
        self,
        rng: np.random.Generator,
        rng_profile: np.random.Generator,
        grids: dict[str, np.ndarray],
        time_cfg: dict[str, Any],
        axis: OODAxis,
    ) -> dict[str, Any]:
        # Reuse the parent's patch / amplitude / IC background unchanged (the
        # void axes never move the patch, so the parent takes its default-patch
        # branch). Then draw the shared void background from the same IC stream.
        latents = super().draw_latents(rng, rng_profile, grids, time_cfg, axis)

        base_lo, base_hi = RC_VOID_RANGES["R_base"]
        y0_lo, y0_hi = RC_VOID_RANGES["y0"]
        sig_lo, sig_hi = RC_VOID_RANGES["sigma"]

        u_base = float(rng.uniform(0.0, 1.0))
        u_amp = float(rng.uniform(0.0, 1.0))
        u_y0 = float(rng.uniform(0.0, 1.0))
        u_sigma = float(rng.uniform(0.0, 1.0))

        R_base_bg = base_lo + u_base * (base_hi - base_lo)
        y0_bg = y0_lo + u_y0 * (y0_hi - y0_lo)
        sigma_bg = sig_lo + u_sigma * (sig_hi - sig_lo)
        # Dependent amp bound, identical to sample_sim_params, so the background
        # void is in-distribution by construction.
        R_amp_bg = u_amp * (R_PEAK_MAX - R_base_bg)

        latents["y_grid"] = np.asarray(grids["y_grid"], dtype=np.float64)
        latents["R_base_bg"] = float(R_base_bg)
        latents["R_amp_bg"] = float(R_amp_bg)
        latents["y0_bg"] = float(y0_bg)
        latents["sigma_bg"] = float(sigma_bg)
        return latents

    def apply_ood_value(
        self, latents: dict[str, Any], axis: OODAxis, value: Any,
    ) -> dict[str, Any]:
        y_grid = latents["y_grid"]
        R_base_bg = float(latents["R_base_bg"])
        R_amp_bg = float(latents["R_amp_bg"])
        y0_bg = float(latents["y0_bg"])
        sigma_bg = float(latents["sigma_bg"])

        # Default the void to the shared background; each axis overrides exactly
        # its swept field (severity-preserving axes also re-solve R_amp).
        R_base, R_amp, y0, sigma = R_base_bg, R_amp_bg, y0_bg, sigma_bg

        if axis.name == "rc_base":
            if float(value) < RC_MIN:
                raise ValueError(
                    f"rc_base value {value} < RC_MIN ({RC_MIN}); the log-normalized "
                    f"R_c(y) channel is undefined at zero floor. Perfect contact is "
                    f"a separately labeled limiting-case experiment."
                )
            R_base = float(value)
        elif axis.name == "rc_amp":
            R_amp = float(value)
        elif axis.name == "rc_sigma":
            sigma_ref = float(axis.id_reference[0])
            S_ref = _void_severity(R_amp_bg, y0_bg, sigma_ref, y_grid)
            sigma = float(value)
            R_amp = S_ref / _void_unit_severity(y0_bg, sigma, y_grid)
        elif axis.name == "rc_y0":
            y0_ref = float(axis.id_reference[0])
            S_ref = _void_severity(R_amp_bg, y0_ref, sigma_bg, y_grid)
            y0 = float(value)
            R_amp = S_ref / _void_unit_severity(y0, sigma_bg, y_grid)
        elif axis.name == "rc_severity":
            R_amp = float(value) * (R_PEAK_MAX - R_base_bg)
        else:
            raise ValueError(
                f"Unknown OOD axis {axis.name!r} for benchmark {self.name!r}."
            )

        a, b, c, d = latents["domain"]
        interface_x = float(latents["interface_x"])
        x_h = float(latents["x_h"])
        y_h = float(latents["y_h"])
        w_h = float(latents["w_h_default"])
        h_h = float(latents["h_h_default"])
        t_off = float(latents["t_off_default"])
        _validate_patch_bounds(
            x_h=x_h, y_h=y_h, w_h=w_h, h_h=h_h, a=a, b=b, c=c, d=d,
        )
        regime = _classify_regime(x_h, interface_x, w_h)

        params = {
            "R_c": float(R_base),
            "interface_x": interface_x,
            "x_h": x_h,
            "y_h": y_h,
            "w_h": w_h,
            "h_h": h_h,
            "A": float(latents["A"]),
            "t_off": t_off,
            "regime": regime,
            "R_c_base": float(R_base),
            "R_c_amp": float(R_amp),
            "R_c_y0": float(y0),
            "R_c_sigma": float(sigma),
            "T0": np.array(latents["T0"], dtype=np.float32),
            "ic_family": latents["ic_family"],
            "ic_params": copy.deepcopy(latents["ic_params"]),
        }
        params["_ood_realized"] = {
            "requested": float(value),
            "R_c_base": float(R_base),
            "R_c_amp": float(R_amp),
            "R_c_y0": float(y0),
            "R_c_sigma": float(sigma),
            "realized_severity": _void_severity(R_amp, y0, sigma, y_grid),
        }
        return params

    def resolution_scale(self, axis: OODAxis, params: dict) -> float | None:
        if axis.resolution_kind == "spatial_gaussian_sigma":
            return float(params["R_c_sigma"])
        return None
