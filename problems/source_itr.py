from __future__ import annotations

from typing import Any

import numpy as np

from problems.base import ProblemDims, empty_forcing_seq
from problems.forcing import (
    FORCING_TEMPORAL_TOKEN_DIM,
    T_EPS,
    _forcing_seq_2tok_from_samples,
    _sample_a,
)
from problems.source import (
    SourceProblem,
    _lhs_unit,
    _patch_center_ranges,
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
        t_bar_norm = (t_j_val - t_s_val) / ds.t_final
        t_s_norm = t_s_val / ds.t_final

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
