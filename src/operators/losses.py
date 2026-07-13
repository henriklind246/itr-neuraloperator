import numpy as np
import torch
import torch.nn as nn

"""Custom loss functions for FNO training.

SpatiallyWeightedMSE upweights the interface region so that
the optimizer receives meaningful gradient signal for the
temperature discontinuity caused by interfacial thermal resistance.

2D geometry: the interface is a vertical slab at x = interface_x,
invariant along y. Weights/masks are therefore constant along the y axis.
"""

class SpatiallyWeightedMSE(nn.Module):
    """MSE loss with spatial weighting that upweights the interface region.

    Weight profile: baseline 1.0 everywhere, rectangular bump of height
    ``interface_weight`` in [interface_x - half_width, interface_x + half_width],
    constant along y. Weights are normalized so their mean equals 1.0
    (loss scale matches plain MSE).

    When ``interface_weight=1.0``, this reduces to standard MSE.

    Parameters
    ----------
    x_grid : np.ndarray
        1D spatial grid along x, shape (Nx,).
    y_grid : np.ndarray
        1D spatial grid along y, shape (Ny,).
    interface_x : float
        Interface location in physical x-coordinates.
    interface_half_width : float
        Half-width of the boosted region in x-coordinates.
    interface_weight : float
        Boost factor for nodes within the interface region.
    """
    def __init__(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        interface_x: float = 0.5,
        interface_half_width: float = 0.05,
        interface_weight: float = 10.0,
    ):
        super().__init__()
        Nx = len(x_grid)
        Ny = len(y_grid)

        self.interface_half_width = float(interface_half_width)
        self.interface_weight = float(interface_weight)

        raw = np.ones((Nx, Ny), dtype=np.float32)
        x_mask = np.abs(x_grid - interface_x) <= interface_half_width
        raw[x_mask, :] = interface_weight

        # Normalize so mean(weights) == 1.0
        raw = raw / raw.mean()

        # Shape (1, Nx, Ny, 1) broadcasts against (B, Nx, Ny, 1)
        weight_tensor = torch.from_numpy(raw).reshape(1, Nx, Ny, 1)
        self.register_buffer("weights", weight_tensor)

        # 1D x-grid kept on-device so per-sample weights can be built from a
        # batch of interface locations (interfaces benchmark).
        x_grid_t = torch.from_numpy(np.asarray(x_grid, dtype=np.float32))
        self.register_buffer("x_grid_t", x_grid_t)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        interface_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted MSE: mean(weights * (y_pred - y_true)^2).

        Parameters
        ----------
        y_pred, y_true : torch.Tensor
            Shape (B, Nx, Ny, 1).
        interface_x : torch.Tensor or None
            Per-sample interface locations, shape (B,). When None, the fixed
            weight band baked at construction is used. When provided, a per-sample
            band centered on each ``interface_x`` is built and normalized to mean
            1.0 per sample.
        """
        if interface_x is None:
            return torch.mean(self.weights * (y_pred - y_true) ** 2)

        band = build_interface_band(
            self.x_grid_t, interface_x, self.interface_half_width
        )  # (B, Nx) bool
        raw = 1.0 + (self.interface_weight - 1.0) * band.to(y_pred.dtype)  # (B, Nx)
        w = raw / raw.mean(dim=1, keepdim=True)  # mean-1 per sample
        w = w.reshape(w.shape[0], w.shape[1], 1, 1)  # (B, Nx, 1, 1)
        return torch.mean(w * (y_pred - y_true) ** 2)


def build_interface_band(
    x_grid_t: torch.Tensor,
    interface_x: torch.Tensor,
    half_width: float,
) -> torch.Tensor:
    """Return per-sample interface band, shape (B, Nx), bool.

    True where ``|x - interface_x| <= half_width``.

    Parameters
    ----------
    x_grid_t : torch.Tensor
        1D spatial grid along x, shape (Nx,).
    interface_x : torch.Tensor
        Per-sample interface locations, shape (B,).
    half_width : float
        Half-width of the band in x-coordinates.
    """
    x = x_grid_t.reshape(1, -1)  # (1, Nx)
    centers = interface_x.reshape(-1, 1).to(x.dtype)  # (B, 1)
    return (torch.abs(x - centers) <= half_width)


def get_batch_interface_x(
    batch: dict,
    device: torch.device,
    *,
    use_per_sample_interface: bool,
) -> torch.Tensor | None:
    """Return per-sample interface locations from a batch, or None.

    Gating is intentionally by the explicit ``use_per_sample_interface`` flag AND
    the presence of a third T_stats column (slot 2 = interface_x). ``source`` has
    slot 2 = 0.5 always, so the flag (off for source) keeps it on the fixed path.
    """
    t_stats = batch.get("T_stats", None)
    if use_per_sample_interface and t_stats is not None and t_stats.shape[1] > 2:
        return t_stats[:, 2].to(device)
    return None


def build_interface_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    interface_x: float = 0.5,
    interface_half_width: float = 0.05,
) -> torch.Tensor:
    """Return boolean tensor of shape (Nx, Ny) for nodes within the interface region.

    The interface is a vertical slab at x = interface_x, so the mask is True on
    full columns and False elsewhere. Used for the monitoring metric
    (interface-specific rel L2), not for the loss itself.
    """
    Ny = len(y_grid)
    x_mask = np.abs(x_grid - interface_x) <= interface_half_width  # (Nx,)
    mask_2d = np.broadcast_to(x_mask[:, None], (len(x_grid), Ny)).copy()
    return torch.from_numpy(mask_2d)


def build_boundary_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    width: float = 0.05,
) -> torch.Tensor:
    """Return boolean tensor of shape (Nx, Ny) for nodes within ``width`` of any
    domain edge.

    The mask is True where x is within ``width`` of the left/right edges (a/b)
    OR y is within ``width`` of the bottom/top edges (c/d). The bounds are read
    from the grid extents, so the band is coordinate-based and therefore
    resolution-agnostic. Used to isolate the FNO ``padding`` confound (the
    absolute pad covers a shrinking fraction of the domain as resolution grows).
    """
    a, b = float(x_grid[0]), float(x_grid[-1])
    c, d = float(y_grid[0]), float(y_grid[-1])
    x_edge = (np.abs(x_grid - a) <= width) | (np.abs(x_grid - b) <= width)  # (Nx,)
    y_edge = (np.abs(y_grid - c) <= width) | (np.abs(y_grid - d) <= width)  # (Ny,)
    mask_2d = x_edge[:, None] | y_edge[None, :]  # (Nx, Ny)
    return torch.from_numpy(mask_2d.copy())


def compute_interface_rel_l2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    iface_mask: torch.Tensor,
) -> float:
    """Compute relative L2 error restricted to the interface region.

    Parameters
    ----------
    y_pred, y_true : torch.Tensor
        Shape (B, Nx, Ny, 1).
    iface_mask : torch.Tensor
        Boolean mask of shape (Nx, Ny) — True for nodes in the interface region.

    Returns
    -------
    float
        Interface-region rel L2 in percent.
    """
    # Advanced indexing with a 2D mask on dims (1,2) collapses to (B, N_iface, 1)
    pred_iface = y_pred[:, iface_mask, :]
    true_iface = y_true[:, iface_mask, :]

    rel_l2 = (
        torch.mean((pred_iface - true_iface) ** 2)
        / torch.mean(true_iface ** 2)
    ) ** 0.5 * 100
    return rel_l2.item()


def physics_residual_loss(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom,
    extra_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared interior CN cell-balance residual (Stage 1, interior-only).

    ``T_n`` / ``T_np1`` are normalized temperature fields one solver step ``dt``
    apart in the training loss layout ``(B, Nx, Ny, 1)`` (a trailing channel of
    1) or ``(B, Nx, Ny)``. The interior residual is the nondimensional per-step
    (normalized) temperature residual directly (the r-coefficients fold in dt,
    the CN 1/2, and the cell capacity), so the loss is ``mean(res^2)`` over the
    interior-mask cells — no extra capacity-scale division on the interior path.

    The mean is over masked cells only (zeroed boundary/Dirichlet cells are
    excluded from the denominator) so the loss magnitude is independent of the
    masked-out fraction.
    """
    from src.physics.fv_residual import interior_cn_residual

    if T_n.dim() == 4:
        T_n = T_n[..., 0]
        T_np1 = T_np1[..., 0]
    res, mask = interior_cn_residual(T_n, T_np1, geom, extra_mask=extra_mask)
    denom = res.shape[0] * int(mask.sum())
    return res.pow(2).sum() / denom


_FULL_BC_REGIONS = ("interior", "left_neumann", "right_dirichlet", "topbot_adiabatic")


def full_bc_physics_loss(
    T_n: torch.Tensor,
    T_np1: torch.Tensor,
    geom,
    bc,
    region_weights: dict | None = None,
    dirichlet_both_ends: bool = False,
    per_sample: bool = False,
    interface_band: bool = False,
) -> dict:
    """Region-partitioned full-BC physics loss for the Stage-2 ``full_bc`` path.

    Computes the per-region mean-squared residual from
    ``fv_residual.full_bc_cn_residual`` and returns a flat dict:

      - ``phys_interior_mse``, ``phys_left_neumann_mse``,
        ``phys_right_dirichlet_mse``, ``phys_topbot_adiabatic_mse`` — the
        per-region diagnostics (mean over that region's cells, all four already
        commensurate normalized per-step temperature errors).
      - ``physics_loss_weighted`` — the gradient signal
        ``sum_region w_region * region_mse`` with ``region_weights`` (default all
        1.0, i.e. each REGION contributes equally, deliberately up-weighting the
        few-celled boundary rows relative to a global all-cell mean).
      - ``physics_loss_allcell_mean`` — the plain all-cell ``mean(r^2)`` over the
        four-region partition (each cell once; unaffected by the weights), for
        comparison against the raw residual behavior.

    With ``dirichlet_both_ends`` the right-Dirichlet MSE averages the violation
    at both model-output times (``T_n`` and ``T_np1``); ``physics_loss_allcell_mean``
    still uses the single ``T_np1`` partition so it stays a true per-cell mean.

    With ``per_sample`` the residual is kept batch-shaped and the dict also
    carries the per-region per-sample means (each ``(B,)``):
    ``interior_per_sample``, ``left_neumann_per_sample``,
    ``topbot_adiabatic_per_sample``, ``right_dirichlet_per_sample`` (the
    right-Dirichlet averages the ``n``/``np1`` halves under
    ``dirichlet_both_ends``). The scalar keys are the batch means of these, so
    they are numerically identical to the flattened default.

    With ``interface_band`` (requires ``per_sample``; raises otherwise) the two
    interior node columns flanking the interface face — full-grid columns
    ``{face_idx, face_idx+1}`` = interior columns ``{face_idx-1, face_idx}`` from
    ``geom.face_idx`` — are split off into a disjoint ``interface_band`` term.
    ``interior_per_sample`` / ``phys_interior_mse`` then become **bulk-only**
    (band columns removed) and two extra keys appear: ``interface_band_per_sample``
    ``(B,)`` and ``phys_interface_band_mse`` (its batch mean). Fails loudly if
    ``geom.face_idx`` is missing or a mapped column is out of range.
    ``interface_band=False`` is byte-for-byte the legacy output.
    """
    from src.physics.fv_residual import full_bc_cn_residual

    if T_n.dim() == 4:
        T_n = T_n[..., 0]
        T_np1 = T_np1[..., 0]

    # Fail loudly: interface_band is a per-sample-only feature. Never silently
    # fall back to a band-less loss (that would make a "band-enabled" run
    # byte-identical to the baseline). Only interface_band=False is legacy-exact.
    if interface_band and not per_sample:
        raise ValueError(
            "full_bc_physics_loss: interface_band=True requires per_sample=True "
            "(the disjoint band split is defined on the batch-shaped interior "
            "residual)."
        )

    if not per_sample:
        parts = full_bc_cn_residual(
            T_n, T_np1, geom, bc, dirichlet_both_ends=dirichlet_both_ends
        )

        region_mse = {name: parts[name].pow(2).mean() for name in _FULL_BC_REGIONS}
        if dirichlet_both_ends:
            region_mse["right_dirichlet"] = 0.5 * (
                parts["right_dirichlet"].pow(2).mean()
                + parts["right_dirichlet_n"].pow(2).mean()
            )

        weights = {name: 1.0 for name in _FULL_BC_REGIONS}
        if region_weights:
            for name, w in region_weights.items():
                if name in weights:
                    weights[name] = float(w)

        weighted = sum(weights[name] * region_mse[name] for name in _FULL_BC_REGIONS)
        allcell = torch.cat([parts[name] for name in _FULL_BC_REGIONS]).pow(2).mean()

        return {
            "physics_loss_weighted": weighted,
            "physics_loss_allcell_mean": allcell,
            "phys_interior_mse": region_mse["interior"],
            "phys_left_neumann_mse": region_mse["left_neumann"],
            "phys_right_dirichlet_mse": region_mse["right_dirichlet"],
            "phys_topbot_adiabatic_mse": region_mse["topbot_adiabatic"],
        }

    parts = full_bc_cn_residual(
        T_n, T_np1, geom, bc,
        dirichlet_both_ends=dirichlet_both_ends, keep_batch=True,
    )

    # Per-region per-sample MSE (mean over the region's cells; leading batch dim
    # kept). Region tensors are (B, ...); reduce every non-batch dim.
    def _ps(t: torch.Tensor) -> torch.Tensor:
        return t.pow(2).flatten(1).mean(dim=1)

    interior_per_sample = _ps(parts["interior"])
    interface_band_per_sample = None
    if interface_band:
        # (b) The band is derived from the per-sample interface face index; a
        # scalar geom (face_idx is None) cannot supply it.
        face_idx = getattr(geom, "face_idx", None)
        if face_idx is None:
            raise ValueError(
                "full_bc_physics_loss: interface_band=True requires "
                "geom.face_idx (per-sample interface face); got None. Build the "
                "geometry with build_cn_geom_per_interface."
            )
        # parts["interior"] is (B, Nx-2, Ny-2). Full-grid column i maps to
        # interior column k = i-1, so the interface-flanking full-grid columns
        # {face_idx, face_idx+1} are interior columns {face_idx-1, face_idx}.
        sq = parts["interior"].square()  # (B, Nx-2, Ny-2)
        B, nx_int, ny_int = sq.shape
        dev = sq.device
        face_idx = face_idx.to(device=dev, dtype=torch.long)  # (B,)
        band_idx = torch.stack([face_idx - 1, face_idx], dim=1)  # (B, 2)
        # (c) Both mapped interior columns must be strictly inside [0, Nx-3].
        if int(band_idx.min()) < 0 or int(band_idx.max()) > nx_int - 1:
            raise ValueError(
                "full_bc_physics_loss: interface band column out of range; "
                f"mapped interior columns {{face_idx-1, face_idx}} must lie in "
                f"[0, {nx_int - 1}] (Nx-3), got min={int(band_idx.min())} "
                f"max={int(band_idx.max())}."
            )
        band_cols = torch.zeros(B, nx_int, dtype=torch.bool, device=dev)
        rows = torch.arange(B, device=dev)[:, None].expand(-1, 2)
        band_cols[rows, band_idx] = True  # (B, Nx-2) bool
        band_mask = band_cols[:, :, None].expand_as(sq)  # (B, Nx-2, Ny-2)
        band_count = band_mask.sum(dim=(1, 2))            # (B,) = 2*(Ny-2)
        bulk_count = (~band_mask).sum(dim=(1, 2))         # (B,)
        interface_band_per_sample = (sq * band_mask).sum(dim=(1, 2)) / band_count
        interior_per_sample = (sq * ~band_mask).sum(dim=(1, 2)) / bulk_count

    per = {
        "interior": interior_per_sample,
        "left_neumann": _ps(parts["left_neumann"]),
        "topbot_adiabatic": _ps(parts["topbot_adiabatic"]),
        "right_dirichlet": _ps(parts["right_dirichlet"]),
    }
    if dirichlet_both_ends:
        per["right_dirichlet"] = 0.5 * (per["right_dirichlet"] + _ps(parts["right_dirichlet_n"]))

    region_mse = {name: per[name].mean() for name in _FULL_BC_REGIONS}

    weights = {name: 1.0 for name in _FULL_BC_REGIONS}
    if region_weights:
        for name, w in region_weights.items():
            if name in weights:
                weights[name] = float(w)

    weighted = sum(weights[name] * region_mse[name] for name in _FULL_BC_REGIONS)
    allcell = torch.cat(
        [parts[name].flatten() for name in _FULL_BC_REGIONS]
    ).pow(2).mean()

    out = {
        "physics_loss_weighted": weighted,
        "physics_loss_allcell_mean": allcell,
        "phys_interior_mse": region_mse["interior"],
        "phys_left_neumann_mse": region_mse["left_neumann"],
        "phys_right_dirichlet_mse": region_mse["right_dirichlet"],
        "phys_topbot_adiabatic_mse": region_mse["topbot_adiabatic"],
        "interior_per_sample": per["interior"],
        "left_neumann_per_sample": per["left_neumann"],
        "topbot_adiabatic_per_sample": per["topbot_adiabatic"],
        "right_dirichlet_per_sample": per["right_dirichlet"],
    }
    if interface_band:
        # interior_* above are now bulk-only (band columns removed); the two
        # terms are disjoint and cell-count-weighted-recombine to the full
        # interior mean-square.
        out["interface_band_per_sample"] = interface_band_per_sample
        out["phys_interface_band_mse"] = interface_band_per_sample.mean()
    return out


# ---------------------------------------------------------------------------
# Unified metric suite (per-sample, normalization-invariant headline + jump +
# Kelvin + tails). Shared by train.py (train_one_epoch / validate) and
# eval.py (evaluate). See the metric definitions in the project plan.
#
# All helpers operate on z-scored (normalized) fields of shape (B, Nx, Ny, 1).
# Kelvin quantities are obtained by a scalar multiply with ``sigma_global``
# because temperature normalization is global (T = T_tilde * sigma + mu).
# ---------------------------------------------------------------------------

# Denominator floors for near-constant fields. With uniform 300 K initial
# conditions some early/low-forcing targets have a near-zero per-sample std (or
# jump), which would make a relative metric blow up; these samples are
# uninformative for a relative metric, so the floor just keeps them finite.
EPS_STD = 1e-6
EPS_JUMP = 1e-6


def interface_flanking_nodes(x_grid: np.ndarray, interface_x: float) -> tuple[int, int]:
    """Return the node indices immediately flanking an interface location.

    Torch/np port of ``visual/dataset_plots.py:_interface_flanking_nodes_from_grid``
    so the monitored interface jump uses the same flanking convention as the
    paper plots. Scalar (fixed-interface) form.
    """
    xg = np.asarray(x_grid, dtype=np.float64)
    iface_idx = int(np.argmin(np.abs(xg - interface_x)))
    left_node = iface_idx - 1 if xg[iface_idx] >= interface_x else iface_idx
    right_node = left_node + 1
    if left_node < 0 or right_node >= len(xg):
        raise ValueError(
            "Interface location is outside the interior of the provided x_grid."
        )
    return left_node, right_node


def interface_flanking_nodes_per_sample(
    x_grid_t: torch.Tensor,
    interface_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized flanking nodes for a batch of interface locations.

    Parameters
    ----------
    x_grid_t : torch.Tensor
        1D spatial grid along x, shape (Nx,).
    interface_x : torch.Tensor
        Per-sample interface locations, shape (B,).

    Returns
    -------
    (left, right) : tuple of torch.Tensor
        Long tensors of shape (B,) with the flanking node indices.
    """
    xg = x_grid_t.reshape(-1)  # (Nx,)
    centers = interface_x.reshape(-1, 1).to(xg.dtype)  # (B, 1)
    iface_idx = torch.argmin(torch.abs(xg.reshape(1, -1) - centers), dim=1)  # (B,)
    x_at_idx = xg[iface_idx]  # (B,)
    left = torch.where(x_at_idx >= interface_x.to(xg.dtype), iface_idx - 1, iface_idx)
    right = left + 1
    nx = xg.shape[0]
    if torch.any(left < 0) or torch.any(right >= nx):
        raise ValueError(
            "Interface location is outside the interior of the provided x_grid."
        )
    return left.long(), right.long()


def per_sample_sq_rms(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS of (pred - true) over all non-batch dims. Shape (B,)."""
    diff = y_pred - y_true
    dims = tuple(range(1, diff.ndim))
    return torch.sqrt(torch.mean(diff ** 2, dim=dims))


def per_sample_nrmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps_std: float = EPS_STD,
) -> torch.Tensor:
    """Per-sample nRMSE = rms_grid(pred-true) / max(std_grid(true), eps_std).

    Shape (B,). Normalization-invariant: scaling pred and true by the same
    constant leaves the ratio unchanged (as long as the floor does not bind),
    so the value is identical in z-scored and Kelvin space.
    """
    dims = tuple(range(1, y_true.ndim))
    rms = per_sample_sq_rms(y_pred, y_true)
    std = torch.std(y_true, dim=dims, unbiased=False)
    return rms / std.clamp_min(eps_std)


def _gather_x_node(field: torch.Tensor, idx) -> torch.Tensor:
    """Select x-node ``idx`` from ``field`` (B, Nx, Ny, 1) -> (B, Ny, 1).

    ``idx`` is either a Python int (fixed interface) or a Long tensor of shape
    (B,) (per-sample interface).
    """
    if torch.is_tensor(idx):
        b, _, ny, c = field.shape
        idx_exp = idx.reshape(b, 1, 1, 1).expand(b, 1, ny, c)
        return torch.gather(field, 1, idx_exp).squeeze(1)
    return field[:, idx, :, :]


def per_sample_node_jump_errors(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    left,
    right,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Node-to-node interface jump error, per sample.

    The jump is extracted from the FNO output field (``y_pred``), never
    re-derived analytically from the truth — this matches the verified plot call
    sites and keeps the diagnostic non-circular.

    Parameters
    ----------
    y_pred, y_true : torch.Tensor
        Shape (B, Nx, Ny, 1) in normalized space.
    left, right : int or torch.Tensor
        Flanking node indices; scalar (fixed interface) or Long (B,) per-sample.

    Returns
    -------
    (err_rms, true_jump_rms) : tuple of torch.Tensor
        Both shape (B,). ``err_rms`` = rms_y(jump_pred - jump_true);
        ``true_jump_rms`` = rms_y(jump_true). Caller scales ``err_rms`` by
        ``sigma_global`` for Kelvin, or divides by max(true_jump_rms, eps) for
        the offset-free normalized jump error.
    """
    pred_jump = _gather_x_node(y_pred, right) - _gather_x_node(y_pred, left)  # (B, Ny, 1)
    true_jump = _gather_x_node(y_true, right) - _gather_x_node(y_true, left)
    dims = tuple(range(1, pred_jump.ndim))
    err_rms = torch.sqrt(torch.mean((pred_jump - true_jump) ** 2, dim=dims))
    true_jump_rms = torch.sqrt(torch.mean(true_jump ** 2, dim=dims))
    return err_rms, true_jump_rms


def per_sample_contact_jump_rmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    left,
    right,
    weight_y: torch.Tensor,
) -> torch.Tensor:
    """Per-sample contact-jump RMSE (normalized), source_itr only.

    The contact jump weights the node jump by the per-row contact law
    ``R_c(y) * G(y)`` (see ``visual/dataset_plots.py:_interface_contact_jump_map``):

        contact_jump(y) = R_c(y) * G(y) * (field[left, y] - field[right, y]).

    Returns the RMS over y of the weighted contact-jump difference per sample,
    shape (B,), in normalized space. The caller multiplies by ``sigma_global``
    to obtain Kelvin (the ~300 K offset cancels in the left-right difference).

    Parameters
    ----------
    weight_y : torch.Tensor
        Per-row weight ``R_c(y) * G(y)``; shape (Ny,) (shared) or (B, Ny).
    """
    pred_left = _gather_x_node(y_pred, left).squeeze(-1)    # (B, Ny)
    pred_right = _gather_x_node(y_pred, right).squeeze(-1)
    true_left = _gather_x_node(y_true, left).squeeze(-1)
    true_right = _gather_x_node(y_true, right).squeeze(-1)
    w = weight_y if weight_y.ndim == 2 else weight_y.reshape(1, -1)
    cj_pred = w * (pred_left - pred_right)   # (B, Ny)
    cj_true = w * (true_left - true_right)
    diff = cj_pred - cj_true
    return torch.sqrt(torch.mean(diff ** 2, dim=1))


def tail_stats(values: torch.Tensor) -> dict:
    """Return distributional stats of a 1D tensor of per-pair metric values.

    Keys: mean, p25, p50, p75, iqr, p90, p99, max. The p25/p50/p75/iqr keys are
    additive; existing callers reading mean/p90/p99/max keep working unchanged.
    """
    v = values.detach().to(torch.float64).reshape(-1)
    if v.numel() == 0:
        return {
            "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "iqr": 0.0,
            "p90": 0.0, "p99": 0.0, "max": 0.0,
        }
    q = torch.quantile(
        v,
        torch.tensor([0.25, 0.50, 0.75, 0.90, 0.99], dtype=v.dtype, device=v.device),
    )
    p25, p50, p75 = float(q[0]), float(q[1]), float(q[2])
    return {
        "mean": float(v.mean()),
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "iqr": p75 - p25,
        "p90": float(q[3]),
        "p99": float(q[4]),
        "max": float(v.max()),
    }
