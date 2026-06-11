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
