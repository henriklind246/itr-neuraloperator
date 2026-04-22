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

        raw = np.ones((Nx, Ny), dtype=np.float32)
        x_mask = np.abs(x_grid - interface_x) <= interface_half_width
        raw[x_mask, :] = interface_weight

        # Normalize so mean(weights) == 1.0
        raw = raw / raw.mean()

        # Shape (1, Nx, Ny, 1) broadcasts against (B, Nx, Ny, 1)
        weight_tensor = torch.from_numpy(raw).reshape(1, Nx, Ny, 1)
        self.register_buffer("weights", weight_tensor)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE: mean(weights * (y_pred - y_true)^2).

        Parameters
        ----------
        y_pred, y_true : torch.Tensor
            Shape (B, Nx, Ny, 1).
        """
        return torch.mean(self.weights * (y_pred - y_true) ** 2)


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
