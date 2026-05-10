import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Time-conditioned 2d FNO --------
#
# Operator-learning task:
# G(T(x, y, t_s), x, y, s_y, Q_y_bins, cond) -> T(x, y, t_j)
#
# where t_bar = t_j - t_s is the lead time, t_s is the absolute source time, and
# T is the globally normalized temperature (using fixed mu_global, sig_global per training set).
# The 28D conditioning vector is injected via Conditional Instance Normalization (CIN).


# --------- SpectralConv2d ---------

class SpectralConv2d(nn.Module):
    """2D Fourier convolution: FFT → mode-wise channel mixing → IFFT.

    When ``spectral_dropout > 0``, random Fourier modes are zeroed during
    training, preventing the model from relying on specific frequencies.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2:int, spectral_dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.spectral_dropout = spectral_dropout

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, inp, weights):
        # (B, C_in, Kx, Ky), (C_in, C_out, Kx, Ky) → (B, C_out, Kx, Ky)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        # x: (B, C, Nx, Ny)
        Nx = x.size(-2)
        Ny = x.size(-1)

        # FFT along both spatial dim → (B, C, Nx//2+1)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        Nx_freq = x_ft.size(-2)
        Ny_freq = x_ft.size(-1)

        mx = min(self.modes1, Nx_freq)
        my = min(self.modes2, Ny_freq)
        # Guard: the x dim is full-length in rfft2, so positive slice [:mx] and
        # negative slice [-mx:] overlap (and silently corrupt each other) when
        # 2*mx > Nx_freq. Force the user to pick modes1 <= Nx_freq // 2.
        if 2 * mx > Nx_freq:
            raise ValueError(
                f"modes1={self.modes1} too large for Nx_freq={Nx_freq}: "
                f"positive and negative x-mode slices overlap (need 2*modes1 <= Nx_freq)."
            )
        out_ft = torch.zeros(x.size(0), self.out_channels, Nx_freq, Ny_freq, dtype=torch.cfloat, device=x.device)
        # positive modes
        out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
        # negative modes
        # for rfft2, the last dimension is one-sided but the x dim. is still positive and negative
        # mx and my are still used for weights2 simply because the slice is still shape (B, C_in, mx, my) so its just
        # illustrating the size of the weight2 learnable tensor
        out_ft[:, :, -mx:, :my] = self.compl_mul2d(x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my])

        # Spectral dropout: randomly zero modes during training.
        # Matches fno1d.SpectralConv1d: no inverse-scaling by 1/(1-p), so
        # training-time spectral magnitude is attenuated vs eval. Intentional
        # (mirrors the 1D baseline); kept consistent across 1D/2D.
        if self.training and self.spectral_dropout > 0:
            mask_x = (torch.rand(mx, device=x.device) >= self.spectral_dropout).to(out_ft.dtype)
            mask_y = (torch.rand(my, device=x.device) >= self.spectral_dropout).to(out_ft.dtype)
            mask = mask_x[:, None] * mask_y[None, :]
            out_ft[:, :, :mx, :my] = out_ft[:, :, :mx, :my] * mask
            out_ft[:, :, -mx:, :my] = out_ft[:, :, -mx:, :my] * mask


        # IFFT back to physical space
        return torch.fft.irfft2(out_ft, s=(Nx, Ny), dim=(-2, -1))


# --------- Conditional Instance Normalization ---------

class ConditionalInstanceNorm2d(nn.Module):
    """Instance normalization + external affine (γ, β) from conditioning MLP."""

    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, gamma, beta, valid_shape: tuple[int, int] | None = None):
        # x: (B, C, Nx, Ny),  gamma/beta: (B, C)
        out = self.norm(x)
        return gamma[:, :, None, None] * out + beta[:, :, None, None]


# --------- Conditioning MLP ---------

class ConditioningMLP(nn.Module):
    """Maps per-sample conditioning vector → per-layer CIN parameters (γ, β).

    Identity init: last linear layer has weight=0 and bias=[1…1, 0…0]
    so that at initialization γ=1, β=0 (plain InstanceNorm, model starts
    as unconditioned FNO).
    """

    def __init__(self, cond_dim: int, hidden_dim: int, n_layers: int, width: int):
        super().__init__()
        self.n_layers = n_layers
        # width of fno layers
        self.width = width

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, n_layers * 2 * width)

        # Identity init for head: γ=1, β=0 at start
        # Bias layout must match reshape(n_layers, 2, width):
        #   [γ_0(width), β_0(width), γ_1(width), β_1(width), ...]
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            bias_3d = torch.zeros(n_layers, 2, width)
            bias_3d[:, 0, :] = 1.0  # γ = 1 for all layers
            # β stays 0
            self.head.bias.copy_(bias_3d.reshape(-1))

    def forward(self, cond):
        # cond: (B, cond_dim)
        h = self.net(cond)
        out = self.head(h)                          # (B, n_layers * 2 * width)
        return out.view(-1, self.n_layers, 2, self.width)  # (B, L, 2, W)


# --------- FNO2d ---------

class FNO2d(nn.Module):
    """Time-conditioned 2D Fourier Neural Operator.

    Forward signature:
        model(spatial, cond) → y_pred

    spatial : (B, Nx, Ny, C_spatial)   — T̃(x, y, t_s), x_norm, y_norm, s_y, and Q_y bins
    cond      : (B, C_cond)         — 28D forcing/contact conditioning vector
    y_pred    : (B, Nx, Ny, out_channels) — predicted T̃(x, y, t_j)
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        in_channels: int = 20,
        out_channels: int = 1,
        n_layers: int = 4,
        cond_dim: int = 28,
        cond_hidden: int = 256,
        dropout: float = 0.0,
        spectral_dropout: float = 0.0
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.padding = 8  # pad spatial dim for non-periodic signals

        # Lift: (B, Nx, Ny, in_channels) → (B, Nx, Ny, width)
        self.linear_p = nn.Linear(in_channels, width)

        # ------- FOURIER LAYERS -------------
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2, spectral_dropout=spectral_dropout) for _ in range(n_layers)
        ])
        # local linear transformations (essentially skip connections)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        # conditional instance norm layers
        self.cin_layers = nn.ModuleList([
            ConditionalInstanceNorm2d(width) for _ in range(n_layers)
        ])

        # Conditioning MLP
        self.cond_mlp = ConditioningMLP(cond_dim, cond_hidden, n_layers, width)

        # Project: (B, Nx, Ny, width) → (B, Nx, Ny, out_channels)
        self.linear_q = nn.Linear(width, 128)
        self.output_layer = nn.Linear(128, out_channels)

        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, spatial, cond):
        """
        spatial : (B, Nx, Ny, in_channels)
        cond      : (B, cond_dim)
        returns   : (B, Nx, Ny, out_channels)
        """
        # Conditioning: compute all (γ_l, β_l) upfront
        cin_params = self.cond_mlp(cond)  # (B, n_layers, 2, width)

        # Lift
        x = self.linear_p(spatial)      # (B, Nx, Ny, width)
        x = x.permute(0, 3, 1, 2)            # (B, width, Nx, Ny)

        Nx0 = x.size(-2)
        Ny0 = x.size(-1)
        x = F.pad(x, (0, self.padding, 0, self.padding))   # (B, width, Nx + pad, Ny + pad)

        # Fourier blocks
        for l in range(self.n_layers):
            gamma = cin_params[:, l, 0, :]  # (B, width)
            beta = cin_params[:, l, 1, :]   # (B, width)

            x1 = self.spectral_layers[l](x)
            x2 = self.conv_layers[l](x)
            x = x1 + x2
            x = self.cin_layers[l](x, gamma, beta, valid_shape=(Nx0, Ny0))
            x = self.drop(self.activation(x))

        # Unpad + Project
        x = x[:, :, :Nx0, :Ny0]                 # (B, width, Nx, Ny)
        x = x.permute(0, 2, 3, 1)            # (B, Nx, Ny, width)
        x = self.drop(self.activation(self.linear_q(x)))  # (B, Nx, Ny, 128)
        x = self.output_layer(x)           # (B, Nx, Ny, out_channels)
        return x
