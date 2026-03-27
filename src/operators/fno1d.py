import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Time-conditioned 1D FNO ---------
#
# Operator-learning task:
#   G_θ(T̃(x, t_s), x, t̄, A, f, R_c) → T̃(x, t_j)
#
# where t̄ = t_j − t_s is the lead time and T̃ denotes per-sample
# normalized temperature.  The conditioning vector (t̄, A, f, R_c)
# is injected via Conditional Instance Normalization (CIN).


# --------- SpectralConv1d ---------

class SpectralConv1d(nn.Module):
    """1D Fourier convolution: FFT → mode-wise channel mixing → IFFT.

    When ``spectral_dropout > 0``, random Fourier modes are zeroed during
    training, preventing the model from relying on specific frequencies.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int, spectral_dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spectral_dropout = spectral_dropout

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, inp, weights):
        # (B, C_in, K), (C_in, C_out, K) → (B, C_out, K)
        return torch.einsum("bik,iok->bok", inp, weights)

    def forward(self, x):
        # x: (B, C, Nx)
        Nx = x.size(-1)

        # FFT along spatial dim → (B, C, Nx//2+1)
        x_ft = torch.fft.rfft(x, dim=-1)
        N_freq = x_ft.size(-1)

        m = min(self.modes, N_freq)
        out_ft = torch.zeros(
            x.size(0), self.out_channels, N_freq,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :m] = self.compl_mul1d(x_ft[:, :, :m], self.weights[:, :, :m])

        # Spectral dropout: randomly zero modes during training
        if self.training and self.spectral_dropout > 0:
            mask = (torch.rand(m, device=x.device) >= self.spectral_dropout).to(out_ft.dtype)
            out_ft[:, :, :m] = out_ft[:, :, :m] * mask

        # IFFT back to physical space
        return torch.fft.irfft(out_ft, n=Nx, dim=-1)


# --------- Conditional Instance Normalization ---------

class ConditionalInstanceNorm1d(nn.Module):
    """InstanceNorm1d + external affine (γ, β) from conditioning MLP."""

    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)

    def forward(self, x, gamma, beta):
        # x: (B, C, Nx),  gamma/beta: (B, C)
        out = self.norm(x)
        return gamma[:, :, None] * out + beta[:, :, None]


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


# --------- FNO1d ---------

class FNO1d(nn.Module):
    """Time-conditioned 1D Fourier Neural Operator.

    Forward signature:
        model(x_spatial, cond) → y_pred

    x_spatial : (B, Nx, C_spatial)   — T̃(x,t_s) and x_norm
    cond      : (B, C_cond)         — (t̄_norm, A_norm, f_norm, R_c_norm)
    y_pred    : (B, Nx, out_channels) — predicted T̃(x, t_j)
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int = 2,
        out_channels: int = 1,
        n_layers: int = 4,
        cond_dim: int = 4,
        cond_hidden: int = 256,
        dropout: float = 0.0,
        spectral_dropout: float = 0.0,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.padding = 8  # pad spatial dim for non-periodic signals

        # Lift: (B, Nx, in_channels) → (B, Nx, width)
        self.linear_p = nn.Linear(in_channels, width)

        # Fourier layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes, spectral_dropout=spectral_dropout)
            for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        ])
        self.cin_layers = nn.ModuleList([
            ConditionalInstanceNorm1d(width) for _ in range(n_layers)
        ])

        # Conditioning MLP
        self.cond_mlp = ConditioningMLP(cond_dim, cond_hidden, n_layers, width)

        # Project: (B, Nx, width) → (B, Nx, out_channels)
        self.linear_q = nn.Linear(width, 128)
        self.output_layer = nn.Linear(128, out_channels)

        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_spatial, cond):
        """
        x_spatial : (B, Nx, in_channels)
        cond      : (B, cond_dim)
        returns   : (B, Nx, out_channels)
        """
        # Conditioning: compute all (γ_l, β_l) upfront
        cin_params = self.cond_mlp(cond)  # (B, n_layers, 2, width)

        # Lift
        x = self.linear_p(x_spatial)      # (B, Nx, width)
        x = x.permute(0, 2, 1)            # (B, width, Nx)

        Nx0 = x.size(-1)
        x = F.pad(x, (0, self.padding))   # (B, width, Nx + pad)

        # Fourier blocks
        for l in range(self.n_layers):
            gamma = cin_params[:, l, 0, :]  # (B, width)
            beta = cin_params[:, l, 1, :]   # (B, width)

            x1 = self.spectral_layers[l](x)
            x2 = self.conv_layers[l](x)
            x = x1 + x2
            x = self.cin_layers[l](x, gamma, beta)
            x = self.drop(self.activation(x))

        # Unpad + Project
        x = x[:, :, :Nx0]                 # (B, width, Nx)
        x = x.permute(0, 2, 1)            # (B, Nx, width)
        x = self.drop(self.activation(self.linear_q(x)))  # (B, Nx, 128)
        x = self.output_layer(x)           # (B, Nx, out_channels)
        return x
