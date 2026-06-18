import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Time-conditioned 2d FNO --------
#
# Operator-learning task:
# G(T(x, y, t_s), x, y, s_y, Q_y_bins, cond_static, forcing_seq) -> T(x, y, t_j)
#
# where t_bar = t_j - t_s is the lead time, t_s is the absolute source time, and
# T is the globally normalized temperature (using fixed mu_global, sig_global per training set).
# spatial input carries 4 base channels + 16 fixed temporal-forcing integral bins
# (Q_y_bins(x, y, k) = s_y(y) * ∫a(t)dt over the k-th subinterval of [t_s, t_j], /q_ref).
# h_a = TemporalForcingEncoder(forcing_seq) feeds two pathways in addition to the bins:
#   1. Spatial: z_a = W h_a, then forcing_field_k(x, y) = s_y(y) * z_{a,k} is concatenated
#      to spatial input as K extra channels (learned spatial pathway).
#   2. Global: [cond_static (15D), h_a (64D)] drives Conditional Instance Normalization.


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
        # Stored as real (..., 2) rather than cfloat: torch.distributed cannot
        # all-reduce complex tensors (gloo errors outright; NCCL goes through a
        # view-as-real path that is not numerically equivalent to the
        # single-process complex gradient). Keeping the leaf parameters real
        # makes DDP reduce them through its standard, correct path. forward()
        # recovers the complex weight via view_as_complex (a zero-copy view), so
        # the math — and the single-process result — is unchanged.
        self.weights1 = nn.Parameter(
            torch.view_as_real(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        )
        self.weights2 = nn.Parameter(
            torch.view_as_real(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
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
        # Real (..., 2) leaf params -> complex view for the mode-wise product.
        weights1 = torch.view_as_complex(self.weights1)
        weights2 = torch.view_as_complex(self.weights2)

        out_ft = torch.zeros(x.size(0), self.out_channels, Nx_freq, Ny_freq, dtype=torch.cfloat, device=x.device)
        # positive modes
        out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], weights1[:, :, :mx, :my])
        # negative modes
        # for rfft2, the last dimension is one-sided but the x dim. is still positive and negative
        # mx and my are still used for weights2 simply because the slice is still shape (B, C_in, mx, my) so its just
        # illustrating the size of the weight2 learnable tensor
        out_ft[:, :, -mx:, :my] = self.compl_mul2d(x_ft[:, :, -mx:, :my], weights2[:, :, :mx, :my])

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
    """Instance normalization + external affine (γ, β) from conditioning MLP.

    When ``valid_shape`` is passed, the per-(B, C) instance statistics are
    computed over the top-left ``(Nx0, Ny0)`` valid region only, so the
    domain-padding buffer (which after the first spectral block holds spurious
    globally-extrapolated values) does not pollute the normalization. The full
    tensor is still normalized with those valid-region statistics; the pad
    region is cropped downstream. With ``valid_shape=None`` this is identical to
    the native ``InstanceNorm2d`` path.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)

    def forward(self, x, gamma, beta, valid_shape: tuple[int, int] | None = None):
        # x: (B, C, Nx, Ny),  gamma/beta: (B, C)
        if valid_shape is None:
            out = self.norm(x)
        else:
            Nx0, Ny0 = valid_shape
            valid = x[:, :, :Nx0, :Ny0]
            mean = valid.mean(dim=(-2, -1), keepdim=True)
            var = valid.var(dim=(-2, -1), keepdim=True, unbiased=False)
            out = (x - mean) / torch.sqrt(var + self.eps)
        return gamma[:, :, None, None] * out + beta[:, :, None, None]


# --------- Conditioning MLP ---------

class ConditioningMLP(nn.Module):
    """Maps per-sample conditioning vector → per-layer CIN parameters (γ, β).

    Soft identity init: head weights are small random (std=1e-3) and bias=[1…1, 0…0]
    so γ≈1, β≈0 at start (model starts *near* an unconditioned FNO) while gradients
    still flow through the conditioning MLP and temporal encoder from step 0.
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

        # Soft identity init: small random weights + bias at γ=1, β=0.
        # Bias layout must match reshape(n_layers, 2, width):
        #   [γ_0(width), β_0(width), γ_1(width), β_1(width), ...]
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
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


# --------- Temporal Forcing Encoder ---------

class TemporalForcingEncoder(nn.Module):
    """forcing_seq: (B, M, token_dim) -> h_a: (B, embed_dim).

    Conv1d-based, mean + max pooling over the M temporal samples. Max pooling
    preserves localized pulse activations that mean alone would smear across
    the sample grid.
    """

    def __init__(self, token_dim: int = 5, hidden: int = 128, embed_dim: int = 64):
        super().__init__()
        self.lift = nn.Linear(token_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.proj = nn.Linear(2 * hidden, embed_dim)
        self.act = nn.GELU()

    def forward(self, z):
        # z: (B, M, token_dim)
        h = self.act(self.lift(z))             # (B, M, hidden)
        h = h.transpose(1, 2)                  # (B, hidden, M)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h_mean = h.mean(dim=-1)                # (B, hidden)
        h_max = h.amax(dim=-1)                 # (B, hidden)
        h = torch.cat([h_mean, h_max], dim=-1) # (B, 2*hidden)
        return self.act(self.proj(h))          # (B, embed_dim)


# --------- FNO2d ---------

class FNO2d(nn.Module):
    """Time-conditioned 2D Fourier Neural Operator.

    Forward signature:
        model(spatial, cond_static, forcing_seq) → y_pred

    spatial      : (B, Nx, Ny, 20)       — T̃(x, y, t_s), x_norm, y_norm, s_y, Q_y_bin_0..Q_y_bin_15
    cond_static  : (B, 15)               — t_bar_norm, t_s_norm, R_c_norm, spatial onehot+params, temporal onehot
    forcing_seq  : (B, M, token_dim)     — 5-D tokens sampled from a(t) over [t_s, t_j]
    y_pred       : (B, Nx, Ny, out_channels) — predicted T̃(x, y, t_j)

    Internally, h_a = TemporalForcingEncoder(forcing_seq) is projected to z_a ∈ R^K and
    s_y * z_a is concatenated as K extra spatial channels before the lift, so linear_p
    receives (in_channels + K) channels (e.g. 20 + 16 = 36 with the default config).
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        in_channels: int = 20,
        out_channels: int = 1,
        n_layers: int = 4,
        cond_static_dim: int = 15,
        cond_hidden: int = 256,
        temporal_token_dim: int = 5,
        temporal_hidden: int = 128,
        forcing_embed_dim: int = 64,
        forcing_spatial_dim: int = 16,
        dropout: float = 0.0,
        spectral_dropout: float = 0.0,
        use_temporal_encoder: bool = True,
        use_forcing_time_aug: bool = False,
        s_y_channel: int = 3,
        padding_reference_resolution: int | None = None,
        padding_mode: str = "zeros",
        cin_exclude_padding: bool = False,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.cond_static_dim = cond_static_dim
        self.temporal_token_dim = temporal_token_dim
        self.temporal_hidden = temporal_hidden
        self.forcing_embed_dim = forcing_embed_dim
        self.forcing_spatial_dim = forcing_spatial_dim
        self.use_temporal_encoder = use_temporal_encoder
        self.use_forcing_time_aug = use_forcing_time_aug
        self.s_y_channel = s_y_channel
        self.padding = 8  # pad spatial dim for non-periodic signals
        self.padding_reference_resolution = padding_reference_resolution
        if padding_mode not in ("zeros", "replicate", "reflect"):
            raise ValueError(
                f"padding_mode must be one of zeros|replicate|reflect, got {padding_mode!r}"
            )
        self.padding_mode = padding_mode
        self.cin_exclude_padding = cin_exclude_padding

        # Spatial-forcing channels (s_y * z_a) are only injected when the temporal
        # branch is active; with the encoder off the lift sees in_channels alone.
        lift_extra = forcing_spatial_dim if use_temporal_encoder else 0
        # Lift: (B, Nx, Ny, in_channels + K) → (B, Nx, Ny, width)
        self.linear_p = nn.Linear(in_channels + lift_extra, width)

        # Project temporal embedding h_a to K spatial-forcing weights; multiplied by s_y(y)
        # to form K extra spatial channels (restores the direct spatial pathway lost when
        # the hand-crafted Q_y_bins were removed).
        if use_temporal_encoder:
            self.forcing_to_spatial = nn.Linear(forcing_embed_dim, forcing_spatial_dim)

        # Time-augmented spatial forcing: fold [t_bar_norm, t_s_norm] into h_a before
        # projecting to spatial-forcing weights, so the learned field can vary with lead.
        if use_temporal_encoder and use_forcing_time_aug:
            self.forcing_aug_mlp = nn.Sequential(
                nn.Linear(forcing_embed_dim + 2, forcing_embed_dim),
                nn.GELU(),
                nn.Linear(forcing_embed_dim, forcing_embed_dim),
            )

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

        # Temporal forcing encoder + Conditioning MLP. With the encoder off, the
        # CIN is driven by cond_static alone (e.g. the source benchmark, which has
        # no forcing_seq).
        if use_temporal_encoder:
            self.temporal_encoder = TemporalForcingEncoder(
                token_dim=temporal_token_dim,
                hidden=temporal_hidden,
                embed_dim=forcing_embed_dim,
            )
            cond_dim = cond_static_dim + forcing_embed_dim
        else:
            cond_dim = cond_static_dim
        self.cond_mlp = ConditioningMLP(
            cond_dim=cond_dim,
            hidden_dim=cond_hidden,
            n_layers=n_layers,
            width=width,
        )

        # Project: (B, Nx, Ny, width) → (B, Nx, Ny, out_channels)
        self.linear_q = nn.Linear(width, 128)
        self.output_layer = nn.Linear(128, out_channels)

        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _padding_for_shape(self, Nx: int, Ny: int) -> tuple[int, int]:
        if self.padding_reference_resolution is None:
            return self.padding, self.padding
        ref = max(int(self.padding_reference_resolution) - 1, 1)
        pad_x = int(round(self.padding * max(int(Nx) - 1, 1) / ref))
        pad_y = int(round(self.padding * max(int(Ny) - 1, 1) / ref))
        return max(pad_x, 0), max(pad_y, 0)

    def forward(self, spatial, cond_static, forcing_seq=None):
        """
        spatial      : (B, Nx, Ny, in_channels)
        cond_static  : (B, cond_static_dim)
        forcing_seq  : (B, M, temporal_token_dim) or None when the temporal
                       encoder is disabled
        returns      : (B, Nx, Ny, out_channels)
        """
        assert spatial.size(-1) == self.in_channels

        if self.use_temporal_encoder:
            # Temporal branch
            h_a = self.temporal_encoder(forcing_seq)              # (B, forcing_embed_dim)

            # Spatial forcing injection: F_k(x, y) = s(y) * z_{a,k}. The s_y channel
            # index is representation-specific (forcing/source: 3; interfaces: 5).
            if self.use_forcing_time_aug:
                t_feats = cond_static[:, 0:2]                     # (B, 2) = [t_bar_norm, t_s_norm]
                h_aug = self.forcing_aug_mlp(torch.cat([h_a, t_feats], dim=-1))
                z_a = self.forcing_to_spatial(h_aug)              # (B, K)
            else:
                z_a = self.forcing_to_spatial(h_a)                # (B, K)
            s_y = spatial[..., self.s_y_channel:self.s_y_channel + 1]  # (B, Nx, Ny, 1)
            Nx, Ny = spatial.size(1), spatial.size(2)
            z_grid = z_a[:, None, None, :].expand(-1, Nx, Ny, -1) # (B, Nx, Ny, K)
            forcing_field = s_y * z_grid                          # (B, Nx, Ny, K)
            spatial_aug = torch.cat([spatial, forcing_field], dim=-1)  # (B, Nx, Ny, in_channels + K)

            # Conditioning MLP (h_a is also routed through CIN)
            cond_full = torch.cat([cond_static, h_a], dim=-1)     # (B, cond_static_dim + forcing_embed_dim)
        else:
            spatial_aug = spatial
            cond_full = cond_static

        cin_params = self.cond_mlp(cond_full)                 # (B, n_layers, 2, width)

        # Lift
        x = self.linear_p(spatial_aug)      # (B, Nx, Ny, width)
        x = x.permute(0, 3, 1, 2)            # (B, width, Nx, Ny)

        Nx0 = x.size(-2)
        Ny0 = x.size(-1)
        pad_x, pad_y = self._padding_for_shape(Nx0, Ny0)
        if self.padding_mode == "zeros":
            x = F.pad(x, (0, pad_y, 0, pad_x))   # zero (constant) buffer
        else:
            # replicate/reflect give a continuous extension at the edge, cutting
            # the spectral leakage that a hard zero step injects near the
            # boundary (and that sharpens with resolution).
            x = F.pad(x, (0, pad_y, 0, pad_x), mode=self.padding_mode)

        # Fourier blocks
        for l in range(self.n_layers):
            gamma = cin_params[:, l, 0, :]  # (B, width)
            beta = cin_params[:, l, 1, :]   # (B, width)

            x1 = self.spectral_layers[l](x)
            x2 = self.conv_layers[l](x)
            x = x1 + x2
            valid_shape = (Nx0, Ny0) if self.cin_exclude_padding else None
            x = self.cin_layers[l](x, gamma, beta, valid_shape=valid_shape)
            x = self.drop(self.activation(x))

        # Unpad + Project
        x = x[:, :, :Nx0, :Ny0]                 # (B, width, Nx, Ny)
        x = x.permute(0, 2, 3, 1)            # (B, Nx, Ny, width)
        x = self.drop(self.activation(self.linear_q(x)))  # (B, Nx, Ny, 128)
        x = self.output_layer(x)           # (B, Nx, Ny, out_channels)
        return x
