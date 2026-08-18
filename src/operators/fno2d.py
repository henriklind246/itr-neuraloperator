import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Time-conditioned 2d FNO --------
#
# Operator-learning task:
# G(T(x, y, t_s), cond_static, forcing_seq) -> T(x, y, t_j)
#
# where t_bar = t_j - t_s is the lead time and T is the globally normalized
# temperature (fixed mu_global, sig_global per training set).
#
# The concrete tensor shapes are NOT fixed here: every model-facing dim
# (in_channels, cond_static_dim, temporal_token_dim, s_y_channel, and the
# use_temporal_encoder / use_forcing_time_aug toggles) is owned by the
# ProblemDims of the active (benchmark, representation) pair and passed in at
# construction. See problems/base.py and problems/<benchmark>.py, and the
# contract table in CLAUDE.md / tests/test_problems.py.
#
# Two input representations share this model:
#   - temporal_encoder (default): the spatial input carries only lean base
#     channels (e.g. forcing: [T̃, x_norm, y_norm, s_y]); the temporal branch is
#     ON. h_a = TemporalForcingEncoder(forcing_seq) feeds two pathways:
#       1. Spatial: z_a = W h_a, then forcing_field_k(x, y) = s(y) * z_{a,k} is
#          concatenated to the spatial input as K learned forcing channels.
#       2. Global: [cond_static, h_a] drives Conditional Instance Normalization.
#     forcing_cond_mode gates which pathway h_a feeds (both | spatial_only |
#     cond_only), the Q1 forcing-routing ablation.
#     forcing_spatial_mode selects the spatial pathway implementation:
#       - broadcast: forcing_field_k(x, y) = s(y) * z_{a,k} (legacy/default)
#       - boundary_extender: boundary tokens [y, s(y), s(y)h_a, t_bar] are
#         cross-attended into learned domain pseudo-extensions. h_a is excluded
#         from the domain queries, so waveform information cannot bypass the
#         boundary pathway. An opt-in scalar-R_c ablation appends normalized
#         contact resistance to the domain queries only.
#       - physics_extender: the same extender with a learned, diffusion-inspired
#         per-head relative-y attention bias. The signed coefficient depends on
#         normalized query depth, lead time, interface crossing, and scalar R_c.
#   - bins: the spatial input additionally carries 16 fixed integral forcing
#     bins (Q_y_bins(x, y, k) = s(y) * ∫a(t)dt over the k-th subinterval of
#     [t_s, t_j], /q_ref); the temporal branch is OFF and forcing_seq is empty,
#     so CIN is driven by cond_static alone.


# --------- SpectralConv2d ---------

class SpectralConv2d(nn.Module):
    """2D Fourier convolution: FFT → mode-wise channel mixing → IFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

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

        # Inherit dtype/device from x_ft so float64 inputs keep complex128
        # precision instead of being silently downcast to complex64.
        out_ft = x_ft.new_zeros(x.size(0), self.out_channels, Nx_freq, Ny_freq)
        # positive modes
        out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft[:, :, :mx, :my], weights1[:, :, :mx, :my])
        # negative modes
        # for rfft2, the last dimension is one-sided but the x dim. is still positive and negative
        # mx and my are still used for weights2 simply because the slice is still shape (B, C_in, mx, my) so its just
        # illustrating the size of the weight2 learnable tensor
        out_ft[:, :, -mx:, :my] = self.compl_mul2d(x_ft[:, :, -mx:, :my], weights2[:, :, :mx, :my])

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


# --------- Boundary-to-domain forcing extender ---------

class BoundaryForcingExtender(nn.Module):
    """Lift a left-boundary forcing representation into domain-wide fields.

    ``h_a`` enters only through the boundary interaction ``s(y) * h_a``. The
    coarse domain queries contain normalized coordinates and the exact lead-time
    scalar supplied by ``cond_static[:, 0]``; the opt-in ablation also appends
    normalized scalar contact resistance. They never receive ``h_a``.

    Blocks follow ``a^(k) = a^(k-1) + FF^(k)(CA^(k)(a^(k-1), q^(0)))``: one
    additive residual per block and deliberately no normalization, so the
    identity path from ``domain_lift`` to ``output_projection`` is unbroken.
    An earlier post-LayerNorm variant put two normalized residuals in that path
    and became untrainable at ``depth=3`` under the standard 2.8e-3 peak LR.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        grid_size: int = 16,
        num_heads: int = 4,
        depth: int = 1,
        s_y_channel: int = 3,
        condition_on_rc: bool = False,
    ):
        super().__init__()
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be positive and divide embed_dim={embed_dim}"
            )
        if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2, 3):
            raise ValueError(
                f"depth must be an integer in {{1, 2, 3}}, got {depth!r}"
            )

        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.depth = depth
        self.s_y_channel = s_y_channel
        self.condition_on_rc = condition_on_rc

        self.boundary_lift = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.domain_lift = nn.Sequential(
            nn.Linear(3 + int(condition_on_rc), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(depth)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(depth)
        ])
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_dim),
        )
        self._init_norm_free_residual()

    def _init_norm_free_residual(self) -> None:
        # Torch's default Linear init (Kaiming-uniform with a=sqrt(5)) has a gain
        # near 0.3. A normalized stack hides that; this one cannot, and it
        # attenuated the boundary signal ~60x between the attention output and
        # the readout. Xavier with the GELU gain on pre-activation layers keeps
        # each block roughly variance-preserving, and zeroed biases stop a branch
        # from emitting a constant that dilutes its h_a-dependent part.
        gelu_gain = nn.init.calculate_gain("relu")
        for block in (
            self.boundary_lift,
            self.domain_lift,
            self.output_projection,
            *self.ffns,
        ):
            nn.init.xavier_uniform_(block[0].weight, gain=gelu_gain)
            nn.init.xavier_uniform_(block[2].weight)
            nn.init.zeros_(block[0].bias)
            nn.init.zeros_(block[2].bias)
        for attention in self.cross_attentions:
            # MultiheadAttention xavier-inits in_proj but leaves out_proj on the
            # default Linear init.
            nn.init.xavier_uniform_(attention.out_proj.weight)

    def enable_diffusion_geometry_bias(
        self,
        hidden_dim: int = 16,
        interface_x_norm: float = 0.5,
    ) -> None:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                "forcing_extender_physics_hidden must be a positive integer, "
                f"got {hidden_dim!r}"
            )
        if (
            isinstance(interface_x_norm, bool)
            or not isinstance(interface_x_norm, (int, float))
            or not 0.0 <= float(interface_x_norm) <= 1.0
        ):
            raise ValueError(
                "forcing_extender_interface_x_norm must be a finite number in "
                f"[0, 1], got {interface_x_norm!r}"
            )
        if getattr(self, "diffusion_geometry_bias_mlps", None) is not None:
            raise RuntimeError("diffusion geometry bias is already enabled")

        self.diffusion_geometry_interface_x_norm = float(interface_x_norm)
        self.diffusion_geometry_bias_mlps = nn.ModuleList()
        for _ in range(self.depth):
            mlp = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.num_heads),
            )
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            self.diffusion_geometry_bias_mlps.append(mlp)

    def _validate_rc_norm(self, rc_norm, batch_size: int) -> None:
        if rc_norm is None:
            raise ValueError(
                "diffusion geometry bias or condition_on_rc=True requires a "
                "normalized scalar R_c tensor."
            )
        if rc_norm.ndim != 2 or rc_norm.shape != (batch_size, 1):
            raise ValueError(
                "rc_norm must have shape (B, 1), got "
                f"{tuple(rc_norm.shape)} for batch size {batch_size}."
            )

    def _diffusion_geometry_context(self, x_norm, t_bar_norm, rc_norm):
        if x_norm.shape != t_bar_norm.shape or x_norm.shape != rc_norm.shape:
            raise ValueError(
                "x_norm, t_bar_norm, and rc_norm must have identical shapes, got "
                f"{tuple(x_norm.shape)}, {tuple(t_bar_norm.shape)}, and "
                f"{tuple(rc_norm.shape)}."
            )
        if x_norm.ndim < 2 or x_norm.shape[-1] != 1:
            raise ValueError(
                "diffusion geometry inputs must end in a singleton feature "
                f"dimension, got {tuple(x_norm.shape)}."
            )
        interface_x = self.diffusion_geometry_interface_x_norm
        crosses_interface = (x_norm > interface_x).to(dtype=x_norm.dtype)
        return torch.cat(
            [x_norm, t_bar_norm, crosses_interface, rc_norm], dim=-1
        )

    def diffusion_geometry_coefficients(self, x_norm, t_bar_norm, rc_norm):
        """Evaluate signed per-block, per-head geometry coefficients."""
        mlps = getattr(self, "diffusion_geometry_bias_mlps", None)
        if mlps is None:
            raise RuntimeError("diffusion geometry bias is not enabled")
        context = self._diffusion_geometry_context(
            x_norm, t_bar_norm, rc_norm
        )
        return torch.stack([mlp(context) for mlp in mlps], dim=0)

    def _build_diffusion_geometry_bias(
        self,
        block_index: int,
        domain_tokens,
        boundary_tokens,
        t_bar_norm,
        rc_norm,
    ):
        mlps = getattr(self, "diffusion_geometry_bias_mlps", None)
        if mlps is None:
            return None

        batch_size, num_queries = domain_tokens.shape[:2]
        self._validate_rc_norm(rc_norm, batch_size)
        x_norm = domain_tokens[..., 0:1]
        t_domain = t_bar_norm[:, None, :].expand(-1, num_queries, -1)
        rc_domain = rc_norm[:, None, :].expand(-1, num_queries, -1)
        context = self._diffusion_geometry_context(x_norm, t_domain, rc_domain)
        coefficients = mlps[block_index](context)

        domain_y = domain_tokens[..., 1]
        boundary_y = boundary_tokens[..., 0]
        delta_y_sq = (domain_y[:, :, None] - boundary_y[:, None, :]).square()
        bias = -coefficients.permute(0, 2, 1)[..., None] * delta_y_sq[:, None, :, :]
        return bias.reshape(
            batch_size * self.num_heads,
            num_queries,
            boundary_tokens.size(1),
        )

    def _build_boundary_tokens(self, h_a, spatial, t_bar_norm):
        # forcing benchmark invariant: normalized y is spatial channel 2 and
        # s(y) is broadcast in x, so the x=0 slice is the boundary function.
        y_boundary = spatial[:, 0, :, 2:3]
        s_y = spatial[:, 0, :, self.s_y_channel:self.s_y_channel + 1]
        Ny = spatial.size(2)
        h_boundary = s_y * h_a[:, None, :]
        t_boundary = t_bar_norm[:, None, :].expand(-1, Ny, -1)
        return torch.cat([y_boundary, s_y, h_boundary, t_boundary], dim=-1)

    def _build_domain_tokens(self, spatial, t_bar_norm, rc_norm=None):
        Nx, Ny = spatial.size(1), spatial.size(2)
        coarse_shape = (min(self.grid_size, Nx), min(self.grid_size, Ny))
        coords = spatial[..., 1:3].permute(0, 3, 1, 2)
        coords = F.interpolate(
            coords,
            size=coarse_shape,
            mode="bilinear",
            align_corners=True,
        )
        coords = coords.permute(0, 2, 3, 1).reshape(spatial.size(0), -1, 2)
        t_domain = t_bar_norm[:, None, :].expand(-1, coords.size(1), -1)
        features = [coords, t_domain]
        if self.condition_on_rc:
            self._validate_rc_norm(rc_norm, spatial.size(0))
            rc_domain = rc_norm[:, None, :].expand(-1, coords.size(1), -1)
            features.append(rc_domain)
        return torch.cat(features, dim=-1), coarse_shape

    def forward(self, h_a, spatial, t_bar_norm, rc_norm=None):
        """Return ``(B, Nx, Ny, out_dim)`` learned pseudo-extensions."""
        boundary_tokens = self._build_boundary_tokens(h_a, spatial, t_bar_norm)
        domain_tokens, coarse_shape = self._build_domain_tokens(
            spatial, t_bar_norm, rc_norm
        )

        boundary = self.boundary_lift(boundary_tokens)
        domain = self.domain_lift(domain_tokens)
        for block_index, (cross_attention, ffn) in enumerate(
            zip(self.cross_attentions, self.ffns)
        ):
            attention_bias = self._build_diffusion_geometry_bias(
                block_index, domain_tokens, boundary_tokens, t_bar_norm, rc_norm
            )
            attended, _ = cross_attention(
                query=domain,
                key=boundary,
                value=boundary,
                attn_mask=attention_bias,
                need_weights=False,
            )
            domain = domain + ffn(attended)

        coarse = self.output_projection(domain)
        coarse = coarse.reshape(
            spatial.size(0), coarse_shape[0], coarse_shape[1], self.out_dim
        ).permute(0, 3, 1, 2)
        extended = F.interpolate(
            coarse,
            size=(spatial.size(1), spatial.size(2)),
            mode="bilinear",
            align_corners=True,
        )
        return extended.permute(0, 2, 3, 1)


# --------- FNO2d ---------

class FNO2d(nn.Module):
    """Time-conditioned 2D Fourier Neural Operator.

    Forward signature:
        model(spatial, cond_static, forcing_seq) → y_pred

    spatial      : (B, Nx, Ny, in_channels)   — T̃(x, y, t_s) plus the active
                   representation's spatial channels (see the module header).
    cond_static  : (B, cond_static_dim)       — lead time and benchmark parameters
    forcing_seq  : (B, M, temporal_token_dim) — tokens sampled from a(t) over
                   [t_s, t_j]; empty (B, 0, 0) in bins mode / when the temporal
                   encoder is disabled
    y_pred       : (B, Nx, Ny, out_channels)  — predicted T̃(x, y, t_j)

    All of in_channels, cond_static_dim, and temporal_token_dim are supplied by
    the resolved ProblemDims for the active (benchmark, representation) pair; the
    ``__init__`` defaults below are placeholders that construction always
    overrides.

    When the temporal encoder's spatial route is active, exactly K forcing
    channels are concatenated before the lift. ``broadcast`` uses s(y) * z_a;
    ``boundary_extender`` replaces those fields with learned pseudo-extensions;
    ``physics_extender`` adds a diffusion-inspired geometry bias to the same
    cross-attention blocks. In every case linear_p receives
    (in_channels + K) channels.
    """

    # NOTE: the in_channels / cond_static_dim / temporal_token_dim defaults below
    # are legacy placeholders (they do not match any current representation). The
    # training and eval paths always pass the resolved ProblemDims values, so the
    # defaults exist only to keep the bare constructor callable.
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
        use_temporal_encoder: bool = True,
        use_forcing_time_aug: bool = False,
        forcing_cond_mode: str = "both",
        forcing_spatial_mode: str = "broadcast",
        forcing_extender_grid_size: int = 16,
        forcing_extender_heads: int = 4,
        forcing_extender_depth: int = 1,
        forcing_extender_condition_on_rc: bool = False,
        forcing_extender_rc_cond_index: int | None = None,
        forcing_extender_physics_hidden: int = 16,
        forcing_extender_interface_x_norm: float = 0.5,
        s_y_channel: int = 3,
        padding_reference_resolution: int | None = None,
        padding_mode: str = "zeros",
        cin_exclude_padding: bool = False,
        hard_right_dirichlet: bool = False,
        t_right_norm: float = 0.0,
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
        # Q1 forcing-routing ablation. The temporal embedding h_a normally feeds
        # two pathways: the spatial injection s_y * z_a and the CIN conditioning
        # (h_a concatenated into cond_full). This gate isolates them:
        #   both        -> h_a drives spatial injection and CIN (default)
        #   spatial_only -> h_a drives only spatial injection; CIN sees cond_static
        #   cond_only    -> h_a drives only CIN; no spatial forcing channels
        if forcing_cond_mode not in ("both", "spatial_only", "cond_only"):
            raise ValueError(
                f"forcing_cond_mode must be one of both|spatial_only|cond_only, "
                f"got {forcing_cond_mode!r}"
            )
        if forcing_cond_mode != "both" and not use_temporal_encoder:
            raise ValueError(
                "forcing_cond_mode only applies with use_temporal_encoder=True; "
                f"got {forcing_cond_mode!r} with the encoder disabled."
            )
        self.forcing_cond_mode = forcing_cond_mode
        self._forcing_to_spatial = use_temporal_encoder and forcing_cond_mode in ("both", "spatial_only")
        self._forcing_to_cond = use_temporal_encoder and forcing_cond_mode in ("both", "cond_only")
        extender_modes = ("boundary_extender", "physics_extender")
        if forcing_spatial_mode not in ("broadcast", *extender_modes):
            raise ValueError(
                "forcing_spatial_mode must be one of "
                "broadcast|boundary_extender|physics_extender, "
                f"got {forcing_spatial_mode!r}"
            )
        if forcing_extender_condition_on_rc:
            if forcing_spatial_mode not in extender_modes:
                raise ValueError(
                    "forcing_extender_condition_on_rc=True requires "
                    "forcing_spatial_mode='boundary_extender' or "
                    "'physics_extender'."
                )
        requires_rc_index = (
            forcing_extender_condition_on_rc
            or forcing_spatial_mode == "physics_extender"
        )
        if requires_rc_index:
            if (
                isinstance(forcing_extender_rc_cond_index, bool)
                or not isinstance(forcing_extender_rc_cond_index, int)
                or not 0 <= forcing_extender_rc_cond_index < cond_static_dim
            ):
                raise ValueError(
                    "the configured forcing extender requires an eligible "
                    "forcing_extender_rc_cond_index in cond_static; got "
                    f"{forcing_extender_rc_cond_index!r} for cond_static_dim="
                    f"{cond_static_dim}."
                )
        if forcing_spatial_mode in extender_modes:
            if not use_temporal_encoder:
                raise ValueError(
                    f"{forcing_spatial_mode} requires use_temporal_encoder=True."
                )
            if forcing_cond_mode == "cond_only":
                raise ValueError(
                    f"{forcing_spatial_mode} requires a spatial forcing route; "
                    "forcing_cond_mode cannot be 'cond_only'."
                )
            if not use_forcing_time_aug:
                raise ValueError(
                    f"{forcing_spatial_mode} requires use_forcing_time_aug=True so it "
                    "receives cond_static[:, 0] as normalized lead time."
                )
        self.forcing_spatial_mode = forcing_spatial_mode
        self.forcing_extender_grid_size = forcing_extender_grid_size
        self.forcing_extender_heads = forcing_extender_heads
        if (
            isinstance(forcing_extender_depth, bool)
            or not isinstance(forcing_extender_depth, int)
            or forcing_extender_depth not in (1, 2, 3)
        ):
            raise ValueError(
                "forcing_extender_depth must be an integer in {1, 2, 3}, "
                f"got {forcing_extender_depth!r}"
            )
        self.forcing_extender_depth = forcing_extender_depth
        self.forcing_extender_condition_on_rc = forcing_extender_condition_on_rc
        self.forcing_extender_rc_cond_index = forcing_extender_rc_cond_index
        self.forcing_extender_physics_hidden = forcing_extender_physics_hidden
        self.forcing_extender_interface_x_norm = forcing_extender_interface_x_norm
        self.s_y_channel = s_y_channel
        self.padding = 8  # pad spatial dim for non-periodic signals
        self.padding_reference_resolution = padding_reference_resolution
        if padding_mode not in ("zeros", "replicate", "reflect"):
            raise ValueError(
                f"padding_mode must be one of zeros|replicate|reflect, got {padding_mode!r}"
            )
        self.padding_mode = padding_mode
        self.cin_exclude_padding = cin_exclude_padding

        # Hard right-Dirichlet output constraint: overwrite the right wall face
        # T[:, Nx-1, :] with the (normalized) fixed boundary temperature so the
        # BC holds exactly rather than only in the soft physics loss. The buffer
        # is persistent (round-trips through checkpoints and applies at eval);
        # when disabled no buffer is registered so old checkpoints load strictly.
        self.hard_right_dirichlet = hard_right_dirichlet
        if hard_right_dirichlet:
            self.register_buffer(
                "t_right_norm", torch.tensor(float(t_right_norm), dtype=torch.float32)
            )

        # The spatial route always contributes K channels, preserving the lift
        # contract across broadcast and extender experiments.
        lift_extra = forcing_spatial_dim if self._forcing_to_spatial else 0
        # Lift: (B, Nx, Ny, in_channels + K) → (B, Nx, Ny, width)
        self.linear_p = nn.Linear(in_channels + lift_extra, width)

        # The spatial route contributes exactly K forcing channels. The learned
        # extender replaces the legacy broadcasts rather than supplementing them.
        if self._forcing_to_spatial:
            if forcing_spatial_mode == "broadcast":
                self.forcing_to_spatial = nn.Linear(forcing_embed_dim, forcing_spatial_dim)
            else:
                self.boundary_extender = BoundaryForcingExtender(
                    embed_dim=forcing_embed_dim,
                    out_dim=forcing_spatial_dim,
                    grid_size=forcing_extender_grid_size,
                    num_heads=forcing_extender_heads,
                    depth=forcing_extender_depth,
                    s_y_channel=s_y_channel,
                    condition_on_rc=forcing_extender_condition_on_rc,
                )

        # Time-augmented spatial forcing: fold t_bar_norm into h_a before
        # projecting to spatial-forcing weights, so the learned field can vary with lead.
        if (
            self._forcing_to_spatial
            and forcing_spatial_mode == "broadcast"
            and use_forcing_time_aug
        ):
            self.forcing_aug_mlp = nn.Sequential(
                nn.Linear(forcing_embed_dim + 1, forcing_embed_dim),
                nn.GELU(),
                nn.Linear(forcing_embed_dim, forcing_embed_dim),
            )

        # ------- FOURIER LAYERS -------------
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)
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
            cond_dim = cond_static_dim + forcing_embed_dim if self._forcing_to_cond else cond_static_dim
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

        # Register the optional MLPs after every shared parameter has been
        # initialized. Otherwise their random hidden layers would advance the
        # RNG and invalidate same-seed ordinary-vs-physics initialization.
        if forcing_spatial_mode == "physics_extender":
            self.boundary_extender.enable_diffusion_geometry_bias(
                hidden_dim=forcing_extender_physics_hidden,
                interface_x_norm=forcing_extender_interface_x_norm,
            )

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

            # Spatial forcing injection is skipped in cond_only mode. Extender
            # mode replaces the K broadcasts rather than adding another route.
            if self._forcing_to_spatial:
                if self.forcing_spatial_mode in ("boundary_extender", "physics_extender"):
                    t_bar_norm = cond_static[:, 0:1]
                    rc_norm = None
                    if (
                        self.forcing_extender_condition_on_rc
                        or self.forcing_spatial_mode == "physics_extender"
                    ):
                        rc_index = self.forcing_extender_rc_cond_index
                        rc_norm = cond_static[:, rc_index:rc_index + 1]
                    forcing_field = self.boundary_extender(
                        h_a, spatial, t_bar_norm, rc_norm
                    )
                else:
                    if self.use_forcing_time_aug:
                        t_feats = cond_static[:, 0:1]             # (B, 1) = [t_bar_norm]
                        h_aug = self.forcing_aug_mlp(torch.cat([h_a, t_feats], dim=-1))
                        z_a = self.forcing_to_spatial(h_aug)      # (B, K)
                    else:
                        z_a = self.forcing_to_spatial(h_a)        # (B, K)
                    s_y = spatial[..., self.s_y_channel:self.s_y_channel + 1]
                    Nx, Ny = spatial.size(1), spatial.size(2)
                    z_grid = z_a[:, None, None, :].expand(-1, Nx, Ny, -1)
                    forcing_field = s_y * z_grid                  # (B, Nx, Ny, K)
                spatial_aug = torch.cat([spatial, forcing_field], dim=-1)  # (B, Nx, Ny, in_channels + K)
            else:
                spatial_aug = spatial

            # Conditioning MLP. h_a is routed through CIN unless the ablation
            # restricts forcing to the spatial pathway (spatial_only).
            if self._forcing_to_cond:
                cond_full = torch.cat([cond_static, h_a], dim=-1)  # (B, cond_static_dim + forcing_embed_dim)
            else:
                cond_full = cond_static
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

        if self.hard_right_dirichlet:
            # Replace the right wall face T[:, Nx-1, :] (dim 1 = Nx) with the
            # constant boundary value. Differentiable, no in-place: the constant
            # column carries zero gradient, so the soft right_dirichlet residual
            # becomes an exact-zero diagnostic.
            edge = self.t_right_norm.to(x.dtype).expand(
                x.shape[0], 1, x.shape[2], x.shape[3]
            )
            x = torch.cat([x[:, :-1], edge], dim=1)
        return x
