from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Continuous Vision Transformer (CViT), PyTorch port ----------------------
#
# Physics-informed operator for the diffusion benchmark. The encoder ingests a
# per-sim field on the grid (the initial condition, 1 channel) and produces a
# token sequence; the decoder queries arbitrary continuous coordinates
# (x, y, t) against those tokens and returns the normalized temperature
# T_tilde = (T - mu) / sigma at each query.
#
# Ported from a JAX/Equinox reference (patch-embed encoder + Fourier-coordinate
# / time-FiLM cross-attention decoder). Two deliberate deviations from a naive
# port, both required for physics-only training:
#
#   1. Explicit softmax attention (not fused SDPA). The PDE residual takes
#      autograd.grad(..., create_graph=True) w.r.t. the query coordinates, i.e.
#      a double backward through the decoder. Fused scaled_dot_product_attention
#      (flash / mem-efficient) kernels raise on double backward, so attention is
#      written out explicitly here. It is used in both encoder and decoder for
#      simplicity; only the decoder strictly needs it.
#
#   2. Hard right-Dirichlet ansatz. Instead of penalizing the x=1 wall, the raw
#      network output is multiplied by g(x) = (1 - x) and offset by
#      t_right_tilde, so T=300 K holds exactly at x=1 for all (y, t).


# --------- activations ---------

class Snake(nn.Module):
    """Snake activation x + (1/alpha) sin^2(alpha x); alpha is trainable."""

    def __init__(self, dim: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        return x + (torch.sin(a * x) ** 2) / a


def _make_activation(name: str, dim: int) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "snake":
        return Snake(dim)
    raise ValueError(f"Unknown activation {name!r}")


# --------- 2D sin-cos positional embedding ---------

def _sincos_pos_embed_2d(emb_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Return a (grid_h*grid_w, emb_dim) fixed 2D sin-cos positional embedding."""
    if emb_dim % 4 != 0:
        raise ValueError(
            f"emb_dim must be divisible by 4 for 2D sin-cos pos-embed, got {emb_dim}."
        )
    grid_y = np.arange(grid_h, dtype=np.float32)
    grid_x = np.arange(grid_w, dtype=np.float32)
    gx, gy = np.meshgrid(grid_x, grid_y)  # each (grid_h, grid_w)
    quarter = emb_dim // 4
    omega = np.arange(quarter, dtype=np.float32) / float(quarter)
    omega = 1.0 / (10000.0 ** omega)  # (quarter,)

    def _embed(pos: np.ndarray) -> np.ndarray:
        out = pos.reshape(-1)[:, None] * omega[None, :]  # (N, quarter)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (N, emb_dim//2)

    emb = np.concatenate([_embed(gy), _embed(gx)], axis=1)  # (N, emb_dim)
    return torch.from_numpy(emb.astype(np.float32))


# --------- MLPs ---------

class MlpBlock(nn.Module):
    """Transformer feed-forward: Linear -> act -> Linear."""

    def __init__(self, dim: int, hidden_dim: int, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = _make_activation(activation, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MLP(nn.Module):
    """Configurable-depth MLP used for the time-FiLM net and the output head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "gelu",
        zero_init_last: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(_make_activation(activation, hidden_dim))
            d = hidden_dim
        last = nn.Linear(d, out_dim)
        if zero_init_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------- explicit (double-backward-safe) multi-head attention ---------

class MultiHeadAttention(nn.Module):
    """Explicit softmax attention: softmax(QK^T / sqrt(d_head)) V.

    Written out (no fused SDPA) so that a second-order autograd.grad through the
    decoder — needed for the PDE residual — does not hit a kernel that rejects
    double backward. q:(B, Nq, D), kv:(B, Nk, D) -> (B, Nq, D).
    """

    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"emb_dim {emb_dim} not divisible by num_heads {num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = q.shape
        qh = self._split(self.q_proj(q))   # (B, H, Nq, hd)
        kh = self._split(self.k_proj(kv))  # (B, H, Nk, hd)
        vh = self._split(self.v_proj(kv))  # (B, H, Nk, hd)
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)       # (B, H, Nq, hd)
        out = out.transpose(1, 2).reshape(B, Nq, self.num_heads * self.head_dim)
        return self.out_proj(out)


# --------- patch embedding ---------

class PatchEmbed2d(nn.Module):
    """Conv2d patch embed; (B, in_ch, H, W) -> (B, H/p * W/p, emb_dim)."""

    def __init__(self, in_ch: int, emb_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        H, W = u.shape[-2], u.shape[-1]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"grid ({H}x{W}) not divisible by patch_size {self.patch_size}."
            )
        x = self.proj(u)  # (B, emb_dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (B, N, emb_dim)


# --------- Fourier coordinate embedding ---------

class FourierEmbed(nn.Module):
    """Random Fourier features: x -> [cos(x @ B), sin(x @ B)].

    The projection ``B`` is a frozen random matrix; ``freq_scale`` sets its
    standard deviation and is a real hyperparameter (it controls the spectral
    bias of the coordinate embedding). ``out_dim`` must be even.
    """

    def __init__(self, in_dim: int, out_dim: int, freq_scale: float):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"FourierEmbed out_dim must be even, got {out_dim}.")
        kernel = torch.randn(in_dim, out_dim // 2) * float(freq_scale)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.kernel  # (..., out_dim//2)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# --------- transformer blocks ---------

class SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + MLP residual block (encoder)."""

    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MlpBlock(emb_dim, int(emb_dim * mlp_ratio), activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h)
        y = self.norm2(x)
        return x + self.mlp(y)


class CrossAttnBlock(nn.Module):
    """Cross-attention block with time-FiLM modulation on the query stream.

    q_inputs are the coordinate queries; kv_inputs are the encoder tokens.
    ``scale``/``shift`` are per-block FiLM parameters produced from the time
    embedding; they modulate the normalized query before attention.
    """

    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float, activation: str):
        super().__init__()
        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_kv = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm_out = nn.LayerNorm(emb_dim)
        self.mlp = MlpBlock(emb_dim, int(emb_dim * mlp_ratio), activation)

    def forward(
        self,
        q_inputs: torch.Tensor,
        kv_inputs: torch.Tensor,
        scale: torch.Tensor | None = None,
        shift: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(q_inputs)
        if scale is not None:
            q = q * (1.0 + scale)
        if shift is not None:
            q = q + shift
        kv = self.norm_kv(kv_inputs)
        x = self.attn(q, kv) + q_inputs
        y = self.norm_out(x)
        return x + self.mlp(y)


# --------- encoder / decoder ---------

class CViTEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        emb_dim: int,
        patch_size: int,
        grid_size: tuple[int, int],
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        activation: str,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2d(in_ch, emb_dim, patch_size)
        grid_h = grid_size[0] // patch_size
        grid_w = grid_size[1] // patch_size
        pos = _sincos_pos_embed_2d(emb_dim, grid_h, grid_w)  # (N, emb_dim)
        self.register_buffer("pos_emb", pos.unsqueeze(0))    # (1, N, emb_dim)
        self.blocks = nn.ModuleList(
            [SelfAttnBlock(emb_dim, num_heads, mlp_ratio, activation) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(u) + self.pos_emb
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class CViTDecoder(nn.Module):
    def __init__(
        self,
        enc_emb_dim: int,
        dec_emb_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        fourier_freq: float,
        activation: str,
        head_layers: int = 2,
        fourier_freq_t: float | None = None,
    ):
        super().__init__()
        self.depth = depth
        self.dec_emb_dim = dec_emb_dim
        # Spatial (x, y in [0, 1]) and temporal (t in [0, t_final]) coordinates
        # live on different ranges, so their Fourier feature scales are decoupled.
        # ``fourier_freq_t`` defaults to ``fourier_freq`` when unset; on the
        # diffusion window t_final=0.3 the shared value 1.0 makes fourier_t nearly
        # constant across the trajectory (near time-blind decoder), so raise it.
        freq_t = float(fourier_freq if fourier_freq_t is None else fourier_freq_t)
        self.fourier_x = FourierEmbed(2, dec_emb_dim, fourier_freq)
        self.fourier_t = FourierEmbed(1, dec_emb_dim, freq_t)
        # Time-FiLM: t-embedding -> per-block (scale, shift). Last layer zero-init
        # so at init the decoder starts from an unmodulated (identity) query.
        self.time_film = MLP(
            in_dim=dec_emb_dim,
            hidden_dim=dec_emb_dim,
            out_dim=depth * 2 * dec_emb_dim,
            num_layers=2,
            activation=activation,
            zero_init_last=True,
        )
        self.proj_kv = nn.Linear(enc_emb_dim, dec_emb_dim)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(dec_emb_dim, num_heads, mlp_ratio, activation) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dec_emb_dim)
        self.head = MLP(dec_emb_dim, dec_emb_dim, out_dim, head_layers, activation)

    def forward(
        self, tokens: torch.Tensor, coords: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # tokens:(B, Nk, enc_emb); coords:(B, Nq, 2); t:(B, Nq, 1)
        B, Nq, _ = coords.shape
        queries = self.fourier_x(coords)          # (B, Nq, dec_emb)
        t_emb = self.fourier_t(t)                 # (B, Nq, dec_emb)
        kv = self.proj_kv(tokens)                 # (B, Nk, dec_emb)
        film = self.time_film(t_emb).view(B, Nq, self.depth, 2, self.dec_emb_dim)
        for i, block in enumerate(self.blocks):
            scale = film[:, :, i, 0, :]
            shift = film[:, :, i, 1, :]
            queries = block(queries, kv, scale=scale, shift=shift)
        queries = self.norm(queries)
        return self.head(queries)                 # (B, Nq, out_dim)


# --------- full model ---------

class CViT(nn.Module):
    """Continuous Vision Transformer with a hard right-Dirichlet ansatz.

    forward(u, coords, t):
        u:      (B, in_ch, H, W)          per-sim field (the IC), on the grid
        coords: (B or 1, Nq, 2)           physical (x, y) in [0, 1]
        t:      (B or 1, Nq, 1)           physical time in [0, t_final]
        ->      (B, Nq, out_dim)          T_tilde = (T - mu) / sigma

    When ``hard_right_dirichlet`` is set, the output is
    ``t_right_tilde + (1 - x) * NN(u, x, y, t)`` so the x=1 Dirichlet wall is
    satisfied exactly; otherwise the raw network output is returned.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_dim: int = 1,
        emb_dim: int = 256,
        dec_emb_dim: int | None = None,
        patch_size: int = 10,
        grid_size: tuple[int, int] = (100, 100),
        depth_enc: int = 4,
        depth_dec: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        fourier_freq: float = 1.0,
        fourier_freq_t: float | None = None,
        activation: str = "gelu",
        hard_right_dirichlet: bool = True,
        t_right_tilde: float = 0.0,
        t_final: float = 1.0,
        hard_left_flux: bool = False,
        left_flux_scale: float = 1.0,
    ):
        super().__init__()
        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.hard_right_dirichlet = bool(hard_right_dirichlet)
        self.hard_left_flux = bool(hard_left_flux)
        self.encoder = CViTEncoder(
            in_ch=in_ch,
            emb_dim=emb_dim,
            patch_size=patch_size,
            grid_size=grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
        )
        self.decoder = CViTDecoder(
            enc_emb_dim=emb_dim,
            dec_emb_dim=dec_emb_dim,
            out_dim=out_dim,
            depth=depth_dec,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            fourier_freq=fourier_freq,
            fourier_freq_t=fourier_freq_t,
            activation=activation,
        )
        self.register_buffer(
            "t_right_tilde", torch.tensor(float(t_right_tilde), dtype=torch.float32)
        )
        # Decoder time is normalized to [0, 1] by dividing the physical query time
        # by t_final *inside* forward. This keeps the temporal Fourier features on
        # the same [0, 1] range as the spatial ones (so a shared fourier_freq is
        # not near time-blind on a small window like t_final=0.3). The division is
        # in-graph, so the PDE residual — which differentiates w.r.t. the physical
        # time leaf — stays exact: autograd's chain rule folds in the 1/t_final
        # factor to yield the true physical T_t. (Rescaling the collocation leaf
        # itself instead would drop that factor and corrupt the residual.) Default
        # 1.0 is a no-op, preserving legacy behavior.
        self.register_buffer(
            "t_norm", torch.tensor(float(t_final), dtype=torch.float32)
        )
        # 1 / (k * sigma): converts an inward left-wall flux q_L into the
        # normalized-temperature slope dT_tilde/dx it must produce. Only read
        # when hard_left_flux is on.
        self.register_buffer(
            "left_flux_scale",
            torch.tensor(float(left_flux_scale), dtype=torch.float32),
        )

    def forward(
        self,
        u: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
        q_left: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = u.shape[0]
        # torch attention does not broadcast batch dims: expand shared dim-1
        # coordinate/time sets to the sim minibatch explicitly.
        if coords.shape[0] == 1 and B > 1:
            coords = coords.expand(B, -1, -1)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1, -1)
        tokens = self.encoder(u)
        raw = self.decoder(tokens, coords, t / self.t_norm)
        if not self.hard_right_dirichlet:
            return raw
        x = coords[..., 0:1]
        if self.hard_left_flux:
            # T = t_right + g*(x - 1) + (1 - x^2)*raw, with g = -q_L/(k*sigma).
            # x=1: both added terms vanish, so the right-Dirichlet wall is kept.
            # x=0: T_x = g + raw_x(0), so the inhomogeneous left-flux condition
            # T_x(0) = g is met by construction and the bc_left residual collapses
            # to the forcing-independent raw_x(0). q_left is a detached, coords-
            # aligned (B, Nq, 1) tensor (None -> g=0, e.g. the t=0 IC where the
            # ramp gives q_L=0), so autograd sees g only as an x-linear term.
            g = torch.zeros_like(x) if q_left is None else -q_left * self.left_flux_scale
            return self.t_right_tilde + g * (x - 1.0) + (1.0 - x * x) * raw
        return self.t_right_tilde + (1.0 - x) * raw


class ForcingCViT(CViT):
    """CViT whose encoder ingests the boundary forcing as a space-time image.

    Identical architecture to :class:`CViT` (patch-embed encoder + Fourier /
    time-FiLM cross-attention decoder + hard right-Dirichlet ansatz), reused
    verbatim; the ONLY difference is the meaning of the encoder input ``u``.

    For the single-slab, forcing-driven benchmark the initial condition is a
    fixed uniform 300 K field, so it carries no per-sim information; the per-sim
    signal is the inhomogeneous left-wall Neumann flux ``q_L(y, t) = a(t)*s(y)``.
    The encoder is therefore conditioned on ``q_L`` rendered as a **boundary
    space-time image**, NOT a 2D spatial (x, y) field:

        u: (B, 1, Ny, Nt)   with axes (rows = left-wall y-nodes, cols = time)
                            and a single channel ``u_enc(y, t) = q_L(y, t)/A_ref``.

    ``grid_size`` here is ``(Ny, Nt)`` (y-resolution x time-resolution of that
    image), so the patch grid is ``(Ny/patch, Nt/patch)``. Do NOT feed a
    temperature or geometry field: the second image axis is time, not x.

    A single input channel is deliberate — the scientific question is whether the
    forcing function image alone suffices; extra channels (``a(t)``, ``s(y)``,
    family one-hots) are only added if the 1-channel version demonstrably fails.

    ``forward(u, coords, t)`` is inherited unchanged, so ``pde_residual.model_xyt``
    and the PINO trainer call it exactly like ``CViT``.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_dim: int = 1,
        emb_dim: int = 256,
        dec_emb_dim: int | None = None,
        patch_size: int = 10,
        grid_size: tuple[int, int] = (100, 100),
        depth_enc: int = 4,
        depth_dec: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        fourier_freq: float = 1.0,
        fourier_freq_t: float | None = None,
        activation: str = "gelu",
        hard_right_dirichlet: bool = True,
        t_right_tilde: float = 0.0,
        t_final: float = 1.0,
        hard_left_flux: bool = False,
        left_flux_scale: float = 1.0,
    ):
        # grid_size is the (Ny, Nt) forcing-image resolution; every block,
        # including the double-backward-safe explicit-attention encoder and the
        # Conv2d PatchEmbed2d (no fused SDPA), is reused from CViT unchanged.
        super().__init__(
            in_ch=in_ch,
            out_dim=out_dim,
            emb_dim=emb_dim,
            dec_emb_dim=dec_emb_dim,
            patch_size=patch_size,
            grid_size=grid_size,
            depth_enc=depth_enc,
            depth_dec=depth_dec,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            fourier_freq=fourier_freq,
            fourier_freq_t=fourier_freq_t,
            activation=activation,
            hard_right_dirichlet=hard_right_dirichlet,
            t_right_tilde=t_right_tilde,
            t_final=t_final,
            hard_left_flux=hard_left_flux,
            left_flux_scale=left_flux_scale,
        )


# --------- interface-benchmark token encoders ---------

class ForcingTokenEncoder(nn.Module):
    """Sampled forcing waveform ``forcing_seq (B, M, token_dim)`` -> forcing
    tokens ``z_f (B, num_tokens, emb_dim)``.

    A lightweight conv1d + mean/max-pool encoder (the FNO's
    ``TemporalForcingEncoder`` pattern, ``fno2d.py:181``), NOT a vision
    transformer: it consumes the same sampled ``a(t)`` function values the FNO
    sees and emits CViT-width cross-attention tokens. The baseline
    ``num_tokens=1`` projects the pooled schedule summary to a single token; a
    larger count adds temporally-resolved read-out tokens. Max pooling keeps
    localized pulse activations that mean pooling would smear.
    """

    def __init__(
        self,
        token_dim: int,
        emb_dim: int,
        num_tokens: int = 1,
        hidden: int = 128,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.emb_dim = int(emb_dim)
        self.lift = nn.Linear(token_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.proj = nn.Linear(2 * hidden, self.num_tokens * self.emb_dim)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        h = self.act(self.lift(z))          # (B, M, hidden)
        h = h.transpose(1, 2)               # (B, hidden, M)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h_mean = h.mean(dim=-1)             # (B, hidden)
        h_max = h.amax(dim=-1)              # (B, hidden)
        h = torch.cat([h_mean, h_max], dim=-1)
        out = self.proj(h)                  # (B, num_tokens*emb_dim)
        return out.view(B, self.num_tokens, self.emb_dim)


class ParamTokenEncoder(nn.Module):
    """Interface scalars ``[x_hat, Rc_hat] (B, n_scalars)`` -> parameter tokens
    ``z_p (B, num_tokens, emb_dim)``.

    ``R_c`` appears in no spatial field, so it must reach the model through this
    branch; ``x_hat`` gives the exact (unpatched) interface location. The raw
    generator parameters never enter here — only the two physically-available
    interface scalars.
    """

    def __init__(
        self,
        n_scalars: int,
        emb_dim: int,
        num_tokens: int = 1,
        hidden: int = 128,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.emb_dim = int(emb_dim)
        self.net = MLP(
            in_dim=n_scalars,
            hidden_dim=hidden,
            out_dim=self.num_tokens * self.emb_dim,
            num_layers=2,
            activation=activation,
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        B = p.shape[0]
        return self.net(p).view(B, self.num_tokens, self.emb_dim)


class InterfaceCViT(nn.Module):
    """Multimodal token-fusion CViT for the ``interfaces`` benchmark.

    All sample-specific physics is turned into decoder cross-attention tokens
    from three heterogeneous streams; the decoder stays a pure coordinate+time
    query with the existing time-only FiLM (``CViTDecoder``). The model is
    conditioned only on information available at inference — the interface
    geometry ``(x_Gamma, R_c)`` and the sampled applied-flux waveform ``a(.)`` —
    never the synthetic generator parameters, so it learns the operator
    ``(x_Gamma, R_c, a(.)) -> T``.

    Streams (each -> ``emb_dim``-wide tokens, plus a learned 3-way modality
    embedding):
      1. spatial  ``z_s``: full ``CViTEncoder`` over ``[T0_tilde, K_norm, D_norm]``.
      2. forcing  ``z_f``: ``ForcingTokenEncoder`` over ``forcing_seq (B,128,2)``.
      3. param    ``z_p``: ``ParamTokenEncoder`` over ``[x_hat, Rc_hat]``.

    ``encode(u_spatial, forcing_seq, params)`` builds the cached latent token set
    (query-independent, so run once per sim); ``decode(latent, coords, t)`` is the
    pure coordinate+time query reused for both CN time slices. The hard
    right-Dirichlet ansatz ``T = t_right_tilde + (1 - x) * raw`` keeps the x=1
    wall exact; the left-wall Neumann is enforced softly by the FV residual (no
    hard-left-flux lifting).
    """

    def __init__(
        self,
        spatial_in_ch: int = 3,
        out_dim: int = 1,
        emb_dim: int = 256,
        dec_emb_dim: int | None = None,
        patch_size: int = 10,
        grid_size: tuple[int, int] = (100, 100),
        depth_enc: int = 4,
        depth_dec: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        fourier_freq: float = 1.0,
        fourier_freq_t: float | None = None,
        activation: str = "gelu",
        hard_right_dirichlet: bool = True,
        t_right_tilde: float = 0.0,
        t_final: float = 1.0,
        temporal_token_dim: int = 2,
        temporal_samples: int = 128,
        num_forcing_tokens: int = 1,
        forcing_hidden: int = 128,
        n_param_scalars: int = 2,
        num_param_tokens: int = 1,
        param_hidden: int = 128,
    ):
        super().__init__()
        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.hard_right_dirichlet = bool(hard_right_dirichlet)
        self.temporal_samples = int(temporal_samples)
        self.spatial_encoder = CViTEncoder(
            in_ch=int(spatial_in_ch),
            emb_dim=emb_dim,
            patch_size=patch_size,
            grid_size=grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
        )
        self.forcing_encoder = ForcingTokenEncoder(
            token_dim=int(temporal_token_dim),
            emb_dim=emb_dim,
            num_tokens=int(num_forcing_tokens),
            hidden=int(forcing_hidden),
        )
        self.param_encoder = ParamTokenEncoder(
            n_scalars=int(n_param_scalars),
            emb_dim=emb_dim,
            num_tokens=int(num_param_tokens),
            hidden=int(param_hidden),
            activation=activation,
        )
        # Learned 3-way modality embedding (spatial / forcing / param) so the
        # decoder can tell the heterogeneous token streams apart.
        self.modality = nn.Parameter(torch.randn(3, emb_dim) * 0.02)
        self.decoder = CViTDecoder(
            enc_emb_dim=emb_dim,
            dec_emb_dim=dec_emb_dim,
            out_dim=out_dim,
            depth=depth_dec,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            fourier_freq=fourier_freq,
            fourier_freq_t=fourier_freq_t,
            activation=activation,
        )
        self.register_buffer(
            "t_right_tilde", torch.tensor(float(t_right_tilde), dtype=torch.float32)
        )
        self.register_buffer(
            "t_norm", torch.tensor(float(t_final), dtype=torch.float32)
        )

    def encode(
        self,
        u_spatial: torch.Tensor,
        forcing_seq: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Build the cached latent token set; shape ``(B, N_s+N_f+N_p, emb_dim)``.

        u_spatial:(B, spatial_in_ch, Nx, Ny); forcing_seq:(B, M, token_dim);
        params:(B, n_param_scalars) = ``[x_hat, Rc_hat]``. Query-independent, so
        run once per sim and reuse for every decode.
        """
        z_s = self.spatial_encoder(u_spatial) + self.modality[0]
        z_f = self.forcing_encoder(forcing_seq) + self.modality[1]
        z_p = self.param_encoder(params) + self.modality[2]
        return torch.cat([z_s, z_f, z_p], dim=1)

    def decode(
        self, latent: torch.Tensor, coords: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Pure coordinate+time query against a cached latent; ``(B, Nq, out_dim)``.

        The decoder takes only ``(coords, t)`` per query — no static conditioning
        enters here — so the same latent serves both CN time slices. Shared
        ``(1, Nq, .)`` coordinate/time sets are expanded to the sim minibatch
        (torch attention does not broadcast batch dims).
        """
        B = latent.shape[0]
        if coords.shape[0] == 1 and B > 1:
            coords = coords.expand(B, -1, -1)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1, -1)
        raw = self.decoder(latent, coords, t / self.t_norm)
        if not self.hard_right_dirichlet:
            return raw
        x = coords[..., 0:1]
        return self.t_right_tilde + (1.0 - x) * raw

    def forward(
        self,
        u_spatial: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
        forcing_seq: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Single-shot ``encode`` + ``decode`` (chunked callers use them directly)."""
        return self.decode(self.encode(u_spatial, forcing_seq, params), coords, t)
