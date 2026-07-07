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
    ):
        super().__init__()
        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.hard_right_dirichlet = bool(hard_right_dirichlet)
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

    def forward(
        self, u: torch.Tensor, coords: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B = u.shape[0]
        # torch attention does not broadcast batch dims: expand shared dim-1
        # coordinate/time sets to the sim minibatch explicitly.
        if coords.shape[0] == 1 and B > 1:
            coords = coords.expand(B, -1, -1)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1, -1)
        tokens = self.encoder(u)
        raw = self.decoder(tokens, coords, t)
        if self.hard_right_dirichlet:
            x = coords[..., 0:1]
            return self.t_right_tilde + (1.0 - x) * raw
        return raw
