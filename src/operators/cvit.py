from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TransitionEncoding:
    memory_tokens: Tensor
    forcing_context: Tensor

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
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        return x + (torch.sin(a * x) ** 2) / a


def _make_activation(name: str, dim: int) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "snake":
        return Snake(dim)
    raise ValueError(f"Unknown activation {name!r}")


# --------- 2D sin-cos positional embedding ---------

def _sincos_pos_embed_2d(emb_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Return the reference-initialized 2D sin-cos positional embedding."""
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

    ``freq_scale`` initializes the trainable projection ``B`` and therefore sets
    the decoder's initial spectral bias. ``out_dim`` must be even.
    """

    def __init__(self, in_dim: int, out_dim: int, freq_scale: float):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"FourierEmbed out_dim must be even, got {out_dim}.")
        kernel = torch.randn(in_dim, out_dim // 2) * float(freq_scale)
        self.kernel = nn.Parameter(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.kernel  # (..., out_dim//2)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def canonicalize_query_coords(
    coords: Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> Tensor:
    """Validate ``(x, y)`` queries on ``[0, 1]^2`` and clamp roundoff at its edges."""
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(
            "coords must have shape (B, Nq, 2) in (x, y) order; "
            f"got {tuple(coords.shape)}."
        )
    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError("coordinate tolerance must be finite and non-negative")
    if not bool(torch.isfinite(coords).all().item()):
        raise ValueError("coords must contain only finite values")
    outside = (coords < -float(tolerance)) | (coords > 1.0 + float(tolerance))
    if bool(outside.any().item()):
        lo = float(coords.detach().amin().cpu())
        hi = float(coords.detach().amax().cpu())
        raise ValueError(
            f"coords must lie in [0, 1] within tolerance {tolerance}; "
            f"observed range [{lo}, {hi}]."
        )
    return coords.clamp(0.0, 1.0)


def sample_source_at_queries(
    source_field: Tensor,
    query_coords: Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    """Bilinearly sample ``(B, 1, Nx, Ny)`` fields at continuous ``(x, y)`` queries."""
    if source_field.ndim != 4 or source_field.shape[1] != 1:
        raise ValueError(
            "source_field must have shape (B, 1, Nx, Ny); "
            f"got {tuple(source_field.shape)}."
        )
    coords = canonicalize_query_coords(query_coords, tolerance=tolerance)
    if coords.shape[0] != source_field.shape[0]:
        raise ValueError(
            "source_field and query_coords batch dimensions must match; "
            f"got {source_field.shape[0]} and {coords.shape[0]}."
        )
    # grid_sample interprets the last coordinate as (width, height). The stored
    # field axes are (Nx, Ny), so physical (x, y) must be supplied as (y, x).
    grid = torch.stack(
        [2.0 * coords[..., 1] - 1.0, 2.0 * coords[..., 0] - 1.0],
        dim=-1,
    ).unsqueeze(2)
    values = F.grid_sample(
        source_field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return values.squeeze(-1).transpose(1, 2), coords


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
        # Equinox optimizes every array leaf in the reference model, including
        # the sin-cos-initialized position array.
        self.pos_emb = nn.Parameter(pos.unsqueeze(0))        # (1, N, emb_dim)
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
        fourier_freq_t: float | None = None,
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
    ):
        super().__init__()
        if film_hidden_layers < 0 or head_hidden_layers < 0:
            raise ValueError("FiLM and head hidden-layer counts must be non-negative.")
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
            num_layers=film_hidden_layers + 1,
            activation=film_activation,
            zero_init_last=True,
        )
        self.proj_kv = nn.Linear(enc_emb_dim, dec_emb_dim)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(dec_emb_dim, num_heads, mlp_ratio, activation) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dec_emb_dim)
        self.head = MLP(
            dec_emb_dim,
            dec_emb_dim,
            out_dim,
            head_hidden_layers + 1,
            head_activation,
        )

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


def _batch_scalar(
    value: Tensor,
    *,
    batch_size: int,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.reshape(-1, 1)
    else:
        if tensor.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{name} batch dimension must be 1 or {batch_size}; "
                f"got {tuple(tensor.shape)}."
            )
        tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must contain one scalar per sample; got {tuple(value.shape)}."
            )
    if tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1)
    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch dimension must be 1 or {batch_size}; "
            f"got {tensor.shape[0]}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class TransitionCViTDecoder(nn.Module):
    def __init__(
        self,
        enc_emb_dim: int,
        dec_emb_dim: int,
        forcing_context_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        fourier_freq: float,
        fourier_freq_source: float,
        fourier_freq_lead: float,
        activation: str,
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
        query_time_conditioning: bool = True,
        film_time_conditioning: bool = True,
        film_init_std: float = 1.0e-3,
    ):
        super().__init__()
        if not query_time_conditioning and not film_time_conditioning:
            raise ValueError(
                "At least one of query_time_conditioning or "
                "film_time_conditioning must be enabled."
            )
        if film_init_std <= 0.0 or not math.isfinite(float(film_init_std)):
            raise ValueError("film_init_std must be finite and positive")
        self.depth = int(depth)
        self.dec_emb_dim = int(dec_emb_dim)
        self.query_time_conditioning = bool(query_time_conditioning)
        self.film_time_conditioning = bool(film_time_conditioning)
        self.fourier_x = FourierEmbed(2, dec_emb_dim, fourier_freq)
        self.fourier_source = FourierEmbed(
            1, dec_emb_dim, fourier_freq_source,
        )
        self.fourier_lead = FourierEmbed(
            1, dec_emb_dim, fourier_freq_lead,
        )
        self.time_conditioner = MLP(
            in_dim=2 + 2 * dec_emb_dim,
            hidden_dim=dec_emb_dim,
            out_dim=dec_emb_dim,
            num_layers=2,
            activation=film_activation,
        )
        if self.query_time_conditioning:
            self.query_time_projection = nn.Linear(dec_emb_dim, dec_emb_dim)
        if self.film_time_conditioning:
            self.time_film = MLP(
                in_dim=dec_emb_dim + forcing_context_dim,
                hidden_dim=dec_emb_dim,
                out_dim=depth * 2 * dec_emb_dim,
                num_layers=film_hidden_layers + 1,
                activation=film_activation,
            )
            film_head = self.time_film.net[-1]
            if not isinstance(film_head, nn.Linear):
                raise TypeError("time_film must end in a Linear layer")
            nn.init.normal_(film_head.weight, mean=0.0, std=float(film_init_std))
            nn.init.zeros_(film_head.bias)
        self.proj_kv = nn.Linear(enc_emb_dim, dec_emb_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    dec_emb_dim, num_heads, mlp_ratio, activation,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dec_emb_dim)
        self.head = MLP(
            dec_emb_dim,
            dec_emb_dim,
            out_dim,
            head_hidden_layers + 1,
            head_activation,
        )

    def forward(
        self,
        encoding: TransitionEncoding,
        coords: Tensor,
        source_time_norm: Tensor,
        lead_time_norm: Tensor,
    ) -> Tensor:
        batch_size = encoding.memory_tokens.shape[0]
        if coords.shape[0] == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1)
        if coords.shape[0] != batch_size:
            raise ValueError(
                "encoding and coords batch dimensions must match; "
                f"got {batch_size} and {coords.shape[0]}."
            )
        source = _batch_scalar(
            source_time_norm, batch_size=batch_size, name="source_time",
        ).to(device=coords.device, dtype=coords.dtype)
        lead = _batch_scalar(
            lead_time_norm, batch_size=batch_size, name="lead_time",
        ).to(device=coords.device, dtype=coords.dtype)
        time_raw = torch.stack([lead[:, 0], source[:, 0]], dim=-1).unsqueeze(1)
        time_features = torch.cat(
            [
                time_raw,
                self.fourier_lead(lead.unsqueeze(1)),
                self.fourier_source(source.unsqueeze(1)),
            ],
            dim=-1,
        )
        time_context = self.time_conditioner(time_features)

        queries = self.fourier_x(coords)
        if self.query_time_conditioning:
            queries = queries + self.query_time_projection(time_context)
        kv = self.proj_kv(encoding.memory_tokens)
        film = None
        if self.film_time_conditioning:
            forcing_context = encoding.forcing_context
            if forcing_context.shape[0] != batch_size:
                raise ValueError(
                    "forcing_context and memory_tokens batch dimensions must match"
                )
            film_input = torch.cat(
                [time_context[:, 0], forcing_context], dim=-1,
            )
            film = self.time_film(film_input).view(
                batch_size, self.depth, 2, self.dec_emb_dim,
            )
        for index, block in enumerate(self.blocks):
            scale = None if film is None else film[:, index, 0, :].unsqueeze(1)
            shift = None if film is None else film[:, index, 1, :].unsqueeze(1)
            queries = block(queries, kv, scale=scale, shift=shift)
        return self.head(self.norm(queries))


def moving_interface_jump_enrichment(
    smooth: torch.Tensor,
    coords: torch.Tensor,
    interface_flux_normalized: torch.Tensor,
    interface_x: torch.Tensor,
    jump_scale: torch.Tensor,
) -> torch.Tensor:
    """Add a moving Heaviside jump while leaving the right subdomain untouched."""
    batch_size = smooth.shape[0]
    interface = torch.as_tensor(
        interface_x, device=smooth.device, dtype=smooth.dtype
    ).reshape(batch_size, 1, 1)
    scale = torch.as_tensor(
        jump_scale, device=smooth.device, dtype=smooth.dtype
    ).reshape(batch_size, 1, 1)
    left = (coords[..., 0:1] < interface).to(smooth.dtype)
    return smooth + left * scale * interface_flux_normalized


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
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
        hard_right_dirichlet: bool = True,
        t_right_tilde: float = 0.0,
        t_final: float = 1.0,
        hard_left_flux: bool = False,
        left_flux_scale: float = 1.0,
    ):
        super().__init__()
        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.hard_right_dirichlet = bool(hard_right_dirichlet)
        if hard_left_flux:
            raise ValueError(
                "hard_left_flux was removed because its global forcing lift was "
                "not consistent with the interior PDE residual; use the soft "
                "forcing-Neumann residual."
            )
        self.hard_left_flux = False
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
            film_hidden_layers=film_hidden_layers,
            film_activation=film_activation,
            head_hidden_layers=head_hidden_layers,
            head_activation=head_activation,
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
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
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
            film_hidden_layers=film_hidden_layers,
            film_activation=film_activation,
            head_hidden_layers=head_hidden_layers,
            head_activation=head_activation,
            hard_right_dirichlet=hard_right_dirichlet,
            t_right_tilde=t_right_tilde,
            t_final=t_final,
            hard_left_flux=hard_left_flux,
            left_flux_scale=left_flux_scale,
        )


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
    geometry ``(x_Gamma, R_c)`` and the applied-flux image ``q_L(y,t)`` —
    never the synthetic generator parameters, so it learns the operator
    ``(x_Gamma, R_c, q_L(y,t)) -> T``.

    Streams (each -> ``emb_dim``-wide tokens, plus a learned 3-way modality
    embedding):
      1. spatial  ``z_s``: full ``CViTEncoder`` over ``[T0_tilde, K_norm, D_norm]``.
      2. forcing  ``z_f``: independent ``CViTEncoder`` over the normalized
         boundary space-time image ``(B,1,Ny_img,Nt_img)``.
      3. param    ``z_p``: ``ParamTokenEncoder`` over ``[x_hat, Rc_hat]``.

    ``encode(u_spatial, forcing_image, params)`` builds the cached latent token set
    (query-independent, so run once per sim); ``decode(latent, coords, t)`` is the
    pure coordinate+time query reused for both CN time slices. The hard
    right-Dirichlet ansatz ``T = t_right_tilde + (1 - x) * raw`` keeps the x=1
    wall exact; the left-wall Neumann is enforced softly by the FV residual (no
    hard-left-flux lifting).
    """

    def __init__(
        self,
        spatial_in_ch: int = 3,
        forcing_in_ch: int = 1,
        out_dim: int = 1,
        emb_dim: int = 256,
        dec_emb_dim: int | None = None,
        patch_size: int = 10,
        grid_size: tuple[int, int] = (100, 100),
        forcing_patch_size: int = 8,
        forcing_grid_size: tuple[int, int] = (96, 256),
        depth_enc: int = 4,
        depth_dec: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        fourier_freq: float = 1.0,
        fourier_freq_t: float | None = None,
        activation: str = "gelu",
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
        hard_right_dirichlet: bool = True,
        t_right_tilde: float = 0.0,
        t_final: float = 1.0,
        n_param_scalars: int = 2,
        num_param_tokens: int = 1,
        param_hidden: int = 128,
        jump_enrichment: bool = False,
        jump_flux_depth: int = 1,
        jump_flux_conditioning: str = "all",
        jump_flux_mode: str = "learned",
        interface_aligned_domains: bool = False,
        interface_flux_gradient_scale: float = 1.0,
        k_left: float = 2.0,
        k_right: float = 1.0,
        interface_x_range: tuple[float, float] = (0.2, 0.8),
        resistance_range: tuple[float, float] = (0.05, 1.0),
    ):
        super().__init__()
        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.hard_right_dirichlet = bool(hard_right_dirichlet)
        self.jump_enrichment = bool(jump_enrichment)
        self.jump_flux_conditioning = str(jump_flux_conditioning)
        if self.jump_flux_conditioning not in {"all", "state_param"}:
            raise ValueError(
                "jump_flux_conditioning must be 'all' or 'state_param'"
            )
        self.jump_flux_mode = str(jump_flux_mode)
        if self.jump_flux_mode not in {
            "learned",
            "left_energy_closure",
            "two_sided_energy_closure",
            "conservative_storage_projection",
        }:
            raise ValueError(
                "jump_flux_mode must be 'learned', 'left_energy_closure', or "
                "'two_sided_energy_closure', or "
                "'conservative_storage_projection'"
            )
        self.interface_aligned_domains = bool(interface_aligned_domains)
        if self.interface_aligned_domains and (
            not self.jump_enrichment or self.jump_flux_mode != "learned"
        ):
            raise ValueError(
                "interface_aligned_domains requires learned jump enrichment"
            )
        if self.interface_aligned_domains and (
            int(spatial_in_ch) != 3 or int(out_dim) != 1
        ):
            raise ValueError(
                "interface_aligned_domains requires spatial_in_ch=3 and out_dim=1"
            )
        if self.interface_aligned_domains and not self.hard_right_dirichlet:
            raise ValueError(
                "interface_aligned_domains requires the hard right Dirichlet boundary"
            )
        if float(interface_flux_gradient_scale) <= 0.0:
            raise ValueError("interface_flux_gradient_scale must be positive")
        if float(k_left) <= 0.0 or float(k_right) <= 0.0:
            raise ValueError("interface conductivities must be positive")
        x_min, x_max = (float(v) for v in interface_x_range)
        r_min, r_max = (float(v) for v in resistance_range)
        if not x_min < x_max or not r_min < r_max:
            raise ValueError("interface and resistance ranges must be increasing")
        if int(forcing_in_ch) != 1 or int(n_param_scalars) != 2:
            raise ValueError(
                "InterfaceCViT requires one forcing-image channel and exactly "
                "two parameter scalars [interface_x, R_c]."
            )
        self.n_param_scalars = 2
        forcing_grid_size = tuple(int(v) for v in forcing_grid_size)
        forcing_patch_size = int(forcing_patch_size)
        if len(forcing_grid_size) != 2 or forcing_patch_size <= 0 or any(
            v <= 0 for v in forcing_grid_size
        ):
            raise ValueError(
                "forcing_grid_size must contain two positive dimensions and "
                f"forcing_patch_size must be positive; got {forcing_grid_size} "
                f"and {forcing_patch_size}."
            )
        if any(v % forcing_patch_size != 0 for v in forcing_grid_size):
            raise ValueError(
                "forcing_grid_size dimensions must be divisible by "
                f"forcing_patch_size; got {forcing_grid_size} and {forcing_patch_size}."
            )
        self.forcing_grid_size = forcing_grid_size
        self.forcing_patch_size = forcing_patch_size
        self.num_forcing_tokens = (
            forcing_grid_size[0] // forcing_patch_size
        ) * (forcing_grid_size[1] // forcing_patch_size)
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
        self.num_spatial_tokens = int(self.spatial_encoder.pos_emb.shape[1])
        self.forcing_encoder = CViTEncoder(
            in_ch=int(forcing_in_ch),
            emb_dim=emb_dim,
            patch_size=forcing_patch_size,
            grid_size=forcing_grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
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
        decoder_kwargs = {
            "enc_emb_dim": emb_dim,
            "dec_emb_dim": dec_emb_dim,
            "depth": depth_dec,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "fourier_freq": fourier_freq,
            "fourier_freq_t": fourier_freq_t,
            "activation": activation,
            "film_hidden_layers": film_hidden_layers,
            "film_activation": film_activation,
            "head_hidden_layers": head_hidden_layers,
            "head_activation": head_activation,
        }
        if self.interface_aligned_domains:
            self.decoder = nn.ModuleDict({
                "left": CViTDecoder(out_dim=1, **decoder_kwargs),
                "right": CViTDecoder(out_dim=1, **decoder_kwargs),
                "interface": CViTDecoder(
                    out_dim=2, depth=int(jump_flux_depth),
                    **{k: v for k, v in decoder_kwargs.items() if k != "depth"},
                ),
            })
            for decoder in self.decoder.values():
                last = decoder.head.net[-1]
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        else:
            self.decoder = CViTDecoder(out_dim=out_dim, **decoder_kwargs)
        if (
            self.jump_enrichment
            and self.jump_flux_mode == "learned"
            and not self.interface_aligned_domains
        ):
            self.interface_flux_decoder = CViTDecoder(
                enc_emb_dim=emb_dim,
                dec_emb_dim=dec_emb_dim,
                out_dim=1,
                depth=int(jump_flux_depth),
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                fourier_freq=fourier_freq,
                fourier_freq_t=fourier_freq_t,
                activation=activation,
                film_hidden_layers=film_hidden_layers,
                film_activation=film_activation,
                head_hidden_layers=head_hidden_layers,
                head_activation=head_activation,
            )
            last = self.interface_flux_decoder.head.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if self.interface_aligned_domains:
            self.register_buffer(
                "interface_flux_gradient_scale",
                torch.tensor(float(interface_flux_gradient_scale), dtype=torch.float32),
            )
            self.register_buffer("k_left", torch.tensor(float(k_left), dtype=torch.float32))
            self.register_buffer("k_right", torch.tensor(float(k_right), dtype=torch.float32))
            self.register_buffer("interface_x_min", torch.tensor(x_min, dtype=torch.float32))
            self.register_buffer("interface_x_max", torch.tensor(x_max, dtype=torch.float32))
            self.register_buffer("resistance_min", torch.tensor(r_min, dtype=torch.float32))
            self.register_buffer("resistance_max", torch.tensor(r_max, dtype=torch.float32))
        self.register_buffer(
            "t_right_tilde", torch.tensor(float(t_right_tilde), dtype=torch.float32)
        )
        self.register_buffer(
            "t_norm", torch.tensor(float(t_final), dtype=torch.float32)
        )

    def encode(
        self,
        u_spatial: torch.Tensor,
        forcing_image: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Build the cached latent token set; shape ``(B, N_s+N_f+N_p, emb_dim)``.

        u_spatial:(B, spatial_in_ch, Nx, Ny); forcing_image:(B, 1, Ny_img, Nt_img);
        params:(B, n_param_scalars) = ``[x_hat, Rc_hat]``. Query-independent, so
        run once per sim and reuse for every decode.
        """
        if tuple(forcing_image.shape[1:]) != (1, *self.forcing_grid_size):
            raise ValueError(
                "forcing_image must have shape (B, 1, Ny_img, Nt_img) with "
                f"(Ny_img, Nt_img)={self.forcing_grid_size}; got "
                f"{tuple(forcing_image.shape)}."
            )
        if params.ndim != 2 or params.shape[1] != self.n_param_scalars:
            raise ValueError(
                "params must have shape (B, 2) = [normalized interface_x, "
                f"normalized R_c]; got {tuple(params.shape)}."
            )
        if self.interface_aligned_domains:
            interface_x, jump_scale = self._physical_interface_scalars(params)
            u_spatial = self._align_spatial_input(u_spatial, interface_x)
        z_s = self.spatial_encoder(u_spatial) + self.modality[0]
        z_f = self.forcing_encoder(forcing_image) + self.modality[1]
        if z_f.shape[1] != self.num_forcing_tokens:
            raise RuntimeError(
                f"Expected {self.num_forcing_tokens} forcing tokens, got {z_f.shape[1]}."
            )
        z_p = self.param_encoder(params) + self.modality[2]
        tokens = torch.cat([z_s, z_f, z_p], dim=1)
        if not self.interface_aligned_domains:
            return tokens
        return {
            "tokens": tokens,
            "interface_x": interface_x,
            "jump_scale": jump_scale,
        }

    def _physical_interface_scalars(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interface_x = self.interface_x_min + params[:, 0] * (
            self.interface_x_max - self.interface_x_min
        )
        resistance = self.resistance_min + params[:, 1] * (
            self.resistance_max - self.resistance_min
        )
        jump_scale = resistance * self.interface_flux_gradient_scale
        return interface_x, jump_scale

    def _align_spatial_input(
        self, u_spatial: torch.Tensor, interface_x: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, nx, ny = u_spatial.shape
        if channels != 3 or nx % 2:
            raise ValueError(
                "interface alignment requires three spatial channels and an even x grid"
            )
        left_size = nx // 2
        temperatures = []
        for batch in range(batch_size):
            split = int(round(float(interface_x[batch].detach().cpu()) * (nx - 1) + 0.5))
            if split < 2 or nx - split < 2:
                raise ValueError("interface alignment requires at least two nodes per domain")
            left = F.interpolate(
                u_spatial[batch : batch + 1, 0:1, :split],
                size=(left_size, ny), mode="bilinear", align_corners=True,
            )
            right = F.interpolate(
                u_spatial[batch : batch + 1, 0:1, split:],
                size=(nx - left_size, ny), mode="bilinear", align_corners=True,
            )
            temperatures.append(torch.cat((left, right), dim=2))
        temperature = torch.cat(temperatures, dim=0)
        material = torch.ones_like(temperature)
        material[:, :, left_size:] = -1.0
        left_distance = torch.linspace(
            -1.0, -1.0 / left_size, left_size,
            device=u_spatial.device, dtype=u_spatial.dtype,
        )
        right_size = nx - left_size
        right_distance = torch.linspace(
            1.0 / right_size, 1.0, right_size,
            device=u_spatial.device, dtype=u_spatial.dtype,
        )
        distance = torch.cat((left_distance, right_distance)).view(1, 1, nx, 1)
        distance = distance.expand(batch_size, 1, nx, ny)
        return torch.cat((temperature, material, distance), dim=1)

    def _interface_head_output(
        self,
        latent: dict[str, torch.Tensor],
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        tokens = latent["tokens"]
        batch_size = tokens.shape[0]
        if y.shape[0] == 1 and batch_size > 1:
            y = y.expand(batch_size, -1, -1)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1, -1)
        if self.jump_flux_conditioning == "state_param":
            forcing_stop = self.num_spatial_tokens + self.num_forcing_tokens
            tokens = torch.cat(
                (tokens[:, : self.num_spatial_tokens], tokens[:, forcing_stop:]),
                dim=1,
            )
        interface_coords = torch.cat((torch.zeros_like(y), y), dim=-1)
        return self.decoder["interface"](tokens, interface_coords, t / self.t_norm)

    def _decode_aligned_rate(
        self,
        latent: dict[str, torch.Tensor],
        coords: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        tokens = latent["tokens"]
        batch_size = tokens.shape[0]
        if coords.shape[0] == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1, -1)
        interface_x = latent["interface_x"].to(coords).view(batch_size, 1, 1)
        jump_scale = latent["jump_scale"].to(coords).view(batch_size, 1, 1)
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        left_x = (x / interface_x).clamp(0.0, 1.0)
        right_length = 1.0 - interface_x
        right_x = ((x - interface_x) / right_length).clamp(0.0, 1.0)
        left_raw = self.decoder["left"](
            tokens, torch.cat((left_x, y), dim=-1), t / self.t_norm
        )
        right_raw = self.decoder["right"](
            tokens, torch.cat((right_x, y), dim=-1), t / self.t_norm
        )
        trace_rate, flux_rate = self._interface_head_output(latent, y, t).split(1, dim=-1)
        distance = x - interface_x
        left_slope = -flux_rate * self.interface_flux_gradient_scale / self.k_left
        right_slope = -flux_rate * self.interface_flux_gradient_scale / self.k_right
        trace = self.t_right_tilde + trace_rate
        left = (
            trace
            + jump_scale * flux_rate
            + left_slope * distance
            + distance.square() * left_raw
        )
        right_quadratic = (
            self.t_right_tilde - trace - right_slope * right_length
        ) / right_length.square()
        right = (
            trace
            + right_slope * distance
            + right_quadratic * distance.square()
            + distance.square() * (right_length - distance) * right_raw
        )
        return torch.where(x < interface_x, left, right)

    def load_state_dict(self, state_dict, strict: bool = True):
        legacy = any(
            key.startswith((
                "forcing_encoder.lift.", "forcing_encoder.conv1.",
                "forcing_encoder.conv2.", "forcing_encoder.proj.",
            ))
            for key in state_dict
        )
        if legacy:
            raise RuntimeError(
                "Legacy InterfaceCViT checkpoint uses waveform_tokens; expected "
                "space_time_image representation version 1. Start a fresh run."
            )
        return super().load_state_dict(state_dict, strict=strict)

    def decode(
        self,
        latent: torch.Tensor | dict[str, torch.Tensor],
        coords: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Pure coordinate+time query against a cached latent; ``(B, Nq, out_dim)``.

        The decoder takes only ``(coords, t)`` per query — no static conditioning
        enters here — so the same latent serves both CN time slices. Shared
        ``(1, Nq, .)`` coordinate/time sets are expanded to the sim minibatch
        (torch attention does not broadcast batch dims).
        """
        if self.interface_aligned_domains:
            if not isinstance(latent, dict):
                raise TypeError("aligned InterfaceCViT requires its encoded latent mapping")
            return self._decode_aligned_rate(latent, coords, t)
        if not isinstance(latent, torch.Tensor):
            raise TypeError("standard InterfaceCViT requires a tensor latent")
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

    def decode_interface_flux(
        self,
        latent: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Query interface flux, or its rate in the aligned-rate variant."""
        if not self.jump_enrichment or self.jump_flux_mode != "learned":
            raise RuntimeError("interface flux decoder is disabled")
        if self.interface_aligned_domains:
            if not isinstance(latent, dict):
                raise TypeError("aligned InterfaceCViT requires its encoded latent mapping")
            return self._interface_head_output(latent, y, t)[..., 1:2]
        if not isinstance(latent, torch.Tensor):
            raise TypeError("standard InterfaceCViT requires a tensor latent")
        batch_size = latent.shape[0]
        if y.shape[0] == 1 and batch_size > 1:
            y = y.expand(batch_size, -1, -1)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1, -1)
        flux_coords = torch.cat((torch.zeros_like(y), y), dim=-1)
        flux_latent = latent
        if self.jump_flux_conditioning == "state_param":
            forcing_stop = self.num_spatial_tokens + self.num_forcing_tokens
            flux_latent = torch.cat(
                (latent[:, : self.num_spatial_tokens], latent[:, forcing_stop:]),
                dim=1,
            )
        return self.interface_flux_decoder(
            flux_latent, flux_coords, t / self.t_norm
        )

    def decode_enriched(
        self,
        latent: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
        interface_x: torch.Tensor,
        jump_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return enriched temperature and its aligned ``q_Gamma/q_ref`` field."""
        if not self.jump_enrichment or self.jump_flux_mode != "learned":
            raise RuntimeError("jump enrichment is disabled")
        if self.interface_aligned_domains:
            raise RuntimeError("aligned InterfaceCViT embeds its interface constraints in decode")
        batch_size = latent.shape[0]
        if coords.shape[0] == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1, -1)
        smooth = self.decode(latent, coords, t)
        flux = self.decode_interface_flux(latent, coords[..., 1:2], t)
        enriched = moving_interface_jump_enrichment(
            smooth, coords, flux, interface_x, jump_scale
        )
        return enriched, flux

    def forward(
        self,
        u_spatial: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
        forcing_image: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Single-shot ``encode`` + ``decode`` (chunked callers use them directly)."""
        return self.decode(self.encode(u_spatial, forcing_image, params), coords, t)


class ForcingICCViT(nn.Module):
    """Two-branch token-fusion CViT for ``diffusion_forcing_single`` with a
    varying initial condition.

    The single-slab forcing benchmark encodes the boundary flux as a space-time
    image ``q_L(y, t)/a_ref`` (:class:`ForcingCViT`). When the IC stops being a
    fixed uniform 300 K field it becomes per-sim information the forcing image
    cannot carry, so a **second encoder branch** ingests the initial temperature
    field and the two token streams are fused for the shared decoder — exactly
    the multimodal recipe of :class:`InterfaceCViT`, minus the interface/param
    stream.

    Streams (each -> ``emb_dim``-wide tokens, plus a learned 2-way modality
    embedding so the decoder can tell them apart):
      1. forcing ``z_f``: :class:`CViTEncoder` over the boundary space-time image
         ``(B, 1, Ny_img, Nt_img)`` (axes rows=y-nodes, cols=time). Reuse the
         successful :class:`ForcingCViT` image config verbatim.
      2. ic ``z_ic``: independent :class:`CViTEncoder` over the normalized
         initial field ``T0_tilde = (T0 - T_right)/sigma_global`` on the data
         grid ``(B, 1, Nx, Ny)``.

    ``encode(u_forcing, u_ic)`` builds the query-independent latent (run once per
    sim, reuse for every decode); ``decode(latent, coords, t)`` is the pure
    coordinate+time query. The hard right-Dirichlet ansatz
    ``T = t_right_tilde + (1 - x) * raw`` keeps the x=1 wall exact; the left-wall
    Neumann stays **soft** (enforced by the PINO residual), so ``decode`` takes
    no ``q_left`` (matching :meth:`InterfaceCViT.decode`).

    The decoder output, the IC encoder input, and the IC loss target must all
    live in the SAME normalized space (frozen ``sigma_global``); this module does
    not normalize — callers pass ``T0_tilde`` already scaled.
    """

    def __init__(
        self,
        forcing_in_ch: int = 1,
        ic_in_ch: int = 1,
        out_dim: int = 1,
        emb_dim: int = 256,
        dec_emb_dim: int | None = None,
        ic_patch_size: int = 10,
        ic_grid_size: tuple[int, int] = (100, 100),
        forcing_patch_size: int = 8,
        forcing_grid_size: tuple[int, int] = (96, 256),
        depth_enc: int = 4,
        depth_dec: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        fourier_freq: float = 1.0,
        fourier_freq_t: float | None = None,
        activation: str = "gelu",
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
        hard_right_dirichlet: bool = True,
        t_right_tilde: float = 0.0,
        t_final: float = 1.0,
    ):
        super().__init__()
        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.hard_right_dirichlet = bool(hard_right_dirichlet)
        if int(forcing_in_ch) != 1:
            raise ValueError(
                "ForcingICCViT requires exactly one forcing-image channel "
                f"q_L(y,t)/a_ref; got forcing_in_ch={forcing_in_ch}."
            )
        forcing_grid_size = tuple(int(v) for v in forcing_grid_size)
        forcing_patch_size = int(forcing_patch_size)
        if len(forcing_grid_size) != 2 or forcing_patch_size <= 0 or any(
            v <= 0 for v in forcing_grid_size
        ):
            raise ValueError(
                "forcing_grid_size must contain two positive dimensions and "
                f"forcing_patch_size must be positive; got {forcing_grid_size} "
                f"and {forcing_patch_size}."
            )
        if any(v % forcing_patch_size != 0 for v in forcing_grid_size):
            raise ValueError(
                "forcing_grid_size dimensions must be divisible by "
                f"forcing_patch_size; got {forcing_grid_size} and {forcing_patch_size}."
            )
        # IC branch runs on the data grid (Nx, Ny). PatchEmbed2d is scalar and
        # drops a border strip on non-divisible grids, so require an exact tiling
        # (the intended config values on the 100x100 grid are ic_patch_size in
        # {10, 20}).
        ic_grid_size = tuple(int(v) for v in ic_grid_size)
        ic_patch_size = int(ic_patch_size)
        if len(ic_grid_size) != 2 or ic_patch_size <= 0 or any(
            v <= 0 for v in ic_grid_size
        ):
            raise ValueError(
                "ic_grid_size must contain two positive dimensions and "
                f"ic_patch_size must be positive; got {ic_grid_size} and "
                f"{ic_patch_size}."
            )
        if ic_grid_size[0] % ic_patch_size != 0 or ic_grid_size[1] % ic_patch_size != 0:
            raise ValueError(
                "ic_grid_size dimensions (Nx, Ny) must both be divisible by "
                f"ic_patch_size; got {ic_grid_size} and {ic_patch_size}."
            )
        self.forcing_grid_size = forcing_grid_size
        self.forcing_patch_size = forcing_patch_size
        self.ic_grid_size = ic_grid_size
        self.ic_patch_size = ic_patch_size
        self.num_forcing_tokens = (
            forcing_grid_size[0] // forcing_patch_size
        ) * (forcing_grid_size[1] // forcing_patch_size)
        self.num_ic_tokens = (
            ic_grid_size[0] // ic_patch_size
        ) * (ic_grid_size[1] // ic_patch_size)
        self.forcing_encoder = CViTEncoder(
            in_ch=int(forcing_in_ch),
            emb_dim=emb_dim,
            patch_size=forcing_patch_size,
            grid_size=forcing_grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
        )
        self.ic_encoder = CViTEncoder(
            in_ch=int(ic_in_ch),
            emb_dim=emb_dim,
            patch_size=ic_patch_size,
            grid_size=ic_grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
        )
        # Learned 2-way modality embedding (forcing / ic) added per token before
        # the streams are concatenated along the token dim — the same scheme as
        # InterfaceCViT.encode, so the decoder cross-attention can distinguish
        # the heterogeneous sources.
        self.modality = nn.Parameter(torch.randn(2, emb_dim) * 0.02)
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
            film_hidden_layers=film_hidden_layers,
            film_activation=film_activation,
            head_hidden_layers=head_hidden_layers,
            head_activation=head_activation,
        )
        self.register_buffer(
            "t_right_tilde", torch.tensor(float(t_right_tilde), dtype=torch.float32)
        )
        self.register_buffer(
            "t_norm", torch.tensor(float(t_final), dtype=torch.float32)
        )

    def encode(
        self, u_forcing: torch.Tensor, u_ic: torch.Tensor
    ) -> torch.Tensor:
        """Build the cached latent token set; ``(B, N_f + N_ic, emb_dim)``.

        u_forcing:(B, 1, Ny_img, Nt_img) boundary space-time image;
        u_ic:(B, 1, Nx, Ny) normalized initial field ``T0_tilde``.
        Query-independent, so run once per sim and reuse for every decode.
        """
        if tuple(u_forcing.shape[1:]) != (1, *self.forcing_grid_size):
            raise ValueError(
                "u_forcing must have shape (B, 1, Ny_img, Nt_img) with "
                f"(Ny_img, Nt_img)={self.forcing_grid_size}; got "
                f"{tuple(u_forcing.shape)}."
            )
        if tuple(u_ic.shape[1:]) != (1, *self.ic_grid_size):
            raise ValueError(
                "u_ic must have shape (B, 1, Nx, Ny) with "
                f"(Nx, Ny)={self.ic_grid_size}; got {tuple(u_ic.shape)}."
            )
        z_f = self.forcing_encoder(u_forcing) + self.modality[0]
        if z_f.shape[1] != self.num_forcing_tokens:
            raise RuntimeError(
                f"Expected {self.num_forcing_tokens} forcing tokens, got {z_f.shape[1]}."
            )
        z_ic = self.ic_encoder(u_ic) + self.modality[1]
        if z_ic.shape[1] != self.num_ic_tokens:
            raise RuntimeError(
                f"Expected {self.num_ic_tokens} IC tokens, got {z_ic.shape[1]}."
            )
        return torch.cat([z_f, z_ic], dim=1)

    def decode(
        self, latent: torch.Tensor, coords: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Pure coordinate+time query against a cached latent; ``(B, Nq, out_dim)``.

        No static conditioning enters here, so the same latent serves every query
        batch. Shared ``(1, Nq, .)`` coordinate/time sets are expanded to the sim
        minibatch (torch attention does not broadcast batch dims). The left wall
        stays soft (no ``q_left`` lifting); only the hard right-Dirichlet ansatz
        is applied.
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
        u_forcing: torch.Tensor,
        u_ic: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Single-shot ``encode`` + ``decode`` (chunked callers use them directly)."""
        return self.decode(self.encode(u_forcing, u_ic), coords, t)


class ForcingTransitionCViT(nn.Module):
    """Causal transition operator over a source state and forcing interval."""

    def __init__(
        self,
        forcing_in_ch: int = 3,
        source_in_ch: int = 1,
        out_dim: int = 1,
        emb_dim: int = 256,
        dec_emb_dim: int | None = None,
        source_patch_size: int = 10,
        source_grid_size: tuple[int, int] = (100, 100),
        forcing_patch_size: int = 8,
        forcing_grid_size: tuple[int, int] = (96, 128),
        depth_enc: int = 4,
        depth_dec: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        fourier_freq: float = 1.0,
        fourier_freq_source: float = 1.0,
        fourier_freq_lead: float = 1.0,
        activation: str = "gelu",
        film_hidden_layers: int = 2,
        film_activation: str = "silu",
        head_hidden_layers: int = 1,
        head_activation: str = "gelu",
        query_time_conditioning: bool = True,
        film_time_conditioning: bool = True,
        film_init_std: float = 1.0e-3,
        coord_tolerance: float = 1.0e-6,
        t_final: float = 1.0,
    ):
        super().__init__()
        if int(forcing_in_ch) != 3:
            raise ValueError(
                "ForcingTransitionCViT requires three forcing channels "
                "[q/A_ref, relative_time, absolute_time]."
            )
        if int(source_in_ch) != 1 or int(out_dim) != 1:
            raise ValueError(
                "ForcingTransitionCViT requires one source and one output channel."
            )
        if t_final <= 0.0 or not math.isfinite(float(t_final)):
            raise ValueError("t_final must be finite and positive")
        if coord_tolerance < 0.0 or not math.isfinite(float(coord_tolerance)):
            raise ValueError("coord_tolerance must be finite and non-negative")

        source_grid_size = tuple(int(value) for value in source_grid_size)
        forcing_grid_size = tuple(int(value) for value in forcing_grid_size)
        source_patch_size = int(source_patch_size)
        forcing_patch_size = int(forcing_patch_size)
        for name, grid_size, patch_size in (
            ("source", source_grid_size, source_patch_size),
            ("forcing", forcing_grid_size, forcing_patch_size),
        ):
            if (
                len(grid_size) != 2
                or patch_size <= 0
                or any(value <= 0 for value in grid_size)
            ):
                raise ValueError(
                    f"{name}_grid_size must contain two positive dimensions and "
                    f"{name}_patch_size must be positive."
                )
            if any(value % patch_size != 0 for value in grid_size):
                raise ValueError(
                    f"{name}_grid_size dimensions must be divisible by "
                    f"{name}_patch_size; got {grid_size} and {patch_size}."
                )

        dec_emb_dim = int(dec_emb_dim) if dec_emb_dim is not None else int(emb_dim)
        self.source_grid_size = source_grid_size
        self.forcing_grid_size = forcing_grid_size
        self.source_patch_size = source_patch_size
        self.forcing_patch_size = forcing_patch_size
        self.num_source_tokens = (
            source_grid_size[0] // source_patch_size
        ) * (source_grid_size[1] // source_patch_size)
        self.num_forcing_tokens = (
            forcing_grid_size[0] // forcing_patch_size
        ) * (forcing_grid_size[1] // forcing_patch_size)
        self.coord_tolerance = float(coord_tolerance)
        self.query_time_conditioning = bool(query_time_conditioning)
        self.film_time_conditioning = bool(film_time_conditioning)
        self.film_init_std = float(film_init_std)

        self.source_encoder = CViTEncoder(
            in_ch=source_in_ch,
            emb_dim=emb_dim,
            patch_size=source_patch_size,
            grid_size=source_grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
        )
        self.forcing_encoder = CViTEncoder(
            in_ch=forcing_in_ch,
            emb_dim=emb_dim,
            patch_size=forcing_patch_size,
            grid_size=forcing_grid_size,
            depth=depth_enc,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
        )
        self.modality = nn.Parameter(torch.randn(2, emb_dim) * 0.02)
        self.forcing_context_projection = MLP(
            in_dim=2 * emb_dim,
            hidden_dim=dec_emb_dim,
            out_dim=dec_emb_dim,
            num_layers=2,
            activation=activation,
        )
        self.decoder = TransitionCViTDecoder(
            enc_emb_dim=emb_dim,
            dec_emb_dim=dec_emb_dim,
            forcing_context_dim=dec_emb_dim,
            out_dim=out_dim,
            depth=depth_dec,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            fourier_freq=fourier_freq,
            fourier_freq_source=fourier_freq_source,
            fourier_freq_lead=fourier_freq_lead,
            activation=activation,
            film_hidden_layers=film_hidden_layers,
            film_activation=film_activation,
            head_hidden_layers=head_hidden_layers,
            head_activation=head_activation,
            query_time_conditioning=query_time_conditioning,
            film_time_conditioning=film_time_conditioning,
            film_init_std=film_init_std,
        )
        self.register_buffer(
            "t_norm", torch.tensor(float(t_final), dtype=torch.float32),
        )

    def encode_source(self, u_source: Tensor) -> Tensor:
        if tuple(u_source.shape[1:]) != (1, *self.source_grid_size):
            raise ValueError(
                "u_source must have shape (B, 1, Nx, Ny) with "
                f"(Nx, Ny)={self.source_grid_size}; got {tuple(u_source.shape)}."
            )
        tokens = self.source_encoder(u_source) + self.modality[0]
        if tokens.shape[1] != self.num_source_tokens:
            raise RuntimeError(
                f"Expected {self.num_source_tokens} source tokens, "
                f"got {tokens.shape[1]}."
            )
        return tokens

    def encode_forcing(self, u_forcing_segment: Tensor) -> Tensor:
        if tuple(u_forcing_segment.shape[1:]) != (
            3, *self.forcing_grid_size,
        ):
            raise ValueError(
                "u_forcing_segment must have shape (B, 3, Ny_img, Nt_img) "
                f"with (Ny_img, Nt_img)={self.forcing_grid_size}; got "
                f"{tuple(u_forcing_segment.shape)}."
            )
        tokens = self.forcing_encoder(u_forcing_segment) + self.modality[1]
        if tokens.shape[1] != self.num_forcing_tokens:
            raise RuntimeError(
                f"Expected {self.num_forcing_tokens} forcing tokens, "
                f"got {tokens.shape[1]}."
            )
        return tokens

    def fuse(
        self,
        source_tokens: Tensor,
        forcing_tokens: Tensor,
    ) -> TransitionEncoding:
        if source_tokens.ndim != 3 or forcing_tokens.ndim != 3:
            raise ValueError("source_tokens and forcing_tokens must be rank-three")
        if source_tokens.shape[0] != forcing_tokens.shape[0]:
            raise ValueError("source and forcing token batch dimensions must match")
        if source_tokens.shape[1] != self.num_source_tokens:
            raise ValueError(
                f"Expected {self.num_source_tokens} source tokens, "
                f"got {source_tokens.shape[1]}."
            )
        if forcing_tokens.shape[1] != self.num_forcing_tokens:
            raise ValueError(
                f"Expected {self.num_forcing_tokens} forcing tokens, "
                f"got {forcing_tokens.shape[1]}."
            )
        forcing_summary = torch.cat(
            [
                forcing_tokens.mean(dim=1),
                forcing_tokens.amax(dim=1),
            ],
            dim=-1,
        )
        return TransitionEncoding(
            memory_tokens=torch.cat([source_tokens, forcing_tokens], dim=1),
            forcing_context=self.forcing_context_projection(forcing_summary),
        )

    def encode(
        self,
        u_forcing_segment: Tensor,
        u_source: Tensor,
    ) -> TransitionEncoding:
        return self.fuse(
            self.encode_source(u_source),
            self.encode_forcing(u_forcing_segment),
        )

    def decode(
        self,
        encoding: TransitionEncoding,
        u_source: Tensor,
        coords: Tensor,
        source_time: Tensor,
        lead_time: Tensor,
    ) -> Tensor:
        batch_size = encoding.memory_tokens.shape[0]
        if u_source.shape[0] != batch_size:
            raise ValueError("encoding and u_source batch dimensions must match")
        if coords.shape[0] == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1)
        source_values, coords = sample_source_at_queries(
            u_source,
            coords,
            tolerance=self.coord_tolerance,
        )
        source = _batch_scalar(
            source_time, batch_size=batch_size, name="source_time",
        ).to(device=coords.device, dtype=coords.dtype)
        lead = _batch_scalar(
            lead_time, batch_size=batch_size, name="lead_time",
        ).to(device=coords.device, dtype=coords.dtype)
        if bool((source < 0.0).any().item()):
            raise ValueError("source_time must be non-negative")
        if bool((lead < 0.0).any().item()):
            raise ValueError("lead_time must be non-negative")
        horizon = self.t_norm.to(device=source.device, dtype=source.dtype)
        if bool((source + lead > horizon + 1.0e-7).any().item()):
            raise ValueError(
                "source_time + lead_time must not exceed t_final"
            )
        source_norm = source / horizon
        lead_norm = lead / horizon
        residual = self.decoder(
            encoding,
            coords,
            source_norm,
            lead_norm,
        )
        x = coords[..., 0:1]
        return source_values + (1.0 - x) * lead_norm.unsqueeze(1) * residual

    def forward(
        self,
        u_forcing_segment: Tensor,
        u_source: Tensor,
        coords: Tensor,
        source_time: Tensor,
        lead_time: Tensor,
    ) -> Tensor:
        return self.decode(
            self.encode(u_forcing_segment, u_source),
            u_source,
            coords,
            source_time,
            lead_time,
        )
