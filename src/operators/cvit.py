from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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


def _sincos_pos_embed_2d(emb_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Return the reference-initialized 2D sin-cos positional embedding."""
    if emb_dim % 4 != 0:
        raise ValueError(
            f"emb_dim must be divisible by 4 for 2D sin-cos pos-embed, got {emb_dim}."
        )
    grid_y = np.arange(grid_h, dtype=np.float32)
    grid_x = np.arange(grid_w, dtype=np.float32)
    gx, gy = np.meshgrid(grid_x, grid_y)
    quarter = emb_dim // 4
    omega = np.arange(quarter, dtype=np.float32) / float(quarter)
    omega = 1.0 / (10000.0 ** omega)

    def _embed(pos: np.ndarray) -> np.ndarray:
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([_embed(gy), _embed(gx)], axis=1)
    return torch.from_numpy(emb.astype(np.float32))


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
        qh = self._split(self.q_proj(q))
        kh = self._split(self.k_proj(kv))
        vh = self._split(self.v_proj(kv))
        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).reshape(B, Nq, self.num_heads * self.head_dim)
        return self.out_proj(out)


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
        x = self.proj(u)
        return x.flatten(2).transpose(1, 2)


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
        proj = x @ self.kernel
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


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
        pos = _sincos_pos_embed_2d(emb_dim, grid_h, grid_w)


        self.pos_emb = nn.Parameter(pos.unsqueeze(0))
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
        use_time_query: bool = True,
    ):
        super().__init__()
        if film_hidden_layers < 0 or head_hidden_layers < 0:
            raise ValueError("FiLM and head hidden-layer counts must be non-negative.")
        self.depth = depth
        self.dec_emb_dim = dec_emb_dim
        self.use_time_query = use_time_query





        freq_t = float(fourier_freq if fourier_freq_t is None else fourier_freq_t)
        self.fourier_x = FourierEmbed(2, dec_emb_dim, fourier_freq)
        if use_time_query:
            self.fourier_t = FourierEmbed(1, dec_emb_dim, freq_t)


        self.time_film = MLP(
            in_dim=dec_emb_dim,
            hidden_dim=dec_emb_dim,
            out_dim=depth * 2 * dec_emb_dim,
            num_layers=film_hidden_layers + 1,
            activation=film_activation,
            zero_init_last=True,
        ) if use_time_query else None
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
        self, tokens: torch.Tensor, coords: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:

        B, Nq, _ = coords.shape
        queries = self.fourier_x(coords)
        kv = self.proj_kv(tokens)
        if self.use_time_query:
            if t is None:
                raise ValueError("An arbitrary-time decoder requires query times")
            film = self.time_film(self.fourier_t(t)).view(B, Nq, self.depth, 2, self.dec_emb_dim)
        for i, block in enumerate(self.blocks):
            scale = film[:, :, i, 0, :] if self.use_time_query else None
            shift = film[:, :, i, 1, :] if self.use_time_query else None
            queries = block(queries, kv, scale=scale, shift=shift)
        queries = self.norm(queries)
        return self.head(queries)


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
         ``(B, 1, Ny_img, Nt_img)`` (axes rows=y-nodes, cols=time). Preserves the
         archived forcing-image architecture.
      2. ic ``z_ic``: independent :class:`CViTEncoder` over the normalized
         initial field ``T0_tilde = (T0 - mu_global)/sigma_global`` on the data
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

    With ``use_time_query=False``, the second branch receives the current state
    and the forcing image carries one interval's average flux. The decoder then
    uses only (x,y) to predict the fixed-dt next field.
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
        use_time_query: bool = True,
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
            use_time_query=use_time_query,
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
        self, latent: torch.Tensor, coords: torch.Tensor, t: torch.Tensor | None = None
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
        if t is not None and t.shape[0] == 1 and B > 1:
            t = t.expand(B, -1, -1)
        raw = self.decoder(latent, coords, t / self.t_norm if t is not None else None)
        if not self.hard_right_dirichlet:
            return raw
        x = coords[..., 0:1]
        return self.t_right_tilde + (1.0 - x) * raw

    def forward(
        self,
        u_forcing: torch.Tensor,
        u_ic: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single-shot ``encode`` + ``decode`` (chunked callers use them directly)."""
        return self.decode(self.encode(u_forcing, u_ic), coords, t)
