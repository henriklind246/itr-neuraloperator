"""Physics-only single-sim overfit diagnostic (PINO objective, one simulation).

Companion to ``diagnose_supervised_overfit.py``. That script showed the CViT CAN
fit one sim's trajectory under a pure DATA objective (rel-L2 well below the 1.0
flat-mean floor), so the decoder is not fatally time-blind. This script asks the
next question: with the SAME model on the SAME single sim, can the PHYSICS-only
objective (interior heat residual + soft zero-Neumann walls + IC anchor) escape
the constant-300 K degenerate solution (rel-L2 == 1.0)?

It reuses the residual functions, data/IC helpers, grid-query validation, and
model builder from ``train_pino`` unchanged, but reimplements collocation locally
so it can (a) up-weight and densify the IC anchor and (b) floor the PDE/BC time
samples at ``t >= t_eps`` (Run E) to test the IC/Neumann corner-conflict theory
without touching the trainer. Diagnostic only; no training/eval path is modified.

Experiment matrix (freq_t=10, flat LR via StepLR gamma=1.0):

  B  IC-only :  --lambda-r 0   --lambda-bc 0  --dense-ic
  C  baseline:  --lambda-r 1   --lambda-ic 1  --lambda-bc 1
  D  IC-up   :  --lambda-ic 100 --dense-ic
  E  IC-up + no-BC-near-t0 : --lambda-ic 100 --dense-ic --t-eps 0.01
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operators.train import (
    _advance_scheduler,
    build_optimizer,
    build_scheduler,
    load_config,
    set_seed,
)
from src.operators.train_pino import (
    WALLS,
    build_cvit,
    build_ic_batch,
    load_diffusion_data,
    validate_rel_l2,
)
from src.operators.utils import resolve_device
from src.physics.pde_residual import (
    diffusion_residual,
    ic_residual,
    neumann_residual,
)

GROUP_ENV = {"benchmark": "BENCHMARK", "representation": "REPRESENTATION"}


def _parse_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _apply_override(config: dict, key_path: str, value: object) -> None:
    parts = key_path.split(".")
    current = config
    for idx, part in enumerate(parts[:-1]):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown config path: {'.'.join(parts[: idx + 1])}")
        current = current[part]
    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise KeyError(f"Unknown config path: {key_path}")
    current[leaf] = value


def _leaf(shape, device, generator, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """Uniform leaf on [lo, hi) that requires grad (for autodiff residuals)."""
    t = lo + (hi - lo) * torch.rand(shape, device=device, generator=generator)
    return t.requires_grad_(True)


def sample_collocation_eps(
    n_r: int,
    n_ic: int,
    n_bc: int,
    t_final: float,
    t_eps: float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    dense_ic: bool,
    device: torch.device,
    generator: torch.Generator | None,
) -> dict[str, Any]:
    """Local collocation with a PDE/BC time floor and optional dense IC.

    Mirrors ``train_pino.sample_collocation`` but (a) draws interior and wall
    times from ``U(t_eps, t_final)`` so the residual/Neumann constraints are not
    imposed in the thin near-IC layer (Run E), and (b) can place the IC anchor on
    EVERY grid node (``dense_ic``) instead of a random subset. IC time stays 0.
    """
    x_r = _leaf((1, n_r, 1), device, generator)
    y_r = _leaf((1, n_r, 1), device, generator)
    t_r = _leaf((1, n_r, 1), device, generator, lo=t_eps, hi=t_final)

    Nx = int(x_grid.numel())
    Ny = int(y_grid.numel())
    if dense_ic:
        gx, gy = torch.meshgrid(
            torch.arange(Nx, device=device), torch.arange(Ny, device=device),
            indexing="ij",
        )
        ix = gx.reshape(-1)
        iy = gy.reshape(-1)
    else:
        ix = torch.randint(0, Nx, (n_ic,), device=device, generator=generator)
        iy = torch.randint(0, Ny, (n_ic,), device=device, generator=generator)
    n_ic_eff = int(ix.numel())
    ic_x = x_grid[ix].view(1, n_ic_eff, 1)
    ic_y = y_grid[iy].view(1, n_ic_eff, 1)
    ic_coords = torch.cat([ic_x, ic_y], dim=-1)
    ic_t = torch.zeros((1, n_ic_eff, 1), device=device)

    def _wall(pin: str):
        free = _leaf((1, n_bc, 1), device, generator)
        tw = _leaf((1, n_bc, 1), device, generator, lo=t_eps, hi=t_final)
        if pin == "left":
            xw = torch.zeros((1, n_bc, 1), device=device, requires_grad=True)
            return xw, free, tw
        if pin == "top":
            yw = torch.ones((1, n_bc, 1), device=device, requires_grad=True)
            return free, yw, tw
        yw = torch.zeros((1, n_bc, 1), device=device, requires_grad=True)
        return free, yw, tw

    walls = {w: _wall(w) for w in WALLS}
    return {
        "interior": (x_r, y_r, t_r),
        "ic": {"coords": ic_coords, "t": ic_t, "ix": ix, "iy": iy},
        "walls": walls,
    }


def _ic_targets_single(
    trajectories: np.ndarray, sim_id: int, ix: torch.Tensor, iy: torch.Tensor,
    mu: float, sigma: float, device: torch.device,
) -> torch.Tensor:
    ix_np = ix.detach().cpu().numpy()
    iy_np = iy.detach().cpu().numpy()
    vals = np.asarray(trajectories[sim_id, 0, :, :], dtype=np.float32)
    vals = vals[ix_np, iy_np]
    vals = (vals - mu) / (sigma + 1e-8)
    return torch.from_numpy(vals).view(1, -1, 1).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Dotted key=value config overrides.")
    parser.add_argument("--sim-id", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lambda-r", type=float, default=1.0)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--n-r", type=int, default=4096)
    parser.add_argument("--n-ic", type=int, default=1024)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument("--dense-ic", action="store_true",
                        help="Anchor the IC on every grid node (overrides --n-ic).")
    parser.add_argument("--t-eps", type=float, default=0.0,
                        help="Floor PDE/BC time samples at t >= t_eps (IC stays t=0).")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--tag", type=str, default="phys_overfit1sim")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", project_root.as_posix())

    dotted: list[str] = []
    for raw in args.overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
        key, val = raw.split("=", 1)
        if key in GROUP_ENV:
            os.environ[GROUP_ENV[key]] = val
        elif key == "model.type":
            continue
        else:
            dotted.append(raw)

    os.environ.setdefault("BENCHMARK", "diffusion")
    os.environ.setdefault("REPRESENTATION", "temporal_encoder")

    config = copy.deepcopy(load_config())
    for raw in dotted:
        key_path, raw_value = raw.split("=", 1)
        _apply_override(config, key_path, _parse_value(raw_value))
    if args.lr is not None:
        config["training"]["learning_rate"] = float(args.lr)

    set_seed(args.seed)
    device = resolve_device(config["training"].get("device", "auto"))
    data = load_diffusion_data(config)
    mu, sigma = data["mu_global"], data["sigma_global"]
    Nx, Ny = int(data["x_grid"].shape[0]), int(data["y_grid"].shape[0])
    t_final = float(data["t_grid"][-1])
    alpha = float(config["training"]["pino"].get("alpha", 1.0))

    sim_id = int(args.sim_id) if args.sim_id is not None else int(data["train_ids"][0])
    ids = np.asarray([sim_id])

    u = build_ic_batch(data["trajectories"], ids, mu, sigma, device)  # (1,1,Nx,Ny)
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)

    fq_t = config["model"]["cvit"].get("fourier_freq_t", None)
    model = build_cvit(config, mu, sigma, grid_size=(Nx, Ny), t_final=t_final).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    out_dir = Path(args.out) if args.out else (
        Path(str(config["paths"]["runs_root"])) / "physics_overfit" / args.tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "overfit_metrics.csv"
    fieldnames = ["epoch", "loss", "loss_r", "loss_ic", "loss_bc",
                  "train_rel_l2", "train_rmse_K"]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    n_ic_report = Nx * Ny if args.dense_ic else args.n_ic
    print(
        f"[phys-overfit] sim_id={sim_id} device={device} epochs={args.epochs} "
        f"grid={Nx}x{Ny} t_final={t_final:.3f} alpha={alpha} | "
        f"lambda=(r={args.lambda_r},ic={args.lambda_ic},bc={args.lambda_bc}) "
        f"n=(r={args.n_r},ic={n_ic_report},bc={args.n_bc}) dense_ic={args.dense_ic} "
        f"t_eps={args.t_eps} | fourier_freq={config['model']['cvit'].get('fourier_freq')} "
        f"fourier_freq_t={fq_t} lr={config['training'].get('learning_rate')} | out={out_dir}",
        flush=True,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        batch = sample_collocation_eps(
            args.n_r, args.n_ic, args.n_bc, t_final, args.t_eps,
            x_grid, y_grid, args.dense_ic, device, gen,
        )
        ic_target = _ic_targets_single(
            data["trajectories"], sim_id, batch["ic"]["ix"], batch["ic"]["iy"],
            mu, sigma, device,
        )

        optimizer.zero_grad(set_to_none=True)
        # Compute only the active terms (skip expensive double-backward when off).
        loss = torch.zeros((), device=device)
        lr_val = float(optimizer.param_groups[0]["lr"])
        val_r = val_ic = val_bc = 0.0

        if args.lambda_ic > 0.0:
            ic = ic_residual(model, u, batch["ic"]["coords"], batch["ic"]["t"], ic_target)
            loss_ic = (ic ** 2).mean()
            val_ic = float(loss_ic.detach().cpu())
            loss = loss + args.lambda_ic * loss_ic

        if args.lambda_r > 0.0:
            x_r, y_r, t_r = batch["interior"]
            r = diffusion_residual(model, u, x_r, y_r, t_r, alpha=alpha)
            loss_r = (r ** 2).mean()
            val_r = float(loss_r.detach().cpu())
            loss = loss + args.lambda_r * loss_r

        if args.lambda_bc > 0.0:
            bc_sq = torch.zeros((), device=device)
            for w in WALLS:
                xw, yw, tw = batch["walls"][w]
                nb = neumann_residual(model, u, xw, yw, tw, w)
                bc_sq = bc_sq + (nb ** 2).mean()
            loss_bc = bc_sq / len(WALLS)
            val_bc = float(loss_bc.detach().cpu())
            loss = loss + args.lambda_bc * loss_bc

        loss.backward()
        optimizer.step()
        _advance_scheduler(scheduler, unit="update", successful_updates=1)
        _advance_scheduler(scheduler, unit="epoch", successful_updates=1)

        row = {"epoch": epoch, "loss": float(loss.detach().cpu()),
               "loss_r": val_r, "loss_ic": val_ic, "loss_bc": val_bc,
               "train_rel_l2": "", "train_rmse_K": ""}
        do_val = (epoch % args.validate_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            model.eval()
            val = validate_rel_l2(model, data, ids, device)
            row["train_rel_l2"] = val["val_rel_l2"]
            row["train_rmse_K"] = val["val_rmse_K"]
            print(
                f"Epoch {epoch}: loss={row['loss']:.6e} "
                f"r={val_r:.3e} ic={val_ic:.3e} bc={val_bc:.3e} "
                f"train_rel_l2={val['val_rel_l2'] * 100:.4f}% "
                f"train_rmse_K={val['val_rmse_K']:.4f}K lr={lr_val:.2e}",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch}: loss={row['loss']:.6e} "
                f"r={val_r:.3e} ic={val_ic:.3e} bc={val_bc:.3e} lr={lr_val:.2e}",
                flush=True,
            )

        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    print(f"[phys-overfit] done -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
