"""Supervised single-sim overfit diagnostic (data MSE, NO physics).

The PINO path floors at val rel-L2 ~= 1.0 (the flat-mean predictor) even when
overfitting a single simulation with a dense IC anchor. That rules out loss
weighting and data volume and points at the model's ability to REPRESENT a
decaying trajectory at all. This script isolates that question: it trains the
CViT on one sim with a pure data objective -- query the model on grid nodes at
saved times and regress the (normalized) stored temperature -- with the residual,
IC, and Neumann terms all removed. If the model cannot drive rel-L2 well below
1.0 here, the failure is architectural (the leading suspect is a near time-blind
decoder: on t in [0, t_final=0.3] the shared fourier_freq=1.0 makes fourier_t
nearly constant across the trajectory).

The decisive comparison is freq_t=1 vs freq_t=10, holding everything else fixed:

  .venv/bin/python scripts/diagnose_supervised_overfit.py \
      model.cvit.fourier_freq_t=1  --epochs 400 --tag freqt1
  .venv/bin/python scripts/diagnose_supervised_overfit.py \
      model.cvit.fourier_freq_t=10 --epochs 400 --tag freqt10

If freq_t=1 stays pinned near 1.0 and freq_t=10 fits (rel-L2 -> small), the
time-blind-decoder diagnosis is causally confirmed and cheap. This is a
diagnostic only; it does not touch the training/eval paths.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.operators.train import (
    build_optimizer,
    build_scheduler,
    load_config,
    set_seed,
)
from src.operators.train_pino import (
    build_cvit,
    load_diffusion_data,
    validate_rel_l2,
)
from src.operators.utils import resolve_device

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Dotted key=value config overrides.")
    parser.add_argument("--sim-id", type=int, default=None,
                        help="Sim to overfit (default: first train id).")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--n-q", type=int, default=4096,
                        help="Random (space-node, time-node) points sampled per step.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training.learning_rate for this diagnostic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument("--tag", type=str, default="overfit1sim")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: runs_root/supervised_overfit/<tag>).")
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

    sim_id = int(args.sim_id) if args.sim_id is not None else int(data["train_ids"][0])
    ids = np.asarray([sim_id])

    # Encoder input: this sim's normalized IC field.  Target: the full normalized
    # trajectory on grid nodes at saved times -- read exactly, no interpolation.
    ic = np.asarray(data["trajectories"][ids, 0, :, :], dtype=np.float32)
    u = torch.from_numpy((ic - mu) / (sigma + 1e-8)).unsqueeze(1).to(device)  # (1,1,Nx,Ny)

    traj = np.asarray(data["trajectories"][sim_id], dtype=np.float32)          # (Nt,Nx,Ny)
    traj_norm = torch.from_numpy((traj - mu) / (sigma + 1e-8)).to(device)
    Nt = traj_norm.shape[0]
    x_grid = torch.as_tensor(data["x_grid"], dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(data["y_grid"], dtype=torch.float32, device=device)
    t_grid = torch.as_tensor(data["t_grid"], dtype=torch.float32, device=device)

    fq_t = config["model"]["cvit"].get("fourier_freq_t", None)
    model = build_cvit(config, mu, sigma, grid_size=(Nx, Ny)).to(device)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    out_dir = Path(args.out) if args.out else (
        Path(str(config["paths"]["runs_root"])) / "supervised_overfit" / args.tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "overfit_metrics.csv"
    fieldnames = ["epoch", "data_mse", "train_rel_l2", "train_rmse_K"]
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    print(
        f"[overfit] sim_id={sim_id} device={device} epochs={args.epochs} n_q={args.n_q} "
        f"grid={Nx}x{Ny} Nt={Nt} | fourier_freq={config['model']['cvit'].get('fourier_freq')} "
        f"fourier_freq_t={fq_t} | lr={config['training'].get('learning_rate')} | out={out_dir}",
        flush=True,
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        # Sample random grid-node/time-node points for this single sim.
        it = torch.randint(0, Nt, (args.n_q,), device=device, generator=gen)
        ix = torch.randint(0, Nx, (args.n_q,), device=device, generator=gen)
        iy = torch.randint(0, Ny, (args.n_q,), device=device, generator=gen)
        coords = torch.stack([x_grid[ix], y_grid[iy]], dim=-1).view(1, args.n_q, 2)
        tq = t_grid[it].view(1, args.n_q, 1)
        target = traj_norm[it, ix, iy].view(1, args.n_q, 1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(u, coords, tq)                 # (1, n_q, 1)
        loss = torch.mean((pred - target) ** 2)     # pure data MSE, no physics
        loss.backward()
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

        row = {"epoch": epoch, "data_mse": float(loss.detach().cpu()),
               "train_rel_l2": "", "train_rmse_K": ""}
        do_val = (epoch % args.validate_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            model.eval()
            # rel-L2 of the model against THIS sim's full trajectory (train==sim).
            val = validate_rel_l2(model, data, ids, device)
            row["train_rel_l2"] = val["val_rel_l2"]
            row["train_rmse_K"] = val["val_rmse_K"]
            print(
                f"Epoch {epoch}: data_mse={row['data_mse']:.6e} "
                f"train_rel_l2={val['val_rel_l2'] * 100:.4f}% "
                f"train_rmse_K={val['val_rmse_K']:.4f}K lr={lr:.2e}",
                flush=True,
            )
        else:
            print(f"Epoch {epoch}: data_mse={row['data_mse']:.6e} lr={lr:.2e}", flush=True)

        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    print(f"[overfit] done -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
