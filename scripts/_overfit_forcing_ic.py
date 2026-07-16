"""Throwaway single-sim overfit probe for diffusion_forcing_single (ForcingICCViT).

Freezes ONE online-sampled problem (IC field + sin/uniform forcing) and its
collocation points for the whole run, then trains physics-only PINO with the
autodiff residual. If a two-branch ForcingICCViT cannot drive the physics loss
of a single fixed problem toward ~0, the bottleneck is capacity/optimization,
not sampling variance. If it can, the varying-IC training gap is a
generalization/optimization problem, not raw capacity.

Freeze mechanism (no code change): keep completed_updates permanently inside the
warmup window (warmup.steps > epochs) and set warmup.resample_every > epochs, so
_should_resample_online_batch is True only at completed_updates == 0. Both the
sim descriptor batch AND the collocation points are then sampled once and reused.
Loss multipliers stay at their 1.0 defaults so the objective is undistorted.

Env knobs (all optional):
  OVERFIT_EPOCHS   default 4000
  OVERFIT_DEVICE   default mps   (mps|cpu|cuda)
  OVERFIT_DATA     default data/diffusion_forcing_new_500
  OVERFIT_RUN      default runs/_overfit_forcing_ic/config0
  OVERFIT_VALEVERY default = epochs (validate only at epoch 0 and the last epoch)

Run:
  PYTHONPATH=. .venv/bin/python scripts/_overfit_forcing_ic.py
Not a unit test; delete after the capacity-vs-optimization question is settled.
"""
import os
import shutil
from pathlib import Path

os.environ["BENCHMARK"] = "diffusion_forcing_single"
os.environ["REPRESENTATION"] = "temporal_encoder"

from src.operators.train import load_config  # noqa: E402
from src.operators.train_pino import run_config_seeds_pino  # noqa: E402

EPOCHS = int(os.environ.get("OVERFIT_EPOCHS", "4000"))
DEVICE = os.environ.get("OVERFIT_DEVICE", "mps")
DATA = Path(os.environ.get("OVERFIT_DATA", "data/_smoke_dfs_varyic"))
RUN = Path(os.environ.get("OVERFIT_RUN", "runs/_overfit_forcing_ic/config0"))
VAL_EVERY = int(os.environ.get("OVERFIT_VALEVERY", str(EPOCHS)))

if RUN.exists():
    shutil.rmtree(RUN)

cfg = load_config()

d = cfg["data"]
d["trajectories.npy"] = str(DATA / "trajectories.npy")
d["x_grid_path"] = str(DATA / "x_grid.npy")
d["y_grid_path"] = str(DATA / "y_grid.npy")
d["t_grid_path"] = str(DATA / "t_grid.npy")
if "sim_params_path" in d:
    d["sim_params_path"] = str(DATA / "sim_params.npy")

t = cfg["training"]
t["epochs"] = EPOCHS
t["validate_every"] = VAL_EVERY
t["device"] = DEVICE

pino = t["pino"]
pino["variant"] = "forcing_ic"
pino["residual_method"] = "autodiff"
pino["sim_batch"] = 1  # single sim
pino["forcing"]["ny_img"] = 96  # 100-grid: 96 % forcing_patch(8) == 0
pino["forcing"]["nt_img"] = 128
pino["forcing"]["a_ref"] = 300.0
pino["forcing"]["temporal_family"] = "sin"
pino["forcing"]["spatial_family"] = "uniform"
# Freeze the single online problem + its collocation points for the whole run.
pino["forcing"]["warmup"]["steps"] = EPOCHS + 1
pino["forcing"]["warmup"]["resample_every"] = EPOCHS + 1
pino["causal"]["enabled"] = False

summary = run_config_seeds_pino(cfg, base_run_dir=RUN, seeds=[42])
print("OVERFIT_SUMMARY", summary)
